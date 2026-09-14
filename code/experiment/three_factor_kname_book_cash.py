#!/usr/bin/env python3
"""pos64 / RSI / -PVR + BOOK + cash>20%. Max 6 names, but 2/4/6 all allowed."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from blend_book import book_ratio, zscore
from family_rotate import fmt, rotate_daily, rotate_quarter, year_pack
from linear_ridge_walkforward import EVAL_END, OUT_DIR, build_panel
from six_name_select import pick_abs_n, pick_ls_n
from two_model_rotate import BT_START
from two_name_select import apply_hold

FACTORS = [("pos_64", 1, "pos64"), ("rsi", 1, "rsi"), ("price_volume_ratio", -1, "pvr")]
SIZES = (1, 2, 3)  # n_each → 2 / 4 / 6 names


def clip_ls(books: list[list[tuple[str, float]]], n_each: int) -> list[tuple[str, float]]:
    agg: dict[str, float] = defaultdict(float)
    for b in books:
        for s, w in b:
            agg[s] += w
    ser = pd.Series(agg, dtype=float)
    if len(ser) < 2:
        return []
    k = min(n_each, len(ser) // 2)
    if k < 1:
        return []
    longs = ser.nlargest(k)
    shorts = ser.drop(index=longs.index).nsmallest(k)
    w = 0.5 / k
    return [(i, w) for i in longs.index] + [(i, -w) for i in shorts.index]


def sized(full: pd.Series, small: pd.Series | None, ratio: pd.Series, t2: float, t0: float, how: str) -> tuple[pd.Series, pd.Series]:
    r = ratio.reindex(full.index)
    cash = (r >= t0).fillna(False)
    mid = (r >= t2).fillna(False) & ~cash
    out = full.copy()
    if how == "switch" and small is not None:
        out[mid] = small.reindex(full.index)[mid]
    elif how == "scale":
        out[mid] = full[mid] * (2.0 / 6.0)
    out[cash] = 0.0
    state = pd.Series("on", index=full.index)
    state[mid] = "mid"
    state[cash] = "0"
    return out, state


def pack(s: pd.Series, state: pd.Series | None = None, cash: pd.Series | None = None) -> dict:
    st = year_pack(s)
    if cash is None:
        cash = s.abs() < 1e-12
        if state is not None:
            cash = state.eq("0")
    sharpes = [r["sharpe"] for r in st["yearly"] if r["sharpe"] is not None]
    st["cash"] = fmt(float(cash.reindex(s.index).fillna(False).mean()), 3)
    st["sharpe_std"] = fmt(float(np.std(sharpes, ddof=1)), 3) if len(sharpes) > 2 else None
    if state is not None:
        st["pct_on"] = fmt(float(state.eq("on").mean()), 3)
        st["pct_mid"] = fmt(float(state.eq("mid").mean()), 3)
    return st


def to_s(rows) -> pd.Series:
    return pd.Series({d: v for d, v in rows}).sort_index().dropna()


def main() -> None:
    print("loading panel...", flush=True)
    cols = ["fwd_ret", "pos_64", "rsi", "price_volume_ratio"]
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=cols)

    fac = {k: {n: [] for n in SIZES} for _, _, k in FACTORS}
    blend = {n: [] for n in SIZES}
    vote = {n: [] for n in SIZES}
    abs_pos = {n: [] for n in (2, 4, 6)}
    abs_zb = {n: [] for n in (2, 4, 6)}

    print("daily books 2/4/6...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        zsc = None
        for col, sgn, _ in FACTORS:
            z = sgn * zscore(day[col].to_numpy())
            zsc = z if zsc is None else zsc + z
        books = {n: [] for n in SIZES}
        for col, sgn, name in FACTORS:
            raw = sgn * day[col].to_numpy()
            for n in SIZES:
                p = pick_ls_n(day, raw, n)
                fac[name][n].append((dt, apply_hold(day, p)))
                books[n].append(p)
        for n in SIZES:
            blend[n].append((dt, apply_hold(day, pick_ls_n(day, zsc, n))))
            vote[n].append((dt, apply_hold(day, clip_ls(books[n], n))))
        for n in (2, 4, 6):
            abs_pos[n].append((dt, apply_hold(day, pick_abs_n(day, day["pos_64"].to_numpy(), n))))
            abs_zb[n].append((dt, apply_hold(day, pick_abs_n(day, zsc, n))))

    s_fac = {k: {n: to_s(fac[k][n]) for n in SIZES} for k in fac}
    idx = s_fac["rsi"][3].index
    for k in s_fac:
        for n in SIZES:
            s_fac[k][n] = s_fac[k][n].reindex(idx)
    s_blend = {n: to_s(blend[n]).reindex(idx) for n in SIZES}
    s_vote = {n: to_s(vote[n]).reindex(idx) for n in SIZES}
    s_abs_p = {n: to_s(abs_pos[n]).reindex(idx) for n in (2, 4, 6)}
    s_abs_z = {n: to_s(abs_zb[n]).reindex(idx) for n in (2, 4, 6)}

    def rot_at(n: int, win: int, margin: float, kind: str):
        d = {k: s_fac[k][n] for k in ("pos64", "rsi", "pvr")}
        if kind == "q":
            pnl, pick = rotate_quarter(d, 60)
        else:
            pnl, pick = rotate_daily(d, win, margin)
        pnl = pnl.reindex(idx)
        pick = pick.reindex(idx).ffill()
        small_n = 1
        p_small = []
        for dt in idx:
            name = pick.loc[dt] if dt in pick.index and pd.notna(pick.loc[dt]) else "rsi"
            p_small.append((dt, s_fac[name][small_n].loc[dt]))
        return pnl, to_s(p_small).reindex(idx), book_ratio(pnl)

    books: list[tuple[str, pd.Series, pd.Series, pd.Series, int]] = []
    # (label, full, small_for_switch, book_ratio_src, n_each)
    for n, tag in ((1, "2名"), (2, "4名"), (3, "6名")):
        for k, cn in (("pos64", "pos64"), ("rsi", "RSI"), ("pvr", "价量")):
            full = s_fac[k][n]
            small = s_fac[k][1]
            books.append((f"固定{cn} {tag}", full, small, book_ratio(full), n))
        books.append((f"z合成 {tag}", s_blend[n], s_blend[1], book_ratio(s_blend[n]), n))
        books.append((f"投票 {tag}", s_vote[n], s_vote[1], book_ratio(s_vote[n]), n))
        r20, r20s, r20b = rot_at(n, 20, 0.0, "d")
        rm, rms, rmb = rot_at(n, 20, 0.03, "d")
        rq, rqs, rqb = rot_at(n, 60, 0.0, "q")
        books.append((f"轮动20d {tag}", r20, r20s, r20b, n))
        books.append((f"轮动20d+3% {tag}", rm, rms, rmb, n))
        books.append((f"季60d {tag}", rq, rqs, rqb, n))
    for n, tag in ((2, "2名|分数|"), (4, "4名|分数|"), (6, "6名|分数|")):
        books.append((f"pos64 {tag}", s_abs_p[n], s_abs_p[2], book_ratio(s_abs_p[n]), n // 2 if n > 1 else 1))
        books.append((f"z合成 {tag}", s_abs_z[n], s_abs_z[2], book_ratio(s_abs_z[n]), n // 2 if n > 1 else 1))

    schemes: dict[str, tuple[pd.Series, pd.Series | None]] = {}

    def add(name, pnl, state=None):
        schemes[name] = (pnl, state)

    for label, full, small, br, _n in books:
        for t0 in (1.05, 1.10, 1.15, 1.20):
            pnl, stt = sized(full, small, br, t2=t0, t0=t0, how="scale")
            add(f"{label} | BOOK≥{t0:.2f}空仓", pnl, stt)
        pnl, stt = sized(full, small, br, t2=1.10, t0=1.25, how="switch")
        add(f"{label} | BOOK 满仓/2名/0 (1.10/1.25)", pnl, stt)

    rows = []
    print(f"{'scheme':58} cash  sh   dd    neg  worst  2026", flush=True)
    for name, (pnl, stt) in schemes.items():
        cash = stt.eq("0") if stt is not None else pnl.abs() < 1e-12
        st = pack(pnl, stt, cash)
        st["scheme"] = name
        rows.append(st)

    ok = [r for r in rows if (r["cash"] or 0) >= 0.20]
    ok.sort(key=lambda r: (
        r["n_neg_years"],
        -(r["worst_year_sharpe"] or -9),
        -(r["max_dd"] or -9),
        -(r["sharpe"] or -9),
    ))
    zeros = [r for r in ok if r["n_neg_years"] == 0]
    print(f"\n{len(ok)} pass cash>=20%  |  0-neg={len(zeros)}", flush=True)
    print("0-neg:", flush=True)
    for r in zeros:
        print(
            f"  {r['scheme']:58} sh={r['sharpe']:.2f} dd={r['max_dd']:.1%} "
            f"w={r['worst_year_sharpe']:.2f} cash={r['cash']:.0%} 26={r['sharpe_2026']} ret={r['return']:+.0%}",
            flush=True,
        )
    print("top 12 by stability (incl. 1+ neg):", flush=True)
    for r in ok[:12]:
        print(
            f"  {r['scheme']:58} sh={r['sharpe']:.2f} dd={r['max_dd']:.1%} "
            f"neg={r['neg_years']} cash={r['cash']:.0%} 26={r['sharpe_2026']}",
            flush=True,
        )

    slim = lambda r: {k: r[k] for k in (
        "scheme", "return", "sharpe", "max_dd", "n_neg_years", "neg_years",
        "worst_year_sharpe", "sharpe_std", "sharpe_2026", "return_2026", "cash", "pct_on", "pct_mid", "yearly",
    ) if k in r}
    path = OUT_DIR / "three_factor_kname_book_cash.json"
    path.write_text(json.dumps({
        "constraint": "max 6 names; 2/4/6 allowed; cash>20%; BOOK; pos64/rsi/-pvr",
        "pass_cash20": [slim(r) for r in ok],
        "zero_neg": [slim(r) for r in zeros],
    }, ensure_ascii=False, indent=2))
    print(f"saved {path}  n_schemes={len(rows)}")


if __name__ == "__main__":
    main()
