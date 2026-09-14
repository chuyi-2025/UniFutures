#!/usr/bin/env python3
"""pos64 / RSI / -PVR + BOOK lots + cash>20%, account max 6 names (3L3S)."""

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
from family_rotate import fmt, rotate_daily, rotate_quarter, trail_sum, year_pack
from linear_ridge_walkforward import EVAL_END, OUT_DIR, build_panel
from six_name_select import pick_ls_n
from two_model_rotate import BT_START
from two_name_dd_control import flatten_after_loss
from two_name_select import apply_hold

FACTORS = [("pos_64", 1, "pos64"), ("rsi", 1, "rsi"), ("price_volume_ratio", -1, "pvr")]


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


def sized(n6: pd.Series, n2: pd.Series, ratio: pd.Series, t2: float, t0: float, how: str) -> tuple[pd.Series, pd.Series]:
    r = ratio.reindex(n6.index)
    cash = (r >= t0).fillna(False)
    mid = (r >= t2).fillna(False) & ~cash
    out = n6.copy()
    if how == "switch":
        out[mid] = n2.reindex(n6.index)[mid]
    else:
        out[mid] = n6[mid] * (2.0 / 6.0)
    out[cash] = 0.0
    state = pd.Series("6", index=n6.index)
    state[mid] = "2"
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
        st["pct_6"] = fmt(float(state.eq("6").mean()), 3)
        st["pct_2"] = fmt(float(state.eq("2").mean()), 3)
    return st


def main() -> None:
    print("loading panel...", flush=True)
    cols = ["fwd_ret", "pos_64", "rsi", "price_volume_ratio"]
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=cols)
    n6: dict[str, list] = {k: [] for _, _, k in FACTORS}
    n2: dict[str, list] = {k: [] for _, _, k in FACTORS}
    blend6, blend2, vote6, vote2 = [], [], [], []
    print("daily 3L3S books (max 6 names)...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        books6, books2 = [], []
        for col, sgn, name in FACTORS:
            sc = sgn * day[col].to_numpy()
            p6 = pick_ls_n(day, sc, 3)
            p2 = pick_ls_n(day, sc, 1)
            n6[name].append((dt, apply_hold(day, p6)))
            n2[name].append((dt, apply_hold(day, p2)))
            books6.append(p6)
            books2.append(p2)
        zsc = None
        for col, sgn, _ in FACTORS:
            z = sgn * zscore(day[col].to_numpy())
            zsc = z if zsc is None else zsc + z
        blend6.append((dt, apply_hold(day, pick_ls_n(day, zsc, 3))))
        blend2.append((dt, apply_hold(day, pick_ls_n(day, zsc, 1))))
        vote6.append((dt, apply_hold(day, clip_ls(books6, 3))))
        vote2.append((dt, apply_hold(day, clip_ls(books2, 1))))

    def to_s(rows):
        return pd.Series({d: v for d, v in rows}).sort_index().dropna()

    s6 = {k: to_s(n6[k]) for k in n6}
    s2 = {k: to_s(n2[k]).reindex(s6["rsi"].index) for k in n2}
    idx = s6["rsi"].index
    for k in s6:
        s6[k] = s6[k].reindex(idx)
        s2[k] = s2[k].reindex(idx)
    zb6 = to_s(blend6).reindex(idx)
    zb2 = to_s(blend2).reindex(idx)
    vt6 = to_s(vote6).reindex(idx)
    vt2 = to_s(vote2).reindex(idx)

    rot20, pick20 = rotate_daily(s6, 20, 0.0)
    rot20m, pick20m = rotate_daily(s6, 20, 0.03)
    rot60q, pick60q = rotate_quarter(s6, 60)
    rot20 = rot20.reindex(idx)
    rot20m = rot20m.reindex(idx)
    rot60q = rot60q.reindex(idx)
    pick20 = pick20.reindex(idx).ffill()
    pick20m = pick20m.reindex(idx).ffill()
    pick60q = pick60q.reindex(idx).ffill()

    def book_of_pick(picks: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        p6, p2 = [], []
        for dt in idx:
            name = picks.loc[dt] if dt in picks.index and pd.notna(picks.loc[dt]) else "rsi"
            p6.append((dt, s6[name].loc[dt]))
            p2.append((dt, s2[name].loc[dt]))
        a = pd.Series({d: v for d, v in p6})
        b = pd.Series({d: v for d, v in p2})
        return a, b, book_ratio(a)

    r20_6, r20_2, r20_b = book_of_pick(pick20)
    rm_6, rm_2, rm_b = book_of_pick(pick20m)
    q_6, q_2, q_b = book_of_pick(pick60q)

    schemes: dict[str, tuple[pd.Series, pd.Series | None]] = {}

    def add(name, pnl, state=None):
        schemes[name] = (pnl, state)

    for label, a, b, br in [
        ("z合成 3L3S", zb6, zb2, book_ratio(zb6)),
        ("投票截6", vt6, vt2, book_ratio(vt6)),
        ("轮动20d", r20_6, r20_2, r20_b),
        ("轮动20d+3%", rm_6, rm_2, rm_b),
        ("季60d", q_6, q_2, q_b),
        ("固定rsi", s6["rsi"], s2["rsi"], book_ratio(s6["rsi"])),
        ("固定pos64", s6["pos64"], s2["pos64"], book_ratio(s6["pos64"])),
        ("固定pvr", s6["pvr"], s2["pvr"], book_ratio(s6["pvr"])),
    ]:
        for t0 in (1.05, 1.10, 1.15, 1.20):
            pnl, stt = sized(a, b, br, t2=t0, t0=t0, how="scale")
            add(f"{label} | BOOK≥{t0:.2f}空仓", pnl, stt)
        pnl, stt = sized(a, b, br, t2=1.10, t0=1.25, how="scale")
        add(f"{label} | BOOK 6/⅓/0 (1.10/1.25)", pnl, stt)
        pnl, stt = sized(a, b, br, t2=1.10, t0=1.25, how="switch")
        add(f"{label} | BOOK 6/2手/0 (1.10/1.25)", pnl, stt)

    trail20 = pd.DataFrame({k: trail_sum(s6[k], 20) for k in s6})
    win_trail = pd.Series({
        dt: trail20.loc[dt, pick20.loc[dt]] if dt in trail20.index and pd.notna(pick20.loc[dt]) else np.nan
        for dt in idx
    })
    gated = r20_6.copy()
    gated[win_trail.reindex(idx).fillna(0) <= 0] = 0.0
    g2 = r20_2.copy()
    g2[win_trail.reindex(idx).fillna(0) <= 0] = 0.0
    cash_gate = win_trail.reindex(idx).fillna(0) <= 0
    for t0 in (1.10, 1.20, 9.0):
        pnl, stt = sized(gated, g2, book_ratio(r20_6), t2=t0, t0=t0, how="scale")
        pnl = pnl.copy()
        pnl[cash_gate] = 0.0
        if stt is not None:
            stt = stt.copy()
            stt[cash_gate] = "0"
        tag = "仅赢家20d>0" if t0 > 5 else f"仅赢家20d>0 + BOOK≥{t0:.2f}空"
        add(f"轮动20d | {tag}", pnl, stt)

    cool = flatten_after_loss(r20_6, -0.02, 5)
    cool2 = flatten_after_loss(r20_2, -0.02, 5)
    pnl, stt = sized(cool, cool2, book_ratio(r20_6), t2=1.10, t0=1.10, how="scale")
    add("轮动20d | 日亏2%空5日 + BOOK≥1.10空", pnl, stt)

    rows = []
    print(f"{'scheme':52} cash  sh   dd    neg  worst  std   2026", flush=True)
    for name, (pnl, stt) in schemes.items():
        cash = stt.eq("0") if stt is not None else pnl.abs() < 1e-12
        st = pack(pnl, stt, cash)
        st["scheme"] = name
        rows.append(st)
        mark = " *" if (st["cash"] or 0) >= 0.20 else ""
        print(
            f"{name:52} {st['cash']:.0%} {st['sharpe']:.2f} {st['max_dd']:.1%} "
            f"n={st['n_neg_years']} w={st['worst_year_sharpe']:.2f} σ={st['sharpe_std']} "
            f"26={st['sharpe_2026']}{mark}",
            flush=True,
        )

    ok = [r for r in rows if (r["cash"] or 0) >= 0.20]
    ok.sort(key=lambda r: (
        r["n_neg_years"],
        -(r["worst_year_sharpe"] or -9),
        -(r["max_dd"] or -9),
        r["sharpe_std"] or 9,
    ))
    slim = lambda r: {k: r[k] for k in (
        "scheme", "return", "sharpe", "max_dd", "n_neg_years", "neg_years",
        "worst_year_sharpe", "sharpe_std", "sharpe_2026", "return_2026", "cash", "pct_6", "pct_2", "yearly",
    ) if k in r}
    path = OUT_DIR / "three_factor_6name_book_cash.json"
    path.write_text(json.dumps({
        "constraint": "max 6 names/day (3L3S); cash>20%; BOOK lots; pos64 / rsi / -pvr",
        "pass_cash20": [slim(r) for r in ok],
        "all": [slim(r) for r in rows],
    }, ensure_ascii=False, indent=2))
    print(f"\n{len(ok)} schemes with cash>=20%  saved -> {path}")
    print("top by stability:", flush=True)
    for r in ok[:12]:
        print(
            f"  {r['scheme']:52} sh={r['sharpe']:.2f} dd={r['max_dd']:.1%} "
            f"neg={r['neg_years']} cash={r['cash']:.0%} 2026={r['sharpe_2026']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
