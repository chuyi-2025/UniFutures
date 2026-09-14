#!/usr/bin/env python3
"""pos64 only: more cash + fewer names. BOOK, hysteresis, extremity, trail, vol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from blend_book import book_ratio
from family_rotate import fmt, trail_sum, year_pack
from linear_ridge_walkforward import EVAL_END, OUT_DIR, build_panel
from six_name_select import pick_ls_n
from three_factor_24_search import flatten_book, pack, vol_scale
from two_model_rotate import BT_START, daily_mkt
from two_name_dd_control import flatten_after_loss
from two_name_select import apply_hold


def pick_band(day: pd.DataFrame, hi: float, lo: float, n_max: int) -> list[tuple[str, float]]:
    s = pd.Series(day["pos_64"].to_numpy(), index=day.index)
    s = s[s.notna() & day["fwd_ret"].notna()]
    long_idx = s[s >= hi].nlargest(n_max).index
    short_idx = s[s <= lo].nsmallest(n_max).index
    n_l, n_s = len(long_idx), len(short_idx)
    out: list[tuple[str, float]] = []
    if n_l:
        w = (0.5 / n_l) if n_s else (1.0 / n_l)
        out += [(day.loc[i, "symbol"], w) for i in long_idx]
    if n_s:
        w = (0.5 / n_s) if n_l else (1.0 / n_s)
        out += [(day.loc[i, "symbol"], -w) for i in short_idx]
    return out


def hysteresis(pnl: pd.Series, exit_thr: float, enter_thr: float) -> pd.Series:
    r = book_ratio(pnl)
    in_cash = False
    flags = []
    for dt in pnl.index:
        rv = r.loc[dt] if dt in r.index else np.nan
        if np.isfinite(rv):
            if in_cash:
                if rv <= enter_thr:
                    in_cash = False
            elif rv >= exit_thr:
                in_cash = True
        flags.append(in_cash)
    return pd.Series(flags, index=pnl.index)


def apply_cash(pnl: pd.Series, cash: pd.Series) -> pd.Series:
    out = pnl.copy()
    out[cash.reindex(pnl.index).fillna(False)] = 0.0
    return out


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=["fwd_ret", "pos_64"])
    mkt = daily_mkt(panel)
    mkt_vol = mkt.rolling(20).std(ddof=1) * np.sqrt(252)
    mkt_med = mkt_vol.rolling(252).median()
    mkt_book = (mkt_vol / mkt_med).shift(1)

    raw: dict[str, list] = {k: [] for k in ("n2", "n4", "n6")}
    meta = []  # date, top, bot, gap, cs_std
    bands: dict[str, list] = {}
    band_keys = []
    for hi, lo, nmax in [
        (0.70, 0.30, 1), (0.70, 0.30, 2),
        (0.80, 0.20, 1), (0.80, 0.20, 2),
        (0.85, 0.15, 1), (0.85, 0.15, 2),
        (0.90, 0.10, 1), (0.90, 0.10, 2),
        (0.75, 0.25, 1), (0.75, 0.25, 2),
    ]:
        k = f"band{hi:.2f}/{lo:.2f}x{nmax}"
        bands[k] = []
        band_keys.append(k)

    print("daily pos64...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        sc = day["pos_64"]
        ok = sc.notna() & day["fwd_ret"].notna()
        s = sc[ok]
        top = float(s.max()) if len(s) else np.nan
        bot = float(s.min()) if len(s) else np.nan
        gap = top - bot if np.isfinite(top) and np.isfinite(bot) else np.nan
        cs = float(s.std(ddof=1)) if len(s) > 2 else np.nan
        meta.append((dt, top, bot, gap, cs))
        p2 = pick_ls_n(day, day["pos_64"].to_numpy(), 1)
        p4 = pick_ls_n(day, day["pos_64"].to_numpy(), 2)
        p6 = pick_ls_n(day, day["pos_64"].to_numpy(), 3)
        raw["n2"].append((dt, apply_hold(day, p2), len(p2)))
        raw["n4"].append((dt, apply_hold(day, p4), len(p4)))
        raw["n6"].append((dt, apply_hold(day, p6), len(p6)))
        for hi, lo, nmax in [
            (0.70, 0.30, 1), (0.70, 0.30, 2),
            (0.80, 0.20, 1), (0.80, 0.20, 2),
            (0.85, 0.15, 1), (0.85, 0.15, 2),
            (0.90, 0.10, 1), (0.90, 0.10, 2),
            (0.75, 0.25, 1), (0.75, 0.25, 2),
        ]:
            k = f"band{hi:.2f}/{lo:.2f}x{nmax}"
            w = pick_band(day, hi, lo, nmax)
            bands[k].append((dt, apply_hold(day, w) if w else 0.0, len(w)))

    def ser(rows):
        pnl = pd.Series({d: p for d, p, _ in rows}).sort_index()
        nn = pd.Series({d: n for d, _, n in rows}).sort_index()
        return pnl, nn

    books = {}
    for k, rows in raw.items():
        books[k] = ser(rows)
    for k, rows in bands.items():
        books[k] = ser(rows)

    idx = books["n4"][0].index
    meta_df = pd.DataFrame(meta, columns=["date", "top", "bot", "gap", "cs"]).set_index("date").reindex(idx)
    mkt_book = mkt_book.reindex(idx)

    rows_out = []

    def add(name, pnl, cash, nn, size_tag):
        live = apply_cash(pnl, cash)
        nn2 = nn.reindex(pnl.index).fillna(0).where(~cash.reindex(pnl.index).fillna(False), 0)
        st = pack(live, cash.reindex(pnl.index).fillna(False), nn2)
        st["scheme"] = name
        st["size_tag"] = size_tag
        rows_out.append(st)

    print("gates...", flush=True)
    for lab, (pnl, nn) in books.items():
        pnl = pnl.reindex(idx)
        nn = nn.reindex(idx).fillna(0)
        size_tag = lab
        # always-on
        add(f"{lab} | 无过滤", pnl, pnl.abs() < 1e-12, nn, size_tag)
        # BOOK flatten
        for thr in (1.00, 1.05, 1.08, 1.10, 1.12, 1.15, 1.20):
            live, cash = flatten_book(pnl, thr)
            add(f"{lab} | BOOK≥{thr:.2f}空", live, cash, nn, size_tag)
        # hysteresis
        for ex, en in ((1.10, 0.95), (1.15, 1.00), (1.10, 1.00), (1.20, 1.05), (1.08, 0.90)):
            cash = hysteresis(pnl, ex, en)
            add(f"{lab} | BOOK出{ex:.2f}进{en:.2f}", apply_cash(pnl, cash), cash, nn, size_tag)
        # trail
        tr = trail_sum(pnl, 20)
        cash = (tr.reindex(idx).fillna(0) <= 0)
        add(f"{lab} | 仅20日累计>0", apply_cash(pnl, cash), cash, nn, size_tag)
        live, bch = flatten_book(pnl, 1.10)
        cash = bch | (tr.reindex(idx).fillna(0) <= 0)
        add(f"{lab} | 20日>0 + BOOK≥1.10", apply_cash(pnl, cash), cash, nn, size_tag)
        # loss flatten
        for th, cd in ((-0.02, 5), (-0.015, 5), (-0.02, 10)):
            cool = flatten_after_loss(pnl, th, cd)
            add(f"{lab} | 日亏{th:.1%}空{cd}日", cool, cool.abs() < 1e-12, nn, size_tag)
            live, bch = flatten_book(cool, 1.10)
            add(f"{lab} | 日亏{th:.1%}空{cd}日+BOOK≥1.10", live, bch, nn, size_tag)
        # mkt vol: cash when market BOOK high (risk-off) or low
        for thr in (1.10, 1.20, 1.30):
            cash = (mkt_book >= thr).fillna(False)
            add(f"{lab} | 市场BOOK≥{thr:.2f}空", apply_cash(pnl, cash), cash, nn, size_tag)
        # vol target + BOOK (cash from BOOK only)
        for tgt, thr in ((0.06, 1.10), (0.06, 1.14), (0.07, 1.10), (0.07, 1.14)):
            vs = vol_scale(pnl, tgt)
            live, cash = flatten_book(vs, thr)
            add(f"{lab} | 波动{int(tgt*100)}%+BOOK≥{thr:.2f}", live, cash, nn, size_tag)

    # CS stretch on top of 1L1S / 2L2S: skip unstretched days
    for lab in ("n2", "n4"):
        pnl, nn = books[lab]
        pnl, nn = pnl.reindex(idx), nn.reindex(idx).fillna(0)
        for g in (0.50, 0.60, 0.70, 0.80):
            cash = (meta_df["gap"] < g).fillna(True)
            add(f"{lab} | 高低差<{g:.2f}则空", apply_cash(pnl, cash), cash, nn, lab)
            live, bch = flatten_book(pnl, 1.10)
            cash2 = cash | bch
            add(f"{lab} | 高低差<{g:.2f}空 + BOOK≥1.10", apply_cash(pnl, cash2), cash2, nn, lab)
        for t in (0.85, 0.90):
            cash = ~((meta_df["top"] >= t) & (meta_df["bot"] <= 1 - t)).fillna(False)
            add(f"{lab} | 非两端{t:.2f}则空", apply_cash(pnl, cash), cash, nn, lab)
            live, bch = flatten_book(pnl, 1.10)
            cash2 = cash | bch
            add(f"{lab} | 非两端{t:.2f}空 + BOOK≥1.10", apply_cash(pnl, cash2), cash2, nn, lab)

    def keep(r):
        sh = r.get("sharpe") or 0
        dd = r.get("max_dd") or 0
        cash = r.get("cash") or 0
        mx = r.get("max_names") or 99
        return sh >= 0.6 and dd > -0.18 and cash >= 0.35 and mx <= 4

    good = [r for r in rows_out if keep(r)]
    good.sort(key=lambda r: (-(r["cash"] or 0), r.get("avg_names_on") or 9, -(r["sharpe"] or 0), -(r["max_dd"] or -9)))
    # also: cash>=40%, names<=2, sh>=0.7
    thin = [
        r for r in rows_out
        if (r.get("cash") or 0) >= 0.40
        and (r.get("avg_names_on") or 9) <= 2.2
        and (r.get("sharpe") or 0) >= 0.7
        and (r.get("max_dd") or 0) > -0.20
    ]
    thin.sort(key=lambda r: (-(r["sharpe"] or 0), -(r["max_dd"] or -9), -(r["cash"] or 0)))

    print(f"n_schemes={len(rows_out)}  cash>=35% names<=4 sh>=0.6 dd>-18%: {len(good)}", flush=True)
    print("MORE CASH / FEWER NAMES (ok quality):", flush=True)
    for r in good[:18]:
        print(
            f"  {r['scheme']:52} cash={r['cash']:.0%} names={r['avg_names_on']}/{r['max_names']} "
            f"sh={r['sharpe']:.2f} dd={r['max_dd']:.1%} neg={r['n_neg_years']} 26={r['sharpe_2026']}",
            flush=True,
        )
    print("THIN: cash>=40% ~2 names sh>=0.7 dd>-20%:", flush=True)
    for r in thin[:12]:
        print(
            f"  {r['scheme']:52} cash={r['cash']:.0%} names={r['avg_names_on']}/{r['max_names']} "
            f"sh={r['sharpe']:.2f} dd={r['max_dd']:.1%} neg={r['n_neg_years']} 26={r['sharpe_2026']}",
            flush=True,
        )

    # Pareto-ish: among cash>=0.35, names<=4, sh>=0.8, dd>-0.12
    tight = [
        r for r in rows_out
        if (r.get("cash") or 0) >= 0.35
        and (r.get("max_names") or 99) <= 4
        and (r.get("sharpe") or 0) >= 0.8
        and (r.get("max_dd") or 0) > -0.12
    ]
    tight.sort(key=lambda r: (-(r["sharpe"] or 0), -(r["max_dd"] or -9)))
    print("TIGHT sh>=0.8 dd>-12% cash>=35% names<=4:", flush=True)
    for r in tight[:12]:
        print(
            f"  {r['scheme']:52} cash={r['cash']:.0%} names={r['avg_names_on']}/{r['max_names']} "
            f"sh={r['sharpe']:.2f} dd={r['max_dd']:.1%} neg={r['n_neg_years']} 26={r['sharpe_2026']}",
            flush=True,
        )

    slim_keys = (
        "scheme", "size_tag", "return", "sharpe", "max_dd", "n_neg_years", "neg_years",
        "worst_year_sharpe", "sharpe_2026", "return_2026", "cash", "avg_names", "avg_names_on",
        "max_names", "yearly",
    )
    slim = lambda r: {k: r[k] for k in slim_keys if k in r}
    path = OUT_DIR / "pos64_more_cash_fewer_names.json"
    path.write_text(json.dumps({
        "goal": "more cash, fewer names, pos64 only",
        "n_schemes": len(rows_out),
        "good": [slim(r) for r in good[:40]],
        "thin": [slim(r) for r in thin[:20]],
        "tight": [slim(r) for r in tight[:20]],
    }, ensure_ascii=False, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
