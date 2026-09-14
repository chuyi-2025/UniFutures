#!/usr/bin/env python3
"""BOOK name-ladder: 6→4→2→0 on pos64 / RSI / z-blend / vote."""

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
from family_rotate import fmt, year_pack
from linear_ridge_walkforward import EVAL_END, OUT_DIR, build_panel
from six_name_select import pick_ls_n
from three_factor_kname_book_cash import clip_ls, pack
from two_model_rotate import BT_START
from two_name_select import apply_hold

FACTORS = [("pos_64", 1, "pos64"), ("rsi", 1, "rsi"), ("price_volume_ratio", -1, "pvr")]


def ladder(s6, s4, s2, ratio, cuts: tuple[float, float, float]) -> tuple[pd.Series, pd.Series]:
    """cuts = (t4, t2, t0): below t4 use 6, [t4,t2) use 4, [t2,t0) use 2, else cash."""
    t4, t2, t0 = cuts
    r = ratio.reindex(s6.index)
    out = s6.copy()
    st = pd.Series("6", index=s6.index)
    m4 = (r >= t4).fillna(False)
    m2 = (r >= t2).fillna(False)
    m0 = (r >= t0).fillna(False)
    out[m4] = s4.reindex(s6.index)[m4]
    st[m4] = "4"
    out[m2] = s2.reindex(s6.index)[m2]
    st[m2] = "2"
    out[m0] = 0.0
    st[m0] = "0"
    return out, st


def to_s(rows):
    return pd.Series({d: v for d, v in rows}).sort_index().dropna()


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=["fwd_ret", "pos_64", "rsi", "price_volume_ratio"])
    store = {k: {n: [] for n in (1, 2, 3)} for k in ("pos64", "rsi", "pvr", "z", "vote")}
    print("daily...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        zsc = None
        books = {n: [] for n in (1, 2, 3)}
        for col, sgn, name in FACTORS:
            raw = sgn * day[col].to_numpy()
            for n in (1, 2, 3):
                p = pick_ls_n(day, raw, n)
                store[name][n].append((dt, apply_hold(day, p)))
                books[n].append(p)
            z = sgn * zscore(day[col].to_numpy())
            zsc = z if zsc is None else zsc + z
        for n in (1, 2, 3):
            store["z"][n].append((dt, apply_hold(day, pick_ls_n(day, zsc, n))))
            store["vote"][n].append((dt, apply_hold(day, clip_ls(books[n], n))))

    series = {k: {n: to_s(store[k][n]) for n in (1, 2, 3)} for k in store}
    idx = series["pos64"][3].index
    for k in series:
        for n in (1, 2, 3):
            series[k][n] = series[k][n].reindex(idx)

    cuts_list = [
        (1.10, 1.20, 1.30),
        (1.10, 1.15, 1.25),
        (1.15, 1.20, 1.30),
        (1.05, 1.15, 1.25),
        (1.10, 1.10, 1.20),  # 6 then skip 4: 6/2/0 at 1.10/1.20
        (1.15, 1.15, 1.20),
    ]
    # also 6/4/0 (no 2-name rung): t4=t2
    cuts_640 = [(1.10, 1.10, 1.25), (1.15, 1.15, 1.25), (1.10, 1.10, 1.20), (1.15, 1.15, 1.20)]

    rows = []
    for fam, lab in (("pos64", "pos64"), ("rsi", "RSI"), ("z", "z合成"), ("vote", "投票")):
        s6, s4, s2 = series[fam][3], series[fam][2], series[fam][1]
        br = book_ratio(s6)
        for cuts in cuts_list + cuts_640:
            pnl, stt = ladder(s6, s4, s2, br, cuts)
            name = f"{lab} 6/4/2/0 {cuts}" if cuts in cuts_list else f"{lab} 6/4/0 {cuts}"
            if cuts[0] == cuts[1]:
                name = f"{lab} 6→2/0 {cuts[0]:.2f}/{cuts[2]:.2f}" if cuts in cuts_list else f"{lab} 6→4/0 {cuts[0]:.2f}/{cuts[2]:.2f}"
            cash = stt.eq("0")
            if float(cash.mean()) < 0.20:
                continue
            st = pack(pnl, stt, cash)
            st["scheme"] = name
            st["pct_6"] = fmt(float(stt.eq("6").mean()), 3)
            st["pct_4"] = fmt(float(stt.eq("4").mean()), 3)
            st["pct_2"] = fmt(float(stt.eq("2").mean()), 3)
            rows.append(st)

    rows.sort(key=lambda r: (r["n_neg_years"], -(r["worst_year_sharpe"] or -9), -(r["max_dd"] or -9), -(r["sharpe"] or -9)))
    print(f"{'scheme':48} cash sh  dd   neg  w    26", flush=True)
    for r in rows[:20]:
        print(
            f"{r['scheme']:48} {r['cash']:.0%} {r['sharpe']:.2f} {r['max_dd']:.1%} "
            f"n={r['n_neg_years']}{r['neg_years']} w={r['worst_year_sharpe']} 26={r['sharpe_2026']}",
            flush=True,
        )
    zeros = [r for r in rows if r["n_neg_years"] == 0]
    print("0-neg", [(r["scheme"], r["sharpe"], r["max_dd"], r["cash"], r["sharpe_2026"]) for r in zeros], flush=True)
    slim = lambda r: {k: r[k] for k in r if k != "yearly" or True}
    path = OUT_DIR / "three_factor_book_ladder.json"
    keep = ("scheme", "return", "sharpe", "max_dd", "n_neg_years", "neg_years", "worst_year_sharpe",
            "sharpe_std", "sharpe_2026", "return_2026", "cash", "pct_6", "pct_4", "pct_2", "yearly")
    path.write_text(json.dumps({
        "constraint": "BOOK name ladder 6/4/2/0; cash>20%",
        "all": [{k: r[k] for k in keep if k in r} for r in rows],
        "zero_neg": [{k: r[k] for k in keep if k in r} for r in zeros],
    }, ensure_ascii=False, indent=2))
    print("saved", path, "n", len(rows))


if __name__ == "__main__":
    main()
