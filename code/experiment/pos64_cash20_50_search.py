#!/usr/bin/env python3
"""Integer-lot search: cash 20-50%, MDD > -10%, Sharpe>1, max 6 names.

Factors: pos64 / RSI / -price_volume_ratio only. Lots in {0,1} or {0,1,2}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from blend_book import zscore
from family_rotate import fmt
from linear_ridge_walkforward import EVAL_END, OUT_DIR, build_panel
from pos64_integer_lot_search import apply_lots, lots_cut, lots_round, scale_of, slim
from six_name_select import pick_ls_n
from three_factor_24_search import flatten_book, pack
from two_model_rotate import BT_START
from two_name_select import apply_hold

NMAP = {
    "pos64-2": 2, "pos64-4": 4, "pos64-6": 6,
    "rsi-2": 2, "rsi-4": 4, "rsi-6": 6,
    "pvr-2": 2, "pvr-4": 4, "pvr-6": 6,
    "z-2": 2, "z-4": 4, "z-6": 6,
    "pos64+价量叠2": 4, "pos64+rsi叠2": 4, "rsi+价量叠2": 4,
    "三因子叠2": 6,
}


def ok(r) -> bool:
    cash = r.get("cash") or 0
    return (
        (r.get("sharpe") or 0) >= 1.0
        and (r.get("max_dd") or 0) > -0.10
        and 0.20 - 1e-9 <= cash <= 0.50 + 1e-9
        and (r.get("max_names") or 99) <= 6
    )


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=["fwd_ret", "pos_64", "rsi", "price_volume_ratio"])
    acc: dict[str, list] = {k: [] for k in NMAP}
    print("daily books...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        day = day.drop_duplicates("symbol").reset_index(drop=True)
        p64 = day["pos_64"].to_numpy()
        rsi = day["rsi"].to_numpy()
        pvr = -day["price_volume_ratio"].to_numpy()
        z = zscore(p64) + zscore(rsi) + zscore(pvr)
        a1, a2, a3 = pick_ls_n(day, p64, 1), pick_ls_n(day, p64, 2), pick_ls_n(day, p64, 3)
        r1, r2, r3 = pick_ls_n(day, rsi, 1), pick_ls_n(day, rsi, 2), pick_ls_n(day, rsi, 3)
        v1, v2, v3 = pick_ls_n(day, pvr, 1), pick_ls_n(day, pvr, 2), pick_ls_n(day, pvr, 3)
        z1, z2, z3 = pick_ls_n(day, z, 1), pick_ls_n(day, z, 2), pick_ls_n(day, z, 3)
        pa, ra, va = apply_hold(day, a1), apply_hold(day, r1), apply_hold(day, v1)
        acc["pos64-2"].append((dt, pa))
        acc["pos64-4"].append((dt, apply_hold(day, a2)))
        acc["pos64-6"].append((dt, apply_hold(day, a3)))
        acc["rsi-2"].append((dt, ra))
        acc["rsi-4"].append((dt, apply_hold(day, r2)))
        acc["rsi-6"].append((dt, apply_hold(day, r3)))
        acc["pvr-2"].append((dt, va))
        acc["pvr-4"].append((dt, apply_hold(day, v2)))
        acc["pvr-6"].append((dt, apply_hold(day, v3)))
        acc["z-2"].append((dt, apply_hold(day, z1)))
        acc["z-4"].append((dt, apply_hold(day, z2)))
        acc["z-6"].append((dt, apply_hold(day, z3)))
        acc["pos64+价量叠2"].append((dt, 0.5 * (pa + va)))
        acc["pos64+rsi叠2"].append((dt, 0.5 * (pa + ra)))
        acc["rsi+价量叠2"].append((dt, 0.5 * (ra + va)))
        acc["三因子叠2"].append((dt, (pa + ra + va) / 3.0))

    books = {k: pd.Series({d: v for d, v in rows}).sort_index().astype(float) for k, rows in acc.items()}
    rows = []

    def add(name, pnl, cash, n_on, lots=None):
        nn = pd.Series(float(n_on), index=pnl.index)
        nn = nn.where(~cash.reindex(pnl.index).fillna(False), 0.0)
        st = pack(pnl, cash, nn)
        st["scheme"] = name
        if lots is not None:
            lt = lots.reindex(pnl.index).fillna(0)
            st["mean_lots"] = fmt(float(lt.mean()), 3)
            st["max_lots"] = int(lt.max())
        rows.append(st)

    print("grids...", flush=True)
    tgts = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
    t0s = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    t2s = (1.40, 1.60, 1.80, 9.0)
    book_thrs = (1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.22, 1.25, 1.28)

    for lab, raw in books.items():
        n_on = NMAP[lab]
        add(f"{lab} | 1手", raw, raw.abs() < 1e-12, n_on, pd.Series(1, index=raw.index))
        for thr in book_thrs:
            live, cash = flatten_book(raw, thr)
            add(f"{lab} | 1手 BOOK≥{thr:.2f}空", live, cash, n_on, (~cash).astype(int))
        for tgt in tgts:
            sc = scale_of(raw, tgt)
            lots = lots_round(sc, cap=1)  # {0,1} only
            pnl, cash = apply_lots(raw, lots)
            add(f"{lab} | 波动{int(tgt*100)}% round{{0,1}}", pnl, cash, n_on, lots)
            lots2 = lots_round(sc, cap=2)
            pnl, cash = apply_lots(raw, lots2)
            add(f"{lab} | 波动{int(tgt*100)}% round{{0,1,2}}", pnl, cash, n_on, lots2)
            live, cash2 = flatten_book(pnl, 1.14)
            add(f"{lab} | 波动{int(tgt*100)}% round{{0,1,2}}+BOOK≥1.14", live, cash2, n_on, lots2.where(~cash2, 0))
            for t0 in t0s:
                for t2 in t2s:
                    lots = lots_cut(sc, t0, t2)
                    pnl, cash = apply_lots(raw, lots)
                    cap_s = "2手" if t2 < 8 else "不上2"
                    add(f"{lab} | 波动{int(tgt*100)}% <{t0:.2f}空 ≥{t2:.2f}{cap_s}", pnl, cash, n_on, lots)

    hits = [r for r in rows if ok(r)]
    hits.sort(key=lambda r: (
        -(r.get("sharpe") or 0),
        -(r.get("max_dd") or -9),
        -(r.get("sharpe_2026") or -9),
        r.get("n_neg_years") or 9,
    ))
    y26 = [r for r in hits if (r.get("sharpe_2026") or -9) >= 0.5]
    y26.sort(key=lambda r: (-(r.get("sharpe_2026") or -9), -(r.get("sharpe") or 0)))
    near = [
        r for r in rows
        if (r.get("sharpe") or 0) >= 0.95
        and (r.get("max_dd") or 0) > -0.11
        and 0.18 <= (r.get("cash") or 0) <= 0.52
        and (r.get("max_names") or 99) <= 6
    ]
    near.sort(key=lambda r: (-(r.get("sharpe") or 0), -(r.get("max_dd") or -9)))

    def pr(title, lst, n=25):
        print(title, len(lst), flush=True)
        for r in lst[:n]:
            print(
                f"  {r['scheme'][:70]:70} sh={r['sharpe']:.3f} dd={r['max_dd']:.1%} "
                f"cash={r['cash']:.0%} neg={r['n_neg_years']} 26={r['sharpe_2026']} "
                f"names={r['max_names']} lots={r.get('mean_lots')}",
                flush=True,
            )

    pr("HITS cash20-50% sh>=1 dd>-10% names<=6", hits, 30)
    pr("HITS with 2026 sh>=0.5", y26, 20)
    if not hits:
        pr("NEAR sh>=0.95 dd>-11% cash18-52%", near, 20)

    # unique families among hits
    fam = {}
    for r in hits:
        key = r["scheme"].split(" |")[0]
        fam.setdefault(key, r)
    print("best per book", flush=True)
    for k, r in fam.items():
        print(
            f"  {r['scheme'][:70]:70} sh={r['sharpe']:.3f} dd={r['max_dd']:.1%} "
            f"cash={r['cash']:.0%} 26={r['sharpe_2026']} neg={r['n_neg_years']}",
            flush=True,
        )

    path = OUT_DIR / "pos64_cash20_50_search.json"
    path.write_text(json.dumps({
        "goal": "integer lots; cash 20-50%; Sharpe>=1; max_dd>-10%; max 6 names; pos64/RSI/PVR",
        "n": len(rows),
        "n_hits": len(hits),
        "hits": [slim(r) for r in hits[:60]],
        "hits_2026": [slim(r) for r in y26[:40]],
        "best_per_book": [slim(r) for r in fam.values()],
    }, ensure_ascii=False, indent=2, default=str))
    print("saved", path, "n", len(rows), "hits", len(hits), flush=True)


if __name__ == "__main__":
    main()
