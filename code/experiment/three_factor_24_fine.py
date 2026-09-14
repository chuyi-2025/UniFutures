#!/usr/bin/env python3
"""Fine grid around 4-name pos64 and pos64+PVR overlay: vol target × BOOK."""

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

from blend_book import book_ratio
from family_rotate import fmt, year_pack
from linear_ridge_walkforward import EVAL_END, OUT_DIR, build_panel
from six_name_select import pick_ls_n
from three_factor_24_search import combine, flatten_book, pack, vol_scale
from two_model_rotate import BT_START
from two_name_select import apply_hold


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=["fwd_ret", "pos_64", "price_volume_ratio"])
    p64_4, pvr_2, p64_2 = [], [], []
    print("daily...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        a = pick_ls_n(day, day["pos_64"].to_numpy(), 2)
        b = pick_ls_n(day, -day["price_volume_ratio"].to_numpy(), 1)
        c = pick_ls_n(day, day["pos_64"].to_numpy(), 1)
        p64_4.append((dt, apply_hold(day, a)))
        pvr_2.append((dt, apply_hold(day, b)))
        p64_2.append((dt, apply_hold(day, c)))

    s64_4 = pd.Series({d: v for d, v in p64_4}).sort_index()
    ov = (pd.Series({d: v for d, v in p64_2}) + pd.Series({d: v for d, v in pvr_2})) / 2
    ov = ov.reindex(s64_4.index)
    nn4 = pd.Series(4.0, index=s64_4.index)
    nn_ov = pd.Series(4.0, index=s64_4.index)  # upper bound; search already measured 3.78

    rows = []

    def add(name, pnl, cash, nn, typical):
        st = pack(pnl, cash, nn.where(~cash, 0))
        st["scheme"] = name
        st["typical"] = typical
        rows.append(st)

    tgts = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    thrs = [1.08, 1.10, 1.12, 1.14, 1.15, 1.16, 1.18, 1.20]
    for lab, raw, nn, typ in (("pos64-4", s64_4, nn4, 4), ("pos64+价量叠2", ov, nn_ov, 4)):
        for tgt in tgts:
            vs = vol_scale(raw, tgt)
            add(f"{lab} | 波动{int(tgt*100)}%", vs, vs.abs() < 1e-12, nn, typ)
            for thr in thrs:
                live, cash = flatten_book(vs, thr)
                add(f"{lab} | 波动{int(tgt*100)}%+BOOK≥{thr:.2f}", live, cash, nn, typ)
                live, cash = flatten_book(raw, thr)
                live2 = vol_scale(live, tgt)
                add(f"{lab} | BOOK≥{thr:.2f}后波动{int(tgt*100)}%", live2, cash, nn, typ)

    def ok(r, cash_min=0.0):
        return (
            (r.get("sharpe") or 0) >= 1.0
            and (r.get("max_dd") or 0) > -0.10
            and (r.get("cash") or 0) >= cash_min
        )

    hits = [r for r in rows if ok(r, 0)]
    hits20 = [r for r in rows if ok(r, 0.20)]
    near = [
        r for r in rows
        if (r.get("sharpe") or 0) >= 0.95
        and (r.get("max_dd") or 0) > -0.105
    ]
    hits.sort(key=lambda r: (-(r["sharpe"] or 0), -(r["max_dd"] or -9)))
    hits20.sort(key=lambda r: (-(r["sharpe"] or 0), -(r["max_dd"] or -9)))
    near.sort(key=lambda r: (-(r["sharpe"] or 0), -(r["max_dd"] or -9)))

    print(f"hits sh>=1 dd>-10%: {len(hits)}  with cash>=20%: {len(hits20)}", flush=True)
    print("HITS", flush=True)
    for r in hits[:15]:
        print(f"  {r['scheme']:48} sh={r['sharpe']:.3f} dd={r['max_dd']:.1%} cash={r['cash']:.0%} neg={r['n_neg_years']} 26={r['sharpe_2026']} ret={r['return']:+.0%}", flush=True)
    print("HITS cash>=20%", flush=True)
    for r in hits20[:15]:
        print(f"  {r['scheme']:48} sh={r['sharpe']:.3f} dd={r['max_dd']:.1%} cash={r['cash']:.0%} neg={r['n_neg_years']} 26={r['sharpe_2026']}", flush=True)
    print("NEAR sh>=0.95 dd>-10.5%", flush=True)
    for r in near[:20]:
        print(f"  {r['scheme']:48} sh={r['sharpe']:.3f} dd={r['max_dd']:.1%} cash={r['cash']:.0%} neg={r['n_neg_years']} 26={r['sharpe_2026']}", flush=True)

    slim_keys = (
        "scheme", "typical", "return", "sharpe", "max_dd", "n_neg_years", "neg_years",
        "worst_year_sharpe", "sharpe_2026", "return_2026", "cash", "yearly",
    )
    slim = lambda r: {k: r[k] for k in slim_keys if k in r}
    path = OUT_DIR / "three_factor_24_fine.json"
    path.write_text(json.dumps({
        "hits": [slim(r) for r in hits],
        "hits_cash20": [slim(r) for r in hits20],
        "near": [slim(r) for r in near[:30]],
    }, ensure_ascii=False, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
