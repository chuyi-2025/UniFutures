#!/usr/bin/env python3
"""Integer 1-lot search with broker margin: drop expensive names, then 3L3S/2L2S.

Universe cap = max 1-lot broker margin (CNY). Account = 1,000,000 CNY.
r = log1p(pnl_cny / 1e6). Cash 20-50%, Sharpe>=1, MDD>-10%, max 6 names.
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
from futures_lot_specs import lot_margin
from linear_ridge_walkforward import EVAL_END, OUT_DIR, build_panel
from pos64_integer_lot_search import apply_lots, lots_cut, scale_of, slim
from pos64_margin_integer_search import dollar_1lot, load_close, net_picks, to_log, CAPITAL
from six_name_select import pick_ls_n
from three_factor_24_search import flatten_book, pack
from two_model_rotate import BT_START

CAPS = (20_000, 30_000, 50_000, 80_000, 10_000_000)
CAP_LAB = {20_000: "一手≤2万", 30_000: "一手≤3万", 50_000: "一手≤5万", 80_000: "一手≤8万", 10_000_000: "不限保证金"}


def attach_margin(day: pd.DataFrame) -> pd.DataFrame:
    mars = [lot_margin(s, p)[0] for s, p in zip(day["symbol"], day["close"])]
    out = day.copy()
    out["lot_margin"] = mars
    return out


def pick_cap(day: pd.DataFrame, score: np.ndarray, n_each: int, cap: float) -> list[tuple[str, float]]:
    d = attach_margin(day)
    d["__sc"] = score
    d = d[d["lot_margin"].notna() & (d["lot_margin"] <= cap)].reset_index(drop=True)
    if len(d) < 2 * n_each:
        return []
    return pick_ls_n(d, d["__sc"].to_numpy(), n_each)


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
    print("closes...", flush=True)
    panel = panel.merge(load_close(), on=["date", "symbol", "code"], how="left")

    recs: dict[str, list] = {}
    print("daily dollar books...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        day = day.drop_duplicates("symbol").reset_index(drop=True)
        p64, rsi, pvr = day["pos_64"].to_numpy(), day["rsi"].to_numpy(), -day["price_volume_ratio"].to_numpy()
        z = zscore(p64) + zscore(rsi) + zscore(pvr)
        for cap in CAPS:
            cl = CAP_LAB[cap]
            specs = [
                (f"pos64-4|{cl}", p64, 2),
                (f"pos64-6|{cl}", p64, 3),
                (f"rsi-4|{cl}", rsi, 2),
                (f"rsi-6|{cl}", rsi, 3),
                (f"pvr-4|{cl}", pvr, 2),
                (f"pvr-6|{cl}", pvr, 3),
                (f"z-6|{cl}", z, 3),
            ]
            picked = {}
            for name, sc, n in specs:
                picks = pick_cap(day, sc, n, cap)
                picked[name] = picks
                recs.setdefault(name, []).append((dt, dollar_1lot(day, picks)))
            a = pick_cap(day, p64, 1, cap)
            v = pick_cap(day, pvr, 1, cap)
            r = pick_cap(day, rsi, 1, cap)
            recs.setdefault(f"pos64+价量叠2|{cl}", []).append((dt, dollar_1lot(day, net_picks(a, v))))
            recs.setdefault(f"pos64+rsi叠2|{cl}", []).append((dt, dollar_1lot(day, net_picks(a, r))))

    series = {}
    for lab, rows in recs.items():
        pnl = pd.Series({d: x[0] for d, x in rows}).sort_index()
        mar = pd.Series({d: x[1] for d, x in rows}).sort_index()
        nn = pd.Series({d: x[3] for d, x in rows}).sort_index()
        r = to_log(pnl / CAPITAL)
        series[lab] = {"r": r, "mar": mar, "nn": nn, "pnl": pnl}

    out_rows = []

    def add(name, lab, r, cash, lots=None):
        info = series[lab]
        nn = info["nn"].reindex(r.index).fillna(0)
        nn = nn.where(~cash.reindex(r.index).fillna(False), 0.0)
        st = pack(r, cash, nn)
        st["scheme"] = name
        inn = ~cash.reindex(r.index).fillna(False)
        mar = info["mar"].reindex(r.index)
        st["mean_margin"] = fmt(float(mar[inn].mean()) if inn.any() else 0, 0)
        st["p90_margin"] = fmt(float(mar[inn].quantile(0.9)) if inn.any() else 0, 0)
        if lots is not None:
            st["mean_lots"] = fmt(float(lots.reindex(r.index).fillna(0).mean()), 3)
        out_rows.append(st)

    print("grids...", flush=True)
    for lab, info in series.items():
        r = info["r"]
        add(f"{lab} | 1手", lab, r, r.abs() < 1e-15)
        for thr in (1.10, 1.14, 1.18, 1.22):
            live, cash = flatten_book(r, thr)
            add(f"{lab} | 1手 BOOK≥{thr:.2f}空", lab, live, cash)
        for tgt in (0.06, 0.07, 0.08):
            sc = scale_of(r, tgt)
            for t0 in (0.55, 0.65, 0.70, 0.75, 0.80):
                lots = lots_cut(sc, t0, 9.0)
                live, cash = apply_lots(r, lots)
                add(f"{lab} | 波动{int(tgt*100)}% <{t0:.2f}空 1手", lab, live, cash, lots)

    hits = [r for r in out_rows if ok(r)]
    hits.sort(key=lambda r: (-(r.get("sharpe") or 0), -(r.get("max_dd") or -9), -(r.get("sharpe_2026") or -9)))
    near = [
        x for x in out_rows
        if (x.get("sharpe") or 0) >= 0.8
        and (x.get("max_dd") or 0) > -0.12
        and 0.15 <= (x.get("cash") or 0) <= 0.55
        and (x.get("max_names") or 99) <= 6
    ]
    near.sort(key=lambda r: (-(r.get("sharpe") or 0), -(r.get("max_dd") or -9)))

    def pr(title, lst, n=25):
        print(title, len(lst), flush=True)
        for r in lst[:n]:
            print(
                f"  {r['scheme'][:78]:78} sh={r['sharpe']:.3f} dd={r['max_dd']:.1%} "
                f"cash={r['cash']:.0%} 26={r['sharpe_2026']} neg={r['n_neg_years']} "
                f"保证金≈{r.get('mean_margin')} names={r['max_names']}",
                flush=True,
            )

    pr("HITS 100万 1手 保证金上限 cash20-50% sh>=1 dd>-10%", hits)
    pr("NEAR sh>=0.8 dd>-12% cash15-55%", near, 20)

    path = OUT_DIR / "pos64_margin_cap_search.json"

    def slim2(r):
        o = slim(r)
        for k in ("mean_margin", "p90_margin", "mean_lots"):
            if k in r:
                o[k] = r[k]
        return o

    path.write_text(json.dumps({
        "capital": CAPITAL,
        "n": len(out_rows),
        "n_hits": len(hits),
        "hits": [slim2(r) for r in hits[:50]],
        "near": [slim2(r) for r in near[:40]],
    }, ensure_ascii=False, indent=2, default=str))
    print("saved", path, "n", len(out_rows), "hits", len(hits), flush=True)


if __name__ == "__main__":
    main()
