#!/usr/bin/env python3
"""Search integer-lot {0,1,2} overlays on 2/4/6-name books.

Fractional vol-target 7% rounded UP to 1 lot destroys the scheme.
Here: round/clip to 0/1/2 (no min-1), plus BOOK flatten / BOOK ladder.
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

from blend_book import book_ratio, zscore
from family_rotate import fmt
from linear_ridge_walkforward import EVAL_END, OUT_DIR, build_panel
from six_name_select import pick_ls_n
from three_factor_24_search import flatten_book, pack
from two_model_rotate import BT_START
from two_name_dd_control import flatten_after_loss
from two_name_select import apply_hold

FACTORS = [("pos_64", 1, "pos64"), ("rsi", 1, "rsi"), ("price_volume_ratio", -1, "pvr")]
WIN = 20
CAP = 2


def scale_of(raw: pd.Series, target: float) -> pd.Series:
    vol = raw.rolling(WIN).std(ddof=1) * np.sqrt(252)
    sc = (target / vol.shift(1)).replace([np.inf, -np.inf], np.nan)
    return sc.clip(lower=0.0, upper=2.5).fillna(1.0)


def apply_lots(raw: pd.Series, lots: pd.Series) -> tuple[pd.Series, pd.Series]:
    lots = lots.reindex(raw.index).fillna(0).astype(int)
    pnl = raw * lots
    cash = lots <= 0
    return pnl, cash


def lots_round(sc: pd.Series, cap: int = CAP) -> pd.Series:
    """round(scale) → {0,1,2}; 0.34→0 not 1."""
    return sc.round().clip(lower=0, upper=cap).astype(int)


def lots_cut(sc: pd.Series, t0: float, t2: float, cap: int = CAP) -> pd.Series:
    out = np.where(sc.to_numpy() < t0, 0, np.where(sc.to_numpy() >= t2, cap, 1))
    return pd.Series(out, index=sc.index, dtype=int)


def lots_book_ladder(raw: pd.Series, t1: float, t2: float) -> pd.Series:
    """BOOK of 1-lot book (lagged): calm→2, mid→1, hot→0."""
    r = book_ratio(raw)
    x = r.to_numpy()
    out = np.ones(len(raw), dtype=int)
    out[np.isnan(x)] = 1
    out[x < t1] = 2
    out[x >= t2] = 0
    return pd.Series(out, index=raw.index)


def slim(st: dict) -> dict:
    keys = (
        "scheme", "return", "sharpe", "max_dd", "n_neg_years", "neg_years",
        "worst_year_sharpe", "sharpe_2026", "return_2026", "cash",
        "avg_names", "avg_names_on", "max_names", "mean_lots", "p90_lots",
    )
    return {k: st[k] for k in keys if k in st}


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=["fwd_ret", "pos_64", "rsi", "price_volume_ratio"])
    books: dict[str, pd.Series] = {}
    print("daily books...", flush=True)
    acc: dict[str, list] = {k: [] for k in (
        "pos64-2", "pos64-4", "pos64-6",
        "rsi-4", "pvr-4", "z-4", "z-6",
        "pos64+pvr-2", "pos64+rsi-2",
    )}
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        day = day.drop_duplicates("symbol").reset_index(drop=True)
        p64 = day["pos_64"].to_numpy()
        rsi = day["rsi"].to_numpy()
        pvr = -day["price_volume_ratio"].to_numpy()
        z = zscore(p64) + zscore(rsi) + zscore(pvr)
        a2 = pick_ls_n(day, p64, 1)
        a4 = pick_ls_n(day, p64, 2)
        a6 = pick_ls_n(day, p64, 3)
        r4 = pick_ls_n(day, rsi, 2)
        v2 = pick_ls_n(day, pvr, 1)
        v4 = pick_ls_n(day, pvr, 2)
        z4 = pick_ls_n(day, z, 2)
        z6 = pick_ls_n(day, z, 3)
        rsi2 = pick_ls_n(day, rsi, 1)
        acc["pos64-2"].append((dt, apply_hold(day, a2)))
        acc["pos64-4"].append((dt, apply_hold(day, a4)))
        acc["pos64-6"].append((dt, apply_hold(day, a6)))
        acc["rsi-4"].append((dt, apply_hold(day, r4)))
        acc["pvr-4"].append((dt, apply_hold(day, v4)))
        acc["z-4"].append((dt, apply_hold(day, z4)))
        acc["z-6"].append((dt, apply_hold(day, z6)))
        acc["pos64+pvr-2"].append((dt, 0.5 * (apply_hold(day, a2) + apply_hold(day, v2))))
        acc["pos64+rsi-2"].append((dt, 0.5 * (apply_hold(day, a2) + apply_hold(day, rsi2))))

    nmap = {
        "pos64-2": 2, "pos64-4": 4, "pos64-6": 6,
        "rsi-4": 4, "pvr-4": 4, "z-4": 4, "z-6": 6,
        "pos64+pvr-2": 4, "pos64+rsi-2": 4,
    }
    for k, rows in acc.items():
        books[k] = pd.Series({d: v for d, v in rows}).sort_index().astype(float)

    rows = []

    def add(name: str, pnl: pd.Series, cash: pd.Series, n_on: int, lots: pd.Series | None = None) -> None:
        nn = pd.Series(float(n_on), index=pnl.index)
        nn = nn.where(~cash.reindex(pnl.index).fillna(False), 0.0)
        st = pack(pnl, cash, nn)
        st["scheme"] = name
        if lots is not None:
            lt = lots.reindex(pnl.index).fillna(0)
            st["mean_lots"] = fmt(float(lt.mean()), 3)
            st["p90_lots"] = fmt(float(lt.quantile(0.9)), 2)
            st["max_lots"] = int(lt.max())
        else:
            inn = ~cash.reindex(pnl.index).fillna(False)
            st["mean_lots"] = fmt(float(inn.mean()), 3)
            st["p90_lots"] = 1.0
            st["max_lots"] = 1
        rows.append(st)

    print("grids...", flush=True)
    tgts = (0.05, 0.06, 0.07, 0.08, 0.10, 0.12)
    t0s = (0.35, 0.45, 0.50, 0.60, 0.70, 0.80)
    t2s = (1.30, 1.50, 1.80, 9.0)
    book_thrs = (1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.25)
    ladder = [(0.90, 1.10), (0.95, 1.12), (1.00, 1.14), (1.00, 1.18), (0.90, 1.14), (0.85, 1.20)]

    for lab, raw in books.items():
        n_on = nmap[lab]
        add(f"{lab} | 1手", raw, raw.abs() < 1e-12, n_on, pd.Series(1, index=raw.index))
        for thr in book_thrs:
            live, cash = flatten_book(raw, thr)
            lots = (~cash).astype(int)
            add(f"{lab} | 1手 BOOK≥{thr:.2f}空", live, cash, n_on, lots)

        cool = flatten_after_loss(raw, -0.02, 5)
        add(f"{lab} | 1手 日亏2%空5日", cool, cool.abs() < 1e-12, n_on, (cool.abs() > 1e-12).astype(int))
        live, cash = flatten_book(cool, 1.14)
        add(f"{lab} | 1手 日亏2%空5日+BOOK≥1.14", live, cash, n_on, (~cash).astype(int))

        for t1, t2 in ladder:
            lots = lots_book_ladder(raw, t1, t2)
            pnl, cash = apply_lots(raw, lots)
            add(f"{lab} | BOOK档 2手<{t1:.2f} / 0手≥{t2:.2f}", pnl, cash, n_on, lots)

        for tgt in tgts:
            sc = scale_of(raw, tgt)
            lots = lots_round(sc)
            pnl, cash = apply_lots(raw, lots)
            add(f"{lab} | 波动{int(tgt*100)}% round{{0,1,2}}", pnl, cash, n_on, lots)
            live, cash2 = flatten_book(pnl, 1.14)
            lots2 = lots.where(~cash2, 0)
            add(f"{lab} | 波动{int(tgt*100)}% round{{0,1,2}}+BOOK≥1.14", live, cash2, n_on, lots2)

            for t0 in t0s:
                for t2 in t2s:
                    lots = lots_cut(sc, t0, t2)
                    pnl, cash = apply_lots(raw, lots)
                    cap_s = "2手" if t2 < 8 else "不上2"
                    add(
                        f"{lab} | 波动{int(tgt*100)}% <{t0:.2f}空 ≥{t2:.2f}{cap_s}",
                        pnl, cash, n_on, lots,
                    )

    def ok(r, sh=0.8, dd=-0.12, y26=0.0, neg=2, names=6):
        return (
            (r.get("sharpe") or 0) >= sh
            and (r.get("max_dd") or 0) > dd
            and (r.get("sharpe_2026") or -9) >= y26
            and (r.get("n_neg_years") or 9) <= neg
            and (r.get("max_names") or 99) <= names
        )

    rows.sort(key=lambda r: (-(r.get("sharpe") or 0), -(r.get("max_dd") or -9)))
    hits4 = [r for r in rows if ok(r, 1.0, -0.10, 0.0, 2, 4)]
    good4 = [r for r in rows if ok(r, 0.80, -0.12, 0.3, 2, 4)]
    good6 = [r for r in rows if ok(r, 0.80, -0.12, 0.3, 2, 6)]
    y26 = [r for r in rows if (r.get("sharpe_2026") or -9) >= 0.8 and (r.get("sharpe") or 0) >= 0.7]
    y26.sort(key=lambda r: (-(r.get("sharpe") or 0), -(r.get("sharpe_2026") or -9)))

    def pr(title, lst, n=20):
        print(title, len(lst), flush=True)
        for r in lst[:n]:
            print(
                f"  {r['scheme'][:72]:72} sh={r['sharpe']:.3f} dd={r['max_dd']:.1%} "
                f"cash={r['cash']:.0%} neg={r['n_neg_years']} 26={r['sharpe_2026']} "
                f"lots={r.get('mean_lots')} names={r['max_names']}",
                flush=True,
            )

    pr("HITS 4名 sh>=1 dd>-10% 26>=0", hits4)
    pr("GOOD 4名 sh>=0.8 dd>-12% 26>=0.3", good4)
    pr("GOOD ≤6名", good6, 15)
    pr("2026 sh>=0.8 且全样本>=0.7", y26, 15)

    # best pos64-4 regardless
    p64 = [r for r in rows if r["scheme"].startswith("pos64-4")]
    p64.sort(key=lambda r: (-(r.get("sharpe") or 0), -(r.get("max_dd") or -9)))
    pr("BEST pos64-4 (by sharpe)", p64, 12)
    p64_26 = sorted(p64, key=lambda r: (-(r.get("sharpe_2026") or -9), -(r.get("sharpe") or 0)))
    pr("BEST pos64-4 (by 2026)", p64_26, 8)

    path = OUT_DIR / "pos64_integer_lot_search.json"
    path.write_text(json.dumps({
        "note": "integer lots {0,1,2}; round(scale) does NOT bump 0.34 up to 1",
        "n": len(rows),
        "hits4": [slim(r) for r in hits4[:40]],
        "good4": [slim(r) for r in good4[:40]],
        "good6": [slim(r) for r in good6[:40]],
        "y2026": [slim(r) for r in y26[:40]],
        "best_pos64_4": [slim(r) for r in p64[:30]],
    }, ensure_ascii=False, indent=2, default=str))
    print("saved", path, "n", len(rows), flush=True)


if __name__ == "__main__":
    main()
