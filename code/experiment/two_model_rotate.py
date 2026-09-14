#!/usr/bin/env python3
"""Quarterly rotate: pos_64 trend vs Ridge, switch on trailing vol. 2010-2026."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from build_feature.build_xgb_feature import FEATURE_NAMES
from linear_ridge_walkforward import (
    EVAL_END,
    L2,
    LS_FRAC,
    MIN_CS,
    MIN_TRAIN_DAYS,
    OUT_DIR,
    SKIP_FEATS,
    build_panel,
    fit_ridge,
    ic_stats,
    ls_pnl,
    metrics_from_pnl,
    predict_ridge,
    rank_ic_daily,
)

VOL_MULT = 1.2
SHORT_WIN = 20
LONG_WIN = 252
BT_START = pd.Timestamp("2010-07-01")


def daily_mkt(panel: pd.DataFrame) -> pd.Series:
    return panel.groupby("date")["fwd_ret"].mean().sort_index()


def choose_model(mkt: pd.Series, q_start: pd.Timestamp) -> tuple[str, dict]:
    hist = mkt[mkt.index < q_start].dropna()
    if len(hist) < SHORT_WIN + 20:
        return "pos64", {"reason": "warm-up", "vol_short": None, "vol_med": None}
    short = float(hist.tail(SHORT_WIN).std(ddof=1) * np.sqrt(252))
    roll = hist.tail(LONG_WIN).rolling(SHORT_WIN).std(ddof=1) * np.sqrt(252)
    med = float(roll.median()) if roll.notna().sum() >= 20 else float(hist.tail(LONG_WIN).std(ddof=1) * np.sqrt(252))
    info = {"vol_short": short, "vol_med": med, "ratio": short / med if med else None}
    if med and np.isfinite(short) and np.isfinite(med) and short > VOL_MULT * med:
        return "ridge", info
    return "pos64", info


def factor_ls(part: pd.DataFrame, score: np.ndarray) -> pd.Series:
    out = []
    part = part.assign(_s=score)
    for dt, day in part.groupby("date"):
        out.append((dt, ls_pnl(day, day["_s"].to_numpy())))
    return pd.Series({d: v for d, v in out})


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def main() -> None:
    print("loading panel from 2010...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01"))
    panel = panel.dropna(subset=["fwd_ret", "pos_64"])
    feats = [c for c in FEATURE_NAMES if c not in SKIP_FEATS and c in panel.columns]
    mkt = daily_mkt(panel)

    print("precomputing ICs...", flush=True)
    ic_map = {f: rank_ic_daily(panel, f) for f in feats}

    quarters = pd.period_range("2010Q3", "2026Q3", freq="Q")
    q_rows = []
    rot_pnl, p64_pnl, ridge_pnl, bh_pnl = [], [], [], []

    for q in quarters:
        q_start, q_end = q.start_time, min(q.end_time, EVAL_END)
        test = panel[(panel["date"] >= q_start) & (panel["date"] <= q_end)]
        train = panel[panel["date"] < q_start]
        if test["date"].nunique() < 15 or test["symbol"].nunique() < 8:
            print(f"skip {q}", flush=True)
            continue
        if test["date"].min() < BT_START:
            continue

        pick, vol_info = choose_model(mkt, q_start)

        p64 = factor_ls(test, test["pos_64"].to_numpy())
        bh = test.groupby("date")["fwd_ret"].mean()

        ridge_ok = train["date"].nunique() >= MIN_TRAIN_DAYS
        use_feats = []
        if ridge_ok:
            ranked = []
            for f in feats:
                st = ic_stats(ic_map[f].loc[ic_map[f].index < q_start])
                ranked.append({"feature": f, **st})
            rtab = pd.DataFrame(ranked).sort_values("t", key=lambda s: s.abs(), ascending=False)
            usable = rtab[(rtab["t"].abs() >= 1.5) & rtab["ic"].notna()]
            use_feats = usable["feature"].head(12).tolist() or rtab.head(8)["feature"].tolist()
            mu, sd, beta = fit_ridge(train, use_feats, l2=L2)
            test = test.copy()
            test["ridge_s"] = predict_ridge(test, use_feats, mu, sd, beta)
            rg = factor_ls(test, test["ridge_s"].to_numpy())
        else:
            rg = p64.copy()
            if pick == "ridge":
                pick = "pos64"
                vol_info["reason"] = "ridge-warmup-fallback"

        chosen = rg if pick == "ridge" else p64
        thin = bool(test["symbol"].nunique() < 20)

        def pack(s):
            return metrics_from_pnl(s)

        row = {
            "quarter": str(q),
            "start": str(q_start.date()),
            "end": str(test["date"].max().date()),
            "n_symbols": int(test["symbol"].nunique()),
            "n_days": int(test["date"].nunique()),
            "picked": pick,
            "thin": thin,
            "vol_short": vol_info.get("vol_short"),
            "vol_med": vol_info.get("vol_med"),
            "vol_ratio": vol_info.get("ratio"),
            "rotate": pack(chosen),
            "pos64": pack(p64),
            "ridge": pack(rg),
            "buyhold": pack(bh),
            "ridge_features": use_feats[:5],
        }
        q_rows.append(row)
        for dt, v in chosen.items():
            rot_pnl.append((dt, v))
        for dt, v in p64.items():
            p64_pnl.append((dt, v))
        for dt, v in rg.items():
            ridge_pnl.append((dt, v))
        for dt, v in bh.items():
            bh_pnl.append((dt, v))
        print(
            f"{q} pick={pick:5} n={row['n_symbols']:2} "
            f"rot={row['rotate']['return']:+.1%} p64={row['pos64']['return']:+.1%} "
            f"rg={row['ridge']['return']:+.1%} bh={row['buyhold']['return']:+.1%} "
            f"vol={row['vol_ratio'] and round(row['vol_ratio'], 2)}",
            flush=True,
        )

    def series(pairs):
        return pd.Series({d: v for d, v in pairs}).sort_index().dropna()

    rot_s, p64_s, rg_s, bh_s = map(series, (rot_pnl, p64_pnl, ridge_pnl, bh_pnl))
    # exclude thin 2026Q3 from headline if last quarter is thin
    core_end = pd.Timestamp("2026-06-30")
    full = {
        "rotate": metrics_from_pnl(rot_s),
        "pos64": metrics_from_pnl(p64_s),
        "ridge": metrics_from_pnl(rg_s),
        "buyhold": metrics_from_pnl(bh_s),
    }
    core = {
        "rotate": metrics_from_pnl(rot_s[rot_s.index <= core_end]),
        "pos64": metrics_from_pnl(p64_s[p64_s.index <= core_end]),
        "ridge": metrics_from_pnl(rg_s[rg_s.index <= core_end]),
        "buyhold": metrics_from_pnl(bh_s[bh_s.index <= core_end]),
    }

    def to_eq(s: pd.Series):
        eq = np.exp(s.cumsum())
        # quarterly last
        qlast = eq.resample("QE").last()
        return (
            [{"date": str(i.date()), "equity": float(v)} for i, v in qlast.items()],
            [{"date": str(i.date()), "equity": float(v)} for i, v in eq.resample("YE").last().items()],
        )

    eq_q, eq_y = {}, {}
    for name, s in [("rotate", rot_s), ("pos64", p64_s), ("ridge", rg_s), ("buyhold", bh_s)]:
        qe, ye = to_eq(s)
        eq_q[name], eq_y[name] = qe, ye

    n_ridge = sum(1 for r in q_rows if r["picked"] == "ridge" and not r["thin"])
    n_p64 = sum(1 for r in q_rows if r["picked"] == "pos64" and not r["thin"])

    payload = {
        "rule": {
            "default": "pos_64 cross-section long/short 20%",
            "alt": "Ridge refit each quarter on top-12 |t| features, same LS",
            "switch": f"at quarter start, if trailing {SHORT_WIN}d mkt vol > {VOL_MULT}x median of rolling {SHORT_WIN}d vol over {LONG_WIN}d → Ridge else pos64",
            "start": str(BT_START.date()),
            "end": str(rot_s.index.max().date()) if len(rot_s) else None,
        },
        "full": full,
        "through_2026q2": core,
        "picks": {"pos64": n_p64, "ridge": n_ridge},
        "quarters": [
            {
                "quarter": r["quarter"],
                "picked": r["picked"],
                "n_symbols": r["n_symbols"],
                "thin": r["thin"],
                "vol_ratio": fmt(r["vol_ratio"], 3),
                "rotate": fmt(r["rotate"]["return"], 4),
                "pos64": fmt(r["pos64"]["return"], 4),
                "ridge": fmt(r["ridge"]["return"], 4),
                "buyhold": fmt(r["buyhold"]["return"], 4),
                "rotate_sharpe": fmt(r["rotate"]["sharpe"], 3),
            }
            for r in q_rows
        ],
        "equity_quarter": eq_q,
        "equity_year": eq_y,
    }
    path = OUT_DIR / "two_model_rotate.json"
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2))
    print(json.dumps(clean({"full": full, "core": core, "picks": payload["picks"]}), indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
