#!/usr/bin/env python3
"""At most 6 names on the quarterly pos64/Ridge book (2010-2026)."""

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
from two_model_rotate import BT_START, choose_model, daily_mkt
from two_name_dd_control import flatten_after_loss
from two_name_select import apply_hold, turnover


def pick_ls_n(day: pd.DataFrame, score: np.ndarray, n_each: int) -> list[tuple[str, float]]:
    s = pd.Series(score, index=day.index)
    s = s[s.notna() & day["fwd_ret"].notna()]
    if len(s) < 2:
        return []
    k = min(n_each, len(s) // 2)
    if k < 1:
        return []
    longs = s.nlargest(k)
    shorts = s.drop(index=longs.index).nsmallest(k)
    w = 0.5 / k
    out = [(day.loc[i, "symbol"], w) for i in longs.index]
    out += [(day.loc[i, "symbol"], -w) for i in shorts.index]
    return out


def pick_abs_n(day: pd.DataFrame, score: np.ndarray, n: int) -> list[tuple[str, float]]:
    s = pd.Series(score, index=day.index)
    s = s[s.notna() & day["fwd_ret"].notna()]
    if s.empty:
        return []
    top = s.abs().nlargest(min(n, len(s)))
    w = 1.0 / len(top)
    return [
        (day.loc[i, "symbol"], (1.0 if s.loc[i] >= 0 else -1.0) * w)
        for i in top.index
    ]


def ser(pairs) -> pd.Series:
    return pd.Series({d: v for d, v in pairs}).sort_index().dropna()


def year_eq(s: pd.Series) -> list[dict]:
    eq = np.exp(s.cumsum())
    return [{"date": str(i.date())[:4], "equity": float(v)} for i, v in eq.resample("YE").last().items()]


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
    return obj


def main() -> None:
    print("loading panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=["fwd_ret", "pos_64"])
    feats = [c for c in FEATURE_NAMES if c not in SKIP_FEATS and c in panel.columns]
    mkt = daily_mkt(panel)
    print("precomputing ICs...", flush=True)
    ic_map = {f: rank_ic_daily(panel, f) for f in feats}

    rules = ["ls2", "ls4", "ls6", "abs6", "unconst"]
    pnls: dict[str, list] = {k: [] for k in rules}
    turns: dict[str, list] = {k: [] for k in rules}
    last_w: dict[str, list | None] = {k: None for k in rules}
    q_summary = []

    quarters = pd.period_range("2010Q3", "2026Q3", freq="Q")
    for q in quarters:
        q_start, q_end = q.start_time, min(q.end_time, EVAL_END)
        test = panel[(panel["date"] >= q_start) & (panel["date"] <= q_end)].copy()
        train = panel[panel["date"] < q_start]
        if test["date"].nunique() < 15 or test["date"].min() < BT_START:
            continue

        pick, _ = choose_model(mkt, q_start)
        if pick == "ridge" and train["date"].nunique() >= MIN_TRAIN_DAYS:
            ranked = [{"feature": f, **ic_stats(ic_map[f].loc[ic_map[f].index < q_start])} for f in feats]
            rtab = pd.DataFrame(ranked).sort_values("t", key=lambda s: s.abs(), ascending=False)
            usable = rtab[(rtab["t"].abs() >= 1.5) & rtab["ic"].notna()]
            use = usable["feature"].head(12).tolist() or rtab.head(8)["feature"].tolist()
            mu, sd, beta = fit_ridge(train, use, l2=L2)
            test["score"] = predict_ridge(test, use, mu, sd, beta)
        else:
            pick = "pos64"
            test["score"] = test["pos_64"]

        q_pnl = {k: [] for k in rules}
        names0: dict[str, list[str]] = {}
        for dt, day in test.groupby("date"):
            sc = day["score"].to_numpy()
            ws = {
                "ls2": pick_ls_n(day, sc, 1),
                "ls4": pick_ls_n(day, sc, 2),
                "ls6": pick_ls_n(day, sc, 3),
                "abs6": pick_abs_n(day, sc, 6),
                "unconst": None,
            }
            if dt == test["date"].min():
                names0 = {k: [s for s, _ in w] for k, w in ws.items() if w is not None}
            for k, w in ws.items():
                if k == "unconst":
                    pnl = ls_pnl(day, sc)
                else:
                    pnl = apply_hold(day, w)
                    turns[k].append(turnover(last_w[k], w))
                    last_w[k] = w
                q_pnl[k].append((dt, pnl))
                pnls[k].append((dt, pnl))

        rec = {
            "quarter": str(q),
            "model": pick,
            "n_symbols": int(test["symbol"].nunique()),
            "thin": test["symbol"].nunique() < 20,
            "names": names0,
        }
        for k in rules:
            rec[k] = metrics_from_pnl(pd.Series({d: v for d, v in q_pnl[k]}))["return"]
        q_summary.append(rec)
        print(
            f"{q} {pick:5} 2={rec['ls2']:+.1%} 4={rec['ls4']:+.1%} "
            f"6={rec['ls6']:+.1%} abs6={rec['abs6']:+.1%} u={rec['unconst']:+.1%} "
            f"open6={names0.get('ls6')}",
            flush=True,
        )

    series = {k: ser(pnls[k]) for k in rules}
    series["ls6_cool"] = flatten_after_loss(series["ls6"], -0.02, 5)

    full = {}
    for k, s in series.items():
        m = metrics_from_pnl(s)
        if k in turns and turns[k]:
            m["turnover"] = float(np.mean(turns[k]))
        m["time_in_mkt"] = float((s.abs() > 1e-12).mean())
        full[k] = m
        print(
            f"{k:10} ret={m['return']:+.1%} sh={m['sharpe']:.2f} "
            f"dd={m['max_dd']:.1%} to={m.get('turnover', float('nan')):.2f} "
            f"in={m['time_in_mkt']:.0%}",
            flush=True,
        )

    keep = ["ls2", "ls4", "ls6", "ls6_cool", "unconst"]
    payload = {
        "rule": "quarterly pos64/Ridge switch; cap names; each side 50% equal-weight",
        "methods": {
            "ls2": "daily 1 long + 1 short",
            "ls4": "daily 2 long + 2 short",
            "ls6": "daily 3 long + 3 short",
            "abs6": "daily top-6 |score|, sign of score",
            "unconst": "daily long top 20% / short bottom 20%",
            "ls6_cool": "ls6 + flatten 5d after daily loss >2%",
        },
        "full": {k: {kk: fmt(vv, 4 if kk != "sharpe" else 3) if kk != "days" else vv for kk, vv in full[k].items()} for k in full},
        "equity_year": {k: year_eq(series[k]) for k in keep},
        "quarters": [
            {
                "quarter": r["quarter"],
                "model": r["model"],
                "n_symbols": r["n_symbols"],
                "ls2": fmt(r["ls2"], 4),
                "ls4": fmt(r["ls4"], 4),
                "ls6": fmt(r["ls6"], 4),
                "abs6": fmt(r["abs6"], 4),
                "unconst": fmt(r["unconst"], 4),
                "open6": r["names"].get("ls6"),
            }
            for r in q_summary
        ],
    }
    path = OUT_DIR / "six_name_select.json"
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
