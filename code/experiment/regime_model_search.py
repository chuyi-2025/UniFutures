#!/usr/bin/env python3
"""Search which models work in which market regimes (2020-2026, all mains)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from linear_ridge_walkforward import (  # noqa: E402
    EVAL_END,
    EVAL_START,
    OUT_DIR,
    build_panel,
    ls_pnl,
    metrics_from_pnl,
    sign_pnl,
)

RESULTS = Path("/home/workspace/lab/UniFutures/data/results")
WF_JSON = OUT_DIR / "walkforward_report.json"

# name, kind (ls|sign), column expression
FACTORS = [
    ("隔夜反转", "ls", "-ret_1d"),
    ("5日反转", "ls", "-momentum_5"),
    ("20日超跌", "ls", "-momentum_20"),
    ("7日新高反转", "ls", "-ratio_max_7"),
    ("5日乖离反转", "ls", "-bias_5"),
    ("RSI反转", "ls", "-(rsi-50)"),
    ("均线趋势20/60", "ls", "ma_ratio_20_60"),
    ("均线趋势30/60", "ls", "ma_ratio_30_60"),
    ("64日位置趋势", "ls", "pos_64"),
    ("20日动量趋势", "ls", "momentum_20"),
    ("隔夜反转TS", "sign", "-ret_1d"),
    ("20日动量TS", "sign", "momentum_20"),
    ("均线趋势TS", "sign", "ma_ratio_20_60-1"),
]


def score_col(df: pd.DataFrame, expr: str) -> np.ndarray:
    loc = {
        "ret_1d": df.get("ret_1d"),
        "momentum_5": df.get("momentum_5"),
        "momentum_20": df.get("momentum_20"),
        "ratio_max_7": df.get("ratio_max_7"),
        "bias_5": df.get("bias_5"),
        "rsi": df.get("rsi"),
        "ma_ratio_20_60": df.get("ma_ratio_20_60"),
        "ma_ratio_30_60": df.get("ma_ratio_30_60"),
        "pos_64": df.get("pos_64"),
    }
    return np.asarray(eval(expr, {"__builtins__": {}}, loc), dtype=float)  # noqa: S307


def load_existing_ew(model: str) -> pd.Series:
    frames = []
    for path in sorted((RESULTS / model).glob("*/daily.csv")):
        d = pd.read_csv(path, usecols=lambda c: c in {"date", "position", "log_ret_1d"})
        if "position" not in d.columns:
            continue
        d["date"] = pd.to_datetime(d["date"])
        d["pnl"] = d["position"].astype(float) * d["log_ret_1d"].fillna(0.0)
        frames.append(d[["date", "pnl"]])
    if not frames:
        return pd.Series(dtype=float)
    all_d = pd.concat(frames, ignore_index=True)
    return all_d.groupby("date")["pnl"].mean().sort_index()


def q_metrics(pnl: pd.Series) -> dict:
    return metrics_from_pnl(pnl)


def fmt(x, nd=3):
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
    wf = json.loads(WF_JSON.read_text())
    qmeta = {r["quarter"]: r for r in wf["quarters"]}
    ridge_q = {
        r["quarter"]: pd.Series(dtype=float) for r in wf["quarters"]
    }  # filled below from equity daily if needed

    print("loading panel...", flush=True)
    panel = build_panel()
    panel = panel[(panel["date"] >= EVAL_START) & (panel["date"] <= EVAL_END)]

    # daily ridge LS from previous run equity path is not stored; recompute via saved? skip, use quarterly totals only
    model_daily: dict[str, dict[str, pd.Series]] = {}

    print("scoring factors...", flush=True)
    for name, kind, expr in FACTORS:
        print(f"  {name}", flush=True)
        q_pnls = {}
        for q, meta in qmeta.items():
            if meta["n_symbols"] < 20:
                continue
            start, end = pd.Timestamp(meta["start"]), pd.Timestamp(meta["end"])
            part = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
            part["_s"] = score_col(part, expr)
            rows = []
            for dt, day in part.groupby("date"):
                sc = day["_s"].to_numpy()
                v = ls_pnl(day, sc) if kind == "ls" else sign_pnl(day, sc)
                rows.append((dt, v))
            q_pnls[q] = pd.Series({d: v for d, v in rows})
        model_daily[name] = q_pnls

    # Ridge quarterly returns already in wf; reconstruct dummy series of constant daily? use stored ls_return only
    model_daily["Ridge多空(每季重估)"] = {}
    for q, meta in qmeta.items():
        if meta["n_symbols"] < 20 or meta.get("ls_return") is None:
            continue
        # approximate: one-period return, metrics from that single number not comparable
        # instead rebuild nothing — we'll attach quarterly return/sharpe from wf directly
        model_daily["Ridge多空(每季重估)"][q] = None

    print("loading PPO/Kronos/XGB...", flush=True)
    existing = {m: load_existing_ew(m) for m in ("ppo", "kronos", "xgb")}

    # quarterly table
    records = []
    for name, q_pnls in model_daily.items():
        if name == "Ridge多空(每季重估)":
            for q, meta in qmeta.items():
                if meta["n_symbols"] < 20:
                    continue
                records.append(
                    {
                        "model": name,
                        "family": "ridge",
                        "quarter": q,
                        "regime": meta["regime"],
                        "return": meta["ls_return"],
                        "sharpe": meta["ls_sharpe"],
                        "n_symbols": meta["n_symbols"],
                    }
                )
            continue
        for q, pnl in q_pnls.items():
            meta = qmeta[q]
            m = q_metrics(pnl)
            records.append(
                {
                    "model": name,
                    "family": "factor",
                    "quarter": q,
                    "regime": meta["regime"],
                    "return": m["return"],
                    "sharpe": m["sharpe"],
                    "n_symbols": meta["n_symbols"],
                }
            )

    for model, series in existing.items():
        for q, meta in qmeta.items():
            start, end = pd.Timestamp(meta["start"]), pd.Timestamp(meta["end"])
            sl = series[(series.index >= start) & (series.index <= end)]
            if len(sl) < 15:
                continue
            m = q_metrics(sl)
            records.append(
                {
                    "model": model.upper(),
                    "family": "existing",
                    "quarter": q,
                    "regime": meta["regime"],
                    "return": m["return"],
                    "sharpe": m["sharpe"],
                    "n_symbols": None,
                }
            )

    rec = pd.DataFrame(records)

    # regime ranking: factor+ridge on full sample; existing separately
    def rank_block(df: pd.DataFrame) -> list[dict]:
        out = []
        for regime, g in df.groupby("regime"):
            rows = []
            for model, gg in g.groupby("model"):
                rows.append(
                    {
                        "model": model,
                        "n_quarters": int(len(gg)),
                        "mean_return": float(gg["return"].mean()),
                        "median_return": float(gg["return"].median()),
                        "win_rate": float((gg["return"] > 0).mean()),
                        "mean_sharpe": float(gg["sharpe"].mean()),
                    }
                )
            rows = sorted(rows, key=lambda r: (r["mean_return"], r["mean_sharpe"]), reverse=True)
            out.append({"regime": regime, "ranking": rows, "winner": rows[0]["model"] if rows else None})
        return out

    factor_rec = rec[rec["family"].isin(["factor", "ridge"])]
    exist_rec = rec[rec["family"] == "existing"]
    regime_factor = rank_block(factor_rec)
    regime_exist = rank_block(exist_rec) if not exist_rec.empty else []

    # lagged-regime switcher among factor models (no look-ahead)
    factor_names = [n for n, _, _ in FACTORS]
    hist = rec[rec["model"].isin(factor_names)].copy()
    switch_rows = []
    quarters = [q for q, m in qmeta.items() if m["n_symbols"] >= 20]
    for i, q in enumerate(quarters):
        if i == 0:
            pick = "隔夜反转"
            rule = "default"
        else:
            prev_reg = qmeta[quarters[i - 1]]["regime"]
            past = hist[(hist["quarter"].isin(quarters[:i])) & (hist["regime"] == prev_reg)]
            if past.empty:
                pick = "隔夜反转"
                rule = f"lag:{prev_reg}:fallback"
            else:
                means = past.groupby("model")["return"].mean().sort_values(ascending=False)
                pick = str(means.index[0])
                rule = f"lag:{prev_reg}"
        sl = hist[(hist["quarter"] == q) & (hist["model"] == pick)]
        if sl.empty:
            continue
        switch_rows.append(
            {
                "quarter": q,
                "regime": qmeta[q]["regime"],
                "picked": pick,
                "rule": rule,
                "return": float(sl["return"].iloc[0]),
                "sharpe": float(sl["sharpe"].iloc[0]),
            }
        )
    switch = pd.DataFrame(switch_rows)
    switch_full = {
        "return": float(np.prod(1 + switch["return"]) - 1) if len(switch) else None,
        "mean_quarter": float(switch["return"].mean()) if len(switch) else None,
        "win_rate": float((switch["return"] > 0).mean()) if len(switch) else None,
        "n": int(len(switch)),
    }
    # also compound approx via sum of log1p
    if len(switch):
        switch_full["compound"] = float(np.expm1(np.log1p(switch["return"]).sum()))

    # best-in-hindsight oracle (look-ahead, upper bound)
    oracle = []
    for q in quarters:
        sl = hist[hist["quarter"] == q]
        if sl.empty:
            continue
        best = sl.sort_values("return", ascending=False).iloc[0]
        oracle.append(float(best["return"]))
    oracle_full = {
        "compound": float(np.expm1(np.log1p(pd.Series(oracle)).sum())) if oracle else None,
        "mean_quarter": float(np.mean(oracle)) if oracle else None,
    }

    # overall factor ranking
    overall = (
        factor_rec.groupby("model")
        .agg(
            n_quarters=("return", "size"),
            mean_return=("return", "mean"),
            win_rate=("return", lambda s: float((s > 0).mean())),
            mean_sharpe=("sharpe", "mean"),
        )
        .reset_index()
        .sort_values("mean_return", ascending=False)
    )

    payload = {
        "note": {
            "universe": "all main contracts, skip 2026Q3 (9 symbols)",
            "regime": "same labels as walkforward (contemporaneous quarter market)",
            "switcher": "pick model using previous quarter realized regime — no look-ahead",
            "existing": "PPO/Kronos/XGB equal-weight of available symbols, 2024-01 onward only",
        },
        "overall_factors": [
            {
                "model": r.model,
                "n_quarters": int(r.n_quarters),
                "mean_return": fmt(r.mean_return, 4),
                "win_rate": fmt(r.win_rate, 3),
                "mean_sharpe": fmt(r.mean_sharpe, 3),
            }
            for r in overall.itertuples()
        ],
        "regime_factors": [
            {
                "regime": blk["regime"],
                "winner": blk["winner"],
                "ranking": [
                    {
                        "model": r["model"],
                        "n_quarters": r["n_quarters"],
                        "mean_return": fmt(r["mean_return"], 4),
                        "win_rate": fmt(r["win_rate"], 3),
                        "mean_sharpe": fmt(r["mean_sharpe"], 3),
                    }
                    for r in blk["ranking"]
                ],
            }
            for blk in regime_factor
        ],
        "regime_existing_2024": [
            {
                "regime": blk["regime"],
                "winner": blk["winner"],
                "ranking": [
                    {
                        "model": r["model"],
                        "n_quarters": r["n_quarters"],
                        "mean_return": fmt(r["mean_return"], 4),
                        "win_rate": fmt(r["win_rate"], 3),
                        "mean_sharpe": fmt(r["mean_sharpe"], 3),
                    }
                    for r in blk["ranking"]
                ],
            }
            for blk in regime_exist
        ],
        "switcher": {
            "quarters": switch_rows,
            "summary": switch_full,
            "oracle_lookhead": oracle_full,
        },
        "quarter_returns": rec.to_dict(orient="records"),
    }
    path = OUT_DIR / "regime_model_search.json"
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2))
    print("WINNERS")
    for blk in regime_factor:
        top = blk["ranking"][:3]
        print(blk["regime"], "->", [(t["model"], round(t["mean_return"], 3)) for t in top])
    print("EXISTING 2024+")
    for blk in regime_exist:
        print(blk["regime"], "->", blk["winner"], blk["ranking"][0])
    print("SWITCH", switch_full, "ORACLE", oracle_full)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
