#!/usr/bin/env python3
"""Cross-sectional top/bottom rank portfolio backtest + IC metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))
sys.path.insert(0, str(HERE.parent / "train" / "rank_lora"))

from common import INIT_CAP, load_symbol_ohlc  # noqa: E402
from config import COST_BPS, HORIZONS  # noqa: E402
from infer_rank_signals import scores_to_positions  # noqa: E402


def apply_cost(daily: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    out = daily.sort_values(["symbol", "date"]).copy()
    parts = []
    for _, g in out.groupby("symbol", sort=False):
        g = g.copy()
        pos = g["position"].to_numpy()
        prev = np.concatenate([[0], pos[:-1]])
        turnover = np.abs(pos - prev)
        g["strategy_ret"] = g["strategy_ret"].to_numpy() - turnover * (cost_bps / 1e4)
        cap = INIT_CAP
        caps = []
        for r in g["strategy_ret"].to_numpy():
            cap *= float(np.exp(r))
            caps.append(cap)
        g["capital"] = caps
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else out


def summarize_symbol(part: pd.DataFrame) -> dict:
    if part.empty:
        return {}
    active = part[part["position"] != 0]
    rets = active["strategy_ret"].to_numpy() if len(active) else np.array([])
    all_r = part["strategy_ret"].to_numpy()
    sharpe = (
        float(all_r.mean() / all_r.std() * np.sqrt(252))
        if len(all_r) > 1 and all_r.std() > 1e-12
        else 0.0
    )
    peak = part["capital"].cummax().to_numpy()
    max_dd = float((part["capital"] / peak - 1.0).min()) if len(part) else 0.0
    final = float(part["capital"].iloc[-1])
    return {
        "symbol": part["symbol"].iloc[0],
        "final_capital": final,
        "return": final / INIT_CAP - 1.0,
        "trade_days": int(len(rets)),
        "win_rate": float((rets > 0).mean()) if len(rets) else 0.0,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "turnover": float(np.abs(np.diff(part["position"].to_numpy(), prepend=0)).sum()),
    }


def ew_sharpe(daily: pd.DataFrame) -> float:
    pivot = daily.pivot_table(index="date", columns="symbol", values="strategy_ret", aggfunc="mean")
    port = pivot.mean(axis=1)
    if len(port) < 2 or port.std() < 1e-12:
        return 0.0
    return float(port.mean() / port.std() * np.sqrt(252))


def rank_ic_series(signals: pd.DataFrame, score_col: str, ret_col: str) -> pd.Series:
    ics = []
    dates = []
    for dt, g in signals.groupby("trade_date"):
        if len(g) < 3:
            continue
        if g[score_col].std() < 1e-12 or g[ret_col].std() < 1e-12:
            continue
        ic = float(g[score_col].corr(g[ret_col], method="spearman"))
        if np.isfinite(ic):
            ics.append(ic)
            dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="rank_ic")


def build_daily_from_positions(pos_df: pd.DataFrame, start, end) -> pd.DataFrame:
    rows = []
    for sym, g in pos_df.groupby("symbol"):
        px = load_symbol_ohlc(sym)
        if px.empty:
            continue
        px = px.copy()
        px["date"] = pd.to_datetime(px["date"]).dt.normalize()
        px = px[(px["date"] >= start) & (px["date"] <= end)].reset_index(drop=True)
        sig = g.set_index("trade_date")["position"].to_dict()
        cap = INIT_CAP
        prev = 0
        for _, row in px.iterrows():
            dt = pd.Timestamp(row["date"]).normalize()
            pos = int(sig.get(dt, 0))
            ret = float(row["log_ret_1d"]) if pd.notna(row["log_ret_1d"]) else 0.0
            if not np.isfinite(ret):
                ret = 0.0
            day_pnl = pos * ret
            if pos != 0:
                cap *= float(np.exp(day_pnl))
            rows.append(
                {
                    "date": dt,
                    "symbol": sym,
                    "code": row.get("code", ""),
                    "position": pos,
                    "log_ret_1d": ret,
                    "strategy_ret": day_pnl,
                    "capital": cap,
                    "turnover_step": abs(pos - prev),
                }
            )
            prev = pos
    return pd.DataFrame(rows)


def run_rank_backtest(
    signals: pd.DataFrame,
    out_dir: Path,
    horizon: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    score_col = f"score_h{horizon}"
    ret_col = f"ret_h{horizon}"
    work = signals.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work = work[(work["trade_date"] >= start) & (work["trade_date"] <= end)]
    if score_col not in work.columns:
        raise SystemExit(f"missing {score_col}")

    ic = rank_ic_series(work, score_col, ret_col) if ret_col in work.columns else pd.Series(dtype=float)
    icir = float(ic.mean() / ic.std() * np.sqrt(252)) if len(ic) > 1 and ic.std() > 1e-12 else 0.0

    pos_df = scores_to_positions(work, score_col, top_pct, bottom_pct)
    # Collapse checks
    collapse = {
        "score_std": float(pos_df[score_col].std()) if score_col in pos_df else float(pos_df["score"].std()) if "score" in pos_df else 0.0,
        "long_rate": float((pos_df["position"] == 1).mean()),
        "short_rate": float((pos_df["position"] == -1).mean()),
        "flat_rate": float((pos_df["position"] == 0).mean()),
        "all_long": bool((pos_df["position"] == 1).all()) if len(pos_df) else False,
        "all_short": bool((pos_df["position"] == -1).all()) if len(pos_df) else False,
    }
    # Prefer renamed score after scores_to_positions keeps original col
    if score_col in pos_df.columns:
        collapse["score_std"] = float(pos_df[score_col].std())

    daily0 = build_daily_from_positions(pos_df, start, end)
    metrics_by_cost = {}
    for bps in COST_BPS:
        daily = apply_cost(daily0, bps) if bps else daily0
        sym_rows = [summarize_symbol(g) for _, g in daily.groupby("symbol")]
        sdf = pd.DataFrame([r for r in sym_rows if r])
        metrics_by_cost[bps] = {
            "cost_bps": bps,
            "median_sharpe": float(sdf["sharpe"].median()) if len(sdf) else 0.0,
            "mean_return": float(sdf["return"].mean()) if len(sdf) else 0.0,
            "median_max_dd": float(sdf["max_dd"].median()) if len(sdf) else 0.0,
            "mean_win_rate": float(sdf["win_rate"].mean()) if len(sdf) else 0.0,
            "ew_sharpe": ew_sharpe(daily),
            "mean_turnover": float(sdf["turnover"].mean()) if len(sdf) else 0.0,
            "n_symbols": int(len(sdf)),
        }
        if bps == 0:
            sdf.to_csv(out_dir / "summary.csv", index=False)
            daily.to_csv(out_dir / "daily.csv", index=False)

    result = {
        "horizon": horizon,
        "rank_ic_mean": float(ic.mean()) if len(ic) else 0.0,
        "rank_icir": icir,
        "n_ic_days": int(len(ic)),
        "collapse": collapse,
        "costs": metrics_by_cost,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if len(ic):
        ic.to_csv(out_dir / "rank_ic_daily.csv", header=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=1, choices=list(HORIZONS))
    ap.add_argument("--start", type=str, default="2025-07-01")
    ap.add_argument("--end", type=str, default="2026-12-31")
    args = ap.parse_args()

    sig = pd.read_parquet(args.signals) if args.signals.suffix == ".parquet" else pd.read_csv(args.signals)
    result = run_rank_backtest(
        sig,
        args.out,
        horizon=args.horizon,
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
