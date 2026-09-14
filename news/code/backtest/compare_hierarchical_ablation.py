#!/usr/bin/env python3
"""Rank hierarchical ablations vs concat/wavg/ema baselines (hold=1, 0/2/5bp)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "train"
HIER = TRAIN / "hierarchical"
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(HIER))
sys.path.insert(0, str(HERE))

from backtest_sentiment import aggregate_daily_signals, run_backtest  # noqa: E402
from common import INIT_CAP, load_symbol_ohlc  # noqa: E402
from config import DATA_ROOT, OOS_END, OOS_START, RESULT_ROOT, SCHEMES  # noqa: E402


COST_BPS = (0, 2, 5)


def apply_cost(daily: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Subtract turnover * cost from strategy_ret and rebuild capital."""
    out = daily.sort_values(["symbol", "date"]).copy()
    parts = []
    for sym, g in out.groupby("symbol", sort=False):
        g = g.copy()
        pos = g["position"].to_numpy()
        prev = np.concatenate([[0], pos[:-1]])
        turnover = np.abs(pos - prev)
        cost = turnover * (cost_bps / 10_000.0)
        g["strategy_ret"] = g["strategy_ret"].to_numpy() - cost
        cap = INIT_CAP
        caps = []
        for r, p in zip(g["strategy_ret"].to_numpy(), pos):
            if p != 0:
                cap *= float(np.exp(r))
            caps.append(cap)
        g["capital"] = caps
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else out


def summarize_frame(part: pd.DataFrame) -> dict:
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
        "final_capital": final,
        "return": final / INIT_CAP - 1.0,
        "trade_days": int(len(rets)),
        "win_rate": float((rets > 0).mean()) if len(rets) else 0.0,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def ew_portfolio_sharpe(daily: pd.DataFrame) -> float:
    if daily.empty:
        return 0.0
    pivot = daily.pivot_table(
        index="date", columns="symbol", values="strategy_ret", aggfunc="mean"
    )
    port = pivot.mean(axis=1)
    if len(port) < 2 or port.std() < 1e-12:
        return 0.0
    return float(port.mean() / port.std() * np.sqrt(252))


def metrics_from_daily(daily: pd.DataFrame, cost_bps: float) -> dict:
    d = apply_cost(daily, cost_bps) if cost_bps else daily
    rows = []
    for sym, g in d.groupby("symbol"):
        sm = summarize_frame(g)
        sm["symbol"] = sym
        rows.append(sm)
    sdf = pd.DataFrame(rows)
    return {
        "cost_bps": cost_bps,
        "median_sharpe": float(sdf["sharpe"].median()) if len(sdf) else 0.0,
        "mean_return": float(sdf["return"].mean()) if len(sdf) else 0.0,
        "median_max_dd": float(sdf["max_dd"].median()) if len(sdf) else 0.0,
        "mean_win_rate": float(sdf["win_rate"].mean()) if len(sdf) else 0.0,
        "n_symbols": int(len(sdf)),
        "ew_sharpe": ew_portfolio_sharpe(d),
    }


def load_baseline_signals(name: str) -> pd.DataFrame | None:
    sent = DATA_ROOT.parent  # news/data/sentiment
    candidates = {
        "concat_L60": sent / "lookback" / "signals_lb_finance_zh_concat_L60.parquet",
        "wavg_exp_L60": sent / "lookback" / "signals_lb_finance_zh_wavg_exp_L60.parquet",
        "ema_sent_10": sent / "innovation" / "ema_sent_10.csv",
    }
    path = candidates.get(name)
    if path is None or not path.exists():
        # try glob
        if name == "wavg_exp_L60":
            hits = list((sent / "lookback").glob("*wavg_exp*L60*"))
            path = hits[0] if hits else None
        elif name == "ema_sent_10":
            hits = list((sent / "innovation").glob("*ema_sent_10*"))
            path = hits[0] if hits else None
        elif name == "concat_L60":
            hits = list((sent / "lookback").glob("*concat*L60*"))
            path = hits[0] if hits else None
    if path is None or not Path(path).exists():
        return None
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def collect_scheme_rows() -> list[dict]:
    rows = []
    for scheme in SCHEMES:
        scheme_dir = DATA_ROOT / "runs" / scheme
        if not scheme_dir.is_dir():
            continue
        for seed_dir in sorted(scheme_dir.glob("seed_*")):
            meta_path = seed_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            daily_path = RESULT_ROOT / scheme / seed_dir.name / "daily_signals.csv"
            # rebuild from per-symbol dailies if needed
            base = {
                "family": "hierarchical",
                "scheme": scheme,
                "seed": seed_dir.name,
                "status": meta.get("status"),
                "streams": "|".join(meta.get("streams") or []),
                "best_val_loss": meta.get("best_val_loss"),
                "n_train": meta.get("n_train"),
                "n_val": meta.get("n_val"),
                "reason": meta.get("reason"),
            }
            sym_dir = RESULT_ROOT / scheme / seed_dir.name
            dailies = list(sym_dir.glob("*/daily.csv")) if sym_dir.exists() else []
            if not dailies:
                rows.append(base)
                continue
            daily = pd.concat([pd.read_csv(p, parse_dates=["date"]) for p in dailies])
            daily["symbol"] = daily["symbol"].astype(str)
            for bps in COST_BPS:
                m = metrics_from_daily(daily, bps)
                rows.append({**base, **m})
    return rows


def collect_baseline_rows(start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    rows = []
    for name in ["concat_L60", "wavg_exp_L60", "ema_sent_10"]:
        sig = load_baseline_signals(name)
        if sig is None or "position" not in sig.columns:
            rows.append(
                {
                    "family": "baseline",
                    "scheme": name,
                    "seed": "-",
                    "status": "missing",
                }
            )
            continue
        sig = sig[(sig["trade_date"] >= start) & (sig["trade_date"] <= end)].copy()
        out = RESULT_ROOT / "baselines" / name
        summary = run_backtest(sig, out, start=start, end=end, hold_days=1)
        dailies = list(out.glob("*/daily.csv"))
        if not dailies:
            rows.append(
                {
                    "family": "baseline",
                    "scheme": name,
                    "seed": "-",
                    "status": "ok",
                    "n_symbols": int(len(summary)),
                }
            )
            continue
        daily = pd.concat([pd.read_csv(p, parse_dates=["date"]) for p in dailies])
        for bps in COST_BPS:
            m = metrics_from_daily(daily, bps)
            rows.append(
                {
                    "family": "baseline",
                    "scheme": name,
                    "seed": "-",
                    "status": "ok",
                    **m,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULT_ROOT / "comparison.csv")
    ap.add_argument("--start", type=str, default=OOS_START)
    ap.add_argument("--end", type=str, default=OOS_END)
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    rows = collect_scheme_rows()
    if not args.skip_baselines:
        rows.extend(collect_baseline_rows(start, end))

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # best single-model stream note
    singles = df[
        (df["family"] == "hierarchical")
        & (df["scheme"].isin([f"H{i:02d}" for i in range(1, 13)]))
        & (df["status"] == "ok")
        & (df.get("cost_bps", 0) == 0)
        & df["best_val_loss"].notna()
    ] if "cost_bps" in df.columns else pd.DataFrame()

    if len(singles):
        g = singles.groupby("scheme", as_index=False)["best_val_loss"].mean()
        best_scheme = g.sort_values("best_val_loss").iloc[0]["scheme"]
        print(f"best single scheme by val loss: {best_scheme}")

    # Val-window ranking helper: median seed portfolio sharpe at 0bp
    if "ew_sharpe" in df.columns and "cost_bps" in df.columns:
        focus = df[(df["family"] == "hierarchical") & (df["cost_bps"] == 0) & (df["status"] == "ok")]
        if len(focus):
            rank = (
                focus.groupby("scheme", as_index=False)["ew_sharpe"]
                .median()
                .sort_values("ew_sharpe", ascending=False)
            )
            rank.to_csv(RESULT_ROOT / "rank_ew_sharpe_0bp.csv", index=False)
            print(rank.head(12).to_string(index=False))

    print(f"[done] {args.out} rows={len(df)}")
    print("NOTE: 2025-07+ OOS is retrospective / not a pristine holdout.")


if __name__ == "__main__":
    main()
