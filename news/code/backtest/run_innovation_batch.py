#!/usr/bin/env python3
"""Backtest and rank the frozen sentiment innovation matrix."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from common import INIT_CAP, RESULTS_ROOT, load_symbol_ohlc  # noqa: E402


DEFAULT_SIGNALS = Path("/home/workspace/lab/UniFutures/news/data/sentiment/innovation")
DEFAULT_OUT = RESULTS_ROOT / "innovation_batch"
COST_BPS = (0, 1, 2, 5)


def sharpe(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or arr.std() <= 1e-12:
        return 0.0
    return float(arr.mean() / arr.std() * math.sqrt(252.0))


def max_drawdown(log_returns: pd.Series | np.ndarray) -> float:
    capital = np.exp(np.cumsum(np.asarray(log_returns, dtype=float)))
    if not len(capital):
        return 0.0
    peak = np.maximum.accumulate(capital)
    return float(np.min(capital / peak - 1.0))


def load_price_cache(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        px = load_symbol_ohlc(symbol)
        if px.empty:
            continue
        px = px.rename(columns={"date": "trade_date"})
        px["trade_date"] = pd.to_datetime(px["trade_date"]).dt.normalize()
        px = px[px["trade_date"].between(start, end)][
            ["trade_date", "code", "log_ret_1d"]
        ].copy()
        cache[symbol] = px
    return cache


def backtest_scheme(
    signals: pd.DataFrame,
    price_cache: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_parts: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for symbol, sig in signals.groupby("symbol", sort=True):
        px = price_cache.get(symbol)
        if px is None or px.empty:
            continue
        part = sig[["trade_date", "position"]].merge(px, on="trade_date", how="inner")
        if part.empty:
            continue
        part = part.sort_values("trade_date").reset_index(drop=True)
        part["symbol"] = symbol
        part["position"] = part["position"].fillna(0.0).clip(-1.0, 1.0)
        part["log_ret_1d"] = part["log_ret_1d"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        part["turnover"] = part["position"].diff().fillna(part["position"]).abs()
        part["gross_ret"] = part["position"] * part["log_ret_1d"]
        summary = {
            "symbol": symbol,
            "days": len(part),
            "active_days": int((part["position"] != 0).sum()),
            "active_ratio": float((part["position"] != 0).mean()),
            "turnover": float(part["turnover"].sum()),
            "avg_abs_position": float(part["position"].abs().mean()),
        }
        for bps in COST_BPS:
            col = f"net_ret_{bps}bp"
            part[col] = part["gross_ret"] - part["turnover"] * bps / 10_000.0
            summary[f"return_{bps}bp"] = float(np.exp(part[col].sum()) - 1.0)
            summary[f"sharpe_{bps}bp"] = sharpe(part[col])
            summary[f"max_dd_{bps}bp"] = max_drawdown(part[col])
        daily_parts.append(part)
        summaries.append(summary)
    return pd.concat(daily_parts, ignore_index=True), pd.DataFrame(summaries)


def portfolio_metrics(daily: pd.DataFrame, bps: int) -> dict:
    col = f"net_ret_{bps}bp"
    portfolio = daily.groupby("trade_date", sort=True)[col].mean()
    return {
        f"ew_portfolio_return_{bps}bp": float(np.exp(portfolio.sum()) - 1.0),
        f"ew_portfolio_sharpe_{bps}bp": sharpe(portfolio),
        f"ew_portfolio_max_dd_{bps}bp": max_drawdown(portfolio),
        f"portfolio_days_{bps}bp": len(portfolio),
    }


def aggregate_scheme(name: str, summary: pd.DataFrame, daily: pd.DataFrame, baseline: bool) -> dict:
    row = {
        "scheme": name,
        "is_baseline": baseline,
        "n_symbols": len(summary),
        "profit_ratio": float((summary["return_0bp"] > 0).mean()),
        "avg_return": float(summary["return_0bp"].mean()),
        "median_return": float(summary["return_0bp"].median()),
        "med_symbol_sharpe": float(summary["sharpe_0bp"].median()),
        "avg_symbol_sharpe": float(summary["sharpe_0bp"].mean()),
        "avg_max_dd": float(summary["max_dd_0bp"].mean()),
        "avg_turnover": float(summary["turnover"].mean()),
        "active_ratio": float(summary["active_days"].sum() / summary["days"].sum()),
    }
    for bps in COST_BPS:
        row[f"avg_return_{bps}bp"] = float(summary[f"return_{bps}bp"].mean())
        row[f"med_symbol_sharpe_{bps}bp"] = float(summary[f"sharpe_{bps}bp"].median())
        row.update(portfolio_metrics(daily, bps))
    return row


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_readme(comparison: pd.DataFrame, out_dir: Path) -> None:
    ranked = comparison.sort_values(
        ["med_symbol_sharpe", "ew_portfolio_sharpe_0bp", "avg_return"],
        ascending=False,
    ).reset_index(drop=True)
    innovations = ranked[~ranked["is_baseline"]].copy()
    baseline = comparison.set_index("scheme").loc["baseline_concat_L60"]
    top = innovations.head(10)

    lines = [
        "# 32 个情绪创新方案批量回测",
        "",
        "> **Retrospective exploratory only.** 2025-07 起的区间已用于 lookback 方案筛选，",
        "> 不是未触碰的最终测试集；以下排名不能当作 pristine OOS 发现。",
        "",
        "## 固定协议",
        "",
        "- 冻结 `finance_zh` lookback 概率；交易日 T 不使用 T 当天研报。",
        "- 所有方案使用相同的 `(symbol, trade_date)` 信号面板，缺失辅助信号按 flat 处理。",
        "- 主排序：品种中位 Sharpe → 等权组合 Sharpe → 品种平均收益。",
        "- 成本：按 `abs(position_t-position_{t-1})` 计单边换手，报告 0/1/2/5bp。",
        "- PPO 历史结果不参与排名：其训练窗、seed 和特征生成口径不可比。",
        "",
        "## Top 10（零成本主排序）",
        "",
        "| 排名 | 方案 | 中位Sharpe | 组合Sharpe | 平均收益 | 盈利品种比 | 平均MDD | 活跃率 | 2bp组合Sharpe |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        lines.append(
            f"| {rank} | `{row['scheme']}` | {row['med_symbol_sharpe']:.2f} | "
            f"{row['ew_portfolio_sharpe_0bp']:.2f} | {fmt_pct(row['avg_return'])} | "
            f"{fmt_pct(row['profit_ratio'])} | {fmt_pct(row['avg_max_dd'])} | "
            f"{fmt_pct(row['active_ratio'])} | {row['ew_portfolio_sharpe_2bp']:.2f} |"
        )

    best = innovations.iloc[0]
    beat_med = best["med_symbol_sharpe"] > baseline["med_symbol_sharpe"]
    beat_ret = innovations.loc[innovations["avg_return"].idxmax(), "avg_return"] > baseline["avg_return"]
    best_ret = innovations.loc[innovations["avg_return"].idxmax()]
    lines += [
        "",
        "## 相对 concat L60 基线",
        "",
        f"- 基线：平均收益 {fmt_pct(baseline['avg_return'])}，"
        f"品种中位 Sharpe {baseline['med_symbol_sharpe']:.2f}，"
        f"等权组合 Sharpe {baseline['ew_portfolio_sharpe_0bp']:.2f}。",
        f"- Sharpe 排名第一：`{best['scheme']}`，中位 Sharpe "
        f"{best['med_symbol_sharpe']:.2f}（{'超过' if beat_med else '未超过'}基线）。",
        f"- 平均收益最高：`{best_ret['scheme']}`，{fmt_pct(best_ret['avg_return'])}"
        f"（{'超过' if beat_ret else '未超过'}基线）。",
        "- 任何胜出仅代表该已反复查看区间内的回顾性结果；确认性结论需要未来新数据。",
        "",
        "完整结果见 `comparison.csv`；逐品种结果和日收益位于各方案子目录。",
    ]
    (out_dir / "innovation_comparison_README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-dir", type=Path, default=DEFAULT_SIGNALS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--start", default="2025-07-01")
    ap.add_argument("--end", default="2026-12-31")
    args = ap.parse_args()

    manifest = pd.read_csv(args.signals_dir / "manifest.csv")
    if int((~manifest["is_baseline"]).sum()) < 32:
        raise SystemExit("manifest contains fewer than 32 innovation schemes")
    first = pd.read_csv(Path(manifest.iloc[0]["path"]))
    symbols = sorted(first["symbol"].astype(str).str.upper().unique())
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    price_cache = load_price_cache(symbols, start, end)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for entry in manifest.itertuples(index=False):
        name = str(entry.scheme)
        signals = pd.read_csv(Path(entry.path))
        signals["symbol"] = signals["symbol"].astype(str).str.upper()
        signals["trade_date"] = pd.to_datetime(signals["trade_date"]).dt.normalize()
        signals = signals[signals["trade_date"].between(start, end)]
        daily, summary = backtest_scheme(signals, price_cache)
        scheme_dir = args.out_dir / name
        scheme_dir.mkdir(parents=True, exist_ok=True)
        daily.to_csv(scheme_dir / "daily.csv", index=False)
        summary.to_csv(scheme_dir / "summary.csv", index=False)
        row = aggregate_scheme(name, summary, daily, bool(entry.is_baseline))
        rows.append(row)
        print(
            f"[backtest] {name}: medS={row['med_symbol_sharpe']:.2f} "
            f"portS={row['ew_portfolio_sharpe_0bp']:.2f} avgR={row['avg_return']:.2%}"
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["med_symbol_sharpe", "ew_portfolio_sharpe_0bp", "avg_return"],
        ascending=False,
    )
    comparison.to_csv(args.out_dir / "comparison.csv", index=False)
    write_readme(comparison, args.out_dir)
    print(f"[done] {args.out_dir / 'comparison.csv'}")
    print(f"[done] {args.out_dir / 'innovation_comparison_README.md'}")


if __name__ == "__main__":
    main()
