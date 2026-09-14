#!/usr/bin/env python3
"""Aggregate and visualize PPO backtest results across symbols."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import BACKTEST_END, BACKTEST_START, RESULTS_DIR, list_symbols

PPO_RESULTS = RESULTS_DIR / "ppo"
ANALYSIS_DIR = PPO_RESULTS / "analysis"
INIT_CAP = 1_000_000.0


def load_summaries() -> pd.DataFrame:
    rows: list[dict] = []
    for sym in list_symbols():
        path = PPO_RESULTS / sym.lower() / "summary.csv"
        if path.exists():
            row = pd.read_csv(path).iloc[0].to_dict()
            row["symbol"] = sym
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("final_capital", "return", "sharpe", "max_dd", "days"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("return", ascending=False).reset_index(drop=True)


def write_report(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary_all.csv", index=False)

    n = len(df)
    pos = int((df["return"] > 0).sum())
    lines = [
        "# PPO 全品种回测分析",
        "",
        f"区间：{BACKTEST_START.date()} ~ {BACKTEST_END.date()}  |  初始资金：100 万/品种",
        "",
        "## 汇总",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 品种数 | {n} |",
        f"| 盈利品种 | {pos} ({pos/n*100:.1f}%) |" if n else "| 盈利品种 | - |",
        f"| 平均收益率 | {df['return'].mean()*100:.2f}% |" if n else "",
        f"| 中位收益率 | {df['return'].median()*100:.2f}% |" if n else "",
        f"| 平均 Sharpe | {df['sharpe'].mean():.2f} |" if n else "",
        f"| 平均最大回撤 | {df['max_dd'].mean()*100:.2f}% |" if n else "",
        "",
        "## 排名（按收益率）",
        "",
        "| 排名 | 品种 | 期末(万) | 收益率 | Sharpe | 最大回撤 | 天数 |",
        "|------|------|---------|--------|--------|----------|------|",
    ]
    for i, row in df.iterrows():
        lines.append(
            f"| {i+1} | {row['symbol']} | {row['final_capital']/1e4:.1f} | "
            f"{row['return']*100:.2f}% | {row['sharpe']:.2f} | {row['max_dd']*100:.2f}% | {int(row['days'])} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return

    rets = df["return"].to_numpy() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(rets, bins=30, color="#1f77b4", edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Return (%)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("PPO return distribution")
    axes[0].grid(True, alpha=0.3)

    top = df.head(15)
    bottom = df.tail(15).iloc[::-1]
    show = pd.concat([top, bottom]).drop_duplicates("symbol")
    colors = ["#2ca02c" if r >= 0 else "#d62728" for r in show["return"]]
    axes[1].barh(show["symbol"], show["return"] * 100, color=colors, alpha=0.85)
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Return (%)")
    axes[1].set_title("Top / bottom symbols")
    axes[1].grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(df["max_dd"] * 100, df["return"] * 100, alpha=0.7, s=40)
    for _, row in df.iterrows():
        if abs(row["return"]) > df["return"].quantile(0.9) or row["return"] < df["return"].quantile(0.1):
            ax.annotate(row["symbol"], (row["max_dd"] * 100, row["return"] * 100), fontsize=8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Max drawdown (%)")
    ax.set_ylabel("Return (%)")
    ax.set_title("Return vs max drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "return_vs_dd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args()

    df = load_summaries()
    if df.empty:
        raise SystemExit(f"no results under {PPO_RESULTS}")

    write_report(df, args.out_dir)
    plot(df, args.out_dir)
    n = len(df)
    pos = int((df["return"] > 0).sum())
    print(f"analyzed {n} symbols: {pos} positive, mean return {df['return'].mean()*100:.2f}%")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
