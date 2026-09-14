#!/usr/bin/env python3
"""Write Kronos-style README for a sentiment backtest result directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def write_readme(
    out_dir: Path,
    title: str,
    setup_rows: list[tuple[str, str]],
    summary_csv: Path | None = None,
) -> Path:
    path = summary_csv or (out_dir / "summary.csv")
    df = pd.read_csv(path)
    ok = df[df.get("status", "ok") == "ok"].copy() if "status" in df.columns else df.copy()
    fail = df[df["status"] == "fail"] if "status" in df.columns else pd.DataFrame()

    lines: list[str] = [f"# {title}", "", "## 实验设置", "", "| 项目 | 配置 |", "|------|------|"]
    for k, v in setup_rows:
        lines.append(f"| {k} | {v} |")
    lines += ["", "## 字段说明（`summary.csv`）", ""]
    lines += [
        "| 字段 | 含义 |",
        "|------|------|",
        "| `symbol` | 品种代码 |",
        "| `return` | 总收益率 |",
        "| `win_rate` | 有仓位日胜率（strategy_ret>0） |",
        "| `sharpe` | 日频策略收益年化 Sharpe（252） |",
        "| `max_dd` | 最大回撤 |",
        "| `trade_days` | 有仓位交易日数 |",
        "",
    ]

    if ok.empty:
        lines += ["## 汇总统计", "", "无成功回测品种。", ""]
    else:
        n = len(ok)
        win_sym = int((ok["return"] > 0).sum())
        lose_sym = int((ok["return"] < 0).sum())
        flat_sym = n - win_sym - lose_sym
        lines += [
            "## 汇总统计（成功品种）",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 成功品种 | {n} |",
            f"| 盈利品种 | {win_sym} / {n}（{win_sym/n*100:.1f}%） |",
            f"| 亏损品种 | {lose_sym} / {n}（{lose_sym/n*100:.1f}%） |",
            f"| 持平品种 | {flat_sym} |",
            f"| 平均收益率 | {ok['return'].mean()*100:.2f}% |",
            f"| 中位收益率 | {ok['return'].median()*100:.2f}% |",
            f"| 平均胜率 | {ok['win_rate'].mean()*100:.1f}% |",
            f"| 中位胜率 | {ok['win_rate'].median()*100:.1f}% |",
            f"| 平均 Sharpe | {ok['sharpe'].mean():.2f} |",
            f"| 中位 Sharpe | {ok['sharpe'].median():.2f} |",
            f"| 平均最大回撤 | {ok['max_dd'].mean()*100:.2f}% |",
            "",
        ]
        best = ok.sort_values("return", ascending=False).head(3)
        worst = ok.sort_values("return", ascending=True).head(3)
        lines.append(
            "表现最好："
            + "、".join(f"`{r.symbol}`（{r['return']*100:.2f}%）" for _, r in best.iterrows())
            + "。"
        )
        lines.append("")
        lines.append(
            "表现最差："
            + "、".join(f"`{r.symbol}`（{r['return']*100:.2f}%）" for _, r in worst.iterrows())
            + "。"
        )
        lines.append("")

        ranked = ok.sort_values("return", ascending=False).reset_index(drop=True)
        lines += [
            "## 排名（按收益率）",
            "",
            "| 排名 | 品种 | 期末(万) | 收益率 | 胜率 | Sharpe | 最大回撤 | 交易日 |",
            "|------|------|---------|--------|------|--------|----------|--------|",
        ]
        for i, r in ranked.iterrows():
            lines.append(
                f"| {i+1} | {r['symbol']} | {r['final_capital']/1e4:.1f} | "
                f"{r['return']*100:.2f}% | {r['win_rate']*100:.1f}% | {r['sharpe']:.2f} | "
                f"{r['max_dd']*100:.2f}% | {int(r['trade_days'])} |"
            )
        lines.append("")

    if not fail.empty:
        lines += ["## 失败品种", "", "| 品种 | 原因 |", "|------|------|"]
        for _, r in fail.iterrows():
            lines.append(f"| {r['symbol']} | {r.get('error', '')} |")
        lines.append("")

    readme = out_dir / "README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[readme] {readme}")
    return readme


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--title", type=str, required=True)
    ap.add_argument("--setup", type=str, nargs="*", default=[], help="key=value pairs")
    args = ap.parse_args()
    setup = []
    for item in args.setup:
        if "=" in item:
            k, v = item.split("=", 1)
            setup.append((k, v))
    write_readme(args.out_dir, args.title, setup)


if __name__ == "__main__":
    main()
