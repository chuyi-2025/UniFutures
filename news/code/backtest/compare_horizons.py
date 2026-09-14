#!/usr/bin/env python3
"""Compare fine-tuned single-model results across prediction horizons."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS = Path("/home/workspace/lab/UniFutures/news/data/results")
MODELS = ("finance_zh", "modernbert", "finbert2")
HORIZONS = (1, 7, 14, 30)


def load_ok(name: str) -> pd.DataFrame | None:
    path = RESULTS / name / "summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    return df


def metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    win = int((df["return"] > 0).sum())
    return {
        "n_sym": n,
        "profit_ratio": win / n if n else 0.0,
        "avg_ret": float(df["return"].mean()) if n else 0.0,
        "med_ret": float(df["return"].median()) if n else 0.0,
        "avg_wr": float(df["win_rate"].mean()) if n else 0.0,
        "avg_sharpe": float(df["sharpe"].mean()) if n else 0.0,
        "med_sharpe": float(df["sharpe"].median()) if n else 0.0,
        "avg_mdd": float(df["max_dd"].mean()) if n else 0.0,
    }


def main() -> None:
    rows = []
    for m in MODELS:
        for h in HORIZONS:
            name = f"sentiment_ft_{m}" if h == 1 else f"sentiment_ft_{m}_h{h}"
            df = load_ok(name)
            if df is None or df.empty:
                continue
            met = metrics(df)
            rows.append({"model": m, "horizon": h, "exp": name, **met})

    tab = pd.DataFrame(rows)
    out_csv = RESULTS / "horizon_comparison.csv"
    tab.to_csv(out_csv, index=False)

    lines = [
        "# 微调单模型：预测 horizon 对比（1 / 7 / 14 / 30 交易日）",
        "",
        "## 设定",
        "",
        "| 项目 | 配置 |",
        "|------|------|",
        "| 训练窗 | 2024-01-01 ~ 2025-06-30 |",
        "| 回测窗 | 2025-07-01 ~ 2026-12-31 |",
        "| 标签 | 研报日后连续 H 个交易日主连累计 log_ret；中性带 `0.001*sqrt(H)` |",
        "| 仓位 | 同日多数表决 → +1/0/-1；回测从入场日起持有 H 个交易日 |",
        "| 模型 | finance_zh / modernbert / finbert2 各自独立微调 |",
        "",
        "## 汇总表",
        "",
        "| 模型 | Horizon | 盈利品种比 | 平均收益 | 中位收益 | 平均胜率 | 平均Sharpe | 中位Sharpe | 平均MDD |",
        "|------|---------|------------|----------|----------|----------|------------|------------|---------|",
    ]
    for _, r in tab.sort_values(["model", "horizon"]).iterrows():
        lines.append(
            f"| {r['model']} | {int(r['horizon'])}d | {r['profit_ratio']*100:.1f}% | "
            f"{r['avg_ret']*100:.2f}% | {r['med_ret']*100:.2f}% | {r['avg_wr']*100:.1f}% | "
            f"{r['avg_sharpe']:.2f} | {r['med_sharpe']:.2f} | {r['avg_mdd']*100:.2f}% |"
        )
    lines.append("")

    # per-model best horizon by avg_ret
    lines += ["## 按模型：相对次日(1d)的变化", ""]
    for m in MODELS:
        sub = tab[tab["model"] == m].sort_values("horizon")
        if sub.empty or 1 not in set(sub["horizon"]):
            continue
        base = sub[sub["horizon"] == 1].iloc[0]
        lines.append(f"### {m}")
        lines.append("")
        lines.append(
            f"次日基线：盈利比 {base['profit_ratio']*100:.1f}% / "
            f"均收益 {base['avg_ret']*100:.2f}% / 中位Sharpe {base['med_sharpe']:.2f}"
        )
        lines.append("")
        lines.append("| Horizon | Δ盈利比(pp) | Δ均收益(pp) | Δ中位Sharpe |")
        lines.append("|---------|-------------|-------------|-------------|")
        for _, r in sub.iterrows():
            if int(r["horizon"]) == 1:
                continue
            lines.append(
                f"| {int(r['horizon'])}d | "
                f"{(r['profit_ratio']-base['profit_ratio'])*100:+.1f} | "
                f"{(r['avg_ret']-base['avg_ret'])*100:+.2f} | "
                f"{r['med_sharpe']-base['med_sharpe']:+.2f} |"
            )
        lines.append("")

    readme = RESULTS / "horizon_comparison_README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {out_csv}")
    print(f"[done] {readme}")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
