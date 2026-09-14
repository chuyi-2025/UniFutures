#!/usr/bin/env python3
"""Compare lookback sentiment schemes (ft finance_zh only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS = Path("/home/workspace/lab/UniFutures/news/data/results")
LOOKBACKS = (3, 7, 14, 30, 60, 180)
PRED_HS = (1, 7)
SCHEMES = (
    "wavg_uniform",
    "wavg_linear",
    "wavg_exp",
    "daily_wavg_linear",
    "concat",
)


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
    # same-day baseline (current best)
    base = load_ok("sentiment_ft_finance_zh")
    if base is not None and not base.empty:
        rows.append({"scheme": "same_day_baseline", "lookback": 0, "pred_h": 1, "exp": "sentiment_ft_finance_zh", **metrics(base)})
    base7 = load_ok("sentiment_ft_finance_zh_h7")
    if base7 is not None and not base7.empty:
        rows.append({"scheme": "same_day_baseline", "lookback": 0, "pred_h": 7, "exp": "sentiment_ft_finance_zh_h7", **metrics(base7)})

    for scheme in SCHEMES:
        for L in LOOKBACKS:
            for h in PRED_HS:
                name = f"sentiment_lb_finance_zh_{scheme}_L{L}_h{h}"
                df = load_ok(name)
                if df is None or df.empty:
                    continue
                rows.append({"scheme": scheme, "lookback": L, "pred_h": h, "exp": name, **metrics(df)})

    tab = pd.DataFrame(rows)
    out_csv = RESULTS / "lookback_comparison.csv"
    tab.to_csv(out_csv, index=False)

    lines = [
        "# 过去 N 天 lookback 情绪方案对比（固定 ft finance_zh）",
        "",
        "## 设定",
        "",
        "| 项目 | 配置 |",
        "|------|------|",
        "| 模型 | 冻结 `ft_models/finance_zh`（不重训） |",
        "| 窗口 | 交易日 T 使用 `report_date ∈ [T−L, T)`，**不含当天** |",
        "| Lookback L | 3 / 7 / 14 / 30 / 60 / 180 自然日 |",
        "| 预测 | 次日 hold=1；未来7日 hold=7 |",
        "| 回测窗 | 2025-07-01 ~ 2026-12-31 |",
        "",
        "### 方案说明",
        "",
        "| scheme | 含义 |",
        "|--------|------|",
        "| same_day_baseline | 原：当天研报 → 次日/7日（对照） |",
        "| wavg_uniform | 窗口内文档概率等权平均 |",
        "| wavg_linear | 文档概率按新旧线性加权 |",
        "| wavg_exp | 文档概率指数衰减加权 |",
        "| daily_wavg_linear | 先按报告日聚合，再按日线性加权 |",
        "| concat | 窗口文本拼接后一次过模型（新→旧，512截断） |",
        "",
        "## 汇总表",
        "",
        "| 方案 | L | 预测H | 盈利品种比 | 平均收益 | 中位收益 | 平均Sharpe | 中位Sharpe | 平均MDD |",
        "|------|---|-------|------------|----------|----------|------------|------------|---------|",
    ]
    if not tab.empty:
        for _, r in tab.sort_values(["pred_h", "scheme", "lookback"]).iterrows():
            lines.append(
                f"| {r['scheme']} | {int(r['lookback'])} | {int(r['pred_h'])}d | "
                f"{r['profit_ratio']*100:.1f}% | {r['avg_ret']*100:.2f}% | {r['med_ret']*100:.2f}% | "
                f"{r['avg_sharpe']:.2f} | {r['med_sharpe']:.2f} | {r['avg_mdd']*100:.2f}% |"
            )
    lines.append("")

    # best per pred_h by avg_ret then med_sharpe
    lines += ["## 各预测 horizon 最优（按均收益，其次中位Sharpe）", ""]
    if not tab.empty:
        for h in PRED_HS:
            sub = tab[tab["pred_h"] == h].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["avg_ret", "med_sharpe"], ascending=False)
            best = sub.iloc[0]
            lines.append(
                f"- **预测 {h}d**：`{best['scheme']}` L={int(best['lookback'])} "
                f"— 均收益 {best['avg_ret']*100:.2f}% / 中位Sharpe {best['med_sharpe']:.2f} / "
                f"盈利比 {best['profit_ratio']*100:.1f}%"
            )
        lines.append("")

    readme = RESULTS / "lookback_comparison_README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {out_csv}")
    print(f"[done] {readme}")
    if not tab.empty:
        print(tab.sort_values(["pred_h", "avg_ret"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
