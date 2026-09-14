#!/usr/bin/env python3
"""Aggregate retrain_v1 scheme results into comparison table + README."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from schemes import RESULT_ROOT, SCHEMES


def sharpe(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or arr.std() <= 1e-12:
        return 0.0
    return float(arr.mean() / arr.std() * math.sqrt(252))


def summarize_scheme(scheme_id: str, title: str, family: str, axis: str) -> dict | None:
    path = RESULT_ROOT / scheme_id / "summary.csv"
    daily_path = RESULT_ROOT / scheme_id / "daily.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "bps" in df.columns:
        zero = df[df["bps"] == 0].copy()
        two = df[df["bps"] == 2].copy()
    else:
        zero = df.copy()
        two = pd.DataFrame()
    if zero.empty:
        return None
    row = {
        "scheme": scheme_id,
        "title": title,
        "family": family,
        "axis": axis,
        "n_symbols": int(zero["symbol"].nunique()) if "symbol" in zero.columns else len(zero),
        "profit_ratio": float((zero["return"] > 0).mean()),
        "avg_return": float(zero["return"].mean()),
        "med_symbol_sharpe": float(zero["sharpe"].median()),
        "avg_symbol_sharpe": float(zero["sharpe"].mean()),
        "avg_max_dd": float(zero["max_dd"].mean()) if "max_dd" in zero.columns else np.nan,
        "active_ratio": float(zero["active_ratio"].mean()) if "active_ratio" in zero.columns else np.nan,
    }
    if not two.empty:
        row["avg_return_2bp"] = float(two["return"].mean())
        row["med_symbol_sharpe_2bp"] = float(two["sharpe"].median())
    if daily_path.exists():
        daily = pd.read_csv(daily_path, parse_dates=["date"])
        port = daily.groupby("date", sort=True)["strategy_ret"].mean()
        row["ew_portfolio_return"] = float(np.exp(port.sum()) - 1.0)
        row["ew_portfolio_sharpe"] = sharpe(port.to_numpy())
    return row


def main() -> None:
    rows = []
    for s in SCHEMES:
        row = summarize_scheme(s.id, s.title, s.family, s.axis)
        if row:
            rows.append(row)
    tab = pd.DataFrame(rows)
    if tab.empty:
        print("no results yet")
        return
    tab = tab.sort_values(
        ["med_symbol_sharpe", "ew_portfolio_sharpe", "avg_return"],
        ascending=False,
        na_position="last",
    )
    out_csv = RESULT_ROOT / "comparison.csv"
    tab.to_csv(out_csv, index=False)

    lines = [
        "# Retrain v1 — 32 差异化重训方案",
        "",
        "> **Retrospective exploratory.** 本批全部重新训练权重，与融合规则矩阵正交。",
        "> OOS 2025-07 起区间此前已被查看，不能当作 pristine holdout。",
        "",
        "## 完成进度",
        "",
        f"- 已出结果：{len(tab)} / 32",
        "",
        "## Top 结果（零成本主排序）",
        "",
        "| 方案 | 族 | 差异轴 | 中位Sharpe | 组合Sharpe | 平均收益 | 盈利比 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in tab.head(15).iterrows():
        lines.append(
            f"| `{r['scheme']}` {r['title']} | {r['family']} | {r['axis']} | "
            f"{r['med_symbol_sharpe']:.2f} | {r.get('ew_portfolio_sharpe', float('nan')):.2f} | "
            f"{r['avg_return']*100:.2f}% | {r['profit_ratio']*100:.1f}% |"
        )
    lines += ["", "完整表见 `comparison.csv`。"]
    readme = RESULT_ROOT / "retrain_comparison_README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {out_csv}")
    print(f"[done] {readme}")
    print(tab[["scheme", "family", "med_symbol_sharpe", "avg_return"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
