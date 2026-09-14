#!/usr/bin/env python3
"""Deep analysis: yearly Sharpe, IC/IR, turnover, 2026 1-lot 4/6 names."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path("/home/workspace/lab/UniFutures/code/experiment")
NEWS_TRAIN = Path(__file__).resolve().parents[1] / "train"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(NEWS_TRAIN))

from common import DATA_DIR, RESULTS_ROOT  # noqa: E402
from linear_ridge_walkforward import (  # noqa: E402
    LS_FRAC,
    MIN_CS,
    build_panel,
    ic_stats,
    ls_pnl,
    metrics_from_pnl,
    rank_ic_daily,
)
from pos64_margin_cap_search import attach_margin  # noqa: E402
from pos64_margin_integer_search import CAPITAL, dollar_1lot, load_close, to_log  # noqa: E402
from six_name_select import pick_ls_n  # noqa: E402
from two_name_select import turnover  # noqa: E402

BT_START = pd.Timestamp("2024-03-08")
OUT = RESULTS_ROOT / "news_deep"
MARGIN_CAP = 50_000.0


def zscore(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return s * 0.0
    return (s - m) / sd


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def picks_ls(day: pd.DataFrame, score: np.ndarray) -> list[tuple[str, float]]:
    d = day.copy()
    d["_sc"] = score
    d = d[d["_sc"].notna() & d["fwd_ret"].notna()].reset_index(drop=True)
    if len(d) < MIN_CS:
        return []
    k = max(3, int(np.floor(len(d) * LS_FRAC)))
    if len(d) < 2 * k:
        return []
    return pick_ls_n(d, d["_sc"].to_numpy(), k)


def run_ls(panel: pd.DataFrame, col: str, *, require: bool = False) -> tuple[pd.Series, pd.Series, pd.Series]:
    pnls, tos, nn = [], [], []
    prev: list[tuple[str, float]] | None = None
    for dt, day in panel.groupby("date"):
        d = day.dropna(subset=[col, "fwd_ret"]) if require else day.dropna(subset=["fwd_ret"])
        if require:
            d = d.dropna(subset=[col])
        if len(d) < MIN_CS:
            prev = None
            continue
        sc = d[col].to_numpy()
        p = ls_pnl(d, sc)
        if not np.isfinite(p):
            prev = None
            continue
        picks = picks_ls(d, sc)
        pnls.append((dt, p))
        tos.append((dt, turnover(prev, picks)))
        nn.append((dt, len(picks)))
        prev = picks
    return (
        pd.Series({d: v for d, v in pnls}).sort_index(),
        pd.Series({d: v for d, v in tos}).sort_index(),
        pd.Series({d: v for d, v in nn}).sort_index(),
    )


def run_blend(panel: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    pnls, tos, nn = [], [], []
    prev = None
    for dt, day in panel.groupby("date"):
        d = day.dropna(subset=cols + ["fwd_ret"])
        if len(d) < MIN_CS:
            prev = None
            continue
        sc = sum(zscore(d[c]) for c in cols) / len(cols)
        d = d.copy()
        d["_sc"] = sc.to_numpy()
        p = ls_pnl(d, d["_sc"].to_numpy())
        if not np.isfinite(p):
            prev = None
            continue
        picks = picks_ls(d, d["_sc"].to_numpy())
        pnls.append((dt, p))
        tos.append((dt, turnover(prev, picks)))
        nn.append((dt, len(picks)))
        prev = picks
    return (
        pd.Series({d: v for d, v in pnls}).sort_index(),
        pd.Series({d: v for d, v in tos}).sort_index(),
        pd.Series({d: v for d, v in nn}).sort_index(),
    )


def yearly(pnl: pd.Series) -> list[dict]:
    rows = []
    for y, g in pnl.groupby(pnl.index.year):
        m = metrics_from_pnl(g)
        rows.append({"year": int(y), "return": fmt(m["return"]), "sharpe": fmt(m["sharpe"], 3), "max_dd": fmt(m["max_dd"]), "days": m["days"]})
    return rows


def day_score(d: pd.DataFrame, col: str, mode: str) -> np.ndarray:
    if mode == "blend":
        return (zscore(d["pos_64"]) + zscore(d["rule_edge"])).to_numpy() / 2.0
    return d[col].to_numpy()


def run_1lot(panel: pd.DataFrame, col: str, n_each: int, *, mode: str = "ls", year: int = 2026) -> dict:
    px = load_close()
    p = panel.merge(px, on=["date", "symbol", "code"], how="left")
    if "close" not in p.columns and "close" in px.columns:
        p = p.rename(columns={"close": "close"})
    pnls, tos, mars, nn = [], [], [], []
    prev = None
    need = ["fwd_ret", "pos_64", "rule_edge"] if mode == "blend" else [col, "fwd_ret"]
    for dt, day in p.groupby("date"):
        if pd.Timestamp(dt).year != year:
            continue
        d = attach_margin(day.drop_duplicates("symbol").reset_index(drop=True))
        d = d[d["lot_margin"].notna() & (d["lot_margin"] <= MARGIN_CAP)]
        d = d.dropna(subset=need)
        if len(d) < 2 * n_each:
            prev = None
            continue
        sc = day_score(d, col, mode)
        picks = pick_ls_n(d, sc, n_each)
        if not picks:
            prev = None
            continue
        cny, mar, _, n = dollar_1lot(d, picks)
        pnls.append((dt, cny))
        mars.append((dt, mar))
        nn.append((dt, n))
        tos.append((dt, turnover(prev, picks)))
        prev = picks
    s = pd.Series({d: v for d, v in pnls}).sort_index()
    log = to_log(s / CAPITAL)
    m = metrics_from_pnl(log)
    return {
        "year": year,
        "n_each": n_each,
        "max_names": 2 * n_each,
        "margin_cap": MARGIN_CAP,
        "return_cny_pct": fmt(s.sum() / CAPITAL),
        "return": fmt(m["return"]),
        "sharpe": fmt(m["sharpe"], 3),
        "max_dd": fmt(m["max_dd"]),
        "days": m["days"],
        "mean_turnover": fmt(pd.Series({d: v for d, v in tos}).mean(), 3),
        "mean_margin": fmt(pd.Series({d: v for d, v in mars}).mean(), 0),
        "mean_names": fmt(pd.Series({d: v for d, v in nn}).mean(), 2),
    }


def analyze(name: str, pnl: pd.Series, to: pd.Series, nn: pd.Series, panel: pd.DataFrame, col: str, require: bool) -> dict:
    m = metrics_from_pnl(pnl)
    ic_p = panel.dropna(subset=[col, "fwd_ret"]) if require else panel.dropna(subset=["fwd_ret"])
    if require:
        ic_p = ic_p.dropna(subset=[col])
    ic = ic_stats(rank_ic_daily(ic_p, col if col != "_blend" else col))
    active = pnl.index
    full_days = int(panel["date"].nunique())
    return {
        "scheme": name,
        "return": fmt(m["return"]),
        "sharpe": fmt(m["sharpe"], 3),
        "max_dd": fmt(m["max_dd"]),
        "days": m["days"],
        "coverage": fmt(len(active) / full_days, 3),
        "mean_ic": fmt(ic.get("ic"), 4),
        "ir": fmt(ic.get("ir"), 3),
        "ic_t": fmt(ic.get("t"), 2),
        "ic_days": ic.get("n"),
        "mean_turnover": fmt(to.mean(), 3),
        "median_turnover": fmt(to.median(), 3),
        "mean_names": fmt(nn.mean(), 2),
        "yearly": yearly(pnl),
        "n_neg_years": sum(1 for r in yearly(pnl) if (r["sharpe"] or 0) < 0),
    }


def main() -> None:
    print("panel...", flush=True)
    panel = build_panel(BT_START).dropna(subset=["fwd_ret", "pos_64"])

    fac_map = {
        "news_edge_ml_L7": (DATA_DIR / "news_factors_L7.parquet", "news_edge"),
        "rule_edge_L7": (DATA_DIR / "rule_factors_L7.parquet", "rule_edge"),
        "rule_lex_L7": (DATA_DIR / "rule_factors_L7.parquet", "rule_lex"),
        "rule_title_L7": (DATA_DIR / "rule_factors_L7.parquet", "rule_title"),
        "bow_ridge_L14": (DATA_DIR / "bow_factors_L14.parquet", "rule_bow"),
    }
    merged = {"pos64": panel.copy()}
    for name, (path, col) in fac_map.items():
        if path.is_file():
            f = pd.read_parquet(path)
            f["date"] = pd.to_datetime(f["date"])
            f["symbol"] = f["symbol"].str.upper()
            merged[name] = panel.merge(f[["date", "symbol", col]], on=["date", "symbol"], how="left")
    if (DATA_DIR / "rule_factors_L7.parquet").is_file():
        rf = pd.read_parquet(DATA_DIR / "rule_factors_L7.parquet")
        rf["date"] = pd.to_datetime(rf["date"])
        rf["symbol"] = rf["symbol"].str.upper()
        bp = panel.merge(rf[["date", "symbol", "rule_edge"]], on=["date", "symbol"], how="left")
        merged["pos64+rule_edge"] = bp

    results = []
    yearly_all = []
    lot_2026 = []

    specs = [
        ("pos64", "pos64", "pos_64", False, "ls"),
        ("news_edge_ml_L7", "news_edge_ml_L7", "news_edge", True, "ls"),
        ("rule_edge_L7", "rule_edge_L7", "rule_edge", True, "ls"),
        ("rule_lex_L7", "rule_lex_L7", "rule_lex", True, "ls"),
        ("rule_title_L7", "rule_title_L7", "rule_title", True, "ls"),
        ("pos64+rule_edge", "pos64+rule_edge", "rule_edge", True, "blend"),
        ("bow_ridge_L14", "bow_ridge_L14", "rule_bow", True, "ls"),
    ]

    for name, key, col, require, mode in specs:
        if key not in merged:
            print(f"skip {name}", flush=True)
            continue
        p = merged[key]
        print(f"run {name}...", flush=True)
        if mode == "blend":
            pnl, to, nn = run_blend(p, ["pos_64", "rule_edge"])
            col_ic = "rule_edge"
        else:
            pnl, to, nn = run_ls(p, col, require=require)
            col_ic = col
        st = analyze(name, pnl, to, nn, p, col_ic, require)
        results.append({k: v for k, v in st.items() if k != "yearly"})
        for yr in st["yearly"]:
            yearly_all.append({"scheme": name, **yr})
        for n_each in (1, 2, 3):
            lot_2026.append({"scheme": name, **run_1lot(p, col, n_each, mode=mode)})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(yearly_all).to_csv(OUT / "yearly.csv", index=False)
    pd.DataFrame(lot_2026).to_csv(OUT / "lot_2026.csv", index=False)
    (OUT / "report.json").write_text(json.dumps({"summary": results, "yearly": yearly_all, "lot_2026": lot_2026}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    write_md(results, yearly_all, lot_2026)
    print(pd.DataFrame(results).to_string(index=False), flush=True)
    print(f"\nsaved {OUT}", flush=True)


def write_md(results, yearly_all, lot_2026) -> None:
    md_path = Path("/home/workspace/lab/UniFutures/docs/linear/NEWS_STRATEGY_ANALYSIS.md")
    lines = [
        "# 新闻/规则策略深度分析",
        "",
        f"> 回测起点 {BT_START.date()}；截面 LS 20%；2026 整数 1 手、保证金≤{MARGIN_CAP/1e4:.0f}万",
        "",
        "## 1. 全样本汇总",
        "",
        "| 方案 | 收益 | Sharpe | 回撤 | 交易天 | 覆盖率 | IC | IR | 均换手 | 均品种数 | 负年 |",
        "|------|------|--------|------|--------|--------|-----|-----|--------|----------|------|",
    ]
    for r in results:
        lines.append(
            f"| {r['scheme']} | {r['return']} | {r['sharpe']} | {r['max_dd']} | {r['days']} | "
            f"{r['coverage']} | {r['mean_ic']} | {r['ir']} | {r['mean_turnover']} | {r['mean_names']} | {r['n_neg_years']} |"
        )
    lines += ["", "## 2. 分年 Sharpe / 收益", ""]
    ydf = pd.DataFrame(yearly_all)
    for sch in ydf["scheme"].unique():
        lines.append(f"### {sch}")
        lines.append("")
        sub = ydf[ydf["scheme"] == sch]
        lines.append("| 年 | 收益 | Sharpe | 回撤 | 天数 |")
        lines.append("|----|------|--------|------|------|")
        for _, row in sub.iterrows():
            lines.append(f"| {int(row['year'])} | {row['return']} | {row['sharpe']} | {row['max_dd']} | {row['days']} |")
        lines.append("")
    lines += ["## 3. 2026 整数 1 手（保证金≤5万）", ""]
    lines.append("| 方案 | 结构 | 收益(对数) | Sharpe | 回撤 | 人民币收益/100万 | 均保证金 | 均换手 |")
    lines.append("|------|------|-----------|--------|------|-----------------|----------|--------|")
    ldf = pd.DataFrame(lot_2026)
    for _, row in ldf.iterrows():
        lines.append(
            f"| {row['scheme']} | {row['max_names']}品种({row['n_each']}L+{row['n_each']}S) | "
            f"{row['return']} | {row['sharpe']} | {row['max_dd']} | {row['return_cny_pct']} | "
            f"{row['mean_margin']} | {row['mean_turnover']} |"
        )
    lines += [
        "",
        "## 4. 方案说明",
        "",
        "| 方案 | 信号来源 |",
        "|------|----------|",
        "| pos64 | 64日价格位置，全品种有覆盖 |",
        "| news_edge_ml_L7 | finance_zh 微调模型，L7 exp 加权 prob多−prob空 |",
        "| rule_edge_L7 | 标题规则+词表，零训练 |",
        "| rule_lex_L7 | 仅正文词表 |",
        "| rule_title_L7 | 仅文件名/标题关键词 |",
        "| pos64+rule_edge | pos64 与 rule_edge 等权 zscore |",
        "| bow_ridge_L14 | 300词 CountVectorizer + Ridge 预测次日收益 |",
        "",
        "## 5. 解读要点",
        "",
        "- **IC/IR**：rule 系因子 IC 高于 ML news_edge；IR>0.5 算稳定，当前多数在 0.1~0.3。",
        "- **换手率**：规则因子换手通常高于 pos64（标题随事件切换快）。",
        "- **1手 2026**：与 daily 定稿不同框架；4品种=2多2空，6品种=3多3空。",
        "- **rule_edge 高 Sharpe** 含大量「上涨点评」事后动量，分年需看 2025/2026 是否持续。",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {md_path}", flush=True)


if __name__ == "__main__":
    main()
