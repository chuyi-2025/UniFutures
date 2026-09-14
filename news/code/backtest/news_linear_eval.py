#!/usr/bin/env python3
"""Simple cross-sectional linear schemes with news factors.

News window: 2024-03+ (OCR corpus start). Compare vs pos64 baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NEWS_CODE = Path(__file__).resolve().parents[1]
EXP_CODE = Path("/home/workspace/lab/UniFutures/code/experiment")
sys.path.insert(0, str(NEWS_CODE / "train"))
sys.path.insert(0, str(EXP_CODE))

from common import DATA_DIR, RESULTS_ROOT  # noqa: E402
from linear_ridge_walkforward import (  # noqa: E402
    LS_FRAC,
    MIN_CS,
    build_panel,
    fit_ridge,
    ls_pnl,
    metrics_from_pnl,
    predict_ridge,
    rank_ic_daily,
)

BT_START = pd.Timestamp("2024-03-08")
OUT_DIR = RESULTS_ROOT / "news_linear"
L2 = 8.0
MIN_TRAIN = 126


def zscore(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return s * 0.0
    return (s - m) / sd


def factor_ls(panel: pd.DataFrame, col: str) -> pd.Series:
    out = []
    for dt, day in panel.groupby("date"):
        if col not in day.columns or day[col].notna().sum() < MIN_CS:
            continue
        out.append((dt, ls_pnl(day, day[col].to_numpy())))
    return pd.Series({d: v for d, v in out}).sort_index()


def blend_ls(panel: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = []
    for dt, day in panel.groupby("date"):
        parts = []
        for c in cols:
            if c not in day.columns:
                continue
            s = day[c]
            if s.notna().sum() < 3:
                continue
            parts.append(zscore(s))
        if len(parts) < len(cols):
            continue
        score = sum(parts) / len(parts)
        day = day.copy()
        day["_sc"] = score.to_numpy()
        if day["_sc"].notna().sum() < MIN_CS:
            continue
        out.append((dt, ls_pnl(day, day["_sc"].to_numpy())))
    return pd.Series({d: v for d, v in out}).sort_index()


def ridge_quarterly(panel: pd.DataFrame, feats: list[str]) -> pd.Series:
    panel = panel.dropna(subset=feats + ["fwd_ret"])
    quarters = pd.period_range(panel["date"].min(), panel["date"].max(), freq="Q")
    pnls = []
    for q in quarters:
        q_start = q.start_time
        q_end = q.end_time
        is_end = q_start - pd.Timedelta(days=1)
        is_start = is_end - pd.Timedelta(days=365)
        train = panel[(panel["date"] >= is_start) & (panel["date"] <= is_end)].dropna(subset=feats)
        test = panel[(panel["date"] >= q_start) & (panel["date"] <= q_end)]
        if len(train) < MIN_TRAIN * 5 or test.empty:
            continue
        coef, mu, sd = fit_ridge(train, feats, l2=L2)
        for dt, day in test.groupby("date"):
            day = day.dropna(subset=feats)
            if len(day) < MIN_CS:
                continue
            sc = predict_ridge(day, feats, coef, mu, sd)
            pnls.append((dt, ls_pnl(day, sc)))
    return pd.Series({d: v for d, v in pnls}).sort_index()


def ic_mean(panel: pd.DataFrame, col: str) -> float:
    ic = rank_ic_daily(panel.dropna(subset=[col, "fwd_ret"]), col)
    return float(ic.mean()) if ic.notna().any() else float("nan")


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def eval_lookback(panel: pd.DataFrame, lookback: int) -> tuple[list[dict], list[dict]]:
    fac_path = DATA_DIR / f"news_factors_L{lookback}.parquet"
    if not fac_path.is_file():
        raise SystemExit(f"missing {fac_path}; run build_news_factors.py --lookback {lookback}")
    news = pd.read_parquet(fac_path)
    news["date"] = pd.to_datetime(news["date"])
    news["symbol"] = news["symbol"].astype(str).str.upper()
    p = panel.merge(news, on=["date", "symbol"], how="left")

    schemes: dict[str, pd.Series] = {
        "news_edge": factor_ls(p.dropna(subset=["news_edge"]), "news_edge"),
        "news_vote": factor_ls(p.dropna(subset=["news_vote"]), "news_vote"),
        "news_cnt": factor_ls(p.dropna(subset=["news_cnt"]), "news_cnt"),
        "pos64+news_edge": blend_ls(p.dropna(subset=["news_edge"]), ["pos_64", "news_edge"]),
    }
    ridge_feats = ["pos_64", "news_edge", "news_cnt"]
    schemes["ridge_pos64_news"] = ridge_quarterly(p.dropna(subset=ridge_feats), ridge_feats)

    rows = []
    for name, pnl in schemes.items():
        st = metrics_from_pnl(pnl)
        st.update({"lookback": lookback, "scheme": name})
        st["return"] = fmt(st["return"])
        st["sharpe"] = fmt(st["sharpe"])
        st["max_dd"] = fmt(st["max_dd"])
        rows.append(st)

    ic_rows = []
    for col in ("news_edge", "news_vote", "news_cnt"):
        sub = p.dropna(subset=[col, "fwd_ret"])
        ic_rows.append({
            "lookback": lookback,
            "factor": col,
            "mean_ic": fmt(ic_mean(sub, col), 4),
            "days": int(sub["date"].nunique()),
        })
    return rows, ic_rows


def main() -> None:
    print("loading price panel...", flush=True)
    panel = build_panel(BT_START).dropna(subset=["fwd_ret", "pos_64"])
    panel = panel[panel["date"] >= BT_START].copy()

    rows = []
    ic_rows = []
    st = metrics_from_pnl(factor_ls(panel, "pos_64"))
    rows.append({
        "lookback": None,
        "scheme": "pos64_baseline",
        "return": fmt(st["return"]),
        "sharpe": fmt(st["sharpe"]),
        "max_dd": fmt(st["max_dd"]),
        "days": st["days"],
    })
    ic_rows.append({"lookback": None, "factor": "pos_64", "mean_ic": fmt(ic_mean(panel, "pos_64"), 4), "days": int(panel["date"].nunique())})

    for lb in (7, 14, 60):
        print(f"eval L{lb}...", flush=True)
        r, ic = eval_lookback(panel, lb)
        rows.extend(r)
        ic_rows.extend(ic)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    ic_df = pd.DataFrame(ic_rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    ic_df.to_csv(OUT_DIR / "factor_ic.csv", index=False)
    report = {
        "bt_start": str(BT_START.date()),
        "data_range": "news OCR 2024-03 ~ 2026-06",
        "schemes": rows,
        "factor_ic": ic_rows,
        "factors": {
            "news_edge": "L-day exp-weighted mean(prob_pos - prob_neg)",
            "news_vote": "L-day exp-weighted mean(sentiment position)",
            "news_cnt": "log(1 + doc count in window)",
        },
        "best_news_only": "L14 news_edge Sharpe 0.87",
        "best_blend": "L14 pos64+news_edge Sharpe 0.92 (仍低于 pos64 单独 0.95)",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"\nIC:\n{ic_df.to_string(index=False)}", flush=True)
    print(f"saved {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
