#!/usr/bin/env python3
"""Backtest rule-based and BoW-Ridge news factors vs pos64."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path("/home/workspace/lab/UniFutures/code/experiment")
NEWS_TRAIN = Path(__file__).resolve().parents[1] / "train"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(NEWS_TRAIN))

from common import DATA_DIR, RESULTS_ROOT  # noqa: E402
from linear_ridge_walkforward import build_panel, ls_pnl, metrics_from_pnl, MIN_CS  # noqa: E402
from news_linear_eval import BT_START, blend_ls, factor_ls, fmt, zscore  # noqa: E402

OUT = RESULTS_ROOT / "news_rule"


def eval_factor(panel: pd.DataFrame, fac: pd.DataFrame, col: str, lookback: int) -> dict:
    p = panel.merge(fac, on=["date", "symbol"], how="left")
    sub = p.dropna(subset=[col])
    pnl = factor_ls(sub, col)
    st = metrics_from_pnl(pnl)
    blend = blend_ls(sub, ["pos_64", col])
    stb = metrics_from_pnl(blend)
    return {
        "lookback": lookback,
        "factor": col,
        "sharpe": fmt(st["sharpe"]),
        "return": fmt(st["return"]),
        "max_dd": fmt(st["max_dd"]),
        "days": st["days"],
        "blend_sharpe": fmt(stb["sharpe"]),
        "blend_return": fmt(stb["return"]),
    }


def main() -> None:
    panel = build_panel(BT_START).dropna(subset=["fwd_ret", "pos_64"])
    rows = []
    st = metrics_from_pnl(factor_ls(panel, "pos_64"))
    rows.append({"lookback": None, "factor": "pos64", "sharpe": fmt(st["sharpe"]), "return": fmt(st["return"]),
                 "max_dd": fmt(st["max_dd"]), "days": st["days"], "blend_sharpe": None, "blend_return": None})

    for lb in (7, 14):
        rule_path = DATA_DIR / f"rule_factors_L{lb}.parquet"
        if rule_path.is_file():
            fac = pd.read_parquet(rule_path)
            fac["date"] = pd.to_datetime(fac["date"])
            for col in ("rule_edge", "rule_title", "rule_lex", "rule_cnt"):
                if col in fac.columns:
                    rows.append(eval_factor(panel, fac, col, lb))

        bow_path = DATA_DIR / f"bow_factors_L{lb}.parquet"
        if bow_path.is_file():
            fac = pd.read_parquet(bow_path)
            fac["date"] = pd.to_datetime(fac["date"])
            if "rule_bow" in fac.columns:
                rows.append(eval_factor(panel, fac.rename(columns={"rule_bow": "rule_bow"}), "rule_bow", lb))

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    (OUT / "report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(df.to_string(index=False), flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
