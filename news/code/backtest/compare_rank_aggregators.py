#!/usr/bin/env python3
"""Compare exp / linear / bigru_pos aggregators vs lookback baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train" / "rank_lora"))

from backtest_rank_portfolio import run_rank_backtest  # noqa: E402
from config import (  # noqa: E402
    AGGREGATORS,
    HORIZONS,
    RESULT_ROOT,
    RUN_ROOT,
    SEEDS,
)


def collect_agg_rows(tag: str, start: str, end: str) -> list[dict]:
    rows = []
    for name in AGGREGATORS:
        for seed in SEEDS:
            run = RUN_ROOT / "aggregators" / tag / name / f"seed_{seed}"
            sig = run / "signals_oos.parquet"
            meta_path = run / "meta.json"
            if not sig.exists():
                continue
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            for h in HORIZONS:
                out = RESULT_ROOT / tag / name / f"seed_{seed}" / f"h{h}"
                metrics = run_rank_backtest(
                    pd.read_parquet(sig),
                    out,
                    horizon=h,
                    start=pd.Timestamp(start),
                    end=pd.Timestamp(end),
                )
                c0 = metrics["costs"][0]
                rows.append(
                    {
                        "family": "rank_lora",
                        "tag": tag,
                        "aggregator": name,
                        "seed": seed,
                        "horizon": h,
                        "val_ic_mean": meta.get("val", {}).get("ic_mean"),
                        "oos_ic_mean": metrics["rank_ic_mean"],
                        "oos_icir": metrics["rank_icir"],
                        "median_sharpe_0bp": c0["median_sharpe"],
                        "ew_sharpe_0bp": c0["ew_sharpe"],
                        "mean_return_0bp": c0["mean_return"],
                        "long_rate": metrics["collapse"]["long_rate"],
                        "short_rate": metrics["collapse"]["short_rate"],
                        "flat_rate": metrics["collapse"]["flat_rate"],
                        "score_std": metrics["collapse"]["score_std"],
                        "all_long": metrics["collapse"]["all_long"],
                        "all_short": metrics["collapse"]["all_short"],
                        "median_sharpe_2bp": metrics["costs"][2]["median_sharpe"],
                        "median_sharpe_5bp": metrics["costs"][5]["median_sharpe"],
                    }
                )
    return rows


def baseline_note() -> list[dict]:
    """Attach existing lookback baseline summaries if present (read-only)."""
    rows = []
    lookback_cmp = Path("/home/workspace/lab/UniFutures/news/data/results/lookback_comparison.csv")
    if lookback_cmp.exists():
        df = pd.read_csv(lookback_cmp)
        for _, r in df.iterrows():
            rows.append(
                {
                    "family": "baseline_lookback",
                    "tag": "existing",
                    "aggregator": f"{r.get('scheme', '')}_L{r.get('lookback', '')}",
                    "seed": "-",
                    "horizon": r.get("hold_days", r.get("horizon", 1)),
                    "mean_return_0bp": r.get("mean_return", r.get("avg_return")),
                    "median_sharpe_0bp": r.get("median_sharpe", r.get("med_sharpe")),
                    "note": "retrospective existing baseline; not re-run here",
                }
            )
    # Hardcoded from README if csv schema differs
    for name, h, mean_ret, med_sh in [
        ("concat_L60", 1, 0.1582, 0.69),
        ("wavg_exp_L60", 1, 0.1572, 0.69),
        ("ema_sent_10", 1, None, None),
    ]:
        rows.append(
            {
                "family": "baseline_ref",
                "aggregator": name,
                "horizon": h,
                "mean_return_0bp": mean_ret,
                "median_sharpe_0bp": med_sh,
                "note": "from lookback_comparison_README / prior runs; retrospective OOS",
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="lora")
    ap.add_argument("--start", type=str, default="2025-07-01")
    ap.add_argument("--end", type=str, default="2026-12-31")
    ap.add_argument("--out", type=Path, default=RESULT_ROOT / "comparison.csv")
    args = ap.parse_args()

    rows = collect_agg_rows(args.tag, args.start, args.end)
    rows.extend(baseline_note())
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    focus = df[df["family"] == "rank_lora"].copy()
    if len(focus):
        rank = (
            focus.groupby(["aggregator", "horizon"], as_index=False)["oos_ic_mean"]
            .mean()
            .sort_values(["horizon", "oos_ic_mean"], ascending=[True, False])
        )
        rank.to_csv(RESULT_ROOT / f"rank_by_ic_{args.tag}.csv", index=False)
        print(rank.to_string(index=False))
        collapsed = focus[focus["all_long"] | focus["all_short"]]
        print(f"collapsed runs: {len(collapsed)} / {len(focus)}")
    print(f"[done] {args.out}")
    print("NOTE: 2025-07+ is retrospective OOS, not a pristine holdout.")


if __name__ == "__main__":
    main()
