#!/usr/bin/env python3
"""Aggregate chunk predictions to one signal per document and symbol."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROB_COLS = ["prob_0", "prob_1", "prob_2"]


def class_to_position(pred_class: int) -> int:
    return {0: 1, 1: 0, 2: -1}[int(pred_class)]


def majority_position(values: pd.Series) -> int:
    total = int(values.astype(int).sum())
    if total > 0:
        return 1
    if total < 0:
        return -1
    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    return 1 if positive > negative else (-1 if negative > positive else 0)


def aggregate(signals: pd.DataFrame, method: str) -> pd.DataFrame:
    keys = ["doc_id", "symbol", "report_date", "trade_date"]
    rows: list[dict] = []
    for key, group in signals.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row["n_chunks"] = len(group)
        if method == "mean":
            probs = group[PROB_COLS].mean().to_numpy(dtype=float)
            pred_class = int(np.argmax(probs))
            row.update({col: float(probs[i]) for i, col in enumerate(PROB_COLS)})
            row["pred_class"] = pred_class
            row["position"] = class_to_position(pred_class)
        else:
            row["position"] = majority_position(group["position"])
            row["pred_class"] = {1: 0, 0: 1, -1: 2}[row["position"]]
            row.update(
                {
                    "vote_positive": int((group["position"] > 0).sum()),
                    "vote_neutral": int((group["position"] == 0).sum()),
                    "vote_negative": int((group["position"] < 0).sum()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()

    signals = pd.read_parquet(args.signals)
    missing = set(PROB_COLS + ["doc_id", "symbol", "report_date", "trade_date", "position"]) - set(
        signals.columns
    )
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for method in ("mean", "vote"):
        out = aggregate(signals, method)
        path = args.out_prefix.with_name(f"{args.out_prefix.name}_{method}.parquet")
        out.to_parquet(path, index=False)
        out.to_csv(path.with_suffix(".csv"), index=False)
        print(
            f"[done] method={method} chunks={len(signals)} "
            f"doc_signals={len(out)} -> {path}"
        )


if __name__ == "__main__":
    main()
