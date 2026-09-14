#!/usr/bin/env python3
"""Convert daily rank scores → top/bottom percentile positions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import BOTTOM_PCT, HORIZONS, TOP_PCT


def scores_to_positions(
    df: pd.DataFrame,
    score_col: str,
    top_pct: float = TOP_PCT,
    bottom_pct: float = BOTTOM_PCT,
) -> pd.DataFrame:
    out = df.copy()
    positions = []
    ranks = []
    for _, g in out.groupby("trade_date", sort=False):
        s = g[score_col].to_numpy(dtype=float)
        n = len(s)
        if n < 3 or np.nanstd(s) < 1e-12:
            positions.extend([0] * n)
            ranks.extend([0.5] * n)
            continue
        # percentile rank in [0,1]
        order = s.argsort().argsort().astype(float)
        pct = (order + 0.5) / n
        pos = np.zeros(n, dtype=int)
        pos[pct >= 1.0 - top_pct] = 1
        pos[pct <= bottom_pct] = -1
        positions.extend(pos.tolist())
        ranks.extend(pct.tolist())
    out["position"] = positions
    out["score_pct"] = ranks
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=1, choices=list(HORIZONS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-pct", type=float, default=TOP_PCT)
    ap.add_argument("--bottom-pct", type=float, default=BOTTOM_PCT)
    args = ap.parse_args()

    df = pd.read_parquet(args.signals) if args.signals.suffix == ".parquet" else pd.read_csv(args.signals)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    col = f"score_h{args.horizon}"
    if col not in df.columns:
        raise SystemExit(f"missing {col}")
    out = scores_to_positions(df, col, args.top_pct, args.bottom_pct)
    # Keep backtest-compatible columns
    out = out.rename(columns={col: "score"})
    keep = [
        "symbol",
        "trade_date",
        "position",
        "score",
        "score_pct",
        f"ret_h{args.horizon}",
        f"valid_h{args.horizon}",
    ]
    keep = [c for c in keep if c in out.columns]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out[keep].to_parquet(args.out, index=False)
    out[keep].to_csv(args.out.with_suffix(".csv"), index=False)
    # Collapse diagnostics
    import json

    pos = out["position"].to_numpy()
    print(
        json.dumps(
            {
                "n": int(len(out)),
                "long_rate": float((pos == 1).mean()),
                "short_rate": float((pos == -1).mean()),
                "flat_rate": float((pos == 0).mean()),
                "score_std": float(out["score"].std()),
                "all_long": bool((pos == 1).all()),
                "all_short": bool((pos == -1).all()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
