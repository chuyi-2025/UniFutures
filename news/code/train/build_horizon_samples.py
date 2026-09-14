#!/usr/bin/env python3
"""Rebuild weak labels for multi-day forward returns from existing OCR samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_DIR,
    forward_horizon_ret,
    load_symbol_ohlc,
    ret_to_label,
)


def _build_cache(symbols: list[str]) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    for sym in tqdm(sorted(set(symbols)), desc="load contracts"):
        df = load_symbol_ohlc(sym)
        if df.empty:
            continue
        cache[sym] = df.set_index("date").sort_index()
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DATA_DIR / "samples.parquet")
    ap.add_argument("--horizon", type=int, required=True, help="trading days ahead (e.g. 7/14/30)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    h = int(args.horizon)
    if h < 1:
        raise SystemExit("horizon must be >= 1")

    base = pd.read_parquet(args.base) if args.base.suffix == ".parquet" else pd.read_csv(args.base)
    base["report_date"] = pd.to_datetime(base["report_date"])
    cache = _build_cache(base["symbol"].astype(str).tolist())

    rows = []
    skip = 0
    for _, row in tqdm(base.iterrows(), total=len(base), desc=f"align h={h}"):
        sym = str(row["symbol"])
        df = cache.get(sym)
        if df is None:
            skip += 1
            continue
        trade_date, fwd = forward_horizon_ret(df, row["report_date"], h)
        if trade_date is None or not pd.notna(fwd):
            skip += 1
            continue
        label = ret_to_label(fwd, horizon=h)
        rows.append(
            {
                "path": row["path"],
                "institution": row["institution"],
                "report_date": pd.Timestamp(row["report_date"]).normalize(),
                "variety_raw": row["variety_raw"],
                "symbol": sym,
                "text": row["text"],
                "trade_date": trade_date.normalize(),
                "log_ret_1d": float(fwd),  # cumulative H-day log ret (kept name for pipeline)
                "horizon": h,
                "label": label,
            }
        )

    out = pd.DataFrame(rows)
    out_path = args.out or (DATA_DIR / f"samples_h{h}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    out.drop(columns=["text"]).to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"[done] h={h} samples={len(out)} skip={skip} -> {out_path}")
    if len(out):
        print(out.groupby("label").size().to_dict())
        print("date range", out["report_date"].min(), "->", out["report_date"].max())


if __name__ == "__main__":
    main()
