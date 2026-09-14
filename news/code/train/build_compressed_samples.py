#!/usr/bin/env python3
"""Attach forward-return weak labels to DeepSeek per-variety compressed records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, forward_horizon_ret, load_symbol_ohlc, ret_to_label  # noqa: E402

DEFAULT_ITEMS = (
    DATA_DIR / "deepseek_variety_compressed" / "compressed_items.parquet"
)
DEFAULT_OUT = DATA_DIR / "deepseek_variety_compressed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 7])
    args = parser.parse_args()

    items = pd.read_parquet(args.items)
    items["report_date"] = pd.to_datetime(items["report_date"]).dt.normalize()
    items["symbol"] = items["symbol"].astype(str).str.upper()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, pd.DataFrame] = {}
    for symbol in tqdm(sorted(items["symbol"].unique()), desc="load prices"):
        prices = load_symbol_ohlc(symbol)
        if not prices.empty:
            cache[symbol] = prices.set_index("date").sort_index()

    for horizon in args.horizon:
        rows: list[dict] = []
        skipped = 0
        for row in tqdm(items.itertuples(index=False), total=len(items), desc=f"labels h{horizon}"):
            prices = cache.get(row.symbol)
            if prices is None:
                skipped += 1
                continue
            trade_date, forward_ret = forward_horizon_ret(
                prices, pd.Timestamp(row.report_date), horizon
            )
            if trade_date is None or not pd.notna(forward_ret):
                skipped += 1
                continue
            data = row._asdict()
            data.update(
                {
                    "trade_date": pd.Timestamp(trade_date).normalize(),
                    f"log_ret_{horizon}d": float(forward_ret),
                    "log_ret_1d": float(forward_ret),
                    "label": ret_to_label(float(forward_ret), horizon=horizon),
                    "label_horizon": int(horizon),
                }
            )
            rows.append(data)

        out = pd.DataFrame(rows)
        path = args.out_dir / f"samples_h{horizon}.parquet"
        out.to_parquet(path, index=False)
        out.drop(columns=["text"], errors="ignore").to_csv(
            path.with_suffix(".csv"), index=False
        )
        print(
            f"[done] h={horizon} rows={len(out)} skipped={skipped} -> {path}"
        )
        if len(out):
            print(out.groupby("label").size().to_dict())


if __name__ == "__main__":
    main()
