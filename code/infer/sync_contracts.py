#!/usr/bin/env python3
"""Append-sync contract CSVs from Tree-Stock -> UniFutures for configured symbols."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONTRACTS_DIR, SRC_CONTRACTS_DIR, SYMBOLS


def append_csv(src: Path, dst: Path) -> int:
    src_df = pd.read_csv(src, parse_dates=["date"])
    if dst.exists():
        dst_df = pd.read_csv(dst, parse_dates=["date"])
        last = dst_df["date"].max()
        add = src_df[src_df["date"] > last].copy()
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        add = src_df
    if add.empty:
        return 0
    add["date"] = add["date"].dt.strftime("%Y-%m-%d")
    add.to_csv(dst, mode="a", header=not dst.exists(), index=False)
    return len(add)


def main() -> None:
    total = 0
    for sym in SYMBOLS:
        src_dir = SRC_CONTRACTS_DIR / sym
        dst_dir = CONTRACTS_DIR / sym
        if not src_dir.is_dir():
            print(f"{sym}: skip (no source dir)")
            continue
        n_sym = 0
        for src in sorted(src_dir.glob("*.csv")):
            n = append_csv(src, dst_dir / src.name)
            if n:
                print(f"{sym}/{src.name}: +{n}")
            n_sym += n
        print(f"{sym}: appended {n_sym} rows")
        total += n_sym
    print(f"done, total appended={total}")


if __name__ == "__main__":
    main()
