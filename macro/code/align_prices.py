#!/usr/bin/env python3
"""Align macro factors (T-1) with AU/AG/SC main-contract forward returns."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from common import FACTOR_DIR, RESULT_DIR, TARGETS, ensure_dirs  # noqa: E402
from linear_ridge_walkforward import TREE, pick_main_fast, _read_contract_csv  # noqa: E402


def load_symbol_fwd(symbol: str) -> pd.DataFrame:
    sym_dir = TREE / symbol
    if not sym_dir.is_dir():
        raise FileNotFoundError(f"no contracts for {symbol}")
    parts = []
    for csv in sorted(sym_dir.glob("*.csv")):
        try:
            raw = _read_contract_csv(csv)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {csv.name}: {exc}", flush=True)
            continue
        parts.append(raw[["date", "code", "close"]])
    df = pd.concat(parts, ignore_index=True).sort_values(["code", "date"])
    nxt = df.groupby("code")["close"].shift(-1)
    same = df.groupby("code")["date"].shift(-1)
    gap = (same - df["date"]).dt.days
    df["fwd_ret"] = np.log(nxt / df["close"])
    df.loc[gap.isna() | (gap > 10), "fwd_ret"] = np.nan
    df["symbol"] = symbol
    return pick_main_fast(df)[["date", "symbol", "code", "close", "fwd_ret"]]


def align_all(symbols: tuple[str, ...] = TARGETS, lag_days: int = 1) -> pd.DataFrame:
    ensure_dirs()
    fac = pd.read_parquet(FACTOR_DIR / "macro_factors_daily.parquet")
    fac["date"] = pd.to_datetime(fac["date"]) + pd.Timedelta(days=lag_days)
    fac = fac.sort_values("date")
    frames = []
    for sym in symbols:
        print(f"[align] {sym}", flush=True)
        px = load_symbol_fwd(sym).sort_values("date")
        m = pd.merge_asof(px, fac, on="date", direction="backward")
        m["symbol"] = sym
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    panel = align_all()
    out = RESULT_DIR / "aligned_panel.parquet"
    ensure_dirs()
    panel.to_parquet(out, index=False)
    print(f"saved {out} rows={len(panel)} {panel['date'].min().date()}->{panel['date'].max().date()}", flush=True)
