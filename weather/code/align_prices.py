#!/usr/bin/env python3
"""Align weather factors with main-contract forward returns for CF/SR/AP/CJ."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from common import FACTOR_DIR, SYMBOLS, ensure_dirs  # noqa: E402
from linear_ridge_walkforward import TREE, pick_main_fast, _read_contract_csv  # noqa: E402


def load_symbol_fwd(symbol: str) -> pd.DataFrame:
    sym_dir = TREE / symbol
    if not sym_dir.is_dir():
        # try lowercase
        sym_dir = TREE / symbol.lower()
    if not sym_dir.is_dir():
        raise FileNotFoundError(f"no contract dir for {symbol} under {TREE}")
    parts = []
    for csv in sorted(sym_dir.glob("*.csv")):
        try:
            raw = _read_contract_csv(csv)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {csv.name}: {exc}", flush=True)
            continue
        parts.append(raw[["date", "code", "close"]])
    if not parts:
        raise RuntimeError(f"no contract CSVs for {symbol}")
    df = pd.concat(parts, ignore_index=True).sort_values(["code", "date"])
    nxt = df.groupby("code")["close"].shift(-1)
    same = df.groupby("code")["date"].shift(-1)
    gap = (same - df["date"]).dt.days
    df["fwd_ret"] = np.log(nxt / df["close"])
    df.loc[gap.isna() | (gap > 10), "fwd_ret"] = np.nan
    df["symbol"] = symbol
    main = pick_main_fast(df)
    return main[["date", "symbol", "code", "close", "fwd_ret"]].sort_values("date")


def load_factors(symbol: str) -> pd.DataFrame:
    path = FACTOR_DIR / f"{symbol}_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing factors {path}")
    fac = pd.read_parquet(path)
    fac["date"] = pd.to_datetime(fac["date"])
    return fac


def align_symbol(symbol: str, weather_lag_days: int = 1) -> pd.DataFrame:
    """Join T-lag weather onto trade dates (no look-ahead)."""
    px = load_symbol_fwd(symbol)
    fac = load_factors(symbol).drop(columns=["symbol"], errors="ignore")
    fac = fac.copy()
    fac["date"] = fac["date"] + pd.Timedelta(days=weather_lag_days)
    # weather is calendar daily; map onto trade days via merge_asof backward
    px = px.sort_values("date")
    fac = fac.sort_values("date")
    out = pd.merge_asof(px, fac, on="date", direction="backward")
    out["symbol"] = symbol
    return out


def align_all(symbols: tuple[str, ...] = SYMBOLS, weather_lag_days: int = 1) -> pd.DataFrame:
    ensure_dirs()
    frames = []
    for sym in symbols:
        print(f"[align] {sym}", flush=True)
        frames.append(align_symbol(sym, weather_lag_days=weather_lag_days))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])


if __name__ == "__main__":
    df = align_all()
    out = FACTOR_DIR / "aligned_panel.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {out} rows={len(df)} symbols={sorted(df['symbol'].unique())}", flush=True)
