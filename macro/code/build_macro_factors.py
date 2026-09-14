#!/usr/bin/env python3
"""Build macro z-score / change factors from daily panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FACTOR_DIR, RAW_DIR, ensure_dirs  # noqa: E402

# Core tradable predictors (avoid leakage from commodity CFDs as primary signal)
CORE = [
    "dxy",
    "usd_broad",
    "us_2y",
    "us_10y",
    "us_curve_10y2y",
    "curve_10_2",
    "hike_proxy",
    "hike_proxy_chg",
    "fed_funds_daily",
    "fed_funds",
    "us_unemp",
    "unemp_chg",
    "us_cpi",
    "cpi_yoy",
    "vix",
    "vix_chg",
    "dxy_ret1",
    "usd_broad_ret1",
]


def expanding_z(s: pd.Series, min_periods: int = 60) -> pd.Series:
    m = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    return (s - m) / sd.replace(0, np.nan)


def rolling_z(s: pd.Series, win: int = 60) -> pd.Series:
    m = s.rolling(win, min_periods=max(20, win // 3)).mean()
    sd = s.rolling(win, min_periods=max(20, win // 3)).std()
    return (s - m) / sd.replace(0, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    ensure_dirs()
    path = RAW_DIR / "_macro_panel_daily.parquet"
    if not path.exists():
        raise SystemExit("missing macro panel; run fetch_macro.py first")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    out = pd.DataFrame(index=df.index)
    for col in CORE:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        out[col] = x
        out[f"{col}_z60"] = rolling_z(x, 60)
        out[f"{col}_z252"] = rolling_z(x, 252)
        out[f"{col}_ma5"] = x.rolling(5, min_periods=3).mean()
        out[f"{col}_ma20"] = x.rolling(20, min_periods=10).mean()
        if col in ("dxy", "usd_broad", "vix", "hike_proxy", "us_2y", "us_10y"):
            out[f"{col}_chg5"] = x.diff(5)
            out[f"{col}_chg20"] = x.diff(20)

    # signed composites (economic intuition)
    if "dxy_z60" in out.columns and "hike_proxy_z60" in out.columns:
        # strong USD + hawkish → often press gold
        out["usd_hawkish"] = out["dxy_z60"].fillna(0) + out["hike_proxy_z60"].fillna(0)
    if "vix_z60" in out.columns and "dxy_z60" in out.columns:
        out["risk_off"] = out["vix_z60"].fillna(0) + out["dxy_z60"].fillna(0)
    if "unemp_chg" in out.columns:
        out["labor_soft"] = expanding_z(out["unemp_chg"].fillna(0), 24)

    out = out.replace([np.inf, -np.inf], np.nan)
    fac_path = FACTOR_DIR / "macro_factors_daily.parquet"
    out.reset_index().to_parquet(fac_path, index=False)
    print(f"saved {fac_path} rows={len(out)} cols={len(out.columns)}", flush=True)


if __name__ == "__main__":
    main()
