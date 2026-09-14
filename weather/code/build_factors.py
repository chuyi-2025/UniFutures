#!/usr/bin/env python3
"""Build production-region weighted weather anomaly factors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FACTOR_DIR, RAW_DIR, ensure_dirs, load_regions  # noqa: E402


def load_station(station_id: str) -> pd.DataFrame:
    p = RAW_DIR / f"{station_id}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing raw weather: {p} (run fetch_open_meteo.py first)")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def region_panel(symbol: str, meta: dict, gdd_base: float) -> pd.DataFrame:
    parts = []
    weights = []
    for st in meta["stations"]:
        df = load_station(st["id"]).copy()
        w = float(st.get("weight", 1.0))
        for col in ("tmin", "tmax", "tmean", "precip", "rh"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("date")[["tmin", "tmax", "tmean", "precip", "rh"]]
        parts.append(df * w)
        weights.append(w)
    wsum = float(sum(weights))
    stacked = parts[0]
    for p in parts[1:]:
        stacked = stacked.add(p, fill_value=0.0)
    out = stacked / wsum
    out = out.reset_index()
    out["gdd"] = np.clip(out["tmean"] - gdd_base, 0.0, None)
    out["frost"] = (out["tmin"] <= 0.0).astype(float)
    # consecutive dry days (precip < 1mm)
    dry = (out["precip"].fillna(0.0) < 1.0).astype(int)
    spell = dry.copy()
    for i in range(1, len(spell)):
        spell.iloc[i] = spell.iloc[i - 1] + 1 if dry.iloc[i] else 0
    out["drought_spell"] = spell.astype(float)
    out["symbol"] = symbol
    out["doy"] = out["date"].dt.dayofyear
    out["month"] = out["date"].dt.month
    return out


def add_climatology_z(df: pd.DataFrame, cols: list[str], min_years: int = 5) -> pd.DataFrame:
    """Z-score vs same calendar day-of-year history (expanding, leave one year out via shift)."""
    out = df.sort_values("date").copy()
    out["year"] = out["date"].dt.year
    for col in cols:
        # expanding mean/std by doy using prior years only
        mu = []
        sd = []
        hist: dict[int, list[float]] = {}
        for doy, val, year in zip(out["doy"], out[col], out["year"], strict=True):
            series = hist.setdefault(int(doy), [])
            if len(series) >= min_years:
                arr = np.asarray(series, dtype=float)
                m = float(np.nanmean(arr))
                s = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else np.nan
            else:
                m, s = np.nan, np.nan
            mu.append(m)
            sd.append(s if s and s > 1e-8 else np.nan)
            if np.isfinite(val):
                series.append(float(val))
        out[f"{col}_clim"] = mu
        out[f"{col}_z"] = (out[col] - out[f"{col}_clim"]) / pd.Series(sd).replace(0, np.nan)
    return out


def add_window_flags(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    out = df.copy()
    pheno = set(int(x) for x in meta.get("phenology_months", []))
    frost_m = set(int(x) for x in meta.get("frost_months", []))
    out["in_pheno"] = out["month"].isin(pheno).astype(int)
    out["in_frost_window"] = out["month"].isin(frost_m).astype(int)
    out["frost_event"] = ((out["frost"] > 0) & (out["in_frost_window"] == 1)).astype(int)
    # stress: hot+dry in season
    out["heat_dry"] = (
        (out["in_pheno"] == 1)
        & (out.get("tmean_z", 0).fillna(0) >= 1.0)
        & (out.get("precip_z", 0).fillna(0) <= -0.5)
    ).astype(int)
    out["drought_stress"] = (
        (out["in_pheno"] == 1) & (out["drought_spell"] >= 10) & (out.get("precip_z", 0).fillna(0) <= -0.5)
    ).astype(int)
    return out


def build_symbol(symbol: str, meta: dict, gdd_base: float) -> pd.DataFrame:
    panel = region_panel(symbol, meta, gdd_base)
    panel = add_climatology_z(panel, ["tmean", "precip", "tmax", "tmin", "rh", "gdd"])
    panel = add_window_flags(panel, meta)
    keep = [
        "date", "symbol", "tmin", "tmax", "tmean", "precip", "rh", "gdd",
        "drought_spell", "frost", "in_pheno", "in_frost_window", "frost_event",
        "heat_dry", "drought_stress",
        "tmean_z", "precip_z", "tmax_z", "tmin_z", "rh_z", "gdd_z",
    ]
    return panel[keep].sort_values("date").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="CF,SR,AP,CJ")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_regions()
    gdd_base = float(cfg.get("defaults", {}).get("gdd_base_c", 10.0))
    want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}

    frames = []
    for sym, meta in cfg["symbols"].items():
        if sym not in want:
            continue
        print(f"[factor] {sym} {meta.get('name_cn')}", flush=True)
        fac = build_symbol(sym, meta, gdd_base)
        path = FACTOR_DIR / f"{sym}_daily.parquet"
        fac.to_parquet(path, index=False)
        print(f"  saved {path} rows={len(fac)} {fac['date'].min().date()}->{fac['date'].max().date()}", flush=True)
        frames.append(fac)

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_path = FACTOR_DIR / "all_symbols_daily.parquet"
        all_df.to_parquet(all_path, index=False)
        print(f"saved {all_path} rows={len(all_df)}", flush=True)


if __name__ == "__main__":
    main()
