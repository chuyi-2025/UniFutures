#!/usr/bin/env python3
"""Fetch world macro series: FRED + Yahoo (DXY, yields, etc.)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW_DIR, ensure_dirs, load_sources  # noqa: E402

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
UA = {"User-Agent": "Mozilla/5.0 (compatible; UniFuturesMacro/1.0)"}


def fetch_fred(series_id: str, start: str) -> pd.DataFrame:
    r = requests.get(
        FRED_CSV,
        params={"id": series_id, "cosd": start},
        timeout=120,
        headers=UA,
    )
    r.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(r.text))
    if df.shape[1] < 2:
        raise RuntimeError(f"bad FRED csv for {series_id}: {df.head()}")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    # FRED uses "." for missing
    return df


def fetch_yahoo(symbol: str, start: str) -> pd.DataFrame:
    # period1/period2 unix
    t0 = int(pd.Timestamp(start).timestamp())
    t1 = int(pd.Timestamp.today().timestamp())
    url = YAHOO_CHART.format(symbol=symbol)
    r = requests.get(
        url,
        params={"period1": t0, "period2": t1, "interval": "1d", "events": "history"},
        timeout=60,
        headers=UA,
    )
    r.raise_for_status()
    payload = r.json()
    res = (payload.get("chart") or {}).get("result") or [None]
    res = res[0]
    if not res:
        raise RuntimeError(f"empty yahoo for {symbol}: {payload}")
    ts = res.get("timestamp") or []
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    close = quote.get("close") or []
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s"), "value": close})
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"]).sort_values("date")


def save_series(name: str, df: pd.DataFrame, source: str, sid: str) -> Path:
    out = df.copy()
    out["name"] = name
    out["source"] = source
    out["series_id"] = sid
    path = RAW_DIR / f"{name}.parquet"
    out.to_parquet(path, index=False)
    return path


def build_macro_panel(start: str) -> pd.DataFrame:
    """Wide daily panel: ffill monthly onto calendar business days."""
    frames = []
    for p in sorted(RAW_DIR.glob("*.parquet")):
        if p.name.startswith("_"):
            continue
        df = pd.read_parquet(p)
        if "value" not in df.columns:
            continue
        name = p.stem
        s = df.set_index(pd.to_datetime(df["date"]))["value"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        frames.append(s.rename(name))
    if not frames:
        raise RuntimeError("no raw macro series")
    wide = pd.concat(frames, axis=1).sort_index()
    wide = wide[wide.index >= pd.Timestamp(start)]
    # daily ffill for mixed freq (unemp/cpi monthly)
    wide = wide.ffill()
    # derived
    if "us_2y" in wide.columns and "fed_funds_daily" in wide.columns:
        wide["hike_proxy"] = wide["us_2y"] - wide["fed_funds_daily"]
    elif "us_2y" in wide.columns and "fed_funds" in wide.columns:
        wide["hike_proxy"] = wide["us_2y"] - wide["fed_funds"]
    if "us_10y" in wide.columns and "us_2y" in wide.columns:
        wide["curve_10_2"] = wide["us_10y"] - wide["us_2y"]
    if "dxy" in wide.columns:
        wide["dxy_ret1"] = np.log(wide["dxy"]).diff()
    if "usd_broad" in wide.columns:
        wide["usd_broad_ret1"] = np.log(wide["usd_broad"]).diff()
    if "vix" in wide.columns:
        wide["vix_chg"] = wide["vix"].diff()
    if "us_unemp" in wide.columns:
        wide["unemp_chg"] = wide["us_unemp"].diff()
    if "us_cpi" in wide.columns:
        wide["cpi_yoy"] = wide["us_cpi"].pct_change(12)
    if "hike_proxy" in wide.columns:
        wide["hike_proxy_chg"] = wide["hike_proxy"].diff()
    path = RAW_DIR / "_macro_panel_daily.parquet"
    wide.reset_index().rename(columns={"index": "date"}).to_parquet(path, index=False)
    # also save with date col name fix
    out = wide.copy()
    out.index.name = "date"
    out.reset_index().to_parquet(path, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_sources()
    start = args.start or cfg.get("defaults", {}).get("start_date", "2015-01-01")

    for spec in cfg.get("fred_series", []):
        sid, name = spec["id"], spec["name"]
        print(f"[fred] {sid} -> {name}", flush=True)
        try:
            df = fetch_fred(sid, start)
            path = save_series(name, df, "fred", sid)
            print(f"  saved {path} rows={len(df)} last={df['date'].max().date()}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {sid}: {exc}", flush=True)
        time.sleep(0.4)

    for spec in cfg.get("yahoo_tickers", []):
        sym, name = spec["symbol"], spec["name"]
        print(f"[yahoo] {sym} -> {name}", flush=True)
        try:
            df = fetch_yahoo(sym, start)
            path = save_series(name, df, "yahoo", sym)
            print(f"  saved {path} rows={len(df)} last={df['date'].max().date()}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {sym}: {exc}", flush=True)
        time.sleep(0.4)

    print("[panel] building daily macro panel...", flush=True)
    panel = build_macro_panel(start)
    print(f"  panel cols={list(panel.columns)} rows={len(panel)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
