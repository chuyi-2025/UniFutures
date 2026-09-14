#!/usr/bin/env python3
"""Fetch historical daily weather for China crop-region points via Open-Meteo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW_DIR, ensure_dirs, load_regions, iter_stations  # noqa: E402

API = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_mean",
]
SLEEP_S = 0.35
MAX_RETRIES = 4


def _fetch_one(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Shanghai",
    }
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(API, params=params, timeout=90)
            if r.status_code in (429, 500, 502, 503):
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(3.0 * (attempt + 1))
                continue
            r.raise_for_status()
            payload = r.json()
            daily = payload.get("daily") or {}
            if not daily or "time" not in daily:
                raise RuntimeError(f"empty daily payload keys={list(payload.keys())} body={str(payload)[:200]}")
            df = pd.DataFrame(daily)
            df = df.rename(
                columns={
                    "time": "date",
                    "temperature_2m_max": "tmax",
                    "temperature_2m_min": "tmin",
                    "temperature_2m_mean": "tmean",
                    "precipitation_sum": "precip",
                    "relative_humidity_2m_mean": "rh",
                }
            )
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "tmin", "tmax", "tmean", "precip", "rh"]]
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"fetch failed lat={lat} lon={lon}: {last_err}")


def station_path(station_id: str) -> Path:
    return RAW_DIR / f"{station_id}.parquet"


def load_existing(station_id: str) -> pd.DataFrame:
    p = station_path(station_id)
    if not p.exists():
        return pd.DataFrame(columns=["date", "tmin", "tmax", "tmean", "precip", "rh"])
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date", keep="last")


def merge_save(station_id: str, new: pd.DataFrame) -> pd.DataFrame:
    old = load_existing(station_id)
    frames = [x for x in (old, new) if x is not None and len(x)]
    out = pd.concat(frames, ignore_index=True) if frames else new.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    out.to_parquet(station_path(station_id), index=False)
    return out


def resolve_range(cfg: dict, start: str | None, end: str | None, station_id: str) -> tuple[str, str]:
    end_ts = pd.Timestamp(end) if end else (pd.Timestamp.today().normalize() - pd.Timedelta(days=1))
    if start:
        start_ts = pd.Timestamp(start)
    else:
        existing = load_existing(station_id)
        if len(existing):
            start_ts = existing["date"].max() + pd.Timedelta(days=1)
        else:
            start_ts = pd.Timestamp(cfg.get("defaults", {}).get("start_date", "2015-01-01"))
    return str(start_ts.date()), str(end_ts.date())


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Open-Meteo daily weather for crop regions")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default: config or incremental)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--symbols", default="CF,SR,AP,CJ", help="comma-separated symbols")
    ap.add_argument("--force-full", action="store_true", help="ignore local cache, refetch from --start/config")
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_regions()
    want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}

    for sym, st in iter_stations(cfg):
        if sym not in want:
            continue
        sid = st["id"]
        if args.force_full:
            start = args.start or cfg.get("defaults", {}).get("start_date", "2015-01-01")
            end = args.end or str((pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).date())
        else:
            start, end = resolve_range(cfg, args.start, args.end, sid)
        if pd.Timestamp(start) > pd.Timestamp(end):
            print(f"[skip] {sid} already up to date through {end}", flush=True)
            continue
        print(f"[fetch] {sym} {sid} {st['name']} {start}->{end} lat={st['lat']} lon={st['lon']}", flush=True)
        df = _fetch_one(float(st["lat"]), float(st["lon"]), start, end)
        df["station_id"] = sid
        df["symbol"] = sym
        out = merge_save(sid, df)
        print(f"  saved {station_path(sid)} rows={len(out)} last={out['date'].max().date()}", flush=True)
        time.sleep(SLEEP_S)

    print("done", flush=True)


if __name__ == "__main__":
    main()
