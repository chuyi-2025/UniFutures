#!/usr/bin/env python3
"""Build sentiment samples from cleaned OCR markdown + next-day main-contract returns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CONTRACTS_DIR,
    DATA_DIR,
    OCR_ROOT,
    REMOVED_SYMBOLS,
    extract_body_text,
    load_symbol_ohlc,
    parse_ocr_header,
    ret_to_label,
)
from symbol_map import map_varieties


def list_available_symbols() -> set[str]:
    return {
        p.name.upper()
        for p in CONTRACTS_DIR.iterdir()
        if p.is_dir() and p.name.upper() not in REMOVED_SYMBOLS
    }


def build_return_cache(symbols: set[str]) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    for sym in tqdm(sorted(symbols), desc="load contracts"):
        df = load_symbol_ohlc(sym)
        if df.empty:
            continue
        df = df.set_index("date").sort_index()
        cache[sym] = df
    return cache


def next_day_ret(df: pd.DataFrame, report_date: pd.Timestamp) -> tuple[pd.Timestamp | None, float]:
    """Return (trade_date, log_ret_1d on trade_date) where trade_date is first session > report_date."""
    if df.empty:
        return None, float("nan")
    rd = pd.Timestamp(report_date)
    if rd.tzinfo is not None:
        rd = rd.tz_localize(None)
    rd = rd.normalize()
    idx = df.index
    future = idx[idx > rd]
    if len(future) == 0:
        return None, float("nan")
    trade_date = future[0]
    ret = float(df.loc[trade_date, "log_ret_1d"])
    return pd.Timestamp(trade_date), ret


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ocr-root", type=Path, default=OCR_ROOT)
    p.add_argument("--out", type=Path, default=DATA_DIR / "samples.parquet")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    available = list_available_symbols()
    files = sorted(args.ocr_root.rglob("*.md"))
    if args.limit > 0:
        files = files[: args.limit]

    # First pass: collect needed symbols from headers
    needed: set[str] = set()
    parsed_rows: list[dict] = []
    skip_parse = skip_map = 0
    for path in tqdm(files, desc="parse ocr"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        hdr = parse_ocr_header(text)
        if hdr is None:
            skip_parse += 1
            continue
        inst, dt, variety = hdr
        if dt is None:
            # fallback: folder YYYYMMDD
            try:
                dt = pd.to_datetime(path.parent.name, format="%Y%m%d")
            except Exception:
                skip_parse += 1
                continue
        syms = map_varieties(variety)
        syms = [s for s in syms if s in available]
        if not syms:
            skip_map += 1
            continue
        body = extract_body_text(text)
        if len(body) < 20:
            skip_parse += 1
            continue
        for sym in syms:
            needed.add(sym)
            parsed_rows.append(
                {
                    "path": str(path),
                    "institution": inst,
                    "report_date": pd.Timestamp(dt).normalize(),
                    "variety_raw": variety,
                    "symbol": sym,
                    "text": body,
                }
            )

    print(f"[parse] rows={len(parsed_rows)} skip_parse={skip_parse} skip_map={skip_map} symbols={len(needed)}")
    cache = build_return_cache(needed)

    out_rows: list[dict] = []
    skip_ret = 0
    for row in tqdm(parsed_rows, desc="align returns"):
        sym = row["symbol"]
        df = cache.get(sym)
        if df is None:
            skip_ret += 1
            continue
        trade_date, ret = next_day_ret(df, row["report_date"])
        if trade_date is None or not pd.notna(ret):
            skip_ret += 1
            continue
        label = ret_to_label(ret)
        out_rows.append(
            {
                **row,
                "trade_date": trade_date.normalize(),
                "log_ret_1d": ret,
                "label": label,
            }
        )

    out = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".csv":
        out.to_csv(args.out, index=False)
    else:
        out.to_parquet(args.out, index=False)
    # also csv for convenience
    csv_path = args.out.with_suffix(".csv")
    out.drop(columns=["text"]).to_csv(csv_path, index=False)
    print(f"[done] samples={len(out)} skip_ret={skip_ret} -> {args.out}")
    if len(out):
        print(out.groupby("label").size().to_dict())
        print("date range", out["report_date"].min(), "->", out["report_date"].max())


if __name__ == "__main__":
    main()
