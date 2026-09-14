#!/usr/bin/env python3
"""Export past-month PPO daily signals for all configured symbols."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

INFER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INFER_DIR))

import config

POS_NAME = {0: "flat", 1: "long", -1: "short"}
COLS = [
    "model",
    "symbol",
    "name_cn",
    "date",
    "code",
    "position",
    "position_name",
    "is_open",
    "capital",
    "action",
    "action_name",
    "pred_ret",
    "signal",
    "pred_class",
]
OUT_NAME = "ppo_month_signals.csv"


def load_symbol_month(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = config.RESULTS_DIR / "ppo" / symbol.lower() / "daily.csv"
    if not path.exists():
        print(f"skip missing {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if df.empty:
        return df

    pos = df["position"].astype(int)
    out = pd.DataFrame(
        {
            "model": "ppo",
            "symbol": symbol,
            "name_cn": config.SYMBOL_CN.get(symbol, symbol),
            "date": df["date"].dt.strftime("%Y-%m-%d"),
            "code": df["code"],
            "position": pos,
            "position_name": pos.map(POS_NAME),
            "is_open": pos != 0,
            "capital": df["capital"].astype(float),
            "action": df["action"].astype(int),
            "action_name": df["action_name"],
            "pred_ret": pd.NA,
            "signal": pd.NA,
            "pred_class": pd.NA,
        }
    )
    return out[COLS]


def main() -> None:
    end = pd.Timestamp(config.INFER_END)
    start = end - timedelta(days=30)
    parts: list[pd.DataFrame] = []

    print(f"ppo month window: {start.date()} ~ {end.date()}")
    for sym in config.SYMBOLS:
        part = load_symbol_month(sym, start, end)
        if part.empty:
            print(f"  {sym}: 0 rows")
            continue
        print(f"  {sym} ({config.SYMBOL_CN[sym]}): {len(part)} rows")
        parts.append(part)

    if not parts:
        raise SystemExit("no PPO daily data found; run run_backtest.py first")

    out = pd.concat(parts, ignore_index=True)
    # keep configured symbol order, then date within each symbol
    out["symbol"] = pd.Categorical(out["symbol"], categories=config.SYMBOLS, ordered=True)
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    out["symbol"] = out["symbol"].astype(str)
    out_path = config.DAILY_PPO_DIR / OUT_NAME
    config.DAILY_PPO_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved -> {out_path} ({len(out)} rows, {out['symbol'].nunique()} symbols)")


if __name__ == "__main__":
    main()
