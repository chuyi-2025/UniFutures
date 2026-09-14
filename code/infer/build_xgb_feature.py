#!/usr/bin/env python3
"""Infer XGB features: 64-day window only, no forward-return filter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

INFER_DIR = Path(__file__).resolve().parent
CODE_DIR = INFER_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(INFER_DIR))

import config
from build_feature.build_xgb_feature import FEATURE_NAMES, N_FEAT, add_features, prepare_df
from shared import HORIZON, WINDOW, ret_to_label


def iter_segments(raw: pd.DataFrame) -> list[pd.DataFrame]:
    out = prepare_df(raw)
    out["_seg"] = (out["code"].astype(str) != out["code"].astype(str).shift()).cumsum()
    return [g.reset_index(drop=True) for _, g in out.groupby("_seg") if len(g) >= WINDOW]


def valid_window(codes: np.ndarray, closes: np.ndarray, i: int) -> bool:
    block = codes[i - WINDOW + 1 : i + 1]
    if len(block) != WINDOW or len(np.unique(block)) != 1:
        return False
    entry = closes[i]
    return bool(np.isfinite(entry) and entry > 0)


def forward_ret_5d(codes: np.ndarray, closes: np.ndarray, dates, i: int) -> tuple[float, object]:
    j = i + HORIZON
    if j >= len(closes):
        return np.nan, pd.NaT
    if len(np.unique(codes[i : j + 1])) != 1:
        return np.nan, pd.NaT
    entry, exit_ = closes[i], closes[j]
    if not (np.isfinite(entry) and np.isfinite(exit_) and entry > 0):
        return np.nan, pd.NaT
    return float(exit_ / entry - 1.0), dates[j]


def build_contract_rows(symbol: str, csv_path: Path) -> list[dict]:
    contract_file = csv_path.stem
    rows: list[dict] = []
    for seg in iter_segments(pd.read_csv(csv_path, parse_dates=["date"])):
        feat = add_features(seg)
        codes = seg["code"].astype(str).values
        closes = seg["close"].astype(float).values
        dates = seg["date"].values
        vals = feat.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)

        for i in range(WINDOW - 1, len(seg)):
            if not valid_window(codes, closes, i):
                continue
            x = vals[i]
            if not np.isfinite(x).all():
                continue
            ret, exit_d = forward_ret_5d(codes, closes, dates, i)
            row = {
                "date": dates[i],
                "symbol": symbol,
                "code": codes[i],
                "contract_file": contract_file,
                "seg_idx": i,
                "label": ret_to_label(ret) if np.isfinite(ret) else -1,
                "ret_5d": ret,
                "exit_date": exit_d,
            }
            for j, v in enumerate(x):
                row[f"f_{j}"] = float(v)
            rows.append(row)
    return rows


def main() -> None:
    all_rows: list[dict] = []
    for symbol in config.SYMBOLS:
        sym_n = 0
        for csv_path in sorted((config.CONTRACTS_DIR / symbol).glob("*.csv")):
            rows = build_contract_rows(symbol, csv_path)
            sym_n += len(rows)
            all_rows.extend(rows)
        print(f"{symbol}: {sym_n} samples")

    df = pd.DataFrame(all_rows).sort_values(["symbol", "date", "code"]).reset_index(drop=True)
    config.XGB_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.XGB_FEATURES_CSV, index=False)
    print(f"saved {config.XGB_FEATURES_CSV} rows={len(df)} feat_dim={N_FEAT}")


if __name__ == "__main__":
    main()
