#!/usr/bin/env python3
"""Build XGB features: compact engineered indicators, all symbols → one CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import CONTRACTS_DIR, FEAT_COLS, HORIZON, REMOVED_SYMBOLS, WINDOW, XGB_FEATURES_CSV, ret_to_label

SPAN = WINDOW + HORIZON

# ~75 scale-free / ratio features; skip 60/720 range ratios (≈1 within 64-day window)
FEATURE_NAMES: tuple[str, ...] = (
    "ret_1d", "momentum_5", "momentum_10", "momentum_20", "momentum_30",
    "ma_ratio_5_20", "ma_ratio_10_30", "ma_ratio_10_60", "ma_ratio_20_60", "ma_ratio_30_60",
    "ma_diff_10_30", "ma_diff_10_60", "ma_diff_30_60",
    "ratio_max_7", "ratio_min_7", "ratio_max_30", "ratio_min_30",
    "ratio_maxmin_7", "ratio_maxmin_30", "range_pct_7", "range_pct_30",
    "pos_64", "vol_pos_64", "ret_mean_5", "ret_std_5", "ret_mean_20", "ret_std_20",
    "volume_ret", "volume_ratio_30", "volume_ma_ratio_10_30", "volume_ma_ratio_30_60",
    "price_volume_ratio", "turnover_rate", "liquidity_10", "liquidity_30",
    "volatility_10", "volatility_30", "volatility_60",
    "rsi", "skew_7", "skew_30", "kurt_7", "kurt_30",
    "bias_5", "bias_10", "bias_30",
    "avg_price_break_5", "avg_price_break_10", "avg_price_break_30",
    "money_flow_ratio_10", "money_flow_ratio_30",
    "body_ratio", "upper_shadow", "lower_shadow", "hl_spread", "gap",
    "month", "weekday", "day",
)
N_FEAT = len(FEATURE_NAMES)


def prepare_df(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.sort_values("date").reset_index(drop=True)
    out["amount"] = out["turnover"].astype(float)
    return out.dropna(subset=FEAT_COLS).reset_index(drop=True)


def iter_segments(raw: pd.DataFrame, *, allow_tail: bool = False) -> list[pd.DataFrame]:
    out = prepare_df(raw)
    out["_seg"] = (out["code"].astype(str) != out["code"].astype(str).shift()).cumsum()
    min_len = WINDOW if allow_tail else SPAN
    return [g.reset_index(drop=True) for _, g in out.groupby("_seg") if len(g) >= min_len]


def add_features(seg: pd.DataFrame) -> pd.DataFrame:
    df = seg.copy()
    c, v = df["close"], df["volume"]
    o, h, l = df["open"], df["high"], df["low"]
    oi = df["open_interest"] if "open_interest" in df.columns else pd.Series(np.nan, index=df.index)

    df["ret_1d"] = c.pct_change()
    for n in (5, 10, 20, 30):
        df[f"momentum_{n}"] = c / c.shift(n) - 1

    ma5, ma10, ma20, ma30, ma60 = (c.rolling(n).mean() for n in (5, 10, 20, 30, 60))
    df["ma_ratio_5_20"] = ma5 / ma20
    df["ma_ratio_10_30"] = ma10 / ma30
    df["ma_ratio_10_60"] = ma10 / ma60
    df["ma_ratio_20_60"] = ma20 / ma60
    df["ma_ratio_30_60"] = ma30 / ma60
    df["ma_diff_10_30"] = (ma10 - ma30) / c
    df["ma_diff_10_60"] = (ma10 - ma60) / c
    df["ma_diff_30_60"] = (ma30 - ma60) / c

    for w in (7, 30):
        mx, mn = c.rolling(w).max(), c.rolling(w).min()
        df[f"ratio_max_{w}"] = c / mx
        df[f"ratio_min_{w}"] = c / mn
        df[f"ratio_maxmin_{w}"] = mx / mn
        df[f"range_pct_{w}"] = (mx - mn) / c

    wmax, wmin = c.rolling(WINDOW).max(), c.rolling(WINDOW).min()
    df["pos_64"] = (c - wmin) / (wmax - wmin + 1e-5)
    vmax, vmin = v.rolling(WINDOW).max(), v.rolling(WINDOW).min()
    df["vol_pos_64"] = (v - vmin) / (vmax - vmin + 1e-5)
    df["ret_mean_5"] = df["ret_1d"].rolling(5).mean()
    df["ret_std_5"] = df["ret_1d"].rolling(5).std()
    df["ret_mean_20"] = df["ret_1d"].rolling(20).mean()
    df["ret_std_20"] = df["ret_1d"].rolling(20).std()

    df["volume_ret"] = v.pct_change()
    vma10, vma30, vma60 = (v.replace(0, np.nan).rolling(n).mean() for n in (10, 30, 60))
    df["volume_ratio_30"] = v / vma30
    df["volume_ma_ratio_10_30"] = vma10 / vma30
    df["volume_ma_ratio_30_60"] = vma30 / vma60
    df["price_volume_ratio"] = c / v.replace(0, np.nan)
    tr = v / oi.replace(0, np.nan)
    df["turnover_rate"] = tr
    df["liquidity_10"] = tr.rolling(10).mean()
    df["liquidity_30"] = tr.rolling(30).mean()

    for w in (10, 30, 60):
        df[f"volatility_{w}"] = c.pct_change().rolling(w).std()

    delta = c.diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean() + 1e-8)
    df["rsi"] = 100 - 100 / (1 + rs)
    for w in (7, 30):
        df[f"skew_{w}"] = c.rolling(w).skew()
        df[f"kurt_{w}"] = c.rolling(w).kurt()
    for w, ma in ((5, ma5), (10, ma10), (30, ma30)):
        df[f"bias_{w}"] = (c - ma) / ma

    avg5 = (o + h + l + c) / 4
    for w in (5, 10, 30):
        df[f"avg_price_break_{w}"] = c / avg5.rolling(w).mean()

    mf = v * c
    df["money_flow_ratio_10"] = mf / mf.rolling(10).mean()
    df["money_flow_ratio_30"] = mf / mf.rolling(30).mean()

    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    df["body_ratio"] = body / rng
    df["upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng
    df["lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng
    df["hl_spread"] = rng / c
    df["gap"] = o / c.shift(1) - 1

    dt = pd.to_datetime(df["date"])
    df["month"] = dt.dt.month
    df["weekday"] = dt.dt.weekday
    df["day"] = dt.dt.day

    arr = df.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64, copy=True)
    arr[np.isinf(arr)] = np.nan
    df.loc[:, FEATURE_NAMES] = arr
    return df


def valid_sample(codes: np.ndarray, closes: np.ndarray, i: int) -> bool:
    block = codes[i - WINDOW + 1 : i + HORIZON + 1]
    if len(block) != SPAN or len(np.unique(block)) != 1:
        return False
    entry, exit_ = closes[i], closes[i + HORIZON]
    return np.isfinite(entry) and np.isfinite(exit_) and entry > 0


def valid_window(codes: np.ndarray, closes: np.ndarray, i: int) -> bool:
    block = codes[i - WINDOW + 1 : i + 1]
    if len(block) != WINDOW or len(np.unique(block)) != 1:
        return False
    entry = closes[i]
    return bool(np.isfinite(entry) and entry > 0)


def build_contract_rows(
    symbol: str,
    csv_path: Path,
    contracts_dir: Path = CONTRACTS_DIR,
    *,
    allow_tail: bool = False,
) -> list[dict]:
    contract_file = csv_path.stem
    rows: list[dict] = []
    for seg in iter_segments(pd.read_csv(csv_path, parse_dates=["date"]), allow_tail=allow_tail):
        feat = add_features(seg)
        codes = seg["code"].astype(str).values
        closes = seg["close"].astype(float).values
        dates = seg["date"].values
        vals = feat.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
        tail_start = len(seg) - HORIZON

        for i in range(WINDOW - 1, tail_start):
            if not valid_sample(codes, closes, i):
                continue
            x = vals[i]
            if not np.isfinite(x).all():
                continue
            ret = closes[i + HORIZON] / closes[i] - 1.0
            row = {
                "date": dates[i],
                "symbol": symbol,
                "code": codes[i],
                "contract_file": contract_file,
                "seg_idx": i,
                "label": ret_to_label(ret),
                "ret_5d": ret,
                "exit_date": dates[i + HORIZON],
            }
            for j, v in enumerate(x):
                row[f"f_{j}"] = float(v)
            rows.append(row)

        if allow_tail:
            for i in range(max(WINDOW - 1, tail_start), len(seg)):
                if not valid_window(codes, closes, i):
                    continue
                x = vals[i]
                if not np.isfinite(x).all():
                    continue
                row = {
                    "date": dates[i],
                    "symbol": symbol,
                    "code": codes[i],
                    "contract_file": contract_file,
                    "seg_idx": i,
                    "label": -1,
                    "ret_5d": np.nan,
                    "exit_date": pd.NaT,
                }
                for j, v in enumerate(x):
                    row[f"f_{j}"] = float(v)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=XGB_FEATURES_CSV)
    parser.add_argument("--symbols", nargs="*", default=None, help="symbols to build (default: all)")
    parser.add_argument("--contracts-dir", type=Path, default=CONTRACTS_DIR)
    parser.add_argument(
        "--allow-tail",
        action="store_true",
        help="emit features for latest days without full forward horizon (infer)",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    symbols = args.symbols or sorted(p.name for p in args.contracts_dir.iterdir() if p.is_dir())
    for symbol in symbols:
        if symbol in REMOVED_SYMBOLS:
            print(f"{symbol}: skip (removed)")
            continue
        sym_n = 0
        for csv_path in sorted((args.contracts_dir / symbol).glob("*.csv")):
            rows = build_contract_rows(
                symbol, csv_path, args.contracts_dir, allow_tail=args.allow_tail
            )
            sym_n += len(rows)
            all_rows.extend(rows)
        print(f"{symbol}: {sym_n} samples")

    df = pd.DataFrame(all_rows).sort_values(["symbol", "date", "code"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"saved {args.out} rows={len(df)} feat_dim={N_FEAT} symbols={df['symbol'].nunique()}")


if __name__ == "__main__":
    main()
