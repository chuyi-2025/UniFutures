#!/usr/bin/env python3
"""Infer GAF features: 64-day window only, no forward-return filter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

INFER_DIR = Path(__file__).resolve().parent
CODE_DIR = INFER_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(INFER_DIR))

import config
from shared import FEAT_COLS, GAF_HORIZON, WINDOW, ret_to_gaf_class

CLIP = 5.0
N_X = WINDOW * len(FEAT_COLS)
X_COLS = [f"x_{i}" for i in range(N_X)]
WD_COLS = [f"wd_{i}" for i in range(WINDOW)]


def normalize_window(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    x = (values - mean) / (std + 1e-5)
    return np.clip(x, -CLIP, CLIP).astype(np.float32)


def prepare_df(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.sort_values("date").reset_index(drop=True)
    out["amount"] = out["turnover"].astype(float)
    return out.dropna(subset=FEAT_COLS).reset_index(drop=True)


def iter_segments(raw: pd.DataFrame) -> list[pd.DataFrame]:
    out = prepare_df(raw)
    out["_seg"] = (out["code"].astype(str) != out["code"].astype(str).shift()).cumsum()
    return [g.reset_index(drop=True) for _, g in out.groupby("_seg") if len(g) >= WINDOW]


def valid_window(codes: np.ndarray, feats: np.ndarray, closes: np.ndarray, i: int) -> bool:
    block = codes[i - WINDOW + 1 : i + 1]
    if len(block) != WINDOW or len(np.unique(block)) != 1:
        return False
    if not np.isfinite(feats[i - WINDOW + 1 : i + 1]).all():
        return False
    return bool(np.isfinite(closes[i]) and closes[i] > 0)


def forward_ret_3d(codes: np.ndarray, closes: np.ndarray, i: int) -> tuple[float, np.datetime64]:
    j = i + GAF_HORIZON
    if j >= len(closes):
        return np.nan, np.datetime64("NaT")
    if len(np.unique(codes[i : j + 1])) != 1:
        return np.nan, np.datetime64("NaT")
    entry, exit_ = closes[i], closes[j]
    if not (np.isfinite(entry) and np.isfinite(exit_) and entry > 0):
        return np.nan, np.datetime64("NaT")
    return float(exit_ / entry - 1.0), np.datetime64("NaT")


def collect_contract(symbol: str, csv_path: Path) -> dict[str, np.ndarray]:
    contract_file = csv_path.stem
    date: list[np.datetime64] = []
    exit_date: list[np.datetime64] = []
    code: list[str] = []
    seg_idx: list[int] = []
    ret_3d: list[np.float32] = []
    label: list[np.int32] = []
    log_ret_1d: list[np.float32] = []
    x_rows: list[np.ndarray] = []
    wd_rows: list[np.ndarray] = []

    raw = pd.read_csv(csv_path, parse_dates=["date"])
    for seg in iter_segments(raw):
        codes = seg["code"].astype(str).values
        closes = seg["close"].astype(float).values
        feats = seg[FEAT_COLS].astype(float).values
        dates = seg["date"].values.astype("datetime64[ns]")

        for i in range(WINDOW - 1, len(seg)):
            if not valid_window(codes, feats, closes, i):
                continue
            win = feats[i - WINDOW + 1 : i + 1]
            normed = normalize_window(win.astype(np.float32, copy=False))
            ret, _ = forward_ret_3d(codes, closes, i)
            date.append(dates[i])
            code.append(codes[i])
            seg_idx.append(i)
            x_rows.append(normed.reshape(-1))
            wd_rows.append(dates[i - WINDOW + 1 : i + 1])
            if np.isfinite(ret):
                ret_3d.append(np.float32(ret))
                label.append(np.int32(ret_to_gaf_class(ret)))
                exit_date.append(dates[i + GAF_HORIZON])
            else:
                ret_3d.append(np.float32(np.nan))
                label.append(np.int32(-1))
                exit_date.append(np.datetime64("NaT"))
            if i + 1 < len(seg):
                log_ret_1d.append(np.float32(np.log(closes[i + 1] / closes[i])))
            else:
                log_ret_1d.append(np.float32(np.nan))

    return {
        "date": np.asarray(date, dtype="datetime64[ns]"),
        "exit_date": np.asarray(exit_date, dtype="datetime64[ns]"),
        "code": np.asarray(code, dtype=object),
        "contract_file": np.full(len(date), contract_file, dtype=object),
        "seg_idx": np.asarray(seg_idx, dtype=np.int32),
        "ret_3d": np.asarray(ret_3d, dtype=np.float32),
        "label": np.asarray(label, dtype=np.int32),
        "log_ret_1d": np.asarray(log_ret_1d, dtype=np.float32),
        "x": np.vstack(x_rows).astype(np.float32, copy=False) if x_rows else np.empty((0, N_X), dtype=np.float32),
        "wd": np.stack(wd_rows) if wd_rows else np.empty((0, WINDOW), dtype="datetime64[ns]"),
    }


def concat_chunks(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = ["date", "exit_date", "code", "contract_file", "seg_idx", "ret_3d", "label", "log_ret_1d", "x", "wd"]
    out: dict[str, np.ndarray] = {}
    for k in keys:
        if k in ("x", "wd"):
            out[k] = np.concatenate([c[k] for c in chunks], axis=0)
        else:
            out[k] = np.concatenate([c[k] for c in chunks])
    return out


def write_symbol_csv(path: Path, symbol: str, data: dict[str, np.ndarray]) -> None:
    n = len(data["date"])
    cols: dict[str, np.ndarray] = {
        "date": data["date"],
        "exit_date": data["exit_date"],
        "symbol": np.full(n, symbol, dtype=object),
        "code": data["code"],
        "contract_file": data["contract_file"],
        "seg_idx": data["seg_idx"],
        "ret_3d": data["ret_3d"],
        "label": data["label"],
        "log_ret_1d": data["log_ret_1d"],
    }
    for j in range(N_X):
        cols[X_COLS[j]] = data["x"][:, j]
    for t in range(WINDOW):
        cols[WD_COLS[t]] = data["wd"][:, t]
    pd.DataFrame(cols).to_csv(path, index=False)


def build_symbol(symbol: str) -> int:
    out_path = config.GAF_FEATURES_DIR / f"{symbol}.csv"
    if out_path.exists():
        out_path.unlink()
    chunks = [
        collect_contract(symbol, csv_path)
        for csv_path in sorted((config.CONTRACTS_DIR / symbol).glob("*.csv"))
    ]
    data = concat_chunks(chunks)
    write_symbol_csv(out_path, symbol, data)
    return len(data["date"])


def merge_symbols() -> None:
    out_dir = config.GAF_FEATURES_DIR
    merged = out_dir / "all.csv"
    if merged.exists():
        merged.unlink()
    parts = sorted(p for p in out_dir.glob("*.csv") if p.name != merged.name)
    n_rows = 0
    for i, path in enumerate(tqdm(parts, desc="merge gaf")):
        df = pd.read_csv(path, parse_dates=["date", "exit_date", *WD_COLS])
        for col in WD_COLS:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
        df.to_csv(merged, mode="w" if i == 0 else "a", header=i == 0, index=False)
        n_rows += len(df)
    print(f"merged gaf -> {merged} rows={n_rows}")


def main() -> None:
    config.GAF_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in tqdm(config.SYMBOLS, desc="gaf"):
        n = build_symbol(symbol)
        tqdm.write(f"{symbol}: {n} -> {config.GAF_FEATURES_DIR / f'{symbol}.csv'}")
    merge_symbols()


if __name__ == "__main__":
    main()
