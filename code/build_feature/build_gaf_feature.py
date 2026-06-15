#!/usr/bin/env python3
"""Build GAF features: numpy column arrays per symbol CSV, then merge."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import (
    CONTRACTS_DIR,
    FEAT_COLS,
    GAF_FEATURES_CSV,
    GAF_FEATURES_DIR,
    GAF_HORIZON,
    WINDOW,
    ret_to_gaf_class,
)

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
    return [g.reset_index(drop=True) for _, g in out.groupby("_seg") if len(g) >= WINDOW + GAF_HORIZON]


def valid_sample(codes: np.ndarray, feats: np.ndarray, closes: np.ndarray, i: int) -> bool:
    block = codes[i - WINDOW + 1 : i + GAF_HORIZON + 1]
    if len(np.unique(block)) != 1:
        return False
    if not np.isfinite(feats[i - WINDOW + 1 : i + 1]).all():
        return False
    entry_close = closes[i]
    exit_close = closes[i + GAF_HORIZON]
    return np.isfinite(entry_close) and np.isfinite(exit_close) and entry_close > 0


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

    for seg in iter_segments(pd.read_csv(csv_path, parse_dates=["date"])):
        codes = seg["code"].astype(str).values
        closes = seg["close"].astype(float).values
        feats = seg[FEAT_COLS].astype(float).values
        dates = seg["date"].values.astype("datetime64[ns]")
        for i in range(WINDOW - 1, len(seg) - GAF_HORIZON):
            if not valid_sample(codes, feats, closes, i):
                continue
            win = feats[i - WINDOW + 1 : i + 1]
            if win.shape != (WINDOW, len(FEAT_COLS)):
                continue
            normed = normalize_window(win.astype(np.float32, copy=False))
            ret = closes[i + GAF_HORIZON] / closes[i] - 1.0
            date.append(dates[i])
            exit_date.append(dates[i + GAF_HORIZON])
            code.append(codes[i])
            seg_idx.append(i)
            ret_3d.append(np.float32(ret))
            label.append(np.int32(ret_to_gaf_class(ret)))
            log_ret_1d.append(np.float32(np.log(closes[i + 1] / closes[i])))
            x_rows.append(normed.reshape(-1))
            wd_rows.append(dates[i - WINDOW + 1 : i + 1])

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
    return {
        "date": np.concatenate([c["date"] for c in chunks]),
        "exit_date": np.concatenate([c["exit_date"] for c in chunks]),
        "code": np.concatenate([c["code"] for c in chunks]),
        "contract_file": np.concatenate([c["contract_file"] for c in chunks]),
        "seg_idx": np.concatenate([c["seg_idx"] for c in chunks]),
        "ret_3d": np.concatenate([c["ret_3d"] for c in chunks]),
        "label": np.concatenate([c["label"] for c in chunks]),
        "log_ret_1d": np.concatenate([c["log_ret_1d"] for c in chunks]),
        "x": np.concatenate([c["x"] for c in chunks], axis=0),
        "wd": np.concatenate([c["wd"] for c in chunks], axis=0),
    }


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
    x = data["x"]
    wd = data["wd"]
    for j in range(N_X):
        cols[X_COLS[j]] = x[:, j]
    for t in range(WINDOW):
        cols[WD_COLS[t]] = wd[:, t]
    pd.DataFrame(cols).to_csv(path, index=False)


def build_symbol(symbol: str) -> int:
    out_path = GAF_FEATURES_DIR / f"{symbol}.csv"
    if out_path.exists():
        out_path.unlink()

    chunks: list[dict[str, np.ndarray]] = []
    for csv_path in sorted((CONTRACTS_DIR / symbol).glob("*.csv")):
        chunks.append(collect_contract(symbol, csv_path))

    data = concat_chunks(chunks)
    write_symbol_csv(out_path, symbol, data)
    return len(data["date"])


def merge_symbols() -> None:
    skip = {GAF_FEATURES_CSV.name, "class_counts_by_symbol.csv"}
    parts = sorted(p for p in GAF_FEATURES_DIR.glob("*.csv") if p.name not in skip)

    n_rows = 0
    for i, path in enumerate(tqdm(parts, desc="merge")):
        df = pd.read_csv(path, parse_dates=["date", "exit_date", *WD_COLS])
        for col in WD_COLS:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
        df.to_csv(GAF_FEATURES_CSV, mode="w" if i == 0 else "a", header=i == 0, index=False)
        n_rows += len(df)

    print(f"merged {len(parts)} symbols -> {GAF_FEATURES_CSV} rows={n_rows}")


def summarize_class_counts() -> pd.DataFrame:
    out_csv = GAF_FEATURES_DIR / "class_counts_by_symbol.csv"
    df = pd.read_csv(GAF_FEATURES_CSV, usecols=["symbol", "label"])
    rows: list[dict[str, int | str]] = []
    for symbol, grp in df.groupby("symbol", sort=True):
        vc = grp["label"].value_counts().sort_index()
        row: dict[str, int | str] = {"symbol": symbol, "total": len(grp)}
        for c in range(6):
            row[f"c{c}"] = int(vc.get(c, 0))
        rows.append(row)
    counts = pd.DataFrame(rows)
    counts.to_csv(out_csv, index=False)

    total = int(counts["total"].sum())
    print(f"class counts saved -> {out_csv} total={total}")
    for c in range(6):
        n = int(counts[f"c{c}"].sum())
        print(f"  class {c}: {n:,} ({100 * n / total:.2f}%)")
    return counts


def verify_symbol(symbol: str) -> None:
    path = GAF_FEATURES_DIR / f"{symbol}.csv"
    df = pd.read_csv(path, parse_dates=["date", "exit_date", *WD_COLS])
    x = df[X_COLS].to_numpy(dtype=np.float32)
    print(
        f"{symbol}: rows={len(df)} x_shape={x.shape} "
        f"x_dtype={x.dtype} wd_ok={df[WD_COLS].notna().all().all()} "
        f"label_counts={df['label'].value_counts().sort_index().to_dict()}"
    )


def main() -> None:
    GAF_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    if GAF_FEATURES_CSV.exists():
        GAF_FEATURES_CSV.unlink()
    symbols = sorted(p.name for p in CONTRACTS_DIR.iterdir() if p.is_dir())
    for symbol in tqdm(symbols, desc="build"):
        n = build_symbol(symbol)
        tqdm.write(f"{symbol}: saved {n} -> {GAF_FEATURES_DIR / f'{symbol}.csv'}")
    merge_symbols()
    summarize_class_counts()


if __name__ == "__main__":
    main()
