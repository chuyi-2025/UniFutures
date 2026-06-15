#!/usr/bin/env python3
"""GAF-CNN MSE training on a futures category (multi-symbol)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import BACKTEST_END, BACKTEST_START, CONTRACTS_DIR, GAF_FEATURES_DIR, MODELS_DIR, TRAIN_END, WINDOW
from train_gaf import BATCH_SIZE, CLIP, DEVICE, EPOCHS, GAF_CNN, generate_gaf_image, run_epoch, to_target

# 有色金属
CATEGORIES: dict[str, list[str]] = {
    "nonferrous": ["CU", "AL", "NI", "SN", "PB", "ZN", "AO", "AD", "BC", "SI", "PS", "LC"],
}


class GafCategoryDataset(Dataset):
    def __init__(self, symbols: list[str], start=None, end=None) -> None:
        parts: list[pd.DataFrame] = []
        for sym in symbols:
            path = GAF_FEATURES_DIR / f"{sym}.csv"
            if not path.exists():
                print(f"skip missing {path}")
                continue
            df = pd.read_csv(path, parse_dates=["date"])
            if start is not None:
                df = df[df["date"] >= pd.Timestamp(start)]
            if end is not None:
                df = df[df["date"] <= pd.Timestamp(end)]
            if not df.empty:
                parts.append(df)
        if not parts:
            raise SystemExit("no data for category symbols")
        self.df = pd.concat(parts, ignore_index=True)
        self.cache: dict[tuple[str, str], pd.DataFrame] = {}

    def _closes(self, row: pd.Series):
        sym = str(row["symbol"])
        key = str(row["contract_file"])
        cache_key = (sym, key)
        if cache_key not in self.cache:
            p = CONTRACTS_DIR / sym / f"{key}.csv"
            seg = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            self.cache[cache_key] = seg
        seg = self.cache[cache_key]
        i = int(row["seg_idx"])
        return seg.iloc[i - WINDOW + 1 : i + 1]["close"].astype(float).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        gaf = generate_gaf_image(self._closes(row))
        x = torch.tensor(gaf, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor([to_target(float(row["ret_3d"]))], dtype=torch.float32)
        return x, y


def train_gaf_category(category: str) -> Path:
    symbols = CATEGORIES[category]
    train_ds = GafCategoryDataset(symbols, end=TRAIN_END)
    valid_ds = GafCategoryDataset(symbols, start=BACKTEST_START, end=BACKTEST_END)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = GAF_CNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"train {category}: symbols={symbols}")
    print(f"train={len(train_ds)} valid={len(valid_ds)} device={DEVICE}")
    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, loss_fn, opt)
        valid_loss = run_epoch(model, valid_loader, loss_fn, None)
        print(f"Epoch {epoch + 1}/{EPOCHS}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"gaf_cnn_{category}.pt"
    torch.save(model.state_dict(), out)
    print(f"saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        default="nonferrous",
        choices=sorted(CATEGORIES),
        help="futures category to train",
    )
    args = parser.parse_args()
    train_gaf_category(args.category)


if __name__ == "__main__":
    main()
