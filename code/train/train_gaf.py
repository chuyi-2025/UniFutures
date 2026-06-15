#!/usr/bin/env python3
"""GAF-CNN MSE training on one symbol (Moses-style)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import (
    BACKTEST_END,
    BACKTEST_START,
    CONTRACTS_DIR,
    GAF_FEATURES_DIR,
    MODELS_DIR,
    TRAIN_END,
    WINDOW,
)

BATCH_SIZE = 32
EPOCHS = 10
CLIP = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_gaf_image(timeseries: np.ndarray) -> np.ndarray:
    min_val = float(np.min(timeseries))
    max_val = float(np.max(timeseries))
    diff = max_val - min_val
    if diff < 1e-6:
        diff = 1.0
    x_scaled = ((timeseries - min_val) / diff) * 2.0 - 1.0
    x_scaled = np.clip(x_scaled, -1.0, 1.0)
    cos_phi = x_scaled
    sin_phi = np.sqrt(1.0 - x_scaled**2)
    return np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)


def to_target(ret: float) -> float:
    return float(np.clip(ret, -CLIP, CLIP) / CLIP)


class GAF_CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.tanh(self.fc2(x))


class GafMseDataset(Dataset):
    def __init__(self, symbol: str, start=None, end=None) -> None:
        path = GAF_FEATURES_DIR / f"{symbol}.csv"
        df = pd.read_csv(path, parse_dates=["date"])
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        self.df = df.reset_index(drop=True)
        self.cache: dict[str, pd.DataFrame] = {}

    def _closes(self, row: pd.Series) -> np.ndarray:
        key = str(row["contract_file"])
        if key not in self.cache:
            p = CONTRACTS_DIR / row["symbol"] / f"{key}.csv"
            seg = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            self.cache[key] = seg
        seg = self.cache[key]
        i = int(row["seg_idx"])
        return seg.iloc[i - WINDOW + 1 : i + 1]["close"].astype(float).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        closes = self._closes(row)
        gaf = generate_gaf_image(closes)
        x = torch.tensor(gaf, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor([to_target(float(row["ret_3d"]))], dtype=torch.float32)
        return x, y


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, opt: optim.Optimizer | None) -> float:
    train = opt is not None
    model.train(train)
    total = 0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        if train:
            opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        if train:
            loss.backward()
            opt.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


def train_gaf(symbol: str) -> Path:
    train_ds = GafMseDataset(symbol, end=TRAIN_END)
    valid_ds = GafMseDataset(symbol, start=BACKTEST_START, end=BACKTEST_END)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = GAF_CNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"train {symbol}: train={len(train_ds)} valid={len(valid_ds)} device={DEVICE}")
    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, loss_fn, opt)
        valid_loss = run_epoch(model, valid_loader, loss_fn, None)
        print(f"Epoch {epoch + 1}/{EPOCHS}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"gaf_cnn_{symbol.lower()}.pt"
    torch.save(model.state_dict(), out)
    print(f"saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SN", help="symbol to train")
    args = parser.parse_args()
    train_gaf(args.symbol.upper())


if __name__ == "__main__":
    main()
