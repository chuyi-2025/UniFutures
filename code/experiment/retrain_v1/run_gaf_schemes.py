#!/usr/bin/env python3
"""Train and backtest G-family GAF retrain schemes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (  # noqa: E402
    BACKTEST_END,
    CONTRACTS_DIR,
    GAF_FEATURES_DIR,
    TRAIN_END,
    WINDOW,
    ret_to_gaf_class,
)
from train_gaf import GAF_CNN, generate_gaf_image  # noqa: E402
from schemes import (  # noqa: E402
    CATEGORY_SYMBOLS,
    EXP_ROOT,
    RESULT_ROOT,
    SEED,
    Scheme,
    by_id,
    schemes_by_family,
)

OOS_START = pd.Timestamp("2025-07-01")
OOS_END = pd.Timestamp("2026-12-31")
INIT_CAP = 1_000_000.0
BATCH_SIZE = 32
EPOCHS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TH = 0.001


class GAF_CNN_CLS(nn.Module):
    def __init__(self, n_classes: int = 6) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class GafSchemeDataset(Dataset):
    def __init__(self, symbols: list[str], horizon: int, clip: float, start=None, end=None, classification: bool = False):
        parts = []
        for sym in symbols:
            path = GAF_FEATURES_DIR / f"{sym}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, parse_dates=["date"])
            if start is not None:
                df = df[df["date"] >= pd.Timestamp(start)]
            if end is not None:
                df = df[df["date"] <= pd.Timestamp(end)]
            if not df.empty:
                parts.append(df)
        if not parts:
            raise SystemExit(f"no data for {symbols}")
        self.df = pd.concat(parts, ignore_index=True)
        self.horizon = horizon
        self.clip = clip
        self.classification = classification
        self.cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._attach_targets()

    def _closes_df(self, sym: str, key: str) -> pd.DataFrame:
        ck = (sym, key)
        if ck not in self.cache:
            p = CONTRACTS_DIR / sym / f"{key}.csv"
            self.cache[ck] = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        return self.cache[ck]

    def _attach_targets(self) -> None:
        self.df = self.df.copy()
        if self.horizon == 3 and "ret_3d" in self.df.columns:
            self.df["_ret"] = self.df["ret_3d"].astype(float)
            self.df = self.df.dropna(subset=["_ret"]).reset_index(drop=True)
            return
        if self.horizon == 1 and "log_ret_1d" in self.df.columns:
            # approximate next-day simple return from same-row log_ret is wrong;
            # use forward shift within contract_file instead.
            self.df = self.df.sort_values(["contract_file", "date"]).reset_index(drop=True)
            self.df["_ret"] = (
                self.df.groupby("contract_file", sort=False)["log_ret_1d"].shift(-1).pipe(np.expm1)
            )
            self.df = self.df.dropna(subset=["_ret"]).reset_index(drop=True)
            return
        # H!=3: vectorized close lookups per contract segment
        rets = np.full(len(self.df), np.nan, dtype=float)
        for (sym, key), idx in self.df.groupby(["symbol", "contract_file"], sort=False).groups.items():
            seg = self._closes_df(str(sym), str(key))
            closes = seg["close"].astype(float).to_numpy()
            for i in idx:
                si = int(self.df.at[i, "seg_idx"])
                if si + self.horizon >= len(closes) or si < 0:
                    continue
                c0 = closes[si]
                c1 = closes[si + self.horizon]
                if c0:
                    rets[i] = c1 / c0 - 1.0
        self.df["_ret"] = rets
        self.df = self.df.dropna(subset=["_ret"]).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seg = self._closes_df(str(row["symbol"]), str(row["contract_file"]))
        i = int(row["seg_idx"])
        closes = seg.iloc[i - WINDOW + 1 : i + 1]["close"].astype(float).to_numpy()
        gaf = generate_gaf_image(closes)
        x = torch.tensor(gaf, dtype=torch.float32).unsqueeze(0)
        ret = float(row["_ret"])
        if self.classification:
            y = torch.tensor(ret_to_gaf_class(ret), dtype=torch.long)
        else:
            y = torch.tensor([float(np.clip(ret, -self.clip, self.clip) / self.clip)], dtype=torch.float32)
        return x, y


def run_epoch(model, loader, loss_fn, opt=None) -> float:
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train:
            opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y if pred.shape == y.shape or y.ndim == 1 and pred.shape[0] == y.shape[0] else y)
        if train:
            loss.backward()
            opt.step()
        total += float(loss.item()) * len(x)
        n += len(x)
    return total / max(n, 1)


def train_scheme(scheme: Scheme) -> Path:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    p = scheme.params
    symbols = CATEGORY_SYMBOLS[p["category"]]
    classification = p["loss"] == "ce6"
    clip = float(p["clip"])
    horizon = int(p["horizon"])

    train_ds = GafSchemeDataset(symbols, horizon, clip, end=TRAIN_END, classification=classification)
    valid_ds = GafSchemeDataset(
        symbols, horizon, clip, start=OOS_START, end=OOS_END, classification=classification
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if classification:
        model = GAF_CNN_CLS(6).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
    else:
        model = GAF_CNN().to(DEVICE)
        if p["loss"] == "huber":
            loss_fn = nn.SmoothL1Loss()
        else:
            loss_fn = nn.MSELoss()

    opt = optim.Adam(model.parameters(), lr=1e-3)
    out = EXP_ROOT / scheme.id
    out.mkdir(parents=True, exist_ok=True)
    print(f"[train] {scheme.id} symbols={symbols} train={len(train_ds)} valid={len(valid_ds)} device={DEVICE}")
    for epoch in range(EPOCHS):
        tr = run_epoch(model, train_loader, loss_fn, opt)
        va = run_epoch(model, valid_loader, loss_fn, None)
        print(f"  epoch {epoch+1}/{EPOCHS} train={tr:.5f} valid={va:.5f}")

    model_path = out / "model.pt"
    torch.save(model.state_dict(), model_path)
    meta = {
        "scheme": scheme.id,
        "title": scheme.title,
        "axis": scheme.axis,
        "params": scheme.params,
        "symbols": symbols,
        "classification": classification,
        "n_train": len(train_ds),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {model_path}")
    return out


def sharpe(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or arr.std() <= 1e-12:
        return 0.0
    return float(arr.mean() / arr.std() * math.sqrt(252))


def signal_from_pred(pred: float | int, classification: bool, clip: float) -> int:
    if classification:
        # classes 0..2 positive-ish, 3..5 negative-ish per ret_to_gaf_class
        cls = int(pred)
        if cls in (1, 2):
            return 1
        if cls in (3, 4):
            return -1
        return 0
    pred_ret = float(pred) * clip
    if pred_ret > TH:
        return 1
    if pred_ret < -TH:
        return -1
    return 0


@torch.inference_mode()
def backtest_scheme(scheme: Scheme) -> pd.DataFrame:
    out = EXP_ROOT / scheme.id
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    classification = bool(meta["classification"])
    clip = float(scheme.params["clip"])
    hold = max(int(scheme.params["horizon"]), 1)
    symbols = meta["symbols"]

    if classification:
        model = GAF_CNN_CLS(6).to(DEVICE)
    else:
        model = GAF_CNN().to(DEVICE)
    model.load_state_dict(torch.load(out / "model.pt", map_location=DEVICE))
    model.eval()

    summaries = []
    daily_parts = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}

    for symbol in symbols:
        path = GAF_FEATURES_DIR / f"{symbol}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[(df["date"] >= OOS_START) & (df["date"] <= OOS_END)].sort_values("date")
        # main-ish: keep farthest-near contract by dropping duplicate dates keep last
        df = df.drop_duplicates("date", keep="last").reset_index(drop=True)
        if df.empty:
            continue
        signals = []
        for row in df.itertuples(index=False):
            ck = (symbol, str(row.contract_file))
            if ck not in cache:
                cache[ck] = pd.read_csv(CONTRACTS_DIR / symbol / f"{row.contract_file}.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            seg = cache[ck]
            i = int(row.seg_idx)
            closes = seg.iloc[i - WINDOW + 1 : i + 1]["close"].astype(float).to_numpy()
            if len(closes) < WINDOW:
                signals.append(0)
                continue
            gaf = generate_gaf_image(closes)
            x = torch.tensor(gaf, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred = model(x)
            if classification:
                signals.append(signal_from_pred(int(pred.argmax(dim=-1).item()), True, clip))
            else:
                signals.append(signal_from_pred(float(pred.item()), False, clip))

        positions = np.zeros(len(df), dtype=float)
        buckets = [[] for _ in range(len(df))]
        for i, sig in enumerate(signals):
            if not sig:
                continue
            for j in range(i, min(i + hold, len(df))):
                buckets[j].append(sig)
        for i, vals in enumerate(buckets):
            if vals:
                m = float(np.mean(vals))
                positions[i] = 1.0 if m > 0 else (-1.0 if m < 0 else 0.0)
        log_r = df["log_ret_1d"].fillna(0.0).to_numpy(dtype=float)
        strat = positions * log_r
        turnover = np.abs(np.diff(positions, prepend=0.0))
        for bps in (0, 2):
            net = strat - turnover * bps / 10000.0
            if bps == 0:
                daily_parts.append(
                    pd.DataFrame(
                        {
                            "date": df["date"],
                            "symbol": symbol,
                            "position": positions,
                            "strategy_ret": net,
                            "turnover": turnover,
                        }
                    )
                )
            capital = INIT_CAP * np.exp(np.cumsum(net))
            peak = np.maximum.accumulate(capital)
            summaries.append(
                {
                    "scheme": scheme.id,
                    "symbol": symbol,
                    "bps": bps,
                    "return": float(capital[-1] / INIT_CAP - 1.0),
                    "sharpe": sharpe(net),
                    "max_dd": float(np.min(capital / peak - 1.0)),
                    "active_ratio": float((positions != 0).mean()),
                    "days": len(df),
                }
            )

    summary = pd.DataFrame(summaries)
    res_dir = RESULT_ROOT / scheme.id
    res_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(res_dir / "summary.csv", index=False)
    if daily_parts:
        pd.concat(daily_parts, ignore_index=True).to_csv(res_dir / "daily.csv", index=False)
    zero = summary[summary["bps"] == 0]
    print(
        f"[bt] {scheme.id}: n={len(zero)} medS={zero['sharpe'].median():.2f} avgR={zero['return'].mean():.2%}"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", nargs="*", default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-bt", action="store_true")
    args = ap.parse_args()
    schemes = [by_id(s) for s in args.scheme] if args.scheme else schemes_by_family("G")
    for scheme in schemes:
        if not args.skip_train:
            train_scheme(scheme)
        if not args.skip_bt:
            backtest_scheme(scheme)


if __name__ == "__main__":
    main()
