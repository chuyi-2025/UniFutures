#!/usr/bin/env python3
"""Sanity checks: label shuffle → ~0 IC; doc-order shuffle → lower IC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import DATA_ROOT, HORIZONS, RUN_ROOT  # noqa: E402
from dataset import DateGroupBatchSampler, WindowRankDataset, collate_windows  # noqa: E402
from losses import daily_rank_ic  # noqa: E402
from model import build_aggregator  # noqa: E402
from train_aggregator import evaluate  # noqa: E402


def label_shuffle_ic(windows: pd.DataFrame, embeds: np.ndarray, seed: int = 0) -> float:
    """Shuffle returns within each trade_date; Rank IC of a fresh model should be ~0."""
    rng = np.random.default_rng(seed)
    w = windows.copy()
    for h in HORIZONS:
        parts = []
        for _, g in w.groupby("trade_date", sort=False):
            gg = g.copy()
            perm = rng.permutation(len(gg))
            gg[f"ret_h{h}"] = gg[f"ret_h{h}"].to_numpy()[perm]
            gg[f"rank_h{h}"] = (
                gg[f"ret_h{h}"].rank(method="average", ascending=True).astype(float)
            )
            parts.append(gg)
        w = pd.concat(parts, ignore_index=True)
    # Recompute group_size
    w["group_size"] = w.groupby("trade_date")["symbol"].transform("count")
    ds = WindowRankDataset(w, embeds, split="val")
    if len(ds) < 10:
        return float("nan")
    loader = DataLoader(
        ds,
        batch_sampler=DateGroupBatchSampler(ds.df, max_rows=64, shuffle=False, seed=seed),
        collate_fn=collate_windows,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_aggregator("exp").to(device)
    return evaluate(model, loader, device)["ic_mean"]


def doc_shuffle_delta(
    windows: pd.DataFrame,
    embeds: np.ndarray,
    model_path: Path | None,
    seed: int = 0,
) -> dict:
    """Compare val IC with intact vs randomly permuted doc idxs in each window."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_aggregator("exp").to(device)
    if model_path and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.to(device)

    clean_ds = WindowRankDataset(windows, embeds, split="val")
    clean_loader = DataLoader(
        clean_ds,
        batch_sampler=DateGroupBatchSampler(clean_ds.df, max_rows=64, shuffle=False, seed=seed),
        collate_fn=collate_windows,
    )
    clean_ic = evaluate(model, clean_loader, device)["ic_mean"]

    rng = np.random.default_rng(seed)
    shuffled = windows.copy()
    new_idxs, new_ages = [], []
    for row in shuffled.itertuples(index=False):
        idxs = list(row.doc_idxs)
        ages = list(row.ages)
        if len(idxs) > 1:
            perm = rng.permutation(len(idxs))
            idxs = [idxs[i] for i in perm]
            ages = [ages[i] for i in perm]
        new_idxs.append(idxs)
        new_ages.append(ages)
    shuffled["doc_idxs"] = new_idxs
    shuffled["ages"] = new_ages
    shuf_ds = WindowRankDataset(shuffled, embeds, split="val")
    shuf_loader = DataLoader(
        shuf_ds,
        batch_sampler=DateGroupBatchSampler(shuf_ds.df, max_rows=64, shuffle=False, seed=seed),
        collate_fn=collate_windows,
    )
    shuf_ic = evaluate(model, shuf_loader, device)["ic_mean"]
    return {
        "clean_ic": clean_ic,
        "shuffled_docs_ic": shuf_ic,
        "delta": shuf_ic - clean_ic,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, default=DATA_ROOT / "windows_L60.parquet")
    ap.add_argument("--embeds", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=RUN_ROOT / "sanity.json")
    args = ap.parse_args()

    windows = pd.read_parquet(args.windows)
    windows["trade_date"] = pd.to_datetime(windows["trade_date"])
    embeds = np.load(args.embeds).astype(np.float32)

    label_ic = label_shuffle_ic(windows, embeds)
    doc = doc_shuffle_delta(windows, embeds, args.model)
    report = {
        "label_shuffle_ic": label_ic,
        "label_shuffle_ok": bool(np.isfinite(label_ic) and abs(label_ic) < 0.05),
        **doc,
        "doc_shuffle_ok": bool(doc["delta"] <= 0.01),  # should not improve
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["label_shuffle_ok"]:
        print("SANITY:FAIL label_shuffle_ic not near zero", file=sys.stderr)
        raise SystemExit(2)
    print("SANITY:PASS")


if __name__ == "__main__":
    main()
