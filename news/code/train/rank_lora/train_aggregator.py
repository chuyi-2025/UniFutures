#!/usr/bin/env python3
"""Train L60 aggregators (exp / linear / bigru_pos) on cached doc embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import (  # noqa: E402
    AGGREGATORS,
    DATA_ROOT,
    EMBED_ROOT,
    FROZEN_FINSENT_NPY,
    HORIZONS,
    RESULT_ROOT,
    RUN_ROOT,
    SEEDS,
)
from dataset import DateGroupBatchSampler, WindowRankDataset, collate_windows  # noqa: E402
from losses import daily_rank_ic, multi_horizon_pairwise_loss  # noqa: E402
from model import build_aggregator  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_scores = {h: [] for h in HORIZONS}
    all_labels = {h: [] for h in HORIZONS}
    all_valid = {h: [] for h in HORIZONS}
    all_gid = []
    losses = []
    for batch in loader:
        embeds = batch["embeds"].to(device)
        ages = batch["ages"].to(device)
        mask = batch["mask"].to(device)
        gid = batch["group_id"].to(device)
        scores = model(embeds, ages, mask)
        label_dict = {h: batch[f"label_h{h}"].to(device) for h in HORIZONS}
        valid_dict = {h: batch[f"valid_h{h}"].to(device) for h in HORIZONS}
        loss, _ = multi_horizon_pairwise_loss(scores, label_dict, valid_dict, gid, HORIZONS)
        losses.append(float(loss.item()))
        all_gid.append(gid.cpu())
        for h in HORIZONS:
            all_scores[h].append(scores[h].float().cpu())
            all_labels[h].append(label_dict[h].cpu())
            all_valid[h].append(valid_dict[h].cpu())
    gids = torch.cat(all_gid) if all_gid else torch.zeros(0)
    metrics = {"loss": float(np.mean(losses) if losses else 0.0)}
    ics = []
    for h in HORIZONS:
        sc = torch.cat(all_scores[h]) if all_scores[h] else torch.zeros(0)
        lb = torch.cat(all_labels[h]) if all_labels[h] else torch.zeros(0)
        va = torch.cat(all_valid[h]) if all_valid[h] else torch.zeros(0, dtype=torch.bool)
        ic = daily_rank_ic(sc, lb, va, gids)
        metrics[f"ic_h{h}"] = ic
        ics.append(ic)
    metrics["ic_mean"] = float(np.mean(ics) if ics else 0.0)
    return metrics


@torch.no_grad()
def predict(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        scores = model(
            batch["embeds"].to(device),
            batch["ages"].to(device),
            batch["mask"].to(device),
        )
        for i in range(len(batch["symbol"])):
            item = {
                "symbol": batch["symbol"][i],
                "trade_date": pd.Timestamp(batch["trade_date"][i]),
            }
            for h in HORIZONS:
                item[f"score_h{h}"] = float(scores[h][i].cpu().item())
                item[f"ret_h{h}"] = float(batch[f"ret_h{h}"][i].item())
                item[f"valid_h{h}"] = bool(batch[f"valid_h{h}"][i].item())
            rows.append(item)
    return pd.DataFrame(rows)


def train_aggregator(
    name: str,
    windows: pd.DataFrame,
    embeds: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    max_rows: int,
    out_dir: Path,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = WindowRankDataset(windows, embeds, split="train")
    val_ds = WindowRankDataset(windows, embeds, split="val")
    oos_ds = WindowRankDataset(windows, embeds, split="oos")
    train_loader = DataLoader(
        train_ds,
        batch_sampler=DateGroupBatchSampler(train_ds.df, max_rows=max_rows, shuffle=True, seed=seed),
        collate_fn=collate_windows,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=DateGroupBatchSampler(val_ds.df, max_rows=max_rows, shuffle=False, seed=seed),
        collate_fn=collate_windows,
    )
    oos_loader = DataLoader(
        oos_ds,
        batch_sampler=DateGroupBatchSampler(oos_ds.df, max_rows=max_rows, shuffle=False, seed=seed),
        collate_fn=collate_windows,
    )

    model = build_aggregator(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_state, best_ic = None, -1e9
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{name} s{seed} ep{epoch}", leave=False):
            opt.zero_grad(set_to_none=True)
            scores = model(
                batch["embeds"].to(device),
                batch["ages"].to(device),
                batch["mask"].to(device),
            )
            label_dict = {h: batch[f"label_h{h}"].to(device) for h in HORIZONS}
            valid_dict = {h: batch[f"valid_h{h}"].to(device) for h in HORIZONS}
            loss, _ = multi_horizon_pairwise_loss(
                scores, label_dict, valid_dict, batch["group_id"].to(device), HORIZONS
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses) if losses else 0.0),
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(row)
        print(json.dumps({"agg": name, "seed": seed, **row}, ensure_ascii=False))
        if val["ic_mean"] > best_ic:
            best_ic = val["ic_mean"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    val = evaluate(model, val_loader, device)
    oos = evaluate(model, oos_loader, device)
    signals = predict(model, oos_loader, device)
    # also dump all splits for analysis
    all_loader = DataLoader(
        WindowRankDataset(windows, embeds, split=None),
        batch_sampler=DateGroupBatchSampler(
            WindowRankDataset(windows, embeds, split=None).df,
            max_rows=max_rows,
            shuffle=False,
            seed=seed,
        ),
        collate_fn=collate_windows,
    )
    all_sig = predict(model, all_loader, device)
    all_sig.to_parquet(out_dir / "signals.parquet", index=False)
    signals.to_parquet(out_dir / "signals_oos.parquet", index=False)
    meta = {
        "aggregator": name,
        "seed": seed,
        "status": "ok",
        "val": val,
        "oos": oos,
        "history": history,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_oos": len(oos_ds),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, default=DATA_ROOT / "windows_L60.parquet")
    ap.add_argument(
        "--embeds",
        type=Path,
        default=None,
        help="npy of doc embeddings aligned to hierarchical doc_idx",
    )
    ap.add_argument("--aggregator", nargs="+", default=list(AGGREGATORS), choices=list(AGGREGATORS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-rows", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", type=str, default="run")
    args = ap.parse_args()

    embeds_path = args.embeds
    if embeds_path is None:
        lora_npy = EMBED_ROOT / "finsent_lora.npy"
        embeds_path = lora_npy if lora_npy.exists() else FROZEN_FINSENT_NPY
    if not embeds_path.exists():
        raise SystemExit(f"missing embeds {embeds_path}")

    windows = pd.read_parquet(args.windows)
    windows["trade_date"] = pd.to_datetime(windows["trade_date"])
    embeds = np.load(embeds_path).astype(np.float32)
    print(f"[embeds] {embeds_path} shape={embeds.shape}")

    if args.smoke:
        args.epochs = 1
        args.seeds = args.seeds[:1]
        # tiny slice
        keep_dates = (
            windows[windows["split"] == "train"]["trade_date"]
            .drop_duplicates()
            .head(10)
            .tolist()
            + windows[windows["split"] == "val"]["trade_date"].drop_duplicates().head(5).tolist()
            + windows[windows["split"] == "oos"]["trade_date"].drop_duplicates().head(5).tolist()
        )
        windows = windows[windows["trade_date"].isin(keep_dates)].reset_index(drop=True)

    summaries = []
    for name in args.aggregator:
        for seed in args.seeds:
            out = RUN_ROOT / "aggregators" / args.tag / name / f"seed_{seed}"
            meta = train_aggregator(
                name, windows, embeds, seed, args.epochs, args.lr, args.max_rows, out
            )
            summaries.append(meta)
            print(
                f"[done] {name} seed{seed} val_ic={meta['val']['ic_mean']:.4f} "
                f"oos_ic={meta['oos']['ic_mean']:.4f}"
            )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "aggregator": m["aggregator"],
                "seed": m["seed"],
                "val_ic_mean": m["val"]["ic_mean"],
                "oos_ic_mean": m["oos"]["ic_mean"],
                **{f"val_ic_h{h}": m["val"][f"ic_h{h}"] for h in HORIZONS},
                **{f"oos_ic_h{h}": m["oos"][f"ic_h{h}"] for h in HORIZONS},
            }
            for m in summaries
        ]
    ).to_csv(RESULT_ROOT / f"aggregator_ic_{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
