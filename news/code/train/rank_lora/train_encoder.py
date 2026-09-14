#!/usr/bin/env python3
"""Train / evaluate Financial-Sentiment Qwen LoRA multi-horizon rank encoder.

16k contexts are encoded one document at a time; pairwise loss is computed
after all docs in a trade_date group are scored (avoids SDPA mask OOM).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import (  # noqa: E402
    DATA_ROOT,
    FINSENT_PATH,
    HEAD_TAIL_HALF,
    HORIZONS,
    MAX_DOC_TOKENS,
    RUN_ROOT,
)
from dataset import DocRankDataset, make_collate_docs, prefix_text, tokenize_head_tail  # noqa: E402
from losses import daily_rank_ic, multi_horizon_pairwise_loss, pairwise_logistic_loss  # noqa: E402
from model import MultiHorizonRankEncoder, load_tokenizer  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def group_indices(df: pd.DataFrame) -> list[list[int]]:
    groups: dict = {}
    for i, d in enumerate(df["trade_date"].tolist()):
        groups.setdefault(pd.Timestamp(d), []).append(i)
    return [idxs for idxs in groups.values() if len(idxs) >= 2]


def encode_one(
    model: MultiHorizonRankEncoder,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int,
    use_amp: bool,
) -> dict[int, torch.Tensor]:
    enc = tokenize_head_tail(tokenizer, text, max_length, HEAD_TAIL_HALF)
    input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
    attn = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        _, scores = model(input_ids, attn)
    return {h: scores[h].reshape(()) for h in HORIZONS}


def score_group(
    model: MultiHorizonRankEncoder,
    tokenizer,
    ds: DocRankDataset,
    idxs: list[int],
    device: torch.device,
    max_length: int,
    use_amp: bool,
    max_docs: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Encode each doc alone; return stacked scores/labels/valid for the group.

    When training, cap |idxs| via max_docs so the retained autograd graph fits in VRAM.
    """
    if max_docs and len(idxs) > max_docs:
        pick = (rng or np.random.default_rng()).choice(len(idxs), size=max_docs, replace=False)
        idxs = [idxs[int(i)] for i in sorted(pick.tolist())]
    score_lists = {h: [] for h in HORIZONS}
    label_lists = {h: [] for h in HORIZONS}
    valid_lists = {h: [] for h in HORIZONS}
    for i in idxs:
        item = ds[i]
        text = prefix_text(item["symbol"], str(ds.df.iloc[i]["text"]))
        sc = encode_one(model, tokenizer, text, device, max_length, use_amp)
        for h in HORIZONS:
            score_lists[h].append(sc[h])
            label_lists[h].append(
                torch.tensor(item[f"label_h{h}"], dtype=torch.float32, device=device)
            )
            valid_lists[h].append(
                torch.tensor(item[f"valid_h{h}"], dtype=torch.bool, device=device)
            )
    scores = {h: torch.stack(score_lists[h]) for h in HORIZONS}
    labels = {h: torch.stack(label_lists[h]) for h in HORIZONS}
    valid = {h: torch.stack(valid_lists[h]) for h in HORIZONS}
    return scores, labels, valid


@torch.no_grad()
def evaluate(
    model: MultiHorizonRankEncoder,
    tokenizer,
    ds: DocRankDataset,
    device: torch.device,
    max_length: int,
    use_amp: bool,
    max_groups: int = 0,
) -> dict:
    model.eval()
    groups = group_indices(ds.df)
    if max_groups > 0:
        groups = groups[:max_groups]
    all_scores = {h: [] for h in HORIZONS}
    all_labels = {h: [] for h in HORIZONS}
    all_valid = {h: [] for h in HORIZONS}
    all_gid = []
    losses = []
    for gi, idxs in enumerate(tqdm(groups, desc="eval groups", leave=False)):
        scores, labels, valid = score_group(
            model, tokenizer, ds, idxs, device, max_length, use_amp
        )
        gid = torch.full((len(idxs),), gi, dtype=torch.long, device=device)
        loss, _ = multi_horizon_pairwise_loss(scores, labels, valid, gid, HORIZONS)
        losses.append(float(loss.item()))
        all_gid.append(gid.cpu())
        for h in HORIZONS:
            all_scores[h].append(scores[h].float().cpu())
            all_labels[h].append(labels[h].cpu())
            all_valid[h].append(valid[h].cpu())
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


def train_one(
    model: MultiHorizonRankEncoder,
    tokenizer,
    train_ds: DocRankDataset,
    val_ds: DocRankDataset,
    device: torch.device,
    epochs: int,
    lr: float,
    max_grad_norm: float,
    use_amp: bool,
    out_dir: Path,
    max_length: int,
    grad_accum: int = 4,
    max_group_docs: int = 4,
    seed: int = 0,
) -> dict:
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    best = {"ic_mean": -1e9}
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        groups = group_indices(train_ds.df)
        order = np.arange(len(groups))
        rng.shuffle(order)
        losses = []
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(order, desc=f"encoder ep{epoch}")
        for step, gi in enumerate(pbar, start=1):
            idxs = groups[int(gi)]
            scores, labels, valid = score_group(
                model, tokenizer, train_ds, idxs, device, max_length, use_amp,
                max_docs=max_group_docs, rng=rng,
            )
            h_losses = []
            for h in HORIZONS:
                hl = pairwise_logistic_loss(scores[h], labels[h], valid[h])
                if torch.isfinite(hl):
                    h_losses.append(hl)
            if not h_losses:
                continue
            loss = torch.stack(h_losses).mean() / max(grad_accum, 1)
            if not loss.requires_grad:
                continue
            loss.backward()
            losses.append(float(loss.item()) * max(grad_accum, 1))
            if step % max(grad_accum, 1) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            pbar.set_postfix(loss=float(np.mean(losses[-20:])), n=len(idxs))
        if any(p.grad is not None for p in model.parameters() if p.requires_grad):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
        train_loss = float(np.mean(losses) if losses else 0.0)
        # Intermediate val uses capped groups; full val happens after training in main()
        val = evaluate(
            model, tokenizer, val_ds, device, max_length, use_amp, max_groups=32
        )
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val["ic_mean"] > best["ic_mean"]:
            best = dict(val)
            best["epoch"] = epoch
            save_checkpoint(model, out_dir / "best")
    return {"history": history, "best": best}


def save_checkpoint(model: MultiHorizonRankEncoder, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    enc = model.encoder
    if hasattr(enc, "save_pretrained"):
        enc.save_pretrained(str(out / "adapter"))
    else:
        torch.save(enc.state_dict(), out / "encoder.pt")
    torch.save(model.heads.state_dict(), out / "heads.pt")


def load_heads(model: MultiHorizonRankEncoder, path: Path) -> None:
    model.heads.load_state_dict(torch.load(path, map_location="cpu"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, default=DATA_ROOT / "doc_targets.parquet")
    ap.add_argument("--out", type=Path, default=RUN_ROOT / "encoder")
    ap.add_argument("--base", type=Path, default=FINSENT_PATH)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=MAX_DOC_TOKENS)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument(
        "--max-group-docs",
        type=int,
        default=4,
        help="cap docs per date group during train (keeps 16k autograd under VRAM)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="1-group smoke at 16k")
    ap.add_argument("--frozen-baseline", action="store_true", help="eval frozen encoder only")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--max-val-groups", type=int, default=0, help="cap val groups (0=all)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.targets)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if args.smoke:
        dates = (
            df[df["split"] == "train"]
            .groupby("trade_date")
            .size()
            .sort_values(ascending=False)
            .head(2)
            .index
        )
        vdates = (
            df[df["split"] == "val"]
            .groupby("trade_date")
            .size()
            .sort_values(ascending=False)
            .head(2)
            .index
        )
        df = df[
            ((df["split"] == "train") & df["trade_date"].isin(dates))
            | ((df["split"] == "val") & df["trade_date"].isin(vdates))
        ].reset_index(drop=True)
        args.epochs = 1
        args.max_val_groups = 2

    tok = load_tokenizer(args.base)
    train_ds = DocRankDataset(df, tok, max_length=args.max_length, split="train")
    val_ds = DocRankDataset(df, tok, max_length=args.max_length, split="val")
    print(f"train_rows={len(train_ds)} val_rows={len(val_ds)} max_length={args.max_length}")
    if len(train_ds) < 4 or len(val_ds) < 2:
        raise SystemExit("insufficient train/val rows for ranking")

    use_lora = not args.no_lora and not args.frozen_baseline
    print(f"[load] {args.base} lora={use_lora}")
    model = MultiHorizonRankEncoder(args.base, use_lora=use_lora).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable={trainable:,} / {total:,} ({100 * trainable / max(total, 1):.2f}%)")

    t0 = time.time()
    # Cap baseline during LoRA train for speed; full val for frozen-baseline / final post
    baseline_groups = args.max_val_groups
    if baseline_groups <= 0:
        if args.frozen_baseline:
            baseline_groups = 0  # all
        elif args.smoke:
            baseline_groups = 8
        else:
            baseline_groups = 32
    print(f"[baseline] frozen/current eval on val (max_groups={baseline_groups or 'all'})")
    baseline = evaluate(
        model,
        tok,
        val_ds,
        device,
        args.max_length,
        use_amp,
        max_groups=baseline_groups,
    )
    print("BASELINE", json.dumps(baseline))
    if device.type == "cuda":
        print(f"peak_vram={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")

    if args.frozen_baseline:
        meta = {"status": "baseline_only", "baseline": baseline, "seconds": time.time() - t0}
        (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(
            f"VERDICT: BASELINE | score={baseline['ic_mean']:.4f} | "
            f"baseline={baseline['ic_mean']:.4f} | delta=+0.0000"
        )
        return

    if args.smoke:
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        groups = group_indices(train_ds.df)
        idxs = groups[0]
        opt.zero_grad(set_to_none=True)
        try:
            scores, labels, valid = score_group(
                model, tok, train_ds, idxs, device, args.max_length, use_amp,
                max_docs=args.max_group_docs,
            )
            h_losses = [
                pairwise_logistic_loss(scores[h], labels[h], valid[h]) for h in HORIZONS
            ]
            loss = torch.stack([x for x in h_losses if torch.isfinite(x)]).mean()
            loss.backward()
            opt.step()
            peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            print(
                f"[smoke] step_ok loss={float(loss.item()):.4f} "
                f"peak_vram={peak:.2f}GB group={len(idxs)} used={min(len(idxs), args.max_group_docs)}"
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("[smoke] OOM at 16k single-doc encoding — blocked")
            raise
        post = evaluate(
            model, tok, val_ds, device, args.max_length, use_amp, max_groups=args.max_val_groups
        )
        save_checkpoint(model, args.out / "smoke")
        delta = post["ic_mean"] - baseline["ic_mean"]
        verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
        meta = {
            "status": "smoke_ok",
            "baseline": baseline,
            "post": post,
            "delta": delta,
            "verdict": verdict,
            "max_length": args.max_length,
            "seconds": time.time() - t0,
        }
        (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(
            f"VERDICT: {verdict} | score={post['ic_mean']:.4f} | "
            f"baseline={baseline['ic_mean']:.4f} | delta={delta:+.4f}"
        )
        return

    result = train_one(
        model,
        tok,
        train_ds,
        val_ds,
        device,
        epochs=args.epochs,
        lr=args.lr,
        max_grad_norm=1.0,
        use_amp=use_amp,
        out_dir=args.out,
        max_length=args.max_length,
        grad_accum=args.grad_accum,
        max_group_docs=args.max_group_docs,
        seed=args.seed,
    )
    if (args.out / "best" / "heads.pt").exists():
        from peft import PeftModel

        model = MultiHorizonRankEncoder(args.base, use_lora=False).to(device)
        model.encoder = PeftModel.from_pretrained(model.encoder, str(args.out / "best" / "adapter"))
        load_heads(model, args.out / "best" / "heads.pt")
        model.to(device)
    post = evaluate(model, tok, val_ds, device, args.max_length, use_amp)
    delta = post["ic_mean"] - baseline["ic_mean"]
    verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
    meta = {
        "status": "ok",
        "baseline": baseline,
        "best": result["best"],
        "post": post,
        "history": result["history"],
        "delta": delta,
        "verdict": verdict,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "trainable": trainable,
        "seconds": time.time() - t0,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"VERDICT: {verdict} | score={post['ic_mean']:.4f} | "
        f"baseline={baseline['ic_mean']:.4f} | delta={delta:+.4f}"
    )


if __name__ == "__main__":
    main()
