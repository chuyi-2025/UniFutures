#!/usr/bin/env python3
"""Zero-shot sentiment inference on OCR samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, WEIGHTS, class3_to_pos, modernbert9_to_pos


def load_samples(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


@torch.inference_mode()
def predict_batch(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred = probs.argmax(axis=-1)
    return pred, probs


def run_model(
    samples: pd.DataFrame,
    model_key: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> pd.DataFrame:
    weight_dir = WEIGHTS[model_key]
    print(f"[load] {model_key} <- {weight_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(weight_dir), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(weight_dir), trust_remote_code=True
    )
    model.eval().to(device)

    preds: list[int] = []
    positions: list[int] = []
    prob_rows: list[list[float]] = []
    texts = samples["text"].astype(str).tolist()
    for i in tqdm(range(0, len(texts), batch_size), desc=f"infer-{model_key}"):
        batch = texts[i : i + batch_size]
        pred, probs = predict_batch(model, tokenizer, batch, device, max_length)
        for j, p in enumerate(pred):
            p = int(p)
            preds.append(p)
            if model_key == "modernbert":
                positions.append(modernbert9_to_pos(p))
            else:
                positions.append(class3_to_pos(p))
            prob_rows.append(probs[j].tolist())

    out = samples.drop(columns=["text"], errors="ignore").copy()
    out["pred_class"] = preds
    out["position"] = positions
    # pad prob columns
    n_cls = len(prob_rows[0]) if prob_rows else 0
    for c in range(n_cls):
        out[f"prob_{c}"] = [row[c] for row in prob_rows]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=DATA_DIR / "samples.parquet")
    ap.add_argument("--model", choices=("finance_zh", "modernbert"), required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    samples = load_samples(args.samples)
    if args.limit > 0:
        samples = samples.head(args.limit).reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = run_model(samples, args.model, args.batch_size, args.max_length, device)
    out_path = args.out or (DATA_DIR / f"signals_zeroshot_{args.model}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    out.drop(columns=[c for c in out.columns if c.startswith("prob_")], errors="ignore").to_csv(
        out_path.with_suffix(".csv"), index=False
    )
    print(f"[done] {len(out)} -> {out_path}")
    print(out["position"].value_counts().to_dict())


if __name__ == "__main__":
    main()
