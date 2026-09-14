#!/usr/bin/env python3
"""Infer fine-tuned next-day sentiment models on all samples and build daily ensemble features for PPO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "news/code/train"))
from common import DATA_DIR as NEWS_SENT_DIR  # noqa: E402
from common import class3_to_pos  # noqa: E402

ROOT = Path("/home/workspace/lab/UniFutures")
FT_ROOT = NEWS_SENT_DIR / "ft_models"
OUT_DOC = NEWS_SENT_DIR / "signals_ft_full_ensemble.parquet"
OUT_DAILY = ROOT / "data/features/sentiment_daily.parquet"
MODELS = ("finance_zh", "modernbert", "finbert2")


@torch.inference_mode()
def infer_model(ckpt: Path, texts: list[str], batch_size: int, max_length: int, device) -> tuple[np.ndarray, np.ndarray]:
    tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt), trust_remote_code=True)
    model.eval().to(device)
    preds, probs_all = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc=ckpt.parent.name):
        batch = texts[i : i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = probs.argmax(axis=-1)
        preds.append(pred)
        probs_all.append(probs)
    return np.concatenate(preds), np.concatenate(probs_all, axis=0)


def majority_pos(vals: list[int]) -> int:
    s = sum(vals)
    if s > 0:
        return 1
    if s < 0:
        return -1
    return 0


def build_daily(doc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sym, dt), grp in doc.groupby(["symbol", "trade_date"], sort=True):
        pos = majority_pos(grp["position"].astype(int).tolist())
        p0 = float(grp["prob_0"].mean())
        p1 = float(grp["prob_1"].mean())
        p2 = float(grp["prob_2"].mean())
        rows.append(
            {
                "symbol": sym,
                "date": pd.Timestamp(dt).normalize(),
                "sent_pos": pos,
                "sent_p0": p0,
                "sent_p1": p1,
                "sent_p2": p2,
                "n_docs": len(grp),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=NEWS_SENT_DIR / "samples.parquet")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    samples = pd.read_parquet(args.samples)
    if args.limit > 0:
        samples = samples.head(args.limit).reset_index(drop=True)
    texts = samples["text"].astype(str).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_probs = []
    all_pos = []
    for name in MODELS:
        ckpt = FT_ROOT / name / "best"
        if not ckpt.is_dir():
            raise SystemExit(f"missing ft checkpoint: {ckpt}")
        print(f"[infer] {name} <- {ckpt}")
        pred, probs = infer_model(ckpt, texts, args.batch_size, args.max_length, device)
        all_probs.append(probs)
        all_pos.append(np.array([class3_to_pos(int(p)) for p in pred], dtype=np.int8))

    # probability average ensemble
    avg_prob = np.mean(np.stack(all_probs, axis=0), axis=0)
    ens_cls = avg_prob.argmax(axis=-1)
    ens_pos = np.array([class3_to_pos(int(c)) for c in ens_cls], dtype=np.int8)
    # also keep vote position for reference
    vote = np.sign(np.sum(np.stack(all_pos, axis=0), axis=0)).astype(np.int8)

    out = samples.drop(columns=["text"], errors="ignore").copy()
    out["position"] = ens_pos  # default: prob-avg ensemble for PPO
    out["position_vote"] = vote
    out["pred_class"] = ens_cls
    for i in range(3):
        out[f"prob_{i}"] = avg_prob[:, i]
    for mi, name in enumerate(MODELS):
        out[f"pos_{name}"] = all_pos[mi]
        for i in range(3):
            out[f"{name}_p{i}"] = all_probs[mi][:, i]

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DOC, index=False)
    print(f"[done] doc signals {len(out)} -> {OUT_DOC}")

    daily = build_daily(out)
    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT_DAILY, index=False)
    daily.to_csv(OUT_DAILY.with_suffix(".csv"), index=False)
    print(f"[done] daily {len(daily)} -> {OUT_DAILY}")
    print(daily["sent_pos"].value_counts().to_dict())


if __name__ == "__main__":
    main()
