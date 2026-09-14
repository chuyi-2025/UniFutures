#!/usr/bin/env python3
"""Fine-tune sentiment classifiers on weak next-day return labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    DATA_DIR,
    FT_TEST_END,
    FT_TEST_START,
    FT_TRAIN_END,
    FT_TRAIN_START,
    ID2LABEL,
    LABEL2ID,
    WEIGHTS,
    class3_to_pos,
)


class NewsDS(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts = df["text"].astype(str).tolist()
        self.labels = [LABEL2ID[x] for x in df["label"].tolist()]
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_model(model_key: str):
    weight_dir = WEIGHTS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(str(weight_dir), trust_remote_code=True)
    if model_key == "finbert2":
        # MLM checkpoint -> classification head
        config = BertConfig.from_pretrained(str(weight_dir))
        config.num_labels = 3
        config.id2label = ID2LABEL
        config.label2id = LABEL2ID
        model = BertForSequenceClassification.from_pretrained(
            str(weight_dir),
            config=config,
            ignore_mismatched_sizes=True,
        )
    elif model_key == "modernbert":
        model = AutoModelForSequenceClassification.from_pretrained(
            str(weight_dir),
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(weight_dir),
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
    return tokenizer, model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    return {"accuracy": acc}


@torch.inference_mode()
def infer_df(model, tokenizer, df: pd.DataFrame, batch_size: int, max_length: int, device) -> pd.DataFrame:
    model.eval()
    preds = []
    positions = []
    prob_rows = []
    texts = df["text"].astype(str).tolist()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = probs.argmax(axis=-1)
        for j, p in enumerate(pred):
            p = int(p)
            preds.append(p)
            positions.append(class3_to_pos(p))
            prob_rows.append(probs[j].tolist())
    out = df.drop(columns=["text"], errors="ignore").copy()
    out["pred_class"] = preds
    out["position"] = positions
    for c in range(3):
        out[f"prob_{c}"] = [row[c] for row in prob_rows]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=DATA_DIR / "samples.parquet")
    ap.add_argument("--model", choices=("finance_zh", "modernbert", "finbert2"), required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--horizon", type=int, default=1, help="tag outputs; use matching samples_hN")
    ap.add_argument("--signal-out", type=Path, default=None)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.samples) if args.samples.suffix == ".parquet" else pd.read_csv(args.samples)
    df["report_date"] = pd.to_datetime(df["report_date"])
    if args.limit > 0:
        df = df.sample(n=min(args.limit, len(df)), random_state=42).reset_index(drop=True)

    train = df[(df["report_date"] >= FT_TRAIN_START) & (df["report_date"] <= FT_TRAIN_END)].reset_index(drop=True)
    test = df[(df["report_date"] >= FT_TEST_START) & (df["report_date"] <= FT_TEST_END)].reset_index(drop=True)
    print(f"[split] h={args.horizon} train={len(train)} test={len(test)}")
    if train.empty or test.empty:
        raise SystemExit("empty train/test split")

    tag = args.model if args.horizon <= 1 else f"{args.model}_h{args.horizon}"
    out_dir = args.out_dir or (DATA_DIR / "ft_models" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args.model)
    train_ds = NewsDS(train, tokenizer, args.max_length)
    # small eval from train tail for logging
    eval_df = train.sample(n=min(512, len(train)), random_state=0)
    eval_ds = NewsDS(eval_df, tokenizer, args.max_length)

    targs = TrainingArguments(
        output_dir=str(out_dir / "runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    ckpt = out_dir / "best"
    trainer.save_model(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))
    (out_dir / "label_map.json").write_text(json.dumps(LABEL2ID, ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    signals = infer_df(model, tokenizer, test, args.batch_size, args.max_length, device)
    if args.signal_out is not None:
        sig_path = args.signal_out
    elif args.horizon <= 1:
        sig_path = DATA_DIR / f"signals_ft_{args.model}.parquet"
    else:
        sig_path = DATA_DIR / f"signals_ft_{args.model}_h{args.horizon}.parquet"
    signals.to_parquet(sig_path, index=False)
    signals.drop(columns=[c for c in signals.columns if c.startswith("prob_")], errors="ignore").to_csv(
        sig_path.with_suffix(".csv"), index=False
    )
    print(f"[done] model={ckpt} signals={sig_path} n={len(signals)}")


if __name__ == "__main__":
    main()
