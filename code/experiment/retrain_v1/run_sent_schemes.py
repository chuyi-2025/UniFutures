#!/usr/bin/env python3
"""Train and backtest S-family sentiment retrain schemes."""

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
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

ROOT = Path("/home/workspace/lab/UniFutures")
sys.path.insert(0, str(ROOT / "news/code/train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_DIR,
    FT_TEST_END,
    FT_TEST_START,
    FT_TRAIN_END,
    FT_TRAIN_START,
    ID2LABEL,
    LABEL2ID,
    WEIGHTS,
    class3_to_pos,
    load_symbol_ohlc,
)
from schemes import EXP_ROOT, RESULT_ROOT, SEED, Scheme, by_id, schemes_by_family  # noqa: E402

INIT_CAP = 1_000_000.0


def relabel(df: pd.DataFrame, ret_th: float, horizon: int) -> pd.DataFrame:
    out = df.copy()
    th = ret_th * float(np.sqrt(max(horizon, 1)))
    ret = out["log_ret_1d"] if horizon <= 1 else out.get("log_ret_h", out["log_ret_1d"])
    # horizon samples store cumulative log in log_ret_1d column historically; keep as-is
    labels = []
    for r in ret.astype(float):
        if not np.isfinite(r) or abs(r) < th:
            labels.append("neutral")
        elif r > 0:
            labels.append("positive")
        else:
            labels.append("negative")
    out["label"] = labels
    return out


class NewsDS(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, mode: str = "cls3"):
        self.texts = df["text"].astype(str).tolist()
        self.mode = mode
        self.tok = tokenizer
        self.max_length = max_length
        if mode == "binary":
            # map positive=1, negative=0
            self.labels = [1 if x == "positive" else 0 for x in df["label"].tolist()]
        elif mode == "regress":
            self.labels = df["log_ret_1d"].astype(float).clip(-0.05, 0.05).tolist()
        else:
            self.labels = [LABEL2ID[x] for x in df["label"].tolist()]

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
        if self.mode == "regress":
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_backbone(model_key: str, mode: str):
    weight_dir = WEIGHTS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(str(weight_dir), trust_remote_code=True)
    if mode == "binary":
        num_labels = 2
        id2label = {0: "negative", 1: "positive"}
        label2id = {"negative": 0, "positive": 1}
    elif mode == "regress":
        num_labels = 1
        id2label = {0: "ret"}
        label2id = {"ret": 0}
    else:
        num_labels = 3
        id2label = ID2LABEL
        label2id = LABEL2ID

    if model_key == "finbert2":
        config = BertConfig.from_pretrained(str(weight_dir))
        config.num_labels = num_labels
        config.id2label = id2label
        config.label2id = label2id
        if mode == "regress":
            config.problem_type = "regression"
        model = BertForSequenceClassification.from_pretrained(
            str(weight_dir), config=config, ignore_mismatched_sizes=True
        )
    else:
        kwargs = dict(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
        if mode == "regress":
            kwargs["problem_type"] = "regression"
        model = AutoModelForSequenceClassification.from_pretrained(str(weight_dir), **kwargs)
    return tokenizer, model


def load_samples(scheme: Scheme) -> pd.DataFrame:
    h = int(scheme.params["horizon"])
    if h <= 1:
        path = DATA_DIR / "samples.parquet"
    else:
        path = DATA_DIR / f"samples_h{h}.parquet"
    df = pd.read_parquet(path)
    df["report_date"] = pd.to_datetime(df["report_date"])
    mode = scheme.params.get("mode", "cls3")
    if mode == "binary":
        df = df[df["label"].isin(["positive", "negative"])].reset_index(drop=True)
    elif mode != "regress":
        df = relabel(df, float(scheme.params.get("ret_th", 0.001)), h)
    return df


@torch.inference_mode()
def infer_positions(model, tokenizer, df: pd.DataFrame, mode: str, batch_size: int, max_length: int, device):
    model.eval()
    texts = df["text"].astype(str).tolist()
    positions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).logits
        if mode == "regress":
            vals = out.squeeze(-1).detach().cpu().numpy()
            for v in vals:
                positions.append(1 if v > 0.001 else (-1 if v < -0.001 else 0))
        elif mode == "binary":
            pred = out.argmax(dim=-1).detach().cpu().numpy()
            for p in pred:
                positions.append(1 if int(p) == 1 else -1)
        else:
            pred = out.argmax(dim=-1).detach().cpu().numpy()
            for p in pred:
                positions.append(class3_to_pos(int(p)))
    return positions


def sharpe(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or arr.std() <= 1e-12:
        return 0.0
    return float(arr.mean() / arr.std() * math.sqrt(252))


def backtest_signals(signals: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    daily_parts = []
    start, end = FT_TEST_START, FT_TEST_END
    for symbol, sig in signals.groupby("symbol", sort=True):
        px = load_symbol_ohlc(symbol)
        if px.empty:
            continue
        px = px[(px["date"] >= start) & (px["date"] <= end)].copy()
        daily = (
            sig.groupby("trade_date", sort=True)["position"]
            .apply(lambda s: int(np.sign(s.sum())) if s.sum() != 0 else 0)
            .rename("position")
            .reset_index()
        )
        daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.normalize()
        part = px.merge(daily, left_on="date", right_on="trade_date", how="left")
        part["position"] = part["position"].fillna(0.0)
        part["log_ret_1d"] = part["log_ret_1d"].fillna(0.0)
        part["strategy_ret"] = part["position"] * part["log_ret_1d"]
        part["turnover"] = part["position"].diff().fillna(part["position"]).abs()
        for bps in (0, 2):
            net = part["strategy_ret"] - part["turnover"] * bps / 10000.0
            if bps == 0:
                daily_parts.append(
                    pd.DataFrame(
                        {
                            "date": part["date"],
                            "symbol": symbol,
                            "position": part["position"],
                            "strategy_ret": net,
                            "turnover": part["turnover"],
                        }
                    )
                )
            capital = INIT_CAP * np.exp(np.cumsum(net.to_numpy()))
            peak = np.maximum.accumulate(capital)
            summaries.append(
                {
                    "symbol": symbol,
                    "bps": bps,
                    "return": float(capital[-1] / INIT_CAP - 1.0),
                    "sharpe": sharpe(net.to_numpy()),
                    "max_dd": float(np.min(capital / peak - 1.0)),
                    "active_ratio": float((part["position"] != 0).mean()),
                    "days": len(part),
                }
            )
    return pd.DataFrame(summaries), (pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame())


def train_scheme(scheme: Scheme) -> Path:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    mode = scheme.params.get("mode", "cls3")
    df = load_samples(scheme)
    train = df[(df["report_date"] >= FT_TRAIN_START) & (df["report_date"] <= FT_TRAIN_END)].reset_index(drop=True)
    test = df[(df["report_date"] >= FT_TEST_START) & (df["report_date"] <= FT_TEST_END)].reset_index(drop=True)
    if train.empty or test.empty:
        raise SystemExit(f"{scheme.id}: empty split train={len(train)} test={len(test)}")

    tokenizer, model = load_backbone(scheme.params["model"], mode)
    out = EXP_ROOT / scheme.id
    out.mkdir(parents=True, exist_ok=True)
    train_ds = NewsDS(train, tokenizer, 512, mode=mode)
    eval_ds = NewsDS(train.sample(n=min(512, len(train)), random_state=0), tokenizer, 512, mode=mode)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if mode == "regress":
            pred = np.asarray(logits).reshape(-1)
            labels = np.asarray(labels).reshape(-1)
            mse = float(np.mean((pred - labels) ** 2))
            return {"mse": mse}
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    targs = TrainingArguments(
        output_dir=str(out / "runs"),
        num_train_epochs=2.0,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy" if mode != "regress" else "mse",
        greater_is_better=mode != "regress",
        fp16=torch.cuda.is_available(),
        report_to=[],
        save_total_limit=1,
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    print(f"[train] {scheme.id} train={len(train)} test={len(test)} mode={mode}")
    trainer.train()
    ckpt = out / "best"
    trainer.save_model(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    positions = infer_positions(model, tokenizer, test, mode, 8, 512, device)
    signals = test.drop(columns=["text"], errors="ignore").copy()
    signals["position"] = positions
    signals.to_parquet(out / "signals.parquet", index=False)
    signals.to_csv(out / "signals.csv", index=False)
    (out / "meta.json").write_text(
        json.dumps(
            {"scheme": scheme.id, "title": scheme.title, "axis": scheme.axis, "params": scheme.params, "n_train": len(train)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def backtest_scheme(scheme: Scheme) -> None:
    out = EXP_ROOT / scheme.id
    signals = pd.read_parquet(out / "signals.parquet")
    summary, daily = backtest_signals(signals)
    summary["scheme"] = scheme.id
    res = RESULT_ROOT / scheme.id
    res.mkdir(parents=True, exist_ok=True)
    summary.to_csv(res / "summary.csv", index=False)
    if not daily.empty:
        daily.to_csv(res / "daily.csv", index=False)
    zero = summary[summary["bps"] == 0]
    print(
        f"[bt] {scheme.id}: n={len(zero)} medS={zero['sharpe'].median():.2f} avgR={zero['return'].mean():.2%}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", nargs="*", default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-bt", action="store_true")
    args = ap.parse_args()
    schemes = [by_id(s) for s in args.scheme] if args.scheme else schemes_by_family("S")
    for scheme in schemes:
        if not args.skip_train:
            train_scheme(scheme)
        if not args.skip_bt:
            backtest_scheme(scheme)


if __name__ == "__main__":
    main()
