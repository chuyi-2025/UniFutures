#!/usr/bin/env python3
"""Concatenate past-L-day texts per (symbol, trade_date) and score with frozen ft finance_zh.

Newest reports first so 512-token truncation keeps recent content.
Window: report_date in [T - lookback, T) — no same-day.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_DIR,
    FT_TEST_END,
    FT_TEST_START,
    class3_to_pos,
    load_symbol_ohlc,
)

CKPT = DATA_DIR / "ft_models" / "finance_zh" / "best"


@torch.inference_mode()
def score_texts(
    model, tokenizer, texts: list[str], batch_size: int, max_length: int, device
) -> tuple[np.ndarray, np.ndarray]:
    probs_all, preds = [], []
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
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        probs_all.append(probs)
        preds.append(probs.argmax(axis=-1))
    return np.concatenate(preds), np.concatenate(probs_all, axis=0)


def build_concat_jobs(
    samples: pd.DataFrame,
    symbols: list[str],
    lookback: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_chars: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sym in symbols:
        px = load_symbol_ohlc(sym)
        if px.empty:
            continue
        dates = px[(px["date"] >= start) & (px["date"] <= end)]["date"].map(
            lambda x: pd.Timestamp(x).normalize()
        )
        sub = samples[samples["symbol"] == sym].sort_values("report_date", ascending=False)
        if sub.empty:
            continue
        rdates = sub["report_date"].to_numpy()
        texts = sub["text"].astype(str).tolist()
        for T in dates:
            t0 = T - pd.Timedelta(days=lookback)
            mask = (rdates >= np.datetime64(t0)) & (rdates < np.datetime64(T))
            if not mask.any():
                continue
            # newest first (sub already descending); keep order of True masks
            idxs = np.where(mask)[0]
            parts = [texts[i] for i in idxs]
            blob = " [SEP] ".join(parts)[:max_chars]
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": T,
                    "report_date": T - pd.Timedelta(days=1),
                    "text": blob,
                    "n_docs": int(mask.sum()),
                    "lookback": lookback,
                    "scheme": "concat",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=DATA_DIR / "samples.parquet")
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--lookback", type=int, nargs="+", default=[3, 7, 14])
    ap.add_argument("--start", type=str, default=str(FT_TEST_START.date()))
    ap.add_argument("--end", type=str, default=str(FT_TEST_END.date()))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR / "lookback")
    ap.add_argument(
        "--tag",
        default="finance_zh",
        help="output filename tag: signals_lb_<tag>_concat_L*.parquet",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.ckpt.is_dir():
        raise SystemExit(f"missing ckpt {args.ckpt}")

    samples = pd.read_parquet(args.samples)
    samples["report_date"] = pd.to_datetime(samples["report_date"]).dt.normalize()
    samples["symbol"] = samples["symbol"].astype(str).str.upper()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    symbols = sorted(samples["symbol"].unique())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(args.ckpt), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.ckpt), trust_remote_code=True
    )
    model.eval().to(device)

    for L in args.lookback:
        print(f"[concat] L={L} building jobs…")
        jobs = build_concat_jobs(samples, symbols, L, start, end, args.max_chars)
        if args.limit > 0:
            jobs = jobs.head(args.limit).reset_index(drop=True)
        if jobs.empty:
            print("  empty; skip")
            continue
        print(f"  scoring n={len(jobs)} on {device}")
        pred, probs = score_texts(
            model, tok, jobs["text"].tolist(), args.batch_size, args.max_length, device
        )
        out = jobs.drop(columns=["text"]).copy()
        out["pred_class"] = pred.astype(int)
        out["position"] = [class3_to_pos(int(p)) for p in pred]
        for i in range(3):
            out[f"prob_{i}"] = probs[:, i]
        path = args.out_dir / f"signals_lb_{args.tag}_concat_L{L}.parquet"
        out.to_parquet(path, index=False)
        out.to_csv(path.with_suffix(".csv"), index=False)
        print(f"  -> {path} nonzero={(out['position']!=0).sum()}")


if __name__ == "__main__":
    main()
