#!/usr/bin/env python3
"""Build full-document, per-symbol token chunks with next-day weak labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import DATA_DIR, WEIGHTS, forward_horizon_ret, load_symbol_ohlc, ret_to_label
from symbol_map import SYMBOL_ALIASES

HIER_ROOT = DATA_DIR / "hierarchical"
DEFAULT_OUT = DATA_DIR / "chunk_samples_h1.parquet"


def canonical_symbol_names() -> dict[str, str]:
    """Use the first configured Chinese alias as a stable display name."""
    names: dict[str, str] = {}
    for alias, symbol in SYMBOL_ALIASES.items():
        if any("\u4e00" <= char <= "\u9fff" for char in alias):
            names.setdefault(symbol, alias)
    return names


def split_with_prefix(
    text: str,
    prefix: str,
    tokenizer,
    max_length: int,
) -> list[tuple[str, int]]:
    """Return text chunks whose retokenized length, including specials, fits."""
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    body_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    body_budget = max_length - special_tokens - len(prefix_ids)
    if body_budget < 1:
        raise ValueError(
            f"prefix consumes token budget: prefix={prefix!r} max_length={max_length}"
        )

    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(body_ids):
        part_ids = body_ids[start : start + body_budget]
        part = tokenizer.decode(
            part_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        sample = prefix + part
        encoded = tokenizer(sample, add_special_tokens=True)["input_ids"]
        # Decoding and re-tokenizing can rarely alter a boundary. Shrink until safe.
        while len(encoded) > max_length and part_ids:
            part_ids = part_ids[:-1]
            part = tokenizer.decode(
                part_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            sample = prefix + part
            encoded = tokenizer(sample, add_special_tokens=True)["input_ids"]
        if part_ids:
            chunks.append((sample, len(encoded)))
            start += len(part_ids)
        else:
            raise ValueError("unable to fit any body token after prefix")
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents", type=Path, default=HIER_ROOT / "documents.parquet")
    ap.add_argument("--links", type=Path, default=HIER_ROOT / "document_symbols.parquet")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit-docs", type=int, default=0)
    args = ap.parse_args()

    documents = pd.read_parquet(args.documents)
    links = pd.read_parquet(args.links)
    documents["report_date"] = pd.to_datetime(documents["report_date"]).dt.normalize()
    links["report_date"] = pd.to_datetime(links["report_date"]).dt.normalize()
    links["symbol"] = links["symbol"].astype(str).str.upper()
    if args.limit_docs:
        documents = documents.iloc[: args.limit_docs].copy()
        links = links[links["doc_idx"].isin(documents["doc_idx"])].copy()

    tokenizer = AutoTokenizer.from_pretrained(
        str(WEIGHTS["finance_zh"]), trust_remote_code=True
    )
    # Full documents are intentionally tokenized before being split.
    tokenizer.model_max_length = 10**9
    symbol_names = canonical_symbol_names()

    symbols = sorted(links["symbol"].unique())
    prices: dict[str, pd.DataFrame] = {}
    for symbol in tqdm(symbols, desc="load prices"):
        frame = load_symbol_ohlc(symbol)
        if not frame.empty:
            prices[symbol] = frame.set_index("date").sort_index()

    merged = links.merge(
        documents[
            [
                "doc_idx",
                "doc_id",
                "path",
                "institution",
                "report_date",
                "variety_raw",
                "text",
            ]
        ],
        on=["doc_idx", "doc_id", "report_date"],
        how="inner",
        validate="many_to_one",
    )

    rows: list[dict] = []
    skipped_return = 0
    for row in tqdm(
        merged.itertuples(index=False), total=len(merged), desc="token chunks"
    ):
        price = prices.get(row.symbol)
        if price is None:
            skipped_return += 1
            continue
        trade_date, forward_ret = forward_horizon_ret(
            price, pd.Timestamp(row.report_date), horizon=1
        )
        if trade_date is None or not pd.notna(forward_ret):
            skipped_return += 1
            continue

        variety_name = symbol_names.get(row.symbol, row.symbol)
        prefix = f"品种:{variety_name}\n"
        chunks = split_with_prefix(
            row.text, prefix, tokenizer, max_length=args.max_length
        )
        n_chunks = len(chunks)
        for chunk_idx, (sample_text, token_count) in enumerate(chunks):
            rows.append(
                {
                    "doc_idx": int(row.doc_idx),
                    "doc_id": row.doc_id,
                    "path": row.path,
                    "institution": row.institution,
                    "report_date": pd.Timestamp(row.report_date),
                    "variety_raw": row.variety_raw,
                    "symbol": row.symbol,
                    "variety_name": variety_name,
                    "chunk_idx": chunk_idx,
                    "n_chunks": n_chunks,
                    "token_count": token_count,
                    "text": sample_text,
                    "trade_date": pd.Timestamp(trade_date).normalize(),
                    "log_ret_1d": float(forward_ret),
                    "label": ret_to_label(float(forward_ret), horizon=1),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no chunk samples produced")
    if int(out["token_count"].max()) > args.max_length:
        raise AssertionError("chunk exceeds max_length")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    out.drop(columns=["text"]).to_csv(args.out.with_suffix(".csv"), index=False)
    print(
        f"[done] rows={len(out)} doc_symbol={out.groupby(['doc_id', 'symbol']).ngroups} "
        f"docs={out['doc_id'].nunique()} symbols={out['symbol'].nunique()} "
        f"max_tokens={out['token_count'].max()} skipped_return={skipped_return} "
        f"-> {args.out}"
    )
    print("labels", out.groupby("label").size().to_dict())
    print(
        "chunks/doc_symbol",
        out.groupby(["doc_id", "symbol"]).size().describe(
            percentiles=[0.5, 0.9, 0.99]
        ).to_dict(),
    )


if __name__ == "__main__":
    main()
