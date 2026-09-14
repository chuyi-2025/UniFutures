#!/usr/bin/env python3
"""Filter full documents to symbol-relevant sentences, then build 512-token samples."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_chunk_samples import canonical_symbol_names
from common import (
    CONTRACTS_DIR,
    DATA_DIR,
    REMOVED_SYMBOLS,
    WEIGHTS,
    forward_horizon_ret,
    load_symbol_ohlc,
    ret_to_label,
)
from symbol_map import SYMBOL_ALIASES, map_varieties

HIER_ROOT = DATA_DIR / "hierarchical"
EMBEDDING_MODEL = Path("/home/workspace/weights/Qwen3-Embedding-0.6B")
DEFAULT_OUT = DATA_DIR / "relevant_chunk_samples_h1.parquet"
QUERY_INSTRUCTION = (
    "Retrieve sentences strongly and specifically relevant to the target commodity, "
    "including its futures market, supply, demand, inventory, cost, profit, price "
    "outlook, and industry chain."
)
BOILERPLATE_RE = re.compile(
    r"(免责声明|风险揭示书|本报告版权|投资者应当|联系电话|联系地址|邮政编码|"
    r"资格证号|从业资格|期货开户|扫码|公众号|数据来源[:：]?$)"
)
PREFERRED_NAMES = {
    "CU": "铜",
    "AL": "铝",
    "ZN": "锌",
    "PB": "铅",
    "NI": "镍",
    "SN": "锡",
    "RB": "螺纹钢",
    "HC": "热卷",
    "RU": "天然橡胶",
    "NR": "20号胶",
    "TA": "PTA",
    "V": "PVC",
    "EC": "集运欧线",
}


def symbol_aliases() -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = defaultdict(list)
    for alias, symbol in SYMBOL_ALIASES.items():
        value = alias.strip()
        if value and value not in aliases[symbol]:
            aliases[symbol].append(value)
    return dict(aliases)


def preferred_symbol_names() -> dict[str, str]:
    names = canonical_symbol_names()
    names.update(PREFERRED_NAMES)
    return names


def build_query(symbol: str, aliases: dict[str, list[str]], names: dict[str, str]) -> str:
    """Build a descriptive query; never rely on a one-character commodity token."""
    name = names.get(symbol, symbol)
    terms = list(aliases.get(symbol, []))
    terms.extend(
        [
            name,
            f"{name}期货",
            f"{name}产业链",
            f"{name}供需",
            f"{name}供应与需求",
            f"{name}库存",
            f"{name}成本利润",
            f"{name}价格走势",
        ]
    )
    unique = list(dict.fromkeys(term for term in terms if term))
    target = "、".join(unique)
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {target}"


def split_sentences(markdown: str) -> list[dict]:
    """Split cleaned Markdown into ordered sentences with nearest heading context."""
    rows: list[dict] = []
    heading = ""
    order = 0
    for raw_line in str(markdown).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("|"):
            continue
        if set(line) <= set("|-: "):
            continue
        if line.startswith("#"):
            heading = re.sub(r"^#+\s*", "", line).strip()
            continue
        line = re.sub(r"^[-*+]\s*", "", line)
        for sentence in re.split(r"(?<=[。！？!?；;])\s*", line):
            sentence = sentence.strip()
            if len(sentence) < 8 or BOILERPLATE_RE.search(sentence):
                continue
            embed_text = f"{heading}：{sentence}" if heading else sentence
            rows.append(
                {
                    "order": order,
                    "heading": heading,
                    "sentence": sentence,
                    "embed_text": embed_text,
                }
            )
            order += 1
    return rows


def last_token_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    if bool((attention_mask[:, -1] == 1).all()):
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(
        last_hidden_state.shape[0], device=last_hidden_state.device
    )
    return last_hidden_state[batch_indices, sequence_lengths]


@torch.inference_mode()
def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    desc: str,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    device = next(model.parameters()).device
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            hidden = model(**batch).last_hidden_state
        pooled = last_token_pool(hidden, batch["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
        vectors.append(pooled.cpu().numpy().astype(np.float16))
    return np.concatenate(vectors, axis=0)


def lexical_match(text: str, aliases: list[str]) -> bool:
    """Direct mention is a strong positive; ignore ambiguous ASCII one-letter codes."""
    for alias in aliases:
        if len(alias) == 1 and alias.isascii():
            continue
        if alias.lower() in text.lower():
            return True
    return False


def selected_lines(sentences: list[dict], selected: np.ndarray) -> list[str]:
    parts: list[str] = []
    previous_heading = None
    for item, keep in zip(sentences, selected):
        if not keep:
            continue
        heading = item["heading"]
        if heading and heading != previous_heading:
            parts.append(f"## {heading}")
            previous_heading = heading
        parts.append(item["sentence"])
    return parts


def pack_lines_with_prefix(
    lines: list[str], prefix: str, tokenizer, max_length: int
) -> list[tuple[str, int]]:
    """Greedily pack whole selected sentences while preserving original text."""

    def token_count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    chunks: list[tuple[str, int]] = []
    current: list[str] = []
    for line in lines:
        candidate = prefix + "\n".join([*current, line])
        count = token_count(candidate)
        if count <= max_length:
            current.append(line)
            continue
        if current:
            text = prefix + "\n".join(current)
            chunks.append((text, token_count(text)))
            current = []
        if token_count(prefix + line) <= max_length:
            current = [line]
            continue

        # Rare OCR run-on sentence: preserve characters and split only this sentence.
        remaining = line
        while remaining:
            low, high = 1, len(remaining)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                if token_count(prefix + remaining[:middle]) <= max_length:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise ValueError("unable to fit sentence characters after prefix")
            text = prefix + remaining[:best]
            chunks.append((text, token_count(text)))
            remaining = remaining[best:]
    if current:
        text = prefix + "\n".join(current)
        chunks.append((text, token_count(text)))
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents", type=Path, default=HIER_ROOT / "documents.parquet")
    ap.add_argument("--links", type=Path, default=HIER_ROOT / "document_symbols.parquet")
    ap.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-score", type=float, default=0.42)
    ap.add_argument("--relative-margin", type=float, default=0.0)
    ap.add_argument("--fallback-score", type=float, default=0.34)
    ap.add_argument("--embedding-batch-size", type=int, default=128)
    ap.add_argument("--embedding-max-length", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit-docs", type=int, default=0)
    args = ap.parse_args()

    documents = pd.read_parquet(args.documents)
    documents["report_date"] = pd.to_datetime(documents["report_date"]).dt.normalize()
    if args.limit_docs:
        documents = documents.iloc[: args.limit_docs].copy()

    # Re-map from the current header and alias table. The persisted historical links
    # can be stale when symbol aliases are expanded (especially multi-variety dailies).
    available = {
        path.name.upper()
        for path in CONTRACTS_DIR.iterdir()
        if path.is_dir() and path.name.upper() not in REMOVED_SYMBOLS
    }
    link_rows = []
    for doc in documents.itertuples(index=False):
        for symbol in map_varieties(str(doc.variety_raw)):
            if symbol in available:
                link_rows.append(
                    {
                        "doc_idx": int(doc.doc_idx),
                        "doc_id": doc.doc_id,
                        "symbol": symbol,
                        "report_date": pd.Timestamp(doc.report_date),
                    }
                )
    links = pd.DataFrame(link_rows)
    if links.empty:
        raise SystemExit("no document-symbol links after current alias mapping")

    all_sentences: list[dict] = []
    document_ranges: dict[int, tuple[int, int]] = {}
    for row in tqdm(documents.itertuples(index=False), total=len(documents), desc="sentences"):
        start = len(all_sentences)
        units = split_sentences(row.text)
        for unit in units:
            unit["doc_idx"] = int(row.doc_idx)
            all_sentences.append(unit)
        document_ranges[int(row.doc_idx)] = (start, len(all_sentences))
    if not all_sentences:
        raise SystemExit("no sentences produced")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_tokenizer = AutoTokenizer.from_pretrained(
        str(args.embedding_model), trust_remote_code=True, padding_side="left"
    )
    if embedding_tokenizer.pad_token is None:
        embedding_tokenizer.pad_token = embedding_tokenizer.eos_token
    embedding_model = AutoModel.from_pretrained(
        str(args.embedding_model),
        trust_remote_code=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).eval().to(device)

    aliases = symbol_aliases()
    names = preferred_symbol_names()
    symbols = sorted(links["symbol"].unique())
    queries = [build_query(symbol, aliases, names) for symbol in symbols]
    query_vectors = encode_texts(
        queries,
        embedding_tokenizer,
        embedding_model,
        args.embedding_batch_size,
        args.embedding_max_length,
        "queries",
    ).astype(np.float32)
    query_index = {symbol: index for index, symbol in enumerate(symbols)}
    sentence_vectors = encode_texts(
        [item["embed_text"] for item in all_sentences],
        embedding_tokenizer,
        embedding_model,
        args.embedding_batch_size,
        args.embedding_max_length,
        "sentence embeddings",
    )
    del embedding_model
    torch.cuda.empty_cache()

    finance_tokenizer = AutoTokenizer.from_pretrained(
        str(WEIGHTS["finance_zh"]), trust_remote_code=True
    )
    finance_tokenizer.model_max_length = 10**9

    price_cache: dict[str, pd.DataFrame] = {}
    for symbol in tqdm(symbols, desc="load prices"):
        frame = load_symbol_ohlc(symbol)
        if not frame.empty:
            price_cache[symbol] = frame.set_index("date").sort_index()

    doc_by_idx = documents.set_index("doc_idx")
    links_by_doc = {
        int(doc_idx): group["symbol"].tolist()
        for doc_idx, group in links.groupby("doc_idx", sort=False)
    }
    rows: list[dict] = []
    filter_stats: list[dict] = []
    skipped_return = 0
    skipped_relevance = 0

    for doc_idx, doc_symbols in tqdm(links_by_doc.items(), desc="filter + chunks"):
        start, end = document_ranges.get(doc_idx, (0, 0))
        if end <= start:
            skipped_relevance += len(doc_symbols)
            continue
        units = all_sentences[start:end]
        vectors = sentence_vectors[start:end].astype(np.float32)
        linked_query_indices = [query_index[symbol] for symbol in doc_symbols]
        scores = vectors @ query_vectors[linked_query_indices].T
        best_linked_scores = scores.max(axis=1)
        doc = doc_by_idx.loc[doc_idx]

        for column, symbol in enumerate(doc_symbols):
            symbol_scores = scores[:, column]
            lexical = np.asarray(
                [
                    lexical_match(item["embed_text"], aliases.get(symbol, []))
                    for item in units
                ],
                dtype=bool,
            )
            semantic = (symbol_scores >= args.min_score) & (
                symbol_scores >= best_linked_scores - args.relative_margin
            )
            selected = lexical | semantic
            if not selected.any():
                best = int(np.argmax(symbol_scores))
                if float(symbol_scores[best]) >= args.fallback_score:
                    selected[best] = True
            if not selected.any():
                skipped_relevance += 1
                continue

            price = price_cache.get(symbol)
            if price is None:
                skipped_return += 1
                continue
            trade_date, forward_ret = forward_horizon_ret(
                price, pd.Timestamp(doc.report_date), horizon=1
            )
            if trade_date is None or not pd.notna(forward_ret):
                skipped_return += 1
                continue

            variety_name = names.get(symbol, symbol)
            prefix = f"品种:{variety_name}\n"
            chunks = pack_lines_with_prefix(
                selected_lines(units, selected),
                prefix,
                finance_tokenizer,
                max_length=args.max_length,
            )
            n_chunks = len(chunks)
            filter_stats.append(
                {
                    "doc_id": doc.doc_id,
                    "symbol": symbol,
                    "total_sentences": len(units),
                    "selected_sentences": int(selected.sum()),
                    "selected_ratio": float(selected.mean()),
                    "max_score": float(symbol_scores.max()),
                    "mean_selected_score": float(symbol_scores[selected].mean()),
                    "lexical_sentences": int(lexical.sum()),
                }
            )
            for chunk_idx, (sample_text, token_count) in enumerate(chunks):
                rows.append(
                    {
                        "doc_idx": int(doc_idx),
                        "doc_id": doc.doc_id,
                        "path": doc.path,
                        "institution": doc.institution,
                        "report_date": pd.Timestamp(doc.report_date),
                        "variety_raw": doc.variety_raw,
                        "symbol": symbol,
                        "variety_name": variety_name,
                        "chunk_idx": chunk_idx,
                        "n_chunks": n_chunks,
                        "token_count": token_count,
                        "total_sentences": len(units),
                        "selected_sentences": int(selected.sum()),
                        "selected_ratio": float(selected.mean()),
                        "max_relevance_score": float(symbol_scores.max()),
                        "text": sample_text,
                        "trade_date": pd.Timestamp(trade_date).normalize(),
                        "log_ret_1d": float(forward_ret),
                        "label": ret_to_label(float(forward_ret), horizon=1),
                    }
                )

    out = pd.DataFrame(rows)
    stats = pd.DataFrame(filter_stats)
    if out.empty:
        raise SystemExit("no relevant chunk samples produced")
    if int(out["token_count"].max()) > args.max_length:
        raise AssertionError("chunk exceeds max_length")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    out.drop(columns=["text"]).to_csv(args.out.with_suffix(".csv"), index=False)
    stats_path = args.out.with_name(f"{args.out.stem}_filter_stats.parquet")
    stats.to_parquet(stats_path, index=False)
    stats.to_csv(stats_path.with_suffix(".csv"), index=False)
    print(
        f"[done] rows={len(out)} doc_symbol={out.groupby(['doc_id', 'symbol']).ngroups} "
        f"docs={out['doc_id'].nunique()} symbols={out['symbol'].nunique()} "
        f"max_tokens={out['token_count'].max()} skipped_return={skipped_return} "
        f"skipped_relevance={skipped_relevance} -> {args.out}"
    )
    print("labels", out.groupby("label").size().to_dict())
    print(
        "selected sentences",
        stats["selected_sentences"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict(),
    )
    print(
        "selected ratio",
        stats["selected_ratio"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict(),
    )
    print(
        "chunks/doc_symbol",
        out.groupby(["doc_id", "symbol"]).size().describe(
            percentiles=[0.5, 0.9, 0.99]
        ).to_dict(),
    )


if __name__ == "__main__":
    main()
