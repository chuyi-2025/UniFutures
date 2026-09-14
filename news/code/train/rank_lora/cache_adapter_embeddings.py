#!/usr/bin/env python3
"""Cache LoRA-adapted FinSent document embeddings (hash-validated)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from config import (  # noqa: E402
    DATA_ROOT,
    EMBED_ROOT,
    FINSENT_PATH,
    HEAD_TAIL_HALF,
    HIER_DATA,
    MAX_DOC_TOKENS,
    RUN_ROOT,
)
from dataset import tokenize_head_tail  # noqa: E402
from model import last_token_pool, load_base_encoder, load_tokenizer  # noqa: E402


@torch.inference_mode()
def encode_docs(
    model,
    tokenizer,
    docs: pd.DataFrame,
    max_length: int,
    half: int,
    device: torch.device,
) -> tuple[np.ndarray, list[int]]:
    vecs = []
    truncated = []
    for row in tqdm(docs.itertuples(index=False), total=len(docs), desc="cache embeds"):
        enc = tokenize_head_tail(tokenizer, str(row.text), max_length, half)
        truncated.append(1 if len(enc["input_ids"]) >= max_length - 1 else 0)
        input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
        attn = torch.tensor([enc["attention_mask"]], dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            pooled = last_token_pool(out.last_hidden_state, attn)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
        vecs.append(pooled.cpu().numpy().astype(np.float16))
    return np.concatenate(vecs, axis=0), truncated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=HIER_DATA / "documents.parquet")
    ap.add_argument("--adapter", type=Path, default=RUN_ROOT / "encoder" / "best" / "adapter")
    ap.add_argument("--base", type=Path, default=FINSENT_PATH)
    ap.add_argument("--out-dir", type=Path, default=EMBED_ROOT)
    ap.add_argument("--max-length", type=int, default=MAX_DOC_TOKENS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", type=str, default="finsent_lora")
    args = ap.parse_args()

    if not args.adapter.is_dir():
        raise SystemExit(f"missing adapter {args.adapter}")

    docs = pd.read_parquet(args.docs).sort_values("doc_idx").reset_index(drop=True)
    if args.limit:
        docs = docs.iloc[: args.limit].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.base)
    base = load_base_encoder(args.base, use_lora=False)
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()
    model.to(device)

    mat, truncated = encode_docs(model, tok, docs, args.max_length, HEAD_TAIL_HALF, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npy = args.out_dir / f"{args.tag}.npy"
    meta_path = args.out_dir / f"{args.tag}.json"
    np.save(npy, mat)
    meta = {
        "tag": args.tag,
        "adapter": str(args.adapter),
        "base": str(args.base),
        "n_docs": int(len(docs)),
        "dim": int(mat.shape[1]),
        "dtype": "float16",
        "max_length": args.max_length,
        "doc_ids": docs["doc_id"].tolist(),
        "content_hashes": docs["content_hash"].tolist(),
        "doc_idxs": docs["doc_idx"].tolist(),
        "truncated": truncated,
        "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1024**3)
        if device.type == "cuda"
        else 0.0,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Validate alignment vs hierarchical manifest order
    hier_docs = pd.read_parquet(HIER_DATA / "documents.parquet", columns=["doc_idx", "doc_id", "content_hash"])
    hier_docs = hier_docs.sort_values("doc_idx").reset_index(drop=True)
    if args.limit == 0:
        assert list(docs["doc_id"]) == list(hier_docs["doc_id"]), "doc_id order mismatch"
        assert list(docs["content_hash"]) == list(hier_docs["content_hash"]), "hash mismatch"
    print(f"[saved] {npy} shape={mat.shape} peak_vram={meta['peak_vram_gb']:.2f}GB")


if __name__ == "__main__":
    main()
