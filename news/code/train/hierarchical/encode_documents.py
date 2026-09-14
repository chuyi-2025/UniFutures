#!/usr/bin/env python3
"""BF16 document encoding — 32k fair cap, head+tail for oversize, middle via hooks."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import (  # noqa: E402
    DATA_ROOT,
    EMBED_ROOT,
    ENCODERS,
    HEAD_TAIL_HALF,
    MAX_DOC_TOKENS,
    required_streams,
)


def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padded = bool(attention_mask[:, 0].sum().item() == attention_mask.shape[0])
    if left_padded:
        return last_hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch = torch.arange(last_hidden.size(0), device=last_hidden.device)
    return last_hidden[batch, seq_lens]


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return nn.functional.normalize(x, p=2, dim=-1)


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("cannot find transformer layers on model")


def tokenize_with_head_tail(
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
    half: int,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Fair 32k: if oversize, keep first half + last half tokens (single forward)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    truncated = len(ids) > max_length
    if truncated:
        ids = ids[:half] + ids[-half:]
    # Add specials without further content truncation when possible.
    bos = [tokenizer.bos_token_id] if getattr(tokenizer, "bos_token_id", None) else []
    eos = [tokenizer.eos_token_id] if getattr(tokenizer, "eos_token_id", None) else []
    # Prefer eos-only for Qwen-style left-padded last-token pooling.
    special_prefix = bos
    special_suffix = eos
    budget = max_length - len(special_prefix) - len(special_suffix)
    if budget < 1:
        budget = max_length
        special_prefix, special_suffix = [], []
    if len(ids) > budget:
        ids = ids[:budget]
        truncated = True
    full = special_prefix + ids + special_suffix
    input_ids = torch.tensor([full], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, truncated


def load_encoder(spec):
    tok = AutoTokenizer.from_pretrained(str(spec.path), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModel.from_pretrained(
        str(spec.path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    model.to("cuda")
    return tok, model


@torch.inference_mode()
def encode_one(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
    half: int,
    middle_layer: int | None,
    want_middle: bool,
) -> dict[str, torch.Tensor | bool]:
    enc, was_truncated = tokenize_with_head_tail(tokenizer, text, max_length, half)
    enc = {k: v.to("cuda") for k, v in enc.items() if torch.is_tensor(v)}
    # ensure attention_mask
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"])

    captured: dict[str, torch.Tensor] = {}
    handle = None
    if want_middle and middle_layer is not None:
        layers = get_decoder_layers(model)
        # L14 -> layers[13]
        idx = int(middle_layer) - 1

        def _hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["middle"] = h

        handle = layers[idx].register_forward_hook(_hook)

    try:
        out = model(**enc, use_cache=False, output_hidden_states=False)
    finally:
        if handle is not None:
            handle.remove()

    hs = out.last_hidden_state
    mask = enc["attention_mask"]
    result: dict[str, torch.Tensor | bool] = {
        "final_last": last_token_pool(hs, mask),
        "final_mean": mean_pool(hs, mask),
        "was_truncated": was_truncated,
    }
    if want_middle:
        if "middle" not in captured:
            raise RuntimeError("middle-layer hook did not fire")
        result["middle_last"] = last_token_pool(captured["middle"], mask)
    return result


def abandon_encoder(key: str, reason: str) -> None:
    path = DATA_ROOT / "abandoned_encoders.json"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = {"reason": reason, "status": "abandoned"}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[abandon] {key}: {reason}")


def write_stream(out_dir: Path, stream: str, vectors: np.ndarray, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stream}.npy", vectors.astype(np.float16))
    (out_dir / f"{stream}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def encode_encoder(
    encoder_key: str,
    docs: pd.DataFrame,
    max_length: int,
    half: int,
    limit: int,
) -> None:
    spec = ENCODERS[encoder_key]
    streams = required_streams(encoder_key)
    out_dir = EMBED_ROOT
    if all((out_dir / f"{s}.npy").exists() for s in streams):
        print(f"[skip] {encoder_key}: streams already cached")
        return

    print(f"[load] {encoder_key} from {spec.path} (bf16, max_len={max_length})")
    torch.cuda.reset_peak_memory_stats()
    try:
        tokenizer, model = load_encoder(spec)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        abandon_encoder(encoder_key, "OOM on model load")
        return
    except Exception as exc:  # noqa: BLE001
        abandon_encoder(encoder_key, f"load failed: {exc}")
        traceback.print_exc()
        return

    n = len(docs) if not limit else min(limit, len(docs))
    subset = docs.iloc[:n].reset_index(drop=True)
    want_middle = any(s.endswith("middle_last") for s in streams)
    dims = {
        "final_last": spec.hidden_size,
        "final_mean": spec.hidden_size,
        "middle_last": spec.hidden_size,
        "final_last_mrl512": 512,
    }
    buffers: dict[str, list[np.ndarray]] = {s.split("__", 1)[1]: [] for s in streams}
    truncated = np.zeros(n, dtype=np.int8)
    oom_docs = 0
    failed_idxs: list[int] = []

    for i in tqdm(range(n), desc=f"encode {encoder_key}"):
        text = str(subset.at[i, "text"])
        try:
            packed = encode_one(
                model,
                tokenizer,
                text,
                max_length=max_length,
                half=half,
                middle_layer=spec.middle_layer,
                want_middle=want_middle,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            oom_docs += 1
            failed_idxs.append(i)
            for suffix in buffers:
                buffers[suffix].append(np.zeros((1, dims[suffix]), dtype=np.float16))
            truncated[i] = 1
            continue

        for suffix in buffers:
            if suffix == "final_last_mrl512":
                vec = packed["final_last"][:, :512]
            else:
                vec = packed[suffix]  # type: ignore[index]
            assert isinstance(vec, torch.Tensor)
            if spec.official_embedding and suffix.startswith("final_last"):
                vec = l2_normalize(vec.float()).to(dtype=vec.dtype)
            buffers[suffix].append(vec.float().cpu().numpy().astype(np.float16))
        truncated[i] = int(bool(packed["was_truncated"]))

    del model
    torch.cuda.empty_cache()
    peak = peak_vram_gb()

    if n > 0 and oom_docs / n > 0.2:
        abandon_encoder(encoder_key, f"OOM on >20% docs ({oom_docs}/{n})")
        return

    meta_base = {
        "encoder": encoder_key,
        "path": str(spec.path),
        "n_docs": n,
        "max_length": max_length,
        "head_tail_half": half,
        "dtype": "float16",
        "official_embedding": spec.official_embedding,
        "oom_docs": oom_docs,
        "failed_idxs": failed_idxs,
        "peak_vram_gb": peak,
        "doc_ids": subset["doc_id"].tolist(),
        "content_hashes": subset["content_hash"].tolist(),
        "truncated": truncated.tolist(),
        "middle_via": "forward_hook",
    }
    for stream in streams:
        suffix = stream.split("__", 1)[1]
        mat = np.concatenate(buffers[suffix], axis=0)
        write_stream(
            out_dir,
            stream,
            mat,
            {**meta_base, "stream": stream, "dim": int(mat.shape[1])},
        )
        print(f"[saved] {stream} shape={mat.shape} peak_vram={peak:.2f}GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=DATA_ROOT / "documents.parquet")
    ap.add_argument(
        "--encoder",
        nargs="+",
        default=list(ENCODERS.keys()),
        choices=list(ENCODERS.keys()),
    )
    ap.add_argument("--max-length", type=int, default=MAX_DOC_TOKENS)
    ap.add_argument("--head-tail-half", type=int, default=HEAD_TAIL_HALF)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke-percentiles", action="store_true")
    args = ap.parse_args()

    if not args.docs.exists():
        raise SystemExit(f"missing {args.docs}; run build_fulltext_manifest.py first")

    docs = pd.read_parquet(args.docs).sort_values("doc_idx").reset_index(drop=True)
    if args.smoke_percentiles:
        lengths = docs["char_count"].to_numpy()
        idxs = []
        for q in (0, 50, 99, 100):
            thr = np.percentile(lengths, q)
            idxs.append(int(np.argmin(np.abs(lengths - thr))))
        # also include true longest
        idxs.append(int(np.argmax(lengths)))
        docs = docs.iloc[sorted(set(idxs))].reset_index(drop=True)
        print(f"[smoke] {len(docs)} docs char_counts={docs['char_count'].tolist()}")

    EMBED_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    order = ["emb", "finsent", "qwen06", "wiro", "fin8b"]
    for key in [k for k in order if k in args.encoder]:
        encode_encoder(
            key,
            docs,
            max_length=args.max_length,
            half=args.head_tail_half,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
