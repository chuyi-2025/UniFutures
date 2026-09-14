#!/usr/bin/env python3
"""Shared configuration for hierarchical Qwen document experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path("/home/workspace/lab/UniFutures")
NEWS_ROOT = ROOT / "news"
WEIGHTS_ROOT = Path("/home/workspace/weights")
OCR_ROOT = NEWS_ROOT / "result" / "unlimited_ocr_by_deepseek"
DATA_ROOT = NEWS_ROOT / "data" / "sentiment" / "hierarchical"
EMBED_ROOT = DATA_ROOT / "doc_embeddings"
RESULT_ROOT = NEWS_ROOT / "data" / "results" / "hierarchical_ablation"
FINANCE_BERT = WEIGHTS_ROOT / "finance-sentiment-zh-base"

MAX_DOC_TOKENS = 32_768
HEAD_TAIL_HALF = 16_384
MAX_WINDOW_DOCS = 256
LOOKBACK_DAYS = 60
TRAIN_START = "2024-03-01"
TRAIN_END = "2024-12-31"
VAL_START = "2025-01-02"
VAL_END = "2025-06-30"
OOS_START = "2025-07-01"
OOS_END = "2026-12-31"
PURGE_DAYS = 1


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    path: Path
    hidden_size: int
    middle_layer: int | None  # 1-indexed transformer block (L14 / L18)
    official_embedding: bool = False


ENCODERS = {
    "emb": EncoderSpec("emb", WEIGHTS_ROOT / "Qwen3-Embedding-0.6B", 1024, None, True),
    "finsent": EncoderSpec(
        "finsent", WEIGHTS_ROOT / "financial-sentiment-Qwen3-0.6B", 1024, None, True
    ),
    "qwen06": EncoderSpec("qwen06", WEIGHTS_ROOT / "Qwen3-0.6B", 1024, 14),
    "wiro": EncoderSpec("wiro", WEIGHTS_ROOT / "WiroAI-Finance-Qwen-1.5B", 1536, 14),
    "fin8b": EncoderSpec("fin8b", WEIGHTS_ROOT / "Qwen-Open-Finance-R-8B", 4096, 18),
}

# Pooling candidates used to pick H21 "best per encoder" on the val window.
ENCODER_POOL_CANDIDATES = {
    "emb": ["emb__final_last"],
    "finsent": ["finsent__final_last"],
    "qwen06": ["qwen06__final_last", "qwen06__final_mean", "qwen06__middle_last"],
    "wiro": ["wiro__final_last", "wiro__final_mean", "wiro__middle_last"],
    "fin8b": ["fin8b__final_last", "fin8b__final_mean", "fin8b__middle_last"],
}


@dataclass(frozen=True)
class Scheme:
    key: str
    streams: tuple[str, ...] = ()
    # concat | gate | best_per_encoder
    fusion: str = "concat"
    # finance_zh | random
    bert_init: str = "finance_zh"
    use_age: bool = True
    # if True, resolve streams from val-best pooling per alive encoder
    resolve_best_pools: bool = False


def _s(*streams: str, **kw) -> Scheme:
    key = kw.pop("key")
    return Scheme(key=key, streams=streams, **kw)


SCHEMES: dict[str, Scheme] = {
    "H01": _s("emb__final_last", key="H01"),
    "H02": _s("emb__final_last_mrl512", key="H02"),
    "H03": _s("finsent__final_last", key="H03"),
    "H04": _s("qwen06__final_last", key="H04"),
    "H05": _s("qwen06__final_mean", key="H05"),
    "H06": _s("qwen06__middle_last", key="H06"),
    "H07": _s("wiro__final_last", key="H07"),
    "H08": _s("wiro__final_mean", key="H08"),
    "H09": _s("wiro__middle_last", key="H09"),
    "H10": _s("fin8b__final_last", key="H10"),
    "H11": _s("fin8b__final_mean", key="H11"),
    "H12": _s("fin8b__middle_last", key="H12"),
    "H13": _s("emb__final_last", "finsent__final_last", key="H13"),
    "H14": _s("emb__final_last", "qwen06__final_last", key="H14"),
    "H15": _s("emb__final_last", "wiro__final_last", key="H15"),
    "H16": _s("emb__final_last", "fin8b__final_last", key="H16"),
    "H17": _s("finsent__final_last", "fin8b__final_last", key="H17"),
    "H18": _s("qwen06__final_last", "wiro__final_last", key="H18"),
    "H19": _s("wiro__final_last", "fin8b__final_last", key="H19"),
    "H20": _s(
        "emb__final_last",
        "finsent__final_last",
        "qwen06__final_last",
        "wiro__final_last",
        "fin8b__final_last",
        key="H20",
    ),
    # H21: val-best pooling per encoder, concat
    "H21": Scheme("H21", fusion="concat", resolve_best_pools=True),
    # H22: same H21 vectors, per-encoder proj + softmax gate
    "H22": Scheme("H22", fusion="gate", resolve_best_pools=True),
    # H23: H21 vectors + randomly initialized BERT
    "H23": Scheme(
        "H23", fusion="concat", resolve_best_pools=True, bert_init="random"
    ),
    # H24: H21 vectors + finance_zh, no age embedding
    "H24": Scheme(
        "H24", fusion="concat", resolve_best_pools=True, use_age=False
    ),
}


def required_streams(encoder: str) -> list[str]:
    if encoder in {"emb", "finsent"}:
        suffixes = ["final_last"]
        if encoder == "emb":
            suffixes.append("final_last_mrl512")
    else:
        suffixes = ["final_last", "final_mean", "middle_last"]
    return [f"{encoder}__{suffix}" for suffix in suffixes]


def scheme_needs_encoder(scheme: Scheme, encoder: str) -> bool:
    if scheme.resolve_best_pools:
        return True
    return any(s.startswith(encoder + "__") for s in scheme.streams)
