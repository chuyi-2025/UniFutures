#!/usr/bin/env python3
"""Configuration for Financial-Sentiment Qwen rank-LoRA + L60 aggregators."""

from __future__ import annotations

from pathlib import Path

ROOT = Path("/home/workspace/lab/UniFutures")
NEWS_ROOT = ROOT / "news"
WEIGHTS_ROOT = Path("/home/workspace/weights")
OCR_ROOT = NEWS_ROOT / "result" / "unlimited_ocr_by_deepseek"
HIER_DATA = NEWS_ROOT / "data" / "sentiment" / "hierarchical"
DATA_ROOT = NEWS_ROOT / "data" / "sentiment" / "rank_lora"
EMBED_ROOT = DATA_ROOT / "doc_embeddings"
RESULT_ROOT = NEWS_ROOT / "data" / "results" / "rank_lora"
RUN_ROOT = DATA_ROOT / "runs"

FINSENT_PATH = WEIGHTS_ROOT / "financial-sentiment-Qwen3-0.6B"
FROZEN_FINSENT_NPY = HIER_DATA / "doc_embeddings" / "finsent__final_last.npy"
FROZEN_FINSENT_META = HIER_DATA / "doc_embeddings" / "finsent__final_last.json"

HORIZONS = (1, 7, 14)
MAX_DOC_TOKENS = 16_384
HEAD_TAIL_HALF = 8_192
LOOKBACK_DAYS = 60
MAX_WINDOW_DOCS = 256
HIDDEN_SIZE = 1024

# Temporal splits (trade_date). Purge = max horizon sessions at boundaries.
TRAIN_START = "2024-03-01"
TRAIN_END = "2024-12-31"
VAL_START = "2025-01-02"
VAL_END = "2025-06-30"
OOS_START = "2025-07-01"
OOS_END = "2026-12-31"
PURGE_SESSIONS = 14

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

# Aggregation
EXP_HALFLIFE = 30.0  # days
TOP_PCT = 0.20
BOTTOM_PCT = 0.20
AGGREGATORS = ("exp", "linear", "bigru_pos")
SEEDS = (0, 1, 2)

# Portfolio / costs
COST_BPS = (0, 2, 5)
