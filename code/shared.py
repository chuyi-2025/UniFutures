from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DIR = "/home/workspace/"
ROOT = Path(f"{DIR}/lab/UniFutures")
CONTRACTS_DIR = ROOT / "data/contracts"
XGB_FEATURES_CSV = ROOT / "data/xgb_features.csv"
GAF_FEATURES_DIR = ROOT / "data/features/gaf"
GAF_FEATURES_CSV = GAF_FEATURES_DIR / "all.csv"
SAMPLES_DIR = ROOT / "data/samples"
MODELS_DIR = ROOT / "data/models"
RESULTS_DIR = ROOT / "data/results"

FEAT_COLS = ["open", "high", "low", "close", "volume", "amount"]
WINDOW = 64
HORIZON = 5
GAF_HORIZON = 3
GAF_N_CLASSES = 6
TRAIN_END = pd.Timestamp("2023-12-31")
BACKTEST_START = pd.Timestamp("2024-01-01")
BACKTEST_END = pd.Timestamp("2026-12-31")
LABEL_BINS = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
N_CLASSES = 8

KRONOS_REPO = Path(f"{DIR}/lab/Kronos")
KRONOS_MODEL = Path(f"{DIR}/weights/Kronos-mini")
KRONOS_TOKENIZER = Path(f"{DIR}/weights/Kronos-Tokenizer-base")

# 低流动性 / 已退市品种，不参与训练与回测
REMOVED_SYMBOLS = frozenset(
    {"WR", "ZC", "RR", "RI", "JR", "LR", "WH", "PM", "FB", "BB"}
)


def ret_to_label(ret: float) -> int:
    return int(np.digitize(ret, LABEL_BINS))


def label_to_signal(label: int) -> int:
    if label == 0:
        return -1
    if label == 7:
        return 1
    return 0


def signal_from_pred_class(pred: int) -> int:
    return label_to_signal(int(pred))


def ret_to_gaf_class(ret: float) -> int:
    """6-class label on forward return: 0~1%, 1~2%, >=2%, <=-2%, -2~-1%, -1~0%."""
    if ret >= 0.02:
        return 2
    if ret >= 0.01:
        return 1
    if ret >= 0.0:
        return 0
    if ret >= -0.01:
        return 5
    if ret >= -0.02:
        return 4
    return 3


def gaf_class_to_signal(cls: int) -> float:
    if cls in (1, 2):
        return 1.0
    if cls in (3, 4):
        return -1.0
    return 0.0
