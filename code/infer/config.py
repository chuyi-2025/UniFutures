"""Infer pipeline config: 9 symbols, last 1 year backtest."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

ROOT = Path("/home/workspace/lab/UniFutures")
CODE_DIR = ROOT / "code"
INFER_DIR = CODE_DIR / "infer"

CONTRACTS_DIR = ROOT / "data/contracts"
SRC_CONTRACTS_DIR = Path("/home/workspace/lab/Tree-Stock/futures/data/all_contracts")
UPDATE_SCRIPT = Path("/home/workspace/lab/Tree-Stock/futures/data/update_all_contracts_from_akshare.py")
VENV_ACTIVATE = Path("/home/env/futures/bin/activate")

MODELS_DIR = ROOT / "data/models"
GAF_FEATURES_DIR = ROOT / "data/infer/features/gaf"
XGB_FEATURES_CSV = ROOT / "data/infer/features/xgb.csv"
PPO_FEATURES_DIR = ROOT / "data/infer/features/ppo"
RESULTS_DIR = ROOT / "data/infer/results"
DAILY_DIR = RESULTS_DIR / "daily"
DAILY_PPO_DIR = DAILY_DIR / "ppo"
DAILY_LINEAR_DIR = DAILY_DIR / "linear"

SYMBOLS = ["SN", "SS", "CF", "JD", "IC", "AU", "AG", "LH", "RU"]
SYMBOL_CN = {
    "SN": "锡",
    "SS": "不锈钢",
    "CF": "棉花",
    "JD": "鸡蛋",
    "IC": "中证500",
    "AU": "黄金",
    "AG": "白银",
    "LH": "生猪",
    "RU": "橡胶",
}

INFER_END = date.today()
INFER_START = INFER_END - timedelta(days=365)

# per-symbol GAF checkpoint (category model when no dedicated one)
GAF_MODEL = {
    "SN": MODELS_DIR / "gaf_cnn_sn.pt",
    "SS": MODELS_DIR / "gaf_cnn_ferrous.pt",
    "CF": MODELS_DIR / "gaf_cnn_softs.pt",
    "JD": MODELS_DIR / "gaf_cnn_grains.pt",
    "IC": MODELS_DIR / "gaf_cnn_chemical.pt",
    "AU": MODELS_DIR / "gaf_cnn_au.pt",
    "AG": MODELS_DIR / "gaf_cnn_au.pt",
    "LH": MODELS_DIR / "gaf_cnn_lh.pt",
    "RU": MODELS_DIR / "gaf_cnn_chemical.pt",
}

PPO_MODEL = {sym: MODELS_DIR / f"ppo_{sym.lower()}.zip" for sym in SYMBOLS}
XGB_MODEL = MODELS_DIR / "xgb_ohlcv.json"
XGB_CODES = MODELS_DIR / "xgb_codes.json"

INIT_CAP = 1_000_000.0
