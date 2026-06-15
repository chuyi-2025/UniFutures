from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import MODELS_DIR, N_CLASSES, REMOVED_SYMBOLS, TRAIN_END, XGB_FEATURES_CSV


def feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]


def train_xgb() -> Path:
    df = pd.read_csv(XGB_FEATURES_CSV, parse_dates=["date", "exit_date"])
    train = df[(df["date"] <= TRAIN_END) & (~df["symbol"].isin(REMOVED_SYMBOLS))].reset_index(drop=True)
    fcols = feat_cols(train)

    symbols = train["symbol"].value_counts()
    print(f"XGB train: {len(train)} samples, {len(symbols)} symbols")
    print(symbols.to_string())

    codes, code_ids = np.unique(train["code"].astype(str), return_inverse=True)
    x = np.concatenate([train[fcols].to_numpy(dtype=np.float32), code_ids.reshape(-1, 1).astype(np.float32)], axis=1)
    y = train["label"].to_numpy(dtype=np.int32)

    vc = pd.Series(y).value_counts().sort_index()
    print("label distribution:")
    for k, v in vc.items():
        print(f"  class {k}: {v}")

    model = xgb.train(
        {
            "objective": "multi:softprob",
            "num_class": N_CLASSES,
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
        },
        xgb.DMatrix(x, label=y),
        num_boost_round=200,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "xgb_ohlcv.json"
    model.save_model(str(out))
    meta = MODELS_DIR / "xgb_codes.json"
    meta.write_text(json.dumps({"codes": codes.tolist()}, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out} codes={len(codes)} samples={len(train)}")
    return out


if __name__ == "__main__":
    train_xgb()
