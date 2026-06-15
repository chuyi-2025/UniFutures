#!/usr/bin/env python3
"""XGB 8-class backtest: main contract, 5-day hold, 1M capital."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import BACKTEST_END, BACKTEST_START, GAF_FEATURES_DIR, HORIZON, MODELS_DIR, RESULTS_DIR, XGB_FEATURES_CSV, label_to_signal
from backtest_kronos import INIT_CAP, pick_main_contract
from execution import HoldExecutor


def feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]


def backtest(symbol: str) -> pd.DataFrame:
    gaf_path = GAF_FEATURES_DIR / f"{symbol}.csv"
    if not gaf_path.exists():
        return pd.DataFrame()

    main = pd.read_csv(gaf_path, parse_dates=["date"])
    main = main[(main["date"] >= BACKTEST_START) & (main["date"] <= BACKTEST_END)]
    main = pick_main_contract(main).reset_index(drop=True)
    if main.empty:
        return pd.DataFrame()

    xgb_df = pd.read_csv(XGB_FEATURES_CSV, parse_dates=["date", "exit_date"])
    xgb_df = xgb_df[(xgb_df["symbol"] == symbol) & (xgb_df["date"] >= BACKTEST_START) & (xgb_df["date"] <= BACKTEST_END)]
    xgb_df = xgb_df.drop_duplicates(subset=["date", "code"], keep="last")
    xgb_idx = xgb_df.set_index(["date", "code"], drop=False)

    meta = json.loads((MODELS_DIR / "xgb_codes.json").read_text(encoding="utf-8"))
    code_to_id = {c: i for i, c in enumerate(meta["codes"])}
    fcols = feat_cols(xgb_df)

    model = xgb.Booster()
    model.load_model(str(MODELS_DIR / "xgb_ohlcv.json"))

    executor = HoldExecutor(HORIZON)
    cap = INIT_CAP
    rows: list[dict] = []

    for _, row in main.iterrows():
        key = (row["date"], row["code"])
        signal = 0
        pred_class = None
        if key in xgb_idx.index:
            xrow = xgb_idx.loc[key]
            if isinstance(xrow, pd.DataFrame):
                xrow = xrow.iloc[-1]
            code_id = code_to_id.get(str(xrow["code"]), -1)
            x = np.concatenate(
                [xrow[fcols].to_numpy(dtype=np.float32), np.array([code_id], dtype=np.float32)]
            ).reshape(1, -1)
            pred_class = int(model.predict(xgb.DMatrix(x)).argmax(axis=1)[0])
            signal = label_to_signal(pred_class)

        log_r = float(row["log_ret_1d"])
        strat_ret, pos = executor.step(signal, log_r)
        cap *= float(np.exp(strat_ret))
        rows.append(
            {
                "date": row["date"],
                "code": row["code"],
                "pred_class": pred_class,
                "signal": signal,
                "position": pos,
                "log_ret_1d": log_r,
                "capital": cap,
            }
        )

    return pd.DataFrame(rows)


def save(symbol: str, part: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out = out_dir or (RESULTS_DIR / "xgb" / symbol.lower())
    out.mkdir(parents=True, exist_ok=True)
    final = float(part["capital"].iloc[-1])
    ret = final / INIT_CAP - 1.0
    pd.DataFrame([{"symbol": symbol, "final_capital": final, "return": ret, "days": len(part)}]).to_csv(
        out / "summary.csv", index=False
    )
    part.to_csv(out / "daily.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(part["date"], part["capital"] / 1e4, linewidth=1.2)
    ax.axhline(INIT_CAP / 1e4, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Capital (10k CNY)")
    ax.set_title(f"XGB {symbol} (main contract, {HORIZON}d hold)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SN", help="symbol to backtest")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory")
    args = parser.parse_args()
    symbol = args.symbol.upper()

    part = backtest(symbol)
    if part.empty:
        raise SystemExit(f"no backtest data for {symbol}")

    out_dir = save(symbol, part, args.out_dir)
    final = float(part["capital"].iloc[-1])
    print(f"{symbol}: final={final:,.0f} ({final/1e4:.1f} 10k CNY) -> {out_dir}")


if __name__ == "__main__":
    main()
