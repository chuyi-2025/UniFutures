#!/usr/bin/env python3
"""GAF-CNN backtest: main contract, 3-day hold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from shared import BACKTEST_END, BACKTEST_START, CONTRACTS_DIR, GAF_FEATURES_DIR, GAF_HORIZON, MODELS_DIR, RESULTS_DIR, WINDOW
from backtest_kronos import INIT_CAP, TH, pick_main_contract
from execution import HoldExecutor
from train_gaf import CLIP, GAF_CNN, generate_gaf_image


def signal_from_pred(pred_ret: float) -> int:
    if pred_ret > TH:
        return 1
    if pred_ret < -TH:
        return -1
    return 0


def closes_for(row: pd.Series, cache: dict[str, pd.DataFrame]) -> np.ndarray:
    key = str(row["contract_file"])
    if key not in cache:
        p = CONTRACTS_DIR / row["symbol"] / f"{key}.csv"
        cache[key] = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    seg = cache[key]
    i = int(row["seg_idx"])
    return seg.iloc[i - WINDOW + 1 : i + 1]["close"].astype(float).to_numpy()


def backtest(symbol: str, model_path: Path) -> pd.DataFrame:
    path = GAF_FEATURES_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"])
    df = df[(df["date"] >= BACKTEST_START) & (df["date"] <= BACKTEST_END)]
    df = pick_main_contract(df)
    if df.empty:
        return pd.DataFrame()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GAF_CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cache: dict[str, pd.DataFrame] = {}
    executor = HoldExecutor(GAF_HORIZON)
    cap = INIT_CAP
    rows: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=symbol):
        gaf = generate_gaf_image(closes_for(row, cache))
        x = torch.tensor(gaf, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_ret = float(model(x).item()) * CLIP

        signal = signal_from_pred(pred_ret)
        log_r = float(row["log_ret_1d"])
        strat_ret, pos = executor.step(signal, log_r)
        cap *= float(np.exp(strat_ret))
        rows.append(
            {
                "date": row["date"],
                "code": row["code"],
                "pred_ret": pred_ret,
                "signal": signal,
                "position": pos,
                "log_ret_1d": log_r,
                "capital": cap,
            }
        )

    return pd.DataFrame(rows)


def save(symbol: str, model_path: Path, part: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (RESULTS_DIR / "gaf" / symbol.lower())
    out_dir.mkdir(parents=True, exist_ok=True)
    final = float(part["capital"].iloc[-1])
    pd.DataFrame([{"symbol": symbol, "final_capital": final, "return": final / INIT_CAP - 1.0, "days": len(part)}]).to_csv(
        out_dir / "summary.csv", index=False
    )
    part.to_csv(out_dir / "daily.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(part["date"], part["capital"] / 1e4, linewidth=1.2)
    ax.axhline(INIT_CAP / 1e4, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Capital (10k CNY)")
    default_model = MODELS_DIR / f"gaf_cnn_{symbol.lower()}.pt"
    title = f"GAF-CNN {symbol} (main contract, {GAF_HORIZON}d hold)"
    if model_path.resolve() != default_model.resolve():
        title += f" [model={model_path.stem}]"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "equity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SN", help="symbol to backtest")
    parser.add_argument("--model-path", default=None, help="model checkpoint (default: models/gaf_cnn_{symbol}.pt)")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory")
    args = parser.parse_args()
    symbol = args.symbol.upper()
    model_path = Path(args.model_path) if args.model_path else MODELS_DIR / f"gaf_cnn_{symbol.lower()}.pt"

    part = backtest(symbol, model_path)
    if part.empty:
        raise SystemExit(f"no backtest data for {symbol}")

    out_dir = save(symbol, model_path, part, args.out_dir)
    final = float(part["capital"].iloc[-1])
    print(f"{symbol}: final={final:,.0f} ({final/1e4:.1f} 10k CNY) -> {out_dir}")


if __name__ == "__main__":
    main()
