#!/usr/bin/env python3
"""Kronos 1d backtest per symbol on GAF normalized windows."""

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
from shared import (
    BACKTEST_END,
    BACKTEST_START,
    CONTRACTS_DIR,
    FEAT_COLS,
    GAF_FEATURES_DIR,
    KRONOS_MODEL,
    KRONOS_REPO,
    KRONOS_TOKENIZER,
    RESULTS_DIR,
    WINDOW,
)

TH = 0.001
INIT_CAP = 1_000_000.0
SAMPLE_COUNT = 5
TEMPERATURE = 1.0
N_FEAT = len(FEAT_COLS)
X_COLS = [f"x_{i}" for i in range(WINDOW * N_FEAT)]
WD_COLS = [f"wd_{i}" for i in range(WINDOW)]
CLOSE_I = FEAT_COLS.index("close")
ROLL_OFFSET = 3


def parse_code_ym(code: str) -> tuple[int, int]:
    yy = int(str(code)[-4:-2])
    mm = int(str(code)[-2:])
    return 2000 + yy, mm


def roll_threshold(dt: pd.Timestamp) -> tuple[int, int]:
    y, m = dt.year, dt.month + ROLL_OFFSET
    while m > 12:
        m -= 12
        y += 1
    return y, m


def pick_main_contract(df: pd.DataFrame) -> pd.DataFrame:
    """One row per date: first contract with (y,m) > current month + 3."""
    picks: list[pd.Series] = []
    for dt, grp in df.groupby("date", sort=True):
        thr = roll_threshold(pd.Timestamp(dt))
        best_row = None
        best_ym = None
        for _, row in grp.iterrows():
            ym = parse_code_ym(row["code"])
            if ym <= thr:
                continue
            if best_ym is None or ym < best_ym:
                best_ym = ym
                best_row = row
        if best_row is not None:
            picks.append(best_row)
    if not picks:
        return pd.DataFrame()
    return pd.DataFrame(picks).sort_values("date").reset_index(drop=True)


def step_pos(pos: int, pred_ret: float) -> int:
    if pred_ret > TH:
        return 0 if pos == -1 else 1
    if pred_ret < -TH:
        return 0 if pos == 1 else -1
    return pos


def step_pos_signal(pos: int, signal: int) -> int:
    """Discrete signal {-1,0,1} with same flip rule as step_pos."""
    if signal == 1:
        return 0 if pos == -1 else 1
    if signal == -1:
        return 0 if pos == 1 else -1
    return pos


def load_contract(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["amount"] = df["turnover"].astype(float)
    return df.sort_values("date").reset_index(drop=True)


def denorm_close(norm_close: float, raw_win: np.ndarray) -> float:
    mean = raw_win[:, CLOSE_I].mean()
    std = raw_win[:, CLOSE_I].std() + 1e-5
    return float(norm_close * std + mean)


class Kronos1d:
    def __init__(
        self,
        device: str | None = None,
        sample_count: int = SAMPLE_COUNT,
        temperature: float = TEMPERATURE,
    ) -> None:
        if str(KRONOS_REPO) not in sys.path:
            sys.path.insert(0, str(KRONOS_REPO))
        from model import Kronos, KronosPredictor, KronosTokenizer
        from model.kronos import calc_time_stamps

        self.calc_time_stamps = calc_time_stamps
        self.sample_count = sample_count
        self.temperature = temperature
        tok = KronosTokenizer.from_pretrained(str(KRONOS_TOKENIZER))
        model = Kronos.from_pretrained(str(KRONOS_MODEL))
        self.pred = KronosPredictor(model, tok, device=device, max_context=512, clip=5.0)
        torch.manual_seed(42)
        np.random.seed(42)

    def pred_ret_1d(self, norm_x: np.ndarray, x_ts: pd.Series, y_ts: pd.Timestamp, raw_win: np.ndarray) -> float:
        x_stamp = self.calc_time_stamps(pd.Series(pd.to_datetime(x_ts))).values.astype(np.float32)
        y_stamp = self.calc_time_stamps(pd.Series([y_ts])).values.astype(np.float32)
        preds = self.pred.generate(
            norm_x[np.newaxis, ...].astype(np.float32),
            x_stamp[np.newaxis, ...],
            y_stamp[np.newaxis, ...],
            pred_len=1,
            T=self.temperature,
            top_k=0,
            top_p=0.9,
            sample_count=self.sample_count,
            verbose=False,
        )
        pred_norm_close = float(preds[0, -1, CLOSE_I])
        pred_close = denorm_close(pred_norm_close, raw_win)
        last_close = float(raw_win[-1, CLOSE_I])
        return pred_close / last_close - 1.0


def backtest_symbol(symbol: str, kronos: Kronos1d, contract_cache: dict) -> pd.DataFrame:
    path = GAF_FEATURES_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date", "exit_date", *WD_COLS])
    df = df[(df["date"] >= BACKTEST_START) & (df["date"] <= BACKTEST_END)]
    df = pick_main_contract(df)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index(drop=True)

    cap = INIT_CAP
    pos = 0
    rows: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=symbol, leave=False):
        key = row["contract_file"]
        if key not in contract_cache:
            contract_cache[key] = load_contract(CONTRACTS_DIR / symbol / f"{key}.csv")
        seg = contract_cache[key]
        i = int(row["seg_idx"])
        raw = seg.iloc[i - WINDOW + 1 : i + 1][FEAT_COLS].to_numpy(dtype=np.float64)

        norm_x = row[X_COLS].to_numpy(dtype=np.float32).reshape(WINDOW, N_FEAT)
        x_ts = pd.to_datetime(row[WD_COLS])
        y_ts = pd.Timestamp(seg.iloc[i + 1]["date"]) if i + 1 < len(seg) else pd.NaT
        if pd.isna(y_ts):
            continue

        pred_ret = kronos.pred_ret_1d(norm_x, x_ts, y_ts, raw)
        new_pos = step_pos(pos, pred_ret)
        log_r = float(row["log_ret_1d"])
        cap *= float(np.exp(new_pos * log_r))

        rows.append(
            {
                "date": row["date"],
                "symbol": symbol,
                "code": row["code"],
                "contract_file": row["contract_file"],
                "pred_ret": pred_ret,
                "position": new_pos,
                "log_ret_1d": log_r,
                "capital": cap,
            }
        )
        pos = new_pos

    return pd.DataFrame(rows)


def plot_equity(part: pd.DataFrame, symbol: str, out_png: Path) -> None:
    d = part.sort_values("date").reset_index(drop=True)
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(d["date"], d["capital"] / 1e4, linewidth=1.2, color="#1f77b4")
    ax.axhline(INIT_CAP / 1e4, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Capital (10k CNY)")
    ax.set_xlabel("Date")
    ax.set_title(f"Kronos {symbol} (main contract, m+{ROLL_OFFSET})")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_symbol(symbol: str, part: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (RESULTS_DIR / "kronos" / symbol.lower())
    out_dir.mkdir(parents=True, exist_ok=True)

    final = float(part["capital"].iloc[-1])
    summary = pd.DataFrame(
        [{"symbol": symbol, "final_capital": final, "return": final / INIT_CAP - 1.0, "days": len(part)}]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    part.to_csv(out_dir / "daily.csv", index=False)
    plot_equity(part, symbol, out_dir / "equity.png")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None, help="single symbol, default all")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory (single-symbol only)")
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    print(f"Kronos predict: sample_count={args.sample_count}, T={args.temperature}")

    symbols = [args.symbol.upper()] if args.symbol else sorted(
        p.stem for p in GAF_FEATURES_DIR.glob("*.csv") if p.name not in ("all.csv", "class_counts_by_symbol.csv")
    )

    kronos = Kronos1d(sample_count=args.sample_count, temperature=args.temperature)
    contract_cache: dict = {}

    for sym in symbols:
        part = backtest_symbol(sym, kronos, contract_cache)
        if part.empty:
            print(f"{sym}: skip (no data)")
            continue
        out_dir = args.out_dir if args.symbol and args.out_dir else None
        out_dir = save_symbol(sym, part, out_dir)
        final = float(part["capital"].iloc[-1])
        print(f"{sym}: final={final:,.0f} ({final/1e4:.1f} 10k CNY) -> {out_dir}")


if __name__ == "__main__":
    main()
