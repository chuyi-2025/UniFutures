#!/usr/bin/env python3
"""Train + backtest PPO for all active symbols."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtest"))
from shared import BACKTEST_END, BACKTEST_START, MODELS_DIR, RESULTS_DIR, TRAIN_END, list_symbols
from backtest_ppo import backtest_df, save
from train_ppo import ExpertStack, PPO_FEATURES_DIR, train_ppo

LOG_PATH = RESULTS_DIR / "ppo" / "roll_log.csv"


def run_symbol(
    symbol: str,
    stack: ExpertStack,
    timesteps: int,
    skip_prepare: bool,
    skip_train: bool,
    skip_backtest: bool,
    verbose: int,
) -> dict:
    sym = symbol.upper()
    row: dict = {"symbol": sym, "status": "ok", "error": ""}

    if not skip_prepare:
        df = stack.build(sym, start=None, end=TRAIN_END)
        if len(df) < 50:
            raise ValueError(f"too few train rows ({len(df)})")
        PPO_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PPO_FEATURES_DIR / f"{sym.lower()}.csv", index=False)

    if not skip_train:
        train_ppo(sym, timesteps=timesteps, end=TRAIN_END, verbose=verbose)

    model_path = MODELS_DIR / f"ppo_{sym.lower()}.zip"
    if not skip_backtest:
        if not model_path.exists():
            raise FileNotFoundError(f"missing model {model_path}")
        bt_df = stack.build(sym, start=BACKTEST_START, end=BACKTEST_END)
        part = backtest_df(bt_df, model_path)
        if part.empty:
            raise ValueError("empty backtest")
        save(sym, part)
        summary = pd.read_csv(RESULTS_DIR / "ppo" / sym.lower() / "summary.csv")
        row.update(summary.iloc[0].to_dict())

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-kronos", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols] if args.symbols else list_symbols()
    print(f"roll PPO: {len(symbols)} symbols, timesteps={args.timesteps}")

    stack = ExpertStack(use_kronos=not args.skip_kronos)
    rows: list[dict] = []
    for i, sym in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {sym}")
        try:
            rows.append(
                run_symbol(
                    sym,
                    stack,
                    timesteps=args.timesteps,
                    skip_prepare=args.skip_prepare,
                    skip_train=args.skip_train,
                    skip_backtest=args.skip_backtest,
                    verbose=args.verbose,
                )
            )
        except Exception as e:
            print(f"  FAIL: {e}")
            rows.append({"symbol": sym, "status": "fail", "error": str(e)})

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(LOG_PATH, index=False)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\ndone: {ok}/{len(symbols)} ok -> {LOG_PATH}")


if __name__ == "__main__":
    main()
