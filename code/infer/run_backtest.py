#!/usr/bin/env python3
"""Run PPO/Kronos/XGB/GAF backtests on infer window and export summaries + signals."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

INFER_DIR = Path(__file__).resolve().parent
CODE_DIR = INFER_DIR.parent
sys.path.insert(0, str(INFER_DIR))
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "backtest"))
sys.path.insert(0, str(CODE_DIR / "train"))

import config
import shared

shared.BACKTEST_START = pd.Timestamp(config.INFER_START)
shared.BACKTEST_END = pd.Timestamp(config.INFER_END)
shared.GAF_FEATURES_DIR = config.GAF_FEATURES_DIR
shared.XGB_FEATURES_CSV = config.XGB_FEATURES_CSV

from backtest_utils import backtest_kronos, install_patches

install_patches()

import train_ppo

train_ppo.PPO_FEATURES_DIR = config.PPO_FEATURES_DIR

from backtest_gaf import backtest as backtest_gaf, save as save_gaf
from backtest_kronos import Kronos1d, save_symbol
from backtest_ppo import backtest as backtest_ppo, save as save_ppo
from backtest_xgb import backtest as backtest_xgb, save as save_xgb

MODELS = ("ppo", "kronos", "xgb", "gaf")
POS_NAME = {0: "flat", 1: "long", -1: "short"}


def metrics(part: pd.DataFrame) -> dict:
    final = float(part["capital"].iloc[-1])
    ret = final / config.INIT_CAP - 1.0
    daily = part["log_ret_1d"].to_numpy() * part["position"].to_numpy()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 1e-8 else 0.0
    peak = part["capital"].cummax().to_numpy()
    max_dd = float((part["capital"].to_numpy() / peak - 1.0).min()) if len(part) else 0.0
    return {"final_capital": final, "return": ret, "days": len(part), "sharpe": sharpe, "max_dd": max_dd}


def latest_signal(model: str, symbol: str, part: pd.DataFrame) -> dict:
    row = part.iloc[-1]
    pos = int(row["position"])
    out = {
        "model": model,
        "symbol": symbol,
        "name_cn": config.SYMBOL_CN[symbol],
        "date": str(row["date"])[:10],
        "code": row["code"],
        "position": pos,
        "position_name": POS_NAME[pos],
        "is_open": pos != 0,
        "capital": float(row["capital"]),
    }
    if model == "ppo":
        out["action"] = int(row["action"])
        out["action_name"] = row["action_name"]
    if model == "kronos":
        out["pred_ret"] = float(row["pred_ret"])
    if model in ("xgb", "gaf"):
        out["signal"] = int(row["signal"])
        if "pred_class" in row:
            out["pred_class"] = row["pred_class"]
        if "pred_ret" in row:
            out["pred_ret"] = float(row["pred_ret"])
    return out


def run_symbol(symbol: str, kronos: Kronos1d, cache: dict) -> tuple[list[dict], list[dict]]:
    sym = symbol.upper()
    summaries: list[dict] = []
    signals: list[dict] = []
    out_root = config.RESULTS_DIR

    ppo_path = config.PPO_MODEL[sym]
    ppo_part = backtest_ppo(sym, ppo_path, build_features=True)
    save_ppo(sym, ppo_part, out_root / "ppo" / sym.lower())
    summaries.append({"model": "ppo", "symbol": sym, **metrics(ppo_part)})
    signals.append(latest_signal("ppo", sym, ppo_part))

    k_part = backtest_kronos(sym, kronos, cache)
    save_symbol(sym, k_part, out_root / "kronos" / sym.lower())
    summaries.append({"model": "kronos", "symbol": sym, **metrics(k_part)})
    signals.append(latest_signal("kronos", sym, k_part))

    x_part = backtest_xgb(sym)
    save_xgb(sym, x_part, out_root / "xgb" / sym.lower())
    summaries.append({"model": "xgb", "symbol": sym, **metrics(x_part)})
    signals.append(latest_signal("xgb", sym, x_part))

    gaf_path = config.GAF_MODEL[sym]
    g_part = backtest_gaf(sym, gaf_path)
    save_gaf(sym, gaf_path, g_part, out_root / "gaf" / sym.lower())
    summaries.append({"model": "gaf", "symbol": sym, **metrics(g_part)})
    signals.append(latest_signal("gaf", sym, g_part))

    return summaries, signals


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PPO_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    all_summary: list[dict] = []
    all_signals: list[dict] = []
    kronos = Kronos1d()
    cache: dict = {}

    print(f"backtest window: {config.INFER_START} ~ {config.INFER_END}")
    for sym in config.SYMBOLS:
        print(f"=== {sym} ({config.SYMBOL_CN[sym]}) ===")
        summaries, signals = run_symbol(sym, kronos, cache)
        all_summary.extend(summaries)
        all_signals.extend(signals)

    summary_df = pd.DataFrame(all_summary)
    signals_df = pd.DataFrame(all_signals)
    daily_dir = config.DAILY_PPO_DIR
    daily_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(daily_dir / "summary.csv", index=False)
    signals_df.to_csv(daily_dir / "latest_signals.csv", index=False)

    print("\n=== summary ===")
    for model in MODELS:
        sub = summary_df[summary_df["model"] == model]
        print(f"\n[{model}]")
        for _, r in sub.iterrows():
            print(
                f"  {r['symbol']}: ret={100*r['return']:.2f}% sharpe={r['sharpe']:.2f} "
                f"max_dd={100*r['max_dd']:.2f}% days={int(r['days'])}"
            )

    print("\n=== latest signals ===")
    for _, r in signals_df.iterrows():
        open_tag = " [OPEN]" if r["is_open"] else ""
        print(f"  {r['model']:7s} {r['symbol']} {r['position_name']:5s} @ {r['date']}{open_tag}")

    print(f"\nsaved -> {daily_dir / 'summary.csv'}")
    print(f"saved -> {daily_dir / 'latest_signals.csv'}")


if __name__ == "__main__":
    main()
