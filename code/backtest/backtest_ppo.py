#!/usr/bin/env python3
"""PPO backtest: main contract, discrete flat/long/short, 1M capital."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from shared import BACKTEST_END, BACKTEST_START, MODELS_DIR, RESULTS_DIR
from backtest_kronos import INIT_CAP
from train_ppo import (
    ExpertStack,
    PPO_FEATURES_DIR,
    PPO_SENT_FEATURES_DIR,
    POS_MAP,
    obs_matrix,
)

ACTION_NAMES = {0: "flat", 1: "long", 2: "short"}


def load_features(
    symbol: str,
    build: bool = False,
    with_sentiment: bool = False,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    start = start or BACKTEST_START
    end = end or BACKTEST_END
    feat_dir = PPO_SENT_FEATURES_DIR if with_sentiment else PPO_FEATURES_DIR
    path = feat_dir / f"{symbol.lower()}.csv"
    if not build and path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        if with_sentiment and not all(c in df.columns for c in ("sent_pos", "sent_p0", "sent_p1", "sent_p2")):
            df = ExpertStack(use_kronos=False).attach_sentiment(df, symbol)
            df = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(df) >= 10:
            return df.reset_index(drop=True)

    print(f"building PPO features for {symbol} [{start.date()} ~ {end.date()}] sent={with_sentiment}")
    return ExpertStack(use_kronos=True).build(symbol, start=start, end=end, with_sentiment=with_sentiment)


def backtest_df(
    df: pd.DataFrame,
    model_path: Path,
    with_sentiment: bool | None = None,
    sent_names: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    model = PPO.load(str(model_path))
    names_guess = list(sent_names) if sent_names is not None else ("sent_pos", "sent_p0", "sent_p1", "sent_p2")
    if with_sentiment is None:
        try:
            with_sentiment = int(model.observation_space.shape[0]) > 15
        except Exception:
            with_sentiment = all(c in df.columns for c in names_guess)
    obs = obs_matrix(df, with_sentiment=with_sentiment, sent_names=sent_names)
    if obs.shape[1] != int(model.observation_space.shape[0]):
        raise ValueError(
            f"obs dim mismatch: features={obs.shape[1]} model={model.observation_space.shape[0]} "
            f"(with_sentiment={with_sentiment})"
        )
    cap = INIT_CAP
    rows: list[dict] = []
    for i in range(len(df)):
        action, _ = model.predict(obs[i], deterministic=True)
        action = int(action)
        pos = POS_MAP[action]
        log_r = float(df.iloc[i]["log_ret_1d"])
        cap *= float(np.exp(pos * log_r))
        rows.append(
            {
                "date": df.iloc[i]["date"],
                "code": df.iloc[i]["code"],
                "action": action,
                "action_name": ACTION_NAMES[action],
                "position": pos,
                "log_ret_1d": log_r,
                "capital": cap,
            }
        )
    return pd.DataFrame(rows)


def backtest(
    symbol: str,
    model_path: Path,
    build_features: bool = False,
    with_sentiment: bool = False,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = load_features(
        symbol,
        build=build_features,
        with_sentiment=with_sentiment,
        start=start,
        end=end,
    )
    return backtest_df(df, model_path, with_sentiment=with_sentiment)


def save(symbol: str, part: pd.DataFrame, out_dir: Path | None = None, title: str | None = None) -> Path:
    out = out_dir or (RESULTS_DIR / "ppo" / symbol.lower())
    out.mkdir(parents=True, exist_ok=True)
    final = float(part["capital"].iloc[-1])
    ret = final / INIT_CAP - 1.0
    daily = part["log_ret_1d"].to_numpy() * part["position"].to_numpy()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 1e-8 else 0.0
    peak = part["capital"].cummax().to_numpy()
    max_dd = float((part["capital"].to_numpy() / peak - 1.0).min()) if len(part) else 0.0
    active = part[part["position"] != 0]
    win_rate = float((active["log_ret_1d"] * active["position"] > 0).mean()) if len(active) else 0.0

    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "final_capital": final,
                "return": ret,
                "days": len(part),
                "trade_days": int(len(active)),
                "win_rate": win_rate,
                "sharpe": sharpe,
                "max_dd": max_dd,
            }
        ]
    ).to_csv(out / "summary.csv", index=False)
    part.to_csv(out / "daily.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(part["date"], part["capital"] / 1e4, linewidth=1.2)
    ax.axhline(INIT_CAP / 1e4, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Capital (10k CNY)")
    ax.set_title(title or f"PPO {symbol} (main contract)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SN")
    parser.add_argument("--model-path", default=None, help="default: models/ppo_{symbol}.zip")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--build-features", action="store_true", help="recompute backtest features")
    parser.add_argument("--with-sentiment", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    default_name = f"ppo_{symbol.lower()}_sent.zip" if args.with_sentiment else f"ppo_{symbol.lower()}.zip"
    model_path = Path(args.model_path) if args.model_path else MODELS_DIR / default_name
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None
    part = backtest(
        symbol,
        model_path,
        build_features=args.build_features,
        with_sentiment=args.with_sentiment,
        start=start,
        end=end,
    )
    if part.empty:
        raise SystemExit(f"no backtest data for {symbol}")

    out_dir = save(symbol, part, args.out_dir)
    final = float(part["capital"].iloc[-1])
    print(f"{symbol}: final={final:,.0f} ({final/INIT_CAP*100-100:.2f}%) -> {out_dir}")


if __name__ == "__main__":
    main()
