#!/usr/bin/env python3
"""PPO: expert features -> discrete flat/long/short."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from gymnasium import spaces
from stable_baselines3 import PPO
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtest"))
from shared import (
    BACKTEST_END,
    BACKTEST_START,
    CONTRACTS_DIR,
    FEAT_COLS,
    GAF_FEATURES_DIR,
    KRONOS_REPO,
    MODELS_DIR,
    N_CLASSES,
    TRAIN_END,
    WINDOW,
    XGB_FEATURES_CSV,
)
from backtest_kronos import pick_main_contract
from train_gaf import CLIP, GAF_CNN, generate_gaf_image

PPO_FEATURES_DIR = Path(__file__).resolve().parents[2] / "data/features/ppo"
PPO_SENT_FEATURES_DIR = Path(__file__).resolve().parents[2] / "data/features/ppo_sent"
SENT_DAILY_PATH = Path(__file__).resolve().parents[2] / "data/features/sentiment_daily.parquet"
GAF_MODELS = ("gaf_cnn_nonferrous", "gaf_cnn_ferrous", "gaf_cnn_oilseeds")
BASE_OBS_NAMES = (
    ["kronos_pred_ret"]
    + [f"{n}_pred_ret" for n in GAF_MODELS]
    + [f"xgb_p{i}" for i in range(N_CLASSES)]
    + ["mom_5d", "rev_5d", "ret_1d_lag"]
)
SENT_OBS_NAMES = ("sent_pos", "sent_p0", "sent_p1", "sent_p2")
# default (original) schema
OBS_NAMES = list(BASE_OBS_NAMES)
OBS_DIM = len(OBS_NAMES)
N_FEAT = len(FEAT_COLS)
X_COLS = [f"x_{i}" for i in range(WINDOW * N_FEAT)]
WD_COLS = [f"wd_{i}" for i in range(WINDOW)]
POS_MAP = {0: 0, 1: 1, 2: -1}


def obs_names(
    with_sentiment: bool = False,
    sent_names: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    if not with_sentiment:
        return list(BASE_OBS_NAMES)
    extra = list(sent_names) if sent_names is not None else list(SENT_OBS_NAMES)
    return list(BASE_OBS_NAMES) + extra


def obs_matrix(
    df: pd.DataFrame,
    with_sentiment: bool | None = None,
    sent_names: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray:
    names_guess = list(sent_names) if sent_names is not None else list(SENT_OBS_NAMES)
    if with_sentiment is None:
        with_sentiment = all(c in df.columns for c in names_guess)
    names = obs_names(with_sentiment, sent_names=sent_names)
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"missing obs columns {missing}; run prepare / build_sentiment_daily")
    return df[names].to_numpy(dtype=np.float32)


def _closes(row: pd.Series, cache: dict[str, pd.DataFrame]) -> np.ndarray:
    key = str(row["contract_file"])
    if key not in cache:
        p = CONTRACTS_DIR / row["symbol"] / f"{key}.csv"
        cache[key] = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    seg = cache[key]
    i = int(row["seg_idx"])
    return seg.iloc[i - WINDOW + 1 : i + 1]["close"].astype(float).to_numpy()


class ExpertStack:
    def __init__(self, use_kronos: bool = True, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gaf: dict[str, GAF_CNN] = {}
        for name in GAF_MODELS:
            m = GAF_CNN().to(self.device)
            m.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location=self.device))
            m.eval()
            self.gaf[name] = m
        self.xgb = xgb.Booster()
        self.xgb.load_model(str(MODELS_DIR / "xgb_ohlcv.json"))
        meta = json.loads((MODELS_DIR / "xgb_codes.json").read_text(encoding="utf-8"))
        self.code_to_id = {c: i for i, c in enumerate(meta["codes"])}
        self.kronos = None
        if use_kronos:
            if str(KRONOS_REPO) not in sys.path:
                sys.path.insert(0, str(KRONOS_REPO))
            from backtest_kronos import Kronos1d

            self.kronos = Kronos1d(device=self.device)
        self._contracts: dict[str, pd.DataFrame] = {}
        self._closes: dict[str, pd.DataFrame] = {}
        self._sent_daily: pd.DataFrame | None = None

    def _load_sentiment(self) -> pd.DataFrame:
        if self._sent_daily is None:
            if not SENT_DAILY_PATH.exists():
                raise FileNotFoundError(
                    f"missing {SENT_DAILY_PATH}; run code/build_feature/build_sentiment_daily.py"
                )
            self._sent_daily = pd.read_parquet(SENT_DAILY_PATH)
            self._sent_daily["date"] = pd.to_datetime(self._sent_daily["date"]).dt.normalize()
            self._sent_daily["symbol"] = self._sent_daily["symbol"].astype(str).str.upper()
        return self._sent_daily

    def attach_sentiment(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Left-join daily ensemble sentiment; missing days -> flat / uniform probs."""
        if df.empty:
            return df
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        sent = self._load_sentiment()
        part = sent[sent["symbol"] == symbol.upper()][
            ["date", "sent_pos", "sent_p0", "sent_p1", "sent_p2"]
        ]
        out = out.merge(part, on="date", how="left")
        out["sent_pos"] = out["sent_pos"].fillna(0.0)
        for c, v in (("sent_p0", 1 / 3), ("sent_p1", 1 / 3), ("sent_p2", 1 / 3)):
            out[c] = out[c].fillna(v)
        return out

    def build(
        self,
        symbol: str,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
        with_sentiment: bool = False,
    ) -> pd.DataFrame:
        main = pd.read_csv(GAF_FEATURES_DIR / f"{symbol.upper()}.csv", parse_dates=["date", *WD_COLS])
        if start is not None:
            main = main[main["date"] >= start]
        if end is not None:
            main = main[main["date"] <= end]
        main = pick_main_contract(main).reset_index(drop=True)
        if main.empty:
            return main

        xgb_df = pd.read_csv(XGB_FEATURES_CSV, parse_dates=["date"])
        xgb_df = xgb_df[xgb_df["symbol"] == symbol.upper()].drop_duplicates(["date", "code"], keep="last")
        xgb_idx = xgb_df.set_index(["date", "code"], drop=False)
        fcols = [c for c in xgb_df.columns if c.startswith("f_")]
        uniform = np.ones(N_CLASSES, dtype=np.float32) / N_CLASSES
        rows: list[dict] = []

        for _, row in tqdm(main.iterrows(), total=len(main), desc=f"ppo {symbol}"):
            closes = _closes(row, self._closes)
            mom = float(closes[-1] / closes[-6] - 1.0) if len(closes) >= 6 else 0.0
            ret_lag = float(np.log(closes[-1] / closes[-2])) if len(closes) >= 2 else 0.0

            vec: dict = {
                "date": row["date"],
                "code": row["code"],
                "log_ret_1d": float(row["log_ret_1d"]),
                "kronos_pred_ret": self._kronos(row, symbol),
                "mom_5d": mom,
                "rev_5d": -mom,
                "ret_1d_lag": ret_lag,
            }
            for name, model in self.gaf.items():
                gaf = generate_gaf_image(closes)
                x = torch.tensor(gaf, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    vec[f"{name}_pred_ret"] = float(model(x).item()) * CLIP

            key = (row["date"], row["code"])
            if key in xgb_idx.index:
                xr = xgb_idx.loc[key]
                if isinstance(xr, pd.DataFrame):
                    xr = xr.iloc[-1]
                cid = self.code_to_id.get(str(xr["code"]), -1)
                xv = np.concatenate([xr[fcols].to_numpy(dtype=np.float32), [cid]], dtype=np.float32).reshape(1, -1)
                probs = self.xgb.predict(xgb.DMatrix(xv)).reshape(-1)
            else:
                probs = uniform
            for i in range(N_CLASSES):
                vec[f"xgb_p{i}"] = float(probs[i])
            rows.append(vec)

        out = pd.DataFrame(rows)
        if with_sentiment:
            out = self.attach_sentiment(out, symbol)
        return out

    def _kronos(self, row: pd.Series, symbol: str) -> float:
        if self.kronos is None:
            return 0.0
        key = str(row["contract_file"])
        if key not in self._contracts:
            p = CONTRACTS_DIR / symbol / f"{key}.csv"
            df = pd.read_csv(p, parse_dates=["date"])
            df["amount"] = df["turnover"].astype(float)
            self._contracts[key] = df.sort_values("date").reset_index(drop=True)
        seg = self._contracts[key]
        i = int(row["seg_idx"])
        if i + 1 >= len(seg):
            return 0.0
        raw = seg.iloc[i - WINDOW + 1 : i + 1][FEAT_COLS].to_numpy(dtype=np.float64)
        norm_x = row[X_COLS].to_numpy(dtype=np.float32).reshape(WINDOW, N_FEAT)
        return self.kronos.pred_ret_1d(norm_x, pd.to_datetime(row[WD_COLS]), pd.Timestamp(seg.iloc[i + 1]["date"]), raw)


class DirectTradingEnv(gym.Env):
    def __init__(
        self,
        features: pd.DataFrame,
        reward_scale: float = 100.0,
        with_sentiment: bool = False,
        sent_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__()
        self.obs_mat = obs_matrix(features, with_sentiment=with_sentiment, sent_names=sent_names)
        self.log_rets = features["log_ret_1d"].to_numpy(dtype=np.float32)
        self.reward_scale = reward_scale
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.obs_mat.shape[1],), dtype=np.float32
        )
        self._t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(len(self.log_rets) - 2, 0)
        self._t = int(self.np_random.integers(0, max_start + 1)) if max_start > 0 else 0
        return self.obs_mat[self._t], {}

    def step(self, action: int):
        pos = POS_MAP[int(action)]
        log_r = float(self.log_rets[self._t])
        self._t += 1
        terminated = self._t >= len(self.log_rets)
        obs = self.obs_mat[self._t] if not terminated else self.obs_mat[-1]
        return obs, pos * log_r * self.reward_scale, terminated, False, {}


def prepare_features(
    symbol: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    use_kronos: bool,
    with_sentiment: bool = False,
    out_dir: Path | None = None,
) -> Path:
    df = ExpertStack(use_kronos=use_kronos).build(symbol, start, end, with_sentiment=with_sentiment)
    if df.empty:
        raise SystemExit(f"no features for {symbol}")
    feat_dir = out_dir or (PPO_SENT_FEATURES_DIR if with_sentiment else PPO_FEATURES_DIR)
    feat_dir.mkdir(parents=True, exist_ok=True)
    out = feat_dir / f"{symbol.lower()}.csv"
    df.to_csv(out, index=False)
    print(f"saved {out} rows={len(df)} sent={with_sentiment}")
    return out


def train_ppo(
    symbol: str,
    timesteps: int = 100_000,
    end: pd.Timestamp | None = TRAIN_END,
    start: pd.Timestamp | None = None,
    verbose: int = 1,
    with_sentiment: bool = False,
    features_dir: Path | None = None,
    model_name: str | None = None,
    sent_names: tuple[str, ...] | list[str] | None = None,
) -> Path:
    feat_dir = features_dir or (PPO_SENT_FEATURES_DIR if with_sentiment else PPO_FEATURES_DIR)
    path = feat_dir / f"{symbol.lower()}.csv"
    if not path.exists():
        prepare_features(
            symbol,
            start=start,
            end=end,
            use_kronos=True,
            with_sentiment=with_sentiment,
            out_dir=feat_dir,
        )
    df = pd.read_csv(path, parse_dates=["date"])
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    df = df.reset_index(drop=True)
    if len(df) < 50:
        raise SystemExit(f"too few rows ({len(df)})")

    env = DirectTradingEnv(df, with_sentiment=with_sentiment, sent_names=sent_names)
    model = PPO("MlpPolicy", env, verbose=verbose, learning_rate=3e-4, device="cpu")
    print(f"train PPO {symbol}: rows={len(df)} obs_dim={env.observation_space.shape[0]} sent={with_sentiment}")
    model.learn(total_timesteps=timesteps)
    name = model_name or (f"ppo_{symbol.lower()}_sent.zip" if with_sentiment else f"ppo_{symbol.lower()}.zip")
    out = MODELS_DIR / name
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SN")
    parser.add_argument("--prepare", action="store_true", help="only build feature CSV")
    parser.add_argument("--skip-kronos", action="store_true")
    parser.add_argument("--with-sentiment", action="store_true", help="append news ensemble obs")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--end", default=str(TRAIN_END.date()))
    parser.add_argument("--start", default=None)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    end = pd.Timestamp(args.end)
    start = pd.Timestamp(args.start) if args.start else None

    if args.prepare:
        prepare_features(
            symbol,
            start,
            end,
            use_kronos=not args.skip_kronos,
            with_sentiment=args.with_sentiment,
        )
        return
    train_ppo(
        symbol,
        timesteps=args.timesteps,
        end=end,
        start=start,
        with_sentiment=args.with_sentiment,
    )


if __name__ == "__main__":
    main()