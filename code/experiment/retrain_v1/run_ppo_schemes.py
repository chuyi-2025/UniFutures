#!/usr/bin/env python3
"""Train and backtest P-family PPO retrain schemes on cached expert features."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO

ROOT = Path("/home/workspace/lab/UniFutures")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import MODELS_DIR, TRAIN_END  # noqa: E402
from schemes import EXP_ROOT, RESULT_ROOT, SEED, Scheme, by_id, schemes_by_family  # noqa: E402

PPO_DIR = ROOT / "data/features/ppo"
PPO_SENT_DIR = ROOT / "data/features/ppo_sent"
LB_L7 = ROOT / "data/features/sentiment_lb_finance_zh_L7.csv"
OOS_START = pd.Timestamp("2025-07-01")
OOS_END = pd.Timestamp("2026-12-31")
INIT_CAP = 1_000_000.0
POS_MAP = {0: 0, 1: 1, 2: -1}
BASE_COLS = [
    "kronos_pred_ret",
    "gaf_cnn_nonferrous_pred_ret",
    "gaf_cnn_ferrous_pred_ret",
    "gaf_cnn_oilseeds_pred_ret",
    *[f"xgb_p{i}" for i in range(8)],
    "mom_5d",
    "rev_5d",
    "ret_1d_lag",
]
SENT_COLS = ["sent_pos", "sent_p0", "sent_p1", "sent_p2"]


def list_symbols(prefer_sent: bool = False) -> list[str]:
    d = PPO_SENT_DIR if prefer_sent and PPO_SENT_DIR.exists() else PPO_DIR
    syms = sorted(p.stem.upper() for p in d.glob("*.csv"))
    # screen universe: prefer symbols with sentiment coverage when available
    if prefer_sent and PPO_SENT_DIR.exists():
        sent = sorted(p.stem.upper() for p in PPO_SENT_DIR.glob("*.csv"))
        if sent:
            return sent
    return syms


def load_features(symbol: str, scheme: Scheme) -> pd.DataFrame:
    obs = scheme.params["obs"]
    if obs in {"sent", "compact_sent"}:
        path = PPO_SENT_DIR / f"{symbol.lower()}.csv"
    else:
        path = PPO_DIR / f"{symbol.lower()}.csv"
    if not path.exists():
        path = PPO_DIR / f"{symbol.lower()}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df["symbol"] = symbol.upper()
    start = scheme.params.get("train_start")
    end = scheme.params.get("train_end", str(TRAIN_END.date()))
    # full history kept; train/oos filtered later
    if obs == "lb_pool_L7" and LB_L7.exists():
        lb = pd.read_csv(LB_L7, parse_dates=["date"])
        lb["symbol"] = lb["symbol"].astype(str).str.upper()
        part = lb[lb["symbol"] == symbol.upper()]
        cols = [c for c in part.columns if c.startswith("sent_") or c in {"n_docs"}]
        df = df.merge(part[["date", *cols]], on="date", how="left")
        for c in cols:
            df[c] = df[c].fillna(0.0 if c != "n_docs" else 0)
    return df


def obs_cols(scheme: Scheme, df: pd.DataFrame) -> list[str]:
    obs = scheme.params["obs"]
    if obs == "no_kronos":
        return [c for c in BASE_COLS if c != "kronos_pred_ret"]
    if obs == "sent":
        return BASE_COLS + [c for c in SENT_COLS if c in df.columns]
    if obs == "compact_sent":
        cols = ["kronos_pred_ret", "gaf_cnn_nonferrous_pred_ret", "mom_5d", "ret_1d_lag"]
        cols += [c for c in SENT_COLS if c in df.columns]
        return cols
    if obs == "lb_pool_L7":
        extra = [c for c in df.columns if c.startswith("sent_") or c == "n_docs"]
        return BASE_COLS + extra
    if obs == "hold_state":
        return BASE_COLS  # hold state appended dynamically in env
    return BASE_COLS


class RetrainEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, scheme: Scheme):
        super().__init__()
        self.scheme = scheme
        self.df = df.reset_index(drop=True)
        self.cols = [c for c in obs_cols(scheme, df) if c in df.columns]
        self.base = self.df[self.cols].to_numpy(dtype=np.float32)
        self.log_rets = self.df["log_ret_1d"].fillna(0.0).to_numpy(dtype=np.float32)
        # precompute H3 forward reward
        self.fwd3 = np.zeros_like(self.log_rets)
        for i in range(len(self.log_rets)):
            self.fwd3[i] = float(self.log_rets[i : i + 3].sum())
        self.reward_kind = scheme.params["reward"]
        self.hold_state = scheme.params["obs"] == "hold_state"
        dim = self.base.shape[1] + (2 if self.hold_state else 0)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self._t = 0
        self._prev_pos = 0.0
        self._hold_age = 0.0
        self._a_exp = 0.0
        self._b_exp = 0.0
        self._eta = 0.01

    def _obs(self):
        o = self.base[self._t]
        if self.hold_state:
            o = np.concatenate([o, np.array([self._prev_pos, self._hold_age], dtype=np.float32)])
        return o.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(len(self.log_rets) - 5, 0)
        self._t = int(self.np_random.integers(0, max_start + 1)) if max_start else 0
        self._prev_pos = 0.0
        self._hold_age = 0.0
        self._a_exp = 0.0
        self._b_exp = 0.0
        return self._obs(), {}

    def step(self, action: int):
        pos = float(POS_MAP[int(action)])
        r1 = float(self.log_rets[self._t])
        kind = self.reward_kind
        if kind == "h3":
            reward = pos * float(self.fwd3[self._t]) * 100.0
        elif kind == "cost":
            reward = pos * r1 * 100.0 - 2.0 * abs(pos - self._prev_pos)
        elif kind == "dsharpe":
            pnl = pos * r1
            self._a_exp = self._a_exp + self._eta * (pnl - self._a_exp)
            self._b_exp = self._b_exp + self._eta * (pnl * pnl - self._b_exp)
            denom = math.sqrt(max(self._b_exp - self._a_exp**2, 1e-8))
            reward = float((self._b_exp * pnl - 0.5 * self._a_exp * pnl * pnl) / (denom**3 + 1e-8))
        else:
            reward = pos * r1 * 100.0

        if pos == self._prev_pos and pos != 0:
            self._hold_age += 1.0
        else:
            self._hold_age = 1.0 if pos != 0 else 0.0
        self._prev_pos = pos
        self._t += 1
        terminated = self._t >= len(self.log_rets)
        obs = self._obs() if not terminated else self.base[-1]
        if self.hold_state and not terminated:
            obs = self._obs()
        elif self.hold_state and terminated:
            obs = np.concatenate([self.base[-1], np.array([self._prev_pos, self._hold_age], dtype=np.float32)])
        return np.asarray(obs, dtype=np.float32), float(reward), terminated, False, {}


def train_symbol(scheme: Scheme, symbol: str) -> Path | None:
    df = load_features(symbol, scheme)
    start = pd.Timestamp(scheme.params.get("train_start", "2000-01-01"))
    end = pd.Timestamp(scheme.params.get("train_end", str(TRAIN_END.date())))
    train = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    if len(train) < 80:
        return None
    env = RetrainEnv(train, scheme)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        seed=SEED,
        device="cpu",
    )
    model.learn(total_timesteps=int(scheme.params.get("timesteps", 40000)))
    out = EXP_ROOT / scheme.id / "models"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ppo_{symbol.lower()}.zip"
    model.save(str(path))
    return path


def sharpe(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or arr.std() <= 1e-12:
        return 0.0
    return float(arr.mean() / arr.std() * math.sqrt(252))


def backtest_symbol(scheme: Scheme, symbol: str) -> dict | None:
    model_path = EXP_ROOT / scheme.id / "models" / f"ppo_{symbol.lower()}.zip"
    if not model_path.exists():
        return None
    df = load_features(symbol, scheme)
    oos = df[(df["date"] >= OOS_START) & (df["date"] <= OOS_END)].reset_index(drop=True)
    if len(oos) < 30:
        return None
    model = PPO.load(str(model_path), device="cpu")
    cols = [c for c in obs_cols(scheme, oos) if c in oos.columns]
    positions = []
    prev = 0
    hold_age = 0.0
    for i in range(len(oos)):
        o = oos.iloc[i][cols].to_numpy(dtype=np.float32)
        if scheme.params["obs"] == "hold_state":
            o = np.concatenate([o, np.array([prev, hold_age], dtype=np.float32)])
        action, _ = model.predict(o, deterministic=True)
        pos = float(POS_MAP[int(action)])
        if pos == prev and pos != 0:
            hold_age += 1.0
        else:
            hold_age = 1.0 if pos != 0 else 0.0
        prev = pos
        positions.append(pos)
    positions = np.asarray(positions, dtype=float)
    log_r = oos["log_ret_1d"].fillna(0.0).to_numpy(dtype=float)
    strat = positions * log_r
    turnover = np.abs(np.diff(positions, prepend=0.0))
    rows = []
    for bps in (0, 2):
        net = strat - turnover * bps / 10000.0
        capital = INIT_CAP * np.exp(np.cumsum(net))
        peak = np.maximum.accumulate(capital)
        rows.append(
            {
                "scheme": scheme.id,
                "symbol": symbol,
                "bps": bps,
                "return": float(capital[-1] / INIT_CAP - 1.0),
                "sharpe": sharpe(net),
                "max_dd": float(np.min(capital / peak - 1.0)),
                "active_ratio": float((positions != 0).mean()),
                "days": len(oos),
            }
        )
    daily = pd.DataFrame(
        {
            "date": oos["date"],
            "symbol": symbol,
            "position": positions,
            "strategy_ret": strat,
            "turnover": turnover,
        }
    )
    return {"summary": rows, "daily": daily}


def run_scheme(scheme: Scheme, symbols: list[str] | None = None) -> None:
    syms = symbols or list_symbols(prefer_sent=scheme.params["obs"] in {"sent", "compact_sent", "lb_pool_L7"})
    # keep runtime manageable for first pass
    if len(syms) > 30:
        syms = syms[:30]
    print(f"[ppo] {scheme.id} symbols={len(syms)} axis={scheme.axis}")
    (EXP_ROOT / scheme.id).mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / scheme.id / "meta.json").write_text(
        json.dumps(
            {"scheme": scheme.id, "title": scheme.title, "axis": scheme.axis, "params": scheme.params, "symbols": syms},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summaries = []
    dailies = []
    for sym in syms:
        try:
            path = train_symbol(scheme, sym)
            if path is None:
                print(f"  skip train {sym}")
                continue
            bt = backtest_symbol(scheme, sym)
            if bt is None:
                continue
            summaries.extend(bt["summary"])
            dailies.append(bt["daily"])
            zero = [r for r in bt["summary"] if r["bps"] == 0][0]
            print(f"  {sym}: ret={zero['return']:.2%} sharpe={zero['sharpe']:.2f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {sym}: {exc}")
    res = RESULT_ROOT / scheme.id
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(res / "summary.csv", index=False)
    if dailies:
        pd.concat(dailies, ignore_index=True).to_csv(res / "daily.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", nargs="*", default=None)
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    schemes = [by_id(s) for s in args.scheme] if args.scheme else schemes_by_family("P")
    for scheme in schemes:
        run_scheme(scheme, args.symbols)


if __name__ == "__main__":
    main()
