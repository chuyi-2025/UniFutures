#!/usr/bin/env python3
"""32 highly diverse retrain schemes for UniFutures retrain_v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path("/home/workspace/lab/UniFutures")
EXP_ROOT = ROOT / "data/experiments/retrain_v1"
RESULT_ROOT = ROOT / "data/results/retrain_v1"
SEED = 42

FERROUS = ["RB", "HC", "I", "J", "JM", "SF", "SM", "SS"]
NONFERROUS = ["CU", "AL", "NI", "SN", "PB", "ZN", "AO", "AD", "SI", "PS", "LC"]
OILSEEDS = ["M", "Y", "P", "OI", "RM", "A", "B", "C", "CS"]


@dataclass(frozen=True)
class Scheme:
    id: str
    family: str  # S/X/G/P
    title: str
    axis: str
    params: dict


SCHEMES: list[Scheme] = [
    Scheme("S01", "S", "sent_zh_h1_th05", "finance_zh H1 RET_TH=0.0005", {"model": "finance_zh", "horizon": 1, "ret_th": 0.0005}),
    Scheme("S02", "S", "sent_zh_h1_th20", "finance_zh H1 RET_TH=0.002", {"model": "finance_zh", "horizon": 1, "ret_th": 0.002}),
    Scheme("S03", "S", "sent_zh_h7", "finance_zh H7 weak labels", {"model": "finance_zh", "horizon": 7, "ret_th": 0.001}),
    Scheme("S04", "S", "sent_zh_h14", "finance_zh H14 weak labels", {"model": "finance_zh", "horizon": 14, "ret_th": 0.001}),
    Scheme("S05", "S", "sent_modernbert_h1", "ModernBERT backbone H1", {"model": "modernbert", "horizon": 1, "ret_th": 0.001}),
    Scheme("S06", "S", "sent_finbert2_h1", "FinBERT2 backbone H1", {"model": "finbert2", "horizon": 1, "ret_th": 0.001}),
    Scheme("S07", "S", "sent_zh_binary", "drop neutral; binary CE", {"model": "finance_zh", "horizon": 1, "mode": "binary"}),
    Scheme("S08", "S", "sent_zh_regress", "MSE on signed return", {"model": "finance_zh", "horizon": 1, "mode": "regress"}),
    Scheme("X01", "X", "xgb_h1_3cls", "1d 3-class softprob", {"horizon": 1, "mode": "3cls"}),
    Scheme("X02", "X", "xgb_h3_8cls", "3d 8-class absolute bins", {"horizon": 3, "mode": "8cls"}),
    Scheme("X03", "X", "xgb_h5_8cls", "5d baseline retrain", {"horizon": 5, "mode": "8cls"}),
    Scheme("X04", "X", "xgb_h10_8cls", "10d 8-class", {"horizon": 10, "mode": "8cls"}),
    Scheme("X05", "X", "xgb_h5_quantile", "5d equal-count 8 bins", {"horizon": 5, "mode": "quantile8"}),
    Scheme("X06", "X", "xgb_h5_extreme", "5d |ret|>=2% only 3-class", {"horizon": 5, "mode": "extreme"}),
    Scheme("X07", "X", "xgb_h5_soft_rank", "rank:pairwise on ret_5d", {"horizon": 5, "mode": "rank"}),
    Scheme("X08", "X", "xgb_h5_ferrous", "ferrous-only universe", {"horizon": 5, "mode": "8cls", "symbols": FERROUS}),
    Scheme("G01", "G", "gaf_nonferrous_mse_h3", "nonferrous MSE H3", {"category": "nonferrous", "horizon": 3, "loss": "mse", "clip": 0.05}),
    Scheme("G02", "G", "gaf_ferrous_mse_h3", "ferrous MSE H3", {"category": "ferrous", "horizon": 3, "loss": "mse", "clip": 0.05}),
    Scheme("G03", "G", "gaf_oilseeds_mse_h3", "oilseeds MSE H3", {"category": "oilseeds", "horizon": 3, "loss": "mse", "clip": 0.05}),
    Scheme("G04", "G", "gaf_nonferrous_mse_h1", "nonferrous MSE H1", {"category": "nonferrous", "horizon": 1, "loss": "mse", "clip": 0.05}),
    Scheme("G05", "G", "gaf_nonferrous_mse_h5", "nonferrous MSE H5", {"category": "nonferrous", "horizon": 5, "loss": "mse", "clip": 0.05}),
    Scheme("G06", "G", "gaf_nonferrous_ce6", "6-class CE head", {"category": "nonferrous", "horizon": 3, "loss": "ce6", "clip": 0.05}),
    Scheme("G07", "G", "gaf_nonferrous_huber", "Huber regression", {"category": "nonferrous", "horizon": 3, "loss": "huber", "clip": 0.05}),
    Scheme("G08", "G", "gaf_nonferrous_clip10", "CLIP=0.10 wider target", {"category": "nonferrous", "horizon": 3, "loss": "mse", "clip": 0.10}),
    Scheme("P01", "P", "ppo_cost_turnover", "reward - λ|Δpos|", {"reward": "cost", "obs": "base", "timesteps": 40000}),
    Scheme("P02", "P", "ppo_sent_cost", "+daily sentiment + cost", {"reward": "cost", "obs": "sent", "timesteps": 40000}),
    Scheme("P03", "P", "ppo_lb_pool_L7", "lookback pool L7 obs", {"reward": "pnl", "obs": "lb_pool_L7", "timesteps": 40000}),
    Scheme("P04", "P", "ppo_reward_h3", "3-day cumulative reward", {"reward": "h3", "obs": "base", "timesteps": 40000}),
    Scheme("P05", "P", "ppo_diff_sharpe", "differential Sharpe reward", {"reward": "dsharpe", "obs": "base", "timesteps": 40000}),
    Scheme("P06", "P", "ppo_no_kronos", "drop Kronos feature", {"reward": "pnl", "obs": "no_kronos", "timesteps": 40000}),
    Scheme("P07", "P", "ppo_hold_state", "prev pos + hold age in obs", {"reward": "cost", "obs": "hold_state", "timesteps": 40000}),
    Scheme("P08", "P", "ppo_compact_2024", "2024-2025H1 + compact sent", {"reward": "cost", "obs": "compact_sent", "train_start": "2024-01-01", "train_end": "2025-06-30", "timesteps": 40000}),
]


def by_id(scheme_id: str) -> Scheme:
    for s in SCHEMES:
        if s.id == scheme_id or s.title == scheme_id:
            return s
    raise KeyError(scheme_id)


def schemes_by_family(family: str) -> list[Scheme]:
    return [s for s in SCHEMES if s.family == family.upper()]


def scheme_dir(scheme: Scheme) -> Path:
    return EXP_ROOT / scheme.id


def scheme_to_dict(scheme: Scheme) -> dict:
    return asdict(scheme)


CATEGORY_SYMBOLS = {
    "nonferrous": NONFERROUS,
    "ferrous": FERROUS,
    "oilseeds": OILSEEDS,
}
