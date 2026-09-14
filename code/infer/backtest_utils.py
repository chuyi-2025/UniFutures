"""Infer-only backtest helpers: main-contract fallback + latest-day inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "backtest"))

from backtest_kronos import (  # noqa: E402
    FEAT_COLS,
    INIT_CAP,
    TH,
    WINDOW,
    X_COLS,
    WD_COLS,
    Kronos1d,
    load_contract,
    parse_code_ym,
    roll_threshold,
    step_pos,
)
from shared import CONTRACTS_DIR, GAF_FEATURES_DIR  # noqa: E402


def pick_main_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Strict m+3 pick; if none qualifies, use farthest-month contract that day."""
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
        if best_row is None:
            fallback_ym = None
            for _, row in grp.iterrows():
                ym = parse_code_ym(row["code"])
                if fallback_ym is None or ym > fallback_ym:
                    fallback_ym = ym
                    best_row = row
        if best_row is not None:
            picks.append(best_row)
    if not picks:
        return pd.DataFrame()
    return pd.DataFrame(picks).sort_values("date").reset_index(drop=True)


def _log_ret(row: pd.Series) -> float:
    v = row["log_ret_1d"]
    return 0.0 if pd.isna(v) else float(v)


def backtest_kronos(symbol: str, kronos: Kronos1d, contract_cache: dict) -> pd.DataFrame:
    import shared

    path = GAF_FEATURES_DIR / f"{symbol}.csv"
    df = pd.read_csv(path, parse_dates=["date", "exit_date", *WD_COLS])
    df = df[(df["date"] >= shared.BACKTEST_START) & (df["date"] <= shared.BACKTEST_END)]
    df = pick_main_contract(df).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    cap = INIT_CAP
    pos = 0
    rows: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"kronos {symbol}", leave=False):
        key = row["contract_file"]
        if key not in contract_cache:
            contract_cache[key] = load_contract(CONTRACTS_DIR / symbol / f"{key}.csv")
        seg = contract_cache[key]
        i = int(row["seg_idx"])
        raw = seg.iloc[i - WINDOW + 1 : i + 1][FEAT_COLS].to_numpy(dtype=np.float64)
        norm_x = row[X_COLS].to_numpy(dtype=np.float32).reshape(WINDOW, len(FEAT_COLS))
        x_ts = pd.to_datetime(row[WD_COLS])
        y_ts = pd.Timestamp(seg.iloc[i + 1]["date"]) if i + 1 < len(seg) else pd.Timestamp(row["date"]) + pd.offsets.BDay(1)

        pred_ret = kronos.pred_ret_1d(norm_x, x_ts, y_ts, raw)
        new_pos = step_pos(pos, pred_ret)
        log_r = _log_ret(row)
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


def install_patches() -> None:
    import backtest_kronos as bk
    import backtest_xgb as bx
    import backtest_gaf as bg
    import train_ppo as tp

    bk.pick_main_contract = pick_main_contract
    tp.pick_main_contract = pick_main_contract
    bx.pick_main_contract = pick_main_contract
    bg.pick_main_contract = pick_main_contract
    bk.backtest_symbol = backtest_kronos

    _orig_xgb = bx.backtest
    _orig_gaf = bg.backtest

    def backtest_xgb(symbol: str):
        part = _orig_xgb(symbol)
        if part.empty:
            return part
        part = part.copy()
        part["log_ret_1d"] = part["log_ret_1d"].fillna(0.0)
        cap = INIT_CAP
        caps = []
        for _, row in part.iterrows():
            cap *= float(np.exp(row["position"] * row["log_ret_1d"]))
            caps.append(cap)
        part["capital"] = caps
        return part

    def backtest_gaf(symbol: str, model_path: Path):
        part = _orig_gaf(symbol, model_path)
        if part.empty:
            return part
        part = part.copy()
        part["log_ret_1d"] = part["log_ret_1d"].fillna(0.0)
        cap = INIT_CAP
        caps = []
        for _, row in part.iterrows():
            cap *= float(np.exp(row["position"] * row["log_ret_1d"]))
            caps.append(cap)
        part["capital"] = caps
        return part

    bx.backtest = backtest_xgb
    bg.backtest = backtest_gaf

    _orig_ppo = tp.backtest if hasattr(tp, "backtest") else None
    import backtest_ppo as bp

    _orig_bt_df = bp.backtest_df

    def backtest_df(
        df,
        model_path,
        with_sentiment=None,
        sent_names=None,
    ):
        df = df.copy()
        df["log_ret_1d"] = df["log_ret_1d"].fillna(0.0)
        return _orig_bt_df(
            df,
            model_path,
            with_sentiment=with_sentiment,
            sent_names=sent_names,
        )

    bp.backtest_df = backtest_df
