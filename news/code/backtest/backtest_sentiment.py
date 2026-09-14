#!/usr/bin/env python3
"""Backtest daily sentiment positions on main-contract next-day returns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from common import (  # noqa: E402
    INIT_CAP,
    RESULTS_ROOT,
    ZEROSHOT_END,
    ZEROSHOT_START,
    load_symbol_ohlc,
)


def majority_position(vals: list[int]) -> int:
    if not vals:
        return 0
    s = sum(vals)
    if s > 0:
        return 1
    if s < 0:
        return -1
    # tie / all flat
    pos = sum(1 for v in vals if v > 0)
    neg = sum(1 for v in vals if v < 0)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def aggregate_daily_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """One position per (symbol, trade_date) via majority vote."""
    rows = []
    for (sym, dt), grp in signals.groupby(["symbol", "trade_date"], sort=True):
        pos = majority_position(grp["position"].astype(int).tolist())
        rows.append({"symbol": sym, "trade_date": pd.Timestamp(dt), "position": pos, "n_docs": len(grp)})
    return pd.DataFrame(rows)


def expand_hold_positions(daily_sig: pd.DataFrame, dates: pd.DatetimeIndex, hold_days: int) -> dict:
    """Map each calendar date -> position in {-1,0,1} after expanding holds.

    Each (trade_date, position) covers the next `hold_days` sessions in `dates`
    starting at trade_date (inclusive). Overlaps: mean then sign.
    """
    if hold_days <= 1:
        return {
            pd.Timestamp(r.trade_date).normalize(): int(r.position)
            for r in daily_sig.itertuples(index=False)
        }

    date_list = [pd.Timestamp(d).normalize() for d in dates]
    loc = {d: i for i, d in enumerate(date_list)}
    buckets: dict[pd.Timestamp, list[float]] = {d: [] for d in date_list}

    for r in daily_sig.itertuples(index=False):
        td = pd.Timestamp(r.trade_date).normalize()
        if td not in loc:
            # signal may fall just before window; skip if not in backtest dates
            continue
        i0 = loc[td]
        pos = float(r.position)
        for j in range(i0, min(i0 + hold_days, len(date_list))):
            buckets[date_list[j]].append(pos)

    out: dict[pd.Timestamp, int] = {}
    for d, vals in buckets.items():
        if not vals:
            out[d] = 0
            continue
        m = float(np.mean(vals))
        out[d] = 1 if m > 0 else (-1 if m < 0 else 0)
    return out


def backtest_symbol(
    symbol: str,
    daily_sig: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    hold_days: int = 1,
) -> pd.DataFrame:
    px = load_symbol_ohlc(symbol)
    if px.empty:
        return pd.DataFrame()
    px = px[(px["date"] >= start) & (px["date"] <= end)].reset_index(drop=True)
    if px.empty:
        return pd.DataFrame()

    sym_sig = daily_sig[daily_sig["symbol"] == symbol]
    sig = expand_hold_positions(sym_sig, pd.DatetimeIndex(px["date"]), hold_days)
    cap = INIT_CAP
    rows = []
    for _, row in px.iterrows():
        dt = pd.Timestamp(row["date"]).normalize()
        pos = int(sig.get(dt, 0))
        ret = float(row["log_ret_1d"]) if pd.notna(row["log_ret_1d"]) else 0.0
        if not np.isfinite(ret):
            ret = 0.0
        day_pnl = pos * ret
        if pos != 0:
            cap *= float(np.exp(pos * ret))
        rows.append(
            {
                "date": dt,
                "symbol": symbol,
                "code": row["code"],
                "position": pos,
                "log_ret_1d": ret,
                "strategy_ret": day_pnl,
                "capital": cap,
            }
        )
    return pd.DataFrame(rows)


def summarize(part: pd.DataFrame) -> dict:
    if part.empty:
        return {}
    active = part[part["position"] != 0]
    rets = active["strategy_ret"].to_numpy() if len(active) else np.array([])
    wins = int((rets > 0).sum()) if len(rets) else 0
    n_trade = int(len(rets))
    win_rate = float(wins / n_trade) if n_trade else 0.0
    all_r = part["strategy_ret"].to_numpy()
    sharpe = (
        float(all_r.mean() / all_r.std() * np.sqrt(252))
        if len(all_r) > 1 and all_r.std() > 1e-12
        else 0.0
    )
    peak = part["capital"].cummax().to_numpy()
    max_dd = float((part["capital"].to_numpy() / peak - 1.0).min()) if len(part) else 0.0
    final = float(part["capital"].iloc[-1])
    return {
        "symbol": part["symbol"].iloc[0],
        "final_capital": final,
        "return": final / INIT_CAP - 1.0,
        "days": int(len(part)),
        "trade_days": n_trade,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "status": "ok",
    }


def run_backtest(
    signals: pd.DataFrame,
    out_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    hold_days: int = 1,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    daily = aggregate_daily_signals(signals)
    daily = daily[(daily["trade_date"] >= start) & (daily["trade_date"] <= end)]
    summaries = []
    for sym in sorted(daily["symbol"].unique()):
        part = backtest_symbol(sym, daily, start, end, hold_days=hold_days)
        if part.empty:
            summaries.append({"symbol": sym, "status": "fail", "error": "no price"})
            continue
        sym_dir = out_dir / sym.lower()
        sym_dir.mkdir(parents=True, exist_ok=True)
        part.to_csv(sym_dir / "daily.csv", index=False)
        sm = summarize(part)
        pd.DataFrame([sm]).to_csv(sym_dir / "summary.csv", index=False)
        summaries.append(sm)
        print(
            f"  {sym}: ret={sm['return']*100:.2f}% sharpe={sm['sharpe']:.2f} "
            f"win={sm['win_rate']*100:.1f}% trades={sm['trade_days']}"
        )
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    daily.to_csv(out_dir / "daily_signals.csv", index=False)
    return summary_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=str, default=str(ZEROSHOT_START.date()))
    ap.add_argument("--end", type=str, default=str(ZEROSHOT_END.date()))
    ap.add_argument("--hold-days", type=int, default=1, help="hold each signal for N trading days")
    args = ap.parse_args()

    if args.signals.suffix == ".parquet":
        signals = pd.read_parquet(args.signals)
    else:
        signals = pd.read_csv(args.signals)
    signals["trade_date"] = pd.to_datetime(signals["trade_date"])
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    print(
        f"[backtest] {args.signals} -> {args.out} "
        f"[{start.date()} ~ {end.date()}] hold={args.hold_days}d"
    )
    run_backtest(signals, args.out, start, end, hold_days=args.hold_days)
    print("[done]", args.out)


if __name__ == "__main__":
    main()
