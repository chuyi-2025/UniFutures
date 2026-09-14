#!/usr/bin/env python3
"""Research backtest: China crop-region weather stress → domestic ag futures.

Research-only (not an account blotter). Position is ±1 notionally; PnL is
fwd_ret * signal with optional hold days after event.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_prices import align_all  # noqa: E402
from common import RESULT_DIR, SYMBOLS, ensure_dirs, load_regions  # noqa: E402

# Supply-shock bias: weather stress → lean long (tighter supply)
HOLD_DAYS = 5
CAPITAL = 1_000_000.0


def year_stats(pnl: pd.Series) -> dict:
    r = pnl.fillna(0.0)
    if len(r) < 5 or r.std() == 0:
        return {"sharpe": None, "return": float(r.sum()), "max_dd": None, "worst": float(r.min()) if len(r) else None}
    # treat pnl as already in return units of capital if scaled; here raw log-ret sum
    sh = float(r.mean() / r.std() * np.sqrt(252))
    nav = (1.0 + r).cumprod()
    dd = float((nav / nav.cummax() - 1.0).min())
    return {"sharpe": round(sh, 3), "return": round(float(r.sum()), 4), "max_dd": round(dd, 4), "worst": round(float(r.min()), 6)}


def build_signal(df: pd.DataFrame, hold_days: int = HOLD_DAYS) -> pd.DataFrame:
    """+1 on frost / heat-dry / drought stress inside phenology; hold N trade days."""
    out = df.sort_values("date").copy()
    event = (
        (out.get("frost_event", 0).fillna(0) > 0)
        | (out.get("heat_dry", 0).fillna(0) > 0)
        | (out.get("drought_stress", 0).fillna(0) > 0)
    )
    # only inside phenology window when flag available
    if "in_pheno" in out.columns:
        event = event & (out["in_pheno"].fillna(0) > 0)
    raw = event.astype(int)
    # hold: once event fires, keep +1 for hold_days trading rows
    sig = np.zeros(len(out), dtype=float)
    left = 0
    for i, e in enumerate(raw.to_numpy()):
        if e:
            left = hold_days
        if left > 0:
            sig[i] = 1.0
            left -= 1
    out["event"] = raw
    out["signal"] = sig
    out["pnl"] = out["signal"] * out["fwd_ret"].fillna(0.0)
    return out


def run_symbol(df: pd.DataFrame, symbol: str, hold_days: int) -> tuple[pd.DataFrame, dict]:
    sub = df[df["symbol"] == symbol].copy()
    sub = build_signal(sub, hold_days=hold_days)
    live = sub.dropna(subset=["fwd_ret"])
    st = year_stats(live["pnl"])
    yearly = []
    for y, g in live.groupby(live["date"].dt.year):
        ys = year_stats(g["pnl"])
        yearly.append({
            "year": int(y),
            "pnl_ret": ys["return"],
            "sharpe": ys["sharpe"],
            "max_dd": ys["max_dd"],
            "worst": ys["worst"],
            "n_events": int(g["event"].sum()),
            "active_days": int((g["signal"] != 0).sum()),
            "days": int(len(g)),
        })
    summary = {
        "symbol": symbol,
        "full": st,
        "n_events": int(live["event"].sum()),
        "active_days": int((live["signal"] != 0).sum()),
        "days": int(len(live)),
        "yearly": yearly,
    }
    return sub, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="CF,SR,AP,CJ")
    ap.add_argument("--hold-days", type=int, default=HOLD_DAYS)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--lag", type=int, default=1, help="weather lag in calendar days")
    args = ap.parse_args()

    ensure_dirs()
    want = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    print("[align panel]", flush=True)
    panel = align_all(symbols=want, weather_lag_days=args.lag)
    start = pd.Timestamp(args.start)
    panel = panel[panel["date"] >= start].copy()

    aligned_path = RESULT_DIR / "aligned_panel.parquet"
    panel.to_parquet(aligned_path, index=False)

    summaries = []
    daily_frames = []
    for sym in want:
        print(f"[backtest] {sym}", flush=True)
        bt, summary = run_symbol(panel, sym, hold_days=args.hold_days)
        summaries.append(summary)
        daily = bt[["date", "symbol", "code", "signal", "event", "fwd_ret", "pnl",
                    "tmean_z", "precip_z", "frost_event", "heat_dry", "drought_stress"]].copy()
        daily["日期"] = daily["date"].dt.strftime("%Y-%m-%d")
        daily_frames.append(daily)
        out_csv = RESULT_DIR / f"{sym}_weather_daily.csv"
        daily.drop(columns=["date"]).to_csv(out_csv, index=False)
        print(
            f"  sharpe={summary['full']['sharpe']} ret={summary['full']['return']} "
            f"events={summary['n_events']} active={summary['active_days']}",
            flush=True,
        )

    # equal-weight book across symbols when each has a row that day
    all_d = pd.concat(daily_frames, ignore_index=True)
    book = (
        all_d.groupby("日期", as_index=False)
        .agg(pnl=("pnl", "mean"), n=("symbol", "count"), n_active=("signal", lambda s: int((s != 0).sum())))
    )
    book_path = RESULT_DIR / "book_ew_daily.csv"
    book.to_csv(book_path, index=False)

    report = {
        "hold_days": args.hold_days,
        "weather_lag_days": args.lag,
        "start": args.start,
        "symbols": list(want),
        "note": "Research PnL = signal * log fwd_ret; not account blotter.",
        "per_symbol": summaries,
        "book_ew": year_stats(book.set_index(pd.to_datetime(book["日期"]))["pnl"]),
        "regions": {s: load_regions()["symbols"][s]["name_cn"] for s in want},
    }
    (RESULT_DIR / "weather_backtest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", RESULT_DIR, flush=True)
    print("book_ew", report["book_ew"], flush=True)


if __name__ == "__main__":
    main()
