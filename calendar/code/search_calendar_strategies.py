#!/usr/bin/env python3
"""Rule search on calendar / roll / delivery-month panels."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from common import PANEL_DIR, RESULT_DIR, ensure_dirs  # noqa: E402
from futures_lot_specs import lot_fee, lot_margin  # noqa: E402

CAPITAL = 1_000_000.0
START = pd.Timestamp("2018-01-01")

# (factor, mode, pnl_col, event_gate)
SCHEMES = [
    # --- delivery month ---
    ("in_deliv_month", "bool_short", "pnl_short_near", "all"),
    ("in_deliv_month", "bool_long", "pnl_long_near", "all"),
    ("pre_deliv", "bool_short", "pnl_short_near", "all"),
    ("pre_deliv", "bool_long", "pnl_long_near", "all"),
    # --- roll window carry MR (calendar-gated spread) ---
    ("carry_z20", "sign_mr", "pnl_long_carry", "in_roll_window"),
    ("carry_z20", "sign_mr", "pnl_long_carry", "pre_roll"),
    ("carry_z60", "sign_mr", "pnl_long_carry", "in_roll_window"),
    # --- roll window outright momentum on main ---
    ("main_mom5", "sign_mom", "pnl_long_main", "in_roll_window"),
    ("main_mom5", "sign_mom", "pnl_long_main", "pre_roll"),
    # --- near OI share (roll pressure) ---
    ("near_oi_share", "low_long_near", "pnl_long_near", "pre_roll"),
    ("near_oi_share", "high_short_near", "pnl_short_near", "pre_roll"),
    # --- calendar month dummies on main ---
    ("cal_month", "month_long", "pnl_long_main", "all"),
    # --- unconditional carry (baseline vs spread doc) ---
    ("carry_z20", "sign_mr", "pnl_long_carry", "all"),
]

GATES = {
    "all": lambda d: np.ones(len(d), dtype=bool),
    "in_roll_window": lambda d: d["in_roll_window"].fillna(False).to_numpy(),
    "pre_roll": lambda d: d["pre_roll"].fillna(False).to_numpy(),
    "in_deliv_month": lambda d: d["in_deliv_month"].fillna(False).to_numpy(),
    "pre_deliv": lambda d: d["pre_deliv"].fillna(False).to_numpy(),
}

THRS = {
    "sign_mr": [0.75, 1.0, 1.25, 1.5],
    "sign_mom": [0.01, 0.02, 0.03],
    "low_long_near": [0.35, 0.40, 0.45],
    "high_short_near": [0.55, 0.60, 0.65],
    "month_long": list(range(1, 13)),
    "bool_short": [1.0],
    "bool_long": [1.0],
}

HOLDS = [1, 3, 5, 10, 15]


def sharpe(r: pd.Series) -> float | None:
    r = r.fillna(0.0)
    if len(r) < 40 or float(r.std()) == 0:
        return None
    return float(r.mean() / r.std() * np.sqrt(252))


def max_dd(r: pd.Series) -> float | None:
    r = r.fillna(0.0)
    if len(r) < 5:
        return None
    nav = (1.0 + r / CAPITAL).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def hold_signal(raw: np.ndarray, hold: int) -> np.ndarray:
    sig = np.zeros(len(raw), dtype=float)
    left = 0
    last = 0.0
    for i, v in enumerate(raw):
        if v != 0:
            left = hold
            last = float(np.sign(v))
        if left > 0:
            sig[i] = last
            left -= 1
    return sig


def margin_one(sym: str, close: float) -> float:
    m, _ = lot_margin(sym, close if np.isfinite(close) else None, broker=True)
    return float(m) if m is not None else np.nan


def run_one(
    df: pd.DataFrame,
    *,
    factor: str,
    mode: str,
    pnl_col: str,
    gate_name: str,
    thr: float,
    hold: int,
) -> dict | None:
    d = df.dropna(subset=[factor, pnl_col]).copy()
    if len(d) < 80:
        return None
    gate = GATES[gate_name](d)
    x = d[factor].to_numpy(dtype=float)
    raw = np.zeros(len(x))

    if mode == "bool_short":
        raw = np.where((x >= thr) & gate, -1.0, 0.0)
    elif mode == "bool_long":
        raw = np.where((x >= thr) & gate, 1.0, 0.0)
    elif mode == "sign_mr":
        raw = np.where((np.abs(x) >= thr) & gate, -np.sign(x), 0.0)
    elif mode == "sign_mom":
        raw = np.where((np.abs(x) >= thr) & gate, np.sign(x), 0.0)
    elif mode == "low_long_near":
        raw = np.where((x <= thr) & gate, 1.0, 0.0)
    elif mode == "high_short_near":
        raw = np.where((x >= thr) & gate, -1.0, 0.0)
    elif mode == "month_long":
        raw = np.where((x == thr) & gate, 1.0, 0.0)
    else:
        return None

    sig = hold_signal(raw, hold) if hold > 1 else raw
    pnl = d[pnl_col].to_numpy(dtype=float) * sig
    pnl_s = pd.Series(pnl, index=d.index).fillna(0.0)
    active = sig != 0
    if active.sum() < 20:
        return None

    sym = str(d["symbol"].iloc[0])
    margins = []
    for r, s in zip(d.itertuples(), sig):
        if s == 0:
            margins.append(0.0)
            continue
        if pnl_col == "pnl_long_carry":
            m1 = margin_one(sym, r.near_close)
            m2 = margin_one(sym, r.far_close)
            margins.append(m1 + m2 if np.isfinite(m1) and np.isfinite(m2) else np.nan)
        elif "near" in pnl_col:
            margins.append(margin_one(sym, r.near_close))
        else:
            margins.append(margin_one(sym, r.main_close))
    avg_margin = float(np.nanmean([m for m in margins if m > 0])) if any(m > 0 for m in margins) else np.nan

    total_pnl = float(pnl_s.sum())
    rets = pnl_s / CAPITAL
    sh = sharpe(rets)
    if sh is None:
        return None

    return {
        "symbol": sym,
        "factor": factor,
        "mode": mode,
        "pnl_col": pnl_col,
        "gate": gate_name,
        "thr": thr,
        "hold": hold,
        "days": len(d),
        "active_days": int(active.sum()),
        "active_pct": float(active.mean()),
        "total_pnl": total_pnl,
        "return": total_pnl / CAPITAL,
        "sharpe": sh,
        "max_dd": max_dd(rets),
        "worst_day": float(pnl_s.min()),
        "avg_margin": avg_margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=str(START.date()))
    args = parser.parse_args()
    ensure_dirs()
    start = pd.Timestamp(args.start)

    rows = []
    for path in sorted(PANEL_DIR.glob("cal_*.parquet")):
        if path.name == "cal_all_calendar.parquet":
            continue
        df = pd.read_parquet(path)
        df = df[df["date"] >= start]
        sym = str(df["symbol"].iloc[0]) if len(df) else path.stem
        print(f"[search] {sym} n={len(df)}", flush=True)
        for factor, mode, pnl_col, gate in SCHEMES:
            for thr, hold in product(THRS.get(mode, [1.0]), HOLDS):
                r = run_one(
                    df,
                    factor=factor,
                    mode=mode,
                    pnl_col=pnl_col,
                    gate_name=gate,
                    thr=thr,
                    hold=hold,
                )
                if r is not None:
                    rows.append(r)

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        print("no results", flush=True)
        return

    all_df = all_df.sort_values("sharpe", ascending=False)
    all_df.to_csv(RESULT_DIR / "calendar_search_all.csv", index=False)

    best = all_df.loc[all_df.groupby("symbol")["sharpe"].idxmax()]
    best.to_csv(RESULT_DIR / "calendar_search_best.csv", index=False)

    pass_df = all_df[(all_df["sharpe"] >= 1.0) & (all_df["active_pct"] >= 0.05)]
    pass_df.to_csv(RESULT_DIR / "calendar_search_pass.csv", index=False)

    # equal-weight book: top scheme per symbol with sharpe>=1
    book_parts = []
    for sym in pass_df["symbol"].unique():
        sub = pass_df[pass_df["symbol"] == sym].sort_values("sharpe", ascending=False).head(1)
        if sub.empty:
            continue
        row = sub.iloc[0]
        path = PANEL_DIR / f"cal_{sym.lower()}.parquet"
        df = pd.read_parquet(path)
        df = df[df["date"] >= start]
        r = run_one(
            df,
            factor=row["factor"],
            mode=row["mode"],
            pnl_col=row["pnl_col"],
            gate_name=row["gate"],
            thr=row["thr"],
            hold=int(row["hold"]),
        )
        if r is None:
            continue
        # rebuild daily pnl for book
        d = df.dropna(subset=[row["factor"], row["pnl_col"]]).copy()
        gate = GATES[row["gate"]](d)
        x = d[row["factor"]].to_numpy(dtype=float)
        mode, thr, hold = row["mode"], row["thr"], int(row["hold"])
        raw = np.zeros(len(x))
        if mode == "bool_short":
            raw = np.where((x >= thr) & gate, -1.0, 0.0)
        elif mode == "bool_long":
            raw = np.where((x >= thr) & gate, 1.0, 0.0)
        elif mode == "sign_mr":
            raw = np.where((np.abs(x) >= thr) & gate, -np.sign(x), 0.0)
        elif mode == "sign_mom":
            raw = np.where((np.abs(x) >= thr) & gate, np.sign(x), 0.0)
        elif mode == "low_long_near":
            raw = np.where((x <= thr) & gate, 1.0, 0.0)
        elif mode == "high_short_near":
            raw = np.where((x >= thr) & gate, -1.0, 0.0)
        elif mode == "month_long":
            raw = np.where((x == thr) & gate, 1.0, 0.0)
        sig = hold_signal(raw, hold) if hold > 1 else raw
        leg = pd.DataFrame({
            "date": d["date"],
            "symbol": sym,
            "pnl": d[row["pnl_col"]].to_numpy(dtype=float) * sig,
        })
        book_parts.append(leg)

    summary = {
        "start": str(start.date()),
        "n_schemes": len(all_df),
        "n_pass": len(pass_df),
        "best_per_symbol": best.to_dict(orient="records"),
    }
    if book_parts:
        book = book_parts[0][["date"]].copy()
        for leg in book_parts:
            book = book.merge(leg[["date", "pnl"]].rename(columns={"pnl": leg["symbol"].iloc[0]}), on="date", how="outer")
        sym_cols = [c for c in book.columns if c != "date"]
        book["pnl_eq"] = book[sym_cols].mean(axis=1)
        book = book.sort_values("date")
        book.to_csv(RESULT_DIR / "calendar_book_daily.csv", index=False)
        rets = book["pnl_eq"].fillna(0.0) / CAPITAL
        summary["book"] = {
            "symbols": sym_cols,
            "sharpe": sharpe(rets),
            "return": float(rets.sum()),
            "max_dd": max_dd(rets),
            "days": len(book),
        }

    (RESULT_DIR / "calendar_search_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
