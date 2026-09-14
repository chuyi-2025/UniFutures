#!/usr/bin/env python3
"""Build cross-product spread & calendar carry panels from Tree-Stock all_contracts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from common import CONFIG_DIR, DATA_DIR, PANEL_DIR, TREE, ensure_dirs  # noqa: E402
from futures_lot_specs import MULTIPLIER  # noqa: E402
from linear_ridge_walkforward import _read_contract_csv, pick_main_fast  # noqa: E402
from speculator_rules import (  # noqa: E402
    contract_ord,
    pick_spec_near_far,
    speculator_can_hold,
)


def load_symbol_all(symbol: str) -> pd.DataFrame:
    sym_dir = TREE / symbol
    if not sym_dir.is_dir():
        return pd.DataFrame()
    parts = []
    for csv in sorted(sym_dir.glob("*.csv")):
        try:
            raw = _read_contract_csv(csv)
        except Exception:
            continue
        parts.append(raw)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    df["_ord"] = df["code"].map(contract_ord)
    return df.dropna(subset=["date", "close", "_ord"]).sort_values(["date", "_ord", "code"])


def fwd_ret_table(raw: pd.DataFrame) -> pd.DataFrame:
    px = raw[["date", "code", "close"]].drop_duplicates().sort_values(["code", "date"])
    nxt = px.groupby("code")["close"].shift(-1)
    gap = px.groupby("code")["date"].shift(-1) - px["date"]
    px["fwd_ret"] = np.log(nxt / px["close"])
    px.loc[gap.dt.days > 10, "fwd_ret"] = np.nan
    return px[["date", "code", "fwd_ret"]]


def load_symbol_main(symbol: str, *, speculator: bool) -> pd.DataFrame:
    df = load_symbol_all(symbol)
    if df.empty:
        return df
    if speculator:
        td = np.sort(df["date"].unique())
        rows = []
        for dt, day in df.groupby("date"):
            elig = []
            for _, r in day.iterrows():
                if speculator_can_hold(dt, str(r["code"]), symbol, td):
                    elig.append(r)
            if not elig:
                continue
            sub = pd.DataFrame(elig)
            main = pick_main_fast(sub[["date", "code", "close"]])
            if main.empty:
                continue
            row = main.iloc[0].copy()
            row["symbol"] = symbol
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
    else:
        main = pick_main_fast(df[["date", "code", "close"]].copy())
        out = main.copy()
        out["symbol"] = symbol
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    nxt = out.groupby("code")["close"].shift(-1)
    gap = out.groupby("code")["date"].shift(-1) - out.groupby("code")["date"].shift(0)
    # simpler: merge fwd from table
    fwd = fwd_ret_table(df)
    out = out.merge(fwd, on=["date", "code"], how="left")
    return out.sort_values("date")


def rank_contracts_day(g: pd.DataFrame) -> pd.DataFrame:
    """Naive: nearest expiry rank1/rank2."""
    rows = []
    for dt, day in g.groupby("date"):
        day = day.sort_values("_ord")
        for rank, (_, r) in enumerate(day.iterrows(), start=1):
            if rank > 3:
                break
            rows.append({
                "date": dt,
                "symbol": r["symbol"],
                "code": r["code"],
                "rank": rank,
                "close": float(r["close"]),
                "_ord": int(r["_ord"]),
                "open_interest": float(r.get("open_interest", np.nan)),
                "volume": float(r.get("volume", np.nan)),
            })
    return pd.DataFrame(rows)


def rank_speculator_carry(g: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Speculator: near=main among eligible, far=next eligible month."""
    td = np.sort(g["date"].unique())
    rows = []
    for dt, day in g.groupby("date"):
        day = day.copy()
        day["date"] = dt
        near_code, near_px, far_code, far_px = pick_spec_near_far(day, symbol, td)
        if not near_code or not far_code:
            continue
        near_ord = contract_ord(near_code)
        far_ord = contract_ord(far_code)
        rows.append({
            "date": dt,
            "symbol": symbol,
            "near_code": near_code,
            "near_close": near_px,
            "near_ord": near_ord,
            "far_code": far_code,
            "far_close": far_px,
            "far_ord": far_ord,
            "in_deliv_month": int(dt.year) * 12 + int(dt.month) == near_ord,
        })
    return pd.DataFrame(rows)


def leg_dollar_pnl(sym: str, close: float, fwd_ret: float, side: int) -> float:
    mult = MULTIPLIER.get(sym)
    if mult is None or not np.isfinite(close) or not np.isfinite(fwd_ret) or close <= 0:
        return np.nan
    val = float(mult) * float(close)
    return side * val * (np.exp(float(fwd_ret)) - 1.0)


def build_cross_spread(
    leg1: str, leg2: str, spread_id: str, kind: str, label: str, *, speculator: bool,
) -> pd.DataFrame:
    m1 = load_symbol_main(leg1, speculator=speculator)
    m2 = load_symbol_main(leg2, speculator=speculator)
    if m1.empty or m2.empty:
        return pd.DataFrame()
    a = m1.rename(columns={
        "code": "code1", "close": "close1", "fwd_ret": "fwd1",
    })[["date", "code1", "close1", "fwd1"]]
    b = m2.rename(columns={
        "code": "code2", "close": "close2", "fwd_ret": "fwd2",
    })[["date", "code2", "close2", "fwd2"]]
    m = a.merge(b, on="date", how="inner").sort_values("date")
    if kind == "ratio":
        m["spread_px"] = m["close1"] / m["close2"].replace(0, np.nan)
    else:
        m["spread_px"] = m["close1"] - m["close2"]
    m["pnl_long_spread"] = [
        leg_dollar_pnl(leg1, r.close1, r.fwd1, +1) - leg_dollar_pnl(leg2, r.close2, r.fwd2, +1)
        for r in m.itertuples()
    ]
    m["pnl_short_spread"] = -m["pnl_long_spread"]
    m["spread_chg"] = m["spread_px"].diff()
    m["spread_fwd"] = m["spread_px"].shift(-1) / m["spread_px"] - 1.0
    gap = m["date"].shift(-1) - m["date"]
    m.loc[gap.dt.days > 10, "spread_fwd"] = np.nan
    m["spread_id"] = spread_id
    m["label"] = label
    m["kind"] = kind
    m["leg1"] = leg1
    m["leg2"] = leg2
    return m


def build_carry_panel(symbol: str, *, speculator: bool) -> pd.DataFrame:
    raw = load_symbol_all(symbol)
    if raw.empty:
        return pd.DataFrame()
    fwd = fwd_ret_table(raw)
    if speculator:
        ranked = rank_speculator_carry(raw, symbol)
        if ranked.empty:
            return pd.DataFrame()
        m = ranked.sort_values("date")
    else:
        ranked = rank_contracts_day(raw)
        if ranked.empty:
            return pd.DataFrame()
        near = ranked[ranked["rank"] == 1].rename(columns={
            "code": "near_code", "close": "near_close", "_ord": "near_ord",
        })
        far = ranked[ranked["rank"] == 2].rename(columns={
            "code": "far_code", "close": "far_close", "_ord": "far_ord",
        })
        m = near.merge(
            far[["date", "symbol", "far_code", "far_close", "far_ord"]],
            on=["date", "symbol"],
            how="inner",
        ).sort_values("date")
        m["in_deliv_month"] = m["near_ord"] == (
            pd.to_datetime(m["date"]).dt.year * 12 + pd.to_datetime(m["date"]).dt.month
        )

    m["spread_px"] = m["near_close"] - m["far_close"]
    m["carry_pct"] = m["spread_px"] / m["near_close"].replace(0, np.nan)
    m = m.merge(
        fwd.rename(columns={"code": "near_code", "fwd_ret": "near_fwd"}),
        on=["date", "near_code"], how="left",
    )
    m = m.merge(
        fwd.rename(columns={"code": "far_code", "fwd_ret": "far_fwd"}),
        on=["date", "far_code"], how="left",
    )
    m["pnl_long_carry"] = [
        leg_dollar_pnl(symbol, r.near_close, r.near_fwd, +1) - leg_dollar_pnl(symbol, r.far_close, r.far_fwd, +1)
        for r in m.itertuples()
    ]
    m["spread_id"] = f"{symbol.lower()}_carry1"
    m["label"] = f"{symbol}近远月"
    m["kind"] = "carry"
    m["leg1"] = symbol
    m["leg2"] = symbol
    return m


def add_factors(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values("date").copy()
    for win in (20, 60, 120):
        m = out["spread_px"].rolling(win, min_periods=max(5, win // 3)).mean()
        sd = out["spread_px"].rolling(win, min_periods=max(5, win // 3)).std()
        out[f"spread_z{win}"] = (out["spread_px"] - m) / sd.replace(0, np.nan)
    if "carry_pct" in out.columns:
        for win in (20, 60):
            m = out["carry_pct"].rolling(win, min_periods=max(5, win // 3)).mean()
            sd = out["carry_pct"].rolling(win, min_periods=max(5, win // 3)).std()
            out[f"carry_z{win}"] = (out["carry_pct"] - m) / sd.replace(0, np.nan)
    out["spread_ma20"] = out["spread_px"].rolling(20, min_periods=5).mean()
    out["spread_ma60"] = out["spread_px"].rolling(60, min_periods=10).mean()
    return out


def build_all(start: pd.Timestamp, panel_dir: Path, *, speculator: bool) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / "spreads.yaml").read_text(encoding="utf-8"))
    panel_dir.mkdir(parents=True, exist_ok=True)
    mode = "speculator" if speculator else "naive"
    cross_rows, carry_rows = [], []

    for sid, spec in cfg.get("cross", {}).items():
        print(f"[cross|{mode}] {sid}...", flush=True)
        df = build_cross_spread(
            spec["leg1"], spec["leg2"], sid,
            spec.get("kind", "cross"), spec.get("label", sid),
            speculator=speculator,
        )
        if df.empty:
            continue
        df = add_factors(df)[df["date"] >= start]
        path = panel_dir / f"cross_{sid}.parquet"
        df.to_parquet(path, index=False)
        cross_rows.append({"spread_id": sid, "days": len(df), "path": path.name})

    for sym in cfg.get("carry", {}).get("symbols", []):
        print(f"[carry|{mode}] {sym}...", flush=True)
        df = build_carry_panel(sym, speculator=speculator)
        if df.empty:
            continue
        df = add_factors(df)[df["date"] >= start]
        path = panel_dir / f"carry_{sym.lower()}.parquet"
        df.to_parquet(path, index=False)
        deliv_pct = float(df["in_deliv_month"].mean()) if "in_deliv_month" in df.columns else None
        carry_rows.append({
            "spread_id": f"{sym.lower()}_carry1",
            "days": len(df),
            "in_deliv_month_pct": deliv_pct,
            "path": path.name,
        })
        print(f"  {len(df)} days, deliv_month={deliv_pct:.1%}" if deliv_pct is not None else f"  {len(df)} days", flush=True)

    parts = [pd.read_parquet(p) for p in sorted(panel_dir.glob("*.parquet")) if p.name != "all_spreads.parquet"]
    if parts:
        pd.concat(parts, ignore_index=True).to_parquet(panel_dir / "all_spreads.parquet", index=False)
    meta = {"mode": mode, "cross": cross_rows, "carry": carry_rows, "start": str(start.date())}
    (panel_dir / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument(
        "--mode", choices=("naive", "speculator", "both"), default="speculator",
        help="naive=rank1/rank2; speculator=exchange delivery rules; both=write both dirs",
    )
    args = parser.parse_args()
    ensure_dirs()
    start = pd.Timestamp(args.start)
    naive_dir = DATA_DIR / "panels_naive"
    spec_dir = DATA_DIR / "panels_speculator"

    if args.mode in ("naive", "both"):
        print("=== NAIVE panels ===", flush=True)
        build_all(start, naive_dir if args.mode == "both" else PANEL_DIR, speculator=False)
    if args.mode in ("speculator", "both"):
        print("=== SPECULATOR panels ===", flush=True)
        build_all(start, spec_dir if args.mode == "both" else PANEL_DIR, speculator=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
