#!/usr/bin/env python3
"""Build calendar / roll / delivery-month factor panels from all_contracts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from common import CONFIG_DIR, PANEL_DIR, TREE, ensure_dirs  # noqa: E402
from futures_lot_specs import MULTIPLIER  # noqa: E402
from linear_ridge_walkforward import _read_contract_csv, pick_main_fast  # noqa: E402


def contract_ord(code: str) -> int | None:
    m = re.match(r"^[A-Z]+(\d{4})$", str(code).upper())
    if not m:
        return None
    yy = 2000 + int(m.group(1)[:2])
    mm = int(m.group(1)[2:])
    return yy * 12 + mm


def contract_ym(code: str) -> tuple[int, int] | None:
    o = contract_ord(code)
    if o is None:
        return None
    return o // 12, o % 12


def load_symbol_all(symbol: str) -> pd.DataFrame:
    sym_dir = TREE / symbol
    if not sym_dir.is_dir():
        return pd.DataFrame()
    parts = []
    for csv in sorted(sym_dir.glob("*.csv")):
        try:
            parts.append(_read_contract_csv(csv))
        except Exception:
            continue
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


def rank_contracts_day(g: pd.DataFrame) -> pd.DataFrame:
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


def leg_pnl(sym: str, close: float, fwd_ret: float, side: int) -> float:
    mult = MULTIPLIER.get(sym)
    if mult is None or not np.isfinite(close) or not np.isfinite(fwd_ret) or close <= 0:
        return np.nan
    val = float(mult) * float(close)
    return side * val * (np.exp(float(fwd_ret)) - 1.0)


def build_symbol_panel(symbol: str, *, roll_offset: int) -> pd.DataFrame:
    raw = load_symbol_all(symbol)
    if raw.empty:
        return pd.DataFrame()

    fwd = fwd_ret_table(raw)
    ranked = rank_contracts_day(raw)
    if ranked.empty:
        return pd.DataFrame()

    near = ranked[ranked["rank"] == 1].rename(columns={
        "code": "near_code",
        "close": "near_close",
        "_ord": "near_ord",
        "open_interest": "near_oi",
        "volume": "near_vol",
    })
    far = ranked[ranked["rank"] == 2].rename(columns={
        "code": "far_code",
        "close": "far_close",
        "_ord": "far_ord",
        "open_interest": "far_oi",
        "volume": "far_vol",
    })
    m = near.merge(
        far[["date", "symbol", "far_code", "far_close", "far_ord", "far_oi", "far_vol"]],
        on=["date", "symbol"],
        how="inner",
    ).sort_values("date")

    m = m.merge(
        fwd.rename(columns={"code": "near_code", "fwd_ret": "near_fwd"}),
        on=["date", "near_code"],
        how="left",
    )
    m = m.merge(
        fwd.rename(columns={"code": "far_code", "fwd_ret": "far_fwd"}),
        on=["date", "far_code"],
        how="left",
    )

    main = pick_main_fast(raw[["date", "code", "close"]].copy())
    main = main.rename(columns={"code": "main_code", "close": "main_close"})
    main = main.merge(
        fwd.rename(columns={"code": "main_code", "fwd_ret": "main_fwd"}),
        on=["date", "main_code"],
        how="left",
    )
    m = m.merge(main[["date", "main_code", "main_close", "main_fwd"]], on="date", how="left")

    dt = pd.to_datetime(m["date"])
    m["trade_ord"] = dt.dt.year * 12 + dt.dt.month
    def _deliv_ord(code: str) -> int | None:
        ym = contract_ym(code)
        if ym is None or ym[1] < 1 or ym[1] > 12:
            return None
        return ym[0] * 12 + ym[1]

    m["near_deliv_ord"] = m["near_code"].map(_deliv_ord)
    m["in_deliv_month"] = m["trade_ord"] == m["near_deliv_ord"]
    m["months_to_deliv"] = m["near_deliv_ord"] - m["trade_ord"]
    m.loc[m["near_deliv_ord"].isna(), "in_deliv_month"] = False

    dtd = []
    for r in m.itertuples():
        ym = contract_ym(r.near_code)
        if ym is None or ym[1] < 1 or ym[1] > 12:
            dtd.append(np.nan)
            continue
        y, mm = ym
        target = pd.Timestamp(y, mm, 1)
        dtd.append(max(0, (target - pd.Timestamp(r.date)).days))
    m["days_to_deliv"] = dtd
    m["pre_deliv"] = (m["days_to_deliv"] <= 15) & (m["days_to_deliv"] > 0)

    m["carry_pct"] = (m["near_close"] - m["far_close"]) / m["near_close"].replace(0, np.nan)
    oi_sum = m["near_oi"].fillna(0) + m["far_oi"].fillna(0)
    m["near_oi_share"] = np.where(oi_sum > 0, m["near_oi"] / oi_sum, np.nan)

    m["main_switched"] = m["main_code"] != m["main_code"].shift(1)
    m["main_switched"] = m["main_switched"].fillna(False)

    pre_n, post_n = 5, 5
    roll_mask = np.zeros(len(m), dtype=bool)
    pre_roll = np.zeros(len(m), dtype=bool)
    ev_idx = np.where(m["main_switched"].to_numpy())[0]
    for i in ev_idx:
        lo = max(0, i - pre_n)
        hi = min(len(m), i + post_n + 1)
        roll_mask[lo:hi] = True
        pre_roll[lo:i] = True
    m["in_roll_window"] = roll_mask
    m["pre_roll"] = pre_roll

    m["cal_month"] = dt.dt.month
    m["cal_quarter_end"] = dt.dt.month.isin([3, 6, 9, 12]) & (dt.dt.day >= 15)

    m["pnl_long_near"] = [
        leg_pnl(symbol, r.near_close, r.near_fwd, +1) for r in m.itertuples()
    ]
    m["pnl_short_near"] = [-x for x in m["pnl_long_near"]]
    m["pnl_long_main"] = [
        leg_pnl(symbol, r.main_close, r.main_fwd, +1) for r in m.itertuples()
    ]
    m["pnl_long_carry"] = [
        leg_pnl(symbol, r.near_close, r.near_fwd, +1) - leg_pnl(symbol, r.far_close, r.far_fwd, +1)
        for r in m.itertuples()
    ]

    m = m.sort_values("date")
    for win in (20, 60):
        mu = m["carry_pct"].rolling(win, min_periods=max(5, win // 3)).mean()
        sd = m["carry_pct"].rolling(win, min_periods=max(5, win // 3)).std()
        m[f"carry_z{win}"] = (m["carry_pct"] - mu) / sd.replace(0, np.nan)
    m["main_mom5"] = m["main_close"].pct_change(5).shift(1)

    m["symbol"] = symbol
    m["roll_offset"] = roll_offset
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-01-01")
    args = parser.parse_args()
    ensure_dirs()
    cfg = yaml.safe_load((CONFIG_DIR / "calendar.yaml").read_text(encoding="utf-8"))
    start = pd.Timestamp(args.start)
    roll_offset = int(cfg.get("roll_offset", 3))

    rows_meta = []
    parts = []
    for sym in cfg.get("symbols", []):
        print(f"[calendar] {sym}...", flush=True)
        df = build_symbol_panel(sym, roll_offset=roll_offset)
        if df.empty:
            print("  skip empty", flush=True)
            continue
        df = df[df["date"] >= start]
        path = PANEL_DIR / f"cal_{sym.lower()}.parquet"
        df.to_parquet(path, index=False)
        rows_meta.append({
            "symbol": sym,
            "days": len(df),
            "from": str(df["date"].min().date()),
            "to": str(df["date"].max().date()),
            "path": path.name,
        })
        parts.append(df)
        print(f"  {len(df)} days -> {path.name}", flush=True)

    if parts:
        all_p = pd.concat(parts, ignore_index=True)
        all_p.to_parquet(PANEL_DIR / "all_calendar.parquet", index=False)
        print(f"[all] {len(all_p)} rows, {all_p['symbol'].nunique()} symbols", flush=True)

    meta = {"symbols": rows_meta, "start": str(start.date()), "roll_offset": roll_offset}
    (PANEL_DIR / "manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print("done.", flush=True)


if __name__ == "__main__":
    main()
