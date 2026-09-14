#!/usr/bin/env python3
"""Blend z(RSI)-z(PVR)+z(vol_pos) 6-hand book + BOOK lot overlay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from family_rotate import fmt, year_pack
from linear_ridge_walkforward import EVAL_END, OUT_DIR, build_panel
from six_name_select import pick_ls_n
from two_model_rotate import BT_START
from two_name_select import apply_hold

SHORT, LONG, THR = 20, 252, 1.1
PARTS = [("rsi", 1), ("price_volume_ratio", -1), ("vol_pos_64", 1)]


def zscore(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    m = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if m.sum() < 5:
        return out
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return out
    out[m] = (x[m] - np.nanmean(x)) / sd
    return out


def book_ratio(pnl: pd.Series, short: int = SHORT, long: int = LONG) -> pd.Series:
    vol = pnl.rolling(short).std(ddof=1) * np.sqrt(252)
    med = vol.rolling(long).median()
    return (vol / med).shift(1)


def apply_switch(n6: pd.Series, n2: pd.Series, ratio: pd.Series, thr: float = THR) -> tuple[pd.Series, pd.Series]:
    r = ratio.reindex(n6.index)
    use2 = r >= thr
    use2 = use2.fillna(False)
    out = n6.copy()
    out[use2] = n2.reindex(n6.index)[use2]
    return out, use2


def live_switch(n6: pd.Series, n2: pd.Series, thr: float = THR) -> tuple[pd.Series, pd.Series]:
    """BOOK from live mixed pnl up to yesterday."""
    idx = n6.index
    live = np.zeros(len(idx))
    use2 = np.zeros(len(idx), dtype=bool)
    buf: list[float] = []
    for i, dt in enumerate(idx):
        if len(buf) >= SHORT + 20:
            s = pd.Series(buf)
            vol = s.rolling(SHORT).std(ddof=1) * np.sqrt(252)
            med = float(vol.tail(LONG).median())
            last = float(vol.iloc[-1]) if np.isfinite(vol.iloc[-1]) else np.nan
            hot = bool(np.isfinite(last) and np.isfinite(med) and med > 0 and last / med >= thr)
        else:
            hot = False
        use2[i] = hot
        v = float(n2.loc[dt] if hot else n6.loc[dt])
        if not np.isfinite(v):
            v = 0.0
        live[i] = v
        buf.append(v)
    return pd.Series(live, index=idx), pd.Series(use2, index=idx)


def main() -> None:
    print("loading panel...", flush=True)
    cols = ["fwd_ret", "rsi", "price_volume_ratio", "vol_pos_64"]
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=cols)
    n6_rows, n2_rows = [], []
    print("daily 6-hand / 2-hand...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        sc = None
        for col, sgn in PARTS:
            z = sgn * zscore(day[col].to_numpy())
            sc = z if sc is None else sc + z
        n6_rows.append((dt, apply_hold(day, pick_ls_n(day, sc, 3))))
        n2_rows.append((dt, apply_hold(day, pick_ls_n(day, sc, 1))))

    n6 = pd.Series({d: v for d, v in n6_rows}).sort_index().dropna()
    n2 = pd.Series({d: v for d, v in n2_rows}).sort_index().dropna()
    n2 = n2.reindex(n6.index)

    ratio_6 = book_ratio(n6)
    scaled = n6 * np.where(ratio_6.reindex(n6.index).fillna(0) >= THR, 2.0 / 6.0, 1.0)
    switched, use2_sig = apply_switch(n6, n2, ratio_6)
    live, use2_live = live_switch(n6, n2)

    schemes = {
        "永远 6 手": n6,
        "永远 2 手": n2,
        "BOOK 缩仓(同6手×1/3)": scaled,
        "BOOK 换 1多1空(信号用6手)": switched,
        "BOOK 换 1多1空(用实盘波动)": live,
    }

    packed = {}
    for name, s in schemes.items():
        st = year_pack(s)
        if name == "BOOK 换 1多1空(信号用6手)":
            st["pct_2hand"] = fmt(float(use2_sig.mean()), 3)
        elif name == "BOOK 换 1多1空(用实盘波动)":
            st["pct_2hand"] = fmt(float(use2_live.mean()), 3)
        elif name == "BOOK 缩仓(同6手×1/3)":
            st["pct_2hand"] = fmt(float((ratio_6.reindex(n6.index).fillna(0) >= THR).mean()), 3)
        elif name == "永远 2 手":
            st["pct_2hand"] = 1.0
        else:
            st["pct_2hand"] = 0.0
        packed[name] = st
        print(
            f"{name:24} ret={st['return']:+.1%} sh={st['sharpe']:.2f} dd={st['max_dd']:.1%} "
            f"neg={st['n_neg_years']}{st['neg_years']} 2026={st['sharpe_2026']} "
            f"2h={st['pct_2hand']:.0%}",
            flush=True,
        )

    yearly_sw = []
    for y, g in switched.groupby(switched.index.year):
        m = year_pack(g)
        yearly_sw.append({
            "year": int(y),
            "return": m["return"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_dd"],
            "pct_2hand": fmt(float(use2_sig.reindex(g.index).mean()), 3),
        })
    packed["BOOK 换 1多1空(信号用6手)"]["yearly"] = yearly_sw

    path = OUT_DIR / "blend_book.json"
    path.write_text(json.dumps({
        "score": "z(rsi) - z(price_volume_ratio) + z(vol_pos_64)",
        "book": "20d vol / 252d median of that vol, shift 1, threshold 1.1",
        "schemes": packed,
    }, ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
