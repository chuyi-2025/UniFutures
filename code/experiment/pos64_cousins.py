#!/usr/bin/env python3
"""Cross-section LS horse race for pos_64 cousins (2010-2026)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from linear_ridge_walkforward import EVAL_END, OUT_DIR, build_panel, ls_pnl, metrics_from_pnl
from two_model_rotate import BT_START

# Same directional use as pos_64: high score → long, low → short.
TREND = [
    "pos_64",
    "ratio_max_7",
    "ratio_max_30",
    "ratio_min_7",
    "ratio_min_30",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "momentum_30",
    "ma_ratio_5_20",
    "ma_ratio_10_30",
    "ma_ratio_10_60",
    "ma_ratio_20_60",
    "ma_ratio_30_60",
    "ma_diff_10_30",
    "ma_diff_10_60",
    "ma_diff_30_60",
    "bias_5",
    "bias_10",
    "bias_30",
    "avg_price_break_5",
    "avg_price_break_10",
    "avg_price_break_30",
    "rsi",
    "ret_mean_5",
    "ret_mean_20",
    "vol_pos_64",
]


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def main() -> None:
    print("loading panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01"))
    core_end = pd.Timestamp("2026-06-30")
    have = [c for c in TREND if c in panel.columns]
    pnls = {c: [] for c in have}
    print("scoring daily LS...", flush=True)
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        for col in have:
            if col not in day.columns:
                continue
            pnls[col].append((dt, ls_pnl(day, day[col].to_numpy())))

    table = []
    for col in have:
        s = pd.Series({d: v for d, v in pnls[col]}).sort_index().dropna()
        s_core = s[s.index <= core_end]
        m = metrics_from_pnl(s)
        mc = metrics_from_pnl(s_core)
        table.append(
            {
                "factor": col,
                "return": fmt(m["return"], 4),
                "sharpe": fmt(m["sharpe"], 3),
                "max_dd": fmt(m["max_dd"], 4),
                "days": m["days"],
                "core_return": fmt(mc["return"], 4),
                "core_sharpe": fmt(mc["sharpe"], 3),
            }
        )
        print(
            f"{col:22} ret={m['return']:+.1%} sh={m['sharpe']:.2f} dd={m['max_dd']:.1%}",
            flush=True,
        )

    table.sort(key=lambda r: (r["sharpe"] is None, -(r["sharpe"] or -999)))
    path = OUT_DIR / "pos64_cousins.json"
    path.write_text(json.dumps({"window": "2010-07-01 ~ 2026-09-07", "ls": "top/bottom 20%", "factors": table}, ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
