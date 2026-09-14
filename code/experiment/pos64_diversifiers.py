#!/usr/bin/env python3
"""Find factors unlike pos_64: other families, low score/PnL correlation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from build_feature.build_xgb_feature import FEATURE_NAMES
from linear_ridge_walkforward import EVAL_END, OUT_DIR, SKIP_FEATS, build_panel, ls_pnl, metrics_from_pnl
from two_model_rotate import BT_START

FAMILY = {
    "ret_1d": "隔夜收益",
    "volume_ret": "成交量",
    "volume_ratio_30": "成交量",
    "volume_ma_ratio_10_30": "成交量",
    "volume_ma_ratio_30_60": "成交量",
    "vol_pos_64": "成交量",
    "price_volume_ratio": "流动性",
    "turnover_rate": "流动性",
    "liquidity_10": "流动性",
    "liquidity_30": "流动性",
    "volatility_10": "波动率",
    "volatility_30": "波动率",
    "volatility_60": "波动率",
    "ret_std_5": "波动率",
    "ret_std_20": "波动率",
    "range_pct_7": "振幅",
    "range_pct_30": "振幅",
    "ratio_maxmin_7": "振幅",
    "ratio_maxmin_30": "振幅",
    "hl_spread": "振幅",
    "skew_7": "高阶矩",
    "skew_30": "高阶矩",
    "kurt_7": "高阶矩",
    "kurt_30": "高阶矩",
    "money_flow_ratio_10": "资金流",
    "money_flow_ratio_30": "资金流",
    "body_ratio": "K线结构",
    "upper_shadow": "K线结构",
    "lower_shadow": "K线结构",
    "gap": "跳空",
}

TREND = {
    "pos_64",
    "rsi",
    "ratio_min_30",
    "ratio_max_30",
    "ratio_min_7",
    "ratio_max_7",
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
    "ret_mean_5",
    "ret_mean_20",
}


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def cs_spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 8:
        return np.nan
    ra = pd.Series(a[mask]).rank().to_numpy()
    rb = pd.Series(b[mask]).rank().to_numpy()
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    print("loading panel...", flush=True)
    panel = build_panel(pd.Timestamp("2010-01-01"))
    cols = [c for c in FEATURE_NAMES if c not in SKIP_FEATS and c in panel.columns]
    others = [c for c in cols if c not in TREND]
    # keep pos_64 as benchmark inside the loop
    score_cols = ["pos_64", *others]
    pnls: dict[str, list] = {c: [] for c in score_cols}
    corrs: dict[str, list] = {c: [] for c in others}

    print(f"scoring {len(others)} non-trend factors...", flush=True)
    n_days = 0
    for dt, day in panel.groupby("date"):
        if dt < BT_START or dt > EVAL_END:
            continue
        n_days += 1
        p64 = day["pos_64"].to_numpy() if "pos_64" in day.columns else None
        for col in score_cols:
            pnls[col].append((dt, ls_pnl(day, day[col].to_numpy())))
        if p64 is not None:
            for col in others:
                corrs[col].append(cs_spearman(day[col].to_numpy(), p64))

    p64s = pd.Series({d: v for d, v in pnls["pos_64"]}).sort_index()
    rows = []
    for col in others:
        s = pd.Series({d: v for d, v in pnls[col]}).sort_index().dropna()
        aligned = pd.concat([s.rename("f"), p64s.rename("p")], axis=1).dropna()
        m = metrics_from_pnl(s)
        mneg = metrics_from_pnl(-s)
        better_flip = abs(mneg["sharpe"]) > abs(m["sharpe"]) and mneg["sharpe"] > m["sharpe"]
        # choose the sign with higher Sharpe (trend vs reversal)
        if mneg["sharpe"] > m["sharpe"]:
            chosen, sign = mneg, -1
        else:
            chosen, sign = m, +1
        score_corr = float(np.nanmean(corrs[col])) if corrs[col] else np.nan
        pnl_corr = float(aligned["f"].corr(aligned["p"])) if len(aligned) > 20 else np.nan
        rows.append(
            {
                "factor": col,
                "family": FAMILY.get(col, "其他"),
                "sign": sign,
                "return": fmt(chosen["return"], 4),
                "sharpe": fmt(chosen["sharpe"], 3),
                "max_dd": fmt(chosen["max_dd"], 4),
                "raw_sharpe": fmt(m["sharpe"], 3),
                "score_corr_pos64": fmt(score_corr, 3),
                "pnl_corr_pos64": fmt(pnl_corr, 3),
                "days": chosen["days"],
            }
        )
        print(
            f"{FAMILY.get(col, '?'):6} {col:22} sign={sign:+d} "
            f"sh={chosen['sharpe']:.2f} scρ={score_corr:+.2f} pnlρ={pnl_corr:+.2f}",
            flush=True,
        )

    rows.sort(key=lambda r: (r["sharpe"] is None, -(r["sharpe"] or -999)))
    # most different among those with some edge
    divers = [r for r in rows if r["sharpe"] is not None and r["sharpe"] >= 0.25]
    divers.sort(key=lambda r: abs(r["pnl_corr_pos64"] or 99))
    payload = {
        "window": "2010-07-01 ~ 2026-09-07",
        "ls": "top/bottom 20%; sign chosen to maximize Sharpe",
        "benchmark": "pos_64",
        "ranked_by_sharpe": rows,
        "low_pnl_corr_with_edge": divers[:12],
        "n_days": n_days,
    }
    path = OUT_DIR / "pos64_diversifiers.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
