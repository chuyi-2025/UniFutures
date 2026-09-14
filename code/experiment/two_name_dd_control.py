#!/usr/bin/env python3
"""Drawdown controls on the daily 1-long/1-short book (2010-2026)."""

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
from linear_ridge_walkforward import (
    EVAL_END,
    L2,
    MIN_TRAIN_DAYS,
    OUT_DIR,
    SKIP_FEATS,
    build_panel,
    fit_ridge,
    ic_stats,
    metrics_from_pnl,
    predict_ridge,
    rank_ic_daily,
)
from two_model_rotate import BT_START, choose_model, daily_mkt
from two_name_select import apply_hold, pick_ls


def build_raw_book() -> pd.Series:
    panel = build_panel(pd.Timestamp("2010-01-01")).dropna(subset=["fwd_ret", "pos_64"])
    feats = [c for c in FEATURE_NAMES if c not in SKIP_FEATS and c in panel.columns]
    mkt = daily_mkt(panel)
    ic_map = {f: rank_ic_daily(panel, f) for f in feats}
    rows = []
    quarters = pd.period_range("2010Q3", "2026Q3", freq="Q")
    for q in quarters:
        q_start, q_end = q.start_time, min(q.end_time, EVAL_END)
        test = panel[(panel["date"] >= q_start) & (panel["date"] <= q_end)].copy()
        train = panel[panel["date"] < q_start]
        if test["date"].nunique() < 15 or test["date"].min() < BT_START:
            continue
        pick, _ = choose_model(mkt, q_start)
        if pick == "ridge" and train["date"].nunique() >= MIN_TRAIN_DAYS:
            ranked = [{"feature": f, **ic_stats(ic_map[f].loc[ic_map[f].index < q_start])} for f in feats]
            rtab = pd.DataFrame(ranked).sort_values("t", key=lambda s: s.abs(), ascending=False)
            usable = rtab[(rtab["t"].abs() >= 1.5) & rtab["ic"].notna()]
            use = usable["feature"].head(12).tolist() or rtab.head(8)["feature"].tolist()
            mu, sd, beta = fit_ridge(train, use, l2=L2)
            test["score"] = predict_ridge(test, use, mu, sd, beta)
        else:
            test["score"] = test["pos_64"]
        for dt, day in test.groupby("date"):
            w = pick_ls(day, day["score"].to_numpy(), False)
            rows.append((dt, apply_hold(day, w)))
    s = pd.Series({d: v for d, v in rows}).sort_index().dropna()
    return s[s.index >= BT_START]


def flatten_after_loss(raw: pd.Series, thresh: float, cool: int) -> pd.Series:
    out = raw.copy()
    skip = 0
    prev = 0.0
    for i, (dt, v) in enumerate(raw.items()):
        if skip > 0:
            out.iloc[i] = 0.0
            skip -= 1
            prev = 0.0
            continue
        out.iloc[i] = v
        if v < thresh:
            skip = cool
        prev = v
    return out


def halt_on_dd(raw: pd.Series, dd_lim: float, cool: int) -> pd.Series:
    out = raw.copy()
    eq = 1.0
    peak = 1.0
    skip = 0
    for i, v in enumerate(raw.to_numpy()):
        if skip > 0:
            out.iloc[i] = 0.0
            skip -= 1
            continue
        eq *= np.exp(v)
        peak = max(peak, eq)
        dd = eq / peak - 1.0
        out.iloc[i] = v
        if dd <= -abs(dd_lim):
            skip = cool
    return out


def resume_when_recovered(raw: pd.Series, dd_lim: float, recover: float) -> pd.Series:
    """Halt after live DD hits lim; stay flat until paper book DD is back above recover."""
    out = raw.copy()
    live = 1.0
    live_peak = 1.0
    paper = 1.0
    paper_peak = 1.0
    halted = False
    for i, v in enumerate(raw.to_numpy()):
        paper *= np.exp(v)
        paper_peak = max(paper_peak, paper)
        if halted:
            out.iloc[i] = 0.0
            if paper / paper_peak - 1.0 > -abs(recover):
                halted = False
            continue
        live *= np.exp(v)
        live_peak = max(live_peak, live)
        out.iloc[i] = v
        if live / live_peak - 1.0 <= -abs(dd_lim):
            halted = True
            paper, paper_peak = live, live_peak
    return out


def vol_target(raw: pd.Series, target: float, lookback: int = 20, cap: float = 2.0) -> pd.Series:
    vol = raw.rolling(lookback).std(ddof=1) * np.sqrt(252)
    scale = (target / vol.shift(1)).clip(upper=cap)
    scale = scale.fillna(1.0)
    return raw * scale


def clip_intraday(raw: pd.Series, stop: float) -> pd.Series:
    """Optimistic: if the day would lose more than stop, realize -stop."""
    return raw.clip(lower=-abs(stop))


def time_in_mkt(s: pd.Series) -> float:
    return float((s.abs() > 1e-12).mean())


def pack(s: pd.Series) -> dict:
    m = metrics_from_pnl(s)
    m["time_in_mkt"] = time_in_mkt(s)
    return m


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main() -> None:
    print("building raw 1L1S book...", flush=True)
    raw = build_raw_book()
    print(f"raw days={len(raw)} {raw.index.min().date()} -> {raw.index.max().date()}", flush=True)

    schemes = {
        "无控制": raw,
        "日亏>1%后空仓5日": flatten_after_loss(raw, -0.01, 5),
        "日亏>2%后空仓5日": flatten_after_loss(raw, -0.02, 5),
        "日亏>3%后空仓5日": flatten_after_loss(raw, -0.03, 5),
        "回撤>8%空仓10日": halt_on_dd(raw, 0.08, 10),
        "回撤>10%空仓10日": halt_on_dd(raw, 0.10, 10),
        "回撤>12%空仓10日": halt_on_dd(raw, 0.12, 10),
        "回撤>10%直到收复到6%": resume_when_recovered(raw, 0.10, 0.06),
        "波动瞄准10%": vol_target(raw, 0.10),
        "波动瞄准12%": vol_target(raw, 0.12),
        "波动10%+回撤10%熔断": halt_on_dd(vol_target(raw, 0.10), 0.10, 10),
        "日内亏损封顶2%(乐观)": clip_intraday(raw, 0.02),
        "日内亏损封顶3%(乐观)": clip_intraday(raw, 0.03),
    }

    table = []
    for name, s in schemes.items():
        m = pack(s)
        table.append({"scheme": name, **m})
        print(
            f"{name:22} ret={m['return']:+.1%} sh={m['sharpe']:.2f} "
            f"dd={m['max_dd']:.1%} in={m['time_in_mkt']:.0%}",
            flush=True,
        )

    # yearly equity for best honest schemes
    def year_eq(s):
        eq = np.exp(s.cumsum())
        return [{"date": str(i.date())[:4], "equity": float(v)} for i, v in eq.resample("YE").last().items()]

    keep = [
        "无控制",
        "波动瞄准10%",
        "回撤>10%空仓10日",
        "波动10%+回撤10%熔断",
        "日亏>1%后空仓5日",
        "日亏>2%后空仓5日",
    ]

    for thresh in (-0.01, -0.02, -0.03):
        n = int((raw < thresh).sum())
        print(f"raw days worse than {thresh:.0%}: {n} / {len(raw)} ({n / len(raw):.1%})", flush=True)
    payload = {
        "baseline": "daily 1 long + 1 short on quarterly pos64/Ridge book",
        "bar": "daily close; halt/vol use only information up to yesterday except optimistic clip",
        "schemes": [
            {
                "scheme": r["scheme"],
                "return": fmt(r["return"], 4),
                "sharpe": fmt(r["sharpe"], 3),
                "max_dd": fmt(r["max_dd"], 4),
                "days": r["days"],
                "time_in_mkt": fmt(r["time_in_mkt"], 3),
            }
            for r in table
        ],
        "equity_year": {k: year_eq(schemes[k]) for k in keep},
    }
    path = OUT_DIR / "two_name_dd_control.json"
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
