#!/usr/bin/env python3
"""Dynamic hold / variable names for fixed rule_edge|no_dianping|L7 signal."""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from common import DATA_DIR, RESULTS_ROOT  # noqa: E402
from linear_ridge_walkforward import build_panel, ic_stats, metrics_from_pnl, rank_ic_daily  # noqa: E402
from rule_factor_mine import batch_aggregate, filter_docs  # noqa: E402
from six_name_select import pick_ls_n  # noqa: E402
from two_name_select import apply_hold  # noqa: E402

BT_START = pd.Timestamp("2024-03-08")
OUT = RESULTS_ROOT / "rule_dynamic"
MAX_NAMES = 4


def calmar_from_pnl(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    if len(pnl) < 30:
        return float("nan")
    m = metrics_from_pnl(pnl)
    dd = m["max_dd"]
    if not np.isfinite(dd) or dd >= -1e-6:
        return float("nan")
    years = len(pnl) / 252.0
    cagr = (1.0 + m["return"]) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    return float(cagr / abs(dd)) if np.isfinite(cagr) else float("nan")


def day_spread(scores: pd.Series) -> float:
    if len(scores) < 2:
        return 0.0
    return float(scores.max() - scores.min())


def pick_variable(
    day: pd.DataFrame,
    *,
    abs_thr: float,
    max_each: int,
    spread: float,
    spread_hi: float,
) -> list[tuple[str, float]]:
    d = day.dropna(subset=["val", "fwd_ret"]).copy()
    if abs_thr > 0:
        d = d[d["val"].abs() >= abs_thr]
    if len(d) < 2:
        return []
    k = 1 if spread < spread_hi else min(max_each, len(d) // 2)
    if k < 1:
        return []
    return pick_ls_n(d, d["val"].to_numpy(), k)


def run_dynamic(
    panel: pd.DataFrame,
    fac: pd.DataFrame,
    *,
    spread_enter: float,
    spread_exit: float,
    min_hold: int,
    abs_thr: float,
    max_each: int,
    spread_hi: float,
    min_docs: int,
    mode: str,
) -> tuple[pd.Series, dict]:
    """mode: hysteresis (enter/exit+min_hold) | daily (flat if weak each day)."""
    p = panel.merge(fac, on=["date", "symbol"], how="left")
    dates = sorted(p["date"].unique())

    picks: list[tuple[str, float]] | None = None
    days_in = 0
    sym_streak: dict[str, int] = {}
    closed: list[int] = []
    pnls: list[tuple[pd.Timestamp, float]] = []
    names_list: list[int] = []
    empty_days = 0

    for dt in dates:
        day = p[p["date"] == dt].drop_duplicates("symbol").reset_index(drop=True)
        d = day.dropna(subset=["val", "fwd_ret"])
        if min_docs > 0:
            d = d[d["n_docs"] >= min_docs]
        spread = day_spread(d["val"]) if len(d) >= 2 else 0.0

        if mode == "daily":
            if spread >= spread_enter:
                new = pick_variable(
                    d, abs_thr=abs_thr, max_each=max_each, spread=spread, spread_hi=spread_hi
                )
                picks = new if new else None
            else:
                picks = None
            days_in = 0
        else:
            # hysteresis: enter when strong, exit when weak after min_hold
            if picks is None:
                if spread >= spread_enter:
                    new = pick_variable(
                        d, abs_thr=abs_thr, max_each=max_each, spread=spread, spread_hi=spread_hi
                    )
                    if new:
                        picks = new
                        days_in = 0
            else:
                days_in += 1
                if days_in >= min_hold and spread < spread_exit:
                    for sym in list(sym_streak.keys()):
                        closed.append(sym_streak.pop(sym))
                    picks = None
                    days_in = 0
                elif days_in >= min_hold and spread >= spread_enter:
                    # optional refresh when still strong
                    new = pick_variable(
                        d, abs_thr=abs_thr, max_each=max_each, spread=spread, spread_hi=spread_hi
                    )
                    if new:
                        old = {s for s, _ in picks}
                        new_s = {s for s, _ in new}
                        if old != new_s:
                            for sym in old - new_s:
                                if sym in sym_streak:
                                    closed.append(sym_streak.pop(sym))
                            picks = new
                            days_in = 0

        if not picks:
            empty_days += 1
            continue

        r = apply_hold(day, picks)
        if np.isfinite(r):
            pnls.append((dt, r))
            names_list.append(len(picks))
            for sym, _ in picks:
                sym_streak[sym] = sym_streak.get(sym, 0) + 1

    closed.extend(sym_streak.values())
    total_days = len(dates)
    pnl_s = pd.Series({d: v for d, v in pnls}).sort_index()
    stats = {
        "median_hold": float(np.median(closed)) if closed else 0.0,
        "mean_names": float(np.mean(names_list)) if names_list else 0.0,
        "max_names": int(max(names_list)) if names_list else 0,
        "empty_ratio": round(empty_days / total_days, 3) if total_days else 1.0,
        "active_days": len(pnl_s),
    }
    return pnl_s, stats


def eval_config(panel, fac, cfg: dict) -> dict:
    ic_p = panel.merge(fac, on=["date", "symbol"], how="left").dropna(subset=["val", "fwd_ret"])
    if cfg["min_docs"] > 0:
        ic_p = ic_p[ic_p["n_docs"] >= cfg["min_docs"]]
    ic = ic_stats(rank_ic_daily(ic_p, "val")) if ic_p["date"].nunique() >= 30 else {}

    pnl, st = run_dynamic(panel, fac, **{k: cfg[k] for k in cfg if k != "name"})
    m = metrics_from_pnl(pnl) if len(pnl) else {"return": np.nan, "sharpe": np.nan, "max_dd": np.nan, "days": 0}
    cal = calmar_from_pnl(pnl)

    yearly_sh = []
    if len(pnl):
        for _, g in pnl.groupby(pnl.index.year):
            mm = metrics_from_pnl(g)
            yearly_sh.append(mm["sharpe"])

    return {
        **cfg,
        "ic": round(float(ic.get("ic", np.nan)), 4) if ic.get("ic") == ic.get("ic") else None,
        "ir": round(float(ic.get("ir", np.nan)), 3) if ic.get("ir") == ic.get("ir") else None,
        "return": round(float(m["return"]), 4) if np.isfinite(m["return"]) else None,
        "sharpe": round(float(m["sharpe"]), 3) if np.isfinite(m["sharpe"]) else None,
        "max_dd": round(float(m["max_dd"]), 4) if np.isfinite(m["max_dd"]) else None,
        "calmar": round(float(cal), 3) if np.isfinite(cal) else None,
        "trade_days": int(m["days"]),
        "n_neg_years": sum(1 for s in yearly_sh if np.isfinite(s) and s < 0),
        **{k: round(v, 3) if isinstance(v, float) else v for k, v in st.items()},
    }


def main() -> None:
    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    sub = filter_docs(docs, "no_dianping")
    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    fac = batch_aggregate(sub, "rule_edge", 7, dates, "uniform")
    print("signal: rule_edge|no_dianping|L7|uniform", flush=True)

    configs: list[dict] = []
    for mode in ("hysteresis", "daily"):
        for spread_enter, spread_exit, min_hold, abs_thr, max_each, spread_hi, min_docs in product(
            (0.30, 0.40, 0.50, 0.60),
            (0.10, 0.15, 0.20, 0.25),
            (5, 7, 10, 14),
            (0.0, 0.15, 0.25),
            (1, 2),
            (0.45, 0.55, 0.65),
            (0, 1),
        ):
            if spread_exit >= spread_enter:
                continue
            if mode == "daily" and min_hold != 5:
                continue  # min_hold unused in daily mode
            name = (
                f"{mode}|enter{spread_enter}|exit{spread_exit}|hold{min_hold}"
                f"|abs{abs_thr}|max{max_each}|hi{spread_hi}|md{min_docs}"
            )
            configs.append({
                "name": name,
                "mode": mode,
                "spread_enter": spread_enter,
                "spread_exit": spread_exit,
                "min_hold": min_hold,
                "abs_thr": abs_thr,
                "max_each": max_each,
                "spread_hi": spread_hi,
                "min_docs": min_docs,
            })

    rows = []
    for i, cfg in enumerate(configs):
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(configs)}]", flush=True)
        rows.append(eval_config(panel, fac, cfg))

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "all_candidates.csv", index=False)

    ok = df[(df["max_names"] <= MAX_NAMES) & (df["trade_days"] >= 60)].copy()
    ok = ok.sort_values(["calmar", "sharpe"], ascending=False, na_position="last")
    ok.to_csv(OUT / "top_ranked.csv", index=False)

    # prefer: calmar high, n_neg_years=0, empty_ratio not too high
    robust = ok[(ok["n_neg_years"] == 0) & (ok["empty_ratio"] <= 0.7)].sort_values(
        ["calmar", "sharpe"], ascending=False
    )
    robust.to_csv(OUT / "top_robust.csv", index=False)

    top = robust.head(15) if len(robust) else ok.head(15)
    (OUT / "report.json").write_text(
        json.dumps({"top15": top.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cols = [
        "name", "sharpe", "calmar", "max_dd", "return", "median_hold",
        "mean_names", "empty_ratio", "trade_days", "n_neg_years",
    ]
    print("\n=== TOP 15 (dynamic, allow flat) ===", flush=True)
    print(top[cols].to_string(index=False), flush=True)
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
