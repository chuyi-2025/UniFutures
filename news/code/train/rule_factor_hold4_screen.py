#!/usr/bin/env python3
"""Screen simple rule factors: max 4 names (2L+2S), sticky hold, median hold > 7d."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from build_news_factors import _weights  # noqa: E402
from common import DATA_DIR, RESULTS_ROOT  # noqa: E402
from linear_ridge_walkforward import (  # noqa: E402
    MIN_CS,
    build_panel,
    ic_stats,
    metrics_from_pnl,
    rank_ic_daily,
)
from rule_factor_mine import batch_aggregate, filter_docs  # noqa: E402
from six_name_select import pick_ls_n  # noqa: E402
from two_name_select import apply_hold  # noqa: E402

BT_START = pd.Timestamp("2024-03-08")
OUT = RESULTS_ROOT / "rule_hold4"
N_EACH = 2  # 2L + 2S = 4 names max
MIN_MEDIAN_HOLD = 7

# Simple search space only
KINDS = ("all", "no_dianping", "strategy")
COLS = ("rule_edge", "rule_lex", "core_lex")
LBS = (7, 14, 30, 60)
WTS = ("uniform",)  # keep weighting simple
MIN_DOCS = (0, 1, 2)
HOLD_DAYS = (7, 10, 14, 20, 30)


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
    if not np.isfinite(cagr):
        return float("nan")
    return float(cagr / abs(dd))


def run_4name_sticky(
    panel: pd.DataFrame,
    fac: pd.DataFrame,
    hold_days: int,
    min_docs: int,
) -> tuple[pd.Series, float, float]:
    """2L+2S sticky hold; return pnl series, median hold, mean names/day."""
    p = panel.merge(fac, on=["date", "symbol"], how="left")
    dates = sorted(p["date"].unique())

    picks: list[tuple[str, float]] | None = None
    days_left = 0
    sym_days: dict[str, int] = {}
    closed_streaks: list[int] = []
    pnls: list[tuple[pd.Timestamp, float]] = []
    names_per_day: list[int] = []

    for dt in dates:
        day = p[p["date"] == dt].drop_duplicates("symbol").reset_index(drop=True)

        if days_left <= 0:
            d = day.dropna(subset=["val", "fwd_ret"])
            if min_docs > 0:
                d = d[d["n_docs"] >= min_docs]
            if len(d) >= 2 * N_EACH:
                new = pick_ls_n(d, d["val"].to_numpy(), N_EACH)
                if new:
                    new_syms = {s for s, _ in new}
                    for sym, streak in list(sym_days.items()):
                        if sym not in new_syms:
                            closed_streaks.append(streak)
                            del sym_days[sym]
                    picks = new
                    days_left = hold_days

        if not picks:
            continue

        r = apply_hold(day, picks)
        if np.isfinite(r):
            pnls.append((dt, r))
            names_per_day.append(len(picks))
            for sym, _ in picks:
                sym_days[sym] = sym_days.get(sym, 0) + 1
            days_left -= 1

    closed_streaks.extend(sym_days.values())
    med_hold = float(np.median(closed_streaks)) if closed_streaks else 0.0
    mean_names = float(np.mean(names_per_day)) if names_per_day else 0.0
    return pd.Series({d: v for d, v in pnls}).sort_index(), med_hold, mean_names


def eval_row(
    panel: pd.DataFrame,
    fac: pd.DataFrame,
    name: str,
    hold_days: int,
    min_docs: int,
) -> dict:
    ic_p = panel.merge(fac, on=["date", "symbol"], how="left")
    ic_p = ic_p.dropna(subset=["val", "fwd_ret"])
    if min_docs > 0:
        ic_p = ic_p[ic_p["n_docs"] >= min_docs]
    ic = ic_stats(rank_ic_daily(ic_p, "val")) if ic_p["date"].nunique() >= 30 else {}

    pnl, med_hold, mean_names = run_4name_sticky(panel, fac, hold_days, min_docs)
    m = metrics_from_pnl(pnl) if len(pnl) else {"return": np.nan, "sharpe": np.nan, "max_dd": np.nan, "days": 0}
    cal = calmar_from_pnl(pnl)

    return {
        "name": name,
        "hold_days": hold_days,
        "min_docs": min_docs,
        "ic": round(float(ic.get("ic", np.nan)), 4) if ic.get("ic") == ic.get("ic") else None,
        "ir": round(float(ic.get("ir", np.nan)), 3) if ic.get("ir") == ic.get("ir") else None,
        "ic_days": int(ic.get("n") or 0),
        "return": round(float(m["return"]), 4) if np.isfinite(m["return"]) else None,
        "sharpe": round(float(m["sharpe"]), 3) if np.isfinite(m["sharpe"]) else None,
        "max_dd": round(float(m["max_dd"]), 4) if np.isfinite(m["max_dd"]) else None,
        "calmar": round(float(cal), 3) if np.isfinite(cal) else None,
        "trade_days": int(m["days"]),
        "median_hold": round(med_hold, 1),
        "mean_names": round(mean_names, 2),
        "pass_hold": med_hold > MIN_MEDIAN_HOLD,
        "pass_names": mean_names <= 4.0 + 1e-9,
    }


def main() -> None:
    rich_path = DATA_DIR / "rule_docs_rich.parquet"
    if not rich_path.is_file():
        from rule_features import scan_rich_docs

        scan_rich_docs().to_parquet(rich_path, index=False)
    docs = pd.read_parquet(rich_path)
    docs["report_date"] = pd.to_datetime(docs["report_date"])

    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    kind_cache = {k: filter_docs(docs, k) for k in KINDS}

    fac_cache: dict[str, pd.DataFrame] = {}
    for kind in KINDS:
        sub = kind_cache[kind]
        if sub.empty:
            continue
        for col in COLS:
            if col not in sub.columns:
                continue
            for lb in LBS:
                for wt in WTS:
                    key = f"{col}|{kind}|L{lb}|{wt}"
                    print(f"agg {key}", flush=True)
                    fac_cache[key] = batch_aggregate(sub, col, lb, dates, wt)

    rows: list[dict] = []
    for key, fac in fac_cache.items():
        for hold in HOLD_DAYS:
            for md in MIN_DOCS:
                tag = f"{key}|hold{hold}|md{md}"
                rows.append(eval_row(panel, fac, tag, hold, md))

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "all_candidates.csv", index=False)

    ok = df[df["pass_hold"] & df["pass_names"]].copy()
    ok = ok.sort_values(["calmar", "sharpe"], ascending=False, na_position="last")
    ok.to_csv(OUT / "filtered.csv", index=False)
    top = ok.head(20)
    top.to_csv(OUT / "top20.csv", index=False)

    (OUT / "report.json").write_text(
        json.dumps(
            {
                "constraints": {"max_names": 4, "min_median_hold": MIN_MEDIAN_HOLD},
                "n_total": len(df),
                "n_pass": len(ok),
                "top10": top.head(10).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== PASS (median_hold>7, names<=4) TOP 10 by Calmar ===", flush=True)
    cols = ["name", "ic", "ir", "sharpe", "calmar", "max_dd", "return", "median_hold", "trade_days"]
    print(top[cols].head(10).to_string(index=False), flush=True)
    print(f"\nn_pass={len(ok)}/{len(df)} saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
