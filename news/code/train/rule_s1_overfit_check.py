#!/usr/bin/env python3
"""Overfit / generalization checks for S1 (rule_edge L30 single + hysteresis)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from common import DATA_DIR, RESULTS_ROOT  # noqa: E402
from linear_ridge_walkforward import build_panel, metrics_from_pnl  # noqa: E402
from rule_edge_dynamic_search import calmar_from_pnl  # noqa: E402
from rule_edge_single_sym_search import pnl_from_picks, precompute_days  # noqa: E402
from rule_factor_mine import batch_aggregate, filter_docs, keep_single_symbol_docs  # noqa: E402

BT_START = pd.Timestamp("2024-03-08")
OUT = RESULTS_ROOT / "rule_dynamic_single" / "overfit"
S1 = {
    "name": "S1",
    "mode": "hysteresis",
    "spread_enter": 0.40,
    "spread_exit": 0.10,
    "min_hold": 5,
    "abs_thr": 0.0,
    "max_each": 2,
    "spread_hi": 0.55,
    "min_docs": 0,
}


def summarize_pnl(pnl: pd.Series) -> dict:
    if pnl is None or len(pnl) < 10:
        return {
            "return": None, "sharpe": None, "max_dd": None, "calmar": None,
            "days": int(len(pnl) if pnl is not None else 0),
        }
    m = metrics_from_pnl(pnl)
    cal = calmar_from_pnl(pnl)
    return {
        "return": round(float(m["return"]), 4) if np.isfinite(m["return"]) else None,
        "sharpe": round(float(m["sharpe"]), 3) if np.isfinite(m["sharpe"]) else None,
        "max_dd": round(float(m["max_dd"]), 4) if np.isfinite(m["max_dd"]) else None,
        "calmar": round(float(cal), 3) if np.isfinite(cal) else None,
        "days": int(m["days"]),
    }


def build_signal(panel: pd.DataFrame, docs: pd.DataFrame, dates: pd.DatetimeIndex):
    sub = filter_docs(docs, "no_dianping")
    fac = batch_aggregate(sub, "rule_edge", 30, dates, "uniform")
    cache = precompute_days(panel, fac)
    return fac, cache


def quarterly_table(pnl: pd.Series) -> pd.DataFrame:
    rows = []
    q = pnl.copy()
    q.index = pd.to_datetime(q.index)
    for period, g in q.groupby(q.index.to_period("Q")):
        s = summarize_pnl(g)
        s["quarter"] = str(period)
        s["start"] = str(g.index.min().date())
        s["end"] = str(g.index.max().date())
        rows.append(s)
    return pd.DataFrame(rows)


def time_split_halves(pnl: pd.Series) -> pd.DataFrame:
    pnl = pnl.sort_index()
    mid = pnl.index[len(pnl) // 2]
    rows = []
    for label, g in [("first_half", pnl.loc[:mid]), ("second_half", pnl.loc[mid:]), ("full", pnl)]:
        rows.append({
            "split": label,
            "start": str(pd.Timestamp(g.index.min()).date()),
            "end": str(pd.Timestamp(g.index.max()).date()),
            **summarize_pnl(g),
        })
    return pd.DataFrame(rows)


def replay_with_exclude(cache: dict, exclude: set[str] | None = None) -> tuple[pd.Series, dict, dict[str, list[float]]]:
    """S1 replay; optional exclude symbols from candidacy. Also collect per-symbol pnl pieces."""
    cfg = S1
    md, abs_thr, max_each, hi = cfg["min_docs"], cfg["abs_thr"], cfg["max_each"], cfg["spread_hi"]
    enter, exit_, min_hold = cfg["spread_enter"], cfg["spread_exit"], cfg["min_hold"]
    exclude = exclude or set()

    picks = None
    days_in = 0
    pnls: list[tuple[pd.Timestamp, float]] = []
    names_list: list[int] = []
    empty_days = 0
    sym_pieces: dict[str, list[float]] = {}

    for dt in cache["dates"]:
        spread = cache["spread"][(dt, md)]
        cand = cache["picks"][(dt, md, abs_thr, max_each, hi)]
        if exclude:
            cand = [(s, w) for s, w in cand if s not in exclude]
            # if exclusion emptied a side, skip day candidacy
            if len(cand) < 2:
                cand = []

        if picks is None:
            if spread >= enter and cand:
                picks = cand
                days_in = 0
        else:
            days_in += 1
            if days_in >= min_hold and spread < exit_:
                picks = None
                days_in = 0
            elif days_in >= min_hold and spread >= enter and cand:
                if {s for s, _ in picks} != {s for s, _ in cand}:
                    picks = cand
                    days_in = 0

        if not picks:
            empty_days += 1
            continue

        ret_map = cache["ret"][dt]
        r = pnl_from_picks(ret_map, picks)
        if np.isfinite(r):
            pnls.append((dt, r))
            names_list.append(len(picks))
            for sym, w in picks:
                if sym in ret_map and np.isfinite(ret_map[sym]):
                    sym_pieces.setdefault(sym, []).append(w * ret_map[sym])

    total = len(cache["dates"])
    pnl_s = pd.Series({d: v for d, v in pnls}).sort_index()
    st = {
        "mean_names": float(np.mean(names_list)) if names_list else 0.0,
        "max_names": int(max(names_list)) if names_list else 0,
        "empty_ratio": round(empty_days / total, 3) if total else 1.0,
        "active_days": len(pnl_s),
    }
    return pnl_s, st, sym_pieces


def symbol_attribution_from_pieces(sym_pieces: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for sym, xs in sym_pieces.items():
        s = pd.Series(xs)
        mu, sd = float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else float("nan")
        sh = mu / sd * np.sqrt(252) if np.isfinite(sd) and sd > 1e-12 else float("nan")
        rows.append({
            "symbol": sym,
            "days": len(xs),
            "sum_pnl": round(float(s.sum()), 4),
            "mean_pnl": round(mu, 6),
            "sharpe_piece": round(float(sh), 3) if np.isfinite(sh) else None,
            "hit_rate": round(float((s > 0).mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("sum_pnl", ascending=False)


def main() -> None:
    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    docs = keep_single_symbol_docs(docs)
    docs = filter_docs(docs, "no_dianping")
    print(f"docs single+no_dianping: {len(docs)} paths={docs['path'].nunique()}", flush=True)

    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    OUT.mkdir(parents=True, exist_ok=True)

    print("build cache...", flush=True)
    _, cache = build_signal(panel, docs, dates)

    print("baseline S1...", flush=True)
    pnl, st, pieces = replay_with_exclude(cache)
    base = {**summarize_pnl(pnl), **st, "calmar": summarize_pnl(pnl)["calmar"]}
    # fix calmar via proper helper
    base["calmar"] = round(float(calmar_from_pnl(pnl)), 3) if len(pnl) >= 30 else None
    print(base, flush=True)

    print("quarterly...", flush=True)
    qdf = quarterly_table(pnl)
    qdf.to_csv(OUT / "quarterly.csv", index=False)
    print(qdf.to_string(index=False), flush=True)

    print("time halves...", flush=True)
    hdf = time_split_halves(pnl)
    hdf.to_csv(OUT / "time_halves.csv", index=False)
    print(hdf.to_string(index=False), flush=True)

    print("symbol attribution...", flush=True)
    adf = symbol_attribution_from_pieces(pieces)
    adf.to_csv(OUT / "symbol_attribution.csv", index=False)
    print(adf.head(15).to_string(index=False), flush=True)
    print("... bottom ...", flush=True)
    print(adf.tail(8).to_string(index=False), flush=True)

    print("leave-one-symbol-out...", flush=True)
    lo_rows = [{"drop": "(none)", **summarize_pnl(pnl), "calmar": base["calmar"], "empty_ratio": st["empty_ratio"]}]
    for sym in adf.head(10)["symbol"].tolist():
        p2, st2, _ = replay_with_exclude(cache, exclude={sym})
        lo_rows.append({
            "drop": sym,
            **summarize_pnl(p2),
            "calmar": round(float(calmar_from_pnl(p2)), 3) if len(p2) >= 30 else None,
            "empty_ratio": st2["empty_ratio"],
        })
    ldf = pd.DataFrame(lo_rows)
    ldf.to_csv(OUT / "leave_one_symbol.csv", index=False)
    print(ldf.to_string(index=False), flush=True)

    print("half-doc ablation...", flush=True)
    paths = docs["path"].drop_duplicates().tolist()
    half_rows = []
    for seed in [0, 1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(paths, size=len(paths) // 2, replace=False))
        sub = docs[docs["path"].isin(keep)].copy()
        print(f"  seed={seed} paths={len(keep)}", flush=True)
        _, c2 = build_signal(panel, sub, dates)
        p2, st2, _ = replay_with_exclude(c2)
        half_rows.append({
            "seed": seed,
            "n_paths": len(keep),
            "n_docs": len(sub),
            "empty_ratio": st2["empty_ratio"],
            **summarize_pnl(p2),
            "calmar": round(float(calmar_from_pnl(p2)), 3) if len(p2) >= 30 else None,
        })
    # hash split by path string
    for bit, label in [(0, "hash0"), (1, "hash1")]:
        sub = docs[docs["path"].map(lambda p: (hash(str(p)) & 1) == bit)].copy()
        print(f"  {label} paths={sub['path'].nunique()}", flush=True)
        _, c2 = build_signal(panel, sub, dates)
        p2, st2, _ = replay_with_exclude(c2)
        half_rows.append({
            "seed": label,
            "n_paths": int(sub["path"].nunique()),
            "n_docs": len(sub),
            "empty_ratio": st2["empty_ratio"],
            **summarize_pnl(p2),
            "calmar": round(float(calmar_from_pnl(p2)), 3) if len(p2) >= 30 else None,
        })
    half = pd.DataFrame(half_rows)
    half.to_csv(OUT / "half_docs.csv", index=False)
    print(half.to_string(index=False), flush=True)

    sh = pd.to_numeric(half["sharpe"], errors="coerce")
    report = {
        "baseline": base,
        "quarterly_positive_sharpe": int((qdf["sharpe"].fillna(-9) > 0).sum()),
        "quarterly_n": int(len(qdf)),
        "half_docs_median_sharpe": None if sh.isna().all() else float(sh.median()),
        "half_docs_min_sharpe": None if sh.isna().all() else float(sh.min()),
        "top_symbols": adf.head(10).to_dict(orient="records"),
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
