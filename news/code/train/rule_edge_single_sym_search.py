#!/usr/bin/env python3
"""Re-search rule strategies after dropping multi-symbol news.

Stage A: sticky 2L2S hold screen over signal specs (single-symbol docs only).
Stage B: dynamic enter/exit grid on top signal specs (precomputed day picks).
"""

from __future__ import annotations

import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from common import DATA_DIR, RESULTS_ROOT  # noqa: E402
from linear_ridge_walkforward import build_panel, ic_stats, metrics_from_pnl, rank_ic_daily  # noqa: E402
from rule_edge_dynamic_search import MAX_NAMES, calmar_from_pnl, day_spread, pick_variable  # noqa: E402
from rule_factor_hold4_screen import eval_row as eval_hold4  # noqa: E402
from rule_factor_mine import batch_aggregate, filter_docs, keep_single_symbol_docs  # noqa: E402
from two_name_select import apply_hold  # noqa: E402

BT_START = pd.Timestamp("2024-03-08")
OUT = RESULTS_ROOT / "rule_dynamic_single"

KINDS = ("no_dianping", "strategy", "weekly")
COLS = ("rule_edge", "rule_lex", "core_lex", "fwd_lex")
LBS = (7, 14, 30)
WTS = ("uniform",)
HOLD_DAYS = (7, 10, 14, 20)
MIN_DOCS_HOLD = (0, 1)
TOP_SIGNALS = 3
ABS_GRID = (0.0, 0.15, 0.25)
MAX_GRID = (1, 2)
HI_GRID = (0.45, 0.55, 0.65)
MD_GRID = (0, 1)


def precompute_days(panel: pd.DataFrame, fac: pd.DataFrame) -> dict:
    """One-time per signal: spreads + pick tables + pnl lookup."""
    p = panel.merge(fac, on=["date", "symbol"], how="left")
    dates = sorted(p["date"].unique())
    out: dict = {"dates": dates, "spread": {}, "picks": {}, "ret": {}, "ic": {}}

    for md in MD_GRID:
        ic_p = p.dropna(subset=["val", "fwd_ret"])
        if md > 0:
            ic_p = ic_p[ic_p["n_docs"] >= md]
        out["ic"][md] = ic_stats(rank_ic_daily(ic_p, "val")) if ic_p["date"].nunique() >= 30 else {}

    for dt in dates:
        day = p[p["date"] == dt].drop_duplicates("symbol").reset_index(drop=True)
        # symbol -> fwd_ret for apply_hold substitute
        ret_map = {
            str(r.symbol): float(r.fwd_ret)
            for r in day.itertuples()
            if pd.notna(r.fwd_ret)
        }
        out["ret"][dt] = ret_map

        for md in MD_GRID:
            d = day.dropna(subset=["val", "fwd_ret"])
            if md > 0:
                d = d[d["n_docs"] >= md]
            sp = day_spread(d["val"]) if len(d) >= 2 else 0.0
            out["spread"][(dt, md)] = sp
            for abs_thr, max_each, hi in product(ABS_GRID, MAX_GRID, HI_GRID):
                picks = pick_variable(
                    d, abs_thr=abs_thr, max_each=max_each, spread=sp, spread_hi=hi
                )
                out["picks"][(dt, md, abs_thr, max_each, hi)] = picks
    return out


def pnl_from_picks(ret_map: dict[str, float], picks: list[tuple[str, float]]) -> float:
    """Match two_name_select.apply_hold: sum(weight * fwd_ret)."""
    if not picks:
        return float("nan")
    pnl = 0.0
    n = 0
    for sym, w in picks:
        if sym in ret_map and np.isfinite(ret_map[sym]):
            pnl += w * ret_map[sym]
            n += 1
    return pnl if n else float("nan")


def run_dynamic_fast(cache: dict, cfg: dict) -> tuple[pd.Series, dict]:
    dates = cache["dates"]
    md = cfg["min_docs"]
    abs_thr = cfg["abs_thr"]
    max_each = cfg["max_each"]
    hi = cfg["spread_hi"]
    mode = cfg["mode"]
    enter = cfg["spread_enter"]
    exit_ = cfg["spread_exit"]
    min_hold = cfg["min_hold"]

    picks: list[tuple[str, float]] | None = None
    days_in = 0
    sym_streak: dict[str, int] = {}
    closed: list[int] = []
    pnls: list[tuple[pd.Timestamp, float]] = []
    names_list: list[int] = []
    empty_days = 0

    for dt in dates:
        spread = cache["spread"][(dt, md)]
        cand = cache["picks"][(dt, md, abs_thr, max_each, hi)]

        if mode == "daily":
            picks = cand if (spread >= enter and cand) else None
            days_in = 0
        else:
            if picks is None:
                if spread >= enter and cand:
                    picks = cand
                    days_in = 0
            else:
                days_in += 1
                if days_in >= min_hold and spread < exit_:
                    for sym in list(sym_streak.keys()):
                        closed.append(sym_streak.pop(sym))
                    picks = None
                    days_in = 0
                elif days_in >= min_hold and spread >= enter and cand:
                    old = {s for s, _ in picks}
                    new_s = {s for s, _ in cand}
                    if old != new_s:
                        for sym in old - new_s:
                            if sym in sym_streak:
                                closed.append(sym_streak.pop(sym))
                        picks = cand
                        days_in = 0

        if not picks:
            empty_days += 1
            continue

        r = pnl_from_picks(cache["ret"][dt], picks)
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


def eval_fast(cache: dict, cfg: dict) -> dict:
    pnl, st = run_dynamic_fast(cache, cfg)
    m = metrics_from_pnl(pnl) if len(pnl) else {"return": np.nan, "sharpe": np.nan, "max_dd": np.nan, "days": 0}
    cal = calmar_from_pnl(pnl)
    yearly_sh = []
    if len(pnl):
        for _, g in pnl.groupby(pnl.index.year):
            yearly_sh.append(metrics_from_pnl(g)["sharpe"])
    ic = cache["ic"].get(cfg["min_docs"], {})
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
    skip_a = os.environ.get("SKIP_STAGE_A") == "1"
    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    n_before = len(docs)
    docs = keep_single_symbol_docs(docs)
    n_paths = int(docs["path"].nunique())
    print(
        f"single-symbol only: rows {len(docs)}/{n_before}, paths={n_paths}, "
        f"symbols={docs['symbol'].nunique()}",
        flush=True,
    )

    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    dates_idx = pd.DatetimeIndex(sorted(panel["date"].unique()))

    fac_cache: dict[str, pd.DataFrame] = {}
    needed_kinds = set(KINDS)
    for kind in needed_kinds:
        sub = filter_docs(docs, kind)
        if sub.empty:
            continue
        for col, lb, wt in product(COLS, LBS, WTS):
            if col not in sub.columns:
                continue
            key = f"{col}|{kind}|L{lb}|{wt}|single"
            print(f"agg {key} docs={len(sub)}", flush=True)
            fac_cache[key] = batch_aggregate(sub, col, lb, dates_idx, wt)

    OUT.mkdir(parents=True, exist_ok=True)

    if skip_a and (OUT / "hold4_filtered.csv").is_file():
        hold_ok = pd.read_csv(OUT / "hold4_filtered.csv")
        print("SKIP_STAGE_A: loaded hold4_filtered.csv", flush=True)
    else:
        hold_rows: list[dict] = []
        for key, fac in fac_cache.items():
            for hold, md in product(HOLD_DAYS, MIN_DOCS_HOLD):
                tag = f"{key}|hold{hold}|md{md}"
                hold_rows.append(eval_hold4(panel, fac, tag, hold, md))
        hold_df = pd.DataFrame(hold_rows)
        hold_df.to_csv(OUT / "hold4_all.csv", index=False)
        hold_ok = hold_df[hold_df["pass_hold"] & hold_df["pass_names"]].copy()
        hold_ok = hold_ok.sort_values(["calmar", "sharpe"], ascending=False, na_position="last")
        hold_ok.to_csv(OUT / "hold4_filtered.csv", index=False)

    print("\n=== Stage A TOP hold4 (single-sym) ===", flush=True)
    cols_a = ["name", "ic", "sharpe", "calmar", "max_dd", "return", "median_hold", "trade_days"]
    print(hold_ok[cols_a].head(12).to_string(index=False), flush=True)

    def signal_key(name: str) -> str:
        return "|".join(str(name).split("|")[:5])

    top_keys: list[str] = []
    for name in hold_ok["name"]:
        k = signal_key(name)
        if k in fac_cache and k not in top_keys:
            top_keys.append(k)
        if len(top_keys) >= TOP_SIGNALS:
            break
    prior = "rule_edge|no_dianping|L7|uniform|single"
    if prior in fac_cache and prior not in top_keys:
        top_keys.append(prior)
    print(f"\nStage B signals ({len(top_keys)}): {top_keys}", flush=True)

    # sanity: compare fast pnl helper vs apply_hold once
    _sig0 = top_keys[0]
    _cache0 = precompute_days(panel, fac_cache[_sig0])
    _dt = _cache0["dates"][100]
    _picks = _cache0["picks"][(_dt, 0, 0.0, 2, 0.55)]
    if _picks:
        day = panel.merge(fac_cache[_sig0], on=["date", "symbol"], how="left")
        day = day[day["date"] == _dt].drop_duplicates("symbol")
        a = apply_hold(day, _picks)
        b = pnl_from_picks(_cache0["ret"][_dt], _picks)
        print(f"pnl sanity apply_hold={a:.6f} fast={b:.6f}", flush=True)

    dyn_rows: list[dict] = []
    for sig in top_keys:
        print(f"\nprecompute + dynamic on {sig}", flush=True)
        cache = precompute_days(panel, fac_cache[sig]) if sig != _sig0 else _cache0
        configs: list[dict] = []
        for mode in ("hysteresis", "daily"):
            for spread_enter, spread_exit, min_hold, abs_thr, max_each, spread_hi, min_docs in product(
                (0.30, 0.40, 0.50, 0.60),
                (0.10, 0.15, 0.20, 0.25),
                (5, 7, 10, 14),
                ABS_GRID,
                MAX_GRID,
                HI_GRID,
                MD_GRID,
            ):
                if spread_exit >= spread_enter:
                    continue
                if mode == "daily" and min_hold != 5:
                    continue
                name = (
                    f"{sig}|{mode}|enter{spread_enter}|exit{spread_exit}|hold{min_hold}"
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

        for i, cfg in enumerate(configs):
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(configs)}] {sig}", flush=True)
            row = eval_fast(cache, cfg)
            row["signal"] = sig
            dyn_rows.append(row)

    dyn = pd.DataFrame(dyn_rows)
    dyn.to_csv(OUT / "dynamic_all.csv", index=False)

    ok = dyn[(dyn["max_names"] <= MAX_NAMES) & (dyn["trade_days"] >= 60)].copy()
    ok = ok.sort_values(["calmar", "sharpe"], ascending=False, na_position="last")
    ok.to_csv(OUT / "dynamic_ranked.csv", index=False)

    robust = ok[(ok["n_neg_years"] == 0) & (ok["empty_ratio"] <= 0.7)].sort_values(
        ["calmar", "sharpe"], ascending=False
    )
    robust.to_csv(OUT / "dynamic_robust.csv", index=False)

    top = robust.head(20) if len(robust) else ok.head(20)
    report = {
        "filter": "single_symbol_docs only (path maps to exactly 1 symbol)",
        "n_docs_single": int(len(docs)),
        "n_paths": n_paths,
        "stage_a_top": hold_ok.head(10).to_dict(orient="records"),
        "stage_b_signals": top_keys,
        "stage_b_top20": top.to_dict(orient="records"),
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    cols_b = [
        "name", "signal", "sharpe", "calmar", "max_dd", "return",
        "median_hold", "mean_names", "empty_ratio", "trade_days", "n_neg_years",
    ]
    print("\n=== Stage B TOP robust dynamic (single-sym) ===", flush=True)
    print(top[cols_b].to_string(index=False), flush=True)
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
