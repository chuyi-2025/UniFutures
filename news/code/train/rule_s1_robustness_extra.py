#!/usr/bin/env python3
"""Extra robustness checks for S1: param neighborhood, L/hold frequency, nulls, drop rates."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from common import DATA_DIR, RESULTS_ROOT  # noqa: E402
from linear_ridge_walkforward import build_panel, metrics_from_pnl  # noqa: E402
from rule_edge_dynamic_search import calmar_from_pnl, day_spread, pick_variable  # noqa: E402
from rule_edge_single_sym_search import pnl_from_picks, precompute_days  # noqa: E402
from rule_factor_mine import batch_aggregate, filter_docs, keep_single_symbol_docs  # noqa: E402
from rule_s1_overfit_check import S1, summarize_pnl  # noqa: E402

BT_START = pd.Timestamp("2024-03-08")
OUT = RESULTS_ROOT / "rule_dynamic_single" / "overfit"


def replay(cache: dict, cfg: dict) -> tuple[pd.Series, dict]:
    md = cfg["min_docs"]
    abs_thr = cfg["abs_thr"]
    max_each = cfg["max_each"]
    hi = cfg["spread_hi"]
    enter = cfg["spread_enter"]
    exit_ = cfg["spread_exit"]
    min_hold = cfg["min_hold"]
    mode = cfg.get("mode", "hysteresis")

    picks = None
    days_in = 0
    pnls: list[tuple[pd.Timestamp, float]] = []
    names_list: list[int] = []
    empty_days = 0

    for dt in cache["dates"]:
        key_spread = (dt, md)
        key_pick = (dt, md, abs_thr, max_each, hi)
        if key_spread not in cache["spread"] or key_pick not in cache["picks"]:
            # fallback: skip if grid point missing
            empty_days += 1
            continue
        spread = cache["spread"][key_spread]
        cand = cache["picks"][key_pick]

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
                    picks = None
                    days_in = 0
                elif days_in >= min_hold and spread >= enter and cand:
                    if {s for s, _ in picks} != {s for s, _ in cand}:
                        picks = cand
                        days_in = 0

        if not picks:
            empty_days += 1
            continue
        r = pnl_from_picks(cache["ret"][dt], picks)
        if np.isfinite(r):
            pnls.append((dt, r))
            names_list.append(len(picks))

    total = len(cache["dates"])
    pnl_s = pd.Series({d: v for d, v in pnls}).sort_index()
    st = {
        "empty_ratio": round(empty_days / total, 3) if total else 1.0,
        "mean_names": float(np.mean(names_list)) if names_list else 0.0,
        "active_days": len(pnl_s),
    }
    return pnl_s, st


def row_from_pnl(pnl: pd.Series, st: dict, **extra) -> dict:
    s = summarize_pnl(pnl)
    cal = round(float(calmar_from_pnl(pnl)), 3) if len(pnl) >= 30 else None
    mn = st.get("mean_names")
    return {
        **extra,
        **s,
        "calmar": cal,
        "empty_ratio": st.get("empty_ratio"),
        "mean_names": round(mn, 3) if isinstance(mn, (int, float)) and mn is not None else mn,
    }


def build_cache(panel, docs, dates, col="rule_edge", lb=30, wt="uniform"):
    sub = filter_docs(docs, "no_dianping")
    fac = batch_aggregate(sub, col, lb, dates, wt)
    return precompute_days(panel, fac)


def precompute_light(panel: pd.DataFrame, fac: pd.DataFrame, cfg: dict) -> dict:
    """Only S1-relevant picks (faster for shuffled factors)."""
    p = panel.merge(fac, on=["date", "symbol"], how="left")
    dates = sorted(p["date"].unique())
    md, abs_thr, max_each, hi = cfg["min_docs"], cfg["abs_thr"], cfg["max_each"], cfg["spread_hi"]
    out = {"dates": dates, "spread": {}, "picks": {}, "ret": {}}
    for dt in dates:
        day = p[p["date"] == dt].drop_duplicates("symbol").reset_index(drop=True)
        out["ret"][dt] = {
            str(r.symbol): float(r.fwd_ret)
            for r in day.itertuples()
            if pd.notna(r.fwd_ret)
        }
        d = day.dropna(subset=["val", "fwd_ret"])
        if md > 0:
            d = d[d["n_docs"] >= md]
        sp = day_spread(d["val"]) if len(d) >= 2 else 0.0
        out["spread"][(dt, md)] = sp
        out["picks"][(dt, md, abs_thr, max_each, hi)] = pick_variable(
            d, abs_thr=abs_thr, max_each=max_each, spread=sp, spread_hi=hi
        )
    return out


def shuffle_fac_cross_section(fac: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle val across symbols within each date (kills ranking signal)."""
    rng = np.random.default_rng(seed)
    f = fac.copy()
    rows = []
    for _, g in f.groupby("date"):
        g = g.copy()
        g["val"] = rng.permutation(g["val"].to_numpy())
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def shuffle_fac_global(fac: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = fac.copy()
    f["val"] = rng.permutation(f["val"].to_numpy())
    return f


def main() -> None:
    docs = pd.read_parquet(DATA_DIR / "rule_docs_rich.parquet")
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    docs = keep_single_symbol_docs(filter_docs(docs, "no_dianping"))
    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    OUT.mkdir(parents=True, exist_ok=True)

    print("base cache L30...", flush=True)
    cache = build_cache(panel, docs, dates, "rule_edge", 30, "uniform")
    base_pnl, base_st = replay(cache, S1)
    print("baseline", row_from_pnl(base_pnl, base_st, tag="S1"), flush=True)

    # ---- 1) trading-param neighborhood (same cache) ----
    print("param neighborhood...", flush=True)
    neigh_rows = []
    for enter in (0.30, 0.35, 0.40, 0.45, 0.50):
        for exit_ in (0.05, 0.10, 0.15, 0.20):
            if exit_ >= enter:
                continue
            for hold in (3, 5, 7, 10, 14):
                for hi in (0.45, 0.55, 0.65):
                    cfg = {**S1, "spread_enter": enter, "spread_exit": exit_, "min_hold": hold, "spread_hi": hi}
                    # need picks for these abs/hi — cache has ABS/MAX/HI grids from precompute_days
                    pnl, st = replay(cache, cfg)
                    neigh_rows.append(row_from_pnl(
                        pnl, st,
                        enter=enter, exit=exit_, min_hold=hold, spread_hi=hi,
                        is_s1=int(enter == 0.4 and exit_ == 0.1 and hold == 5 and hi == 0.55),
                    ))
    neigh = pd.DataFrame(neigh_rows).sort_values("sharpe", ascending=False)
    neigh.to_csv(OUT / "param_neighborhood.csv", index=False)
    print(neigh.head(10).to_string(index=False), flush=True)
    print("neigh sharpe: median", neigh["sharpe"].median(), "p10", neigh["sharpe"].quantile(0.1),
          "frac>1", (neigh["sharpe"] > 1).mean(), flush=True)

    # ---- 2) hold / enter frequency slices (1D sweeps) ----
    print("1D sweeps...", flush=True)
    sweep_rows = []
    for hold in (1, 3, 5, 7, 10, 14, 20):
        cfg = {**S1, "min_hold": hold}
        pnl, st = replay(cache, cfg)
        sweep_rows.append(row_from_pnl(pnl, st, axis="min_hold", value=hold))
    for enter in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60):
        cfg = {**S1, "spread_enter": enter, "spread_exit": min(0.10, enter - 0.05)}
        if cfg["spread_exit"] >= enter:
            cfg["spread_exit"] = enter * 0.5
        pnl, st = replay(cache, cfg)
        sweep_rows.append(row_from_pnl(pnl, st, axis="spread_enter", value=enter))
    for hi in (0.35, 0.45, 0.55, 0.65, 0.75):
        cfg = {**S1, "spread_hi": hi}
        # hi=0.35/0.75 may miss cache — skip if empty
        pnl, st = replay(cache, cfg)
        if st["active_days"] == 0 and hi not in (0.45, 0.55, 0.65):
            continue
        sweep_rows.append(row_from_pnl(pnl, st, axis="spread_hi", value=hi))
    sweeps = pd.DataFrame(sweep_rows)
    sweeps.to_csv(OUT / "param_1d_sweeps.csv", index=False)
    print(sweeps.to_string(index=False), flush=True)

    # ---- 3) lookback L + weight + signal col ----
    print("lookback / weight / col...", flush=True)
    sig_rows = []
    for lb in (7, 14, 21, 30, 45, 60):
        for wt in ("uniform", "exp"):
            print(f"  L{lb} {wt}", flush=True)
            c = build_cache(panel, docs, dates, "rule_edge", lb, wt)
            pnl, st = replay(c, S1)
            sig_rows.append(row_from_pnl(pnl, st, col="rule_edge", lookback=lb, weight=wt))
    for col in ("rule_lex", "core_lex", "fwd_lex", "rule_title"):
        # rule_title may not exist in rich docs as aggregate col — check
        print(f"  col {col} L30", flush=True)
        try:
            c = build_cache(panel, docs, dates, col, 30, "uniform")
        except Exception as e:
            print("  skip", col, e, flush=True)
            continue
        pnl, st = replay(c, S1)
        sig_rows.append(row_from_pnl(pnl, st, col=col, lookback=30, weight="uniform"))
    sigdf = pd.DataFrame(sig_rows).sort_values("sharpe", ascending=False)
    sigdf.to_csv(OUT / "signal_lookback.csv", index=False)
    print(sigdf.to_string(index=False), flush=True)

    # ---- 4) null: shuffle cross-section / global ----
    print("null shuffles...", flush=True)
    sub = filter_docs(docs, "no_dianping")
    fac0 = batch_aggregate(sub, "rule_edge", 30, dates, "uniform")
    null_rows = []
    for kind in ("cs", "global"):
        for seed in range(20):
            f2 = shuffle_fac_cross_section(fac0, seed) if kind == "cs" else shuffle_fac_global(fac0, seed)
            c2 = precompute_light(panel, f2, S1)
            pnl, st = replay(c2, S1)
            null_rows.append(row_from_pnl(pnl, st, null=kind, seed=seed))
            if (seed + 1) % 5 == 0:
                print(f"  {kind} seed {seed+1}/20", flush=True)
    nulldf = pd.DataFrame(null_rows)
    nulldf.to_csv(OUT / "null_shuffle.csv", index=False)
    for kind, g in nulldf.groupby("null"):
        sh = g["sharpe"].dropna()
        print(f"null {kind}: mean={sh.mean():.3f} median={sh.median():.3f} "
              f"p95={sh.quantile(0.95):.3f} frac>S1={(sh > 2.856).mean():.3f} frac>1={(sh > 1).mean():.3f}",
              flush=True)

    # ---- 5) drop-rate curve ----
    print("drop-rate curve...", flush=True)
    paths = docs["path"].drop_duplicates().to_numpy()
    drop_rows = []
    for keep_frac in (0.25, 0.50, 0.75, 1.0):
        for seed in range(5 if keep_frac < 1 else 1):
            rng = np.random.default_rng(seed)
            n = max(1, int(len(paths) * keep_frac))
            keep = set(rng.choice(paths, size=n, replace=False)) if keep_frac < 1 else set(paths)
            sub_docs = docs[docs["path"].isin(keep)]
            print(f"  keep={keep_frac} seed={seed} n={len(keep)}", flush=True)
            c = build_cache(panel, sub_docs, dates, "rule_edge", 30, "uniform")
            pnl, st = replay(c, S1)
            drop_rows.append(row_from_pnl(pnl, st, keep_frac=keep_frac, seed=seed, n_paths=len(keep)))
    dropdf = pd.DataFrame(drop_rows)
    dropdf.to_csv(OUT / "drop_rate.csv", index=False)
    print(dropdf.groupby("keep_frac")["sharpe"].agg(["mean", "median", "min", "max"]).to_string(), flush=True)

    # ---- 6) calendar month odd/even + skip-every-other-active-day ----
    print("frequency / calendar splits...", flush=True)
    freq_rows = []
    # odd/even months on realized pnl
    for label, mask in [
        ("odd_months", base_pnl.index.month % 2 == 1),
        ("even_months", base_pnl.index.month % 2 == 0),
        ("active_days_odd", np.arange(len(base_pnl)) % 2 == 0),
        ("active_days_even", np.arange(len(base_pnl)) % 2 == 1),
    ]:
        g = base_pnl.loc[mask] if isinstance(mask, pd.Series) else base_pnl.iloc[mask]
        freq_rows.append(row_from_pnl(g, {"empty_ratio": None, "mean_names": None}, split=label))
    # trade only every k-th calendar day while in position — approximate by thinning pnl days
    for k in (2, 3, 5):
        g = base_pnl.iloc[::k]
        freq_rows.append(row_from_pnl(g, {"empty_ratio": None, "mean_names": None}, split=f"thin_every_{k}"))
    freqdf = pd.DataFrame(freq_rows)
    freqdf.to_csv(OUT / "frequency_splits.csv", index=False)
    print(freqdf.to_string(index=False), flush=True)

    # ---- 7) expanding year OOS feel: metrics by year already; add rolling 6m ----
    print("rolling 6m...", flush=True)
    roll_rows = []
    idx = base_pnl.sort_index().index
    for i in range(len(idx)):
        end = idx[i]
        start = end - pd.Timedelta(days=183)
        g = base_pnl.loc[(base_pnl.index > start) & (base_pnl.index <= end)]
        if len(g) < 40:
            continue
        # only record month-ends approx
        if i < len(idx) - 1 and idx[i + 1].month == end.month:
            continue
        s = summarize_pnl(g)
        if s["sharpe"] is None:
            continue
        roll_rows.append({"end": str(end.date()), **s})
    rolldf = pd.DataFrame(roll_rows)
    rolldf.to_csv(OUT / "rolling_6m.csv", index=False)
    if len(rolldf):
        print("rolling6m sharpe: median", rolldf["sharpe"].median(),
              "min", rolldf["sharpe"].min(), "frac>0", (rolldf["sharpe"] > 0).mean(), flush=True)

    report = {
        "baseline_sharpe": 2.856,
        "neigh_n": int(len(neigh)),
        "neigh_sharpe_median": float(neigh["sharpe"].median()),
        "neigh_sharpe_p10": float(neigh["sharpe"].quantile(0.1)),
        "neigh_frac_sharpe_gt_1": float((neigh["sharpe"] > 1).mean()),
        "null_cs_mean": float(nulldf.loc[nulldf["null"] == "cs", "sharpe"].mean()),
        "null_cs_p95": float(nulldf.loc[nulldf["null"] == "cs", "sharpe"].quantile(0.95)),
        "null_global_mean": float(nulldf.loc[nulldf["null"] == "global", "sharpe"].mean()),
        "null_global_p95": float(nulldf.loc[nulldf["null"] == "global", "sharpe"].quantile(0.95)),
        "drop_median_by_frac": dropdf.groupby("keep_frac")["sharpe"].median().to_dict(),
    }
    (OUT / "robustness_extra_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nREPORT", json.dumps(report, indent=2), flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
