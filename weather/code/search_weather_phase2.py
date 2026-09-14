#!/usr/bin/env python3
"""Phase-2 weather search: per-symbol, multi-horizon IC, time-split OOS.

Focus on schemes where Sharpe and IC/IR agree, with half-sample confirmation.
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULT_DIR, SYMBOLS, ensure_dirs  # noqa: E402
from search_weather_strategies import (  # noqa: E402
    FACTOR_SPECS,
    factor_series,
    hold_signal,
    make_position,
    prepare_features,
    sharpe,
    max_dd,
    spearman_ic,
)

OUT = RESULT_DIR / "search"


def _spearman(x, y) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = spearman_ic(x, y)
    return v if v == v else float("nan")


def add_horizons(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol", group_keys=False)
    # fwd_ret is already 1-day log return; build 5/10d forward sums of future 1d rets
    out["fwd_ret_5"] = g["fwd_ret"].transform(lambda s: s.shift(-1).rolling(5).sum().shift(-4))
    # simpler: sum of next h days: r_t + r_{t+1}+... = use reverse cum then
    def fwd_sum(s: pd.Series, h: int) -> pd.Series:
        # at t, sum of fwd_ret[t], fwd_ret[t+1], ... fwd_ret[t+h-1] — but fwd_ret[t] is already next-day
        # So h-day return starting tomorrow ≈ sum of h consecutive fwd_ret from t
        x = s.fillna(0.0)
        c = x[::-1].cumsum()[::-1]
        fut = c - c.shift(-h).fillna(0.0)
        fut.iloc[-h:] = np.nan
        return fut

    out["fwd_ret_5"] = g["fwd_ret"].transform(lambda s: fwd_sum(s, 5))
    out["fwd_ret_10"] = g["fwd_ret"].transform(lambda s: fwd_sum(s, 10))
    return out


def ic_metrics(fac: pd.Series, ret: pd.Series, dates: pd.Series) -> dict:
    sub = pd.DataFrame({"f": fac, "r": ret, "date": dates}).dropna()
    if len(sub) < 60:
        return {"ic": None, "ir": None, "ic_pos": None, "n_m": 0}
    ic = spearman_ic(sub["f"], sub["r"], min_n=30)
    months = []
    for _, g in sub.groupby(pd.to_datetime(sub["date"]).dt.to_period("M")):
        if len(g) < 8:
            continue
        months.append(spearman_ic(g["f"], g["r"], min_n=8))
    months = [m for m in months if m == m]
    if len(months) < 8:
        return {"ic": round(ic, 4) if ic == ic else None, "ir": None, "ic_pos": None, "n_m": len(months)}
    arr = np.asarray(months, float)
    ir = float(arr.mean() / arr.std()) if arr.std() > 1e-12 else None
    return {
        "ic": round(float(ic), 4) if ic == ic else None,
        "ir": round(ir, 3) if ir is not None else None,
        "ic_pos": round(float((arr > 0).mean()), 3),
        "n_m": len(months),
    }


def time_split_sharpe(pnl: pd.Series) -> dict:
    pnl = pnl.sort_index()
    if len(pnl) < 80:
        return {"sh_full": sharpe(pnl), "sh_first": None, "sh_second": None, "both_pos": False}
    mid = pnl.index[len(pnl) // 2]
    a = pnl[pnl.index <= mid]
    b = pnl[pnl.index > mid]
    sa, sb = sharpe(a), sharpe(b)
    return {
        "sh_full": sharpe(pnl),
        "sh_first": sa,
        "sh_second": sb,
        "both_pos": bool(sa is not None and sb is not None and sa > 0 and sb > 0),
        "mid": str(pd.Timestamp(mid).date()),
    }


def run_leg(df: pd.DataFrame, factor: str, mode: str, thr: float, direction: float, hold: int, pheno: bool) -> pd.DataFrame:
    g = df.sort_values("date").copy()
    f = factor_series(g, factor)
    ph = g["in_pheno"] if pheno else None
    pos = make_position(f, mode, thr, direction, hold, ph)
    g["signal"] = pos
    g["pnl"] = pos * g["fwd_ret"].fillna(0.0).to_numpy()
    g["_fac"] = f * direction
    return g


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(RESULT_DIR / "aligned_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= "2018-01-01"].copy()
    panel = prepare_features(panel)
    panel = add_horizons(panel)

    factors = [
        "rh_z", "tmean_z", "precip_z", "neg_precip_z", "tmax_z", "tmin_z",
        "tmean_z_ma5", "tmean_z_ma10", "precip_z_ma5", "neg_precip_z_ma5",
        "dry_hot", "wet_cool", "heat_minus_precip", "frost_score", "gdd_z",
    ]
    modes = ["cont", "sign", "long_only"]
    thrs = [0.5, 1.0, 1.5]
    holds = [1, 3, 5, 10]
    dirs = [1.0, -1.0]
    phenos = [True, False]
    horizons = ["fwd_ret", "fwd_ret_5", "fwd_ret_10"]

    rows = []
    # --- A) factor IC board multi-horizon per symbol ---
    ic_rows = []
    for sym, g in panel.groupby("symbol"):
        for fname in factors:
            fac = factor_series(g, fname)
            for direction in dirs:
                for pheno in phenos:
                    gg = g[g["in_pheno"] > 0] if pheno else g
                    for hz in horizons:
                        m = ic_metrics(fac.reindex(gg.index) * direction, gg[hz], gg["date"])
                        ic_rows.append({
                            "symbol": sym, "factor": fname, "direction": direction,
                            "pheno_only": pheno, "horizon": hz, **m,
                        })
    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(OUT / "ic_ir_horizons.csv", index=False)

    # --- B) per-symbol trading search ---
    print("per-symbol trading grid...", flush=True)
    for sym in SYMBOLS:
        g0 = panel[panel["symbol"] == sym]
        for fname, mode, thr, hold, direction, pheno in product(factors, modes, thrs, holds, dirs, phenos):
            if mode == "cont" and (thr != 0.5 or hold != 1):
                # only one cont setting per direction/pheno
                if not (thr == 0.5 and hold == 1):
                    continue
                thr_use, hold_use = 0.0, 1
                mode_use = "cont"
            else:
                thr_use, hold_use, mode_use = thr, hold, mode
            bt = run_leg(g0, fname, mode_use, thr_use, direction, hold_use, pheno)
            live = bt.dropna(subset=["fwd_ret"])
            act = int((live["signal"].abs() > 1e-8).sum())
            if act < 30:
                continue
            pnl = live.set_index("date")["pnl"]
            split = time_split_sharpe(pnl)
            sh = split["sh_full"]
            if sh is None:
                continue
            ic1 = ic_metrics(live["_fac"], live["fwd_ret"], live["date"])
            ic5 = ic_metrics(live["_fac"], live["fwd_ret_5"], live["date"])
            row = {
                "symbol": sym,
                "factor": fname,
                "mode": mode_use,
                "thr": thr_use,
                "hold": hold_use,
                "direction": direction,
                "pheno_only": pheno,
                "sharpe": round(sh, 3),
                "sh_first": None if split["sh_first"] is None else round(split["sh_first"], 3),
                "sh_second": None if split["sh_second"] is None else round(split["sh_second"], 3),
                "oos_both_pos": split["both_pos"],
                "ret": round(float(pnl.sum()), 4),
                "max_dd": None if max_dd(pnl) is None else round(max_dd(pnl), 4),
                "active": act,
                "ic1": ic1["ic"],
                "ir1": ic1["ir"],
                "ic5": ic5["ic"],
                "ir5": ic5["ir"],
            }
            # quality score: sharpe + oos bonus + ic/ir
            q = (row["sharpe"] or 0)
            if row["oos_both_pos"]:
                q += 0.35
            q += 0.4 * (row["ir5"] or 0) + 0.3 * (row["ir1"] or 0)
            q += 1.5 * abs(row["ic5"] or 0) + 1.0 * abs(row["ic1"] or 0)
            # penalize if halves disagree in sign hard
            if row["sh_first"] is not None and row["sh_second"] is not None:
                if row["sh_first"] * row["sh_second"] < 0:
                    q -= 0.4
            row["quality"] = round(q, 4)
            rows.append(row)

    res = pd.DataFrame(rows)
    res = res.sort_values(["quality", "sharpe"], ascending=False)
    res.to_csv(OUT / "per_symbol_schemes.csv", index=False)

    # quality tiers
    robust = res[
        (res["sharpe"] >= 0.6)
        & (res["oos_both_pos"])
        & (res["ir5"].fillna(-9) >= 0.15)
        & (res["ic5"].abs().fillna(0) >= 0.02)
    ].copy()
    soft = res[
        (res["sharpe"] >= 0.5)
        & (res["oos_both_pos"])
        & ((res["ir1"].fillna(-9) >= 0.1) | (res["ir5"].fillna(-9) >= 0.1))
    ].copy()
    robust.to_csv(OUT / "robust_schemes.csv", index=False)
    soft.to_csv(OUT / "soft_robust_schemes.csv", index=False)

    # EW book from best robust-or-soft per symbol
    picks = []
    src = robust if len(robust) else soft
    for sym in SYMBOLS:
        sub = src[src["symbol"] == sym]
        if sub.empty:
            sub = res[res["symbol"] == sym].head(1)
        if sub.empty:
            continue
        picks.append(sub.iloc[0].to_dict())

    book_parts = []
    for p in picks:
        g = panel[panel["symbol"] == p["symbol"]]
        bt = run_leg(g, p["factor"], p["mode"], float(p["thr"]), float(p["direction"]), int(p["hold"]), bool(p["pheno_only"]))
        book_parts.append(bt[["date", "symbol", "signal", "pnl", "fwd_ret"]])
    if book_parts:
        legs = pd.concat(book_parts, ignore_index=True)
        book = legs.groupby("date")["pnl"].mean().sort_index()
        split = time_split_sharpe(book)
        legs.assign(日期=lambda d: d["date"].dt.strftime("%Y-%m-%d")).drop(columns=["date"]).to_csv(
            OUT / "picked_legs_daily.csv", index=False
        )
        pd.DataFrame({"日期": book.index.strftime("%Y-%m-%d"), "pnl": book.values}).to_csv(
            OUT / "picked_book_daily.csv", index=False
        )
    else:
        split = {}
        book = pd.Series(dtype=float)

    # top IC factors overall
    ic_rank = (
        ic_df.dropna(subset=["ir"])
        .sort_values(["ir", "ic"], ascending=False)
        .groupby(["symbol", "horizon"], as_index=False)
        .head(3)
    )
    ic_rank.to_csv(OUT / "ic_ir_top_by_symbol_horizon.csv", index=False)

    report = {
        "n_per_symbol_schemes": int(len(res)),
        "n_robust": int(len(robust)),
        "n_soft_robust": int(len(soft)),
        "robust_criteria": "sharpe>=0.6 & both halves>0 & ir5>=0.15 & |ic5|>=0.02",
        "soft_criteria": "sharpe>=0.5 & both halves>0 & (ir1>=0.1|ir5>=0.1)",
        "picks": picks,
        "picked_book": {
            "sharpe": split.get("sh_full"),
            "sh_first": split.get("sh_first"),
            "sh_second": split.get("sh_second"),
            "oos_both_pos": split.get("both_pos"),
            "ret": None if book.empty else round(float(book.sum()), 4),
        },
        "top_per_symbol": {
            sym: res[res["symbol"] == sym].head(5)[
                ["factor", "mode", "thr", "hold", "direction", "pheno_only", "sharpe", "sh_first", "sh_second",
                 "oos_both_pos", "ic1", "ir1", "ic5", "ir5", "quality"]
            ].to_dict(orient="records")
            for sym in SYMBOLS
        },
        "best_ic_samples": ic_df.dropna(subset=["ir"]).sort_values("ir", ascending=False).head(15).to_dict(orient="records"),
    }
    (OUT / "phase2_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("robust", len(robust), "soft", len(soft), flush=True)
    print("picks", picks, flush=True)
    print("book", report["picked_book"], flush=True)
    for sym in SYMBOLS:
        print(f"\n=== {sym} top3 ===", flush=True)
        print(res[res.symbol == sym].head(3)[
            ["factor", "mode", "thr", "hold", "direction", "pheno_only", "sharpe", "sh_first", "sh_second", "ic5", "ir5", "quality"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
