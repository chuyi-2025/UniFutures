#!/usr/bin/env python3
"""Search multiple weather signal schemes; rank by Sharpe / IC / IR.

Uses aligned panel (weather T-1 already applied in results/aligned_panel.parquet
or rebuilds via align_prices). Research-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_prices import align_all  # noqa: E402
from common import RESULT_DIR, SYMBOLS, ensure_dirs  # noqa: E402

OUT = RESULT_DIR / "search"
MIN_ACTIVE = 40  # min days with non-zero position for a scheme to count


def spearman_ic(x: pd.Series, y: pd.Series, min_n: int = 30) -> float:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    m = a.notna() & b.notna()
    if int(m.sum()) < min_n:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = a[m].corr(b[m], method="spearman")
    return float(v) if v == v else float("nan")


def sharpe(r: pd.Series) -> float | None:
    r = r.fillna(0.0)
    if len(r) < 30 or float(r.std()) == 0:
        return None
    return float(r.mean() / r.std() * np.sqrt(252))


def max_dd(r: pd.Series) -> float | None:
    r = r.fillna(0.0)
    if len(r) < 5:
        return None
    nav = (1.0 + r).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def hold_signal(raw: np.ndarray, hold: int) -> np.ndarray:
    """Propagate non-zero sign for `hold` trading days after each event."""
    sig = np.zeros(len(raw), dtype=float)
    left = 0
    last = 0.0
    for i, v in enumerate(raw):
        if v != 0:
            left = hold
            last = float(np.sign(v))
        if left > 0:
            sig[i] = last
            left -= 1
    return sig


def rolling_z(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=max(3, win // 2)).mean()
    sd = s.rolling(win, min_periods=max(3, win // 2)).std()
    return (s - m) / sd.replace(0, np.nan)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol", group_keys=False)
    for col in ("tmean_z", "precip_z", "tmax_z", "tmin_z", "rh_z", "gdd_z", "drought_spell"):
        if col not in out.columns:
            continue
        out[f"{col}_ma5"] = g[col].transform(lambda s: s.rolling(5, min_periods=3).mean())
        out[f"{col}_ma10"] = g[col].transform(lambda s: s.rolling(10, min_periods=5).mean())
    # composites
    out["heat_minus_precip"] = out["tmean_z"].fillna(0) - out["precip_z"].fillna(0)
    out["dry_hot"] = out["tmean_z"].fillna(0) + (-out["precip_z"].fillna(0))
    out["frost_score"] = out["frost_event"].fillna(0) * 2.0 + np.clip(-out["tmin_z"].fillna(0), 0, None)
    out["wet_cool"] = (-out["tmean_z"].fillna(0)) + out["precip_z"].fillna(0)
    # cross-sectional rank of dry_hot within day (among ag names)
    out["dry_hot_cs"] = out.groupby("date")["dry_hot"].rank(pct=True) - 0.5
    out["precip_z_cs"] = out.groupby("date")["precip_z"].rank(pct=True) - 0.5
    return out


FACTOR_SPECS = {
    # continuous factors (higher → lean long if direction=+1)
    "tmean_z": "tmean_z",
    "precip_z": "precip_z",
    "neg_precip_z": None,  # special
    "tmax_z": "tmax_z",
    "tmin_z": "tmin_z",
    "rh_z": "rh_z",
    "gdd_z": "gdd_z",
    "drought_spell": "drought_spell",
    "tmean_z_ma5": "tmean_z_ma5",
    "precip_z_ma5": "precip_z_ma5",
    "neg_precip_z_ma5": None,
    "tmean_z_ma10": "tmean_z_ma10",
    "precip_z_ma10": "precip_z_ma10",
    "heat_minus_precip": "heat_minus_precip",
    "dry_hot": "dry_hot",
    "wet_cool": "wet_cool",
    "frost_score": "frost_score",
    "dry_hot_cs": "dry_hot_cs",
    "precip_z_cs": "precip_z_cs",
}


def factor_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "neg_precip_z":
        return -df["precip_z"]
    if name == "neg_precip_z_ma5":
        return -df["precip_z_ma5"]
    col = FACTOR_SPECS[name]
    return df[col]


def eval_ic_ir(df: pd.DataFrame, factor: pd.Series, pheno_only: bool) -> dict:
    sub = df.copy()
    sub["_f"] = factor
    if pheno_only and "in_pheno" in sub.columns:
        sub = sub[sub["in_pheno"].fillna(0) > 0]
    # daily pooled IC
    ic_all = spearman_ic(sub["_f"], sub["fwd_ret"])
    # monthly IC series → IR
    sub = sub.dropna(subset=["_f", "fwd_ret"])
    if sub.empty:
        return {"ic": None, "ir": None, "ic_pos_frac": None, "n_ic_months": 0}
    months = []
    for _, g in sub.groupby(sub["date"].dt.to_period("M")):
        if len(g) < 10:
            continue
        months.append(spearman_ic(g["_f"], g["fwd_ret"]))
    months = [x for x in months if np.isfinite(x)]
    if len(months) < 6:
        ir = None
        pos = None
    else:
        arr = np.asarray(months, dtype=float)
        ir = float(arr.mean() / arr.std()) if arr.std() > 1e-12 else None
        pos = float((arr > 0).mean())
    return {
        "ic": None if not np.isfinite(ic_all) else round(float(ic_all), 4),
        "ir": None if ir is None else round(ir, 3),
        "ic_pos_frac": None if pos is None else round(pos, 3),
        "n_ic_months": len(months),
    }


def make_position(factor: pd.Series, mode: str, thr: float, direction: float, hold: int, in_pheno: pd.Series | None) -> np.ndarray:
    f = factor.fillna(0.0).to_numpy(dtype=float)
    if in_pheno is not None:
        mask = in_pheno.fillna(0).to_numpy() > 0
    else:
        mask = np.ones(len(f), dtype=bool)

    if mode == "cont":
        # continuous: clip z to [-2,2] / 2 → [-1,1]
        raw = np.clip(f, -2.0, 2.0) / 2.0 * direction
        raw = np.where(mask, raw, 0.0)
        return raw  # no hold for continuous
    if mode == "sign":
        raw = np.sign(f) * direction
        raw = np.where(mask & (np.abs(f) >= thr), raw, 0.0)
        return hold_signal(raw, hold) if hold > 1 else raw
    if mode == "long_only":
        raw = np.where(mask & (f * direction >= thr), direction, 0.0)
        return hold_signal(raw, hold) if hold > 1 else raw
    if mode == "short_only":
        raw = np.where(mask & (f * direction <= -thr), -direction, 0.0)
        # direction already encodes preferred side; use sign of stress
        raw = np.where(mask & ((f * direction) <= -thr), -1.0 * np.sign(direction or 1), 0.0)
        return hold_signal(raw, hold) if hold > 1 else raw
    raise ValueError(mode)


def eval_scheme(
    df: pd.DataFrame,
    factor_name: str,
    mode: str,
    thr: float,
    direction: float,
    hold: int,
    pheno_only: bool,
    scope: str,
) -> dict | None:
    work = df.copy()
    fac = factor_series(work, factor_name)
    ic = eval_ic_ir(work, fac * direction, pheno_only=pheno_only)

    if scope == "by_symbol":
        pnls = []
        act = 0
        for sym, g in work.groupby("symbol"):
            g = g.sort_values("date")
            f = factor_series(g, factor_name)
            ph = g["in_pheno"] if pheno_only else None
            pos = make_position(f, mode, thr, direction, hold, ph)
            pnl = pos * g["fwd_ret"].fillna(0.0).to_numpy()
            s = pd.Series(pnl, index=g["date"])
            pnls.append(s)
            act += int((np.abs(pos) > 1e-8).sum())
        # equal-weight calendar book
        book = pd.concat(pnls, axis=1).mean(axis=1).sort_index()
    else:
        # pooled single series (all symbols stacked) — use CS book equal weight by date
        rows = []
        for sym, g in work.groupby("symbol"):
            g = g.sort_values("date")
            f = factor_series(g, factor_name)
            ph = g["in_pheno"] if pheno_only else None
            pos = make_position(f, mode, thr, direction, hold, ph)
            tmp = g[["date"]].copy()
            tmp["pnl"] = pos * g["fwd_ret"].fillna(0.0).to_numpy()
            tmp["active"] = np.abs(pos) > 1e-8
            rows.append(tmp)
        allr = pd.concat(rows, ignore_index=True)
        book = allr.groupby("date")["pnl"].mean().sort_index()
        act = int(allr["active"].sum())

    if act < MIN_ACTIVE:
        return None
    sh = sharpe(book)
    if sh is None:
        return None
    return {
        "factor": factor_name,
        "mode": mode,
        "thr": thr,
        "direction": direction,
        "hold": hold,
        "pheno_only": pheno_only,
        "scope": scope,
        "sharpe": round(sh, 3),
        "ret": round(float(book.sum()), 4),
        "max_dd": None if max_dd(book) is None else round(max_dd(book), 4),
        "worst": round(float(book.min()), 6),
        "active_legs": act,
        "ic": ic["ic"],
        "ir": ic["ir"],
        "ic_pos_frac": ic["ic_pos_frac"],
        "n_ic_months": ic["n_ic_months"],
        "score": None,  # filled later
    }


def score_row(r: dict) -> float:
    """Composite: prefer Sharpe + IR + |IC| with mild activity."""
    sh = r.get("sharpe") or 0.0
    ir = r.get("ir") or 0.0
    ic = abs(r.get("ic") or 0.0)
    return float(sh) + 0.5 * float(ir) + 2.0 * float(ic)


def per_symbol_best(df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Also report best continuous factor IC/IR per symbol (no trading)."""
    rows = []
    for sym, g in df.groupby("symbol"):
        for fname in FACTOR_SPECS:
            fac = factor_series(g, fname)
            for pheno in (False, True):
                for direction in (1.0, -1.0):
                    ic = eval_ic_ir(g, fac * direction, pheno_only=pheno)
                    if ic["ic"] is None:
                        continue
                    rows.append({
                        "symbol": sym,
                        "factor": fname,
                        "direction": direction,
                        "pheno_only": pheno,
                        **ic,
                    })
    rows.sort(key=lambda x: (x.get("ir") or -9, abs(x.get("ic") or 0)), reverse=True)
    return rows[: top_n * 4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)

    aligned = RESULT_DIR / "aligned_panel.parquet"
    if args.rebuild or not aligned.exists():
        print("rebuilding aligned panel...", flush=True)
        panel = align_all(symbols=SYMBOLS, weather_lag_days=1)
        panel = panel[panel["date"] >= pd.Timestamp(args.start)].copy()
        panel.to_parquet(aligned, index=False)
    else:
        panel = pd.read_parquet(aligned)
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel[panel["date"] >= pd.Timestamp(args.start)].copy()

    panel = prepare_features(panel)
    print(f"panel rows={len(panel)} {panel['date'].min().date()}->{panel['date'].max().date()}", flush=True)

    modes = ["cont", "sign", "long_only"]
    thrs = [0.0, 0.5, 1.0, 1.5]
    holds = [1, 3, 5, 10]
    directions = [1.0, -1.0]
    phenos = [True, False]

    results = []
    n_try = 0
    for fname, mode, thr, hold, direction, pheno in product(
        FACTOR_SPECS.keys(), modes, thrs, holds, directions, phenos
    ):
        # prune silly combos
        if mode == "cont" and (thr != 0.0 or hold != 1):
            continue
        if mode != "cont" and thr == 0.0 and mode == "sign":
            # sign with thr0 ok
            pass
        n_try += 1
        row = eval_scheme(panel, fname, mode, thr, direction, hold, pheno, scope="ew_book")
        if row is None:
            continue
        row["score"] = round(score_row(row), 4)
        results.append(row)
        if len(results) % 50 == 0:
            print(f"  kept {len(results)} / tried ~{n_try}", flush=True)

    res = pd.DataFrame(results)
    if res.empty:
        print("no schemes passed filters", flush=True)
        return
    res = res.sort_values(["score", "sharpe", "ir"], ascending=False)
    res.to_csv(OUT / "all_schemes.csv", index=False)

    # filters for "good" schemes
    good = res[
        (res["sharpe"] >= 0.5)
        & (res["ir"].fillna(-9) >= 0.2)
        & (res["ic"].abs().fillna(0) >= 0.02)
    ].copy()
    good.to_csv(OUT / "good_schemes.csv", index=False)

    top = res.head(30)
    top.to_csv(OUT / "top30.csv", index=False)

    # IC leaderboard (factor diagnostics)
    ic_board = per_symbol_best(panel, top_n=8)
    pd.DataFrame(ic_board).to_csv(OUT / "ic_ir_by_symbol.csv", index=False)

    # re-export best scheme daily
    best = res.iloc[0].to_dict()
    print("BEST", best, flush=True)
    work = panel.copy()
    rows = []
    for sym, g in work.groupby("symbol"):
        g = g.sort_values("date")
        f = factor_series(g, best["factor"])
        ph = g["in_pheno"] if best["pheno_only"] else None
        pos = make_position(f, best["mode"], float(best["thr"]), float(best["direction"]), int(best["hold"]), ph)
        tmp = g[["date", "symbol", "fwd_ret"]].copy()
        tmp["signal"] = pos
        tmp["pnl"] = pos * g["fwd_ret"].fillna(0.0).to_numpy()
        tmp["factor"] = f.to_numpy()
        rows.append(tmp)
    daily = pd.concat(rows, ignore_index=True)
    book = daily.groupby("date", as_index=False).agg(pnl=("pnl", "mean"), n_active=("signal", lambda s: int((s.abs() > 1e-8).sum())))
    book["日期"] = pd.to_datetime(book["date"]).dt.strftime("%Y-%m-%d")
    book.drop(columns=["date"]).to_csv(OUT / "best_book_daily.csv", index=False)
    daily.assign(日期=lambda d: d["date"].dt.strftime("%Y-%m-%d")).drop(columns=["date"]).to_csv(
        OUT / "best_legs_daily.csv", index=False
    )

    # also best among good if any
    best_good = good.iloc[0].to_dict() if len(good) else None

    report = {
        "n_tried_approx": n_try,
        "n_kept": int(len(res)),
        "n_good": int(len(good)),
        "best": best,
        "best_good": best_good,
        "top10": top.head(10).to_dict(orient="records"),
        "criteria_good": "sharpe>=0.5 & ir>=0.2 & |ic|>=0.02",
        "note": "IC/IR on factor*direction vs fwd_ret; Sharpe on EW book of signal*fwd_ret",
    }
    (OUT / "search_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"kept={len(res)} good={len(good)} -> {OUT}", flush=True)
    print("top5 sharpe/ic/ir:", flush=True)
    print(top.head(5)[["factor", "mode", "thr", "direction", "hold", "pheno_only", "sharpe", "ic", "ir", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
