#!/usr/bin/env python3
"""Search simple macro strategies for AU/AG/SC: Sharpe, maxDD, IC, IR."""

from __future__ import annotations

import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_prices import align_all  # noqa: E402
from common import RESULT_DIR, TARGETS, ensure_dirs  # noqa: E402

OUT = RESULT_DIR / "search"
MIN_ACTIVE = 40

# Prefer economically linked factors; exclude commodity CFD levels as primary
FACTORS = [
    "dxy_z60",
    "dxy_chg5",
    "dxy_chg20",
    "usd_broad_z60",
    "hike_proxy_z60",
    "hike_proxy_chg",
    "us_2y_z60",
    "us_2y_chg5",
    "us_10y_z60",
    "us_curve_10y2y_z60",
    "curve_10_2_z60",
    "vix_z60",
    "vix_chg",
    "unemp_chg",
    "us_unemp_z60",
    "cpi_yoy_z60",
    "usd_hawkish",
    "risk_off",
    "labor_soft",
]


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
    if len(r) < 40 or float(r.std()) == 0:
        return None
    return float(r.mean() / r.std() * np.sqrt(252))


def max_dd(r: pd.Series) -> float | None:
    r = r.fillna(0.0)
    if len(r) < 5:
        return None
    nav = (1.0 + r).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def hold_signal(raw: np.ndarray, hold: int) -> np.ndarray:
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


def make_pos(f: pd.Series, mode: str, thr: float, direction: float, hold: int) -> np.ndarray:
    x = f.fillna(0.0).to_numpy(dtype=float)
    if mode == "cont":
        return np.clip(x, -2, 2) / 2.0 * direction
    if mode == "sign":
        raw = np.where(np.abs(x) >= thr, np.sign(x) * direction, 0.0)
        return hold_signal(raw, hold) if hold > 1 else raw
    if mode == "long_only":
        raw = np.where(x * direction >= thr, direction, 0.0)
        return hold_signal(raw, hold) if hold > 1 else raw
    raise ValueError(mode)


def ic_metrics(fac: pd.Series, ret: pd.Series, dates: pd.Series) -> dict:
    sub = pd.DataFrame({"f": fac, "r": ret, "date": dates}).dropna()
    if len(sub) < 80:
        return {"ic": None, "ir": None, "ic_pos": None, "n_m": 0}
    ic = spearman_ic(sub["f"], sub["r"], min_n=40)
    months = []
    for _, g in sub.groupby(pd.to_datetime(sub["date"]).dt.to_period("M")):
        if len(g) < 8:
            continue
        months.append(spearman_ic(g["f"], g["r"], min_n=8))
    months = [m for m in months if m == m]
    if len(months) < 8:
        return {"ic": None if ic != ic else round(ic, 4), "ir": None, "ic_pos": None, "n_m": len(months)}
    arr = np.asarray(months, float)
    ir = float(arr.mean() / arr.std()) if arr.std() > 1e-12 else None
    return {
        "ic": round(float(ic), 4) if ic == ic else None,
        "ir": round(ir, 3) if ir is not None else None,
        "ic_pos": round(float((arr > 0).mean()), 3),
        "n_m": len(months),
    }


def time_split(pnl: pd.Series) -> dict:
    pnl = pnl.sort_index()
    mid = pnl.index[len(pnl) // 2]
    a, b = pnl[pnl.index <= mid], pnl[pnl.index > mid]
    sa, sb = sharpe(a), sharpe(b)
    return {
        "sh_full": sharpe(pnl),
        "sh_first": sa,
        "sh_second": sb,
        "both_pos": bool(sa is not None and sb is not None and sa > 0 and sb > 0),
        "mid": str(pd.Timestamp(mid).date()),
    }


def fwd_sum(s: pd.Series, h: int) -> pd.Series:
    x = s.fillna(0.0)
    c = x[::-1].cumsum()[::-1]
    fut = c - c.shift(-h).fillna(0.0)
    fut.iloc[-h:] = np.nan
    return fut


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    aligned = RESULT_DIR / "aligned_panel.parquet"
    if not aligned.exists():
        print("building aligned panel...", flush=True)
        panel = align_all()
        panel.to_parquet(aligned, index=False)
    else:
        panel = pd.read_parquet(aligned)
        panel["date"] = pd.to_datetime(panel["date"])

    panel = panel[panel["date"] >= "2018-01-01"].copy()
    panel = panel.sort_values(["symbol", "date"])
    panel["fwd_ret_5"] = panel.groupby("symbol")["fwd_ret"].transform(lambda s: fwd_sum(s, 5))

    available = [f for f in FACTORS if f in panel.columns]
    print(f"factors available {len(available)}/{len(FACTORS)}", flush=True)

    # IC board
    ic_rows = []
    for sym, g in panel.groupby("symbol"):
        for fac in available:
            for direction in (1.0, -1.0):
                for hz in ("fwd_ret", "fwd_ret_5"):
                    m = ic_metrics(g[fac] * direction, g[hz], g["date"])
                    ic_rows.append({"symbol": sym, "factor": fac, "direction": direction, "horizon": hz, **m})
    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(OUT / "ic_ir_board.csv", index=False)

    modes = ["cont", "sign", "long_only"]
    thrs = [0.5, 1.0, 1.5]
    holds = [1, 3, 5, 10]
    rows = []
    print("trading grid...", flush=True)
    for sym in TARGETS:
        g0 = panel[panel["symbol"] == sym]
        for fac, mode, thr, hold, direction in product(available, modes, thrs, holds, (1.0, -1.0)):
            if mode == "cont" and not (thr == 0.5 and hold == 1):
                continue
            thr_u, hold_u = (0.0, 1) if mode == "cont" else (thr, hold)
            pos = make_pos(g0[fac], mode, thr_u, direction, hold_u)
            live = g0.copy()
            live["signal"] = pos
            live["pnl"] = pos * live["fwd_ret"].fillna(0.0).to_numpy()
            act = int((np.abs(pos) > 1e-8).sum())
            if act < MIN_ACTIVE:
                continue
            pnl = live.set_index("date")["pnl"]
            split = time_split(pnl)
            sh = split["sh_full"]
            if sh is None:
                continue
            ic1 = ic_metrics(g0[fac] * direction, g0["fwd_ret"], g0["date"])
            ic5 = ic_metrics(g0[fac] * direction, g0["fwd_ret_5"], g0["date"])
            dd = max_dd(pnl)
            q = (sh or 0) + (0.35 if split["both_pos"] else 0)
            q += 0.5 * (ic5["ir"] or 0) + 0.3 * (ic1["ir"] or 0)
            q += 2.0 * abs(ic5["ic"] or 0) + 1.0 * abs(ic1["ic"] or 0)
            if dd is not None:
                q += min(0.3, max(0.0, -dd) * -1)  # milder dd helps a bit: dd is negative
                q += 0.15 * max(0.0, 0.15 + (dd or 0))  # reward dd > -15%
            if split["sh_first"] and split["sh_second"] and split["sh_first"] * split["sh_second"] < 0:
                q -= 0.4
            rows.append({
                "symbol": sym,
                "factor": fac,
                "mode": mode,
                "thr": thr_u,
                "hold": hold_u,
                "direction": direction,
                "sharpe": round(sh, 3),
                "sh_first": None if split["sh_first"] is None else round(split["sh_first"], 3),
                "sh_second": None if split["sh_second"] is None else round(split["sh_second"], 3),
                "oos_both_pos": split["both_pos"],
                "max_dd": None if dd is None else round(dd, 4),
                "ret": round(float(pnl.sum()), 4),
                "active": act,
                "ic1": ic1["ic"],
                "ir1": ic1["ir"],
                "ic5": ic5["ic"],
                "ir5": ic5["ir"],
                "quality": round(q, 4),
            })

    res = pd.DataFrame(rows).sort_values(["quality", "sharpe"], ascending=False)
    res.to_csv(OUT / "all_schemes.csv", index=False)

    robust = res[
        (res["sharpe"] >= 0.6)
        & (res["oos_both_pos"])
        & (res["ir5"].fillna(-9) >= 0.1)
        & (res["ic5"].abs().fillna(0) >= 0.02)
        & (res["max_dd"].fillna(-1) >= -0.25)
    ]
    soft = res[
        (res["sharpe"] >= 0.5)
        & (res["oos_both_pos"])
        & ((res["ir1"].fillna(-9) >= 0.08) | (res["ir5"].fillna(-9) >= 0.08))
        & (res["max_dd"].fillna(-1) >= -0.30)
    ]
    robust.to_csv(OUT / "robust_schemes.csv", index=False)
    soft.to_csv(OUT / "soft_robust_schemes.csv", index=False)

    picks = []
    for sym in TARGETS:
        pool = robust[robust["symbol"] == sym]
        if pool.empty:
            pool = soft[soft["symbol"] == sym]
        if pool.empty:
            pool = res[res["symbol"] == sym]
        picks.append(pool.iloc[0].to_dict())

    # book
    parts = []
    for p in picks:
        g = panel[panel["symbol"] == p["symbol"]].copy()
        pos = make_pos(g[p["factor"]], p["mode"], float(p["thr"]), float(p["direction"]), int(p["hold"]))
        g["signal"] = pos
        g["pnl"] = pos * g["fwd_ret"].fillna(0.0).to_numpy()
        parts.append(g[["date", "symbol", "signal", "pnl", "fwd_ret"]])
    legs = pd.concat(parts, ignore_index=True)
    book = legs.groupby("date")["pnl"].mean().sort_index()
    bsplit = time_split(book)
    legs.assign(日期=lambda d: d["date"].dt.strftime("%Y-%m-%d")).drop(columns=["date"]).to_csv(
        OUT / "picked_legs_daily.csv", index=False
    )
    pd.DataFrame({"日期": book.index.strftime("%Y-%m-%d"), "pnl": book.values}).to_csv(
        OUT / "picked_book_daily.csv", index=False
    )

    report = {
        "n_schemes": int(len(res)),
        "n_robust": int(len(robust)),
        "n_soft": int(len(soft)),
        "factors": available,
        "picks": picks,
        "book": {
            "sharpe": bsplit.get("sh_full"),
            "sh_first": bsplit.get("sh_first"),
            "sh_second": bsplit.get("sh_second"),
            "oos_both_pos": bsplit.get("both_pos"),
            "max_dd": max_dd(book),
            "ret": round(float(book.sum()), 4),
            "mid": bsplit.get("mid"),
        },
        "best_ic": ic_df.dropna(subset=["ir"]).sort_values("ir", ascending=False).head(12).to_dict(orient="records"),
        "note": "Research signal*log fwd_ret; macro T-1; hike_proxy = US2Y - daily FF (not CME FedWatch scrape).",
    }

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, str):
            return o
        try:
            if pd.isna(o):
                return None
        except Exception:
            pass
        return o

    (OUT / "search_report.json").write_text(json.dumps(conv(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print("robust", len(robust), "soft", len(soft), flush=True)
    print("picks:", flush=True)
    for p in picks:
        print(
            f"  {p['symbol']} {p['factor']} {p['mode']} thr={p['thr']} hold={p['hold']} dir={p['direction']} "
            f"sh={p['sharpe']} dd={p['max_dd']} ic5={p['ic5']} ir5={p['ir5']}",
            flush=True,
        )
    print("book", report["book"], flush=True)
    print("top IC/IR:", flush=True)
    print(ic_df.dropna(subset=["ir"]).sort_values("ir", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
