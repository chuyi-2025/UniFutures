#!/usr/bin/env python3
"""Walk-forward Ridge every quarter, 2020-2026, nearly all main contracts.

Each quarter: re-rank features by IS IC, refit one Ridge, evaluate that quarter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from build_feature.build_xgb_feature import (  # noqa: E402
    FEATURE_NAMES,
    WINDOW,
    add_features,
    iter_segments,
    valid_window,
)
from shared import REMOVED_SYMBOLS  # noqa: E402

XGB_PATH = Path("/home/workspace/lab/UniFutures/data/xgb_features.csv")
INFER_XGB = Path("/home/workspace/lab/UniFutures/data/infer/features/xgb.csv")
TREE = Path("/home/workspace/lab/Tree-Stock/futures/data/all_contracts")
OUT_DIR = Path("/home/workspace/lab/UniFutures/data/infer/results/linear_factor")
DEAD = frozenset({"WT", "RO", "ER", "WS", "ME", "TC"})
SKIP_FEATS = {"month", "weekday", "day"}
INIT_CAP = 1_000_000.0
EVAL_START = pd.Timestamp("2020-01-01")
EVAL_END = pd.Timestamp("2026-09-08")
L2 = 8.0
MIN_CS = 15
MIN_TRAIN_DAYS = 252
LS_FRAC = 0.2


def pick_main_fast(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    code = out["code"].astype(str)
    yy = 2000 + pd.to_numeric(code.str[-4:-2], errors="coerce")
    mm = pd.to_numeric(code.str[-2:], errors="coerce")
    out["_ord"] = yy * 12 + mm
    out = out.dropna(subset=["_ord"])
    dt = pd.to_datetime(out["date"])
    thr_m = dt.dt.month + 3
    extra = (thr_m - 1) // 12
    thr_y = dt.dt.year + extra
    thr_m = ((thr_m - 1) % 12) + 1
    out["_ok"] = out["_ord"] > (thr_y * 12 + thr_m)
    ok = out[out["_ok"]]
    pick = ok.sort_values(["date", "_ord"]).groupby("date", as_index=False).head(1)
    missing = set(out["date"].unique()) - set(pick["date"].unique())
    if missing:
        fb = out[out["date"].isin(missing)].sort_values(["date", "_ord"], ascending=[True, False])
        fb = fb.groupby("date", as_index=False).head(1)
        pick = pd.concat([pick, fb], ignore_index=True)
    return pick.drop(columns=["_ord", "_ok"]).sort_values("date").reset_index(drop=True)


def load_fwd_ret() -> pd.DataFrame:
    frames = []
    for sym_dir in sorted(p for p in TREE.iterdir() if p.is_dir()):
        sym = sym_dir.name.upper()
        if sym in REMOVED_SYMBOLS or sym in DEAD:
            continue
        parts = []
        for csv in sorted(sym_dir.glob("*.csv")):
            try:
                raw = _read_contract_csv(csv)
            except Exception as exc:  # noqa: BLE001
                print(f"  ret skip {csv}: {exc}", flush=True)
                continue
            parts.append(raw[["date", "code", "close"]])
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True).sort_values(["code", "date"])
        nxt = df.groupby("code")["close"].shift(-1)
        same = df.groupby("code")["date"].shift(-1)
        gap = (same - df["date"]).dt.days
        df["fwd_ret"] = np.log(nxt / df["close"])
        df.loc[gap.isna() | (gap > 10), "fwd_ret"] = np.nan
        df["symbol"] = sym
        frames.append(df[["date", "symbol", "code", "fwd_ret"]])
    return pd.concat(frames, ignore_index=True)


def _read_contract_csv(csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv, encoding="utf-8")
    raw.columns = [str(c).replace("\ufeff", "") for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["code"] = raw["code"].astype(str)
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    return raw.dropna(subset=["date", "close"])


def build_tail_features(min_date: pd.Timestamp) -> pd.DataFrame:
    """Rebuild recent features per contract file; skip files that fail."""
    chunks: list[pd.DataFrame] = []
    for sym_dir in sorted(p for p in TREE.iterdir() if p.is_dir()):
        sym = sym_dir.name.upper()
        if sym in REMOVED_SYMBOLS or sym in DEAD:
            continue
        for csv in sorted(sym_dir.glob("*.csv")):
            try:
                raw = _read_contract_csv(csv)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {csv.name}: {exc}", flush=True)
                continue
            if raw.empty or raw["date"].max() < min_date:
                continue
            try:
                recs = []
                for seg in iter_segments(raw, allow_tail=True):
                    feat = add_features(seg)
                    codes = seg["code"].astype(str).to_numpy()
                    closes = seg["close"].astype(float).to_numpy()
                    dates = pd.to_datetime(seg["date"]).to_numpy()
                    vals = feat.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
                    for i in range(WINDOW - 1, len(seg)):
                        dt = pd.Timestamp(dates[i])
                        if dt < min_date:
                            continue
                        if not valid_window(codes, closes, i):
                            continue
                        x = vals[i]
                        if not np.isfinite(x).all():
                            continue
                        recs.append((dt, codes[i], *x.tolist()))
                if recs:
                    cols = ["date", "code", *FEATURE_NAMES]
                    part = pd.DataFrame(recs, columns=cols)
                    part["symbol"] = sym
                    chunks.append(part)
            except Exception as exc:  # noqa: BLE001
                print(f"  feat fail {sym}/{csv.name}: {exc}", flush=True)
                continue
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def load_hist_features() -> pd.DataFrame:
    frames = []
    for path in (XGB_PATH, INFER_XGB):
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        rename = {f"f_{i}": name for i, name in enumerate(FEATURE_NAMES) if f"f_{i}" in df.columns}
        df = df.rename(columns=rename)
        keep = ["date", "symbol", "code", *FEATURE_NAMES]
        df = df[[c for c in keep if c in df.columns]]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out = out[~out["symbol"].isin(REMOVED_SYMBOLS | DEAD)]
    out = out.drop_duplicates(["date", "symbol", "code"], keep="last")
    return out


def build_panel(hist_start: pd.Timestamp | None = None) -> pd.DataFrame:
    print("loading historical features...", flush=True)
    hist = load_hist_features()
    print(f"  hist rows={len(hist)} last={hist['date'].max().date()}", flush=True)
    print("skipping Tree-Stock tail rebuild (native crash risk); post-2026-06-05 only infer 9 symbols", flush=True)
    tail = pd.DataFrame()
    feat = pd.concat([hist, tail], ignore_index=True) if not tail.empty else hist
    feat = feat.drop_duplicates(["date", "symbol", "code"], keep="last")

    print("picking main contracts...", flush=True)
    mains = []
    for sym, grp in feat.groupby("symbol"):
        p = pick_main_fast(grp)
        if not p.empty:
            mains.append(p)
    main = pd.concat(mains, ignore_index=True)

    print("loading forward returns...", flush=True)
    rets = load_fwd_ret()
    panel = main.merge(rets, on=["date", "symbol", "code"], how="left")
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    start = hist_start if hist_start is not None else pd.Timestamp("2010-01-01")
    panel = panel[(panel["date"] >= start) & (panel["date"] <= EVAL_END)]
    print(
        f"panel rows={len(panel)} symbols={panel['symbol'].nunique()} "
        f"{panel['date'].min().date()} -> {panel['date'].max().date()} "
        f"fwd_ret coverage={panel['fwd_ret'].notna().mean():.2%}",
        flush=True,
    )
    return panel


def rank_ic_daily(panel: pd.DataFrame, col: str) -> pd.Series:
    x = panel.pivot_table(index="date", columns="symbol", values=col, aggfunc="last")
    y = panel.pivot_table(index="date", columns="symbol", values="fwd_ret", aggfunc="last")
    xr = x.rank(axis=1)
    yr = y.rank(axis=1)
    mask = xr.notna() & yr.notna()
    n = mask.sum(axis=1)
    xr = xr.where(mask)
    yr = yr.where(mask)
    xm = xr.mean(axis=1)
    ym = yr.mean(axis=1)
    xd = xr.sub(xm, axis=0)
    yd = yr.sub(ym, axis=0)
    num = (xd * yd).sum(axis=1)
    den = np.sqrt((xd * xd).sum(axis=1) * (yd * yd).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    ic[n < MIN_CS] = np.nan
    return ic


def ic_stats(ic: pd.Series) -> dict:
    ic = ic.dropna()
    n = int(len(ic))
    if n < 20:
        return {"ic": np.nan, "ir": np.nan, "t": np.nan, "n": n}
    mu = float(ic.mean())
    sd = float(ic.std(ddof=1))
    ir = mu / sd if sd > 1e-12 else np.nan
    t = mu / (sd / np.sqrt(n)) if sd > 1e-12 else np.nan
    return {"ic": mu, "ir": ir, "t": t, "n": n}


def fit_ridge(train: pd.DataFrame, feats: list[str], l2: float = L2):
    X = train[feats].to_numpy(dtype=float)
    y = train["fwd_ret"].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z = (X - mu) / sd
    A = np.column_stack([np.ones(len(Z)), Z])
    xtx = A.T @ A
    xty = A.T @ y
    reg = np.eye(xtx.shape[0]) * l2
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xtx + reg, xty)
    return mu, sd, beta


def predict_ridge(df: pd.DataFrame, feats: list[str], mu, sd, beta) -> np.ndarray:
    X = df[feats].to_numpy(dtype=float)
    Z = (X - mu) / sd
    pred = beta[0] + Z @ beta[1:]
    pred[~np.isfinite(X).all(axis=1)] = np.nan
    return pred


def ls_pnl(day: pd.DataFrame, score: np.ndarray) -> float:
    s = pd.Series(score, index=day.index)
    y = day["fwd_ret"]
    ok = s.notna() & y.notna()
    s, y = s[ok], y[ok]
    n = int(len(s))
    k = max(3, int(np.floor(n * LS_FRAC)))
    if n < 2 * k:
        return np.nan
    w = pd.Series(0.0, index=s.index)
    w.loc[s.nlargest(k).index] = 1.0 / k
    w.loc[s.nsmallest(k).index] = -1.0 / k
    return float((w * y).sum())


def sign_pnl(day: pd.DataFrame, score: np.ndarray) -> float:
    pos = np.sign(score)
    y = day["fwd_ret"].to_numpy()
    mask = np.isfinite(pos) & np.isfinite(y) & (pos != 0)
    if mask.sum() < 5:
        return np.nan
    return float(np.mean(pos[mask] * y[mask]))


def bh_pnl(day: pd.DataFrame) -> float:
    y = day["fwd_ret"].dropna()
    return float(y.mean()) if len(y) else np.nan


def metrics_from_pnl(pnl: pd.Series) -> dict:
    pnl = pnl.dropna()
    if pnl.empty:
        return {"return": np.nan, "sharpe": np.nan, "max_dd": np.nan, "days": 0}
    cap = INIT_CAP * np.exp(pnl.cumsum())
    ret = float(cap.iloc[-1] / INIT_CAP - 1.0)
    sharpe = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)) if pnl.std(ddof=1) > 1e-12 else 0.0
    peak = cap.cummax()
    max_dd = float((cap / peak - 1.0).min())
    return {"return": ret, "sharpe": sharpe, "max_dd": max_dd, "days": int(len(pnl))}


def classify_regime(q_ret: float, q_vol: float, vol_cut: float) -> str:
    if not np.isfinite(q_ret) or not np.isfinite(q_vol):
        return "未知"
    if q_ret >= 0.06:
        return "上涨"
    if q_ret <= -0.06:
        return "下跌"
    if q_vol >= vol_cut:
        return "高波震荡"
    return "低波震荡"


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
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
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    feats = [c for c in FEATURE_NAMES if c not in SKIP_FEATS and c in panel.columns]

    print("precomputing daily ICs...", flush=True)
    ic_map = {f: rank_ic_daily(panel, f) for f in feats}

    quarters = pd.period_range("2020Q1", "2026Q3", freq="Q")
    q_rows = []
    eq_ls = []
    eq_sign = []
    eq_bh = []
    feat_hist = []

    for q in quarters:
        q_start, q_end = q.start_time, min(q.end_time, EVAL_END)
        train = panel[(panel["date"] < q_start) & panel["fwd_ret"].notna()]
        test = panel[(panel["date"] >= q_start) & (panel["date"] <= q_end) & panel["fwd_ret"].notna()]
        if train["date"].nunique() < MIN_TRAIN_DAYS or test["date"].nunique() < 15:
            print(f"skip {q}: train_days={train['date'].nunique()} test_days={test['date'].nunique()}", flush=True)
            continue

        ranked = []
        for f in feats:
            st = ic_stats(ic_map[f].loc[ic_map[f].index < q_start])
            ranked.append({"feature": f, **st})
        rtab = pd.DataFrame(ranked)
        rtab["abs_t"] = rtab["t"].abs()
        rtab = rtab.sort_values("abs_t", ascending=False)
        usable = rtab[(rtab["t"].abs() >= 1.5) & rtab["ic"].notna()]
        use_feats = usable["feature"].head(12).tolist() or rtab.head(8)["feature"].tolist()

        mu, sd, beta = fit_ridge(train, use_feats)
        test = test.copy()
        test["score"] = predict_ridge(test, use_feats, mu, sd, beta)

        daily_ls, daily_sign, daily_bh = [], [], []
        for dt, day in test.groupby("date"):
            sc = day["score"].to_numpy()
            daily_ls.append((dt, ls_pnl(day, sc)))
            daily_sign.append((dt, sign_pnl(day, sc)))
            daily_bh.append((dt, bh_pnl(day)))
        ls = pd.Series({d: v for d, v in daily_ls})
        sg = pd.Series({d: v for d, v in daily_sign})
        bh = pd.Series({d: v for d, v in daily_bh})

        oos_ic = []
        for dt, day in test.groupby("date"):
            if day["score"].notna().sum() < MIN_CS:
                continue
            xr = day["score"].rank()
            yr = day["fwd_ret"].rank()
            mask = xr.notna() & yr.notna()
            if mask.sum() < MIN_CS:
                continue
            oos_ic.append(float(xr[mask].corr(yr[mask], method="pearson")))
        oos = ic_stats(pd.Series(oos_ic))

        m_ls, m_sg, m_bh = metrics_from_pnl(ls), metrics_from_pnl(sg), metrics_from_pnl(bh)
        q_vol = float(bh.std(ddof=1) * np.sqrt(252)) if bh.std(ddof=1) > 0 else np.nan
        row = {
            "quarter": str(q),
            "start": str(q_start.date()),
            "end": str(test["date"].max().date()),
            "n_symbols": int(test["symbol"].nunique()),
            "n_days": int(test["date"].nunique()),
            "n_train_days": int(train["date"].nunique()),
            "features": use_feats,
            "top_feature": use_feats[0] if use_feats else None,
            "ridge_oos_ic": oos["ic"],
            "ridge_oos_ir": oos["ir"],
            "ridge_oos_t": oos["t"],
            "ls": m_ls,
            "sign_ew": m_sg,
            "buyhold": m_bh,
            "mkt_vol": q_vol,
        }
        q_rows.append(row)
        feat_hist.append({"quarter": str(q), "features": use_feats})
        for dt, v in ls.items():
            eq_ls.append({"date": dt, "pnl": v, "quarter": str(q)})
        for dt, v in sg.items():
            eq_sign.append({"date": dt, "pnl": v, "quarter": str(q)})
        for dt, v in bh.items():
            eq_bh.append({"date": dt, "pnl": v, "quarter": str(q)})
        print(
            f"{q} n={row['n_symbols']} days={row['n_days']} "
            f"LS={m_ls['return']:+.1%} sh={m_ls['sharpe']:.2f} "
            f"sign={m_sg['return']:+.1%} BH={m_bh['return']:+.1%} "
            f"IC={oos['ic'] if oos['ic']==oos['ic'] else float('nan'):.3f} top={use_feats[0]}",
            flush=True,
        )

    vols = [r["mkt_vol"] for r in q_rows if np.isfinite(r["mkt_vol"])]
    vol_cut = float(np.median(vols)) if vols else 0.15
    for r in q_rows:
        r["regime"] = classify_regime(r["buyhold"]["return"], r["mkt_vol"], vol_cut)

    def to_equity(recs):
        s = pd.Series({r["date"]: r["pnl"] for r in recs}).sort_index().dropna()
        eq = np.exp(s.cumsum())
        return [{"date": str(i.date()), "equity": float(v)} for i, v in eq.items()]

    regime_sum = []
    for name, grp in pd.DataFrame(
        [
            {
                "regime": r["regime"],
                "ls": r["ls"]["return"],
                "sign": r["sign_ew"]["return"],
                "bh": r["buyhold"]["return"],
                "ic": r["ridge_oos_ic"],
                "sharpe": r["ls"]["sharpe"],
            }
            for r in q_rows
        ]
    ).groupby("regime"):
        regime_sum.append(
            {
                "regime": name,
                "n_quarters": int(len(grp)),
                "mean_ls": float(grp["ls"].mean()),
                "win_ls": int((grp["ls"] > 0).sum()),
                "mean_sign": float(grp["sign"].mean()),
                "mean_bh": float(grp["bh"].mean()),
                "mean_ic": float(grp["ic"].mean()),
                "mean_sharpe": float(grp["sharpe"].mean()),
            }
        )

    payload = {
        "coverage": {
            "symbols": int(panel.loc[panel["date"] >= EVAL_START, "symbol"].nunique()),
            "symbol_list": sorted(panel.loc[panel["date"] >= EVAL_START, "symbol"].unique().tolist()),
            "start": "2020-01-01",
            "end": str(panel["date"].max().date()),
            "n_quarters": len(q_rows),
            "removed": sorted(REMOVED_SYMBOLS | DEAD),
            "refit": "every calendar quarter, expanding train before quarter start",
            "model": "one Ridge on top-12 |t|>=1.5 features, predict next-day log return",
            "ls": "long top 20% / short bottom 20% of that day's mains",
            "vol_cut": vol_cut,
        },
        "quarters": [
            {
                "quarter": r["quarter"],
                "start": r["start"],
                "end": r["end"],
                "regime": r["regime"],
                "n_symbols": r["n_symbols"],
                "n_days": r["n_days"],
                "top_feature": r["top_feature"],
                "features": r["features"],
                "oos_ic": fmt(r["ridge_oos_ic"], 4),
                "oos_ir": fmt(r["ridge_oos_ir"], 3),
                "oos_t": fmt(r["ridge_oos_t"], 2),
                "ls_return": fmt(r["ls"]["return"], 4),
                "ls_sharpe": fmt(r["ls"]["sharpe"], 3),
                "ls_mdd": fmt(r["ls"]["max_dd"], 4),
                "sign_return": fmt(r["sign_ew"]["return"], 4),
                "sign_sharpe": fmt(r["sign_ew"]["sharpe"], 3),
                "bh_return": fmt(r["buyhold"]["return"], 4),
                "mkt_vol": fmt(r["mkt_vol"], 3),
            }
            for r in q_rows
        ],
        "regime_summary": regime_sum,
        "feature_hist": feat_hist,
        "equity": {
            "ls": to_equity(eq_ls),
            "sign": to_equity(eq_sign),
            "buyhold": to_equity(eq_bh),
        },
        "full": {
            "ls": metrics_from_pnl(pd.Series({r["date"]: r["pnl"] for r in eq_ls})),
            "sign": metrics_from_pnl(pd.Series({r["date"]: r["pnl"] for r in eq_sign})),
            "buyhold": metrics_from_pnl(pd.Series({r["date"]: r["pnl"] for r in eq_bh})),
        },
    }
    path = OUT_DIR / "walkforward_report.json"
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2))
    pd.DataFrame(payload["quarters"]).to_csv(OUT_DIR / "walkforward_quarters.csv", index=False)
    print(json.dumps(clean({"coverage": payload["coverage"], "full": payload["full"], "regime_summary": regime_sum}), ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
