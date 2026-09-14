#!/usr/bin/env python3
"""Linear factor IC/IR/turnover/t-stat + 3-month main-contract backtest.

Compares V1-V4 factor iterations and a Ridge combo against PPO/Kronos/XGB/GAF
on the 9 infer symbols (same universe as latest_signals.csv).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CODE_DIR = Path(__file__).resolve().parents[1]
INFER_DIR = CODE_DIR / "infer"
sys.path.insert(0, str(INFER_DIR))
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "backtest"))

import config
from build_feature.build_xgb_feature import FEATURE_NAMES

INIT_CAP = 1_000_000.0
SYMBOLS = list(config.SYMBOLS)
SYMBOL_CN = dict(config.SYMBOL_CN)
XGB_PATH = config.XGB_FEATURES_CSV
GAF_DIR = config.GAF_FEATURES_DIR
RESULT_DIR = config.RESULTS_DIR
OUT_DIR = RESULT_DIR / "linear_factor"
IS_END = pd.Timestamp("2026-06-08")
OOS_START = pd.Timestamp("2026-06-09")
OOS_END = pd.Timestamp("2026-09-08")
SKIP_FEATS = {"month", "weekday", "day"}
STYLE_FEATS = ("volatility_20", "ret_std_20", "liquidity_30")


def pick_main_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized m+3 main-contract pick with farthest-month fallback."""
    out = df.copy()
    code = out["code"].astype(str)
    yy = 2000 + code.str[-4:-2].astype(int)
    mm = code.str[-2:].astype(int)
    out["_ord"] = yy * 12 + mm
    dt = pd.to_datetime(out["date"])
    thr_m = dt.dt.month + 3
    extra = (thr_m - 1) // 12
    thr_y = dt.dt.year + extra
    thr_m = ((thr_m - 1) % 12) + 1
    thr_ord = thr_y * 12 + thr_m
    out["_ok"] = out["_ord"] > thr_ord
    ok = out[out["_ok"]]
    pick = ok.sort_values(["date", "_ord"]).groupby("date", as_index=False).head(1)
    missing = set(out["date"].unique()) - set(pick["date"].unique())
    if missing:
        fb = out[out["date"].isin(missing)].sort_values(["date", "_ord"], ascending=[True, False])
        fb = fb.groupby("date", as_index=False).head(1)
        pick = pd.concat([pick, fb], ignore_index=True)
    return pick.drop(columns=["_ord", "_ok"]).sort_values("date").reset_index(drop=True)


def load_main_panel() -> pd.DataFrame:
    xgb = pd.read_csv(XGB_PATH, parse_dates=["date"])
    xgb = xgb[xgb["symbol"].isin(SYMBOLS)].copy()
    rename = {f"f_{i}": name for i, name in enumerate(FEATURE_NAMES) if f"f_{i}" in xgb.columns}
    xgb = xgb.rename(columns=rename)
    frames = []
    for sym, grp in xgb.groupby("symbol"):
        main = pick_main_fast(grp)
        if main.empty:
            continue
        gaf_path = GAF_DIR / f"{sym}.csv"
        gaf = pd.read_csv(gaf_path, parse_dates=["date"], usecols=["date", "code", "log_ret_1d"])
        main = main.merge(gaf, on=["date", "code"], how="left")
        frames.append(main)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    panel["fwd_ret"] = panel["log_ret_1d"]
    return panel


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return np.nan
    if np.nanstd(a[mask]) < 1e-12 or np.nanstd(b[mask]) < 1e-12:
        return np.nan
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def cs_ic_series(df: pd.DataFrame, col: str) -> pd.Series:
    rows = []
    for dt, grp in df.groupby("date"):
        if len(grp) < 5:
            continue
        rows.append((dt, _safe_spearman(grp[col].to_numpy(), grp["fwd_ret"].to_numpy())))
    if not rows:
        return pd.Series(dtype=float)
    idx, vals = zip(*rows)
    return pd.Series(vals, index=pd.DatetimeIndex(idx), dtype=float)


def ts_ic_mean(df: pd.DataFrame, col: str) -> float:
    ics = []
    for _, grp in df.groupby("symbol"):
        ics.append(_safe_spearman(grp[col].to_numpy(), grp["fwd_ret"].to_numpy()))
    return float(np.nanmean(ics)) if ics else np.nan


def ic_stats(ic: pd.Series) -> dict:
    ic = ic.dropna()
    n = int(len(ic))
    if n < 5:
        return {"ic": np.nan, "ir": np.nan, "t": np.nan, "n": n}
    mu = float(ic.mean())
    sd = float(ic.std(ddof=1)) if n > 1 else np.nan
    ir = mu / sd if sd and sd > 1e-12 else np.nan
    t = mu / (sd / np.sqrt(n)) if sd and sd > 1e-12 else np.nan
    return {"ic": mu, "ir": ir, "t": t, "n": n}


def ls_weights(grp: pd.DataFrame, col: str, n_side: int = 3) -> pd.Series:
    s = grp[col]
    valid = s.dropna()
    w = pd.Series(0.0, index=grp.index)
    if len(valid) < n_side * 2:
        return w
    long_idx = valid.nlargest(n_side).index
    short_idx = valid.nsmallest(n_side).index
    w.loc[long_idx] = 1.0 / n_side
    w.loc[short_idx] = -1.0 / n_side
    return w


def factor_turnover(df: pd.DataFrame, col: str, n_side: int = 3) -> float:
    weights = []
    dates = []
    for dt, grp in df.groupby("date"):
        w = ls_weights(grp, col, n_side)
        weights.append(pd.Series(w.to_numpy(), index=grp["symbol"].to_numpy()))
        dates.append(dt)
    if len(weights) < 2:
        return np.nan
    turns = []
    prev = weights[0]
    for cur in weights[1:]:
        aligned = pd.concat([prev, cur], axis=1).fillna(0.0)
        turns.append(0.5 * float(np.abs(aligned.iloc[:, 1] - aligned.iloc[:, 0]).sum()))
        prev = cur
    return float(np.mean(turns)) if turns else np.nan


def sign_turnover(df: pd.DataFrame, col: str) -> float:
    turns = []
    for _, grp in df.groupby("symbol"):
        pos = np.sign(grp[col].to_numpy())
        pos = np.where(np.isfinite(pos), pos, 0.0)
        if len(pos) < 2:
            continue
        turns.append(0.5 * float(np.mean(np.abs(np.diff(pos)))))
    return float(np.nanmean(turns)) if turns else np.nan


def eval_factor(df: pd.DataFrame, col: str) -> dict:
    ic = cs_ic_series(df, col)
    stats = ic_stats(ic)
    stats["ts_ic"] = ts_ic_mean(df, col)
    stats["turnover"] = factor_turnover(df, col)
    stats["sign_turnover"] = sign_turnover(df, col)
    return stats


def neutralize_cs(df: pd.DataFrame, col: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for _, grp in df.groupby("date"):
        x = grp[col]
        mu = x.mean()
        sd = x.std(ddof=0)
        if pd.notna(sd) and sd > 1e-12:
            out.loc[grp.index] = (x - mu) / sd
        else:
            out.loc[grp.index] = x - mu
    return out


def ema_by_symbol(df: pd.DataFrame, col: str, span: int = 5) -> pd.Series:
    return df.groupby("symbol")[col].transform(lambda s: s.ewm(span=span, adjust=False).mean())


def residualize_style(df: pd.DataFrame, col: str, styles: tuple[str, ...] = STYLE_FEATS) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for _, grp in df.groupby("date"):
        y = grp[col].to_numpy(dtype=float)
        cols = [c for c in styles if c in grp.columns]
        X = grp[cols].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        resid = np.full(len(grp), np.nan)
        if mask.sum() >= (X.shape[1] + 2):
            A = np.column_stack([np.ones(mask.sum()), X[mask]])
            try:
                beta, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
                pred = A @ beta
                resid[np.where(mask)[0]] = y[mask] - pred
            except np.linalg.LinAlgError:
                resid[np.where(mask)[0]] = y[mask] - np.nanmean(y[mask])
        out.loc[grp.index] = resid
    return out


def backtest_sign(df: pd.DataFrame, col: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    part = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    rows = []
    for sym, grp in part.groupby("symbol"):
        g = grp.sort_values("date").reset_index(drop=True)
        pos = np.sign(g[col].to_numpy())
        pos = np.where(np.isfinite(pos), pos, 0.0)
        ret = g["fwd_ret"].fillna(0.0).to_numpy()
        pnl = pos * ret
        cap = INIT_CAP * np.exp(np.cumsum(pnl))
        daily = pd.DataFrame(
            {
                "date": g["date"],
                "symbol": sym,
                "code": g["code"],
                "position": pos.astype(int),
                "fwd_ret": ret,
                "pnl": pnl,
                "capital": cap,
            }
        )
        rows.append(daily)
    daily = pd.concat(rows, ignore_index=True)
    return {"daily": daily, "per_symbol": symbol_metrics(daily), "portfolio": portfolio_metrics(daily)}


def backtest_ls(df: pd.DataFrame, col: str, start: pd.Timestamp, end: pd.Timestamp, n_side: int = 3) -> dict:
    part = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    pieces = []
    for dt, grp in part.groupby("date"):
        w = ls_weights(grp, col, n_side)
        rec = grp[["date", "symbol", "code", "fwd_ret"]].copy()
        rec["weight"] = w.to_numpy()
        rec["pnl"] = rec["weight"] * rec["fwd_ret"].fillna(0.0)
        rec["position"] = np.sign(rec["weight"]).astype(int)
        pieces.append(rec)
    daily = pd.concat(pieces, ignore_index=True)
    port = daily.groupby("date", as_index=False)["pnl"].sum().sort_values("date")
    port["capital"] = INIT_CAP * np.exp(port["pnl"].cumsum())
    return {"daily": daily, "portfolio_daily": port, "portfolio": _metrics_from_pnl(port["pnl"].to_numpy(), port["capital"].to_numpy())}


def symbol_metrics(daily: pd.DataFrame) -> list[dict]:
    out = []
    for sym, g in daily.groupby("symbol"):
        g = g.sort_values("date")
        out.append({"symbol": sym, **_metrics_from_pnl(g["pnl"].to_numpy(), g["capital"].to_numpy())})
    return out


def portfolio_metrics(daily: pd.DataFrame) -> dict:
    port = daily.groupby("date", as_index=False)["pnl"].mean().sort_values("date")
    cap = INIT_CAP * np.exp(port["pnl"].cumsum())
    return _metrics_from_pnl(port["pnl"].to_numpy(), cap.to_numpy())


def _metrics_from_pnl(pnl: np.ndarray, capital: np.ndarray | None = None) -> dict:
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[np.isfinite(pnl)]
    if capital is None:
        capital = INIT_CAP * np.exp(np.cumsum(pnl))
    ret = float(capital[-1] / INIT_CAP - 1.0) if len(capital) else np.nan
    sharpe = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)) if len(pnl) > 1 and pnl.std(ddof=1) > 1e-12 else 0.0
    peak = np.maximum.accumulate(capital)
    max_dd = float((capital / peak - 1.0).min()) if len(capital) else 0.0
    return {"return": ret, "sharpe": sharpe, "max_dd": max_dd, "days": int(len(pnl))}


def existing_model_oos() -> list[dict]:
    rows = []
    for model in ("ppo", "kronos", "xgb", "gaf"):
        for sym in SYMBOLS:
            path = RESULT_DIR / model / sym.lower() / "daily.csv"
            if not path.exists():
                continue
            d = pd.read_csv(path, parse_dates=["date"])
            d = d[(d["date"] >= OOS_START) & (d["date"] <= OOS_END)].copy()
            if d.empty:
                continue
            pnl = d["position"].to_numpy() * d["log_ret_1d"].fillna(0.0).to_numpy()
            cap = INIT_CAP * np.exp(np.cumsum(pnl))
            rows.append(
                {
                    "model": model,
                    "symbol": sym,
                    "name_cn": SYMBOL_CN[sym],
                    **_metrics_from_pnl(pnl, cap),
                    "last_date": str(d["date"].iloc[-1].date()),
                    "last_code": str(d["code"].iloc[-1]),
                    "last_position": int(d["position"].iloc[-1]),
                }
            )
    return rows


def existing_portfolio(model_rows: list[dict]) -> list[dict]:
    out = []
    by_model: dict[str, list] = {}
    for r in model_rows:
        by_model.setdefault(r["model"], []).append(r)
    for model, items in by_model.items():
        rets = [x["return"] for x in items]
        sharpes = [x["sharpe"] for x in items]
        dds = [x["max_dd"] for x in items]
        out.append(
            {
                "model": model,
                "mean_return": float(np.mean(rets)),
                "median_return": float(np.median(rets)),
                "mean_sharpe": float(np.mean(sharpes)),
                "mean_max_dd": float(np.mean(dds)),
                "win_symbols": int(sum(1 for x in rets if x > 0)),
                "n_symbols": len(items),
            }
        )
    return out


def fit_ridge(train: pd.DataFrame, feats: list[str], l2: float = 5.0) -> tuple[np.ndarray, list[str]]:
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
    return np.concatenate([[1.0], mu, sd, beta]), feats


def apply_ridge(df: pd.DataFrame, pack: np.ndarray, feats: list[str]) -> pd.Series:
    n = len(feats)
    mu = pack[1 : 1 + n]
    sd = pack[1 + n : 1 + 2 * n]
    beta = pack[1 + 2 * n :]
    X = df[feats].to_numpy(dtype=float)
    Z = (X - mu) / sd
    pred = beta[0] + Z @ beta[1:]
    pred[~np.isfinite(X).all(axis=1)] = np.nan
    return pd.Series(pred, index=df.index)


def latest_signals(df: pd.DataFrame, col: str) -> list[dict]:
    last = df[df["date"] == df["date"].max()].copy()
    rows = []
    for _, r in last.iterrows():
        score = r[col]
        pos = int(np.sign(score)) if np.isfinite(score) else 0
        rows.append(
            {
                "symbol": r["symbol"],
                "name_cn": SYMBOL_CN.get(r["symbol"], r["symbol"]),
                "date": str(pd.Timestamp(r["date"]).date()),
                "code": r["code"],
                "score": None if not np.isfinite(score) else float(score),
                "position": pos,
                "position_name": {1: "long", -1: "short", 0: "flat"}[pos],
            }
        )
    return rows


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), nd)


def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading panel...", flush=True)
    panel = load_main_panel()
    print(f"panel rows={len(panel)} dates={panel['date'].nunique()} last={panel['date'].max().date()}", flush=True)
    panel = panel.dropna(subset=["fwd_ret"], how="any")
    is_df = panel[panel["date"] <= IS_END].copy()
    oos_df = panel[(panel["date"] >= OOS_START) & (panel["date"] <= OOS_END)].copy()

    feat_rows = []
    for name in FEATURE_NAMES:
        if name in SKIP_FEATS or name not in is_df.columns:
            continue
        st = eval_factor(is_df, name)
        feat_rows.append({"feature": name, **st})
    feat_tab = pd.DataFrame(feat_rows)
    feat_tab["abs_ic"] = feat_tab["ic"].abs()
    feat_tab = feat_tab.sort_values(["abs_ic", "t"], ascending=False)
    feat_tab.to_csv(OUT_DIR / "feature_ic_is.csv", index=False)

    # V1: oversold rebound = -20d momentum (slide analog)
    panel["v1"] = -panel["momentum_20"]
    # also try the empirically strongest raw feature (signed by IS IC)
    top = feat_tab.iloc[0]
    top_name = str(top["feature"])
    top_sign = 1.0 if top["ic"] >= 0 else -1.0
    panel["v1_best"] = top_sign * panel[top_name]

    # V2: cross-sectional neutralize
    panel["v2"] = neutralize_cs(panel, "v1")
    panel["v2_best"] = neutralize_cs(panel, "v1_best")

    # V3: EMA smooth
    panel["v3"] = ema_by_symbol(panel, "v2", span=5)
    panel["v3_best"] = ema_by_symbol(panel, "v2_best", span=5)

    # V4: strip vol/liquidity style
    panel["v4"] = residualize_style(panel, "v3")
    panel["v4_best"] = residualize_style(panel, "v3_best")

    is_df = panel[panel["date"] <= IS_END].copy()
    oos_df = panel[(panel["date"] >= OOS_START) & (panel["date"] <= OOS_END)].copy()

    versions = [
        ("V1 初始超跌反弹", "v1", "-momentum_20"),
        ("V2 截面中性化", "v2", "cs z-score of V1"),
        ("V3 信号平滑", "v3", "EMA5 of V2"),
        ("V4 剥离波动/流动性风格", "v4", "residual vs vol/liquidity"),
        (f"V1-best {top_name}", "v1_best", f"{top_sign:+.0f}*{top_name}"),
        ("V4-best", "v4_best", f"V1-best through V2-V4"),
    ]

    version_stats = []
    version_bt = []
    version_latest = {}
    for label, col, recipe in versions:
        is_s = eval_factor(is_df, col)
        oos_s = eval_factor(oos_df, col)
        bt_sign = backtest_sign(panel, col, OOS_START, OOS_END)
        bt_ls = backtest_ls(panel, col, OOS_START, OOS_END)
        version_stats.append(
            {
                "version": label,
                "col": col,
                "recipe": recipe,
                "is": is_s,
                "oos": oos_s,
            }
        )
        version_bt.append(
            {
                "version": label,
                "col": col,
                "sign_ew": bt_sign["portfolio"],
                "ls_top3": bt_ls["portfolio"],
                "per_symbol": bt_sign["per_symbol"],
            }
        )
        version_latest[col] = latest_signals(panel, col)
        bt_sign["daily"].to_csv(OUT_DIR / f"daily_{col}_sign.csv", index=False)

    # Ridge on top |t|>=1.5 features (IS only)
    usable = feat_tab[(feat_tab["t"].abs() >= 1.5) & feat_tab["ic"].notna()]
    ridge_feats = usable["feature"].head(12).tolist()
    if len(ridge_feats) < 4:
        ridge_feats = feat_tab.head(8)["feature"].tolist()
    pack, ridge_feats = fit_ridge(is_df, ridge_feats, l2=8.0)
    panel["ridge"] = apply_ridge(panel, pack, ridge_feats)
    is_df = panel[panel["date"] <= IS_END].copy()
    oos_df = panel[(panel["date"] >= OOS_START) & (panel["date"] <= OOS_END)].copy()
    ridge_is = eval_factor(is_df, "ridge")
    ridge_oos = eval_factor(oos_df, "ridge")
    ridge_bt = backtest_sign(panel, "ridge", OOS_START, OOS_END)
    ridge_ls = backtest_ls(panel, "ridge", OOS_START, OOS_END)
    ridge_bt["daily"].to_csv(OUT_DIR / "daily_ridge_sign.csv", index=False)

    model_rows = existing_model_oos()
    model_port = existing_portfolio(model_rows)

    # equal-weight existing-model daily portfolio for equity overlay
    model_eq = {}
    for model in ("ppo", "kronos", "xgb", "gaf"):
        frames = []
        for sym in SYMBOLS:
            path = RESULT_DIR / model / sym.lower() / "daily.csv"
            if not path.exists():
                continue
            d = pd.read_csv(path, parse_dates=["date"])
            d = d[(d["date"] >= OOS_START) & (d["date"] <= OOS_END)]
            if d.empty:
                continue
            frames.append(d[["date", "position", "log_ret_1d"]].assign(symbol=sym))
        if not frames:
            continue
        all_d = pd.concat(frames)
        all_d["pnl"] = all_d["position"] * all_d["log_ret_1d"].fillna(0.0)
        port = all_d.groupby("date")["pnl"].mean().sort_index()
        eq = (1.0 + (np.exp(port.cumsum()) - 1.0))  # start at 1 via exp
        eq = np.exp(port.cumsum())
        model_eq[model] = [{"date": str(i.date()), "equity": float(v)} for i, v in eq.items()]

    v4_daily = ridge_bt["daily"]
    v4_port = v4_daily.groupby("date")["pnl"].mean().sort_index()
    linear_eq = [{"date": str(i.date()), "equity": float(v)} for i, v in np.exp(v4_port.cumsum()).items()]
    v4sign = backtest_sign(panel, "v4", OOS_START, OOS_END)["daily"]
    v4_eq_s = v4sign.groupby("date")["pnl"].mean().sort_index()
    v4_eq = [{"date": str(i.date()), "equity": float(v)} for i, v in np.exp(v4_eq_s.cumsum()).items()]

    latest_existing = pd.read_csv(RESULT_DIR / "daily" / "ppo" / "latest_signals.csv")
    latest_cmp = []
    v4_map = {r["symbol"]: r for r in version_latest["v4"]}
    ridge_map = {r["symbol"]: r for r in latest_signals(panel, "ridge")}
    for sym in SYMBOLS:
        row = {"symbol": sym, "name_cn": SYMBOL_CN[sym]}
        for model in ("ppo", "kronos", "xgb", "gaf"):
            sub = latest_existing[(latest_existing["model"] == model) & (latest_existing["symbol"] == sym)]
            row[model] = sub["position_name"].iloc[0] if len(sub) else None
        row["linear_v4"] = v4_map.get(sym, {}).get("position_name")
        row["ridge"] = ridge_map.get(sym, {}).get("position_name")
        row["date"] = v4_map.get(sym, {}).get("date")
        row["code"] = v4_map.get(sym, {}).get("code")
        latest_cmp.append(row)

    coverage = {
        "symbols": SYMBOLS,
        "n_is": int((panel["date"] <= IS_END).sum()),
        "n_oos": int(((panel["date"] >= OOS_START) & (panel["date"] <= OOS_END)).sum()),
        "is_end": str(IS_END.date()),
        "oos_start": str(OOS_START.date()),
        "oos_end": str(OOS_END.date()),
        "dates_oos": int(oos_df["date"].nunique()),
        "panel_last": str(panel["date"].max().date()),
        "top_raw_feature": top_name,
        "top_raw_sign": top_sign,
        "ridge_features": ridge_feats,
    }

    payload = {
        "coverage": coverage,
        "feature_ic": [
            {
                "feature": r.feature,
                "ic": fmt(r.ic, 4),
                "ir": fmt(r.ir, 3),
                "t": fmt(r.t, 2),
                "ts_ic": fmt(r.ts_ic, 4),
                "turnover": fmt(r.turnover, 3),
                "sign_turnover": fmt(r.sign_turnover, 3),
            }
            for r in feat_tab.head(15).itertuples()
        ],
        "versions": [
            {
                "version": v["version"],
                "recipe": v["recipe"],
                "is_ic": fmt(v["is"]["ic"], 4),
                "is_ir": fmt(v["is"]["ir"], 3),
                "is_t": fmt(v["is"]["t"], 2),
                "is_turnover": fmt(v["is"]["turnover"], 3),
                "oos_ic": fmt(v["oos"]["ic"], 4),
                "oos_ir": fmt(v["oos"]["ir"], 3),
                "oos_t": fmt(v["oos"]["t"], 2),
                "oos_turnover": fmt(v["oos"]["turnover"], 3),
            }
            for v in version_stats
        ],
        "ridge": {
            "features": ridge_feats,
            "is": {k: fmt(ridge_is[k], 4 if "ic" in k else 3) for k in ridge_is},
            "oos": {k: fmt(ridge_oos[k], 4 if "ic" in k else 3) for k in ridge_oos},
            "sign_ew": {k: fmt(ridge_bt["portfolio"][k], 4) for k in ridge_bt["portfolio"]},
            "ls_top3": {k: fmt(ridge_ls["portfolio"][k], 4) for k in ridge_ls["portfolio"]},
            "per_symbol": [
                {"symbol": r["symbol"], "name_cn": SYMBOL_CN[r["symbol"]], **{k: fmt(r[k], 4) for k in ("return", "sharpe", "max_dd")}}
                for r in ridge_bt["per_symbol"]
            ],
        },
        "linear_backtest": [
            {
                "version": b["version"],
                "sign_ew": {k: fmt(b["sign_ew"][k], 4) for k in b["sign_ew"]},
                "ls_top3": {k: fmt(b["ls_top3"][k], 4) for k in b["ls_top3"]},
                "per_symbol": [
                    {"symbol": r["symbol"], "name_cn": SYMBOL_CN[r["symbol"]], **{k: fmt(r[k], 4) for k in ("return", "sharpe", "max_dd")}}
                    for r in b["per_symbol"]
                ],
            }
            for b in version_bt
        ],
        "existing_3m": [
            {
                "model": r["model"],
                "symbol": r["symbol"],
                "name_cn": r["name_cn"],
                "return": fmt(r["return"], 4),
                "sharpe": fmt(r["sharpe"], 3),
                "max_dd": fmt(r["max_dd"], 4),
                "last_position": r["last_position"],
            }
            for r in model_rows
        ],
        "existing_portfolio": model_port,
        "latest_compare": latest_cmp,
        "equity": {"ridge": linear_eq, "v4": v4_eq, **model_eq},
        "thresholds": {
            "ic_effective": 0.03,
            "ir_effective": 0.5,
            "t_effective": 2.0,
        },
    }

    out_json = OUT_DIR / "report.json"
    out_json.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2))
    print(json.dumps(clean({"coverage": coverage, "versions": payload["versions"], "ridge": payload["ridge"], "existing_portfolio": model_port}), ensure_ascii=False, indent=2))
    print(f"saved -> {out_json}")


if __name__ == "__main__":
    main()
