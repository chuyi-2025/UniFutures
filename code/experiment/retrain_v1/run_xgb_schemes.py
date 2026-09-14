#!/usr/bin/env python3
"""Train and backtest X-family retrain schemes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (  # noqa: E402
    GAF_FEATURES_DIR,
    LABEL_BINS,
    REMOVED_SYMBOLS,
    TRAIN_END,
    XGB_FEATURES_CSV,
    ret_to_label,
)
from schemes import EXP_ROOT, RESULT_ROOT, SEED, Scheme, by_id, schemes_by_family  # noqa: E402

OOS_START = pd.Timestamp("2025-07-01")
OOS_END = pd.Timestamp("2026-12-31")
INIT_CAP = 1_000_000.0
LABEL_BINS_ARR = np.asarray(LABEL_BINS, dtype=float)


def feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]


def enrich_returns(xgb_df: pd.DataFrame) -> pd.DataFrame:
    """Attach ret_1d/ret_3d/ret_10d from GAF forward log returns."""
    cache_path = EXP_ROOT / "_cache" / "gaf_fwd_returns.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        fwd = pd.read_parquet(cache_path)
    else:
        parts = []
        for path in sorted(GAF_FEATURES_DIR.glob("*.csv")):
            if path.stem in {"all", "class_counts_by_symbol"}:
                continue
            df = pd.read_csv(
                path,
                usecols=["date", "symbol", "code", "contract_file", "log_ret_1d", "ret_3d"],
                parse_dates=["date"],
            )
            df["symbol"] = df["symbol"].astype(str).str.upper()
            df = df.sort_values(["contract_file", "date"]).reset_index(drop=True)
            g = df.groupby("contract_file", sort=False)["log_ret_1d"]
            # forward return from close_t to close_{t+H} = sum of log_ret on t+1..t+H
            df["fwd_log_1"] = g.shift(-1)
            df["fwd_log_3"] = g.transform(lambda s: s.shift(-1) + s.shift(-2) + s.shift(-3))
            df["fwd_log_5"] = g.transform(
                lambda s: sum(s.shift(-i) for i in range(1, 6))
            )
            df["fwd_log_10"] = g.transform(
                lambda s: sum(s.shift(-i) for i in range(1, 11))
            )
            df["ret_1d"] = np.expm1(df["fwd_log_1"])
            df["ret_3d_fwd"] = np.expm1(df["fwd_log_3"])
            df["ret_5d_fwd"] = np.expm1(df["fwd_log_5"])
            df["ret_10d"] = np.expm1(df["fwd_log_10"])
            parts.append(
                df[
                    [
                        "date",
                        "symbol",
                        "code",
                        "ret_1d",
                        "ret_3d_fwd",
                        "ret_5d_fwd",
                        "ret_10d",
                        "ret_3d",
                    ]
                ]
            )
        fwd = pd.concat(parts, ignore_index=True)
        fwd.to_parquet(cache_path, index=False)
        print(f"[cache] wrote {cache_path} rows={len(fwd)}")

    out = xgb_df.merge(fwd, on=["date", "symbol", "code"], how="left", suffixes=("", "_g"))
    # prefer rebuilt forwards; fall back to stored ret_5d / ret_3d
    out["ret_1d"] = out["ret_1d"].fillna(np.nan)
    out["ret_3d_use"] = out["ret_3d_fwd"].fillna(out.get("ret_3d"))
    out["ret_5d_use"] = out["ret_5d_fwd"].fillna(out["ret_5d"])
    out["ret_10d_use"] = out["ret_10d"]
    return out


def pick_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    if horizon == 1:
        return df["ret_1d"]
    if horizon == 3:
        return df["ret_3d_use"]
    if horizon == 5:
        return df["ret_5d_use"]
    if horizon == 10:
        return df["ret_10d_use"]
    raise ValueError(horizon)


def make_labels(df: pd.DataFrame, scheme: Scheme) -> pd.DataFrame:
    p = scheme.params
    horizon = int(p["horizon"])
    mode = p["mode"]
    work = df.copy()
    if symbols := p.get("symbols"):
        work = work[work["symbol"].isin(symbols)]
    rets = pick_return(work, horizon)
    work = work.assign(_ret=rets).dropna(subset=["_ret"]).reset_index(drop=True)

    if mode == "8cls":
        work["y"] = work["_ret"].map(lambda r: int(np.digitize(r, LABEL_BINS_ARR)))
        work["num_class"] = 8
        work["objective"] = "multi:softprob"
    elif mode == "3cls":
        work["y"] = np.where(work["_ret"] > 0.001, 2, np.where(work["_ret"] < -0.001, 0, 1))
        work["num_class"] = 3
        work["objective"] = "multi:softprob"
    elif mode == "quantile8":
        train_mask = work["date"] <= TRAIN_END
        qs = work.loc[train_mask, "_ret"].quantile(np.linspace(0, 1, 9)).to_numpy()
        qs[0], qs[-1] = -np.inf, np.inf
        work["y"] = np.digitize(work["_ret"], qs[1:-1])
        work["num_class"] = 8
        work["objective"] = "multi:softprob"
        work.attrs["quantile_edges"] = qs.tolist()
    elif mode == "extreme":
        work = work[work["_ret"].abs() >= 0.02].reset_index(drop=True)
        work["y"] = np.where(work["_ret"] > 0, 2, 0)
        # keep middle class unused; map to {0,2} only — use 2-class
        work["y"] = (work["_ret"] > 0).astype(int)
        work["num_class"] = 2
        work["objective"] = "multi:softprob"
    elif mode == "rank":
        work["y"] = work["_ret"].astype(float)
        work["num_class"] = 0
        work["objective"] = "rank:pairwise"
    else:
        raise ValueError(mode)
    return work


def signal_from_pred(pred, mode: str, num_class: int) -> int:
    if mode == "rank":
        return 1 if pred > 0 else (-1 if pred < 0 else 0)
    cls = int(pred)
    if mode in {"3cls"}:
        return {0: -1, 1: 0, 2: 1}.get(cls, 0)
    if mode == "extreme":
        return 1 if cls == 1 else -1
    # 8-class
    if cls == 0:
        return -1
    if cls == num_class - 1:
        return 1
    return 0


def train_scheme(scheme: Scheme, enriched: pd.DataFrame) -> Path:
    out = EXP_ROOT / scheme.id
    out.mkdir(parents=True, exist_ok=True)
    labeled = make_labels(enriched, scheme)
    train = labeled[
        (labeled["date"] <= TRAIN_END) & (~labeled["symbol"].isin(REMOVED_SYMBOLS))
    ].reset_index(drop=True)
    if len(train) < 1000:
        raise SystemExit(f"{scheme.id}: too few train rows {len(train)}")

    fcols = feat_cols(train)
    codes, code_ids = np.unique(train["code"].astype(str), return_inverse=True)
    x = np.concatenate(
        [train[fcols].to_numpy(dtype=np.float32), code_ids.reshape(-1, 1).astype(np.float32)],
        axis=1,
    )
    y = train["y"].to_numpy()
    objective = train["objective"].iloc[0]
    num_class = int(train["num_class"].iloc[0])

    params = {"eta": 0.1, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "seed": SEED}
    if objective.startswith("multi"):
        params.update({"objective": objective, "num_class": num_class, "eval_metric": "mlogloss"})
        dtrain = xgb.DMatrix(x, label=y)
    else:
        # pairwise rank needs group sizes
        train = train.sort_values(["date", "symbol"]).reset_index(drop=True)
        codes, code_ids = np.unique(train["code"].astype(str), return_inverse=True)
        x = np.concatenate(
            [train[fcols].to_numpy(dtype=np.float32), code_ids.reshape(-1, 1).astype(np.float32)],
            axis=1,
        )
        y = train["y"].to_numpy()
        groups = train.groupby("date", sort=False).size().to_numpy()
        params.update({"objective": "rank:pairwise", "eval_metric": "ndcg"})
        dtrain = xgb.DMatrix(x, label=y)
        dtrain.set_group(groups)

    model = xgb.train(params, dtrain, num_boost_round=200)
    model_path = out / "model.json"
    model.save_model(str(model_path))
    meta = {
        "scheme": scheme.id,
        "title": scheme.title,
        "axis": scheme.axis,
        "params": scheme.params,
        "codes": codes.tolist(),
        "num_class": num_class,
        "objective": objective,
        "n_train": len(train),
        "feat_cols": fcols,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] {scheme.id} n={len(train)} -> {model_path}")
    return out


def sharpe(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or arr.std() <= 1e-12:
        return 0.0
    return float(arr.mean() / arr.std() * math.sqrt(252))


def backtest_scheme(scheme: Scheme, enriched: pd.DataFrame) -> pd.DataFrame:
    out = EXP_ROOT / scheme.id
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    model = xgb.Booster()
    model.load_model(str(out / "model.json"))
    code_to_id = {c: i for i, c in enumerate(meta["codes"])}
    fcols = meta["feat_cols"]
    mode = scheme.params["mode"]
    num_class = int(meta["num_class"])
    hold = max(int(scheme.params["horizon"]), 1)

    oos = enriched[
        (enriched["date"] >= OOS_START)
        & (enriched["date"] <= OOS_END)
        & (~enriched["symbol"].isin(REMOVED_SYMBOLS))
    ].copy()
    if symbols := scheme.params.get("symbols"):
        oos = oos[oos["symbol"].isin(symbols)]
    oos = oos.dropna(subset=fcols).sort_values(["symbol", "date"]).reset_index(drop=True)

    summaries = []
    daily_parts = []
    for symbol, part in oos.groupby("symbol", sort=True):
        part = part.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
        if part.empty:
            continue
        code_ids = part["code"].astype(str).map(lambda c: code_to_id.get(c, -1)).to_numpy(dtype=np.float32)
        x = np.concatenate(
            [part[fcols].to_numpy(dtype=np.float32), code_ids.reshape(-1, 1)], axis=1
        )
        dmat = xgb.DMatrix(x)
        if mode == "rank":
            preds = model.predict(dmat)
            signals = np.array([signal_from_pred(p, mode, num_class) for p in preds], dtype=float)
        else:
            probs = model.predict(dmat)
            if probs.ndim == 1:
                cls = (probs >= 0.5).astype(int)
            else:
                cls = probs.argmax(axis=1)
            signals = np.array([signal_from_pred(c, mode, num_class) for c in cls], dtype=float)

        # hold expansion: mean then sign
        positions = np.zeros(len(part), dtype=float)
        buckets = [[] for _ in range(len(part))]
        for i, sig in enumerate(signals):
            if sig == 0:
                continue
            for j in range(i, min(i + hold, len(part))):
                buckets[j].append(sig)
        for i, vals in enumerate(buckets):
            if vals:
                m = float(np.mean(vals))
                positions[i] = 1.0 if m > 0 else (-1.0 if m < 0 else 0.0)

        # need next-day log return from gaf-ish: use ret_1d as simple? Better use log from close changes.
        # Approximate: strategy earns position_t * log_ret realized on t (as elsewhere).
        # Enrichment ret_1d is FORWARD; for PnL we need same-day log. Merge from original ret path:
        # Use np.log1p of concurrent move: from GAF file log_ret_1d on same date.
        # Reload quickly from part if we stored — we didn't. Use 0 if missing.
        # Attach via enriched columns: use pick of log from forward inverse is wrong.
        # Fall back: read from gaf cache file columns — add log_ret_1d to enrichment next time.
        # Temporary: compute from ret_5d_use/5 as rough — NO.
        # Load symbol gaf log_ret_1d map:
        gpath = GAF_FEATURES_DIR / f"{symbol}.csv"
        g = pd.read_csv(gpath, usecols=["date", "code", "log_ret_1d"], parse_dates=["date"])
        g = g.drop_duplicates(["date", "code"], keep="last")
        part = part.merge(g, on=["date", "code"], how="left")
        log_r = part["log_ret_1d"].fillna(0.0).to_numpy(dtype=float)
        strat = positions * log_r
        turnover = np.abs(np.diff(positions, prepend=0.0))
        for bps in (0, 2):
            net = strat - turnover * bps / 10000.0
            # store only 0bp in daily; summary both
            if bps == 0:
                daily = pd.DataFrame(
                    {
                        "date": part["date"],
                        "symbol": symbol,
                        "position": positions,
                        "strategy_ret": net,
                        "turnover": turnover,
                    }
                )
                daily_parts.append(daily)
            capital = INIT_CAP * np.exp(np.cumsum(net))
            peak = np.maximum.accumulate(capital)
            summaries.append(
                {
                    "scheme": scheme.id,
                    "symbol": symbol,
                    "bps": bps,
                    "return": float(capital[-1] / INIT_CAP - 1.0),
                    "sharpe": sharpe(net),
                    "max_dd": float(np.min(capital / peak - 1.0)),
                    "active_ratio": float((positions != 0).mean()),
                    "days": len(part),
                }
            )

    summary = pd.DataFrame(summaries)
    res_dir = RESULT_ROOT / scheme.id
    res_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(res_dir / "summary.csv", index=False)
    if daily_parts:
        pd.concat(daily_parts, ignore_index=True).to_csv(res_dir / "daily.csv", index=False)
    zero = summary[summary["bps"] == 0]
    print(
        f"[bt] {scheme.id}: n={len(zero)} medS={zero['sharpe'].median():.2f} "
        f"avgR={zero['return'].mean():.2%}"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", nargs="*", default=None)
    ap.add_argument("--family", default="X")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-bt", action="store_true")
    args = ap.parse_args()

    if args.scheme:
        schemes = [by_id(s) for s in args.scheme]
    else:
        schemes = schemes_by_family(args.family)

    print("[load] xgb features…")
    xgb_df = pd.read_csv(XGB_FEATURES_CSV, parse_dates=["date", "exit_date"])
    xgb_df["symbol"] = xgb_df["symbol"].astype(str).str.upper()
    enriched = enrich_returns(xgb_df)

    for scheme in schemes:
        if not args.skip_train:
            train_scheme(scheme, enriched)
        if not args.skip_bt:
            backtest_scheme(scheme, enriched)


if __name__ == "__main__":
    main()
