#!/usr/bin/env python3
"""Build lagged lookback sentiment daily features (finance_zh only) for PPO.

For each calendar trading day T and lookback L:
  - collect report days in [T-L, T) (no same-day)
  - emit rolling pooled probs (linear wavg over days) as sent_* (4-dim)
  - also emit stacked last-L day features as sent_d{k}_* for k=0..L-1
    (k=0 = most recent available day before T; missing -> uniform/flat)

Writes:
  data/features/sentiment_lb_finance_zh_L{L}.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "news/code/train"))
from common import DATA_DIR as NEWS_SENT_DIR  # noqa: E402
from common import class3_to_pos, load_symbol_ohlc  # noqa: E402

ROOT = Path("/home/workspace/lab/UniFutures")
DOC_SIG = NEWS_SENT_DIR / "signals_ft_full_ensemble.parquet"
PCOLS = ("finance_zh_p0", "finance_zh_p1", "finance_zh_p2")
OUT_DIR = ROOT / "data/features"


def docs_to_report_daily(docs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sym, rd), grp in docs.groupby(["symbol", "report_date"], sort=True):
        p = grp[list(PCOLS)].mean().to_numpy(dtype=float)
        rows.append(
            {
                "symbol": str(sym).upper(),
                "report_date": pd.Timestamp(rd).normalize(),
                "p0": float(p[0]),
                "p1": float(p[1]),
                "p2": float(p[2]),
                "pos": float(class3_to_pos(int(np.argmax(p)))),
                "n_docs": len(grp),
            }
        )
    return pd.DataFrame(rows)


def linear_w(ages: np.ndarray, lookback: int) -> np.ndarray:
    ages = np.asarray(ages, dtype=float)
    w = (lookback + 1.0 - ages).clip(min=0.0)
    s = float(w.sum())
    return w / s if s > 0 else w


def build_for_symbol(daily: pd.DataFrame, sym: str, lookback: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    px = load_symbol_ohlc(sym)
    if px.empty:
        return pd.DataFrame()
    # need history before start for lookback
    hist_start = start - pd.Timedelta(days=lookback + 5)
    dates = px[(px["date"] >= hist_start) & (px["date"] <= end)]["date"].map(
        lambda x: pd.Timestamp(x).normalize()
    )
    sub = daily[daily["symbol"] == sym].sort_values("report_date")
    if sub.empty:
        return pd.DataFrame()
    rdates = sub["report_date"].to_numpy()
    mat = sub[["p0", "p1", "p2", "pos"]].to_numpy(dtype=float)

    rows = []
    for T in dates:
        if T < start or T > end:
            continue
        t0 = T - pd.Timedelta(days=lookback)
        mask = (rdates >= np.datetime64(t0)) & (rdates < np.datetime64(T))
        idxs = np.where(mask)[0]
        if len(idxs):
            ages = np.maximum(
                (T - pd.to_datetime(rdates[idxs])).days.to_numpy(dtype=float),
                1.0,
            )
            w = linear_w(ages, lookback)
            p = (mat[idxs, :3] * w[:, None]).sum(axis=0)
            pos = float(class3_to_pos(int(np.argmax(p))))
            n_docs = int(sub.iloc[idxs]["n_docs"].sum()) if "n_docs" in sub.columns else int(len(idxs))
            order = np.argsort(-rdates[idxs].astype("datetime64[ns]").astype(np.int64))
            ordered = idxs[order]
        else:
            p = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
            pos = 0.0
            n_docs = 0
            ordered = np.array([], dtype=int)

        feat = {
            "symbol": sym,
            "date": T,
            "sent_pos": pos,
            "sent_p0": float(p[0]),
            "sent_p1": float(p[1]),
            "sent_p2": float(p[2]),
            "n_docs": n_docs,
        }
        for k in range(lookback):
            if k < len(ordered):
                i = int(ordered[k])
                feat[f"sent_d{k}_pos"] = float(mat[i, 3])
                feat[f"sent_d{k}_p0"] = float(mat[i, 0])
                feat[f"sent_d{k}_p1"] = float(mat[i, 1])
                feat[f"sent_d{k}_p2"] = float(mat[i, 2])
            else:
                feat[f"sent_d{k}_pos"] = 0.0
                feat[f"sent_d{k}_p0"] = 1 / 3
                feat[f"sent_d{k}_p1"] = 1 / 3
                feat[f"sent_d{k}_p2"] = 1 / 3
        rows.append(feat)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=DOC_SIG)
    ap.add_argument("--lookback", type=int, nargs="+", default=[3, 7, 14])
    ap.add_argument("--start", type=str, default="2024-01-01")
    ap.add_argument("--end", type=str, default="2026-12-31")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    docs = pd.read_parquet(args.docs)
    docs["report_date"] = pd.to_datetime(docs["report_date"]).dt.normalize()
    docs["symbol"] = docs["symbol"].astype(str).str.upper()
    daily = docs_to_report_daily(docs)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    symbols = sorted(daily["symbol"].unique())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for L in args.lookback:
        print(f"[lb-daily] L={L}")
        parts = []
        for sym in symbols:
            part = build_for_symbol(daily, sym, L, start, end)
            if not part.empty:
                parts.append(part)
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        path = args.out_dir / f"sentiment_lb_finance_zh_L{L}.parquet"
        out.to_parquet(path, index=False)
        out.to_csv(path.with_suffix(".csv"), index=False)
        print(f"  -> {path} rows={len(out)} syms={out['symbol'].nunique() if len(out) else 0}")


if __name__ == "__main__":
    main()
