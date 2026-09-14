#!/usr/bin/env python3
"""Build simple daily news factors per (symbol, trade_date).

No same-day leak: for trading day T use reports with report_date in [T-L, T).

Factors:
  news_edge  — exp-weighted mean(prob_pos - prob_neg)
  news_vote  — exp-weighted mean(position in {-1,0,+1})
  news_cnt   — log(1 + doc count in window)
  n_docs     — raw doc count
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, class3_to_pos, load_symbol_ohlc  # noqa: E402

DEFAULT_SIG = DATA_DIR / "signals_ft_full_ensemble.parquet"
PCOLS = ("finance_zh_p0", "finance_zh_p1", "finance_zh_p2")


def _weights(ages, lookback: int) -> np.ndarray:
    ages = np.asarray(ages, dtype=float)
    half = max(lookback / 2.0, 1.0)
    w = np.exp(-np.log(2.0) * ages / half)
    s = float(w.sum())
    return w / s if s > 0 else w


def build_factors(
    docs: pd.DataFrame,
    *,
    lookback: int = 7,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pcols: tuple[str, str, str] = PCOLS,
) -> pd.DataFrame:
    rows: list[dict] = []
    symbols = sorted(docs["symbol"].unique())
    for sym in symbols:
        px = load_symbol_ohlc(sym)
        if px.empty:
            continue
        dates = px[(px["date"] >= start) & (px["date"] <= end)]["date"].map(
            lambda x: pd.Timestamp(x).normalize()
        )
        sub = docs[docs["symbol"] == sym].sort_values("report_date")
        if sub.empty:
            continue
        rdates = sub["report_date"].to_numpy()
        probs = sub[list(pcols)].to_numpy(dtype=float)
        pos_arr = np.array([class3_to_pos(int(np.argmax(p))) for p in probs])
        edge_arr = probs[:, 0] - probs[:, 2]
        for T in dates:
            t0 = T - pd.Timedelta(days=lookback)
            mask = (rdates >= np.datetime64(t0)) & (rdates < np.datetime64(T))
            if not mask.any():
                continue
            ages = np.array([(T - pd.Timestamp(d)).days for d in rdates[mask]], dtype=float)
            ages = np.maximum(ages, 1.0)
            w = _weights(ages, lookback)
            edge = float((edge_arr[mask] * w).sum())
            vote = float((pos_arr[mask] * w).sum())
            n = int(mask.sum())
            rows.append({
                "date": T,
                "symbol": sym,
                "news_edge": edge,
                "news_vote": vote,
                "news_cnt": float(np.log1p(n)),
                "n_docs": n,
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=7)
    ap.add_argument("--start", default="2024-03-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--signal", type=Path, default=DEFAULT_SIG)
    args = ap.parse_args()

    if not args.signal.is_file():
        raise SystemExit(f"missing {args.signal}")
    docs = pd.read_parquet(args.signal)
    docs["report_date"] = pd.to_datetime(docs["report_date"]).dt.normalize()
    docs["symbol"] = docs["symbol"].astype(str).str.upper()
    for c in PCOLS:
        if c not in docs.columns:
            raise SystemExit(f"missing {c} in {args.signal}")

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    fac = build_factors(docs, lookback=args.lookback, start=start, end=end)
    out = DATA_DIR / f"news_factors_L{args.lookback}.parquet"
    fac.to_parquet(out, index=False)
    print(
        f"saved {out} rows={len(fac)} symbols={fac['symbol'].nunique()} "
        f"{fac['date'].min().date()} -> {fac['date'].max().date()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
