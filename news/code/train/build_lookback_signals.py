#!/usr/bin/env python3
"""Build lookback sentiment signals from frozen ft finance_zh doc scores.

Timing (no same-day leak):
  For trading day T, use reports with report_date in [T - lookback, T).
  Signal is applied on T (backtest hold_days = predict horizon).

Schemes:
  wavg_uniform / wavg_linear / wavg_exp  — pool doc probs in the window
  daily_wavg_linear — first mean probs per report_date, then linear lookback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_DIR,
    FT_TEST_END,
    FT_TEST_START,
    class3_to_pos,
    load_symbol_ohlc,
)

DOC_SIG = DATA_DIR / "signals_ft_full_ensemble.parquet"
POS_COL = "pos_finance_zh"
PCOLS = ("finance_zh_p0", "finance_zh_p1", "finance_zh_p2")


def _normalize(df: pd.DataFrame, pcols: tuple[str, str, str]) -> pd.DataFrame:
    out = df.copy()
    out["report_date"] = pd.to_datetime(out["report_date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    for c in pcols:
        if c not in out.columns:
            raise SystemExit(f"missing {c} in {DOC_SIG}; run build_sentiment_daily.py")
    return out


def _weights(ages: np.ndarray, lookback: int, scheme: str) -> np.ndarray:
    ages = np.asarray(ages, dtype=float)
    if scheme == "wavg_uniform" or scheme.endswith("uniform"):
        w = np.ones_like(ages)
    elif scheme == "wavg_linear" or scheme.endswith("linear"):
        w = (lookback + 1.0 - ages).clip(min=0.0)
    elif scheme == "wavg_exp" or scheme.endswith("exp"):
        half = max(lookback / 2.0, 1.0)
        w = np.exp(-np.log(2.0) * ages / half)
    else:
        raise ValueError(scheme)
    s = float(w.sum())
    return w / s if s > 0 else w


def _pos_from_probs(p: np.ndarray) -> int:
    return int(class3_to_pos(int(np.argmax(p))))


def build_wavg_doc(
    docs: pd.DataFrame,
    symbols: list[str],
    lookback: int,
    scheme: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pcols: tuple[str, str, str] = PCOLS,
) -> pd.DataFrame:
    """Doc-level weighted average over lookback calendar window."""
    rows: list[dict] = []
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
        for T in dates:
            t0 = T - pd.Timedelta(days=lookback)
            mask = (rdates >= np.datetime64(t0)) & (rdates < np.datetime64(T))
            if not mask.any():
                continue
            ages = ((T - pd.to_datetime(rdates[mask])).days).astype(float)
            # age 0 would be same-day; window excludes T so ages >= 1 if report on T-1
            ages = np.maximum(ages, 1.0)
            w = _weights(ages, lookback, scheme)
            p = (probs[mask] * w[:, None]).sum(axis=0)
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": T,
                    "report_date": T - pd.Timedelta(days=1),
                    "position": _pos_from_probs(p),
                    "pred_class": int(np.argmax(p)),
                    "prob_0": float(p[0]),
                    "prob_1": float(p[1]),
                    "prob_2": float(p[2]),
                    "n_docs": int(mask.sum()),
                    "lookback": lookback,
                    "scheme": scheme,
                }
            )
    return pd.DataFrame(rows)


def build_daily_wavg(
    docs: pd.DataFrame,
    symbols: list[str],
    lookback: int,
    scheme: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pcols: tuple[str, str, str] = PCOLS,
) -> pd.DataFrame:
    """Mean probs per report_date, then weighted lookback across days."""
    daily_rows = []
    for (sym, rd), grp in docs.groupby(["symbol", "report_date"], sort=True):
        p = grp[list(pcols)].mean().to_numpy(dtype=float)
        daily_rows.append(
            {
                "symbol": sym,
                "report_date": pd.Timestamp(rd).normalize(),
                "prob_0": float(p[0]),
                "prob_1": float(p[1]),
                "prob_2": float(p[2]),
                "n_docs": len(grp),
            }
        )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return daily

    rows: list[dict] = []
    for sym in symbols:
        px = load_symbol_ohlc(sym)
        if px.empty:
            continue
        dates = px[(px["date"] >= start) & (px["date"] <= end)]["date"].map(
            lambda x: pd.Timestamp(x).normalize()
        )
        sub = daily[daily["symbol"] == sym].sort_values("report_date")
        if sub.empty:
            continue
        rdates = sub["report_date"].to_numpy()
        probs = sub[["prob_0", "prob_1", "prob_2"]].to_numpy(dtype=float)
        ndocs = sub["n_docs"].to_numpy()
        for T in dates:
            t0 = T - pd.Timedelta(days=lookback)
            mask = (rdates >= np.datetime64(t0)) & (rdates < np.datetime64(T))
            if not mask.any():
                continue
            ages = np.maximum(((T - pd.to_datetime(rdates[mask])).days).astype(float), 1.0)
            w = _weights(ages, lookback, scheme)
            p = (probs[mask] * w[:, None]).sum(axis=0)
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": T,
                    "report_date": T - pd.Timedelta(days=1),
                    "position": _pos_from_probs(p),
                    "pred_class": int(np.argmax(p)),
                    "prob_0": float(p[0]),
                    "prob_1": float(p[1]),
                    "prob_2": float(p[2]),
                    "n_docs": int(ndocs[mask].sum()),
                    "n_days": int(mask.sum()),
                    "lookback": lookback,
                    "scheme": scheme,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=DOC_SIG)
    ap.add_argument("--lookback", type=int, nargs="+", default=[3, 7, 14])
    ap.add_argument(
        "--scheme",
        choices=(
            "wavg_uniform",
            "wavg_linear",
            "wavg_exp",
            "daily_wavg_linear",
            "daily_wavg_exp",
            "daily_wavg_uniform",
        ),
        nargs="+",
        default=["wavg_linear", "wavg_exp", "wavg_uniform", "daily_wavg_linear"],
    )
    ap.add_argument("--start", type=str, default=str(FT_TEST_START.date()))
    ap.add_argument("--end", type=str, default=str(FT_TEST_END.date()))
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR / "lookback")
    ap.add_argument(
        "--prob-cols",
        nargs=3,
        default=list(PCOLS),
        metavar=("P0", "P1", "P2"),
        help="three probability columns in --docs",
    )
    ap.add_argument(
        "--tag",
        default="finance_zh",
        help="output filename tag: signals_lb_<tag>_<scheme>_L*.parquet",
    )
    args = ap.parse_args()

    pcols = tuple(args.prob_cols)
    docs = _normalize(pd.read_parquet(args.docs), pcols)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    symbols = sorted(docs["symbol"].unique())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for scheme in args.scheme:
        for L in args.lookback:
            print(f"[build] scheme={scheme} L={L}")
            if scheme.startswith("daily_"):
                sig = build_daily_wavg(
                    docs, symbols, L, scheme, start, end, pcols=pcols
                )
            else:
                sig = build_wavg_doc(
                    docs, symbols, L, scheme, start, end, pcols=pcols
                )
            out = args.out_dir / f"signals_lb_{args.tag}_{scheme}_L{L}.parquet"
            sig.to_parquet(out, index=False)
            sig.to_csv(out.with_suffix(".csv"), index=False)
            n_pos = int((sig["position"] != 0).sum()) if len(sig) else 0
            print(f"  -> {out} rows={len(sig)} nonzero={n_pos}")


if __name__ == "__main__":
    main()
