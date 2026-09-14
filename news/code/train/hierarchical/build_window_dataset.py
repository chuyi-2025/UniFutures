#!/usr/bin/env python3
"""Build L60 (symbol, trade_date) windows: doc idxs, ages, labels, splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    LABEL2ID,
    load_symbol_ohlc,
    normalize_ts,
    ret_to_label,
)
from config import (  # noqa: E402
    DATA_ROOT,
    LOOKBACK_DAYS,
    MAX_WINDOW_DOCS,
    OOS_END,
    OOS_START,
    PURGE_DAYS,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)


def split_name(trade_date: pd.Timestamp) -> str:
    t = normalize_ts(trade_date)
    if pd.Timestamp(TRAIN_START) <= t <= pd.Timestamp(TRAIN_END):
        return "train"
    # purge 1 session between train and val
    if pd.Timestamp(VAL_START) <= t <= pd.Timestamp(VAL_END):
        return "val"
    if pd.Timestamp(OOS_START) <= t <= pd.Timestamp(OOS_END):
        return "oos"
    return "other"


def build_windows(
    docs: pd.DataFrame,
    links: pd.DataFrame,
    lookback: int,
    max_docs: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    doc_dates = {
        int(r.doc_idx): normalize_ts(r.report_date)
        for r in docs[["doc_idx", "report_date"]].itertuples(index=False)
    }
    rows: list[dict] = []
    for sym in tqdm(sorted(links["symbol"].unique()), desc="windows"):
        px = load_symbol_ohlc(sym)
        if px.empty:
            continue
        px = px.copy()
        px["date"] = px["date"].map(normalize_ts)
        px = px[(px["date"] >= start) & (px["date"] <= end)].reset_index(drop=True)
        if px.empty:
            continue
        # newest first for truncation, then reverse to oldest→newest for model
        sub = links[links["symbol"] == sym].sort_values(
            ["report_date", "doc_idx"], ascending=[False, False]
        )
        if sub.empty:
            continue
        rdates = sub["report_date"].map(normalize_ts).to_numpy(dtype="datetime64[ns]")
        doc_idxs = sub["doc_idx"].to_numpy(dtype=np.int32)
        for _, prow in px.iterrows():
            T = normalize_ts(prow["date"])
            t0 = T - pd.Timedelta(days=lookback)
            mask = (rdates >= np.datetime64(t0)) & (rdates < np.datetime64(T))
            if not mask.any():
                continue
            chosen = doc_idxs[mask][:max_docs]  # newest-first truncate
            chosen = chosen[::-1].copy()  # oldest → newest
            ages = []
            for did in chosen:
                rd = doc_dates[int(did)]
                ages.append(int((T - rd).days))
            ret = float(prow["log_ret_1d"]) if pd.notna(prow["log_ret_1d"]) else float("nan")
            if not np.isfinite(ret):
                continue
            label = ret_to_label(ret, horizon=1)
            rows.append(
                {
                    "symbol": sym,
                    "trade_date": T,
                    "n_docs": int(len(chosen)),
                    "doc_idxs": chosen.tolist(),
                    "ages": ages,
                    "log_ret_1d": ret,
                    "label": label,
                    "label_id": LABEL2ID[label],
                    "split": split_name(T),
                    "lookback": lookback,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=DATA_ROOT / "documents.parquet")
    ap.add_argument("--links", type=Path, default=DATA_ROOT / "document_symbols.parquet")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "windows_L60.parquet")
    ap.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--max-docs", type=int, default=MAX_WINDOW_DOCS)
    ap.add_argument("--start", type=str, default=TRAIN_START)
    ap.add_argument("--end", type=str, default=OOS_END)
    args = ap.parse_args()

    docs = pd.read_parquet(args.docs, columns=["doc_idx", "doc_id", "report_date"])
    docs["report_date"] = pd.to_datetime(docs["report_date"]).map(normalize_ts)
    links = pd.read_parquet(args.links)
    links["report_date"] = pd.to_datetime(links["report_date"]).map(normalize_ts)
    links["symbol"] = links["symbol"].astype(str).str.upper()
    links = links[links["doc_idx"].isin(set(docs["doc_idx"]))].copy()

    # drop train/val boundary purge days from train only is handled by split dates;
    # additionally drop any val rows within PURGE_DAYS of TRAIN_END if needed
    _ = PURGE_DAYS

    windows = build_windows(
        docs,
        links,
        args.lookback,
        args.max_docs,
        pd.Timestamp(args.start),
        pd.Timestamp(args.end),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(args.out, index=False)
    windows.drop(columns=["doc_idxs", "ages"]).to_csv(
        args.out.with_suffix(".csv"), index=False
    )
    summary = {
        "n_windows": int(len(windows)),
        "lookback": args.lookback,
        "max_docs": args.max_docs,
        "splits": windows["split"].value_counts().to_dict() if len(windows) else {},
        "label_counts": windows["label"].value_counts().to_dict() if len(windows) else {},
        "symbols": int(windows["symbol"].nunique()) if len(windows) else 0,
        "mean_n_docs": float(windows["n_docs"].mean()) if len(windows) else 0.0,
        "p95_n_docs": float(windows["n_docs"].quantile(0.95)) if len(windows) else 0.0,
    }
    (args.out.parent / "windows_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
