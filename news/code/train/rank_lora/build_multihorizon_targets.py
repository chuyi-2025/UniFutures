#!/usr/bin/env python3
"""Build multi-horizon cross-sectional ranking targets + L60 windows.

Labels follow price-side X07: keep raw forward returns, group by trade_date,
train with pairwise ranking. Docs enter on first session AFTER report_date.
Windows use report_date ∈ [T-60, T). Splits purge max-horizon sessions.
"""

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

from common import load_symbol_ohlc, normalize_ts  # noqa: E402
from config import (  # noqa: E402
    DATA_ROOT,
    HIER_DATA,
    HORIZONS,
    LOOKBACK_DAYS,
    MAX_WINDOW_DOCS,
    OOS_END,
    OOS_START,
    PURGE_SESSIONS,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)


def session_index(px: pd.DataFrame) -> dict[pd.Timestamp, int]:
    dates = [normalize_ts(d) for d in px["date"].tolist()]
    return {d: i for i, d in enumerate(dates)}


def forward_ret_from_entry(px: pd.DataFrame, entry: pd.Timestamp, horizon: int) -> float:
    """Cumulative log return over `horizon` sessions starting AT entry (inclusive)."""
    idx = session_index(px)
    e = normalize_ts(entry)
    if e not in idx:
        return float("nan")
    i0 = idx[e]
    i1 = i0 + horizon
    if i1 > len(px):
        return float("nan")
    rets = px["log_ret_1d"].iloc[i0:i1].astype(float)
    if rets.isna().any():
        return float("nan")
    return float(rets.sum())


def first_session_after(px: pd.DataFrame, report_date: pd.Timestamp) -> pd.Timestamp | None:
    rd = normalize_ts(report_date)
    future = px[px["date"] > rd]
    if future.empty:
        return None
    return normalize_ts(future["date"].iloc[0])


def assign_split(trade_date: pd.Timestamp, session_dates: list[pd.Timestamp]) -> str:
    """Assign train/val/oos/other with PURGE_SESSIONS trading-day embargo."""
    t = normalize_ts(trade_date)
    train_end = pd.Timestamp(TRAIN_END)
    val_start = pd.Timestamp(VAL_START)
    val_end = pd.Timestamp(VAL_END)
    oos_start = pd.Timestamp(OOS_START)

    # Exclude last PURGE_SESSIONS of train and first PURGE of val/oos by sessions
    sd = sorted(session_dates)
    train_sessions = [d for d in sd if pd.Timestamp(TRAIN_START) <= d <= train_end]
    val_sessions = [d for d in sd if val_start <= d <= val_end]
    oos_sessions = [d for d in sd if oos_start <= d <= pd.Timestamp(OOS_END)]

    if len(train_sessions) > PURGE_SESSIONS:
        train_ok_end = train_sessions[-(PURGE_SESSIONS + 1)]
    else:
        train_ok_end = train_sessions[0] if train_sessions else train_end

    if len(val_sessions) > PURGE_SESSIONS:
        val_ok_start = val_sessions[PURGE_SESSIONS]
        val_ok_end = val_sessions[-(PURGE_SESSIONS + 1)] if len(val_sessions) > 2 * PURGE_SESSIONS else val_sessions[-1]
    else:
        val_ok_start = val_start
        val_ok_end = val_end

    if len(oos_sessions) > PURGE_SESSIONS:
        oos_ok_start = oos_sessions[PURGE_SESSIONS]
    else:
        oos_ok_start = oos_start

    if pd.Timestamp(TRAIN_START) <= t <= train_ok_end:
        return "train"
    if val_ok_start <= t <= val_ok_end:
        return "val"
    if oos_ok_start <= t <= pd.Timestamp(OOS_END):
        return "oos"
    return "other"


def build_doc_symbol_targets(docs: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """One row per (doc, symbol) with entry trade_date and H1/H7/H14 returns."""
    symbols = sorted(links["symbol"].unique())
    px_cache: dict[str, pd.DataFrame] = {}
    for sym in tqdm(symbols, desc="load prices"):
        px = load_symbol_ohlc(sym)
        if px.empty:
            continue
        px = px.copy()
        px["date"] = px["date"].map(normalize_ts)
        px_cache[sym] = px

    doc_map = docs.set_index("doc_idx")
    rows: list[dict] = []
    for r in tqdm(links.itertuples(index=False), total=len(links), desc="doc targets"):
        sym = str(r.symbol).upper()
        px = px_cache.get(sym)
        if px is None or px.empty:
            continue
        if int(r.doc_idx) not in doc_map.index:
            continue
        doc = doc_map.loc[int(r.doc_idx)]
        report_date = normalize_ts(r.report_date)
        entry = first_session_after(px, report_date)
        if entry is None:
            continue
        item = {
            "doc_idx": int(r.doc_idx),
            "doc_id": str(doc["doc_id"]),
            "symbol": sym,
            "report_date": report_date,
            "trade_date": entry,
            "institution": doc.get("institution", ""),
            "text": doc["text"],
            "char_count": int(doc["char_count"]),
            "path": doc["path"],
            "content_hash": doc["content_hash"],
        }
        ok = True
        for h in HORIZONS:
            ret = forward_ret_from_entry(px, entry, h)
            item[f"ret_h{h}"] = ret
            item[f"valid_h{h}"] = bool(np.isfinite(ret))
            if h == max(HORIZONS) and not np.isfinite(ret):
                ok = False
        # Keep if at least H1 is valid
        if not item["valid_h1"]:
            continue
        rows.append(item)
    return pd.DataFrame(rows)


def add_cross_section_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Within each trade_date, rank symbols by return (higher = better)."""
    out = df.copy()
    for h in HORIZONS:
        col = f"ret_h{h}"
        rank_col = f"rank_h{h}"
        # Dense rank ascending on return so larger return -> larger rank for pairwise
        out[rank_col] = (
            out.groupby("trade_date", sort=False)[col]
            .rank(method="average", ascending=True)
            .astype(float)
        )
        # Invalidate rank when return missing
        out.loc[~out[f"valid_h{h}"], rank_col] = np.nan
    # group size for pairwise
    out["group_size"] = out.groupby("trade_date")["symbol"].transform("count")
    return out


def build_windows(
    links: pd.DataFrame,
    docs: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """(symbol, trade_date) windows with doc idxs + ages + multi-horizon labels."""
    # Daily target: one label per (symbol, trade_date) from price, not from a single doc
    symbols = sorted(links["symbol"].unique())
    doc_dates = {
        int(r.doc_idx): normalize_ts(r.report_date)
        for r in docs[["doc_idx", "report_date"]].itertuples(index=False)
    }

    # Precompute daily returns for each symbol/horizon for window labels
    px_cache: dict[str, pd.DataFrame] = {}
    all_sessions: list[pd.Timestamp] = []
    for sym in symbols:
        px = load_symbol_ohlc(sym)
        if px.empty:
            continue
        px = px.copy()
        px["date"] = px["date"].map(normalize_ts)
        px_cache[sym] = px
        all_sessions.extend(px["date"].tolist())
    all_sessions = sorted(set(all_sessions))

    # Index links
    link_by_sym: dict[str, pd.DataFrame] = {}
    for sym, g in links.groupby("symbol"):
        gg = g.sort_values(["report_date", "doc_idx"], ascending=[False, False]).copy()
        gg["report_date"] = gg["report_date"].map(normalize_ts)
        link_by_sym[str(sym).upper()] = gg

    rows: list[dict] = []
    start = pd.Timestamp(TRAIN_START)
    end = pd.Timestamp(OOS_END)
    for sym, px in tqdm(px_cache.items(), desc="windows"):
        sub = link_by_sym.get(sym)
        if sub is None or sub.empty:
            continue
        rdates = sub["report_date"].to_numpy(dtype="datetime64[ns]")
        doc_idxs = sub["doc_idx"].to_numpy(dtype=np.int32)
        pxw = px[(px["date"] >= start) & (px["date"] <= end)].reset_index(drop=True)
        for _, prow in pxw.iterrows():
            T = normalize_ts(prow["date"])
            t0 = T - pd.Timedelta(days=LOOKBACK_DAYS)
            mask = (rdates >= np.datetime64(t0)) & (rdates < np.datetime64(T))
            if not mask.any():
                continue
            chosen = doc_idxs[mask][:MAX_WINDOW_DOCS][::-1].copy()  # oldest→newest
            ages = [int((T - doc_dates[int(d)]).days) for d in chosen]
            item = {
                "symbol": sym,
                "trade_date": T,
                "n_docs": int(len(chosen)),
                "doc_idxs": chosen.tolist(),
                "ages": ages,
                "lookback": LOOKBACK_DAYS,
                "split": assign_split(T, all_sessions),
            }
            any_valid = False
            for h in HORIZONS:
                ret = forward_ret_from_entry(px, T, h)
                item[f"ret_h{h}"] = ret
                item[f"valid_h{h}"] = bool(np.isfinite(ret))
                any_valid = any_valid or item[f"valid_h{h}"]
            if not any_valid or item["split"] == "other":
                continue
            rows.append(item)

    win = pd.DataFrame(rows)
    if win.empty:
        return win
    # Cross-section ranks on window trade_dates
    for h in HORIZONS:
        col = f"ret_h{h}"
        win[f"rank_h{h}"] = (
            win.groupby("trade_date", sort=False)[col]
            .rank(method="average", ascending=True)
            .astype(float)
        )
        win.loc[~win[f"valid_h{h}"], f"rank_h{h}"] = np.nan
    win["group_size"] = win.groupby("trade_date")["symbol"].transform("count")
    return win


def audit(targets: pd.DataFrame, windows: pd.DataFrame) -> dict:
    summary = {
        "n_doc_symbol_rows": int(len(targets)),
        "n_docs": int(targets["doc_idx"].nunique()) if len(targets) else 0,
        "n_symbols_targets": int(targets["symbol"].nunique()) if len(targets) else 0,
        "n_windows": int(len(windows)),
        "window_splits": windows["split"].value_counts().to_dict() if len(windows) else {},
        "mean_n_docs": float(windows["n_docs"].mean()) if len(windows) else 0.0,
        "p95_n_docs": float(windows["n_docs"].quantile(0.95)) if len(windows) else 0.0,
        "horizons": list(HORIZONS),
        "purge_sessions": PURGE_SESSIONS,
        "lookback_days": LOOKBACK_DAYS,
    }
    for h in HORIZONS:
        for split in ("train", "val", "oos"):
            part = windows[(windows["split"] == split) & windows[f"valid_h{h}"]]
            if part.empty:
                continue
            # Rank IC of raw return vs itself = 1; report return std instead
            summary[f"{split}_h{h}_n"] = int(len(part))
            summary[f"{split}_h{h}_ret_std"] = float(part[f"ret_h{h}"].std())
            summary[f"{split}_h{h}_group_mean"] = float(part["group_size"].mean())
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=HIER_DATA / "documents.parquet")
    ap.add_argument("--links", type=Path, default=HIER_DATA / "document_symbols.parquet")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT)
    args = ap.parse_args()

    docs = pd.read_parquet(args.docs)
    docs["report_date"] = pd.to_datetime(docs["report_date"]).map(normalize_ts)
    links = pd.read_parquet(args.links)
    links["report_date"] = pd.to_datetime(links["report_date"]).map(normalize_ts)
    links["symbol"] = links["symbol"].astype(str).str.upper()
    links = links[links["doc_idx"].isin(set(docs["doc_idx"]))].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] doc-symbol multi-horizon targets")
    targets = build_doc_symbol_targets(docs, links)
    targets = add_cross_section_ranks(targets)
    # Encoder training uses report-entry trade_date split (same assign_split)
    all_sessions = sorted(
        {
            normalize_ts(d)
            for sym in targets["symbol"].unique()
            for d in load_symbol_ohlc(sym)["date"].tolist()
        }
    )
    targets["split"] = targets["trade_date"].map(lambda t: assign_split(t, all_sessions))
    targets.to_parquet(args.out_dir / "doc_targets.parquet", index=False)
    targets.drop(columns=["text"], errors="ignore").to_csv(
        args.out_dir / "doc_targets.csv", index=False
    )

    print("[2/3] L60 multi-horizon windows")
    windows = build_windows(links, docs, targets)
    windows.to_parquet(args.out_dir / "windows_L60.parquet", index=False)
    windows.drop(columns=["doc_idxs", "ages"]).to_csv(
        args.out_dir / "windows_L60.csv", index=False
    )

    print("[3/3] audit")
    summary = audit(targets, windows)
    # Encoder-side split counts
    summary["doc_target_splits"] = targets["split"].value_counts().to_dict()
    (args.out_dir / "targets_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
