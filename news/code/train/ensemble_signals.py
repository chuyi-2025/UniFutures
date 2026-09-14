#!/usr/bin/env python3
"""Ensemble fine-tuned sentiment signals: majority vote and probability average."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from common import DATA_DIR, class3_to_pos  # noqa: E402


def load_sig(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def merge_on_keys(dfs: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    base = dfs[0][keys].drop_duplicates()
    for i, df in enumerate(dfs):
        cols = keys + ["position"] + [c for c in df.columns if c.startswith("prob_")]
        part = df[cols].copy()
        rename = {"position": f"position_{i}"}
        for c in list(part.columns):
            if c.startswith("prob_"):
                rename[c] = f"m{i}_{c}"
        part = part.rename(columns=rename)
        base = base.merge(part, on=keys, how="outer")
    return base


def build_vote(merged: pd.DataFrame, n_models: int) -> pd.DataFrame:
    pos_cols = [f"position_{i}" for i in range(n_models)]
    rows = []
    for _, r in merged.iterrows():
        votes = [int(r[c]) for c in pos_cols if pd.notna(r.get(c))]
        if not votes:
            pos = 0
        else:
            s = sum(votes)
            pos = 1 if s > 0 else (-1 if s < 0 else 0)
        rows.append(pos)
    out = merged[["path", "institution", "report_date", "variety_raw", "symbol", "trade_date", "log_ret_1d", "label"]].copy()
    # some keys may be missing if outer merge — rebuild from available
    keep = [c for c in ["path", "institution", "report_date", "variety_raw", "symbol", "trade_date", "log_ret_1d", "label"] if c in merged.columns]
    out = merged[keep].copy()
    out["position"] = rows
    out["pred_class"] = [class3_to_pos_inv(p) for p in rows]
    return out


def class3_to_pos_inv(pos: int) -> int:
    if pos > 0:
        return 0
    if pos < 0:
        return 2
    return 1


def build_prob_avg(merged: pd.DataFrame, n_models: int) -> pd.DataFrame:
    rows_pos = []
    rows_cls = []
    for _, r in merged.iterrows():
        acc = np.zeros(3, dtype=float)
        cnt = 0
        for i in range(n_models):
            probs = [r.get(f"m{i}_prob_{k}") for k in range(3)]
            if any(pd.isna(x) for x in probs):
                continue
            acc += np.asarray(probs, dtype=float)
            cnt += 1
        if cnt == 0:
            cls = 1
        else:
            cls = int(np.argmax(acc / cnt))
        rows_cls.append(cls)
        rows_pos.append(class3_to_pos(cls))
    keep = [c for c in ["path", "institution", "report_date", "variety_raw", "symbol", "trade_date", "log_ret_1d", "label"] if c in merged.columns]
    out = merged[keep].copy()
    out["position"] = rows_pos
    out["pred_class"] = rows_cls
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--signals",
        type=Path,
        nargs="+",
        default=[
            DATA_DIR / "signals_ft_finance_zh.parquet",
            DATA_DIR / "signals_ft_modernbert.parquet",
            DATA_DIR / "signals_ft_finbert2.parquet",
        ],
    )
    ap.add_argument("--out-vote", type=Path, default=DATA_DIR / "signals_ensemble_vote.parquet")
    ap.add_argument("--out-prob", type=Path, default=DATA_DIR / "signals_ensemble_prob.parquet")
    args = ap.parse_args()

    dfs = [load_sig(p) for p in args.signals]
    keys = ["path", "symbol", "trade_date"]
    # ensure unique doc-symbol-trade rows
    dfs = [d.drop_duplicates(keys) for d in dfs]
    merged = dfs[0][keys + ["institution", "report_date", "variety_raw", "log_ret_1d", "label", "position"] + [c for c in dfs[0].columns if c.startswith("prob_")]].copy()
    merged = merged.rename(columns={"position": "position_0", **{c: f"m0_{c}" for c in merged.columns if c.startswith("prob_")}})
    for i, d in enumerate(dfs[1:], start=1):
        cols = keys + ["position"] + [c for c in d.columns if c.startswith("prob_")]
        part = d[cols].rename(columns={"position": f"position_{i}", **{c: f"m{i}_{c}" for c in d.columns if c.startswith("prob_")}})
        merged = merged.merge(part, on=keys, how="inner")

    n_models = len(dfs)
    vote = build_vote(merged, n_models)
    prob = build_prob_avg(merged, n_models)
    args.out_vote.parent.mkdir(parents=True, exist_ok=True)
    vote.to_parquet(args.out_vote, index=False)
    prob.to_parquet(args.out_prob, index=False)
    vote.to_csv(args.out_vote.with_suffix(".csv"), index=False)
    prob.to_csv(args.out_prob.with_suffix(".csv"), index=False)
    print(f"[done] vote={len(vote)} -> {args.out_vote}")
    print(f"[done] prob={len(prob)} -> {args.out_prob}")


if __name__ == "__main__":
    main()
