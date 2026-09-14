#!/usr/bin/env python3
"""Mine rule-only news factors; rank by IC/IR. Optimized batch aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from build_news_factors import _weights  # noqa: E402
from common import DATA_DIR, RESULTS_ROOT, load_symbol_ohlc  # noqa: E402
from linear_ridge_walkforward import (  # noqa: E402
    MIN_CS,
    build_panel,
    ic_stats,
    ls_pnl,
    metrics_from_pnl,
    rank_ic_daily,
)
from rule_features import scan_rich_docs  # noqa: E402

BT_START = pd.Timestamp("2024-03-08")
OUT = RESULTS_ROOT / "rule_mine"

# Focused search space
SPECS: list[tuple[str, str, int, str]] = []
for kind in ("all", "no_dianping", "strategy", "daily", "weekly"):
    for col in ("fwd_lex", "core_lex", "sd_lex", "rule_lex", "rule_edge"):
        for lb in (3, 7, 14, 30):
            for wt in ("exp", "uniform"):
                SPECS.append((kind, col, lb, wt))


def filter_docs(docs: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "all":
        return docs
    if kind == "no_dianping":
        return docs[docs["kind"] != "dianping"]
    return docs[docs["kind"] == kind]


def keep_single_symbol_docs(docs: pd.DataFrame) -> pd.DataFrame:
    """Drop reports mapped to multiple symbols (早评/指数/席位等污染截面)."""
    if docs.empty or "path" not in docs.columns:
        return docs
    n_sym = docs.groupby("path")["symbol"].nunique()
    single = n_sym[n_sym == 1].index
    return docs[docs["path"].isin(single)].copy()


def batch_aggregate(
    docs: pd.DataFrame,
    col: str,
    lookback: int,
    dates: pd.DatetimeIndex,
    weight: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sym, sub in docs.groupby("symbol"):
        sub = sub.sort_values("report_date")
        rdates = sub["report_date"].to_numpy(dtype="datetime64[ns]")
        vals = sub[col].to_numpy(dtype=float)
        for T in dates:
            t0 = np.datetime64(T - pd.Timedelta(days=lookback))
            t1 = np.datetime64(T)
            mask = (rdates >= t0) & (rdates < t1)
            if not mask.any():
                continue
            ages = np.array([(T - pd.Timestamp(d)).days for d in rdates[mask]], dtype=float)
            ages = np.maximum(ages, 1.0)
            if weight == "exp":
                w = _weights(ages, lookback)
            else:
                w = np.ones_like(ages) / len(ages)
            rows.append({
                "date": T,
                "symbol": sym,
                "val": float((vals[mask] * w).sum()),
                "n_docs": int(mask.sum()),
                "std": float(np.std(vals[mask])) if mask.sum() > 1 else 0.0,
            })
    return pd.DataFrame(rows)


def eval_factor(panel: pd.DataFrame, fac: pd.DataFrame, name: str, min_docs: int = 0) -> dict:
    p = panel.merge(fac, on=["date", "symbol"], how="left")
    if min_docs > 0:
        p = p[p["n_docs"] >= min_docs]
    p = p.dropna(subset=["val", "fwd_ret"])
    if p["date"].nunique() < 30:
        return {"name": name, "ic": None, "ir": None, "sharpe": None, "trade_days": 0}
    p = p.rename(columns={"val": name})
    ic = ic_stats(rank_ic_daily(p, name))
    pnls = []
    for _, day in p.groupby("date"):
        if len(day) < MIN_CS:
            continue
        r = ls_pnl(day, day[name].to_numpy())
        if np.isfinite(r):
            pnls.append(r)
    m = metrics_from_pnl(pd.Series(pnls))
    return {
        "name": name,
        "ic": round(float(ic["ic"]), 4) if ic.get("ic") == ic.get("ic") else None,
        "ir": round(float(ic["ir"]), 3) if ic.get("ir") == ic.get("ir") else None,
        "ic_t": round(float(ic["t"]), 2) if ic.get("t") == ic.get("t") else None,
        "ic_days": int(ic.get("n") or 0),
        "sharpe": round(float(m["sharpe"]), 3),
        "return": round(float(m["return"]), 4),
        "max_dd": round(float(m["max_dd"]), 4),
        "trade_days": int(m["days"]),
    }


def main() -> None:
    rich_path = DATA_DIR / "rule_docs_rich.parquet"
    if not rich_path.is_file():
        print("scan rich docs...", flush=True)
        scan_rich_docs().to_parquet(rich_path, index=False)
    docs = pd.read_parquet(rich_path)
    docs["report_date"] = pd.to_datetime(docs["report_date"])
    print(f"docs={len(docs)}", flush=True)

    panel = build_panel(BT_START).dropna(subset=["fwd_ret"])
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    kind_cache = {k: filter_docs(docs, k) for k in {s[0] for s in SPECS}}

    fac_cache: dict[str, pd.DataFrame] = {}
    for i, (kind, col, lb, wt) in enumerate(SPECS):
        key = f"{col}|{kind}|L{lb}|{wt}"
        print(f"[{i+1}/{len(SPECS)}] {key}", flush=True)
        sub = kind_cache[kind]
        if sub.empty or col not in sub.columns:
            continue
        fac_cache[key] = batch_aggregate(sub, col, lb, dates, wt)

    rows: list[dict] = []
    for key, fac in fac_cache.items():
        rows.append(eval_factor(panel, fac, key, 0))
        rows.append(eval_factor(panel, fac, f"{key}|min3", 3))

    # derived
    for kind in ("no_dianping", "strategy"):
        for lb in (7, 14, 30):
            k1 = f"fwd_lex|{kind}|L{lb}|exp"
            if k1 not in fac_cache:
                continue
            fac = fac_cache[k1].copy()
            fac["val"] = fac["val"] * (1.0 - fac["std"].clip(0, 1))
            rows.append(eval_factor(panel, fac, f"consensus_fwd|{kind}|L{lb}|min2", 2))
        k1, k2 = f"fwd_lex|{kind}|L{14}|exp", f"sd_lex|{kind}|L{14}|exp"
        if k1 in fac_cache and k2 in fac_cache:
            m = fac_cache[k1].merge(fac_cache[k2], on=["date", "symbol"], suffixes=("_a", "_b"))
            m["val"] = m["val_a"] + 0.5 * m["val_b"]
            m["n_docs"] = m[["n_docs_a", "n_docs_b"]].max(axis=1)
            rows.append(eval_factor(panel, m, f"fwd+0.5sd|{kind}|L14|min2", 2))

    df = pd.DataFrame(rows).sort_values(["ir", "ic"], ascending=False, na_position="last")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "all_candidates.csv", index=False)
    top_ic = df.nlargest(20, "ic")
    top_ir = df.nlargest(20, "ir")
    top_ic.to_csv(OUT / "top_ic.csv", index=False)
    top_ir.to_csv(OUT / "top_ir.csv", index=False)

    best = df.dropna(subset=["ic", "ir"]).head(5)
    (OUT / "report.json").write_text(
        json.dumps({"top5_ir": best.to_dict(orient="records"), "top_ic": top_ic.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== TOP 10 IR ===", flush=True)
    print(top_ir.head(10)[["name", "ic", "ir", "ic_t", "sharpe", "trade_days"]].to_string(index=False), flush=True)
    print("\n=== TOP 10 IC ===", flush=True)
    print(top_ic.head(10)[["name", "ic", "ir", "ic_t", "sharpe", "trade_days"]].to_string(index=False), flush=True)

    def _save_factor(full_name: str, out_col: str, out_path: Path, min_docs: int = 0) -> None:
        base = full_name.split("|min")[0]
        if base not in fac_cache:
            return
        outf = fac_cache[base].copy()
        if min_docs > 0:
            outf = outf[outf["n_docs"] >= min_docs]
        outf = outf.rename(columns={"val": out_col})
        outf.to_parquet(out_path, index=False)
        print(f"saved {out_path} ({full_name}, n={len(outf)})", flush=True)

    # peak IC (sparse) + balanced IC/IR
    _save_factor("rule_edge|strategy|L3|exp|min3", "rule_best", DATA_DIR / "rule_factor_best.parquet", 3)
    _save_factor("rule_edge|strategy|L7|uniform|min3", "rule_balanced", DATA_DIR / "rule_factor_balanced.parquet", 3)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
