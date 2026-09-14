#!/usr/bin/env python3
"""Rule search on cross-product spreads & calendar carry panels."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/home/workspace/lab/UniFutures/code/experiment")))

from common import DATA_DIR, PANEL_DIR, RESULT_DIR, ensure_dirs  # noqa: E402
from futures_lot_specs import lot_fee, lot_margin  # noqa: E402

CAPITAL = 1_000_000.0
START = pd.Timestamp("2018-01-01")


def sharpe(r: pd.Series) -> float | None:
    r = r.fillna(0.0)
    if len(r) < 40 or float(r.std()) == 0:
        return None
    return float(r.mean() / r.std() * np.sqrt(252))


def max_dd(r: pd.Series) -> float | None:
    r = r.fillna(0.0)
    if len(r) < 5:
        return None
    nav = (1.0 + r).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def hold_signal(raw: np.ndarray, hold: int) -> np.ndarray:
    sig = np.zeros(len(raw), dtype=float)
    left = 0
    last = 0.0
    for i, v in enumerate(raw):
        if v != 0:
            left = hold
            last = float(np.sign(v))
        if left > 0:
            sig[i] = last
            left -= 1
    return sig


def spread_margin(row: pd.Series) -> float:
    leg1, leg2 = str(row["leg1"]), str(row["leg2"])
    c1 = float(row.get("close1", row.get("near_close", np.nan)))
    c2 = float(row.get("close2", row.get("far_close", np.nan)))
    m1, _ = lot_margin(leg1, c1 if np.isfinite(c1) else None, broker=True)
    m2, _ = lot_margin(leg2, c2 if np.isfinite(c2) else None, broker=True)
    if m1 is None or m2 is None:
        return np.nan
    return float(m1) + float(m2)


def spread_fee(row: pd.Series, *, opening: bool) -> float:
    leg1, leg2 = str(row["leg1"]), str(row["leg2"])
    c1 = float(row.get("close1", row.get("near_close", np.nan)))
    c2 = float(row.get("close2", row.get("far_close", np.nan)))
    kind = "open" if opening else "close_prev"
    try:
        return float(lot_fee(leg1, c1, kind, 1)) + float(lot_fee(leg2, c2, kind, 1))
    except Exception:
        return 0.0


def run_scheme(
    df: pd.DataFrame,
    *,
    factor: str,
    mode: str,
    thr: float,
    direction: float,
    hold: int,
    mean_revert: bool,
) -> dict | None:
    d = df.dropna(subset=[factor]).copy()
    if len(d) < 80:
        return None
    x = d[factor].to_numpy(dtype=float)
    raw = np.zeros(len(x))
    if mode == "sign":
        if mean_revert:
            raw = np.where(np.abs(x) >= thr, -np.sign(x), 0.0)
        else:
            raw = np.where(np.abs(x) >= thr, np.sign(x) * direction, 0.0)
    elif mode == "long_only":
        raw = np.where(x * direction >= thr, direction, 0.0)
    else:
        return None
    sig = hold_signal(raw, hold) if hold > 1 else raw
    d["signal"] = sig

    pnl_col = "pnl_long_spread" if "pnl_long_spread" in d.columns else "pnl_long_carry"
    prev = 0.0
    pnls = []
    fees = []
    margins = []
    for _, r in d.iterrows():
        cur = float(r["signal"])
        gross = cur * float(r[pnl_col]) if cur != 0 and np.isfinite(r[pnl_col]) else 0.0
        fee = 0.0
        if cur != prev:
            if prev != 0:
                fee += spread_fee(r, opening=False)
            if cur != 0:
                fee += spread_fee(r, opening=True)
        pnls.append(gross - fee)
        fees.append(fee)
        margins.append(spread_margin(r) if cur != 0 else 0.0)
        prev = cur

    d["pnl"] = pnls
    d["ret"] = d["pnl"] / CAPITAL
    r = d["ret"]
    active = (d["signal"] != 0).sum()
    if active < 30:
        return None
    return {
        "factor": factor,
        "mode": mode,
        "thr": thr,
        "direction": direction,
        "hold": hold,
        "mean_revert": mean_revert,
        "days": len(d),
        "active_days": int(active),
        "active_pct": round(active / len(d), 3),
        "total_pnl": round(float(d["pnl"].sum()), 0),
        "return": round(float(r.sum()), 4),
        "sharpe": sharpe(r),
        "max_dd": max_dd(r),
        "worst_day": round(float(d["pnl"].min()), 0),
        "avg_margin": round(float(np.mean([m for m in margins if m > 0])), 0) if any(m > 0 for m in margins) else None,
        "total_fee": round(float(sum(fees)), 0),
    }


def search_panel(path: Path) -> list[dict]:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= START].sort_values("date")
    sid = str(df["spread_id"].iloc[0])
    label = str(df["label"].iloc[0]) if "label" in df.columns else sid
    kind = str(df["kind"].iloc[0]) if "kind" in df.columns else "cross"

    factors = [c for c in df.columns if c.startswith("spread_z") or c.startswith("carry_z")]
    if not factors:
        return []

    rows = []
    for fac, thr, hold, mr in product(
        factors,
        [0.75, 1.0, 1.25, 1.5, 2.0],
        [1, 3, 5, 10, 15],
        [True, False],
    ):
        for mode in ("sign",):
            rec = run_scheme(
                df, factor=fac, mode=mode, thr=thr, direction=1.0,
                hold=hold, mean_revert=mr,
            )
            if rec and rec.get("sharpe") is not None:
                rec.update({"spread_id": sid, "label": label, "kind": kind})
                rows.append(rec)
    return rows


def run_search(panel_dir: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (panel_dir / "all_spreads.parquet").exists():
        print(f"panels missing at {panel_dir} — run build_panels.py first", flush=True)
        sys.exit(1)

    all_rows: list[dict] = []
    for path in sorted(panel_dir.glob("*.parquet")):
        if path.name == "all_spreads.parquet":
            continue
        print(f"search {path.name}...", flush=True)
        all_rows.extend(search_panel(path))

    if not all_rows:
        print("no results", flush=True)
        sys.exit(1)

    res = pd.DataFrame(all_rows).sort_values(["sharpe", "return"], ascending=False)
    res.to_csv(out_dir / "spread_search_all.csv", index=False)

    best = res.sort_values("sharpe", ascending=False).groupby("spread_id", as_index=False).head(1)
    best.to_csv(out_dir / "spread_search_best.csv", index=False)

    ok = res[
        (res["sharpe"] >= 0.8)
        & (res["max_dd"].fillna(-1) > -0.15)
        & (res["active_days"] >= 40)
    ].sort_values("sharpe", ascending=False)
    ok.to_csv(out_dir / "spread_search_pass.csv", index=False)

    summary = {
        "panel_dir": str(panel_dir),
        "n_schemes": len(res),
        "n_spreads": int(res["spread_id"].nunique()),
        "n_pass": len(ok),
        "top10": ok.head(10).to_dict(orient="records"),
        "best_per_spread": best.to_dict(orient="records"),
    }
    (out_dir / "spread_search_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
    )
    return best


def compare_modes(naive_best: pd.DataFrame, spec_best: pd.DataFrame) -> pd.DataFrame:
    cols = ["spread_id", "label", "kind", "factor", "thr", "hold", "mean_revert",
            "sharpe", "return", "max_dd", "active_pct", "total_pnl"]
    n = naive_best[cols].rename(columns={
        c: f"naive_{c}" for c in cols if c not in ("spread_id", "label", "kind")
    })
    s = spec_best[cols].rename(columns={
        c: f"spec_{c}" for c in cols if c not in ("spread_id", "label", "kind")
    })
    m = n.merge(s, on=["spread_id", "label", "kind"], how="outer")
    m["sharpe_delta"] = m["spec_sharpe"] - m["naive_sharpe"]
    m["return_delta"] = m["spec_return"] - m["naive_return"]
    return m.sort_values("spec_sharpe", ascending=False, na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-dir", type=Path, default=PANEL_DIR)
    parser.add_argument("--compare", action="store_true", help="run naive+speculator and write comparison")
    args = parser.parse_args()
    ensure_dirs()

    if args.compare:
        naive_dir = DATA_DIR / "panels_naive"
        spec_dir = DATA_DIR / "panels_speculator"
        naive_out = RESULT_DIR / "search_naive"
        spec_out = RESULT_DIR / "search_speculator"
        print("=== NAIVE search ===", flush=True)
        naive_best = run_search(naive_dir, naive_out)
        print("\n=== SPECULATOR search ===", flush=True)
        spec_best = run_search(spec_dir, spec_out)
        cmp = compare_modes(naive_best, spec_best)
        cmp_path = RESULT_DIR / "search_speculator_vs_naive.csv"
        cmp.to_csv(cmp_path, index=False)
        print("\n=== Speculator vs Naive (best per spread) ===")
        show = ["spread_id", "label", "kind",
                "naive_sharpe", "spec_sharpe", "sharpe_delta",
                "naive_return", "spec_return", "return_delta"]
        print(cmp[show].to_string(index=False))
        print(f"\nWrote {cmp_path}", flush=True)
        return

    out = RESULT_DIR / "search"
    best = run_search(args.panel_dir, out)
    cols = ["spread_id", "label", "factor", "thr", "hold", "mean_revert", "sharpe", "return", "max_dd", "active_pct"]
    print("\n=== Best per spread ===")
    print(best[cols].to_string(index=False))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
