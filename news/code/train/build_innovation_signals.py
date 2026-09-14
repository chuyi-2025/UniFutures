#!/usr/bin/env python3
"""Build a frozen 32-scheme innovation matrix from cached lookback signals.

All schemes use report windows ending before the trade date.  The script only
transforms cached probabilities and prices; it performs no model inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, FT_TEST_END, FT_TEST_START, load_symbol_ohlc  # noqa: E402


LOOKBACK_DIR = DATA_DIR / "lookback"
DEFAULT_OUT = DATA_DIR / "innovation"
KEYS = ["symbol", "trade_date"]
PROB_COLS = ["prob_0", "prob_1", "prob_2"]


def load_signal(scheme: str, lookback: int, prefix: str) -> pd.DataFrame:
    path = LOOKBACK_DIR / f"signals_lb_finance_zh_{scheme}_L{lookback}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    keep = KEYS + PROB_COLS + ["position", "n_docs"]
    missing = set(keep) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[keep].rename(
        columns={c: f"{prefix}_{c}" for c in keep if c not in KEYS}
    )


def attach_prices(panel: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol in sorted(panel["symbol"].unique()):
        px = load_symbol_ohlc(symbol)
        if px.empty:
            continue
        px = px[["date", "close", "log_ret_1d"]].copy()
        px["trade_date"] = pd.to_datetime(px.pop("date")).dt.normalize()
        px["symbol"] = symbol
        # A signal applied to row T earns log_ret_1d on T, so every technical
        # filter must be known at T-1.  Shift all price-derived features.
        px["signal_close"] = px["close"].shift(1)
        px["ret_vol20"] = px["log_ret_1d"].rolling(20, min_periods=10).std().shift(1)
        px["ann_vol20"] = px["ret_vol20"] * np.sqrt(252.0)
        px["ma20"] = px["close"].rolling(20, min_periods=10).mean().shift(1)
        px["ma60"] = px["close"].rolling(60, min_periods=20).mean().shift(1)
        px["vol_rank252"] = px["ret_vol20"].rolling(252, min_periods=40).rank(pct=True)
        parts.append(px)
    if not parts:
        raise RuntimeError("no price data available")
    return panel.merge(pd.concat(parts, ignore_index=True), on=KEYS, how="left")


def sign(values: pd.Series | np.ndarray) -> pd.Series:
    if isinstance(values, pd.Series):
        return np.sign(values).astype(float)
    return pd.Series(np.sign(values), dtype=float)


def consecutive_gate(direction: pd.Series, groups: pd.Series, days: int) -> pd.Series:
    same = direction.groupby(groups, sort=False).transform(
        lambda x: x.eq(x.shift(1)).rolling(days - 1, min_periods=days - 1).sum()
    )
    return direction.where((direction != 0) & (same >= days - 1), 0.0)


def cooldown_flips(direction: pd.Series, groups: pd.Series, cooldown: int = 2) -> pd.Series:
    out = pd.Series(0.0, index=direction.index)
    for _, idx in direction.groupby(groups, sort=False).groups.items():
        wait = 0
        previous = 0.0
        for i in idx:
            current = float(direction.loc[i])
            if current and previous and current != previous:
                wait = cooldown
            if wait:
                out.loc[i] = 0.0
                wait -= 1
            else:
                out.loc[i] = current
            if current:
                previous = current
    return out


def majority(votes: list[pd.Series], threshold: int | None = None) -> pd.Series:
    total = pd.concat(votes, axis=1).fillna(0.0).sum(axis=1)
    if threshold is None:
        return sign(total)
    return sign(total).where(total.abs() >= threshold, 0.0)


def normalized_entropy(frame: pd.DataFrame) -> pd.Series:
    p = frame.clip(1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(3.0)


def make_output(panel: pd.DataFrame, position: pd.Series, scheme: str) -> pd.DataFrame:
    pos = pd.to_numeric(position, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    return pd.DataFrame(
        {
            "symbol": panel["symbol"],
            "trade_date": panel["trade_date"],
            "position": pos,
            "n_docs": panel["c60_n_docs"].fillna(0).astype(int),
            "scheme": scheme,
        }
    )


def build_schemes(panel: pd.DataFrame) -> dict[str, pd.Series]:
    cedge = panel["c60_prob_0"] - panel["c60_prob_2"]
    eedge = panel["e60_prob_0"] - panel["e60_prob_2"]
    cdir, edir = sign(cedge), sign(eedge)
    maxp = panel[[f"c60_prob_{i}" for i in range(3)]].max(axis=1)
    entropy = normalized_entropy(panel[[f"c60_prob_{i}" for i in range(3)]])
    # This checkpoint is deliberately low-confidence (most max probabilities
    # are below 0.50), so gates are calibrated to its probability scale.
    conf_size = ((maxp - 1 / 3) / (0.50 - 1 / 3)).clip(0.0, 1.0)

    schemes: dict[str, pd.Series] = {
        "conf_maxp_045": cdir.where(maxp >= 0.45, 0.0),
        "conf_maxp_0475": cdir.where(maxp >= 0.475, 0.0),
        "conf_maxp_049": cdir.where(maxp >= 0.49, 0.0),
        "conf_entropy_087": cdir.where(entropy <= 0.87, 0.0),
        "conf_margin_003": cdir.where(cedge.abs() >= 0.03, 0.0),
        "conf_margin_006": cdir.where(cedge.abs() >= 0.06, 0.0),
        "soft_edge": cedge,
        "soft_edge_thr003": cdir.where(cedge.abs() >= 0.03, 0.0),
        "size_by_conf": cdir * conf_size,
        "vol_target_10": cdir * (0.10 / panel["ann_vol20"]).clip(upper=1.0).fillna(0.0),
        "persist_2": consecutive_gate(cdir, panel["symbol"], 2),
        "persist_3": consecutive_gate(cdir, panel["symbol"], 3),
        "ema_sent_5": sign(cedge.groupby(panel["symbol"]).transform(lambda x: x.ewm(span=5).mean())),
        "ema_sent_10": sign(cedge.groupby(panel["symbol"]).transform(lambda x: x.ewm(span=10).mean())),
        "flip_cooldown_2": cooldown_flips(cdir, panel["symbol"]),
        "min_docs_3": cdir.where(panel["c60_n_docs"] >= 3, 0.0),
        "min_docs_5": cdir.where(panel["c60_n_docs"] >= 5, 0.0),
        "min_docs_10": cdir.where(panel["c60_n_docs"] >= 10, 0.0),
        "docs_conf_size": cedge * (panel["c60_n_docs"] / 5.0).clip(upper=1.0),
        "conf_weighted_concat_exp": (
            cedge * cedge.abs() + eedge * eedge.abs()
        ) / (cedge.abs() + eedge.abs()).replace(0.0, np.nan),
        "blend_concat_exp": sign(0.5 * cedge + 0.5 * eedge),
        "vote_concat_exp_uni7": majority(
            [cdir, edir, sign(panel["u7_prob_0"] - panel["u7_prob_2"])]
        ),
        "and_concat_exp": cdir.where(cdir == edir, 0.0),
        "or_concat_exp": pd.Series(
            np.where(cedge.abs() >= eedge.abs(), cdir, edir), index=panel.index
        ),
        "trend_gate_20": cdir.where(
            ((cdir > 0) & (panel["signal_close"] > panel["ma20"]))
            | ((cdir < 0) & (panel["signal_close"] < panel["ma20"])),
            0.0,
        ),
        "trend_gate_60": cdir.where(
            ((cdir > 0) & (panel["signal_close"] > panel["ma60"]))
            | ((cdir < 0) & (panel["signal_close"] < panel["ma60"])),
            0.0,
        ),
        "vol_hi_skip": cdir.where(panel["vol_rank252"] <= 0.80, 0.0),
        "vol_lo_only": cdir.where(panel["vol_rank252"] <= 0.50, 0.0),
        "crossL_concat_majority": majority(
            [
                sign(panel[f"c{lookback}_prob_0"] - panel[f"c{lookback}_prob_2"])
                for lookback in (14, 30, 60)
            ]
        ),
        "cross_scheme_consensus": majority(
            [
                cdir,
                edir,
                sign(panel["c30_prob_0"] - panel["c30_prob_2"]),
                sign(panel["e30_prob_0"] - panel["e30_prob_2"]),
            ],
            threshold=2,
        ),
        "long_bias_08": cdir.where(cdir <= 0, cdir * 0.8),
        "short_bias_08": cdir.where(cdir >= 0, cdir * 0.8),
    }
    # Frozen references are emitted separately and are not counted among the 32.
    schemes["baseline_concat_L60"] = cdir
    schemes["baseline_wavg_exp_L60"] = edir
    if len(schemes) != 34:
        raise AssertionError(f"expected 32 schemes + 2 baselines, got {len(schemes)}")
    return schemes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--start", default=str(FT_TEST_START.date()))
    ap.add_argument("--end", default=str(FT_TEST_END.date()))
    args = ap.parse_args()

    sources = [
        load_signal("concat", 60, "c60"),
        load_signal("wavg_exp", 60, "e60"),
        load_signal("wavg_uniform", 7, "u7"),
        load_signal("concat", 14, "c14"),
        load_signal("concat", 30, "c30"),
        load_signal("wavg_exp", 30, "e30"),
    ]
    panel = sources[0]
    for source in sources[1:]:
        panel = panel.merge(source, on=KEYS, how="left")
    panel = panel.sort_values(KEYS).reset_index(drop=True)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    panel = panel[panel["trade_date"].between(start, end)].reset_index(drop=True)
    panel = attach_prices(panel)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    schemes = build_schemes(panel)
    manifest: list[dict] = []
    for name, position in schemes.items():
        output = make_output(panel, position, name)
        path = args.out_dir / f"{name}.csv"
        output.to_csv(path, index=False)
        manifest.append(
            {
                "scheme": name,
                "path": str(path),
                "is_baseline": name.startswith("baseline_"),
                "rows": len(output),
                "symbols": output["symbol"].nunique(),
                "start": output["trade_date"].min().date().isoformat(),
                "end": output["trade_date"].max().date().isoformat(),
                "active_ratio": float((output["position"] != 0).mean()),
            }
        )
        print(f"[write] {name}: rows={len(output)} active={manifest[-1]['active_ratio']:.1%}")

    pd.DataFrame(manifest).to_csv(args.out_dir / "manifest.csv", index=False)
    metadata = {
        "status": "retrospective_exploratory",
        "selection_warning": "2025-07 onward was previously inspected; it is not a pristine holdout.",
        "scheme_count": 32,
        "baseline_count": 2,
        "source": "cached frozen finance_zh lookback probabilities",
        "timing": "report_date in [T-L,T), no same-day reports",
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {args.out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
