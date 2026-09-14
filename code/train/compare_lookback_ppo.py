#!/usr/bin/env python3
"""PPO + lookback finance_zh features (pool 4-dim or stacked L×4).

Modes:
  pool  — obs = experts(15) + sent_pos/p0/p1/p2 from lookback linear pool
  stack — obs = experts(15) + sent_d{k}_* for k=0..L-1

Train 2024-01-01~2025-06-30, OOS 2025-07-01~2026-12-31.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtest"))
from shared import MODELS_DIR, RESULTS_DIR, list_symbols  # noqa: E402
from backtest_ppo import backtest_df, save  # noqa: E402
from train_ppo import (  # noqa: E402
    SENT_OBS_NAMES,
    train_ppo,
)

FEAT_ROOT = Path("/home/workspace/lab/UniFutures/data/features")
BASE_FEAT = RESULTS_DIR / "ppo_sentiment_cmp" / "features_base"
CMP_ROOT = RESULTS_DIR / "ppo_lookback_cmp"

TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2025-06-30")
TEST_START = pd.Timestamp("2025-07-01")
TEST_END = pd.Timestamp("2026-12-31")


def stack_sent_names(L: int) -> list[str]:
    names = []
    for k in range(L):
        names += [f"sent_d{k}_pos", f"sent_d{k}_p0", f"sent_d{k}_p1", f"sent_d{k}_p2"]
    return names


def attach_lb(
    base: pd.DataFrame, sent: pd.DataFrame, sym: str, mode: str, L: int
) -> tuple[pd.DataFrame, list[str]]:
    out = base.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    part = sent[sent["symbol"].astype(str).str.upper() == sym.upper()].copy()
    part["date"] = pd.to_datetime(part["date"]).dt.normalize()
    if mode == "pool":
        cols = ["date", "sent_pos", "sent_p0", "sent_p1", "sent_p2"]
        names = list(SENT_OBS_NAMES)
    else:
        names = stack_sent_names(L)
        cols = ["date"] + names
    part = part[cols]
    # drop existing sent cols if any
    drop = [c for c in out.columns if c.startswith("sent_")]
    out = out.drop(columns=drop, errors="ignore")
    out = out.merge(part, on="date", how="left")
    if mode == "pool":
        out["sent_pos"] = out["sent_pos"].fillna(0.0)
        for c, v in (("sent_p0", 1 / 3), ("sent_p1", 1 / 3), ("sent_p2", 1 / 3)):
            out[c] = out[c].fillna(v)
    else:
        for k in range(L):
            out[f"sent_d{k}_pos"] = out[f"sent_d{k}_pos"].fillna(0.0)
            for c, v in (
                (f"sent_d{k}_p0", 1 / 3),
                (f"sent_d{k}_p1", 1 / 3),
                (f"sent_d{k}_p2", 1 / 3),
            ):
                out[c] = out[c].fillna(v)
    return out, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, nargs="+", default=[3, 7, 14])
    ap.add_argument("--mode", choices=("pool", "stack"), nargs="+", default=["pool", "stack"])
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--timesteps", type=int, default=80_000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    symbols = [s.upper() for s in args.symbols] if args.symbols else list_symbols()
    # only symbols with base features + original ppo
    symbols = [
        s
        for s in symbols
        if (BASE_FEAT / f"{s.lower()}.csv").exists() and (MODELS_DIR / f"ppo_{s.lower()}.zip").exists()
    ]
    if args.limit > 0:
        symbols = symbols[: args.limit]
    print(f"[ppo-lb] symbols={len(symbols)} modes={args.mode} L={args.lookback}")

    cmp_rows = []
    for L in args.lookback:
        sent_path = FEAT_ROOT / f"sentiment_lb_finance_zh_L{L}.parquet"
        if not sent_path.exists():
            print(f"[skip] missing {sent_path}")
            continue
        sent = pd.read_parquet(sent_path)
        for mode in args.mode:
            tag = f"{mode}_L{L}"
            feat_dir = CMP_ROOT / "features" / tag
            feat_dir.mkdir(parents=True, exist_ok=True)
            for i, sym in enumerate(symbols, 1):
                print(f"\n[{tag}] {i}/{len(symbols)} {sym}")
                base = pd.read_csv(BASE_FEAT / f"{sym.lower()}.csv", parse_dates=["date"])
                feat, names = attach_lb(base, sent, sym, mode, L)
                feat_path = feat_dir / f"{sym.lower()}.csv"
                feat.to_csv(feat_path, index=False)

                model_name = f"ppo_{sym.lower()}_lb_{tag}.zip"
                model_path = MODELS_DIR / model_name
                if not args.skip_train:
                    train_ppo(
                        sym,
                        timesteps=args.timesteps,
                        start=TRAIN_START,
                        end=TRAIN_END,
                        verbose=args.verbose,
                        with_sentiment=True,
                        features_dir=feat_dir,
                        model_name=model_name,
                        sent_names=names,
                    )

                test = feat[(feat["date"] >= TEST_START) & (feat["date"] <= TEST_END)].reset_index(drop=True)
                out_dir = CMP_ROOT / tag / sym.lower()
                part = backtest_df(test, model_path, with_sentiment=True, sent_names=names)
                save(sym, part, out_dir, title=f"PPO+lb {tag} {sym}")
                sm = pd.read_csv(out_dir / "summary.csv").iloc[0].to_dict()
                cmp_rows.append(
                    {
                        "symbol": sym,
                        "mode": mode,
                        "lookback": L,
                        "return": sm["return"],
                        "sharpe": sm["sharpe"],
                        "max_dd": sm["max_dd"],
                        "win_rate": sm["win_rate"],
                        "trade_days": sm["trade_days"],
                    }
                )

    if not cmp_rows:
        print("[done] no rows")
        return
    tab = pd.DataFrame(cmp_rows)
    tab.to_csv(CMP_ROOT / "comparison.csv", index=False)

    # aggregate
    lines = [
        "# PPO × lookback finance_zh 特征对比",
        "",
        "| 模式 | L | 品种数 | 盈利比 | 平均收益 | 中位收益 | 平均Sharpe | 中位Sharpe | 平均MDD |",
        "|------|---|--------|--------|----------|----------|------------|------------|---------|",
    ]
    for (mode, L), g in tab.groupby(["mode", "lookback"]):
        n = len(g)
        lines.append(
            f"| {mode} | {int(L)} | {n} | {(g['return']>0).mean()*100:.1f}% | "
            f"{g['return'].mean()*100:.2f}% | {g['return'].median()*100:.2f}% | "
            f"{g['sharpe'].mean():.2f} | {g['sharpe'].median():.2f} | {g['max_dd'].mean()*100:.2f}% |"
        )
    (CMP_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
