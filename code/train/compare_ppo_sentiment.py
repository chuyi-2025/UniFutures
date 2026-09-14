#!/usr/bin/env python3
"""Train PPO+sentiment and compare vs original PPO on the same OOS window.

Fair protocol:
  - Sentiment experts: next-day 3-model FT ensemble (prob-avg) daily features
  - PPO+sent train: 2024-01-01 ~ 2025-06-30 (obs = 15 experts + 4 sentiment)
  - Ablation PPO (no sent, same train window): optional
  - Original PPO: existing ppo_{sym}.zip trained ≤2023-12-31
  - OOS backtest: 2025-07-01 ~ 2026-12-31
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
    ExpertStack,
    PPO_FEATURES_DIR,
    PPO_SENT_FEATURES_DIR,
    SENT_DAILY_PATH,
    train_ppo,
)

TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2025-06-30")
TEST_START = pd.Timestamp("2025-07-01")
TEST_END = pd.Timestamp("2026-12-31")

CMP_ROOT = RESULTS_DIR / "ppo_sentiment_cmp"


def symbols_with_sentiment() -> list[str]:
    sent = pd.read_parquet(SENT_DAILY_PATH)
    sent_syms = set(sent["symbol"].astype(str).str.upper())
    return [s for s in list_symbols() if s in sent_syms and (MODELS_DIR / f"ppo_{s.lower()}.zip").exists()]


def ensure_features(stack: ExpertStack, sym: str) -> tuple[Path, Path]:
    """Build base features once, then derive sentiment-augmented copy."""
    base_dir = CMP_ROOT / "features_base"
    sent_dir = PPO_SENT_FEATURES_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    sent_dir.mkdir(parents=True, exist_ok=True)
    base_path = base_dir / f"{sym.lower()}.csv"
    sent_path = sent_dir / f"{sym.lower()}.csv"

    if base_path.exists():
        base_df = pd.read_csv(base_path, parse_dates=["date"])
        ok_base = (
            len(base_df) >= 50
            and base_df["date"].min() <= TRAIN_START
            and base_df["date"].max() >= TEST_START
        )
    else:
        ok_base = False
    if not ok_base:
        base_df = stack.build(sym, start=TRAIN_START, end=TEST_END, with_sentiment=False)
        if base_df.empty:
            raise ValueError("empty features")
        base_df.to_csv(base_path, index=False)

    if sent_path.exists():
        sent_df = pd.read_csv(sent_path, parse_dates=["date"])
        ok_sent = (
            len(sent_df) >= 50
            and sent_df["date"].min() <= TRAIN_START
            and sent_df["date"].max() >= TEST_START
            and all(c in sent_df.columns for c in ("sent_pos", "sent_p0", "sent_p1", "sent_p2"))
        )
    else:
        ok_sent = False
    if not ok_sent:
        sent_df = stack.attach_sentiment(pd.read_csv(base_path, parse_dates=["date"]), sym)
        sent_df.to_csv(sent_path, index=False)
    return base_path, sent_path


def run_backtest_slice(
    sym: str,
    feat_path: Path,
    model_path: Path,
    out_dir: Path,
    with_sentiment: bool,
    title: str,
) -> dict:
    df = pd.read_csv(feat_path, parse_dates=["date"])
    df = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].reset_index(drop=True)
    if with_sentiment and "sent_pos" not in df.columns:
        df = ExpertStack(use_kronos=False).attach_sentiment(df, sym)
    part = backtest_df(df, model_path, with_sentiment=with_sentiment)
    if part.empty:
        raise ValueError("empty backtest")
    save(sym, part, out_dir, title=title)
    sm = pd.read_csv(out_dir / "summary.csv").iloc[0].to_dict()
    sm["status"] = "ok"
    return sm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true", help="skip same-window no-sentiment PPO")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not SENT_DAILY_PATH.exists():
        raise SystemExit(f"missing {SENT_DAILY_PATH}; run build_sentiment_daily.py first")

    symbols = [s.upper() for s in args.symbols] if args.symbols else symbols_with_sentiment()
    if args.limit > 0:
        symbols = symbols[: args.limit]
    print(f"[cmp] symbols={len(symbols)} train={TRAIN_START.date()}~{TRAIN_END.date()} test={TEST_START.date()}~{TEST_END.date()}")

    stack = ExpertStack(use_kronos=True)
    rows = []

    for i, sym in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {sym}")
        row = {"symbol": sym}
        try:
            if not args.skip_prepare:
                base_path, sent_path = ensure_features(stack, sym)
            else:
                base_path = CMP_ROOT / "features_base" / f"{sym.lower()}.csv"
                sent_path = PPO_SENT_FEATURES_DIR / f"{sym.lower()}.csv"

            # --- original PPO (≤2023 train) ---
            orig_model = MODELS_DIR / f"ppo_{sym.lower()}.zip"
            orig_out = CMP_ROOT / "original" / sym.lower()
            sm_o = run_backtest_slice(sym, base_path, orig_model, orig_out, False, f"PPO original {sym}")
            row.update({f"orig_{k}": sm_o[k] for k in ("return", "sharpe", "max_dd", "win_rate", "trade_days")})

            # --- PPO + sentiment ---
            sent_model = MODELS_DIR / f"ppo_{sym.lower()}_sent.zip"
            if not args.skip_train:
                train_ppo(
                    sym,
                    timesteps=args.timesteps,
                    start=TRAIN_START,
                    end=TRAIN_END,
                    verbose=args.verbose,
                    with_sentiment=True,
                    features_dir=PPO_SENT_FEATURES_DIR,
                    model_name=f"ppo_{sym.lower()}_sent.zip",
                )
            sent_out = CMP_ROOT / "sent" / sym.lower()
            sm_s = run_backtest_slice(sym, sent_path, sent_model, sent_out, True, f"PPO+sent {sym}")
            row.update({f"sent_{k}": sm_s[k] for k in ("return", "sharpe", "max_dd", "win_rate", "trade_days")})

            # --- ablation: same train window, no sentiment ---
            if not args.skip_ablation:
                abl_model = MODELS_DIR / f"ppo_{sym.lower()}_abl2024.zip"
                abl_feat = CMP_ROOT / "features_base"
                if not args.skip_train:
                    train_ppo(
                        sym,
                        timesteps=args.timesteps,
                        start=TRAIN_START,
                        end=TRAIN_END,
                        verbose=args.verbose,
                        with_sentiment=False,
                        features_dir=abl_feat,
                        model_name=f"ppo_{sym.lower()}_abl2024.zip",
                    )
                abl_out = CMP_ROOT / "ablation" / sym.lower()
                sm_a = run_backtest_slice(sym, base_path, abl_model, abl_out, False, f"PPO abl2024 {sym}")
                row.update({f"abl_{k}": sm_a[k] for k in ("return", "sharpe", "max_dd", "win_rate", "trade_days")})

            row["status"] = "ok"
            print(
                f"  orig={row['orig_return']*100:.2f}% sent={row['sent_return']*100:.2f}%"
                + (f" abl={row['abl_return']*100:.2f}%" if "abl_return" in row else "")
            )
        except Exception as e:
            row["status"] = "fail"
            row["error"] = str(e)
            print(f"  FAIL: {e}")
        rows.append(row)

    CMP_ROOT.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(CMP_ROOT / "comparison.csv", index=False)

    ok = tab[tab["status"] == "ok"].copy()
    lines = [
        "# PPO × 次日情绪融合对比",
        "",
        "## 设定",
        "",
        "| 项目 | 配置 |",
        "|------|------|",
        "| 情绪专家 | 次日三模型微调融合（概率平均）→ `sent_pos` + `sent_p0..2` |",
        "| 原始 PPO | 既有 `ppo_{sym}.zip`，专家特征训练 ≤2023-12-31，obs=15 |",
        "| PPO+sent | 新增 4 维情绪观测，训练窗 2024-01-01~2025-06-30，obs=19 |",
        "| 消融 PPO | 同训练窗、无情绪，obs=15（控制训练区间差异） |",
        "| 回测窗 | 2025-07-01 ~ 2026-12-31 |",
        "| 资金 | 100 万/品种；`capital *= exp(pos × log_ret_1d)` |",
        "",
    ]
    if ok.empty:
        lines += ["无成功品种。", ""]
    else:
        def block(prefix: str, name: str) -> list[str]:
            if f"{prefix}_return" not in ok.columns:
                return []
            r = ok[f"{prefix}_return"]
            s = ok[f"{prefix}_sharpe"]
            n = len(ok)
            win = int((r > 0).sum())
            return [
                f"### {name}",
                "",
                f"| 指标 | 值 |",
                f"|------|-----|",
                f"| 品种数 | {n} |",
                f"| 盈利品种 | {win}/{n}（{win/n*100:.1f}%） |",
                f"| 平均收益 | {r.mean()*100:.2f}% |",
                f"| 中位收益 | {r.median()*100:.2f}% |",
                f"| 平均 Sharpe | {s.mean():.2f} |",
                f"| 中位 Sharpe | {s.median():.2f} |",
                f"| 平均 MDD | {ok[f'{prefix}_max_dd'].mean()*100:.2f}% |",
                "",
            ]

        lines += ["## 汇总", ""]
        lines += block("orig", "原始 PPO")
        lines += block("sent", "PPO + 情绪融合")
        lines += block("abl", "消融（同窗无情绪）")

        if "sent_return" in ok.columns and "orig_return" in ok.columns:
            d = ok["sent_return"] - ok["orig_return"]
            lines += [
                "## 相对原始 PPO（sent − orig）",
                "",
                f"- 平均收益差：{d.mean()*100:+.2f} pp",
                f"- 中位收益差：{d.median()*100:+.2f} pp",
                f"- sent 胜出品种：{int((d > 0).sum())}/{len(ok)}",
                "",
            ]
        if "sent_return" in ok.columns and "abl_return" in ok.columns:
            d2 = ok["sent_return"] - ok["abl_return"]
            lines += [
                "## 相对同窗消融（sent − abl）",
                "",
                f"- 平均收益差：{d2.mean()*100:+.2f} pp",
                f"- 中位收益差：{d2.median()*100:+.2f} pp",
                f"- sent 胜出品种：{int((d2 > 0).sum())}/{len(ok)}",
                "",
            ]

        lines += [
            "## 品种明细（按 sent 收益排序）",
            "",
            "| 品种 | orig收益 | sent收益 | abl收益 | orig Sharpe | sent Sharpe | Δsent-orig |",
            "|------|----------|----------|---------|-------------|-------------|------------|",
        ]
        show = ok.sort_values("sent_return", ascending=False)
        for _, r in show.iterrows():
            abl = f"{r['abl_return']*100:.2f}%" if "abl_return" in r and pd.notna(r.get("abl_return")) else "-"
            lines.append(
                f"| {r['symbol']} | {r['orig_return']*100:.2f}% | {r['sent_return']*100:.2f}% | {abl} | "
                f"{r['orig_sharpe']:.2f} | {r['sent_sharpe']:.2f} | "
                f"{(r['sent_return']-r['orig_return'])*100:+.2f} pp |"
            )
        lines.append("")

    readme = CMP_ROOT / "README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[done] {CMP_ROOT / 'comparison.csv'}")
    print(f"[done] {readme}")


if __name__ == "__main__":
    main()
