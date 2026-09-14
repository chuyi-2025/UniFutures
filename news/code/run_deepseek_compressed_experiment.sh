#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/workspace/lab/UniFutures/news"
PY="/home/env/futures/bin/python"
TRAIN="$ROOT/code/train"
BT="$ROOT/code/backtest"
DATA="$ROOT/data/sentiment"
COMP="$DATA/deepseek_variety_compressed"
MODEL="$DATA/ft_models/finance_zh_deepseek_compressed"
FROZEN_CKPT="$DATA/ft_models/finance_zh/best"
SIGNALS="$DATA/signals_ft_finance_zh_deepseek_compressed.parquet"
SIGNALS_FROZEN="$DATA/signals_ft_finance_zh_deepseek_frozen.parquet"
LOOKBACK="$COMP/lookback"
RESULTS="$ROOT/data/results/deepseek_compressed"
TAG="finance_zh_deepseek_compressed"
TAG_FROZEN="finance_zh_deepseek_frozen"

mkdir -p "$LOOKBACK" "$RESULTS"

"$PY" - <<'PY'
import json
import sys
from pathlib import Path

root = Path("/home/workspace/lab/UniFutures/news/data/sentiment/deepseek_variety_compressed")
summary_path = root / "summary.json"
items_path = root / "compressed_items.parquet"
if not summary_path.exists() or not items_path.exists():
    raise SystemExit("missing compression outputs")
summary = json.loads(summary_path.read_text())
json_n = len(list((root / "json").glob("*.json")))
docs_total = int(summary.get("docs_total", 0))
compressed_docs = int(summary.get("compressed_docs", 0))
failed = int(summary.get("failed", 0))
# Require a completed full-corpus run, not the early smoke summary.
if failed != 0:
    raise SystemExit(f"compression still has failures: {failed}")
if docs_total < 1000:
    raise SystemExit(f"unexpected docs_total={docs_total}")
if json_n < int(0.95 * docs_total):
    raise SystemExit(f"incomplete json cache: json_files={json_n} docs_total={docs_total}")
# Empty 品种分析 is valid for some reports; require high json coverage and enough mapped docs.
if compressed_docs < int(0.90 * docs_total):
    raise SystemExit(
        f"incomplete compression: compressed_docs={compressed_docs} "
        f"docs_total={docs_total} json_files={json_n}"
    )
print(
    f"[gate] compression ok docs={compressed_docs}/{docs_total} "
    f"rows={summary.get('compressed_rows')} json={json_n}"
)
PY

"$PY" "$TRAIN/build_compressed_samples.py" \
  --items "$COMP/compressed_items.parquet" \
  --out-dir "$COMP" \
  --horizon 1 7

# Arm A: retrain finance_zh on compressed per-symbol texts (user request).
"$PY" "$TRAIN/train_finetune.py" \
  --samples "$COMP/samples_h1.parquet" \
  --model finance_zh \
  --out-dir "$MODEL" \
  --signal-out "$SIGNALS" \
  --epochs 2 \
  --batch-size 16 \
  --max-length 512

"$PY" "$TRAIN/build_lookback_signals.py" \
  --docs "$SIGNALS" \
  --prob-cols prob_0 prob_1 prob_2 \
  --lookback 60 \
  --scheme wavg_exp \
  --tag "$TAG" \
  --out-dir "$LOOKBACK"

"$PY" "$TRAIN/infer_lookback_concat.py" \
  --samples "$COMP/samples_h1.parquet" \
  --ckpt "$MODEL/best" \
  --lookback 60 \
  --tag "$TAG" \
  --out-dir "$LOOKBACK" \
  --batch-size 32 \
  --max-length 512 \
  --max-chars 4000

# Arm B: freeze original lookback ckpt, only swap compressed texts (apples-to-apples).
if [[ -d "$FROZEN_CKPT" ]]; then
  "$PY" - <<PY
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

samples = pd.read_parquet("$COMP/samples_h1.parquet")
ckpt = Path("$FROZEN_CKPT")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    str(ckpt), trust_remote_code=True
).eval().to(device)

texts = samples["text"].astype(str).tolist()
probs_all = []
batch_size = 32
with torch.inference_mode():
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        probs_all.append(torch.softmax(model(**enc).logits, dim=-1).cpu())
probs = torch.cat(probs_all, dim=0).numpy()
out = samples.drop(columns=["text"], errors="ignore").copy()
out["pred_class"] = probs.argmax(axis=1).astype(int)
out["position"] = out["pred_class"].map({0: 1, 1: 0, 2: -1}).astype(int)
for i in range(3):
    out[f"prob_{i}"] = probs[:, i]
path = Path("$SIGNALS_FROZEN")
out.to_parquet(path, index=False)
out.to_csv(path.with_suffix(".csv"), index=False)
print(f"[frozen-score] n={len(out)} -> {path}")
PY

  "$PY" "$TRAIN/build_lookback_signals.py" \
    --docs "$SIGNALS_FROZEN" \
    --prob-cols prob_0 prob_1 prob_2 \
    --lookback 60 \
    --scheme wavg_exp \
    --tag "$TAG_FROZEN" \
    --out-dir "$LOOKBACK"

  "$PY" "$TRAIN/infer_lookback_concat.py" \
    --samples "$COMP/samples_h1.parquet" \
    --ckpt "$FROZEN_CKPT" \
    --lookback 60 \
    --tag "$TAG_FROZEN" \
    --out-dir "$LOOKBACK" \
    --batch-size 32 \
    --max-length 512 \
    --max-chars 4000
fi

for arm in "$TAG" "$TAG_FROZEN"; do
  for scheme in concat wavg_exp; do
    signal="$LOOKBACK/signals_lb_${arm}_${scheme}_L60.parquet"
    [[ -f "$signal" ]] || continue
    for hold in 1 7; do
      "$PY" "$BT/backtest_sentiment.py" \
        --signals "$signal" \
        --out "$RESULTS/${arm}_${scheme}_L60_h${hold}" \
        --start 2025-07-01 \
        --end 2026-12-31 \
        --hold-days "$hold"
    done
  done
done

"$PY" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/home/workspace/lab/UniFutures/news/data/results/deepseek_compressed")
baseline_dir = Path("/home/workspace/lab/UniFutures/news/data/results")
rows = []
for arm in ("finance_zh_deepseek_compressed", "finance_zh_deepseek_frozen"):
    for scheme in ("concat", "wavg_exp"):
        for hold in (1, 7):
            path = root / f"{arm}_{scheme}_L60_h{hold}" / "summary.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "status" in df.columns:
                df = df[df["status"] == "ok"]
            rows.append(
                {
                    "arm": arm,
                    "scheme": scheme,
                    "lookback": 60,
                    "hold_days": hold,
                    "n_symbols": len(df),
                    "profit_ratio": float((df["return"] > 0).mean()) if len(df) else 0.0,
                    "avg_return": float(df["return"].mean()) if len(df) else 0.0,
                    "median_return": float(df["return"].median()) if len(df) else 0.0,
                    "avg_sharpe": float(df["sharpe"].mean()) if len(df) else 0.0,
                    "median_sharpe": float(df["sharpe"].median()) if len(df) else 0.0,
                    "avg_max_dd": float(df["max_dd"].mean()) if len(df) else 0.0,
                }
            )

# Reference rows from the original frozen lookback comparison, if present.
for scheme in ("concat", "wavg_exp"):
    for hold in (1, 7):
        path = baseline_dir / f"sentiment_lb_finance_zh_{scheme}_L60_h{hold}" / "summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "status" in df.columns:
            df = df[df["status"] == "ok"]
        rows.append(
            {
                "arm": "original_fulltext_frozen",
                "scheme": scheme,
                "lookback": 60,
                "hold_days": hold,
                "n_symbols": len(df),
                "profit_ratio": float((df["return"] > 0).mean()) if len(df) else 0.0,
                "avg_return": float(df["return"].mean()) if len(df) else 0.0,
                "median_return": float(df["return"].median()) if len(df) else 0.0,
                "avg_sharpe": float(df["sharpe"].mean()) if len(df) else 0.0,
                "median_sharpe": float(df["sharpe"].median()) if len(df) else 0.0,
                "avg_max_dd": float(df["max_dd"].mean()) if len(df) else 0.0,
            }
        )

out = pd.DataFrame(rows)
out.to_csv(root / "comparison.csv", index=False)
lines = [
    "# DeepSeek 品种压缩：concat L60 与指数衰减对比",
    "",
    "- `finance_zh_deepseek_compressed`：压缩文本上重训 finance_zh。",
    "- `finance_zh_deepseek_frozen`：冻结原 lookback ckpt，仅替换压缩文本（更接近 lookback README 协议）。",
    "- `original_fulltext_frozen`：原全文 lookback 对照。",
    "- 交易日 T 仅使用 `[T-60, T)` 研报，不含当天。",
    "- 所有新产物写入独立目录，未覆盖原 `samples.parquet` / `lookback/`。",
    "",
    "| 臂 | 方案 | 持有期 | 品种数 | 盈利品种比 | 平均收益 | 中位收益 | 平均Sharpe | 中位Sharpe | 平均MDD |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in out.itertuples(index=False):
    lines.append(
        f"| {row.arm} | {row.scheme} L60 | {row.hold_days}d | {row.n_symbols} | "
        f"{row.profit_ratio:.1%} | {row.avg_return:.2%} | {row.median_return:.2%} | "
        f"{row.avg_sharpe:.2f} | {row.median_sharpe:.2f} | {row.avg_max_dd:.2%} |"
    )
(root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out.to_string(index=False))
print(root / "REPORT.md")
PY
