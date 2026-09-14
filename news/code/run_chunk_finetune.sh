#!/usr/bin/env bash
# Full-text token chunks -> finance_zh H1 fine-tune -> mean/vote backtests.
set -euo pipefail

ROOT="/home/workspace/lab/UniFutures/news"
PY="/home/env/futures/bin/python"
TRAIN="$ROOT/code/train"
BT="$ROOT/code/backtest"
DATA="$ROOT/data/sentiment"
RESULTS="$ROOT/data/results"

SAMPLES="$DATA/chunk_samples_h1.parquet"
RAW_SIGNALS="$DATA/signals_ft_finance_zh_chunks_h1.parquet"
AGG_PREFIX="$DATA/signals_ft_finance_zh_chunks_h1"
MODEL="$DATA/ft_models/finance_zh_chunks_h1"

export PYTHONUNBUFFERED=1

"$PY" "$TRAIN/build_chunk_samples.py" \
  --out "$SAMPLES" \
  --max-length 512

"$PY" "$TRAIN/train_finetune.py" \
  --model finance_zh \
  --samples "$SAMPLES" \
  --out-dir "$MODEL" \
  --signal-out "$RAW_SIGNALS" \
  --horizon 1 \
  --epochs 2 \
  --batch-size 16 \
  --max-length 512

"$PY" "$TRAIN/aggregate_chunk_signals.py" \
  --signals "$RAW_SIGNALS" \
  --out-prefix "$AGG_PREFIX"

for METHOD in mean vote; do
  OUT="$RESULTS/sentiment_ft_finance_zh_chunks_h1_${METHOD}"
  "$PY" "$BT/backtest_sentiment.py" \
    --signals "${AGG_PREFIX}_${METHOD}.parquet" \
    --out "$OUT" \
    --start 2025-07-01 \
    --end 2026-12-31 \
    --hold-days 1
  "$PY" "$BT/write_readme.py" \
    --out-dir "$OUT" \
    --title "全文分块 finance_zh 次日回测（${METHOD} 聚合）" \
    --setup \
      "文本=全文按 finance_zh tokenizer 切块，每块含品种前缀且总长≤512 token" \
      "多品种=同一全文分别复制给每个品种" \
      "chunk聚合=${METHOD}，先聚合为每文档每品种一票，再按同日文档多数表决" \
      "标签=对应品种次日主连涨跌，|ret|<0.001为中性" \
      "训练窗=2024-01-01 ~ 2025-06-30" \
      "回测窗=2025-07-01 ~ 2026-12-31" \
      "初始资金=100万/品种"
done

echo "[ALL DONE] chunk finance_zh H1 experiment"
