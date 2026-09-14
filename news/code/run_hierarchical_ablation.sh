#!/usr/bin/env bash
# Hierarchical full-document encoder ablation — aligned to plan.
set -euo pipefail

ROOT=/home/workspace/lab/UniFutures
PY=/home/env/futures/bin/python
HIER="$ROOT/news/code/train/hierarchical"
BT="$ROOT/news/code/backtest"
TRAIN="$ROOT/news/code/train"
DATA="$ROOT/news/data/sentiment/hierarchical"
RESULT="$ROOT/news/data/results/hierarchical_ablation"
MAX_LEN=32768

mkdir -p "$DATA" "$RESULT"

echo "===== Stage 0a: fulltext manifest ====="
"$PY" "$HIER/build_fulltext_manifest.py"

echo "===== Stage 0b: BF16 smoke (shortest/median/p99/longest) ====="
"$PY" "$HIER/encode_documents.py" --smoke-percentiles --max-length "$MAX_LEN" \
  --encoder emb finsent qwen06 wiro fin8b || true

echo "===== clearing smoke caches ====="
rm -f "$DATA/doc_embeddings"/*.npy "$DATA/doc_embeddings"/*.json

echo "===== Stage 1: full encode @ ${MAX_LEN} (bf16; OOM => abandon) ====="
for ENC in emb finsent qwen06 wiro fin8b; do
  echo "----- encode $ENC -----"
  "$PY" "$HIER/encode_documents.py" --encoder "$ENC" --max-length "$MAX_LEN" || true
done

echo "===== Stage 2: build L60 windows ====="
"$PY" "$HIER/build_window_dataset.py"

echo "===== Stage 3a: train single-model H01-H12 (3 seeds) ====="
for S in $(seq -w 1 12); do
  SCH="H${S}"
  for SEED in 0 1 2; do
    echo "----- train $SCH seed=$SEED -----"
    "$PY" "$HIER/train_ablation.py" --scheme "$SCH" --seed "$SEED" || true
  done
done

echo "===== Stage 3b: resolve best pools for H21-H24 ====="
"$PY" "$HIER/train_ablation.py" --scheme H01 --seed 0 --resolve-best-pools || true
cat "$RESULT/best_pools.json" 2>/dev/null || true

echo "===== Stage 3c: train concat/gate/structure H13-H24 ====="
for S in $(seq -w 13 24); do
  SCH="H${S}"
  for SEED in 0 1 2; do
    echo "----- train $SCH seed=$SEED -----"
    "$PY" "$HIER/train_ablation.py" --scheme "$SCH" --seed "$SEED" || true
  done
done

echo "===== Stage 4: ensure lookback baselines exist ====="
if [[ ! -f "$ROOT/news/data/sentiment/lookback/signals_lb_finance_zh_concat_L60.parquet" ]]; then
  echo "building concat L60 baseline signals..."
  "$PY" "$TRAIN/infer_lookback_concat.py" --lookback 60 || true
fi
if ! ls "$ROOT/news/data/sentiment/lookback/"*wavg_exp*L60* >/dev/null 2>&1; then
  echo "building wavg_exp L60 baseline signals..."
  "$PY" "$TRAIN/build_lookback_signals.py" --scheme wavg_exp --lookback 60 || true
fi

echo "===== Stage 5: comparison + baselines (0/2/5bp) ====="
"$PY" "$BT/compare_hierarchical_ablation.py"
echo "[done] hierarchical ablation — see $RESULT/comparison.csv"
echo "NOTE: 2025-07+ is retrospective OOS, not pristine holdout."
