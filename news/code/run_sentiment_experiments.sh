#!/usr/bin/env bash
# End-to-end news sentiment experiments: zeroshot -> finetune -> ensemble
set -euo pipefail
ROOT="/home/workspace/lab/UniFutures/news"
TRAIN="${ROOT}/code/train"
BT="${ROOT}/code/backtest"
DATA="${ROOT}/data/sentiment"
RES="${ROOT}/data/results"
PY="/home/env/futures/bin/python"
export PYTHONUNBUFFERED=1

mkdir -p "${DATA}" "${RES}"

echo "======== [1/4] build dataset ========"
"${PY}" "${TRAIN}/build_dataset.py" --out "${DATA}/samples.parquet"

echo "======== [2/4] zeroshot infer + backtest ========"
"${PY}" "${TRAIN}/infer_zeroshot.py" --model finance_zh --out "${DATA}/signals_zeroshot_finance_zh.parquet"
"${PY}" "${TRAIN}/infer_zeroshot.py" --model modernbert --out "${DATA}/signals_zeroshot_modernbert.parquet"

"${PY}" "${BT}/backtest_sentiment.py" \
  --signals "${DATA}/signals_zeroshot_finance_zh.parquet" \
  --out "${RES}/sentiment_zeroshot_finance_zh" \
  --start 2024-01-01 --end 2026-12-31
"${PY}" "${BT}/write_readme.py" \
  --out-dir "${RES}/sentiment_zeroshot_finance_zh" \
  --title "新闻情绪零样本回测（finance-sentiment-zh-base）" \
  --setup "模型=finance-sentiment-zh-base" "模式=零样本" "回测区间=2024-01-01 ~ 2026-12-31" \
         "信号=pos多/neu空/neg空" "收益=研报日后首个交易日主连 log_ret" "初始资金=100万/品种"

"${PY}" "${BT}/backtest_sentiment.py" \
  --signals "${DATA}/signals_zeroshot_modernbert.parquet" \
  --out "${RES}/sentiment_zeroshot_modernbert" \
  --start 2024-01-01 --end 2026-12-31
"${PY}" "${BT}/write_readme.py" \
  --out-dir "${RES}/sentiment_zeroshot_modernbert" \
  --title "新闻情绪零样本回测（modernbert-fingpt）" \
  --setup "模型=modernbert-fingpt" "模式=零样本" "回测区间=2024-01-01 ~ 2026-12-31" \
         "信号=9档映射到多/空/平" "收益=研报日后首个交易日主连 log_ret" "初始资金=100万/品种"

echo "======== [3/4] finetune three models ========"
for m in finance_zh modernbert finbert2; do
  "${PY}" "${TRAIN}/train_finetune.py" --model "${m}"
  "${PY}" "${BT}/backtest_sentiment.py" \
    --signals "${DATA}/signals_ft_${m}.parquet" \
    --out "${RES}/sentiment_ft_${m}" \
    --start 2025-07-01 --end 2026-12-31
  "${PY}" "${BT}/write_readme.py" \
    --out-dir "${RES}/sentiment_ft_${m}" \
    --title "新闻情绪微调回测（${m}）" \
    --setup "模型=${m}" "模式=微调" "训练窗=2024-01-01 ~ 2025-06-30" \
           "回测窗=2025-07-01 ~ 2026-12-31" "标签=次日收益弱监督(|ret|<0.001中性)" \
           "初始资金=100万/品种"
done

echo "======== [4/4] ensemble ========"
"${PY}" "${TRAIN}/ensemble_signals.py"
"${PY}" "${BT}/backtest_sentiment.py" \
  --signals "${DATA}/signals_ensemble_vote.parquet" \
  --out "${RES}/sentiment_ensemble_vote" \
  --start 2025-07-01 --end 2026-12-31
"${PY}" "${BT}/write_readme.py" \
  --out-dir "${RES}/sentiment_ensemble_vote" \
  --title "新闻情绪融合回测（多数表决）" \
  --setup "融合=三模型仓位多数表决" "回测窗=2025-07-01 ~ 2026-12-31" "初始资金=100万/品种"

"${PY}" "${BT}/backtest_sentiment.py" \
  --signals "${DATA}/signals_ensemble_prob.parquet" \
  --out "${RES}/sentiment_ensemble_prob" \
  --start 2025-07-01 --end 2026-12-31
"${PY}" "${BT}/write_readme.py" \
  --out-dir "${RES}/sentiment_ensemble_prob" \
  --title "新闻情绪融合回测（概率平均）" \
  --setup "融合=三模型 softmax 概率平均后 argmax" "回测窗=2025-07-01 ~ 2026-12-31" "初始资金=100万/品种"

echo "[ALL DONE] results under ${RES}"
