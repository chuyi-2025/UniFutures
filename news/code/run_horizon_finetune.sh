#!/usr/bin/env bash
# Fine-tune single models for horizons 7/14/30 and backtest with H-day hold.
set -euo pipefail
ROOT="/home/workspace/lab/UniFutures/news"
TRAIN="${ROOT}/code/train"
BT="${ROOT}/code/backtest"
DATA="${ROOT}/data/sentiment"
RES="${ROOT}/data/results"
PY="/home/env/futures/bin/python"
export PYTHONUNBUFFERED=1

mkdir -p "${DATA}" "${RES}"

for H in 7 14 30; do
  echo "======== build samples h=${H} ========"
  "${PY}" "${TRAIN}/build_horizon_samples.py" --horizon "${H}" --out "${DATA}/samples_h${H}.parquet"
done

for H in 7 14 30; do
  for M in finance_zh modernbert finbert2; do
    BS=8
    if [[ "${M}" == "finance_zh" ]]; then BS=16; fi
    echo "======== finetune ${M} h=${H} ========"
    "${PY}" "${TRAIN}/train_finetune.py" \
      --model "${M}" \
      --horizon "${H}" \
      --samples "${DATA}/samples_h${H}.parquet" \
      --batch-size "${BS}" \
      --epochs 2

    OUT="${RES}/sentiment_ft_${M}_h${H}"
    "${PY}" "${BT}/backtest_sentiment.py" \
      --signals "${DATA}/signals_ft_${M}_h${H}.parquet" \
      --out "${OUT}" \
      --start 2025-07-01 --end 2026-12-31 \
      --hold-days "${H}"

    "${PY}" "${BT}/write_readme.py" \
      --out-dir "${OUT}" \
      --title "新闻情绪微调回测（${M}，前瞻${H}日）" \
      --setup "模型=${M}" "模式=微调" "预测horizon=${H}交易日累计收益" \
             "训练窗=2024-01-01 ~ 2025-06-30" "回测窗=2025-07-01 ~ 2026-12-31" \
             "持仓=信号日起持有${H}个交易日" "标签=弱监督(|ret|<0.001*sqrt(H)中性)" \
             "初始资金=100万/品种"
  done
done

"${PY}" "${BT}/compare_horizons.py"
echo "[ALL DONE] horizon experiments"
