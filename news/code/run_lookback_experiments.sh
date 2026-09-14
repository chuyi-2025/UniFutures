#!/usr/bin/env bash
# Lookback experiments: past L days -> predict 1d / 7d, fixed ft finance_zh
set -euo pipefail
ROOT="/home/workspace/lab/UniFutures/news"
UF="/home/workspace/lab/UniFutures"
TRAIN="${ROOT}/code/train"
BT="${ROOT}/code/backtest"
DATA="${ROOT}/data/sentiment"
RES="${ROOT}/data/results"
LB="${DATA}/lookback"
PY="/home/env/futures/bin/python"
export PYTHONUNBUFFERED=1

mkdir -p "${LB}" "${RES}"

echo "======== [1/5] lookback wavg signals ========"
"${PY}" "${TRAIN}/build_lookback_signals.py" \
  --lookback 3 7 14 \
  --scheme wavg_uniform wavg_linear wavg_exp daily_wavg_linear \
  --out-dir "${LB}"

echo "======== [2/5] lookback concat infer ========"
"${PY}" "${TRAIN}/infer_lookback_concat.py" \
  --lookback 3 7 14 \
  --out-dir "${LB}"

echo "======== [3/5] backtest lookback × {1d,7d} ========"
for scheme in wavg_uniform wavg_linear wavg_exp daily_wavg_linear concat; do
  for L in 3 7 14; do
    SIG="${LB}/signals_lb_finance_zh_${scheme}_L${L}.parquet"
    if [[ ! -f "${SIG}" ]]; then
      echo "[skip] missing ${SIG}"
      continue
    fi
    for H in 1 7; do
      OUT="${RES}/sentiment_lb_finance_zh_${scheme}_L${L}_h${H}"
      echo "---- ${scheme} L=${L} h=${H} ----"
      "${PY}" "${BT}/backtest_sentiment.py" \
        --signals "${SIG}" \
        --out "${OUT}" \
        --start 2025-07-01 --end 2026-12-31 \
        --hold-days "${H}"
      "${PY}" "${BT}/write_readme.py" \
        --out-dir "${OUT}" \
        --title "lookback ${scheme} L=${L} predict=${H}d（ft finance_zh）" \
        --setup "模型=ft finance_zh冻结" "窗口=[T-L,T)不含当天" "L=${L}" \
               "方案=${scheme}" "持有=${H}交易日" "回测=2025-07-01~2026-12-31"
    done
  done
done

echo "======== [4/5] compare lookback sentiment ========"
"${PY}" "${BT}/compare_lookback.py"

echo "======== [5/5] lookback daily feats + PPO ========"
"${PY}" "${UF}/code/build_feature/build_lookback_sent_daily.py" --lookback 3 7 14
"${PY}" "${UF}/code/train/compare_lookback_ppo.py" \
  --lookback 3 7 14 \
  --mode pool stack \
  --timesteps 80000

echo "[ALL DONE] see ${RES}/lookback_comparison_README.md and ${UF}/data/results/ppo_lookback_cmp/README.md"
