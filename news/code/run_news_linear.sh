#!/usr/bin/env bash
# Build simple news factors and run cross-sectional linear backtest.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV_ACTIVATE:-/home/env/futures/bin/activate}"
# shellcheck disable=SC1090
source "${VENV}"
cd "${SCRIPT_DIR}/train"
for L in 7 14 60; do
  python3 build_news_factors.py --lookback "${L}"
done
cd "${SCRIPT_DIR}/backtest"
python3 news_linear_eval.py
