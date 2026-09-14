#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/workspace/lab/UniFutures"
PYTHON="/home/env/futures/bin/python"

cd "$ROOT"

"$PYTHON" news/code/train/build_innovation_signals.py
"$PYTHON" news/code/backtest/run_innovation_batch.py

echo "[done] news/data/results/innovation_batch/innovation_comparison_README.md"
