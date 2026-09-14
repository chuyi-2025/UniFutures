#!/usr/bin/env bash
# End-to-end: DeepSeek variety compression (resumable) -> train/backtest.
set -euo pipefail

ROOT="/home/workspace/lab/UniFutures/news"
PY="/home/env/futures/bin/python"
COMP="$ROOT/data/sentiment/deepseek_variety_compressed"
LOG_DIR="$COMP/logs"
mkdir -p "$LOG_DIR"

echo "[$(date -Is)] start compress+train pipeline" | tee -a "$LOG_DIR/orchestrator.log"

# Resume compression until the completion gate passes.
for attempt in 1 2 3 4 5 6 7 8; do
  echo "[$(date -Is)] compress attempt=$attempt" | tee -a "$LOG_DIR/orchestrator.log"
  set +e
  "$PY" "$ROOT/code/deepseek_preprocess/compress_by_variety.py" --workers 16 \
    >>"$LOG_DIR/compress.log" 2>&1
  rc=$?
  set -e
  echo "[$(date -Is)] compress exit=$rc" | tee -a "$LOG_DIR/orchestrator.log"

  if "$PY" - <<'PY'
import json
import sys
from pathlib import Path

root = Path("/home/workspace/lab/UniFutures/news/data/sentiment/deepseek_variety_compressed")
summary_path = root / "summary.json"
if not summary_path.exists():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text())
json_n = len(list((root / "json").glob("*.json")))
docs_total = int(summary.get("docs_total", 0))
compressed_docs = int(summary.get("compressed_docs", 0))
failed = int(summary.get("failed", 0))
print(
    {
        "docs_total": docs_total,
        "compressed_docs": compressed_docs,
        "failed": failed,
        "json_n": json_n,
    },
    flush=True,
)
ok = (
    failed == 0
    and docs_total >= 1000
    and json_n >= int(0.95 * docs_total)
    # Some reports legitimately yield empty 品种分析; require enough mapped rows.
    and compressed_docs >= int(0.90 * docs_total)
)
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[$(date -Is)] compression gate passed" | tee -a "$LOG_DIR/orchestrator.log"
    break
  fi
  if [[ "$attempt" -eq 8 ]]; then
    echo "[$(date -Is)] compression gate failed after retries" | tee -a "$LOG_DIR/orchestrator.log" >&2
    exit 3
  fi
  sleep 30
done

echo "[$(date -Is)] start train/backtest" | tee -a "$LOG_DIR/orchestrator.log"
bash "$ROOT/code/run_deepseek_compressed_experiment.sh" \
  >>"$LOG_DIR/train_backtest.log" 2>&1
echo "[$(date -Is)] pipeline finished" | tee -a "$LOG_DIR/orchestrator.log"
