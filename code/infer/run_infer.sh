#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/env/futures/bin/activate

TREE_LOG="${TREE_LOG:-/home/workspace/lab/Tree-Stock/futures/data/update_all_contracts.log}"
SKIP_UPDATE="${SKIP_UPDATE:-0}"
WAIT_SECS="${WAIT_SECS:-720}"   # when skipping update, wait up to 12m for 19:30 Done
POLL_SECS="${POLL_SECS:-15}"

ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }

echo "[$(ts)] infer start"

if [[ "${SKIP_UPDATE}" == "1" ]]; then
  echo "[1/5] skip akshare update (SKIP_UPDATE=1); wait evening Done if needed"
  START_SIZE=0
  if [[ -f "${TREE_LOG}" ]]; then
    START_SIZE="$(stat -c%s "${TREE_LOG}")"
  fi
  # If a Done. already landed in the last 25 minutes, do not wait.
  RECENT_DONE=0
  if [[ -f "${TREE_LOG}" ]]; then
    if python3 - <<PY
from pathlib import Path
from datetime import datetime, timedelta
p = Path("${TREE_LOG}")
text = p.read_text(errors="ignore").splitlines()
cutoff = datetime.now() - timedelta(minutes=25)
# log has no timestamps on Done; use file mtime as proxy when last line is Done
if text and text[-1].startswith("Done."):
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    raise SystemExit(0 if mtime >= cutoff else 1)
raise SystemExit(1)
PY
    then
      RECENT_DONE=1
      echo "[$(ts)] recent Tree-Stock Done found"
    fi
  fi
  if [[ "${RECENT_DONE}" -ne 1 && "${WAIT_SECS}" -gt 0 ]]; then
    echo "[$(ts)] wait Tree-Stock Done (timeout ${WAIT_SECS}s)"
    DEADLINE=$((SECONDS + WAIT_SECS))
    DONE=0
    while (( SECONDS < DEADLINE )); do
      if [[ -f "${TREE_LOG}" ]]; then
        if tail -c "+$((START_SIZE + 1))" "${TREE_LOG}" 2>/dev/null | grep -q '^Done\.'; then
          DONE=1
          echo "[$(ts)] Tree-Stock update finished"
          break
        fi
      fi
      sleep "${POLL_SECS}"
    done
    if [[ "${DONE}" -ne 1 ]]; then
      echo "[$(ts)] WARN: Tree-Stock Done not seen, continue" >&2
    fi
  fi
else
  echo "[1/5] update source contracts (Tree-Stock, append mode)"
  python3 "/home/workspace/lab/Tree-Stock/futures/data/update_all_contracts_from_akshare.py" --no-sleep
fi

echo "[2/5] sync contracts -> UniFutures (append)"
python3 "${SCRIPT_DIR}/sync_contracts.py"

echo "[3/5] build infer features"
bash "${SCRIPT_DIR}/run_latest.sh"

echo "[4/5] backtest + latest signals"
CUDA_VISIBLE_DEVICES="" python3 "${SCRIPT_DIR}/run_backtest.py"

echo "[5/5] export PPO multi-symbol last-month signals"
python3 "${SCRIPT_DIR}/export_ppo_month.py"

echo "[$(ts)] done -> $(python3 -c "import sys; sys.path.insert(0,'${SCRIPT_DIR}'); import config; print(config.DAILY_PPO_DIR)")"
