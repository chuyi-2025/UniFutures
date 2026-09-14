#!/usr/bin/env bash
# Daily live run for the final scheme (pos64+RSI overlay, 一手≤5万, BOOK≥1.18).
# Cron 19:50: contracts already pulled at 19:30; blotter only (SKIP_UPDATE=1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/home/env/futures/bin/activate}"
UPDATE_SCRIPT="${UPDATE_SCRIPT:-/home/workspace/lab/Tree-Stock/futures/data/update_all_contracts_from_akshare.py}"
TREE_LOG="${TREE_LOG:-/home/workspace/lab/Tree-Stock/futures/data/update_all_contracts.log}"
SKIP_UPDATE="${SKIP_UPDATE:-0}"

ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }

echo "[$(ts)] final scheme start"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtualenv not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"

if [[ "${SKIP_UPDATE}" != "1" ]]; then
  echo "[$(ts)] update Tree-Stock contracts (today)"
  python3 "${UPDATE_SCRIPT}" --no-sleep >> "${TREE_LOG}" 2>&1
  echo "[$(ts)] contracts updated"
else
  echo "[$(ts)] skip contract update (SKIP_UPDATE=1)"
fi

cd "${SCRIPT_DIR}"
echo "[$(ts)] blotter"
python3 "${SCRIPT_DIR}/pos64_rsi_overlay_2026_blotter.py"
echo "[$(ts)] final scheme done"
