#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_final_scheme.sh"
LOG_FILE="${SCRIPT_DIR}/run_final_scheme.log"
VENV_ACTIVATE="${VENV_ACTIVATE:-/home/env/futures/bin/activate}"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Run script not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtualenv activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi

chmod +x "${RUN_SCRIPT}"

# After 19:30 contract pull; skip re-download.
CRON_CMD="cd \"${SCRIPT_DIR}\" && /usr/bin/env bash -lc 'source \"${VENV_ACTIVATE}\" && SKIP_UPDATE=1 bash \"${RUN_SCRIPT}\" >> \"${LOG_FILE}\" 2>&1'"
CRON_EXPR="50 19 * * *"
CRON_LINE="${CRON_EXPR} ${CRON_CMD}"

TMP_CRON="$(mktemp)"
crontab -l 2>/dev/null | awk '!/UniFutures\/code\/experiment\/run_final_scheme\.sh/' > "${TMP_CRON}" || true
echo "${CRON_LINE}" >> "${TMP_CRON}"
crontab "${TMP_CRON}"
rm -f "${TMP_CRON}"

echo "Installed cron job:"
echo "${CRON_LINE}"
echo "Current crontab:"
crontab -l
