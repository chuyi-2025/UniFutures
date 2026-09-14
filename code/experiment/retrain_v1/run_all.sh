#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/workspace/lab/UniFutures
PY=/home/env/futures/bin/python
cd "$ROOT"
export PYTHONUNBUFFERED=1

mkdir -p data/results/retrain_v1
STATUS=data/results/retrain_v1/serial_status.tsv
MASTER_LOG=data/results/retrain_v1/serial_nohup.log

exec > >(tee -a "$MASTER_LOG") 2>&1

if [[ ! -f "$STATUS" ]]; then
  printf "scheme\tstatus\tstarted_at\tfinished_at\n" > "$STATUS"
fi

run_scheme() {
  local scheme="$1"
  local runner="$2"
  local summary="data/results/retrain_v1/${scheme}/summary.csv"
  if [[ -s "$summary" ]]; then
    echo "[skip] $scheme already complete: $summary"
    return 0
  fi

  local started
  started="$(date --iso-8601=seconds)"
  echo "[start] $scheme at $started"
  if "$PY" -u "$runner" --scheme "$scheme"; then
    local finished
    finished="$(date --iso-8601=seconds)"
    printf "%s\tcompleted\t%s\t%s\n" "$scheme" "$started" "$finished" >> "$STATUS"
    echo "[complete] $scheme at $finished"
  else
    local finished
    finished="$(date --iso-8601=seconds)"
    printf "%s\tfailed\t%s\t%s\n" "$scheme" "$started" "$finished" >> "$STATUS"
    echo "[failed] $scheme at $finished; continuing serial queue"
  fi
}

echo "[queue] 32 schemes; completed summaries are skipped"
for scheme in X01 X02 X03 X04 X05 X06 X07 X08; do
  run_scheme "$scheme" code/experiment/retrain_v1/run_xgb_schemes.py
done
for scheme in G01 G02 G03 G04 G05 G06 G07 G08; do
  run_scheme "$scheme" code/experiment/retrain_v1/run_gaf_schemes.py
done
for scheme in S01 S02 S03 S04 S05 S06 S07 S08; do
  run_scheme "$scheme" code/experiment/retrain_v1/run_sent_schemes.py
done
for scheme in P01 P02 P03 P04 P05 P06 P07 P08; do
  run_scheme "$scheme" code/experiment/retrain_v1/run_ppo_schemes.py
done

echo "[compare] aggregate all available results"
"$PY" -u code/experiment/retrain_v1/compare_retrain.py || true
echo "[done] serial queue finished at $(date --iso-8601=seconds)"
