#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/env/futures/bin/activate

echo "[build] infer features (window-only, no forward filter)"
python3 "${SCRIPT_DIR}/build_features.py"

echo "[done] features ready"
