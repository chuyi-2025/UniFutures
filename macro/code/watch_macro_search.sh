#!/usr/bin/env bash
# Auto-resume macro account search until wall clock end.
set -euo pipefail
ROOT=/home/workspace/lab/UniFutures
OUT=$ROOT/macro/data/results/search_account
END_EPOCH=${1:?usage: watch_macro_search.sh END_EPOCH}
LOG=$OUT/watch.log
cd "$ROOT"
echo "[$(date)] watch until $(date -d @$END_EPOCH)" | tee -a "$LOG"
while (( $(date +%s) < END_EPOCH )); do
  left=$(( END_EPOCH - $(date +%s) ))
  mins=$(python3 -c "print(max(1, round($left/60, 2)))")
  echo "[$(date)] start resume minutes=$mins" | tee -a "$LOG"
  setsid python3 -u macro/code/search_macro_account.py --minutes "$mins" --resume --skip-phase-a \
    </dev/null >>"$OUT/run_watch.log" 2>&1 &
  pid=$!
  echo "$pid" > "$OUT/watch.pid"
  wait "$pid" || true
  ec=$?
  echo "[$(date)] exit=$ec — sleep 3s then resume if time left" | tee -a "$LOG"
  # finalize checkpoint every crash
  python3 - <<'PY' || true
from pathlib import Path
import json, pandas as pd, numpy as np, sys
sys.path.insert(0,'macro/code')
from search_macro_account import OUT, CAPITAL, START, FACTORS, BookEngine
cp=OUT/'checkpoint_schemes.csv'
if not cp.exists() or cp.stat().st_size<100: raise SystemExit
res=pd.read_csv(cp).sort_values(['quality','sharpe'],ascending=False)
res=res.drop_duplicates(subset=['factor','mode','thr','direction','hold','symbols','max_names','max_lots','margin_cap','util'],keep='first')
robust=res[(res.sharpe>=0.8)&(res.oos_both_pos)&(res.max_dd.fillna(-1)>=-0.20)&(res.ir_mean.fillna(-9)>=0.08)]
soft=res[(res.sharpe>=0.6)&(res.oos_both_pos)&(res.max_dd.fillna(-1)>=-0.25)]
res.to_csv(OUT/'all_account_schemes.csv',index=False)
robust.to_csv(OUT/'robust_account.csv',index=False)
soft.to_csv(OUT/'soft_account.csv',index=False)
res.head(50).to_csv(OUT/'top50_account.csv',index=False)
print('checkpoint finalized', len(res), 'best_sh', res.sharpe.max(), 'best_q', res.quality.max())
PY
  sleep 3
done
echo "[$(date)] wall done" | tee -a "$LOG"
