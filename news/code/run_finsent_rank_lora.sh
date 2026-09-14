#!/usr/bin/env bash
# Financial-Sentiment Qwen rank-LoRA pipeline
set -euo pipefail

PY="${PY:-/home/env/futures/bin/python}"
ROOT="/home/workspace/lab/UniFutures/news"
RL="$ROOT/code/train/rank_lora"
BT="$ROOT/code/backtest"
DATA="$ROOT/data/sentiment/rank_lora"
RUNS="$DATA/runs"
LOG="$DATA/pipeline.log"
RESULT="$ROOT/data/results/rank_lora"

mkdir -p "$DATA" "$RUNS" "$RESULT"
exec > >(tee -a "$LOG") 2>&1

echo "===== $(date -Is) finsent rank lora start ====="
cd "$RL"

echo "----- [1] build multi-horizon targets -----"
"$PY" build_multihorizon_targets.py --out-dir "$DATA"

echo "----- [2] frozen baseline Rank IC -----"
"$PY" train_encoder.py --frozen-baseline --out "$RUNS/encoder_baseline"

echo "----- [3] LoRA 1-step smoke (16k) -----"
"$PY" train_encoder.py --smoke --out "$RUNS/encoder_smoke" --max-rows 1

echo "----- [4] LoRA full train -----"
"$PY" train_encoder.py --out "$RUNS/encoder" --epochs 2 --grad-accum 4 --max-group-docs 4 --lr 1e-4

VERDICT=$(python3 - <<'PY'
import json
from pathlib import Path
m=json.loads(Path("/home/workspace/lab/UniFutures/news/data/sentiment/rank_lora/runs/encoder/meta.json").read_text())
print(m.get("verdict","REGRESSION"))
print(m.get("post",{}).get("ic_mean",0), m.get("baseline",{}).get("ic_mean",0), m.get("delta",0))
PY
)
echo "encoder verdict lines:"
echo "$VERDICT"
VNAME=$(echo "$VERDICT" | head -1)
DELTA=$(echo "$VERDICT" | awk 'NR==2{print $3}')

# Gate: compare LoRA full-val IC to saved frozen baseline (not capped in-run baseline)
"$PY" - <<'PY'
import json, sys
from pathlib import Path
root = Path("/home/workspace/lab/UniFutures/news/data/sentiment/rank_lora/runs")
m = json.loads((root / "encoder" / "meta.json").read_text())
bpath = root / "encoder_baseline" / "meta.json"
frozen = json.loads(bpath.read_text())["baseline"]["ic_mean"] if bpath.exists() else m["baseline"]["ic_mean"]
post = float(m.get("post", {}).get("ic_mean", -1))
best = float(m.get("best", {}).get("ic_mean", post))
delta = post - frozen
print(f"GATE check post={post:.4f} best_capped={best:.4f} frozen={frozen:.4f} delta={delta:+.4f}")
gate_path = root / "gate.json"
gate = {"post": post, "frozen": frozen, "delta": delta, "pass": post > frozen}
gate_path.write_text(json.dumps(gate, indent=2))
if post <= frozen:
    print("GATE:STOP LoRA full-val Rank IC did not beat frozen baseline")
    sys.exit(3)
print("GATE:PASS")
PY
GATE_RC=$?
USE_LORA_EMBEDS=1
if [[ "$GATE_RC" -ne 0 ]]; then
  echo "[info] continuing with frozen-embed aggregators + LoRA cache for diagnostics"
  USE_LORA_EMBEDS=0
fi

echo "----- [5] cache adapter embeddings -----"
"$PY" cache_adapter_embeddings.py \
  --adapter "$RUNS/encoder/best/adapter" \
  --out-dir "$DATA/doc_embeddings" \
  --tag finsent_lora

echo "----- [6] aggregator smoke on frozen embeds -----"
"$PY" train_aggregator.py --smoke --tag frozen_smoke \
  --embeds "$ROOT/data/sentiment/hierarchical/doc_embeddings/finsent__final_last.npy" \
  --aggregator exp linear bigru_pos

echo "----- [7] sanity checks -----"
"$PY" sanity_checks.py \
  --embeds "$DATA/doc_embeddings/finsent_lora.npy" \
  --out "$RUNS/sanity.json" \
  --model "$RUNS/aggregators/frozen_smoke/exp/seed_0/model.pt" || {
  echo "GATE:STOP sanity failed"
  exit 4
}

echo "----- [8] full aggregators 3 seeds -----"
if [[ "$USE_LORA_EMBEDS" -eq 1 ]]; then
  "$PY" train_aggregator.py --tag lora \
    --embeds "$DATA/doc_embeddings/finsent_lora.npy" \
    --aggregator exp linear bigru_pos \
    --seeds 0 1 2 --epochs 5
  COMPARE_TAG=lora
else
  "$PY" train_aggregator.py --tag frozen_fallback \
    --embeds "$ROOT/data/sentiment/hierarchical/doc_embeddings/finsent__final_last.npy" \
    --aggregator exp linear bigru_pos \
    --seeds 0 1 2 --epochs 5
  COMPARE_TAG=frozen_fallback
fi

echo "----- [9] backtest + compare -----"
"$PY" "$BT/compare_rank_aggregators.py" --tag "$COMPARE_TAG" --out "$RESULT/comparison.csv"

echo "===== $(date -Is) finsent rank lora done ====="
