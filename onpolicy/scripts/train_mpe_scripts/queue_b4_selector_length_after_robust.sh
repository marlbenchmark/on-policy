#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/autodl-tmp/on-policy}"
ROBUST_PID="${1:-116081}"
RESULTS="onpolicy/scripts/results/MPE/simple_spread"
ROBUST_LOG="$RESULTS/robust_baseline_logs/master.log"

cd "$ROOT"
echo "WAIT robust pipeline pid=$ROBUST_PID"
while ! grep -q "ROBUST_BASELINES_3SEED_COMPLETE" "$ROBUST_LOG"; do
  if ! kill -0 "$ROBUST_PID" 2>/dev/null; then
    echo "ERROR: robust pipeline ended without completion marker" >&2
    exit 1
  fi
  sleep 60
done

echo "START selector length training"
bash onpolicy/scripts/train_mpe_scripts/run_b4_selector_length_3seed.sh
echo "START selector length validation"
bash onpolicy/scripts/eval_b4_selector_length_validation_3seed.sh
echo "SELECTOR_LENGTH_PIPELINE_COMPLETE"
