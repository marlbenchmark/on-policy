#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for the already-running seed4--5 extension, then execute both frozen suites.
ROOT="${ROOT:-/root/autodl-tmp/on-policy}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
RESULTS="$ROOT/onpolicy/scripts/results/MPE/simple_spread"
TRAIN_LOG="$RESULTS/robust_baseline_logs/robust_seed4_5_master.log"
PIPELINE_LOG="$RESULTS/robust_baseline_logs/robust_5seed_pipeline.log"

while ! grep -q "ROBUST_BASELINES_SEED4_5_COMPLETE" "$TRAIN_LOG"; do
  if grep -q "ERROR:" "$TRAIN_LOG"; then
    echo "seed4--5 training reported an error" >&2
    exit 1
  fi
  sleep 60
done

cd "$ROOT"
bash onpolicy/scripts/eval/eval_mpe_robust_baselines_5seed.sh
"$PYTHON" onpolicy/scripts/eval/summarize_mpe_robust_5seed.py \
  --results_root "$RESULTS/headroom_b4" \
  --input_template 'robust_comparison_trainseed{seed}_seed60000_200' \
  --suite id --output_dir "$RESULTS/headroom_b4/robust_comparison_5seed_summary"
bash onpolicy/scripts/eval/eval_mpe_robust_ood_5seed.sh
"$PYTHON" onpolicy/scripts/eval/summarize_mpe_robust_5seed.py \
  --results_root "$RESULTS/headroom_b4" \
  --input_template 'robust_ood_v1_trainseed{seed}_seed90000_200' \
  --suite ood --output_dir "$RESULTS/headroom_b4/robust_ood_v1_5seed_summary"

echo "ROBUST_5SEED_PIPELINE_COMPLETE" | tee -a "$PIPELINE_LOG"
