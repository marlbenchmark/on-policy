#!/usr/bin/env bash
set -Eeuo pipefail

# Resumable training-then-validation pipeline.  Individual stages skip only
# when both actor and critic checkpoints (or complete evaluation outputs) exist.

ROOT="${ROOT:-/root/autodl-tmp/on-policy}"

cd "$ROOT"
echo "START selector length training"
bash onpolicy/scripts/train_mpe_scripts/run_b4_selector_length_3seed.sh
echo "START selector length validation"
bash onpolicy/scripts/eval_b4_selector_length_validation_3seed.sh
echo "SELECTOR_LENGTH_PIPELINE_COMPLETE"
