#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for the explicit pipeline completion marker before reading CSV files;
# checkpoint existence alone is not a training-completion signal.

ROOT="${ROOT:-/root/autodl-tmp/on-policy}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
RESULTS="onpolicy/scripts/results/MPE/simple_spread"
PIPELINE_LOG="$RESULTS/selector_length_logs/queue_master.log"
SUMMARY_LOG="$RESULTS/selector_length_logs/summary.log"

cd "$ROOT"
echo "WAIT selector length validation"
while ! grep -q "SELECTOR_LENGTH_PIPELINE_COMPLETE" "$PIPELINE_LOG"; do
  if grep -q "ERROR:" "$PIPELINE_LOG"; then
    echo "ERROR: selector length pipeline reported a failure" >&2
    exit 1
  fi
  sleep 60
done

echo "START selector length summary"
"$PYTHON" onpolicy/scripts/eval/summarize_b4_selector_length.py 2>&1 | tee "$SUMMARY_LOG"
echo "SELECTOR_LENGTH_SUMMARY_COMPLETE"
