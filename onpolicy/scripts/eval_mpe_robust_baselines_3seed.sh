#!/usr/bin/env bash
set -Eeuo pipefail

# Locked robust-baseline comparison pool: eval seeds 60000-60199 are not used
# by the selector-length validation or its final-test reservation.

ROOT="${ROOT:-/root/autodl-tmp/on-policy}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
RESULTS="onpolicy/scripts/results/MPE/simple_spread"
EVAL_START=60000
EVAL_EPISODES=200
LOG_DIR="$RESULTS/robust_baseline_eval_logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"

for train_seed in 1 2 3; do
  if [[ "$train_seed" == 1 ]]; then
    b4_actor="$RESULTS/selective_mappo/b4_selector_1m/run1/models/actor.pt"
  else
    b4_actor="$RESULTS/selective_mappo/b4_selector_1m_seed${train_seed}/run1/models/actor.pt"
  fi
  robust_b0="$RESULTS/rmappo/robust_b0_20m_seed${train_seed}/run1/models/actor.pt"
  robust_b2="$RESULTS/ua_rep_mappo/robust_b2_20m_seed${train_seed}/run1/models/actor.pt"
  output="$RESULTS/headroom_b4/robust_comparison_trainseed${train_seed}_seed${EVAL_START}_${EVAL_EPISODES}"
  for actor in "$b4_actor" "$robust_b0" "$robust_b2"; do
    [[ -s "$actor" ]] || { echo "missing actor: $actor" >&2; exit 1; }
  done
  if [[ -s "$output/episode_results.csv" && -s "$output/summary.json" ]]; then
    echo "SKIP robust evaluation trainseed${train_seed}"
    continue
  fi
  [[ ! -e "$output" ]] || {
    echo "incomplete output exists: $output" >&2
    exit 1
  }
  echo "START robust evaluation trainseed${train_seed}"
  env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    onpolicy/scripts/eval/eval_mpe_b4.py \
    --algorithm_name selective_mappo --scenario_name simple_spread \
    --num_agents 3 --num_landmarks 3 --episode_length 25 \
    --eval_episodes "$EVAL_EPISODES" --eval_batch_size 25 \
    --eval_seed_start "$EVAL_START" --seed "$train_seed" \
    --checkpoint_activation tanh --b3_threshold 0.03 \
    --risk_soft_scale 200 --selector_ablations \
    --b4_actor "$b4_actor" --robust_b0_actor "$robust_b0" \
    --robust_b2_actor "$robust_b2" --output_dir "$output" \
    --noise_stds 0.05 0.10 0.20 0.30 \
    --mask_probs 0.10 0.20 0.30 0.50 --delay_steps 1 2 3 \
    2>&1 | tee "$LOG_DIR/trainseed${train_seed}.log"
  [[ -s "$output/episode_results.csv" && -s "$output/summary.json" ]] || {
    echo "incomplete robust evaluation: trainseed${train_seed}" >&2
    exit 1
  }
  echo "DONE robust evaluation trainseed${train_seed}"
done

echo "ROBUST_BASELINE_EVAL_3SEED_COMPLETE"
