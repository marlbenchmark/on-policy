#!/usr/bin/env bash
set -Eeuo pipefail

# The 70000-70499 pool is validation-only.  Do not use the reserved
# 80000-80199 final-test pool until one method has passed validation.

ROOT="${ROOT:-/root/autodl-tmp/on-policy}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
RESULTS="onpolicy/scripts/results/MPE/simple_spread"
EVAL_START=70000
EVAL_EPISODES=500
LOG_DIR="$RESULTS/selector_length_eval_logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"

for seed in 1 2 3; do
  for millions in 1 2 5; do
    if [[ "$millions" == 1 ]]; then
      if [[ "$seed" == 1 ]]; then
        actor="$RESULTS/selective_mappo/b4_selector_1m/run1/models/actor.pt"
      else
        actor="$RESULTS/selective_mappo/b4_selector_1m_seed${seed}/run1/models/actor.pt"
      fi
    else
      actor="$RESULTS/selective_mappo/b4_selector_length_${millions}m_seed${seed}/run1/models/actor.pt"
    fi
    output="$RESULTS/headroom_b4/selector_length_${millions}m_trainseed${seed}_validation_seed${EVAL_START}_${EVAL_EPISODES}"
    [[ -s "$actor" ]] || { echo "missing actor: $actor" >&2; exit 1; }
    if [[ -s "$output/episode_results.csv" && -s "$output/summary.json" ]]; then
      echo "SKIP ${millions}m trainseed${seed} validation"
      continue
    fi
    [[ ! -e "$output" ]] || {
      echo "incomplete output exists: $output" >&2
      exit 1
    }
    echo "START ${millions}m trainseed${seed} validation"
    env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
      onpolicy/scripts/eval/eval_mpe_b4.py \
      --algorithm_name selective_mappo --scenario_name simple_spread \
      --num_agents 3 --num_landmarks 3 --episode_length 25 \
      --eval_episodes "$EVAL_EPISODES" --eval_batch_size 25 \
      --eval_seed_start "$EVAL_START" --seed "$seed" \
      --checkpoint_activation tanh --b3_threshold 0.03 \
      --risk_soft_scale 200 --selector_ablations \
      --b4_actor "$actor" --output_dir "$output" \
      --noise_stds 0.05 0.10 0.20 0.30 \
      --mask_probs 0.10 0.20 0.30 0.50 --delay_steps 1 2 3 \
      2>&1 | tee "$LOG_DIR/${millions}m_trainseed${seed}.log"
    [[ -s "$output/episode_results.csv" && -s "$output/summary.json" ]] || {
      echo "incomplete selector validation: ${millions}m seed${seed}" >&2
      exit 1
    }
    echo "DONE ${millions}m trainseed${seed} validation"
  done
done

echo "B4_SELECTOR_LENGTH_VALIDATION_3SEED_COMPLETE"
