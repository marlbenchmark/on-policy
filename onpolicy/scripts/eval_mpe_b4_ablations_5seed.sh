#!/usr/bin/env bash
set -Eeuo pipefail

# Phase-A selector ablation on an evaluation pool disjoint from threshold
# selection and later selector-length validation seeds.

ROOT="${ROOT:-/root/autodl-tmp/on-policy}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
RESULTS="onpolicy/scripts/results/MPE/simple_spread"
EVAL_START=50000
EVAL_EPISODES=200

cd "$ROOT"
mkdir -p "$RESULTS/selector_ablation_logs"

for train_seed in 1 2 3 4 5; do
  if [[ "$train_seed" == 1 ]]; then
    actor="$RESULTS/selective_mappo/b4_selector_1m/run1/models/actor.pt"
  else
    actor="$RESULTS/selective_mappo/b4_selector_1m_seed${train_seed}/run1/models/actor.pt"
  fi
  output="$RESULTS/headroom_b4/selector_ablations_trainseed${train_seed}_seed${EVAL_START}_${EVAL_EPISODES}"
  log="$RESULTS/selector_ablation_logs/trainseed${train_seed}.log"
  [[ -s "$actor" ]] || { echo "missing actor: $actor" >&2; exit 1; }
  if [[ -s "$output/episode_results.csv" && -s "$output/summary.json" ]]; then
    echo "SKIP trainseed${train_seed}: completed output exists"
    continue
  fi
  [[ ! -e "$output" ]] || {
    echo "incomplete output exists: $output" >&2
    exit 1
  }
  echo "START selector ablations trainseed${train_seed}"
  env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    onpolicy/scripts/eval/eval_mpe_b4.py \
    --algorithm_name selective_mappo --scenario_name simple_spread \
    --num_agents 3 --num_landmarks 3 --episode_length 25 \
    --eval_episodes "$EVAL_EPISODES" --eval_batch_size 25 \
    --eval_seed_start "$EVAL_START" --seed "$train_seed" \
    --checkpoint_activation tanh --b3_threshold 0.03 \
    --risk_soft_scale 200 --selector_ablations \
    --b4_actor "$actor" --output_dir "$output" \
    --noise_stds 0.05 0.10 0.20 0.30 \
    --mask_probs 0.10 0.20 0.30 0.50 --delay_steps 1 2 3 \
    2>&1 | tee "$log"
  [[ -s "$output/episode_results.csv" && -s "$output/summary.json" ]] || {
    echo "incomplete evaluation: trainseed${train_seed}" >&2
    exit 1
  }
  echo "DONE selector ablations trainseed${train_seed}"
done

echo "B4_SELECTOR_ABLATIONS_5SEED_COMPLETE"
