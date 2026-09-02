#!/usr/bin/env bash
set -Eeuo pipefail

# Frozen OOD suite v1. Do not change conditions after inspecting its returns.
# Seeds 90000--90199 are disjoint from training and all earlier validation suites.
ROOT="${ROOT:-/root/autodl-tmp/on-policy}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
RESULTS="onpolicy/scripts/results/MPE/simple_spread"
EVAL_START=90000
EVAL_EPISODES=200
SUITE=robust_ood_v1
LOG_DIR="$RESULTS/${SUITE}_logs_5seed"
MANIFEST="onpolicy/scripts/eval/robust_ood_v1_manifest.json"
MANIFEST_SHA256="461242c4742a9e224bc2e8cb5fd01a0209cb9a7466667cc4b4981b7a9856d87a"

cd "$ROOT"
mkdir -p "$LOG_DIR"
printf '%s  %s\n' "$MANIFEST_SHA256" "$MANIFEST" | sha256sum --check --status || {
  echo "OOD suite manifest changed after freeze" >&2
  exit 1
}

for train_seed in 1 2 3 4 5; do
  if [[ "$train_seed" == 1 ]]; then
    b4_actor="$RESULTS/selective_mappo/b4_selector_1m/run1/models/actor.pt"
  else
    b4_actor="$RESULTS/selective_mappo/b4_selector_1m_seed${train_seed}/run1/models/actor.pt"
  fi
  robust_b0="$RESULTS/rmappo/robust_b0_20m_seed${train_seed}/run1/models/actor.pt"
  robust_b2="$RESULTS/ua_rep_mappo/robust_b2_20m_seed${train_seed}/run1/models/actor.pt"
  output="$RESULTS/headroom_b4/${SUITE}_trainseed${train_seed}_seed${EVAL_START}_${EVAL_EPISODES}"
  for actor in "$b4_actor" "$robust_b0" "$robust_b2"; do
    [[ -s "$actor" ]] || { echo "missing actor: $actor" >&2; exit 1; }
  done
  if [[ -s "$output/episode_results.csv" && -s "$output/summary.json" ]]; then
    echo "SKIP complete OOD evaluation trainseed${train_seed}"
    continue
  fi
  [[ ! -e "$output" ]] || {
    echo "incomplete output exists: $output" >&2
    exit 1
  }
  echo "START OOD evaluation trainseed${train_seed}"
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
    --noise_stds 0.15 0.25 0.40 --mask_probs 0.40 0.60 --delay_steps 4 \
    --composite_conditions \
      noise=0.20+mask=0.30 \
      noise=0.20+delay=2 \
      mask=0.30+delay=2 \
    2>&1 | tee "$LOG_DIR/trainseed${train_seed}.log"
  [[ -s "$output/episode_results.csv" && -s "$output/summary.json" ]] || {
    echo "incomplete OOD evaluation: trainseed${train_seed}" >&2
    exit 1
  }
  echo "DONE OOD evaluation trainseed${train_seed}"
done

echo "ROBUST_OOD_V1_EVAL_5SEED_COMPLETE"
