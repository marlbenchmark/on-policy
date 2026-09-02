#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/on-policy
PYTHON=/root/miniconda3/bin/python
GPU=${GPU:-0}
RESULTS=onpolicy/scripts/results/MPE/simple_spread
OUT_ROOT="$RESULTS/headroom_b4/observation_repair_final_v1"
mkdir -p "$OUT_ROOT"

for train_seed in 1 2 3 4 5; do
  if [[ "$train_seed" == 1 ]]; then
    actor="$RESULTS/selective_mappo/b4_selector_1m/run1/models/actor.pt"
  else
    actor="$RESULTS/selective_mappo/b4_selector_1m_seed${train_seed}/run1/models/actor.pt"
  fi
  output="$OUT_ROOT/trainseed${train_seed}_seed93000_200"
  if [[ -s "$output/summary.json" && -s "$output/episode_results.csv" ]]; then
    echo "SKIP complete trainseed${train_seed}"
    continue
  fi
  [[ -s "$actor" ]] || { echo "missing actor: $actor" >&2; exit 2; }
  [[ ! -e "$output" ]] || {
    echo "incomplete output exists; inspect before rerun: $output" >&2
    exit 3
  }
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    onpolicy/scripts/eval/eval_mpe_observation_repair.py \
    --algorithm_name selective_mappo --scenario_name simple_spread \
    --num_agents 3 --num_landmarks 3 --episode_length 25 \
    --eval_episodes 200 --eval_batch_size 25 \
    --eval_seed_start 93000 --eval_protocol final --seed "$train_seed" \
    --checkpoint_activation tanh --b3_threshold 0.03 \
    --risk_soft_scale 200 --repair_smooth_alpha 0.50 \
    --repair_max_velocity 0.20 --repair_modes risk_smooth \
    --b4_actor "$actor" --output_dir "$output" \
    --noise_stds 0.30 0.40 --mask_probs 0.50 0.60 \
    --delay_steps 3 4 \
    --composite_conditions noise=0.20+mask=0.30 \
      noise=0.20+delay=2 mask=0.30+delay=2 \
    2>&1 | tee "$OUT_ROOT/trainseed${train_seed}.log"
done

echo OBSERVATION_REPAIR_FINAL_5SEED_COMPLETE
