#!/usr/bin/env bash
set -Eeuo pipefail

# Train 2M and 5M selectors independently from each seed's same 100K warm
# checkpoint.  The legacy --use_ReLU flag selects Tanh (store_false).

ROOT="${ROOT:-/root/autodl-tmp/on-policy}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
RESULTS="onpolicy/scripts/results/MPE/simple_spread"
LOG_DIR="$RESULTS/selector_length_logs"

cd "$ROOT"
mkdir -p "$LOG_DIR"

run_stage() {
  local name="$1"
  local models="$2"
  shift 2
  if [[ -s "$models/actor.pt" && -s "$models/critic.pt" ]]; then
    echo "SKIP $name: completed checkpoint exists"
    return
  fi
  [[ ! -e "${models%/run1/models}" ]] || {
    echo "ERROR: incomplete output exists for $name" >&2
    exit 1
  }
  echo "START $name"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  [[ -s "$models/actor.pt" && -s "$models/critic.pt" ]] || {
    echo "ERROR: missing checkpoint after $name" >&2
    exit 1
  }
  echo "DONE $name"
}

for seed in 1 2 3; do
  if [[ "$seed" == 1 ]]; then
    warm_models="$RESULTS/selective_mappo/b4_selector_100k/run1/models"
    b0_models="$RESULTS/rmappo/check/run7/models"
  else
    warm_models="$RESULTS/selective_mappo/b4_selector_100k_seed${seed}/run1/models"
    b0_models="$RESULTS/rmappo/check_seed${seed}/run1/models"
  fi
  for millions in 2 5; do
    steps=$((millions * 1000000))
    exp="b4_selector_length_${millions}m_seed${seed}"
    models="$RESULTS/selective_mappo/$exp/run1/models"
    run_stage "$exp" "$models" \
      env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
      onpolicy/scripts/train/train_mpe.py \
      --env_name MPE --algorithm_name selective_mappo \
      --experiment_name "$exp" --scenario_name simple_spread \
      --num_agents 3 --num_landmarks 3 --seed "$seed" \
      --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 \
      --episode_length 25 --num_env_steps "$steps" --ppo_epoch 10 \
      --data_chunk_length 10 --use_ReLU --gain 0.01 \
      --lr 1e-4 --critic_lr 7e-4 --use_linear_lr_decay \
      --model_dir "$warm_models" --cstm_b0_model_dir "$b0_models" \
      --cstm_latent_dim 32 --cstm_num_heads 3 \
      --cstm_bootstrap_prob 0.5 --cstm_random_prior_scale 0.5 \
      --cstm_use_separate_detector --cstm_use_uncertainty_feature \
      --cstm_selector_hidden_dim 64 \
      --cstm_selector_initial_threshold 0.03 \
      --cstm_selector_initial_scale 200 \
      --cstm_selector_fallback_cost 0.005 \
      --cstm_selector_coverage_coef 0.01 \
      --cstm_selector_coverage_target 0.5 \
      --cstm_selector_entropy_coef 0.001 \
      --cstm_selector_clean_probability 0.25 --use_wandb
  done
done

echo "B4_SELECTOR_LENGTH_3SEED_COMPLETE"
