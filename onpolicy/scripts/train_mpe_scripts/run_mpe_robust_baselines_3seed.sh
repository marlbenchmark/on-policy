#!/usr/bin/env bash
set -Eeuo pipefail

# Train corruption-exposed B0 and B2 baselines.  The legacy --use_ReLU flag is
# store_false in config.py, so passing it selects Tanh (matching evaluation).

ROOT="${ROOT:-/root/autodl-tmp/on-policy}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
RESULTS="onpolicy/scripts/results/MPE/simple_spread"
LOG_DIR="$RESULTS/robust_baseline_logs"
mkdir -p "$LOG_DIR"
cd "$ROOT"

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
  b0_exp="robust_b0_20m_seed${seed}"
  b2_exp="robust_b2_20m_seed${seed}"
  b0_models="$RESULTS/rmappo/$b0_exp/run1/models"
  b2_models="$RESULTS/ua_rep_mappo/$b2_exp/run1/models"
  common=(
    --env_name MPE --scenario_name simple_spread
    --num_agents 3 --num_landmarks 3 --seed "$seed"
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1
    --episode_length 25 --num_env_steps 20000000 --ppo_epoch 10
    --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4
    --mpe_use_corruption_training --use_wandb
  )
  run_stage "$b0_exp" "$b0_models" \
    env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    onpolicy/scripts/train/train_mpe.py --algorithm_name rmappo \
    --experiment_name "$b0_exp" "${common[@]}"
  run_stage "$b2_exp" "$b2_models" \
    env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    onpolicy/scripts/train/train_mpe.py --algorithm_name ua_rep_mappo \
    --experiment_name "$b2_exp" "${common[@]}" \
    --cstm_latent_dim 32 --cstm_aux_coef 0.1 --cstm_num_heads 3 \
    --cstm_bootstrap_prob 0.8 --cstm_use_uncertainty_feature
done

echo "ROBUST_BASELINES_3SEED_COMPLETE"
