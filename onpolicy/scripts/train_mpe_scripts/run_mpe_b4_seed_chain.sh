#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/autodl-tmp/on-policy}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  cat <<'EOF'
Usage: run_mpe_b4_seed_chain.sh SEED [--dry-run] [--resume] [--train-only]

Runs one independent B4 training chain in this order:
  B0 20M -> B2 20M -> frozen detector 1M -> freeze verification
  -> B4 warm-up 100K -> B4 1M -> freeze verification -> final evaluation

SEED must be >= 2. Existing completed stages are reused only with --resume.
The evaluation uses the locked, previously unseen seeds 40000-40199.
EOF
}

[[ $# -ge 1 ]] || { usage; exit 2; }
SEED="$1"
shift
[[ "$SEED" =~ ^[0-9]+$ ]] && (( SEED >= 2 )) || {
  echo "ERROR: SEED must be an integer >= 2" >&2
  exit 2
}

DRY_RUN=0
RESUME=0
RUN_EVAL=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --resume) RESUME=1 ;;
    --train-only) RUN_EVAL=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

cd "$ROOT"

EXPECTED_COMMIT="e14232e665dcfd9f7359bb98e98709a21c2e6e29"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
git merge-base --is-ancestor "$EXPECTED_COMMIT" HEAD || {
  echo "ERROR: required B4 base commit $EXPECTED_COMMIT is not an ancestor of $ACTUAL_COMMIT" >&2
  exit 1
}

RESULTS="onpolicy/scripts/results/MPE/simple_spread"
B0_EXP="check_seed${SEED}"
B2_EXP="b2_ua_rep_mappo_seed${SEED}"
DET_EXP="b2_detector_v5_frozen_1m_seed${SEED}"
B4_WARM_EXP="b4_selector_100k_seed${SEED}"
B4_EXP="b4_selector_1m_seed${SEED}"

B0_MODELS="$RESULTS/rmappo/$B0_EXP/run1/models"
B2_MODELS="$RESULTS/ua_rep_mappo/$B2_EXP/run1/models"
DET_MODELS="$RESULTS/ua_rep_mappo/$DET_EXP/run1/models"
B4_WARM_MODELS="$RESULTS/selective_mappo/$B4_WARM_EXP/run1/models"
B4_MODELS="$RESULTS/selective_mappo/$B4_EXP/run1/models"
EVAL_DIR="$RESULTS/headroom_b4/formal_trainseed${SEED}_seed40000_200"
LOG_DIR="$RESULTS/multiseed_logs/trainseed${SEED}"
mkdir -p "$LOG_DIR"

run() {
  if (( DRY_RUN )); then
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_stage() {
  local name="$1"
  local models="$2"
  shift 2
  if [[ -s "$models/actor.pt" && -s "$models/critic.pt" ]]; then
    if (( RESUME )); then
      echo "SKIP $name: completed checkpoint exists at $models"
      return
    fi
    echo "ERROR: $name output already exists; use --resume to validate and reuse it" >&2
    exit 1
  fi
  if [[ -e "${models%/run1/models}" ]]; then
    echo "ERROR: incomplete output directory exists for $name:" >&2
    echo "  ${models%/run1/models}" >&2
    echo "Move it aside explicitly before restarting this stage." >&2
    exit 1
  fi
  echo "START $name"
  if (( DRY_RUN )); then
    run "$@"
  else
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    [[ -s "$models/actor.pt" && -s "$models/critic.pt" ]] || {
      echo "ERROR: $name finished without actor.pt and critic.pt" >&2
      exit 1
    }
  fi
  echo "DONE $name"
}

COMMON=(
  --env_name MPE --scenario_name simple_spread
  --num_agents 3 --num_landmarks 3 --seed "$SEED"
  --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1
  --episode_length 25 --ppo_epoch 10 --use_ReLU --gain 0.01
  --use_wandb
)

echo "B4 multi-seed chain: seed=$SEED commit=$ACTUAL_COMMIT gpu=$GPU"
run "$PYTHON" tests/test_selective_mappo.py

run_stage b0_20m "$B0_MODELS" \
  env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  onpolicy/scripts/train/train_mpe.py \
  --algorithm_name rmappo --experiment_name "$B0_EXP" \
  "${COMMON[@]}" --num_env_steps 20000000 \
  --lr 7e-4 --critic_lr 7e-4

run_stage b2_20m "$B2_MODELS" \
  env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  onpolicy/scripts/train/train_mpe.py \
  --algorithm_name ua_rep_mappo --experiment_name "$B2_EXP" \
  "${COMMON[@]}" --num_env_steps 20000000 \
  --lr 7e-4 --critic_lr 7e-4 \
  --cstm_latent_dim 32 --cstm_aux_coef 0.1 --cstm_num_heads 3 \
  --cstm_bootstrap_prob 0.8 --cstm_use_uncertainty_feature

run_stage detector_1m "$DET_MODELS" \
  env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  onpolicy/scripts/train/train_mpe.py \
  --algorithm_name ua_rep_mappo --experiment_name "$DET_EXP" \
  "${COMMON[@]}" --num_env_steps 1000000 \
  --lr 7e-4 --critic_lr 7e-4 --model_dir "$B2_MODELS" \
  --cstm_latent_dim 32 --cstm_aux_coef 0.1 --cstm_num_heads 3 \
  --cstm_bootstrap_prob 0.5 --cstm_random_prior_scale 0.5 \
  --cstm_use_separate_detector --cstm_detector_aux_coef 0.1 \
  --cstm_detector_only --cstm_use_uncertainty_feature \
  --cstm_ood_rank_coef 1.0 --cstm_ood_rank_margin 0.02 \
  --cstm_ood_noise_std 0.2 --cstm_ood_mask_prob 0.3 \
  --cstm_ood_delay_prob 0.3 --cstm_uncertainty_cal_coef 1.0 \
  --cstm_uncertainty_target_scale 0.05 --cstm_uncertainty_corr_coef 0.01

if (( ! DRY_RUN )); then
  run "$PYTHON" onpolicy/scripts/tools/verify_b4_seed_chain.py detector \
    --base-actor "$B2_MODELS/actor.pt" \
    --trained-actor "$DET_MODELS/actor.pt" \
    --base-critic "$B2_MODELS/critic.pt" \
    --trained-critic "$DET_MODELS/critic.pt"
fi

run_stage b4_100k "$B4_WARM_MODELS" \
  env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  onpolicy/scripts/train/train_mpe.py \
  --algorithm_name selective_mappo --experiment_name "$B4_WARM_EXP" \
  "${COMMON[@]}" --num_env_steps 100000 --data_chunk_length 10 \
  --lr 1e-4 --critic_lr 7e-4 --model_dir "$DET_MODELS" \
  --cstm_b0_model_dir "$B0_MODELS" \
  --cstm_latent_dim 32 --cstm_num_heads 3 --cstm_bootstrap_prob 0.5 \
  --cstm_random_prior_scale 0.5 --cstm_use_separate_detector \
  --cstm_use_uncertainty_feature --cstm_selector_hidden_dim 64 \
  --cstm_selector_initial_threshold 0.03 --cstm_selector_initial_scale 200 \
  --cstm_selector_fallback_cost 0.005 --cstm_selector_coverage_coef 0.01 \
  --cstm_selector_coverage_target 0.5 --cstm_selector_entropy_coef 0.001 \
  --cstm_selector_clean_probability 0.25

run_stage b4_1m "$B4_MODELS" \
  env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  onpolicy/scripts/train/train_mpe.py \
  --algorithm_name selective_mappo --experiment_name "$B4_EXP" \
  "${COMMON[@]}" --num_env_steps 1000000 --data_chunk_length 10 \
  --lr 1e-4 --critic_lr 7e-4 --use_linear_lr_decay \
  --model_dir "$B4_WARM_MODELS" --cstm_b0_model_dir "$B0_MODELS" \
  --cstm_latent_dim 32 --cstm_num_heads 3 --cstm_bootstrap_prob 0.5 \
  --cstm_random_prior_scale 0.5 --cstm_use_separate_detector \
  --cstm_use_uncertainty_feature --cstm_selector_hidden_dim 64 \
  --cstm_selector_initial_threshold 0.03 --cstm_selector_initial_scale 200 \
  --cstm_selector_fallback_cost 0.005 --cstm_selector_coverage_coef 0.01 \
  --cstm_selector_coverage_target 0.5 --cstm_selector_entropy_coef 0.001 \
  --cstm_selector_clean_probability 0.25

if (( ! DRY_RUN )); then
  run "$PYTHON" onpolicy/scripts/tools/verify_b4_seed_chain.py b4 \
    --b0-actor "$B0_MODELS/actor.pt" \
    --b2-actor "$DET_MODELS/actor.pt" \
    --trained-actor "$B4_MODELS/actor.pt"
fi

if (( RUN_EVAL )); then
  if [[ -s "$EVAL_DIR/episode_results.csv" ]]; then
    if (( RESUME )); then
      echo "SKIP evaluation: $EVAL_DIR/episode_results.csv exists"
    else
      echo "ERROR: evaluation output exists; use --resume to reuse it" >&2
      exit 1
    fi
  else
    [[ ! -e "$EVAL_DIR" ]] || {
      echo "ERROR: incomplete evaluation directory exists: $EVAL_DIR" >&2
      exit 1
    }
    echo "START formal evaluation seeds 40000-40199"
    run env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
      onpolicy/scripts/eval/eval_mpe_b4.py \
      --algorithm_name selective_mappo --scenario_name simple_spread \
      --num_agents 3 --num_landmarks 3 --episode_length 25 \
      --eval_episodes 200 --eval_batch_size 25 --eval_seed_start 40000 \
      --seed "$SEED" --checkpoint_activation tanh --b3_threshold 0.03 \
      --b4_actor "$B4_MODELS/actor.pt" --output_dir "$EVAL_DIR" \
      --noise_stds 0.05 0.10 0.20 0.30 \
      --mask_probs 0.10 0.20 0.30 0.50 --delay_steps 1 2 3 \
      --write_step_log
    if (( ! DRY_RUN )); then
      [[ -s "$EVAL_DIR/episode_results.csv" && -s "$EVAL_DIR/steps.csv" ]] || {
        echo "ERROR: evaluation outputs are incomplete" >&2
        exit 1
      }
      run "$PYTHON" onpolicy/scripts/tools/verify_b4_seed_chain.py gate-curve \
        --steps "$EVAL_DIR/steps.csv" --output "$EVAL_DIR/gate_curve.csv"
    fi
    echo "DONE formal evaluation"
  fi
fi

echo "B4_SEED_CHAIN_COMPLETE seed=$SEED"
echo "B4 checkpoint: $B4_MODELS/actor.pt"
(( RUN_EVAL )) && echo "Evaluation: $EVAL_DIR"
