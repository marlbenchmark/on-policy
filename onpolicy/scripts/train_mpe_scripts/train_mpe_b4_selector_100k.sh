#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)

cd "$ROOT"
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python \
  onpolicy/scripts/train/train_mpe.py \
  --env_name MPE \
  --algorithm_name selective_mappo \
  --experiment_name b4_selector_100k \
  --scenario_name simple_spread \
  --num_agents 3 --num_landmarks 3 --seed 1 \
  --n_training_threads 1 --n_rollout_threads 128 \
  --num_mini_batch 1 --episode_length 25 --num_env_steps 100000 \
  --ppo_epoch 10 --data_chunk_length 10 \
  --use_ReLU --gain 0.01 --lr 1e-4 --critic_lr 7e-4 \
  --model_dir onpolicy/scripts/results/MPE/simple_spread/ua_rep_mappo/b2_detector_v5_frozen_1m/run1/models \
  --cstm_b0_model_dir onpolicy/scripts/results/MPE/simple_spread/rmappo/check/run7/models \
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
  --cstm_selector_clean_probability 0.25 \
  --use_wandb
