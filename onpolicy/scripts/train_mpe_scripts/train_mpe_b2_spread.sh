#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python \
  "$SCRIPT_DIR/../train/train_mpe.py" \
  --env_name MPE \
  --algorithm_name ua_rep_mappo \
  --experiment_name b2_ua_rep_mappo \
  --scenario_name simple_spread \
  --num_agents 3 \
  --num_landmarks 3 \
  --seed 1 \
  --n_training_threads 1 \
  --n_rollout_threads 128 \
  --num_mini_batch 1 \
  --episode_length 25 \
  --num_env_steps 20000000 \
  --ppo_epoch 10 \
  --use_ReLU \
  --gain 0.01 \
  --lr 7e-4 \
  --critic_lr 7e-4 \
  --cstm_latent_dim 32 \
  --cstm_aux_coef 0.1 \
  --cstm_num_heads 3 \
  --cstm_bootstrap_prob 0.8 \
  --cstm_use_uncertainty_feature \
  --use_wandb
