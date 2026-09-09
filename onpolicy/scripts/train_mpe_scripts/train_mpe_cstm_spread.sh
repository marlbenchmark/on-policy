#!/bin/sh
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="cstm_mappo"
exp="b1_liam_mappo"
seed_max=1

for seed in `seq ${seed_max}`; do
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/../train/train_mpe.py" \
        --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
        --scenario_name ${scenario} --num_agents ${num_agents} \
        --num_landmarks ${num_landmarks} --seed ${seed} \
        --n_training_threads 1 --n_rollout_threads 128 \
        --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
        --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
        --cstm_latent_dim 32 --cstm_aux_coef 0.1 --use_wandb
done
