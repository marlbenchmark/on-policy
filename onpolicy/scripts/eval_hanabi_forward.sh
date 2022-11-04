#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Full"
num_agents=2
algo="mappo"
exp="check"
seed_max=1
ulimit -n 22222

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python eval/eval_hanabi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --hanabi_name ${hanabi} --num_agents ${num_agents} --seed 1 --n_training_threads 128 --n_rollout_threads 1 \
    --n_eval_rollout_threads 1000 --num_mini_batch 4 --episode_length 100 --num_env_steps 10000000000000 --ppo_epoch 15 \
    --gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --use_eval --use_wandb --use_recurrent_policy \
    --entropy_coef 0.015 --model_dir "xxx"
    echo "training is done!"
done
