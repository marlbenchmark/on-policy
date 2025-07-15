#!/bin/sh
env="StarCraft2"
map="3s5z_vs_3s6z"
algo="mappo"
exp="eval_experiment02"
num_seeds=32

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed amount is ${num_seeds}"
seed=0
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 python ../eval/eval_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 1 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 64 \
--add_agent_id true \
--model_dir "/workspace/mounted/on-policy/results/training/experiment02/files" \
--num_eval_seeds ${num_seeds} \
--cuda false \
--user_name "luca-mertens-kiel-university"
