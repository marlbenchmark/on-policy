#!/bin/sh
# exp params
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="rmappo"
exp="render"
seed=1

# football params
num_agents=3

# --save_videos is preferred instead of --save_gifs 
# because .avi file is much smaller than .gif file

echo "render ${render_episodes} episodes"

CUDA_VISIBLE_DEVICES=0 python render/render_football.py --env_name ${env} --scenario_name ${scenario} \
--algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} --num_agents ${num_agents} --representation "simple115v2" \
--use_render --render_episodes 10 --n_rollout_threads 1 --model_dir "xxx" --save_videos --user_name "xxx" 
