#!/bin/sh
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

env="MPE"
scenario="simple_reference"
num_landmarks=3
num_agents=2
algo="rmappo"
exp="check"
seed_max=1

case "$(basename "$0")" in
  train_mpe_reference.sh)
    scenario="simple_reference"
    num_landmarks=3
    num_agents=2
    num_env_steps=3000000
    ppo_epoch=15
    use_relu=''
    ;;
  train_mpe_comm.sh)
    scenario="simple_speaker_listener"
    num_landmarks=3
    num_agents=2
    num_env_steps=2000000
    ppo_epoch=15
    use_relu='--use_ReLU'
    ;;
  train_mpe_spread.sh)
    scenario="simple_spread"
    num_landmarks=3
    num_agents=3
    num_env_steps=20000000
    ppo_epoch=10
    use_relu='--use_ReLU'
    ;;
esac

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/../train/train_mpe.py" --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps ${num_env_steps} \
    --ppo_epoch ${ppo_epoch} ${use_relu} --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --use_wandb
done
