#!/bin/sh
# Train RMAPPO on any of the four supported PettingZoo environments.
#
# Usage:
#   ./train_pettingzoo.sh <scenario>
#
# Scenarios:
#   spread           -- simple_spread_v3          (3 agents, cooperative coverage)
#   speaker_listener -- simple_speaker_listener_v4 (2 agents, comm + navigation)
#   reference        -- simple_reference_v3        (2 agents, referential comm)
#   pistonball       -- pistonball_v6              (20 pistons, cooperative control)
#
# Example:
#   ./train_pettingzoo.sh spread

ENV="MPE"
ALGO="rmappo"
EXP="check"
SEED_MAX=1

# --------------------------------------------------------------------------- #
# Per-scenario configuration                                                   #
# --------------------------------------------------------------------------- #

case "$1" in

    spread)
        SCENARIO="simple_spread"
        NUM_AGENTS=3
        NUM_LANDMARKS=3
        N_ROLLOUT_THREADS=128
        EPISODE_LENGTH=25
        NUM_ENV_STEPS=20000000
        PPO_EPOCH=10
        EXTRA_FLAGS="--use_ReLU"
        ;;

    speaker_listener)
        # NOTE: simple_speaker_listener uses the separated runner because the
        # speaker and listener have different observation and action spaces.
        # --share_policy must NOT be passed for this scenario.
        SCENARIO="simple_speaker_listener"
        NUM_AGENTS=2
        NUM_LANDMARKS=3
        N_ROLLOUT_THREADS=128
        EPISODE_LENGTH=25
        NUM_ENV_STEPS=2000000
        PPO_EPOCH=15
        EXTRA_FLAGS=""
        ;;

    reference)
        SCENARIO="simple_reference"
        NUM_AGENTS=2
        NUM_LANDMARKS=3
        N_ROLLOUT_THREADS=128
        EPISODE_LENGTH=25
        NUM_ENV_STEPS=3000000
        PPO_EPOCH=15
        EXTRA_FLAGS=""
        ;;

    pistonball)
        # Pistonball uses discrete actions (continuous=False in the factory).
        # n_rollout_threads is lower than MPE because pistonball observations
        # are much larger (flattened images).
        SCENARIO="pistonball"
        NUM_AGENTS=20
        NUM_LANDMARKS=0
        N_ROLLOUT_THREADS=32
        EPISODE_LENGTH=125
        NUM_ENV_STEPS=10000000
        PPO_EPOCH=10
        EXTRA_FLAGS=""
        ;;

    *)
        echo "Usage: $0 {spread|speaker_listener|reference|pistonball}"
        exit 1
        ;;

esac

# --------------------------------------------------------------------------- #
# Training loop                                                                #
# --------------------------------------------------------------------------- #

echo "env=${ENV}  scenario=${SCENARIO}  algo=${ALGO}  exp=${EXP}  seeds=1..${SEED_MAX}"

for seed in $(seq ${SEED_MAX}); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py \
        --env_name ${ENV} \
        --algorithm_name ${ALGO} \
        --experiment_name ${EXP} \
        --scenario_name ${SCENARIO} \
        --num_agents ${NUM_AGENTS} \
        --num_landmarks ${NUM_LANDMARKS} \
        --seed ${seed} \
        --n_training_threads 1 \
        --n_rollout_threads ${N_ROLLOUT_THREADS} \
        --num_mini_batch 1 \
        --episode_length ${EPISODE_LENGTH} \
        --num_env_steps ${NUM_ENV_STEPS} \
        --ppo_epoch ${PPO_EPOCH} \
        --gain 0.01 \
        --lr 7e-4 \
        --critic_lr 7e-4 \
        --wandb_name "xxx" \
        --user_name "yuchao" \
        ${EXTRA_FLAGS}
done
