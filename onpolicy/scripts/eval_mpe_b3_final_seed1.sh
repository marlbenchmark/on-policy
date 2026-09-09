#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BASE_PYTHON=/root/miniconda3/bin/python

cd "$ROOT"
"$BASE_PYTHON" onpolicy/scripts/eval/eval_mpe_b3_gate.py \
  --algorithm_name cstm_mappo --scenario_name simple_spread \
  --num_agents 3 --num_landmarks 3 --episode_length 25 \
  --eval_episodes 200 --eval_batch_size 25 --eval_seed_start 30000 \
  --train_seed 1 --checkpoint_activation tanh \
  --detector_random_prior_scale 0.5 --gate_reduce mean --thresholds 0.03 \
  --b0_actor onpolicy/scripts/results/MPE/simple_spread/rmappo/check/run7/models/actor.pt \
  --b2_actor onpolicy/scripts/results/MPE/simple_spread/ua_rep_mappo/b2_detector_v5_frozen_1m/run1/models/actor.pt \
  --output_dir onpolicy/scripts/results/MPE/simple_spread/headroom_b3/final_mean_0.03_seed30000_200 \
  --noise_stds 0.05 0.10 0.20 0.30 \
  --mask_probs 0.10 0.20 0.30 0.50 --delay_steps 1 2 3 \
  --write_step_log
