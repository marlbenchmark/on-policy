#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BASE_PYTHON=/root/miniconda3/bin/python

cd "$ROOT"
"$BASE_PYTHON" onpolicy/scripts/eval/eval_mpe_headroom.py \
  --algorithm_name cstm_mappo \
  --scenario_name simple_spread \
  --num_agents 3 \
  --num_landmarks 3 \
  --episode_length 25 \
  --eval_episodes 200 \
  --eval_batch_size 50 \
  --eval_seed_start 10000 \
  --train_seed 1 \
  --checkpoint_activation tanh \
  --b0_actor onpolicy/scripts/results/MPE/simple_spread/rmappo/check/run7/models/actor.pt \
  --b1_actor onpolicy/scripts/results/MPE/simple_spread/cstm_mappo/b1_liam_mappo/run1/models/actor.pt \
  --b2_actor onpolicy/scripts/results/MPE/simple_spread/ua_rep_mappo/b2_ua_rep_mappo/run1/models/actor.pt \
  --output_dir onpolicy/scripts/results/MPE/simple_spread/headroom_b2/seed1 \
  --noise_stds 0.05 0.10 0.20 0.30 \
  --mask_probs 0.10 0.20 0.30 0.50 \
  --delay_steps 1 2 3 \
  --cuda
