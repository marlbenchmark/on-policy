#!/usr/bin/env python3
"""
Train RMAPPO on any of the four supported PettingZoo environments.

Run one environment:
    python train_pettingzoo.py                  # default: spread
    python train_pettingzoo.py speaker_listener
    python train_pettingzoo.py reference
    python train_pettingzoo.py pistonball

Or import and call the per-environment helpers directly.
"""

import os
import subprocess
import sys
from pathlib import Path


# Repo root (three levels above this file: train_mpe_scripts/ → scripts/ → onpolicy/ → root)
TRAIN_CWD = Path(__file__).resolve().parents[3]

ENV = "MPE"
ALGO = "rmappo"
EXP = "check"
SEED_MAX = 1
STARTING_SEED = 1

# share_policy=True  → shared runner  (do NOT pass --share_policy on the CLI)
# share_policy=False → separated runner (DO pass --share_policy on the CLI)
#
# This is inverted from the flag name because config.py declares the flag as
# action='store_false', default=True.  Passing --share_policy on the command
# line therefore sets all_args.share_policy = False (separated).
SCENARIOS = {
    "spread": {
        "scenario_name": "simple_spread",
        "num_agents": 3,
        "num_landmarks": 3,
        "n_rollout_threads": 128,
        "episode_length": 25,
        "num_env_steps": 20_000_000,
        "ppo_epoch": 10,
        "share_policy": True,
        "extra_flags": ["--use_ReLU"],
    },
    "speaker_listener": {
        # Must use the separated runner: the training script asserts that
        # share_policy=True with simple_speaker_listener is invalid.
        "scenario_name": "simple_speaker_listener",
        "num_agents": 2,
        "num_landmarks": 3,
        "n_rollout_threads": 128,
        "episode_length": 25,
        "num_env_steps": 2_000_000,
        "ppo_epoch": 15,
        "share_policy": False,
        "extra_flags": [],
    },
    "reference": {
        "scenario_name": "simple_reference",
        "num_agents": 2,
        "num_landmarks": 3,
        "n_rollout_threads": 128,
        "episode_length": 25,
        "num_env_steps": 3_000_000,
        "ppo_epoch": 15,
        "share_policy": True,
        "extra_flags": [],
    },
    "pistonball": {
        # n_rollout_threads is lower than MPE because pistonball observations
        # are flattened images (~72 000 dims per piston).
        # episode_length matches PettingZoo's default max_cycles=125.
        "scenario_name": "pistonball",
        "num_agents": 20,
        "num_landmarks": 0,
        "n_rollout_threads": 32,
        "episode_length": 125,
        "num_env_steps": 10_000_000,
        "ppo_epoch": 10,
        "share_policy": True,
        "extra_flags": [],
    },
}


def _run_training(env_key: str, exp: str = EXP, seed_max: int = SEED_MAX,
                  starting_seed: int = STARTING_SEED) -> None:
    cfg = SCENARIOS[env_key]
    scenario = cfg["scenario_name"]

    print(
        f"env={ENV}  scenario={scenario}  algo={ALGO}  "
        f"exp={exp}  seeds={starting_seed}..{seed_max}"
    )

    for seed in range(starting_seed, seed_max + 1):
        print(f"seed is {seed}:")

        cmd = [
            sys.executable, "-m", "onpolicy.scripts.train.train_mpe",
            "--env_name",          ENV,
            "--algorithm_name",    ALGO,
            "--experiment_name",   exp,
            "--scenario_name",     scenario,
            "--num_agents",        str(cfg["num_agents"]),
            "--num_landmarks",     str(cfg["num_landmarks"]),
            "--seed",              str(seed),
            "--n_training_threads", "1",
            "--n_rollout_threads", str(cfg["n_rollout_threads"]),
            "--num_mini_batch",    "1",
            "--episode_length",    str(cfg["episode_length"]),
            "--num_env_steps",     str(cfg["num_env_steps"]),
            "--ppo_epoch",         str(cfg["ppo_epoch"]),
            "--gain",              "0.01",
            "--lr",                "7e-4",
            "--critic_lr",         "7e-4",
            "--wandb_name",        "xxx",
            "--user_name",         "yuchao",
        ]

        # Passing --share_policy sets all_args.share_policy=False (separated runner).
        if not cfg["share_policy"]:
            cmd.append("--share_policy")

        cmd.extend(cfg["extra_flags"])

        subprocess.run(
            cmd,
            check=True,
            cwd=TRAIN_CWD,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )


def spread() -> None:
    _run_training("spread")


def speaklist() -> None:
    _run_training("speaker_listener")


def reference() -> None:
    _run_training("reference")


def pistonball() -> None:
    _run_training("pistonball")


_DISPATCH = {
    "spread":           spread,
    "speaker_listener": speaklist,
    "reference":        reference,
    "pistonball":       pistonball,
}

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "spread"
    if key not in _DISPATCH:
        print(f"Unknown scenario {key!r}. Choose from: {', '.join(_DISPATCH)}")
        sys.exit(1)
    _DISPATCH[key]()
