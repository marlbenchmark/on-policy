#!/usr/bin/env python
import argparse
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.mpe.perturbations import TeammatePositionCorruptor
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.cstm_mappo.algorithm.dual_policy_actor import B1Actor


def parse_args():
    parser = get_config()
    parser.add_argument("--scenario_name", type=str, default="simple_spread")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--b0_actor", type=Path, required=True)
    parser.add_argument("--b1_actor", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--eval_seed_start", type=int, default=10000)
    parser.add_argument("--eval_batch_size", type=int, default=50)
    parser.add_argument("--noise_stds", type=float, nargs="*",
                        default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--mask_probs", type=float, nargs="*",
                        default=[0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--delay_steps", type=int, nargs="*",
                        default=[1, 2, 3])
    parser.add_argument("--checkpoint_activation", choices=("tanh", "relu"),
                        default="tanh")
    return parser.parse_args()


def load_actors(args, device):
    common = copy.deepcopy(args)
    common.use_recurrent_policy = True
    common.use_naive_recurrent_policy = False
    common.use_ReLU = args.checkpoint_activation == "relu"

    b0_args = copy.deepcopy(common)
    b0_args.algorithm_name = "rmappo"
    # Build one environment for spaces. Environment construction consumes
    # NumPy randomness, but every episode is explicitly reseeded before reset.
    space_env = MPEEnv(common)
    b0 = R_Actor(
        b0_args, space_env.observation_space[0],
        space_env.action_space[0], device)
    b1_args = copy.deepcopy(common)
    b1_args.algorithm_name = "cstm_mappo"
    b1 = B1Actor(
        b1_args, space_env.observation_space[0], space_env.action_space[0],
        args.num_agents, device)

    b0.load_state_dict(torch.load(args.b0_actor, map_location=device))
    b1.load_state_dict(torch.load(args.b1_actor, map_location=device))
    b0.eval()
    b1.eval()
    action_dim = space_env.action_space[0].n
    space_env.close()
    return {"B0": b0, "B1": b1}, action_dim


@torch.no_grad()
def run_episode_batch(args, actor, action_dim, condition, eval_seeds,
                      condition_index):
    envs, observations, corruptors = [], [], []
    for eval_seed in eval_seeds:
        env = MPEEnv(args)
        env.seed(eval_seed)
        envs.append(env)
        observations.append(np.asarray(env.reset(), dtype=np.float32))
        corruptor = TeammatePositionCorruptor(
            condition[0], condition[1], args.num_agents, args.num_landmarks,
            seed=eval_seed * 1009 + condition_index * 9176)
        corruptor.reset()
        corruptors.append(corruptor)

    batch_size = len(envs)
    observations = np.asarray(observations, dtype=np.float32)
    rnn_states = np.zeros(
        (batch_size, args.num_agents, args.recurrent_N, args.hidden_size),
        dtype=np.float32)
    masks = np.ones((batch_size * args.num_agents, 1), dtype=np.float32)
    episode_returns = np.zeros(batch_size, dtype=np.float64)

    for _ in range(args.episode_length):
        actor_obs = np.stack([
            corruptor.transform(obs)
            for corruptor, obs in zip(corruptors, observations)
        ])
        actions, _, next_rnn_states = actor(
            actor_obs.reshape(batch_size * args.num_agents, -1),
            rnn_states.reshape(
                batch_size * args.num_agents, args.recurrent_N,
                args.hidden_size),
            masks, deterministic=True)
        action_indices = actions.detach().cpu().numpy().reshape(
            batch_size, args.num_agents)
        env_actions = np.eye(action_dim, dtype=np.float32)[action_indices]
        next_observations = []
        for env_index, env in enumerate(envs):
            obs, rewards, _, _ = env.step(env_actions[env_index])
            next_observations.append(obs)
            episode_returns[env_index] += float(np.mean(rewards))
        observations = np.asarray(next_observations, dtype=np.float32)
        rnn_states = next_rnn_states.detach().cpu().numpy().reshape(
            batch_size, args.num_agents, args.recurrent_N, args.hidden_size)

    for env in envs:
        env.close()
    return episode_returns.tolist()


def make_conditions(args):
    conditions = [("clean", 0.0)]
    conditions.extend(("noise", float(x)) for x in args.noise_stds if x > 0)
    conditions.extend(("mask", float(x)) for x in args.mask_probs if x > 0)
    conditions.extend(("delay", int(x)) for x in args.delay_steps if x > 0)
    return conditions


def summarize_pair(b0, b1):
    b0 = np.asarray(b0, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    delta = b1 - b0
    oracle = np.maximum(b0, b1)

    def basic(values):
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        sem = std / math.sqrt(len(values)) if values.size else 0.0
        return {
            "mean": float(values.mean()),
            "std": std,
            "sem": sem,
            "ci95_low": float(values.mean() - 1.96 * sem),
            "ci95_high": float(values.mean() + 1.96 * sem),
        }

    b0_stats, b1_stats, delta_stats = basic(b0), basic(b1), basic(delta)
    oracle_mean = float(oracle.mean())
    return {
        "episodes": int(len(b0)),
        "B0": b0_stats,
        "B1": b1_stats,
        "delta_B1_minus_B0": delta_stats,
        "oracle_mean": oracle_mean,
        "best_fixed_mean": max(b0_stats["mean"], b1_stats["mean"]),
        "oracle_gap": oracle_mean - max(b0_stats["mean"], b1_stats["mean"]),
        "b1_episode_win_rate": float((b1 > b0).mean()),
    }


def write_outputs(args, raw_rows, summaries):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "episode_returns.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)

    summary_rows = []
    for item in summaries:
        summary_rows.append({
            "corruption_type": item["corruption_type"],
            "level": item["level"],
            "episodes": item["episodes"],
            "b0_mean": item["B0"]["mean"],
            "b1_mean": item["B1"]["mean"],
            "delta_b1_minus_b0": item["delta_B1_minus_B0"]["mean"],
            "delta_ci95_low": item["delta_B1_minus_B0"]["ci95_low"],
            "delta_ci95_high": item["delta_B1_minus_B0"]["ci95_high"],
            "oracle_mean": item["oracle_mean"],
            "oracle_gap": item["oracle_gap"],
            "b1_episode_win_rate": item["b1_episode_win_rate"],
        })
    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    args = parse_args()
    if args.scenario_name != "simple_spread":
        raise ValueError("this evaluator currently validates simple_spread only")
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actors, action_dim = load_actors(args, device)
    conditions = make_conditions(args)
    raw_rows, summaries = [], []

    print("device={} episodes={} conditions={}".format(
        device, args.eval_episodes, len(conditions)))
    for condition_index, condition in enumerate(conditions):
        returns = {"B0": [], "B1": []}
        eval_seeds = [args.eval_seed_start + episode_id
                      for episode_id in range(args.eval_episodes)]
        for algorithm, actor in actors.items():
            for batch_start in range(0, args.eval_episodes,
                                     args.eval_batch_size):
                batch_seeds = eval_seeds[
                    batch_start:batch_start + args.eval_batch_size]
                returns[algorithm].extend(run_episode_batch(
                    args, actor, action_dim, condition, batch_seeds,
                    condition_index))

        for episode_id, eval_seed in enumerate(eval_seeds):
            for algorithm, actor in actors.items():
                raw_rows.append({
                    "algorithm": algorithm,
                    "train_seed": args.train_seed,
                    "eval_seed": eval_seed,
                    "corruption_type": condition[0],
                    "level": condition[1],
                    "episode_id": episode_id,
                    "episode_return": returns[algorithm][episode_id],
                })
        summary = summarize_pair(returns["B0"], returns["B1"])
        summary.update({"corruption_type": condition[0], "level": condition[1]})
        summaries.append(summary)
        print("{:>5} {:>4}: B0={:8.3f} B1={:8.3f} delta={:+7.3f} "
              "CI=[{:+7.3f},{:+7.3f}] oracle_gap={:.3f}".format(
                  condition[0], str(condition[1]), summary["B0"]["mean"],
                  summary["B1"]["mean"],
                  summary["delta_B1_minus_B0"]["mean"],
                  summary["delta_B1_minus_B0"]["ci95_low"],
                  summary["delta_B1_minus_B0"]["ci95_high"],
                  summary["oracle_gap"]))

    write_outputs(args, raw_rows, summaries)
    print("saved results to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
