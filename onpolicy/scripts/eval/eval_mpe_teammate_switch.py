#!/usr/bin/env python
"""Cheap focal-agent headroom test under teammate policy changes."""
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
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.cstm_mappo.algorithm.dual_policy_actor import B1Actor
from onpolicy.scripts.eval.eval_mpe_b4 import (
    load_actor,
    risk_only_b2_gate,
)


VARIANTS = ("B0", "B2", "RISK_SOFT", "B4")
CONDITIONS = (
    "stable_b0",
    "stable_b2",
    "b2_to_b0_alt",
    "b2_to_b2_alt",
    "b2_to_random",
    "b2_to_sticky",
    "b2_to_delayed",
    "b2_random_recover",
)


def parse_args():
    parser = get_config()
    parser.add_argument("--scenario_name", type=str, default="simple_spread")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--b4_actor", type=Path, required=True)
    parser.add_argument("--b0_controller", type=Path, required=True)
    parser.add_argument("--b0_alt_controller", type=Path, required=True)
    parser.add_argument("--b2_controller", type=Path, required=True)
    parser.add_argument("--b2_alt_controller", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--eval_seed_start", type=int, default=91000)
    parser.add_argument("--eval_batch_size", type=int, default=25)
    parser.add_argument("--switch_step", type=int, default=12)
    parser.add_argument("--recovery_step", type=int, default=18)
    parser.add_argument("--teammate_delay", type=int, default=2)
    parser.add_argument("--b3_threshold", type=float, default=0.03)
    parser.add_argument("--risk_soft_scale", type=float, default=200.0)
    parser.add_argument("--checkpoint_activation", choices=("tanh", "relu"),
                        default="tanh")
    return parser.parse_args()


def load_controller(args, path, kind, device):
    common = copy.deepcopy(args)
    common.use_recurrent_policy = True
    common.use_naive_recurrent_policy = False
    common.use_ReLU = args.checkpoint_activation == "relu"
    env = MPEEnv(common)
    if kind == "b0":
        common.algorithm_name = "rmappo"
        actor = R_Actor(common, env.observation_space[0],
                        env.action_space[0], device)
    else:
        common.algorithm_name = "ua_rep_mappo"
        common.cstm_num_heads = 3
        common.cstm_use_separate_detector = False
        common.cstm_use_uncertainty_feature = True
        actor = B1Actor(common, env.observation_space[0],
                        env.action_space[0], args.num_agents, device)
    actor.load_state_dict(torch.load(path, map_location=device))
    actor.eval()
    env.close()
    return actor


def basic(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std,
            "ci95_low": mean - half, "ci95_high": mean + half}


def controller_step(actor, observations, states, masks, args):
    batch_size = len(observations)
    flat_obs = observations.reshape(batch_size * args.num_agents, -1)
    flat_states = states.reshape(
        batch_size * args.num_agents, args.recurrent_N, args.hidden_size)
    actions, _, next_states = actor(
        flat_obs, flat_states, masks, deterministic=True)
    actions = actions.cpu().numpy().reshape(batch_size, args.num_agents)
    next_states = next_states.cpu().numpy().reshape(states.shape)
    return actions, next_states


def choose_teammates(condition, step, main_b0, alt_b0, main_b2, alt_b2,
                     random_actions, sticky_actions, b2_history, args):
    if condition == "stable_b0":
        return main_b0
    if condition == "stable_b2":
        return main_b2
    if step < args.switch_step:
        return main_b2
    if condition == "b2_to_b0_alt":
        return alt_b0
    if condition == "b2_to_b2_alt":
        return alt_b2
    if condition == "b2_to_random":
        return random_actions[:, step]
    if condition == "b2_to_sticky":
        return sticky_actions
    if condition == "b2_to_delayed":
        delayed_index = max(0, len(b2_history) - 1 - args.teammate_delay)
        return b2_history[delayed_index]
    if condition == "b2_random_recover":
        return (random_actions[:, step] if step < args.recovery_step
                else main_b2)
    raise ValueError(condition)


@torch.no_grad()
def run_batch(args, actor, controllers, action_dim, condition,
              condition_index, seeds):
    records = []
    random_by_seed = {}
    for seed in seeds:
        rng = np.random.default_rng(seed * 1009 + condition_index * 104729)
        random_by_seed[seed] = rng.integers(
            0, action_dim,
            size=(args.episode_length, args.num_agents), dtype=np.int64)
        for variant in VARIANTS:
            env = MPEEnv(args)
            env.seed(seed)
            records.append({"seed": seed, "variant": variant, "env": env,
                            "obs": np.asarray(env.reset(), dtype=np.float32)})

    batch_size = len(records)
    state_shape = (batch_size, args.num_agents, args.recurrent_N,
                   args.hidden_size)
    b0_states = np.zeros(state_shape, dtype=np.float32)
    b2_states = np.zeros(state_shape, dtype=np.float32)
    controller_states = {
        name: np.zeros(state_shape, dtype=np.float32)
        for name in controllers
    }
    masks = np.ones((batch_size * args.num_agents, 1), dtype=np.float32)
    episode_returns = np.zeros(batch_size, dtype=np.float64)
    pre_returns = np.zeros(batch_size, dtype=np.float64)
    post_returns = np.zeros(batch_size, dtype=np.float64)
    pre_risk = np.zeros(batch_size, dtype=np.float64)
    post_risk = np.zeros(batch_size, dtype=np.float64)
    pre_fallback = np.zeros(batch_size, dtype=np.float64)
    post_fallback = np.zeros(batch_size, dtype=np.float64)
    pre_error = np.zeros(batch_size, dtype=np.float64)
    post_error = np.zeros(batch_size, dtype=np.float64)
    pre_count = post_count = 0
    step_rows = []
    b2_history = []
    sticky_actions = None
    observations = np.stack([record["obs"] for record in records])
    random_actions = np.stack([random_by_seed[record["seed"]]
                               for record in records])

    for step in range(args.episode_length):
        flat_obs = observations.reshape(batch_size * args.num_agents, -1)
        flat_b0 = b0_states.reshape(
            batch_size * args.num_agents, args.recurrent_N, args.hidden_size)
        flat_b2 = b2_states.reshape(
            batch_size * args.num_agents, args.recurrent_N, args.hidden_size)
        mixed, next_b2, next_b0, diagnostics, b0_probs, b2_probs = \
            actor._distribution_outputs(flat_obs, flat_b2, flat_b0, masks)
        _, detector_probs, _, _ = actor.b2_actor.detector_outputs(
            flat_obs, flat_b2, masks)

        b0_actions = b0_probs.argmax(-1).cpu().numpy().reshape(
            batch_size, args.num_agents)
        b2_actions = b2_probs.argmax(-1).cpu().numpy().reshape(
            batch_size, args.num_agents)
        b4_actions = mixed.argmax(-1).cpu().numpy().reshape(
            batch_size, args.num_agents)
        b0_agent_probs = b0_probs.reshape(
            batch_size, args.num_agents, action_dim)
        b2_agent_probs = b2_probs.reshape(
            batch_size, args.num_agents, action_dim)
        risk = diagnostics["detector_risk"].cpu().numpy().reshape(
            batch_size, args.num_agents)[:, 0]
        risk_gate = risk_only_b2_gate(
            risk, args.b3_threshold, args.risk_soft_scale)
        risk_probs = (torch.as_tensor(risk_gate, device=b0_probs.device,
                                      dtype=b0_probs.dtype).unsqueeze(-1)
                      * b2_agent_probs[:, 0]
                      + torch.as_tensor(1.0 - risk_gate,
                                        device=b0_probs.device,
                                        dtype=b0_probs.dtype).unsqueeze(-1)
                      * b0_agent_probs[:, 0])
        risk_actions = risk_probs.argmax(-1).cpu().numpy()

        controller_actions = {}
        for name, controller in controllers.items():
            actions, states = controller_step(
                controller, observations, controller_states[name], masks, args)
            controller_actions[name] = actions
            controller_states[name] = states
        main_b2 = controller_actions["b2"]
        b2_history.append(main_b2.copy())
        if step == args.switch_step - 1:
            sticky_actions = main_b2.copy()
        teammate_actions = choose_teammates(
            condition, step, controller_actions["b0"],
            controller_actions["b0_alt"], main_b2,
            controller_actions["b2_alt"], random_actions, sticky_actions,
            b2_history, args).copy()

        choices = {"B0": b0_actions[:, 0], "B2": b2_actions[:, 0],
                   "RISK_SOFT": risk_actions, "B4": b4_actions[:, 0]}
        executed = teammate_actions.copy()
        for index, record in enumerate(records):
            executed[index, 0] = choices[record["variant"]][index]

        if executed.min() < 0 or executed.max() >= action_dim:
            ranges = {name: (int(values.min()), int(values.max()))
                      for name, values in controller_actions.items()}
            ranges["executed"] = (int(executed.min()), int(executed.max()))
            raise ValueError("invalid discrete action ranges: {}".format(ranges))

        detector_predictions = detector_probs.argmax(-1).cpu().numpy().reshape(
            batch_size, args.num_agents, args.num_agents - 1)
        focal_error = (detector_predictions[:, 0]
                       != executed[:, 1:]).mean(axis=1)
        fallback = 1.0 - risk_gate
        if step < args.switch_step:
            pre_risk += risk
            pre_fallback += fallback
            pre_error += focal_error
            pre_count += 1
            phase = "pre"
        else:
            post_risk += risk
            post_fallback += fallback
            post_error += focal_error
            post_count += 1
            phase = ("recovery" if condition == "b2_random_recover"
                     and step >= args.recovery_step else "post")

        for index, record in enumerate(records):
            step_rows.append({
                "condition": condition, "eval_seed": record["seed"],
                "algorithm": record["variant"], "step": step,
                "phase": phase, "focal_risk": risk[index],
                "focal_fallback": fallback[index],
                "focal_prediction_error": focal_error[index],
                "focal_action": executed[index, 0],
                "teammate_1_action": executed[index, 1],
                "teammate_2_action": executed[index, 2],
            })

        env_actions = np.eye(action_dim, dtype=np.float32)[executed]
        next_observations = []
        for index, record in enumerate(records):
            obs, rewards, _, _ = record["env"].step(env_actions[index])
            next_observations.append(np.asarray(obs, dtype=np.float32))
            reward = float(np.mean(rewards))
            episode_returns[index] += reward
            if step < args.switch_step:
                pre_returns[index] += reward
            else:
                post_returns[index] += reward
        observations = np.stack(next_observations)
        b0_states = next_b0.cpu().numpy().reshape(state_shape)
        b2_states = next_b2.cpu().numpy().reshape(state_shape)

    rows = []
    for index, record in enumerate(records):
        rows.append({
            "condition": condition, "eval_seed": record["seed"],
            "algorithm": record["variant"],
            "episode_return": episode_returns[index],
            "pre_return": pre_returns[index],
            "post_return": post_returns[index],
            "pre_risk": pre_risk[index] / pre_count,
            "post_risk": post_risk[index] / post_count,
            "risk_delta": (post_risk[index] / post_count
                           - pre_risk[index] / pre_count),
            "pre_fallback": pre_fallback[index] / pre_count,
            "post_fallback": post_fallback[index] / post_count,
            "fallback_delta": (post_fallback[index] / post_count
                               - pre_fallback[index] / pre_count),
            "pre_prediction_error": pre_error[index] / pre_count,
            "post_prediction_error": post_error[index] / post_count,
        })
        record["env"].close()
    return rows, step_rows


def summarize(rows, conditions=CONDITIONS):
    result = {}
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        arrays = {}
        post_arrays = {}
        diagnostics = {}
        for variant in VARIANTS:
            selected = sorted(
                (row for row in condition_rows if row["algorithm"] == variant),
                key=lambda row: row["eval_seed"])
            arrays[variant] = np.asarray(
                [row["episode_return"] for row in selected])
            post_arrays[variant] = np.asarray(
                [row["post_return"] for row in selected])
            diagnostics[variant] = {
                metric: basic([row[metric] for row in selected])
                for metric in ("pre_risk", "post_risk", "risk_delta",
                               "pre_fallback", "post_fallback",
                               "fallback_delta", "pre_prediction_error",
                               "post_prediction_error")
            }
        comparisons = {}
        post_comparisons = {}
        for left, right in (("B2", "B0"), ("RISK_SOFT", "B0"),
                            ("RISK_SOFT", "B2"), ("B4", "RISK_SOFT")):
            comparisons[left + "-" + right] = basic(
                arrays[left] - arrays[right])
            post_comparisons[left + "-" + right] = basic(
                post_arrays[left] - post_arrays[right])
        oracle = np.maximum(arrays["B0"], arrays["B2"])
        best_fixed = max(arrays["B0"].mean(), arrays["B2"].mean())
        result[condition] = {
            "returns": {name: basic(values)
                        for name, values in arrays.items()},
            "post_returns": {name: basic(values)
                             for name, values in post_arrays.items()},
            "paired_deltas": comparisons,
            "post_paired_deltas": post_comparisons,
            "oracle_mean": float(oracle.mean()),
            "oracle_gap": float(oracle.mean() - best_fixed),
            "diagnostics": diagnostics,
        }
    return result


def main():
    args = parse_args()
    if not (0 < args.switch_step < args.recovery_step < args.episode_length):
        raise ValueError("require 0 < switch < recovery < episode length")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actor, action_dim = load_actor(args, device)
    controllers = {
        "b0": load_controller(args, args.b0_controller, "b0", device),
        "b0_alt": load_controller(
            args, args.b0_alt_controller, "b0", device),
        "b2": load_controller(args, args.b2_controller, "b2", device),
        "b2_alt": load_controller(
            args, args.b2_alt_controller, "b2", device),
    }
    seeds = [args.eval_seed_start + offset
             for offset in range(args.eval_episodes)]
    all_rows, all_step_rows = [], []
    print("device={} seeds={}--{} conditions={}".format(
        device, seeds[0], seeds[-1], len(CONDITIONS)), flush=True)
    for condition_index, condition in enumerate(CONDITIONS):
        condition_rows, condition_step_rows = [], []
        for start in range(0, len(seeds), args.eval_batch_size):
            rows, step_rows = run_batch(
                args, actor, controllers, action_dim, condition,
                condition_index,
                seeds[start:start + args.eval_batch_size])
            condition_rows.extend(rows)
            condition_step_rows.extend(step_rows)
        all_rows.extend(condition_rows)
        all_step_rows.extend(condition_step_rows)
        partial = summarize(condition_rows, (condition,))[condition]
        print("{} B0={:.3f} B2={:.3f} Risk={:.3f} dB2={:+.3f} "
              "post_dB2={:+.3f} dRiskB0={:+.3f} risk_delta={:+.5f} "
              "oracle_gap={:.3f}".format(
                  condition, partial["returns"]["B0"]["mean"],
                  partial["returns"]["B2"]["mean"],
                  partial["returns"]["RISK_SOFT"]["mean"],
                  partial["paired_deltas"]["B2-B0"]["mean"],
                  partial["post_paired_deltas"]["B2-B0"]["mean"],
                  partial["paired_deltas"]["RISK_SOFT-B0"]["mean"],
                  partial["diagnostics"]["RISK_SOFT"]["risk_delta"]["mean"],
                  partial["oracle_gap"]), flush=True)

    with (args.output_dir / "episode_results.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output_dir / "step_results.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_step_rows[0]))
        writer.writeheader()
        writer.writerows(all_step_rows)
    with (args.output_dir / "summary.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(summarize(all_rows), handle, indent=2, ensure_ascii=False)
    print("saved {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
