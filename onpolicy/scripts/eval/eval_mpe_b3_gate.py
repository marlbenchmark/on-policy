#!/usr/bin/env python
"""Paired B3 threshold-gate evaluation for simple_spread.

Every policy variant advances independent B0 and B2 recurrent states on every
step.  A gated variant executes B0 per agent when detector risk is strictly
greater than its threshold, otherwise it executes B2.
"""
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
    parser.add_argument("--b2_actor", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--eval_seed_start", type=int, default=20000)
    parser.add_argument("--eval_batch_size", type=int, default=25)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--gate_reduce", choices=("mean", "max"), default="mean")
    parser.add_argument("--noise_stds", type=float, nargs="*",
                        default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--mask_probs", type=float, nargs="*",
                        default=[0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--delay_steps", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--checkpoint_activation", choices=("tanh", "relu"),
                        default="tanh")
    parser.add_argument("--detector_random_prior_scale", type=float, default=0.5,
                        help="must match the frozen detector checkpoint")
    parser.add_argument("--write_step_log", action="store_true")
    return parser.parse_args()


def _threshold_label(value):
    return format(float(value), ".8g")


def make_variants(thresholds, reducer):
    variants = [{"name": "B0", "threshold": -math.inf},
                {"name": "B2", "threshold": math.inf}]
    seen = set()
    for value in thresholds:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("thresholds must be finite and nonnegative")
        if value in seen:
            raise ValueError("duplicate threshold: {}".format(value))
        seen.add(value)
        variants.append({
            "name": "gate_{}_{}".format(reducer, _threshold_label(value)),
            "threshold": value,
        })
    return variants


def reduce_detector_risk(uncertainty, reducer):
    """Reduce per-teammate risk to one score per controlled agent."""
    if reducer == "mean":
        return uncertainty.mean(axis=-1)
    if reducer == "max":
        return uncertainty.max(axis=-1)
    raise ValueError("unknown reducer: {}".format(reducer))


def select_actions(b0_actions, b2_actions, risk, threshold):
    """Return executed actions and B0 fallback mask (strictly risk > t)."""
    fallback = risk > threshold
    return np.where(fallback, b0_actions, b2_actions), fallback


def load_actors(args, device):
    common = copy.deepcopy(args)
    common.use_recurrent_policy = True
    common.use_naive_recurrent_policy = False
    common.use_ReLU = args.checkpoint_activation == "relu"
    space_env = MPEEnv(common)

    b0_args = copy.deepcopy(common)
    b0_args.algorithm_name = "rmappo"
    b0 = R_Actor(b0_args, space_env.observation_space[0],
                 space_env.action_space[0], device)

    b2_args = copy.deepcopy(common)
    b2_args.algorithm_name = "ua_rep_mappo"
    b2_args.cstm_num_heads = 3
    b2_args.cstm_use_separate_detector = True
    b2_args.cstm_use_uncertainty_feature = True
    b2_args.cstm_random_prior_scale = args.detector_random_prior_scale
    b2 = B1Actor(b2_args, space_env.observation_space[0],
                 space_env.action_space[0], args.num_agents, device)

    b0.load_state_dict(torch.load(args.b0_actor, map_location=device))
    b2.load_state_dict(torch.load(args.b2_actor, map_location=device))
    b0.eval()
    b2.eval()
    action_dim = space_env.action_space[0].n
    space_env.close()
    return b0, b2, action_dim


def make_conditions(args):
    conditions = [("clean", 0.0)]
    conditions.extend(("noise", float(x)) for x in args.noise_stds if x > 0)
    conditions.extend(("mask", float(x)) for x in args.mask_probs if x > 0)
    conditions.extend(("delay", int(x)) for x in args.delay_steps if x > 0)
    return conditions


def basic(values):
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    sem = std / math.sqrt(values.size) if values.size else 0.0
    mean = float(values.mean()) if values.size else 0.0
    return {"mean": mean, "std": std, "sem": sem,
            "ci95_low": mean - 1.96 * sem,
            "ci95_high": mean + 1.96 * sem}


def _new_gate_counts(batch_size):
    zeros = lambda: np.zeros(batch_size, dtype=np.float64)
    return {key: zeros() for key in (
        "agent_steps", "fallback", "prediction_error", "error_steps",
        "fallback_error", "fallback_no_error", "action_disagreement")}


@torch.no_grad()
def run_batch(args, b0, b2, action_dim, condition, condition_index,
              seeds, variants, step_writer=None):
    records = []
    for seed in seeds:
        for variant_index, variant in enumerate(variants):
            env = MPEEnv(args)
            env.seed(seed)
            observation = np.asarray(env.reset(), dtype=np.float32)
            corruptor = TeammatePositionCorruptor(
                condition[0], condition[1], args.num_agents,
                args.num_landmarks,
                seed=seed * 1009 + condition_index * 9176)
            corruptor.reset()
            records.append({"seed": seed, "variant_index": variant_index,
                            "variant": variant, "env": env,
                            "obs": observation, "corruptor": corruptor})

    batch_size = len(records)
    state_shape = (batch_size, args.num_agents, args.recurrent_N,
                   args.hidden_size)
    b0_states = np.zeros(state_shape, dtype=np.float32)
    b2_states = np.zeros(state_shape, dtype=np.float32)
    masks = np.ones((batch_size * args.num_agents, 1), dtype=np.float32)
    returns = np.zeros(batch_size, dtype=np.float64)
    counts = _new_gate_counts(batch_size)

    for step in range(args.episode_length):
        actor_obs = np.stack([
            record["corruptor"].transform(record["obs"])
            for record in records])
        flat_obs = actor_obs.reshape(batch_size * args.num_agents, -1)
        flat_b0_states = b0_states.reshape(
            batch_size * args.num_agents, args.recurrent_N, args.hidden_size)
        flat_b2_states = b2_states.reshape(
            batch_size * args.num_agents, args.recurrent_N, args.hidden_size)

        b0_action_tensor, _, next_b0_states = b0(
            flat_obs, flat_b0_states, masks, deterministic=True)
        head_logits, mean_probs, uncertainty, _ = b2.detector_outputs(
            flat_obs, flat_b2_states, masks)
        b2_action_tensor, _, next_b2_states = b2(
            flat_obs, flat_b2_states, masks, deterministic=True)

        b0_actions = b0_action_tensor.cpu().numpy().reshape(
            batch_size, args.num_agents)
        b2_actions = b2_action_tensor.cpu().numpy().reshape(
            batch_size, args.num_agents)
        uncertainty_np = uncertainty.cpu().numpy().reshape(
            batch_size, args.num_agents, args.num_agents - 1)
        risk_mean = uncertainty_np.mean(axis=-1)
        risk_max = uncertainty_np.max(axis=-1)
        risk = reduce_detector_risk(uncertainty_np, args.gate_reduce)

        executed = np.empty_like(b0_actions)
        fallback = np.zeros_like(b0_actions, dtype=bool)
        for index, record in enumerate(records):
            executed[index], fallback[index] = select_actions(
                b0_actions[index], b2_actions[index], risk[index],
                record["variant"]["threshold"])

        predictions = mean_probs.argmax(dim=-1).cpu().numpy().reshape(
            batch_size, args.num_agents, args.num_agents - 1)
        targets = np.stack([
            np.stack([np.delete(row, agent_id)
                      for agent_id in range(args.num_agents)])
            for row in executed])
        prediction_errors = (predictions != targets).astype(np.float64)
        error_mean = prediction_errors.mean(axis=-1)
        error_any = error_mean > 0.0
        action_disagreement = b0_actions != b2_actions

        counts["agent_steps"] += args.num_agents
        counts["fallback"] += fallback.sum(axis=1)
        counts["prediction_error"] += prediction_errors.sum(axis=(1, 2))
        counts["error_steps"] += error_any.sum(axis=1)
        counts["fallback_error"] += (fallback & error_any).sum(axis=1)
        counts["fallback_no_error"] += (fallback & ~error_any).sum(axis=1)
        counts["action_disagreement"] += action_disagreement.sum(axis=1)

        if step_writer is not None:
            for index, record in enumerate(records):
                for agent_id in range(args.num_agents):
                    step_writer.writerow({
                        "train_seed": args.train_seed,
                        "eval_seed": record["seed"],
                        "corruption_type": condition[0],
                        "level": condition[1],
                        "variant": record["variant"]["name"],
                        "threshold": record["variant"]["threshold"],
                        "gate_reduce": args.gate_reduce,
                        "step": step,
                        "agent_id": agent_id,
                        "uncertainty_mean": risk_mean[index, agent_id],
                        "uncertainty_max": risk_max[index, agent_id],
                        "gate_risk": risk[index, agent_id],
                        "fallback": int(fallback[index, agent_id]),
                        "b0_action": int(b0_actions[index, agent_id]),
                        "b2_action": int(b2_actions[index, agent_id]),
                        "executed_action": int(executed[index, agent_id]),
                        "actions_disagree": int(
                            action_disagreement[index, agent_id]),
                        "prediction_error_mean": error_mean[index, agent_id],
                        "prediction_error_any": int(error_any[index, agent_id]),
                    })

        env_actions = np.eye(action_dim, dtype=np.float32)[executed]
        for index, record in enumerate(records):
            obs, rewards, _, _ = record["env"].step(env_actions[index])
            record["obs"] = np.asarray(obs, dtype=np.float32)
            returns[index] += float(np.mean(rewards))
        b0_states = next_b0_states.cpu().numpy().reshape(state_shape)
        b2_states = next_b2_states.cpu().numpy().reshape(state_shape)

    rows = []
    teammate_predictions_per_agent_step = args.num_agents - 1
    for index, record in enumerate(records):
        agent_steps = counts["agent_steps"][index]
        error_steps = counts["error_steps"][index]
        no_error_steps = agent_steps - error_steps
        fallback_count = counts["fallback"][index]
        rows.append({
            "algorithm": record["variant"]["name"],
            "train_seed": args.train_seed,
            "eval_seed": record["seed"],
            "corruption_type": condition[0],
            "level": condition[1],
            "episode_return": returns[index],
            "threshold": record["variant"]["threshold"],
            "gate_reduce": args.gate_reduce,
            "fallback_rate": fallback_count / agent_steps,
            "prediction_error_rate": counts["prediction_error"][index] /
                (agent_steps * teammate_predictions_per_agent_step),
            "prediction_error_recall": (counts["fallback_error"][index] /
                error_steps) if error_steps else 0.0,
            "false_fallback_rate": (counts["fallback_no_error"][index] /
                no_error_steps) if no_error_steps else 0.0,
            "fallback_error_rate": (counts["fallback_error"][index] /
                fallback_count) if fallback_count else 0.0,
            "b0_b2_action_disagreement_rate":
                counts["action_disagreement"][index] / agent_steps,
        })
    for record in records:
        record["env"].close()
    return rows


def summarize_condition(rows, variants):
    by_variant = {item["name"]: [] for item in variants}
    for row in rows:
        by_variant[row["algorithm"]].append(row)
    for value in by_variant.values():
        value.sort(key=lambda row: row["eval_seed"])
    b0 = np.asarray([row["episode_return"] for row in by_variant["B0"]])
    b2 = np.asarray([row["episode_return"] for row in by_variant["B2"]])
    oracle = np.maximum(b0, b2)
    summaries = []
    for variant in variants:
        name = variant["name"]
        variant_rows = by_variant[name]
        returns = np.asarray([row["episode_return"] for row in variant_rows])
        summary = {
            "algorithm": name,
            "threshold": variant["threshold"],
            "episodes": int(returns.size),
            "return": basic(returns),
            "delta_vs_B0": basic(returns - b0),
            "delta_vs_B2": basic(returns - b2),
            "win_rate_vs_B0": float((returns > b0).mean()),
            "win_rate_vs_B2": float((returns > b2).mean()),
            "oracle_mean": float(oracle.mean()),
            "oracle_gap": float(oracle.mean() - returns.mean()),
        }
        for metric in ("fallback_rate", "prediction_error_rate",
                       "prediction_error_recall", "false_fallback_rate",
                       "fallback_error_rate", "b0_b2_action_disagreement_rate"):
            summary[metric] = basic([row[metric] for row in variant_rows])
        summaries.append(summary)
    return summaries


def write_summary_csv(path, summaries):
    rows = []
    for item in summaries:
        rows.append({
            "corruption_type": item["corruption_type"], "level": item["level"],
            "algorithm": item["algorithm"], "threshold": item["threshold"],
            "episodes": item["episodes"], "return_mean": item["return"]["mean"],
            "return_ci95_low": item["return"]["ci95_low"],
            "return_ci95_high": item["return"]["ci95_high"],
            "delta_vs_b0": item["delta_vs_B0"]["mean"],
            "delta_vs_b0_ci95_low": item["delta_vs_B0"]["ci95_low"],
            "delta_vs_b0_ci95_high": item["delta_vs_B0"]["ci95_high"],
            "delta_vs_b2": item["delta_vs_B2"]["mean"],
            "delta_vs_b2_ci95_low": item["delta_vs_B2"]["ci95_low"],
            "delta_vs_b2_ci95_high": item["delta_vs_B2"]["ci95_high"],
            "win_rate_vs_b0": item["win_rate_vs_B0"],
            "win_rate_vs_b2": item["win_rate_vs_B2"],
            "oracle_mean": item["oracle_mean"], "oracle_gap": item["oracle_gap"],
            "fallback_rate": item["fallback_rate"]["mean"],
            "prediction_error_rate": item["prediction_error_rate"]["mean"],
            "prediction_error_recall": item["prediction_error_recall"]["mean"],
            "false_fallback_rate": item["false_fallback_rate"]["mean"],
            "fallback_error_rate": item["fallback_error_rate"]["mean"],
            "action_disagreement_rate":
                item["b0_b2_action_disagreement_rate"]["mean"],
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.scenario_name != "simple_spread":
        raise ValueError("this evaluator currently validates simple_spread only")
    variants = make_variants(args.thresholds, args.gate_reduce)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    b0, b2, action_dim = load_actors(args, device)
    conditions = make_conditions(args)
    print("device={} episodes={} conditions={} variants={} reducer={}".format(
        device, args.eval_episodes, len(conditions), len(variants),
        args.gate_reduce), flush=True)

    step_handle = None
    step_writer = None
    if args.write_step_log:
        step_handle = (args.output_dir / "gate_steps.csv").open(
            "w", newline="", encoding="utf-8")
        fields = ["train_seed", "eval_seed", "corruption_type", "level",
                  "variant", "threshold", "gate_reduce", "step", "agent_id",
                  "uncertainty_mean", "uncertainty_max", "gate_risk", "fallback",
                  "b0_action", "b2_action", "executed_action", "actions_disagree",
                  "prediction_error_mean", "prediction_error_any"]
        step_writer = csv.DictWriter(step_handle, fieldnames=fields)
        step_writer.writeheader()

    all_rows, all_summaries = [], []
    seeds = [args.eval_seed_start + i for i in range(args.eval_episodes)]
    for condition_index, condition in enumerate(conditions):
        condition_rows = []
        for start in range(0, len(seeds), args.eval_batch_size):
            condition_rows.extend(run_batch(
                args, b0, b2, action_dim, condition, condition_index,
                seeds[start:start + args.eval_batch_size], variants, step_writer))
        condition_summaries = summarize_condition(condition_rows, variants)
        for item in condition_summaries:
            item.update({"corruption_type": condition[0], "level": condition[1]})
        all_rows.extend(condition_rows)
        all_summaries.extend(condition_summaries)
        best = max(condition_summaries[2:], key=lambda item: item["return"]["mean"])
        print("{:>5} {:>4}: B0={:8.3f} B2={:8.3f} best={} {:8.3f} "
              "dB2={:+.3f} fb={:.3f}".format(
                  condition[0], str(condition[1]),
                  condition_summaries[0]["return"]["mean"],
                  condition_summaries[1]["return"]["mean"],
                  best["algorithm"], best["return"]["mean"],
                  best["delta_vs_B2"]["mean"],
                  best["fallback_rate"]["mean"]), flush=True)

    if step_handle is not None:
        step_handle.close()
    with (args.output_dir / "episode_results.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(all_summaries, handle, indent=2, ensure_ascii=False)
    write_summary_csv(args.output_dir / "summary.csv", all_summaries)
    print("saved results to {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
