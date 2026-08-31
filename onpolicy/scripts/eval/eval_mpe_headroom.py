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
    parser.add_argument("--b2_actor", type=Path, required=True)
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
    parser.add_argument(
        "--b2_uncertainty_override", type=float, default=None,
        help=("constant uncertainty consumed by the B2 policy; diagnostics "
              "still report predicted uncertainty"))
    parser.add_argument(
        "--b2_disable_uncertainty_adapter", action="store_true",
        help="bypass the complete B2 uncertainty adapter, including its bias")
    parser.add_argument(
        "--b2_disable_teammate_policy", action="store_true",
        help="bypass B2 teammate latent fusion while retaining diagnostics")
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
    b1_args.cstm_num_heads = 1
    b1_args.cstm_use_uncertainty_feature = False
    b1_args.cstm_random_prior_scale = 0.0
    b1_args.cstm_use_separate_detector = False
    b1_args.cstm_detector_aux_coef = 0.0
    b1 = B1Actor(
        b1_args, space_env.observation_space[0], space_env.action_space[0],
        args.num_agents, device)

    b2_args = copy.deepcopy(common)
    b2_args.algorithm_name = "ua_rep_mappo"
    b2_args.cstm_num_heads = 3
    if not b2_args.cstm_use_separate_detector:
        b2_args.cstm_use_uncertainty_feature = True
    b2 = B1Actor(
        b2_args, space_env.observation_space[0], space_env.action_space[0],
        args.num_agents, device)

    b0.load_state_dict(torch.load(args.b0_actor, map_location=device))
    b1.load_state_dict(torch.load(args.b1_actor, map_location=device))
    b2.load_state_dict(torch.load(args.b2_actor, map_location=device))
    b2.set_uncertainty_override(args.b2_uncertainty_override)
    if args.b2_disable_uncertainty_adapter:
        b2.use_uncertainty_feature = False
    if args.b2_disable_teammate_policy:
        b2.use_teammate_policy = False
    b0.eval()
    b1.eval()
    b2.eval()
    action_dim = space_env.action_space[0].n
    space_env.close()
    return {"B0": b0, "B1": b1, "B2": b2}, action_dim


def _finish_diagnostics(count, sum_error, sum_uncertainty,
                        sum_uncertainty_sq, sum_uncertainty_error,
                        sum_head_disagreement, sum_predictive_entropy,
                        sum_head_entropy, sum_prediction_confidence,
                        sum_pairwise_js):
    if count <= 0:
        raise ValueError("B2 diagnostics received no predictions")
    mean_error = sum_error / count
    mean_uncertainty = sum_uncertainty / count
    uncertainty_variance = max(
        0.0, sum_uncertainty_sq / count - mean_uncertainty ** 2)
    error_variance = max(0.0, mean_error - mean_error ** 2)
    covariance = (sum_uncertainty_error / count
                  - mean_uncertainty * mean_error)
    denominator = math.sqrt(uncertainty_variance * error_variance)
    correlation = covariance / denominator if denominator > 1e-12 else 0.0
    return {
        "prediction_count": int(count),
        "prediction_error": float(mean_error),
        "prediction_accuracy": float(1.0 - mean_error),
        "mean_uncertainty": float(mean_uncertainty),
        "uncertainty_std": float(math.sqrt(uncertainty_variance)),
        "uncertainty_error_correlation": float(correlation),
        "head_action_disagreement": float(sum_head_disagreement / count),
        "mean_predictive_entropy": float(sum_predictive_entropy / count),
        "mean_head_entropy": float(sum_head_entropy / count),
        "mean_prediction_confidence": float(
            sum_prediction_confidence / count),
        "mean_pairwise_js": float(sum_pairwise_js / count),
        "_sum_error": float(sum_error),
        "_sum_uncertainty": float(sum_uncertainty),
        "_sum_uncertainty_sq": float(sum_uncertainty_sq),
        "_sum_uncertainty_error": float(sum_uncertainty_error),
        "_sum_head_disagreement": float(sum_head_disagreement),
        "_sum_predictive_entropy": float(sum_predictive_entropy),
        "_sum_head_entropy": float(sum_head_entropy),
        "_sum_prediction_confidence": float(sum_prediction_confidence),
        "_sum_pairwise_js": float(sum_pairwise_js),
    }


@torch.no_grad()
def run_episode_batch(args, algorithm, actor, action_dim, condition,
                      eval_seeds, condition_index):
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
    diagnostic_sums = None
    if algorithm == "B2":
        diagnostic_sums = {
            "count": np.zeros(batch_size, dtype=np.int64),
            "error": np.zeros(batch_size, dtype=np.float64),
            "uncertainty": np.zeros(batch_size, dtype=np.float64),
            "uncertainty_sq": np.zeros(batch_size, dtype=np.float64),
            "uncertainty_error": np.zeros(batch_size, dtype=np.float64),
            "head_disagreement": np.zeros(batch_size, dtype=np.float64),
            "predictive_entropy": np.zeros(batch_size, dtype=np.float64),
            "head_entropy": np.zeros(batch_size, dtype=np.float64),
            "prediction_confidence": np.zeros(batch_size, dtype=np.float64),
            "pairwise_js": np.zeros(batch_size, dtype=np.float64),
        }

    for _ in range(args.episode_length):
        actor_obs = np.stack([
            corruptor.transform(obs)
            for corruptor, obs in zip(corruptors, observations)
        ])
        flat_obs = actor_obs.reshape(batch_size * args.num_agents, -1)
        flat_states = rnn_states.reshape(
            batch_size * args.num_agents, args.recurrent_N,
            args.hidden_size)
        if algorithm == "B2":
            head_logits, mean_probs, uncertainty, _ = \
                actor.teammate_diagnostics(flat_obs, flat_states, masks)
        actions, _, next_rnn_states = actor(
            flat_obs, flat_states, masks, deterministic=True)
        action_indices = actions.detach().cpu().numpy().reshape(
            batch_size, args.num_agents)

        if algorithm == "B2":
            targets = np.stack([
                np.delete(action_indices, agent_id, axis=1)
                for agent_id in range(args.num_agents)
            ], axis=1)
            predictions = mean_probs.argmax(dim=-1).cpu().numpy().reshape(
                batch_size, args.num_agents, args.num_agents - 1)
            errors = (predictions != targets).astype(np.float64)
            head_probs = torch.softmax(head_logits, dim=-1)
            eps = torch.finfo(head_probs.dtype).eps
            predictive_entropy = -(
                mean_probs * mean_probs.clamp_min(eps).log()).sum(-1)
            predictive_entropy = (
                predictive_entropy / math.log(action_dim)).cpu().numpy().reshape(
                    batch_size, args.num_agents, args.num_agents - 1)
            per_head_entropy = -(
                head_probs * head_probs.clamp_min(eps).log()).sum(-1)
            mean_head_entropy = (
                per_head_entropy.mean(dim=1) / math.log(action_dim)
            ).cpu().numpy().reshape(
                batch_size, args.num_agents, args.num_agents - 1)
            prediction_confidence = mean_probs.max(
                dim=-1).values.cpu().numpy().reshape(
                    batch_size, args.num_agents, args.num_agents - 1)
            pairwise_js_values = []
            for left in range(3):
                for right in range(left + 1, 3):
                    pair_mean = 0.5 * (
                        head_probs[:, left] + head_probs[:, right])
                    pair_entropy = -(
                        pair_mean * pair_mean.clamp_min(eps).log()).sum(-1)
                    component_entropy = 0.5 * (
                        per_head_entropy[:, left]
                        + per_head_entropy[:, right])
                    pairwise_js_values.append(
                        (pair_entropy - component_entropy)
                        / math.log(action_dim))
            pairwise_js = torch.stack(pairwise_js_values).mean(
                dim=0).cpu().numpy().reshape(
                    batch_size, args.num_agents, args.num_agents - 1)
            uncertainty_np = uncertainty.cpu().numpy().reshape(
                batch_size, args.num_agents, args.num_agents - 1)
            head_predictions = head_logits.argmax(dim=-1).cpu().numpy().reshape(
                batch_size, args.num_agents, 3, args.num_agents - 1)
            action_votes = np.eye(action_dim, dtype=np.int64)[
                head_predictions].sum(axis=2)
            head_disagreement = 1.0 - action_votes.max(axis=-1) / 3.0
            reduce_axes = (1, 2)
            samples_per_episode = args.num_agents * (args.num_agents - 1)
            diagnostic_sums["count"] += samples_per_episode
            diagnostic_sums["error"] += errors.sum(axis=reduce_axes)
            diagnostic_sums["uncertainty"] += uncertainty_np.sum(
                axis=reduce_axes)
            diagnostic_sums["uncertainty_sq"] += (
                uncertainty_np ** 2).sum(axis=reduce_axes)
            diagnostic_sums["uncertainty_error"] += (
                uncertainty_np * errors).sum(axis=reduce_axes)
            diagnostic_sums["head_disagreement"] += head_disagreement.sum(
                axis=reduce_axes)
            diagnostic_sums["predictive_entropy"] += predictive_entropy.sum(
                axis=reduce_axes)
            diagnostic_sums["head_entropy"] += mean_head_entropy.sum(
                axis=reduce_axes)
            diagnostic_sums["prediction_confidence"] += \
                prediction_confidence.sum(axis=reduce_axes)
            diagnostic_sums["pairwise_js"] += pairwise_js.sum(
                axis=reduce_axes)

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
    episode_diagnostics = None
    if algorithm == "B2":
        episode_diagnostics = [
            _finish_diagnostics(
                diagnostic_sums["count"][i], diagnostic_sums["error"][i],
                diagnostic_sums["uncertainty"][i],
                diagnostic_sums["uncertainty_sq"][i],
                diagnostic_sums["uncertainty_error"][i],
                diagnostic_sums["head_disagreement"][i],
                diagnostic_sums["predictive_entropy"][i],
                diagnostic_sums["head_entropy"][i],
                diagnostic_sums["prediction_confidence"][i],
                diagnostic_sums["pairwise_js"][i])
            for i in range(batch_size)
        ]
    return episode_returns.tolist(), episode_diagnostics


def make_conditions(args):
    conditions = [("clean", 0.0)]
    conditions.extend(("noise", float(x)) for x in args.noise_stds if x > 0)
    conditions.extend(("mask", float(x)) for x in args.mask_probs if x > 0)
    conditions.extend(("delay", int(x)) for x in args.delay_steps if x > 0)
    return conditions


def basic(values):
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values)) if values.size else 0.0
    return {
        "mean": float(values.mean()),
        "std": std,
        "sem": sem,
        "ci95_low": float(values.mean() - 1.96 * sem),
        "ci95_high": float(values.mean() + 1.96 * sem),
    }


def summarize_returns(returns):
    arrays = {key: np.asarray(value, dtype=np.float64)
              for key, value in returns.items()}
    result = {
        "episodes": int(len(arrays["B0"])),
        "returns": {key: basic(value) for key, value in arrays.items()},
        "paired_deltas": {},
        "episode_win_rates": {},
    }
    for left, right in (("B1", "B0"), ("B2", "B0"), ("B2", "B1")):
        name = "{}_minus_{}".format(left, right)
        result["paired_deltas"][name] = basic(arrays[left] - arrays[right])
        result["episode_win_rates"]["{}_over_{}".format(left, right)] = \
            float((arrays[left] > arrays[right]).mean())
    stacked = np.stack([arrays[key] for key in ("B0", "B1", "B2")])
    oracle_mean = float(stacked.max(axis=0).mean())
    best_fixed_mean = max(
        result["returns"][key]["mean"] for key in ("B0", "B1", "B2"))
    result["oracle_mean"] = oracle_mean
    result["best_fixed_mean"] = best_fixed_mean
    result["oracle_gap"] = oracle_mean - best_fixed_mean
    return result


def summarize_diagnostics(episode_diagnostics):
    total_count = sum(item["prediction_count"] for item in episode_diagnostics)
    aggregate = _finish_diagnostics(
        total_count,
        sum(item["_sum_error"] for item in episode_diagnostics),
        sum(item["_sum_uncertainty"] for item in episode_diagnostics),
        sum(item["_sum_uncertainty_sq"] for item in episode_diagnostics),
        sum(item["_sum_uncertainty_error"] for item in episode_diagnostics),
        sum(item["_sum_head_disagreement"] for item in episode_diagnostics),
        sum(item["_sum_predictive_entropy"] for item in episode_diagnostics),
        sum(item["_sum_head_entropy"] for item in episode_diagnostics),
        sum(item["_sum_prediction_confidence"] for item in episode_diagnostics),
        sum(item["_sum_pairwise_js"] for item in episode_diagnostics))
    return {key: value for key, value in aggregate.items()
            if not key.startswith("_")}


def add_relative_to_clean(summaries, results_by_condition):
    clean_key = ("clean", 0.0)
    clean_returns = results_by_condition[clean_key]["returns"]
    clean_diagnostics = results_by_condition[clean_key]["diagnostics"]
    diagnostic_metrics = (
        "prediction_error", "prediction_accuracy", "mean_uncertainty",
        "uncertainty_std", "uncertainty_error_correlation",
        "head_action_disagreement", "mean_predictive_entropy",
        "mean_head_entropy", "mean_prediction_confidence",
        "mean_pairwise_js")
    for summary in summaries:
        key = (summary["corruption_type"], summary["level"])
        current = results_by_condition[key]
        relative = {"return_deltas": {}, "B2_diagnostic_deltas": {}}
        for algorithm in ("B0", "B1", "B2"):
            relative["return_deltas"][algorithm] = basic(
                np.asarray(current["returns"][algorithm])
                - np.asarray(clean_returns[algorithm]))
        for metric in diagnostic_metrics:
            relative["B2_diagnostic_deltas"][metric] = basic([
                current_item[metric] - clean_item[metric]
                for current_item, clean_item in zip(
                    current["diagnostics"], clean_diagnostics)
            ])
        summary["relative_to_clean"] = relative


def write_outputs(args, raw_rows, diagnostic_rows, summaries):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "episode_returns.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)

    diagnostic_path = args.output_dir / "b2_episode_diagnostics.csv"
    with diagnostic_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(diagnostic_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    summary_rows = []
    for item in summaries:
        summary_rows.append({
            "corruption_type": item["corruption_type"],
            "level": item["level"],
            "episodes": item["episodes"],
            "b0_mean": item["returns"]["B0"]["mean"],
            "b1_mean": item["returns"]["B1"]["mean"],
            "b2_mean": item["returns"]["B2"]["mean"],
            "delta_b1_minus_b0": item["paired_deltas"]["B1_minus_B0"]["mean"],
            "delta_b2_minus_b0": item["paired_deltas"]["B2_minus_B0"]["mean"],
            "delta_b2_minus_b1": item["paired_deltas"]["B2_minus_B1"]["mean"],
            "oracle_mean": item["oracle_mean"],
            "oracle_gap": item["oracle_gap"],
            "b2_prediction_error": item["B2_diagnostics"]["prediction_error"],
            "b2_prediction_accuracy": item["B2_diagnostics"]["prediction_accuracy"],
            "b2_mean_uncertainty": item["B2_diagnostics"]["mean_uncertainty"],
            "b2_uncertainty_std": item["B2_diagnostics"]["uncertainty_std"],
            "b2_uncertainty_error_correlation": item["B2_diagnostics"]["uncertainty_error_correlation"],
            "b2_head_action_disagreement": item["B2_diagnostics"]["head_action_disagreement"],
            "b2_mean_predictive_entropy": item["B2_diagnostics"]["mean_predictive_entropy"],
            "b2_mean_head_entropy": item["B2_diagnostics"]["mean_head_entropy"],
            "b2_mean_prediction_confidence": item["B2_diagnostics"]["mean_prediction_confidence"],
            "b2_mean_pairwise_js": item["B2_diagnostics"]["mean_pairwise_js"],
            "b0_return_delta_from_clean": item["relative_to_clean"]["return_deltas"]["B0"]["mean"],
            "b1_return_delta_from_clean": item["relative_to_clean"]["return_deltas"]["B1"]["mean"],
            "b2_return_delta_from_clean": item["relative_to_clean"]["return_deltas"]["B2"]["mean"],
            "b2_error_delta_from_clean": item["relative_to_clean"]["B2_diagnostic_deltas"]["prediction_error"]["mean"],
            "b2_uncertainty_delta_from_clean": item["relative_to_clean"]["B2_diagnostic_deltas"]["mean_uncertainty"]["mean"],
            "b2_correlation_delta_from_clean": item["relative_to_clean"]["B2_diagnostic_deltas"]["uncertainty_error_correlation"]["mean"],
            "b2_head_disagreement_delta_from_clean": item["relative_to_clean"]["B2_diagnostic_deltas"]["head_action_disagreement"]["mean"],
            "b2_predictive_entropy_delta_from_clean": item["relative_to_clean"]["B2_diagnostic_deltas"]["mean_predictive_entropy"]["mean"],
            "b2_confidence_delta_from_clean": item["relative_to_clean"]["B2_diagnostic_deltas"]["mean_prediction_confidence"]["mean"],
            "b2_pairwise_js_delta_from_clean": item["relative_to_clean"]["B2_diagnostic_deltas"]["mean_pairwise_js"]["mean"],
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
    raw_rows, diagnostic_rows, summaries = [], [], []
    results_by_condition = {}

    if args.b2_disable_teammate_policy:
        uncertainty_mode = "teammate_policy_disabled"
    elif args.b2_disable_uncertainty_adapter:
        uncertainty_mode = "adapter_disabled"
    else:
        uncertainty_mode = (
            "predicted" if args.b2_uncertainty_override is None
            else "constant:{}".format(args.b2_uncertainty_override))
    print("device={} episodes={} conditions={} b2_uncertainty={}".format(
        device, args.eval_episodes, len(conditions), uncertainty_mode))
    for condition_index, condition in enumerate(conditions):
        returns = {"B0": [], "B1": [], "B2": []}
        b2_episode_diagnostics = []
        eval_seeds = [args.eval_seed_start + episode_id
                      for episode_id in range(args.eval_episodes)]
        for algorithm, actor in actors.items():
            for batch_start in range(0, args.eval_episodes,
                                     args.eval_batch_size):
                batch_seeds = eval_seeds[
                    batch_start:batch_start + args.eval_batch_size]
                batch_returns, batch_diagnostics = run_episode_batch(
                    args, algorithm, actor, action_dim, condition,
                    batch_seeds, condition_index)
                returns[algorithm].extend(batch_returns)
                if algorithm == "B2":
                    b2_episode_diagnostics.extend(batch_diagnostics)

        for episode_id, eval_seed in enumerate(eval_seeds):
            for algorithm in actors:
                raw_rows.append({
                    "algorithm": algorithm,
                    "train_seed": args.train_seed,
                    "eval_seed": eval_seed,
                    "corruption_type": condition[0],
                    "level": condition[1],
                    "episode_id": episode_id,
                    "episode_return": returns[algorithm][episode_id],
                })
            public_diagnostics = {
                key: value
                for key, value in b2_episode_diagnostics[episode_id].items()
                if not key.startswith("_")
            }
            diagnostic_rows.append({
                "train_seed": args.train_seed,
                "eval_seed": eval_seed,
                "corruption_type": condition[0],
                "level": condition[1],
                "episode_id": episode_id,
                **public_diagnostics,
            })
        summary = summarize_returns(returns)
        summary["B2_diagnostics"] = summarize_diagnostics(
            b2_episode_diagnostics)
        summary.update({"corruption_type": condition[0], "level": condition[1]})
        summaries.append(summary)
        results_by_condition[(condition[0], condition[1])] = {
            "returns": returns,
            "diagnostics": b2_episode_diagnostics,
        }
        diag = summary["B2_diagnostics"]
        print("{:>5} {:>4}: B0={:8.3f} B1={:8.3f} B2={:8.3f} "
              "B2acc={:.3f} u={:.6f} corr={:+.3f} disagree={:.3f}".format(
                  condition[0], str(condition[1]),
                  summary["returns"]["B0"]["mean"],
                  summary["returns"]["B1"]["mean"],
                  summary["returns"]["B2"]["mean"],
                  diag["prediction_accuracy"], diag["mean_uncertainty"],
                  diag["uncertainty_error_correlation"],
                  diag["head_action_disagreement"]))

    add_relative_to_clean(summaries, results_by_condition)
    write_outputs(args, raw_rows, diagnostic_rows, summaries)
    print("saved results to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
