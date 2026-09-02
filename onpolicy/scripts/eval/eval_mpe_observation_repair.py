#!/usr/bin/env python
"""Paired development-set smoke test for online observation repair."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.mpe.observation_repair import RiskTriggeredObservationRepair
from onpolicy.scripts.eval.eval_mpe_b4 import (
    SequentialCorruptor, load_actor, make_conditions, risk_only_b2_gate)


POLICIES = ("B0", "B2", "RISK_SOFT")


def parse_args():
    parser = get_config()
    parser.add_argument("--scenario_name", type=str, default="simple_spread")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--b4_actor", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--eval_seed_start", type=int, default=92000)
    parser.add_argument("--eval_protocol",
                        choices=("development", "final", "mechanism"),
                        default="development")
    parser.add_argument("--eval_batch_size", type=int, default=20)
    parser.add_argument("--b3_threshold", type=float, default=0.03)
    parser.add_argument("--risk_soft_scale", type=float, default=200.0)
    parser.add_argument("--repair_smooth_alpha", type=float, default=0.35)
    parser.add_argument("--repair_max_velocity", type=float, default=0.20)
    parser.add_argument("--repair_modes", nargs="*",
                        choices=RiskTriggeredObservationRepair.VALID_MODES,
                        default=list(RiskTriggeredObservationRepair.VALID_MODES))
    parser.add_argument("--repair_policies", nargs="*", choices=POLICIES,
                        default=list(POLICIES),
                        help="policies evaluated behind each repair mode")
    parser.add_argument("--noise_stds", type=float, nargs="*",
                        default=[0.30, 0.40])
    parser.add_argument("--mask_probs", type=float, nargs="*",
                        default=[0.50, 0.60])
    parser.add_argument("--delay_steps", type=int, nargs="*", default=[3, 4])
    parser.add_argument("--composite_conditions", type=str, nargs="*",
                        default=["noise=0.20+mask=0.30",
                                 "noise=0.20+delay=2",
                                 "mask=0.30+delay=2"])
    parser.add_argument("--checkpoint_activation", choices=("tanh", "relu"),
                        default="tanh")
    return parser.parse_args()


def basic(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values)) if len(values) else 0.0
    return {"mean": mean, "std": std, "sem": sem,
            "ci95_low": mean - 1.96 * sem,
            "ci95_high": mean + 1.96 * sem}


def variant_names(modes, repair_policies=POLICIES):
    names = list(POLICIES)
    for mode in modes:
        names.extend("{}_{}".format(mode.upper(), policy)
                     for policy in repair_policies)
    return tuple(names)


def split_variant(variant):
    if variant in POLICIES:
        return None, variant
    for policy in POLICIES:
        suffix = "_" + policy
        if variant.endswith(suffix):
            return variant[:-len(suffix)].lower(), policy
    raise ValueError("unrecognized variant: {}".format(variant))


@torch.no_grad()
def run_batch(args, actor, action_dim, condition, condition_index, seeds,
              variants):
    records = []
    for seed in seeds:
        for variant in variants:
            mode, _ = split_variant(variant)
            env = MPEEnv(args)
            env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            corruptor = SequentialCorruptor(
                condition[2], args.num_agents, args.num_landmarks,
                seed=seed * 1009 + condition_index * 9176)
            corruptor.reset()
            repair = None
            if mode is not None:
                repair = RiskTriggeredObservationRepair(
                    mode, args.num_agents, args.num_landmarks,
                    risk_threshold=args.b3_threshold,
                    smooth_alpha=args.repair_smooth_alpha,
                    max_velocity=args.repair_max_velocity)
            repair_rng = np.random.default_rng(
                seed * 1000003 + condition_index * 9176 + 71)
            records.append({"seed": seed, "variant": variant, "env": env,
                            "obs": obs, "corruptor": corruptor,
                            "repair": repair, "repair_rng": repair_rng})

    batch_size = len(records)
    state_shape = (batch_size, args.num_agents, args.recurrent_N,
                   args.hidden_size)
    b2_states = np.zeros(state_shape, dtype=np.float32)
    b0_states = np.zeros(state_shape, dtype=np.float32)
    masks = np.ones((batch_size * args.num_agents, 1), dtype=np.float32)
    returns = np.zeros(batch_size, dtype=np.float64)
    sums = {name: np.zeros(batch_size, dtype=np.float64) for name in (
        "raw_risk", "final_risk", "requested", "intervention", "missing",
        "abs_change")}

    for _ in range(args.episode_length):
        raw_obs = np.stack([record["corruptor"].transform(record["obs"])
                            for record in records])
        flat_raw = raw_obs.reshape(batch_size * args.num_agents, -1)
        flat_b2 = b2_states.reshape(
            batch_size * args.num_agents, args.recurrent_N, args.hidden_size)
        flat_b0 = b0_states.reshape(
            batch_size * args.num_agents, args.recurrent_N, args.hidden_size)

        _, _, _, raw_diag, _, _ = actor._distribution_outputs(
            flat_raw, flat_b2, flat_b0, masks)
        raw_risk = raw_diag["detector_risk"].cpu().numpy().reshape(
            batch_size, args.num_agents)

        policy_obs = raw_obs.copy()
        for index, record in enumerate(records):
            if record["repair"] is None:
                continue
            mode = record["repair"].mode
            trigger_mask = None
            position_slice = record["repair"].position_slice
            raw_block = raw_obs[index, :, position_slice].reshape(
                args.num_agents, args.num_agents - 1, 2)
            missing = np.linalg.norm(raw_block, axis=-1) <= \
                record["repair"].zero_tolerance
            if mode == "random_smooth":
                high_risk = np.broadcast_to(
                    raw_risk[index, :, None] > args.b3_threshold,
                    missing.shape)
                eligible = record["repair"].valid & (~missing)
                count = int((high_risk & eligible).sum())
                candidates = np.flatnonzero(eligible.reshape(-1))
                trigger_mask = np.zeros_like(missing, dtype=bool)
                if count:
                    selected = record["repair_rng"].choice(
                        candidates, size=min(count, len(candidates)),
                        replace=False)
                    trigger_mask.reshape(-1)[selected] = True
            elif mode == "oracle_smooth":
                clean_block = record["obs"][:, position_slice].reshape(
                    args.num_agents, args.num_agents - 1, 2)
                trigger_mask = np.linalg.norm(
                    raw_block - clean_block, axis=-1) > 1e-7
            policy_obs[index], stats = record["repair"].transform(
                raw_obs[index], raw_risk[index], trigger_mask=trigger_mask)
            sums["requested"][index] += stats["requested_rate"]
            sums["intervention"][index] += stats["intervention_rate"]
            sums["missing"][index] += stats["missing_rate"]
            sums["abs_change"][index] += stats["mean_abs_change"]

        flat_policy_obs = policy_obs.reshape(batch_size * args.num_agents, -1)
        _, next_b2, next_b0, diagnostics, b0_probs, b2_probs = \
            actor._distribution_outputs(
                flat_policy_obs, flat_b2, flat_b0, masks)
        final_risk = diagnostics["detector_risk"].cpu().numpy().reshape(
            batch_size, args.num_agents)
        risk_gate = risk_only_b2_gate(
            final_risk, args.b3_threshold, args.risk_soft_scale)
        risk_gate_t = torch.as_tensor(
            risk_gate.reshape(-1, 1), device=b0_probs.device,
            dtype=b0_probs.dtype)
        probability_choices = {
            "B0": b0_probs,
            "B2": b2_probs,
            "RISK_SOFT": risk_gate_t * b2_probs
                         + (1.0 - risk_gate_t) * b0_probs,
        }
        action_choices = {
            name: probs.argmax(-1).cpu().numpy().reshape(
                batch_size, args.num_agents)
            for name, probs in probability_choices.items()
        }
        executed = np.stack([
            action_choices[split_variant(record["variant"])[1]][index]
            for index, record in enumerate(records)])
        env_actions = np.eye(action_dim, dtype=np.float32)[executed]
        for index, record in enumerate(records):
            obs, rewards, _, _ = record["env"].step(env_actions[index])
            record["obs"] = np.asarray(obs, dtype=np.float32)
            returns[index] += float(np.mean(rewards))
        sums["raw_risk"] += raw_risk.mean(axis=1)
        sums["final_risk"] += final_risk.mean(axis=1)
        b2_states = next_b2.cpu().numpy().reshape(state_shape)
        b0_states = next_b0.cpu().numpy().reshape(state_shape)

    rows = []
    for index, record in enumerate(records):
        rows.append({
            "algorithm": record["variant"],
            "eval_seed": record["seed"],
            "corruption_type": condition[0],
            "level": condition[1],
            "episode_return": returns[index],
            "raw_detector_risk": sums["raw_risk"][index] / args.episode_length,
            "final_detector_risk": sums["final_risk"][index] / args.episode_length,
            "repair_requested_rate": sums["requested"][index] / args.episode_length,
            "repair_intervention_rate": sums["intervention"][index] / args.episode_length,
            "observed_missing_rate": sums["missing"][index] / args.episode_length,
            "repair_mean_abs_change": sums["abs_change"][index] / args.episode_length,
        })
    for record in records:
        record["env"].close()
    return rows


def summarize(rows, variants):
    grouped = {variant: [] for variant in variants}
    for row in rows:
        grouped[row["algorithm"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["eval_seed"])
    arrays = {variant: np.asarray(
        [row["episode_return"] for row in grouped[variant]])
        for variant in variants}
    output = []
    for variant in variants:
        _, policy = split_variant(variant)
        item = {
            "algorithm": variant,
            "episodes": len(arrays[variant]),
            "return": basic(arrays[variant]),
            "paired_delta_vs_B0": basic(arrays[variant] - arrays["B0"]),
            "paired_delta_vs_raw_policy": basic(
                arrays[variant] - arrays[policy]),
        }
        for metric in ("raw_detector_risk", "final_detector_risk",
                       "repair_requested_rate", "repair_intervention_rate",
                       "observed_missing_rate", "repair_mean_abs_change"):
            item[metric] = basic([row[metric] for row in grouped[variant]])
        output.append(item)
    return output


def main():
    args = parse_args()
    bands = {"development": (92000, 93000), "final": (93000, 93200),
             "mechanism": (94000, 95000)}
    lower, upper = bands[args.eval_protocol]
    if args.eval_seed_start < lower or args.eval_seed_start >= upper:
        raise ValueError("{} eval seeds must stay in [{}, {})".format(
            args.eval_protocol, lower, upper))
    if args.eval_seed_start + args.eval_episodes > upper:
        raise ValueError("{} evaluation exceeds reserved seed band".format(
            args.eval_protocol))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actor, action_dim = load_actor(args, device)
    variants = variant_names(args.repair_modes, args.repair_policies)
    seeds = [args.eval_seed_start + index
             for index in range(args.eval_episodes)]
    conditions = make_conditions(args)
    all_rows, all_summaries = [], []
    print("device={} seeds={}..{} variants={} conditions={}".format(
        device, seeds[0], seeds[-1], len(variants), len(conditions)),
        flush=True)
    for condition_index, condition in enumerate(conditions):
        condition_rows = []
        for start in range(0, len(seeds), args.eval_batch_size):
            condition_rows.extend(run_batch(
                args, actor, action_dim, condition, condition_index,
                seeds[start:start + args.eval_batch_size], variants))
        summaries = summarize(condition_rows, variants)
        for item in summaries:
            item.update({"corruption_type": condition[0],
                         "level": condition[1]})
        all_rows.extend(condition_rows)
        all_summaries.extend(summaries)
        ranked = sorted(summaries,
                        key=lambda item: item["return"]["mean"], reverse=True)
        print("{:>16} {:>10}: best={} {:8.3f}; B0={:8.3f}; Risk={:8.3f}".format(
            condition[0], str(condition[1]), ranked[0]["algorithm"],
            ranked[0]["return"]["mean"], summaries[0]["return"]["mean"],
            summaries[2]["return"]["mean"]), flush=True)

    aggregate = summarize(all_rows, variants)
    with (args.output_dir / "episode_results.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    payload = {"protocol": {
        "eval_protocol": args.eval_protocol,
        "eval_seed_start": args.eval_seed_start,
        "eval_episodes": args.eval_episodes,
        "repair_modes": args.repair_modes,
        "repair_policies": args.repair_policies,
        "repair_smooth_alpha": args.repair_smooth_alpha,
        "repair_max_velocity": args.repair_max_velocity,
        "risk_threshold": args.b3_threshold,
        "conditions": [{"type": item[0], "level": item[1]}
                       for item in conditions],
    }, "by_condition": all_summaries, "aggregate": aggregate}
    with (args.output_dir / "summary.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print("saved results to {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
