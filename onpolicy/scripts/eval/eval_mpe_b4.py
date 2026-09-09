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
from onpolicy.algorithms.selective_mappo.selective_actor import SelectiveActor
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.cstm_mappo.algorithm.dual_policy_actor import B1Actor


def parse_args():
    parser = get_config()
    parser.add_argument("--scenario_name", type=str, default="simple_spread")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--b4_actor", type=Path, required=True)
    parser.add_argument("--robust_b0_actor", type=Path)
    parser.add_argument("--robust_b2_actor", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--eval_seed_start", type=int, default=20000)
    parser.add_argument("--eval_batch_size", type=int, default=25)
    parser.add_argument("--b3_threshold", type=float, default=0.03)
    parser.add_argument("--selector_ablations", action="store_true",
                        help="evaluate risk-only and coverage-matched random gates")
    parser.add_argument("--risk_soft_scale", type=float, default=200.0)
    parser.add_argument("--noise_stds", type=float, nargs="*",
                        default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--mask_probs", type=float, nargs="*",
                        default=[0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--delay_steps", type=int, nargs="*",
                        default=[1, 2, 3])
    parser.add_argument(
        "--composite_conditions", type=str, nargs="*", default=[],
        help=("ordered unseen corruptions, for example "
              "noise=0.20+mask=0.30; order is applied left-to-right"))
    parser.add_argument("--checkpoint_activation", choices=("tanh", "relu"),
                        default="tanh")
    parser.add_argument("--write_step_log", action="store_true")
    return parser.parse_args()


def parse_composite_condition(text):
    """Parse and label an ordered composition without hiding its components."""
    specs = []
    for item in text.split("+"):
        try:
            corruption_type, raw_level = item.split("=", 1)
        except ValueError as exc:
            raise ValueError("invalid composite condition: {}".format(text)) from exc
        corruption_type = corruption_type.strip().lower()
        if corruption_type not in ("noise", "mask", "delay"):
            raise ValueError("unsupported composite corruption: {}".format(
                corruption_type))
        level = float(raw_level)
        if level <= 0 or (corruption_type == "delay" and not level.is_integer()):
            raise ValueError("invalid {} level: {}".format(
                corruption_type, raw_level))
        level = int(level) if corruption_type == "delay" else level
        specs.append((corruption_type, level))
    if len(specs) < 2:
        raise ValueError("a composite condition needs at least two components")
    name = "+".join(kind for kind, _ in specs)
    label = "+".join(str(level) for _, level in specs)
    return name, label, tuple(specs)


class SequentialCorruptor:
    """Apply a frozen list of corruptions left-to-right."""

    def __init__(self, specs, num_agents, num_landmarks, seed):
        self.corruptors = [
            TeammatePositionCorruptor(
                kind, level, num_agents, num_landmarks,
                seed=seed + component_index * 104729)
            for component_index, (kind, level) in enumerate(specs)
        ]

    def reset(self):
        for corruptor in self.corruptors:
            corruptor.reset()

    def transform(self, observation):
        for corruptor in self.corruptors:
            observation = corruptor.transform(observation)
        return observation


def make_conditions(args):
    # The third item preserves the exact executable corruption specification.
    conditions = [("clean", 0.0, (("clean", 0.0),))]
    conditions.extend(("noise", float(x), (("noise", float(x)),))
                      for x in args.noise_stds if x > 0)
    conditions.extend(("mask", float(x), (("mask", float(x)),))
                      for x in args.mask_probs if x > 0)
    conditions.extend(("delay", int(x), (("delay", int(x)),))
                      for x in args.delay_steps if x > 0)
    conditions.extend(parse_composite_condition(text)
                      for text in args.composite_conditions)
    return conditions


def basic(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    sem = std / math.sqrt(values.size) if values.size else 0.0
    return {"mean": mean, "std": std, "sem": sem,
            "ci95_low": mean - 1.96 * sem,
            "ci95_high": mean + 1.96 * sem}


def risk_only_b2_gate(risk, threshold=0.03, scale=200.0):
    """Fixed monotonic soft gate: high detector risk means less B2 use."""
    logits = np.clip(scale * (threshold - np.asarray(risk)), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def coverage_matched_random_gate(b2_gate, variants, rng):
    """Assign shuffled B4 gate values to the random arm."""
    output = np.asarray(b2_gate).copy()
    targets = [index for index, variant in enumerate(variants)
               if variant == "MATCHED_RANDOM"]
    sources = [index for index, variant in enumerate(variants)
               if variant == "B4"]
    if not targets:
        return output
    if len(targets) != len(sources):
        raise ValueError("MATCHED_RANDOM and B4 batch sizes must match")
    values = output[sources].reshape(-1).copy()
    output[targets] = values[rng.permutation(values.size)].reshape(
        len(targets), *output.shape[1:])
    return output


def load_actor(args, device):
    common = copy.deepcopy(args)
    common.algorithm_name = "selective_mappo"
    common.use_recurrent_policy = True
    common.use_naive_recurrent_policy = False
    common.use_ReLU = args.checkpoint_activation == "relu"
    common.cstm_num_heads = 3
    common.cstm_use_separate_detector = True
    common.cstm_use_uncertainty_feature = True
    common.cstm_random_prior_scale = 0.5
    space_env = MPEEnv(common)
    actor = SelectiveActor(
        common, space_env.observation_space[0], space_env.action_space[0],
        args.num_agents, device)
    actor.load_state_dict(torch.load(args.b4_actor, map_location=device))
    actor.eval()
    action_dim = space_env.action_space[0].n
    space_env.close()
    return actor, action_dim


def load_robust_actors(args, device):
    paths = (args.robust_b0_actor, args.robust_b2_actor)
    if paths == (None, None):
        return None
    if any(path is None for path in paths):
        raise ValueError("both robust actor paths must be provided")
    common = copy.deepcopy(args)
    common.use_recurrent_policy = True
    common.use_naive_recurrent_policy = False
    common.use_ReLU = args.checkpoint_activation == "relu"
    common.cstm_num_heads = 3
    common.cstm_use_separate_detector = False
    common.cstm_use_uncertainty_feature = True
    common.cstm_random_prior_scale = 0.0
    space_env = MPEEnv(common)
    common.algorithm_name = "rmappo"
    b0 = R_Actor(common, space_env.observation_space[0],
                 space_env.action_space[0], device)
    common.algorithm_name = "ua_rep_mappo"
    b2 = B1Actor(common, space_env.observation_space[0],
                 space_env.action_space[0], args.num_agents, device)
    b0.load_state_dict(torch.load(args.robust_b0_actor, map_location=device))
    b2.load_state_dict(torch.load(args.robust_b2_actor, map_location=device))
    b0.eval(); b2.eval(); space_env.close()
    return {"ROBUST_B0": b0, "ROBUST_B2": b2}


@torch.no_grad()
def run_batch(args, actor, action_dim, condition, condition_index, seeds,
              variants, step_writer=None, robust_actors=None):
    records = []
    for seed in seeds:
        for variant in variants:
            env = MPEEnv(args)
            env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            corruptor = SequentialCorruptor(
                condition[2], args.num_agents, args.num_landmarks,
                seed=seed * 1009 + condition_index * 9176)
            corruptor.reset()
            records.append({"seed": seed, "variant": variant, "env": env,
                            "obs": obs, "corruptor": corruptor})
    batch_size = len(records)
    state_shape = (batch_size, args.num_agents, args.recurrent_N,
                   args.hidden_size)
    b2_states = np.zeros(state_shape, dtype=np.float32)
    b0_states = np.zeros(state_shape, dtype=np.float32)
    masks = np.ones((batch_size * args.num_agents, 1), dtype=np.float32)
    robust_states = ({name: np.zeros(state_shape, dtype=np.float32)
                      for name in robust_actors} if robust_actors else {})
    returns = np.zeros(batch_size, dtype=np.float64)
    metric_names = ("agent_steps", "b3_fallback", "soft_fallback",
                    "hard_fallback", "risk_soft_fallback",
                    "matched_random_fallback", "branch_disagreement",
                    "b4_equals_b0", "b4_equals_b2", "risk")
    sums = {key: np.zeros(batch_size, dtype=np.float64)
            for key in metric_names}
    variant_names = [record["variant"] for record in records]
    random_rng = np.random.default_rng(
        condition_index * 1000003 + seeds[0] * 9176 + len(seeds))

    for step in range(args.episode_length):
        actor_obs = np.stack([
            record["corruptor"].transform(record["obs"])
            for record in records])
        flat_obs = actor_obs.reshape(batch_size * args.num_agents, -1)
        flat_b2 = b2_states.reshape(
            batch_size * args.num_agents, args.recurrent_N, args.hidden_size)
        flat_b0 = b0_states.reshape(
            batch_size * args.num_agents, args.recurrent_N, args.hidden_size)
        mixed, next_b2, next_b0, diagnostics, b0_probs, b2_probs = \
            actor._distribution_outputs(flat_obs, flat_b2, flat_b0, masks)
        b0_actions = b0_probs.argmax(-1).cpu().numpy().reshape(
            batch_size, args.num_agents)
        b2_actions = b2_probs.argmax(-1).cpu().numpy().reshape(
            batch_size, args.num_agents)
        b4_actions = mixed.argmax(-1).cpu().numpy().reshape(
            batch_size, args.num_agents)
        risk = diagnostics["detector_risk"].cpu().numpy().reshape(
            batch_size, args.num_agents)
        b2_gate = diagnostics["b2_gate"].cpu().numpy().reshape(
            batch_size, args.num_agents)
        b3_fallback = risk > args.b3_threshold
        b3_actions = np.where(b3_fallback, b0_actions, b2_actions)
        choices = {"B0": b0_actions, "B2": b2_actions,
                   "B3": b3_actions, "B4": b4_actions}
        next_robust_states = {}
        for name, robust_actor in (robust_actors or {}).items():
            robust_actions, _, next_states = robust_actor(
                flat_obs, robust_states[name].reshape(
                    batch_size * args.num_agents, args.recurrent_N,
                    args.hidden_size), masks, deterministic=True)
            choices[name] = robust_actions.cpu().numpy().reshape(
                batch_size, args.num_agents)
            next_robust_states[name] = next_states.cpu().numpy().reshape(
                state_shape)
        risk_soft_gate = risk_only_b2_gate(
            risk, args.b3_threshold, args.risk_soft_scale)
        matched_random_gate = coverage_matched_random_gate(
            b2_gate, variant_names, random_rng)
        if args.selector_ablations:
            risk_gate_t = torch.as_tensor(
                risk_soft_gate.reshape(-1, 1), device=b0_probs.device,
                dtype=b0_probs.dtype)
            random_gate_t = torch.as_tensor(
                matched_random_gate.reshape(-1, 1), device=b0_probs.device,
                dtype=b0_probs.dtype)
            risk_soft_actions = (risk_gate_t * b2_probs
                                 + (1.0 - risk_gate_t) * b0_probs).argmax(-1)
            matched_random_actions = (random_gate_t * b2_probs
                                      + (1.0 - random_gate_t)
                                      * b0_probs).argmax(-1)
            choices["RISK_SOFT"] = risk_soft_actions.cpu().numpy().reshape(
                batch_size, args.num_agents)
            choices["MATCHED_RANDOM"] = \
                matched_random_actions.cpu().numpy().reshape(
                    batch_size, args.num_agents)
        executed = np.stack([
            choices[record["variant"]][index]
            for index, record in enumerate(records)])
        disagreement = b0_actions != b2_actions

        sums["agent_steps"] += args.num_agents
        sums["b3_fallback"] += b3_fallback.sum(axis=1)
        sums["soft_fallback"] += (1.0 - b2_gate).sum(axis=1)
        sums["hard_fallback"] += (b2_gate < 0.5).sum(axis=1)
        sums["risk_soft_fallback"] += (1.0 - risk_soft_gate).sum(axis=1)
        sums["matched_random_fallback"] += \
            (1.0 - matched_random_gate).sum(axis=1)
        sums["branch_disagreement"] += disagreement.sum(axis=1)
        sums["b4_equals_b0"] += (b4_actions == b0_actions).sum(axis=1)
        sums["b4_equals_b2"] += (b4_actions == b2_actions).sum(axis=1)
        sums["risk"] += risk.sum(axis=1)

        if step_writer is not None:
            for index, record in enumerate(records):
                for agent_id in range(args.num_agents):
                    step_writer.writerow({
                        "eval_seed": record["seed"],
                        "corruption_type": condition[0], "level": condition[1],
                        "algorithm": record["variant"], "step": step,
                        "agent_id": agent_id,
                        "detector_risk": risk[index, agent_id],
                        "b2_gate": b2_gate[index, agent_id],
                        "soft_fallback": 1.0 - b2_gate[index, agent_id],
                        "b3_fallback": int(b3_fallback[index, agent_id]),
                        "b0_action": int(b0_actions[index, agent_id]),
                        "b2_action": int(b2_actions[index, agent_id]),
                        "b3_action": int(b3_actions[index, agent_id]),
                        "b4_action": int(b4_actions[index, agent_id]),
                        "executed_action": int(executed[index, agent_id]),
                    })

        env_actions = np.eye(action_dim, dtype=np.float32)[executed]
        for index, record in enumerate(records):
            obs, rewards, _, _ = record["env"].step(env_actions[index])
            record["obs"] = np.asarray(obs, dtype=np.float32)
            returns[index] += float(np.mean(rewards))
        b2_states = next_b2.cpu().numpy().reshape(state_shape)
        b0_states = next_b0.cpu().numpy().reshape(state_shape)
        robust_states.update(next_robust_states)

    rows = []
    for index, record in enumerate(records):
        denom = sums["agent_steps"][index]
        rows.append({
            "algorithm": record["variant"], "eval_seed": record["seed"],
            "corruption_type": condition[0], "level": condition[1],
            "episode_return": returns[index],
            "mean_detector_risk": sums["risk"][index] / denom,
            "b3_fallback_rate": sums["b3_fallback"][index] / denom,
            "b4_soft_fallback_rate": sums["soft_fallback"][index] / denom,
            "b4_hard_fallback_rate": sums["hard_fallback"][index] / denom,
            "risk_soft_fallback_rate":
                sums["risk_soft_fallback"][index] / denom,
            "matched_random_fallback_rate":
                sums["matched_random_fallback"][index] / denom,
            "branch_action_disagreement_rate":
                sums["branch_disagreement"][index] / denom,
            "b4_action_equals_b0_rate": sums["b4_equals_b0"][index] / denom,
            "b4_action_equals_b2_rate": sums["b4_equals_b2"][index] / denom,
        })
    for record in records:
        record["env"].close()
    return rows


def summarize(rows, variants):
    by_variant = {variant: [] for variant in variants}
    for row in rows:
        by_variant[row["algorithm"]].append(row)
    for values in by_variant.values():
        values.sort(key=lambda row: row["eval_seed"])
    arrays = {key: np.asarray([row["episode_return"] for row in values])
              for key, values in by_variant.items()}
    oracle = np.maximum(arrays["B0"], arrays["B2"])
    summaries = []
    metrics = ("mean_detector_risk", "b3_fallback_rate",
               "b4_soft_fallback_rate", "b4_hard_fallback_rate",
               "risk_soft_fallback_rate", "matched_random_fallback_rate",
               "branch_action_disagreement_rate",
               "b4_action_equals_b0_rate", "b4_action_equals_b2_rate")
    for variant in variants:
        item = {"algorithm": variant, "episodes": len(arrays[variant]),
                "return": basic(arrays[variant]),
                "oracle_mean": float(oracle.mean()),
                "oracle_gap": float(oracle.mean() - arrays[variant].mean()),
                "paired_deltas": {}, "win_rates": {}}
        for baseline in ("B0", "B2", "B3"):
            item["paired_deltas"]["vs_" + baseline] = basic(
                arrays[variant] - arrays[baseline])
            item["win_rates"]["vs_" + baseline] = float(
                (arrays[variant] > arrays[baseline]).mean())
        for metric in metrics:
            item[metric] = basic([row[metric]
                                  for row in by_variant[variant]])
        summaries.append(item)
    return summaries


def main():
    args = parse_args()
    variants = ("B0", "B2", "B3", "B4")
    if args.selector_ablations:
        variants = ("B0", "B2", "B3", "RISK_SOFT",
                    "MATCHED_RANDOM", "B4")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actor, action_dim = load_actor(args, device)
    robust_actors = load_robust_actors(args, device)
    if robust_actors:
        variants = variants[:-1] + ("ROBUST_B0", "ROBUST_B2", "B4")
    step_handle = None
    step_writer = None
    if args.write_step_log:
        step_handle = (args.output_dir / "steps.csv").open(
            "w", newline="", encoding="utf-8")
        fields = ("eval_seed", "corruption_type", "level", "algorithm",
                  "step", "agent_id", "detector_risk", "b2_gate",
                  "soft_fallback", "b3_fallback", "b0_action", "b2_action",
                  "b3_action", "b4_action", "executed_action")
        step_writer = csv.DictWriter(step_handle, fieldnames=fields)
        step_writer.writeheader()
    all_rows, all_summaries = [], []
    seeds = [args.eval_seed_start + i for i in range(args.eval_episodes)]
    conditions = make_conditions(args)
    print("device={} episodes={} conditions={}".format(
        device, len(seeds), len(conditions)), flush=True)
    for condition_index, condition in enumerate(conditions):
        condition_rows = []
        for start in range(0, len(seeds), args.eval_batch_size):
            condition_rows.extend(run_batch(
                args, actor, action_dim, condition, condition_index,
                seeds[start:start + args.eval_batch_size], variants,
                step_writer, robust_actors))
        condition_summaries = summarize(condition_rows, variants)
        for item in condition_summaries:
            item.update({"corruption_type": condition[0],
                         "level": condition[1]})
        all_rows.extend(condition_rows)
        all_summaries.extend(condition_summaries)
        values = {item["algorithm"]: item["return"]["mean"]
                  for item in condition_summaries}
        b4 = condition_summaries[-1]
        extra = ""
        if args.selector_ablations:
            extra = " RiskSoft={:8.3f} Random={:8.3f}".format(
                values["RISK_SOFT"], values["MATCHED_RANDOM"])
        if robust_actors:
            extra += " RobustB0={:8.3f} RobustB2={:8.3f}".format(
                values["ROBUST_B0"], values["ROBUST_B2"])
        print("{:>5} {:>4}: B0={:8.3f} B2={:8.3f} B3={:8.3f} B4={:8.3f}{} "
              "dB3={:+.3f} fb={:.3f}".format(
                  condition[0], str(condition[1]), values["B0"], values["B2"],
                  values["B3"], values["B4"], extra,
                  b4["paired_deltas"]["vs_B3"]["mean"],
                  b4["b4_soft_fallback_rate"]["mean"]), flush=True)
    if step_handle is not None:
        step_handle.close()
    with (args.output_dir / "episode_results.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader(); writer.writerows(all_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as h:
        json.dump(all_summaries, h, indent=2, ensure_ascii=False)
    print("saved results to {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
