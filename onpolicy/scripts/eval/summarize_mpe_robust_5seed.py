#!/usr/bin/env python
"""Condition-balanced, training-seed-level robust baseline statistics."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ALGORITHMS = ("B0", "B2", "B3", "RISK_SOFT", "B4",
              "ROBUST_B0", "ROBUST_B2")
COMPARISONS = (
    ("ROBUST_B2", "ROBUST_B0"),
    ("ROBUST_B0", "RISK_SOFT"),
    ("ROBUST_B2", "RISK_SOFT"),
    ("ROBUST_B2", "B4"),
    ("RISK_SOFT", "B4"),
    ("ROBUST_B0", "B0"),
    ("ROBUST_B2", "B2"),
)
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--input_template", required=True,
                        help="directory name containing {seed}")
    parser.add_argument("--suite", choices=("id", "ood"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def condition_key(row):
    return row["corruption_type"], row["level"]


def condition_groups(row, suite):
    kind = row["corruption_type"]
    level = float(row["level"]) if "+" not in row["level"] else None
    groups = ["all_conditions"]
    if kind == "clean":
        return groups + ["clean"]
    groups.append("all_corruptions" if suite == "id" else "all_ood")
    if "+" in kind:
        return groups + ["composite"]
    groups.extend(["single", kind])
    if suite == "id":
        strong = ((kind == "noise" and level >= 0.20)
                  or (kind == "mask" and level >= 0.30)
                  or (kind == "delay" and level >= 2))
        if strong:
            groups.append("strong_known")
    else:
        extrapolation = ((kind == "noise" and level > 0.30)
                         or (kind == "mask" and level > 0.50)
                         or (kind == "delay" and level > 3))
        groups.append("single_extrapolation" if extrapolation
                      else "single_interpolation")
    return groups


def t95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    critical = T975.get(len(values), 1.96)
    half = critical * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n_train_seeds": int(len(values)), "mean": mean, "std": std,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "positive_seeds": int((values > 0).sum()),
            "values": [float(value) for value in values]}


def verdict(stats):
    if stats["ci95_low"] > 0:
        return "positive"
    if stats["ci95_high"] < 0:
        return "negative"
    return "inconclusive"


def load_seed_rows(args, train_seed):
    directory = args.results_root / args.input_template.format(seed=train_seed)
    path = directory / "episode_results.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    present = {row["algorithm"] for row in rows}
    missing = set(ALGORITHMS) - present
    if missing:
        raise ValueError("{} missing algorithms {}".format(path, sorted(missing)))
    return rows


def seed_condition_means(rows):
    buckets = defaultdict(list)
    group_members = defaultdict(set)
    for row in rows:
        key = condition_key(row)
        buckets[(key, row["algorithm"])].append(float(row["episode_return"]))
    result = {}
    for (key, algorithm), values in buckets.items():
        result[(key, algorithm)] = float(np.mean(values))
    return result


def main():
    args = parse_args()
    per_seed = {}
    group_conditions = defaultdict(set)
    for train_seed in range(1, 6):
        rows = load_seed_rows(args, train_seed)
        per_seed[train_seed] = seed_condition_means(rows)
        for row in rows:
            for group in condition_groups(row, args.suite):
                group_conditions[group].add(condition_key(row))

    report = {"suite": args.suite, "training_seeds": list(range(1, 6)),
              "groups": {}}
    for group, conditions in sorted(group_conditions.items()):
        seed_means = {algorithm: [] for algorithm in ALGORITHMS}
        for train_seed in range(1, 6):
            for algorithm in ALGORITHMS:
                values = [per_seed[train_seed][(key, algorithm)]
                          for key in sorted(conditions)]
                seed_means[algorithm].append(float(np.mean(values)))
        group_report = {
            "condition_count": len(conditions),
            "conditions": [list(key) for key in sorted(conditions)],
            "algorithms": {algorithm: t95(values)
                           for algorithm, values in seed_means.items()},
            "paired_comparisons": {},
        }
        for left, right in COMPARISONS:
            deltas = np.asarray(seed_means[left]) - np.asarray(seed_means[right])
            stats = t95(deltas)
            stats["verdict"] = verdict(stats)
            group_report["paired_comparisons"][left + "-" + right] = stats
        report["groups"][group] = group_report

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary_5seed.json"
    md_path = args.output_dir / "summary_5seed.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    lines = ["# Robust baseline five-seed summary", "",
             "All confidence intervals use paired training-seed differences "
             "and Student's t (n=5, df=4). Conditions are weighted equally.", ""]
    for group, values in report["groups"].items():
        lines.extend(["## {} ({} conditions)".format(
            group, values["condition_count"]), "",
            "| comparison | mean delta | 95% CI | positive seeds | verdict |",
            "|---|---:|---:|---:|---|"])
        for name, stats in values["paired_comparisons"].items():
            lines.append("| {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {}/5 | {} |".format(
                name, stats["mean"], stats["ci95_low"], stats["ci95_high"],
                stats["positive_seeds"], stats["verdict"]))
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("saved {} and {}".format(json_path, md_path))


if __name__ == "__main__":
    main()
