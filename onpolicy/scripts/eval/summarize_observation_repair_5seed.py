#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path

import numpy as np


T95 = {2: 12.7062, 3: 4.3027, 4: 3.1824, 5: 2.7764}


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    critical = T95.get(len(values), 1.96)
    half = critical * sd / math.sqrt(len(values))
    return {"mean": mean, "std": sd, "ci95_low": mean - half,
            "ci95_high": mean + half, "positive_seeds": int((values > 0).sum()),
            "values": values.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    runs = []
    for seed in range(1, 6):
        path = args.root / "trainseed{}_seed93000_200".format(seed) / "summary.json"
        with path.open(encoding="utf-8") as handle:
            runs.append(json.load(handle))

    condition_keys = []
    for item in runs[0]["by_condition"]:
        if item["algorithm"] == "RISK_SMOOTH_RISK_SOFT":
            condition_keys.append((item["corruption_type"], str(item["level"])))
    result = {"primary_comparison":
              "RISK_SMOOTH_RISK_SOFT minus RISK_SOFT",
              "inferential_unit": "training_seed", "conditions": []}
    comparison_specs = {
        "repair_risk_vs_risk": ("RISK_SMOOTH_RISK_SOFT", "RISK_SOFT"),
        "repair_risk_vs_b0": ("RISK_SMOOTH_RISK_SOFT", "B0"),
        "repair_b0_vs_b0": ("RISK_SMOOTH_B0", "B0"),
        "repair_risk_vs_repair_b0":
            ("RISK_SMOOTH_RISK_SOFT", "RISK_SMOOTH_B0"),
    }
    comparison_matrices = {name: [] for name in comparison_specs}
    for run in runs:
        lookup = {(item["corruption_type"], str(item["level"]), item["algorithm"]): item
                  for item in run["by_condition"]}
        for name, (left, right) in comparison_specs.items():
            comparison_matrices[name].append([
                lookup[(kind, level, left)]["return"]["mean"]
                - lookup[(kind, level, right)]["return"]["mean"]
                for kind, level in condition_keys])
    comparison_matrices = {
        name: np.asarray(values)
        for name, values in comparison_matrices.items()
    }
    matrix = comparison_matrices["repair_risk_vs_risk"]
    for index, (kind, level) in enumerate(condition_keys):
        result["conditions"].append({"corruption_type": kind, "level": level,
                                     "paired_delta": stats(matrix[:, index])})
    groups = {
        "all": np.ones(len(condition_keys), dtype=bool),
        "clean": np.asarray([kind == "clean" for kind, _ in condition_keys]),
        "noise": np.asarray([kind == "noise" for kind, _ in condition_keys]),
        "mask": np.asarray([kind == "mask" for kind, _ in condition_keys]),
        "delay": np.asarray([kind == "delay" for kind, _ in condition_keys]),
        "composite": np.asarray(["+" in kind for kind, _ in condition_keys]),
    }
    result["comparisons"] = {
        comparison: {
            name: stats(values[:, selected].mean(axis=1))
            for name, selected in groups.items()
        }
        for comparison, values in comparison_matrices.items()
    }
    result["groups"] = result["comparisons"]["repair_risk_vs_risk"]
    output = args.root / "five_seed_summary.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    for name, item in result["groups"].items():
        print("{:<10} {:+.4f} [{:+.4f}, {:+.4f}] {}/5 positive".format(
            name, item["mean"], item["ci95_low"], item["ci95_high"],
            item["positive_seeds"]))
    print("saved {}".format(output))


if __name__ == "__main__":
    main()
