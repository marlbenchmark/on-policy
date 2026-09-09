#!/usr/bin/env python3
"""Summarize the held-out B4 selector-length validation without extra dependencies."""

import csv
import json
from pathlib import Path


ROOT = Path("onpolicy/scripts/results/MPE/simple_spread")
EVAL_START = 70000
EVAL_EPISODES = 500
LENGTHS = (1, 2, 5)
SEEDS = (1, 2, 3)
# A context selector must beat both the fixed risk gate and an exact
# coverage-matched random control by a practically meaningful margin.  Merely
# producing a positive floating-point delta is not mechanism evidence.
MIN_PRACTICAL_GAIN = 0.15


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def paired_delta(rows, left, right, subset):
    selected = [r for r in rows if subset(r)]
    by_algorithm = {}
    for row in selected:
        key = (int(row["eval_seed"]), row["corruption_type"], row["level"])
        by_algorithm.setdefault(row["algorithm"], {})[key] = float(row["episode_return"])
    common = sorted(set(by_algorithm[left]) & set(by_algorithm[right]))
    if not common:
        raise RuntimeError(f"no paired rows for {left} and {right}")
    return mean([by_algorithm[left][key] - by_algorithm[right][key] for key in common])


def main():
    records = []
    for length in LENGTHS:
        for seed in SEEDS:
            directory = ROOT / "headroom_b4" / (
                f"selector_length_{length}m_trainseed{seed}_validation_seed"
                f"{EVAL_START}_{EVAL_EPISODES}"
            )
            csv_path = directory / "episode_results.csv"
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            algorithms = {r["algorithm"] for r in rows}
            required = {"B4", "RISK_SOFT", "MATCHED_RANDOM"}
            if not required <= algorithms:
                raise RuntimeError(f"{csv_path}: missing {sorted(required - algorithms)}")
            all_rows = lambda _: True
            clean = lambda r: r["corruption_type"] == "clean"
            corrupt = lambda r: r["corruption_type"] != "clean"
            records.append(
                {
                    "length_m": length,
                    "train_seed": seed,
                    "b4_minus_risk_all": paired_delta(rows, "B4", "RISK_SOFT", all_rows),
                    "b4_minus_risk_clean": paired_delta(rows, "B4", "RISK_SOFT", clean),
                    "b4_minus_risk_corrupt": paired_delta(rows, "B4", "RISK_SOFT", corrupt),
                    "b4_minus_matched_random_all": paired_delta(
                        rows, "B4", "MATCHED_RANDOM", all_rows
                    ),
                }
            )

    summaries = []
    for length in LENGTHS:
        group = [r for r in records if r["length_m"] == length]
        deltas = [r["b4_minus_risk_all"] for r in group]
        clean_deltas = [r["b4_minus_risk_clean"] for r in group]
        random_deltas = [r["b4_minus_matched_random_all"] for r in group]
        summaries.append(
            {
                "length_m": length,
                "mean_b4_minus_risk_all": mean(deltas),
                "mean_b4_minus_risk_clean": mean(clean_deltas),
                "mean_b4_minus_risk_corrupt": mean(
                    [r["b4_minus_risk_corrupt"] for r in group]
                ),
                "mean_b4_minus_matched_random_all": mean(random_deltas),
                "positive_seeds_vs_risk": sum(value > 0 for value in deltas),
                "passes_numeric_rule": (
                    all(value > 0 for value in deltas)
                    and mean(deltas) >= MIN_PRACTICAL_GAIN
                    and mean(clean_deltas) >= -0.10
                    and all(value > 0 for value in random_deltas)
                    and mean(random_deltas) >= MIN_PRACTICAL_GAIN
                ),
            }
        )

    passing = [s for s in summaries if s["passes_numeric_rule"]]
    if passing:
        selected = max(passing, key=lambda s: s["mean_b4_minus_risk_all"])
        # Gate saturation/stability is stored in training logs rather than the
        # evaluation CSV, so a numeric pass still requires a manual log audit.
        decision = "NUMERIC_PASS_REQUIRES_GATE_AUDIT"
        selected_length = selected["length_m"]
        rationale = (
            f"{selected_length}M passes the return/random-control numeric rule. "
            "Audit gate stability before deciding whether to use the untouched "
            "80000-80199 final-test pool."
        )
    else:
        decision = "USE_RISK_ONLY_AS_MAIN_METHOD"
        selected_length = None
        rationale = (
            "No training length passes the predeclared consistency/effect/clean/random-control "
            "rule. Extra selector training does not rescue the learned gate on this validation set."
        )

    output = ROOT / "headroom_b4" / "selector_length_validation_decision"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "decision": decision,
        "selected_length_m": selected_length,
        "rationale": rationale,
        "records": records,
        "summaries": summaries,
    }
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# B4 selector training-length validation",
        "",
        f"Decision: **{decision}**",
        "",
        rationale,
        "",
        "| Length | mean B4-risk | corrupt | clean | B4-random | positive seeds | pass |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['length_m']}M | {item['mean_b4_minus_risk_all']:+.4f} | "
            f"{item['mean_b4_minus_risk_corrupt']:+.4f} | "
            f"{item['mean_b4_minus_risk_clean']:+.4f} | "
            f"{item['mean_b4_minus_matched_random_all']:+.4f} | "
            f"{item['positive_seeds_vs_risk']}/3 | "
            f"{'yes' if item['passes_numeric_rule'] else 'no'} |"
        )
    lines.extend(["", "## Per-seed paired deltas", ""])
    for row in records:
        lines.append(
            f"- {row['length_m']}M seed {row['train_seed']}: "
            f"B4-risk(all)={row['b4_minus_risk_all']:+.4f}, "
            f"clean={row['b4_minus_risk_clean']:+.4f}, "
            f"corrupt={row['b4_minus_risk_corrupt']:+.4f}, "
            f"B4-random={row['b4_minus_matched_random_all']:+.4f}"
        )
    (output / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
