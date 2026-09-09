#!/usr/bin/env python
"""Exact frozen-branch checks and empirical B4 gate-curve export."""

import argparse
import csv
from pathlib import Path

import torch


def load(path):
    return torch.load(Path(path), map_location="cpu")


def assert_exact(left, right, label):
    if left.keys() != right.keys():
        missing = sorted(left.keys() - right.keys())
        extra = sorted(right.keys() - left.keys())
        raise AssertionError(
            f"{label} key mismatch: missing={missing[:5]} extra={extra[:5]}")
    changed = [key for key in left if not torch.equal(left[key], right[key])]
    if changed:
        raise AssertionError(f"{label} changed tensors: {changed[:10]}")
    print(f"PASS {label}: {len(left)} tensors exactly equal")


def verify_detector(args):
    base_actor = load(args.base_actor)
    trained_actor = load(args.trained_actor)
    shared = {key: trained_actor[key] for key in base_actor}
    assert_exact(base_actor, shared, "detector frozen B2 actor")
    new_keys = sorted(set(trained_actor) - set(base_actor))
    if not new_keys or not all(
            key.startswith("uncertainty_detector.") for key in new_keys):
        raise AssertionError(f"unexpected detector-only keys: {new_keys[:10]}")
    print(f"PASS detector additions: {len(new_keys)} detector tensors")
    assert_exact(load(args.base_critic), load(args.trained_critic),
                 "detector frozen B2 critic")


def prefixed(state, prefix):
    return {key[len(prefix):]: value for key, value in state.items()
            if key.startswith(prefix)}


def verify_b4(args):
    trained = load(args.trained_actor)
    assert_exact(load(args.b0_actor), prefixed(trained, "b0_actor."),
                 "B4 frozen B0 branch")
    assert_exact(load(args.b2_actor), prefixed(trained, "b2_actor."),
                 "B4 frozen B2+detector branch")
    selector = [key for key in trained
                if not key.startswith(("b0_actor.", "b2_actor."))]
    if not selector:
        raise AssertionError("B4 checkpoint has no selector tensors")
    print(f"PASS B4 selector tensors present: {len(selector)}")


def gate_curve(args):
    values = []
    with Path(args.steps).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("algorithm") == "B4":
                values.append((float(row["detector_risk"]),
                               float(row["b2_gate"])))
    if not values:
        raise AssertionError("no B4 rows found in steps.csv")
    values.sort()
    bins = max(1, min(args.bins, len(values)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "bin", "count", "risk_min", "risk_max", "mean_risk",
            "mean_b2_gate", "mean_fallback_probability"])
        writer.writeheader()
        for index in range(bins):
            lo = index * len(values) // bins
            hi = (index + 1) * len(values) // bins
            chunk = values[lo:hi]
            mean_risk = sum(item[0] for item in chunk) / len(chunk)
            mean_gate = sum(item[1] for item in chunk) / len(chunk)
            writer.writerow({
                "bin": index + 1, "count": len(chunk),
                "risk_min": chunk[0][0], "risk_max": chunk[-1][0],
                "mean_risk": mean_risk, "mean_b2_gate": mean_gate,
                "mean_fallback_probability": 1.0 - mean_gate,
            })
    print(f"PASS gate curve: {len(values)} B4 agent-steps -> {output}")


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    detector = sub.add_parser("detector")
    detector.add_argument("--base-actor", required=True)
    detector.add_argument("--trained-actor", required=True)
    detector.add_argument("--base-critic", required=True)
    detector.add_argument("--trained-critic", required=True)
    detector.set_defaults(func=verify_detector)
    b4 = sub.add_parser("b4")
    b4.add_argument("--b0-actor", required=True)
    b4.add_argument("--b2-actor", required=True)
    b4.add_argument("--trained-actor", required=True)
    b4.set_defaults(func=verify_b4)
    curve = sub.add_parser("gate-curve")
    curve.add_argument("--steps", required=True)
    curve.add_argument("--output", required=True)
    curve.add_argument("--bins", type=int, default=20)
    curve.set_defaults(func=gate_curve)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
