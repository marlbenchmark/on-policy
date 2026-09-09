import numpy as np

from onpolicy.scripts.eval.eval_mpe_b4 import (
    coverage_matched_random_gate,
    parse_composite_condition,
    risk_only_b2_gate,
)


def test_risk_only_gate_is_monotonic_and_centered():
    risk = np.array([0.0, 0.01, 0.03, 0.05, 0.10])
    gate = risk_only_b2_gate(risk, threshold=0.03, scale=200.0)
    assert np.all(gate[:-1] > gate[1:])
    np.testing.assert_allclose(gate[2], 0.5, rtol=0, atol=1e-12)


def test_coverage_matched_random_gate_preserves_values_and_other_variants():
    gate = np.arange(24, dtype=np.float64).reshape(6, 4) / 24.0
    variants = ["B0", "MATCHED_RANDOM", "B4",
                "MATCHED_RANDOM", "B4", "B3"]
    shuffled = coverage_matched_random_gate(
        gate, variants, np.random.default_rng(7))
    random_rows = [1, 3]
    source_rows = [2, 4]
    other_rows = [0, 2, 4, 5]
    np.testing.assert_array_equal(shuffled[other_rows], gate[other_rows])
    np.testing.assert_array_equal(
        np.sort(shuffled[random_rows].reshape(-1)),
        np.sort(gate[source_rows].reshape(-1)))
    assert not np.array_equal(shuffled[random_rows], gate[random_rows])


def test_composite_condition_parser_preserves_order_and_levels():
    condition = parse_composite_condition("noise=0.20+mask=0.30+delay=2")
    assert condition == (
        "noise+mask+delay", "0.2+0.3+2",
        (("noise", 0.2), ("mask", 0.3), ("delay", 2)))


if __name__ == "__main__":
    test_risk_only_gate_is_monotonic_and_centered()
    test_coverage_matched_random_gate_preserves_values_and_other_variants()
    test_composite_condition_parser_preserves_order_and_levels()
    print("B4_EVAL_ABLATION_TESTS_PASSED 3")
