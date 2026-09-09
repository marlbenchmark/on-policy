import numpy as np
import unittest

from onpolicy.scripts.eval.eval_mpe_b3_gate import (
    make_variants,
    reduce_detector_risk,
    select_actions,
)


class B3GateTest(unittest.TestCase):
    def test_detector_risk_reductions_record_mean_and_max_semantics(self):
        uncertainty = np.asarray([[[0.01, 0.03], [0.02, 0.08]]])
        np.testing.assert_allclose(
            reduce_detector_risk(uncertainty, "mean"), [[0.02, 0.05]])
        np.testing.assert_allclose(
            reduce_detector_risk(uncertainty, "max"), [[0.03, 0.08]])

    def test_gate_falls_back_only_when_risk_strictly_exceeds_threshold(self):
        b0 = np.asarray([1, 2, 3])
        b2 = np.asarray([4, 4, 4])
        actions, fallback = select_actions(
            b0, b2, np.asarray([0.01, 0.02, 0.03]), 0.02)
        np.testing.assert_array_equal(fallback, [False, False, True])
        np.testing.assert_array_equal(actions, [4, 4, 3])

    def test_variants_include_exact_fixed_policy_controls(self):
        variants = make_variants([0.01, 0.025], "mean")
        self.assertEqual(variants[0], {"name": "B0", "threshold": -np.inf})
        self.assertEqual(variants[1], {"name": "B2", "threshold": np.inf})
        self.assertEqual([item["name"] for item in variants[2:]],
                         ["gate_mean_0.01", "gate_mean_0.025"])

    def test_invalid_thresholds_are_rejected(self):
        for thresholds in ([-0.1], [float("nan")], [0.1, 0.1]):
            with self.subTest(thresholds=thresholds):
                with self.assertRaises(ValueError):
                    make_variants(thresholds, "mean")


if __name__ == "__main__":
    unittest.main()
