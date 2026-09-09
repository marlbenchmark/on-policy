from types import SimpleNamespace

import numpy as np

from onpolicy.runner.shared.robust_mpe_runner import build_corruption_training


def args(**overrides):
    values = dict(
        scenario_name="simple_spread",
        seed=7,
        num_landmarks=3,
        cstm_selector_clean_probability=0.25,
        cstm_selector_noise_levels=[0.05, 0.10, 0.20, 0.30],
        cstm_selector_mask_levels=[0.10, 0.20, 0.30, 0.50],
        cstm_selector_delay_levels=[1, 2, 3],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_corruption_mix_is_deterministic_and_complete():
    first, _ = build_corruption_training(args(), 128, 3)
    second, _ = build_corruption_training(args(), 128, 3)
    assert first == second
    assert len(first) == 128
    assert {kind for kind, _ in first} == {"clean", "noise", "mask", "delay"}


def test_all_clean_mix_preserves_observations():
    conditions, corruptors = build_corruption_training(
        args(cstm_selector_clean_probability=1.0), 4, 3)
    assert conditions == [("clean", 0.0)] * 4
    obs = np.arange(54, dtype=np.float32).reshape(3, 18)
    for corruptor in corruptors:
        np.testing.assert_array_equal(corruptor.transform(obs), obs)


if __name__ == "__main__":
    test_corruption_mix_is_deterministic_and_complete()
    test_all_clean_mix_preserves_observations()
    print("ROBUST_MPE_RUNNER_TESTS_PASSED 2")
