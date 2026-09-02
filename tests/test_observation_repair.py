import numpy as np

from onpolicy.envs.mpe.observation_repair import RiskTriggeredObservationRepair
from onpolicy.envs.mpe.perturbations import teammate_position_slice


def observation(block):
    obs = np.zeros((3, 18), dtype=np.float32)
    obs[:, teammate_position_slice(3, 3)] = np.asarray(
        block, dtype=np.float32).reshape(3, -1)
    return obs


def extracted(obs):
    return obs[:, teammate_position_slice(3, 3)].reshape(3, 2, 2)


def test_mask_hold_fills_only_missing_slots_after_history():
    repair = RiskTriggeredObservationRepair("mask_hold", 3, 3)
    first = np.arange(12, dtype=np.float32).reshape(3, 2, 2) + 1
    repair.transform(observation(first), np.ones(3))
    second = first + 1
    second[0, 1] = 0
    output, stats = repair.transform(observation(second), np.ones(3))
    actual = extracted(output)
    np.testing.assert_allclose(actual[0, 1], first[0, 1])
    np.testing.assert_allclose(actual[0, 0], second[0, 0])
    assert stats["intervention_rate"] == 1.0 / 6.0


def test_risk_hold_does_not_change_clean_low_risk_observation():
    repair = RiskTriggeredObservationRepair("risk_hold", 3, 3)
    first = np.arange(12, dtype=np.float32).reshape(3, 2, 2) + 1
    repair.transform(observation(first), np.zeros(3))
    second = first + 0.1
    output, stats = repair.transform(observation(second), np.zeros(3))
    np.testing.assert_allclose(extracted(output), second)
    assert stats["intervention_rate"] == 0.0


def test_velocity_extrapolates_only_triggered_agent():
    repair = RiskTriggeredObservationRepair(
        "risk_velocity", 3, 3, max_velocity=0.2)
    first = np.arange(12, dtype=np.float32).reshape(3, 2, 2) + 1
    second = first + 0.1
    repair.transform(observation(first), np.zeros(3))
    repair.transform(observation(second), np.zeros(3))
    third = second + 5.0
    output, stats = repair.transform(
        observation(third), np.array([0.2, 0.0, 0.0]))
    actual = extracted(output)
    np.testing.assert_allclose(actual[0], second[0] + 0.1, atol=1e-6)
    np.testing.assert_allclose(actual[1:], third[1:])
    assert stats["intervention_rate"] == 2.0 / 6.0


def test_first_step_missing_has_no_fabricated_repair():
    repair = RiskTriggeredObservationRepair("risk_hold", 3, 3)
    zeros = np.zeros((3, 2, 2), dtype=np.float32)
    output, stats = repair.transform(observation(zeros), np.ones(3))
    np.testing.assert_allclose(extracted(output), zeros)
    assert stats["requested_rate"] == 1.0
    assert stats["intervention_rate"] == 0.0


def test_always_smooth_intervenes_everywhere_after_history():
    repair = RiskTriggeredObservationRepair(
        "always_smooth", 3, 3, smooth_alpha=0.5)
    first = np.arange(12, dtype=np.float32).reshape(3, 2, 2) + 1
    second = first + 2
    repair.transform(observation(first), np.zeros(3))
    output, stats = repair.transform(observation(second), np.zeros(3))
    np.testing.assert_allclose(extracted(output), first + 1)
    assert stats["intervention_rate"] == 1.0


def test_external_trigger_controls_random_and_oracle_modes():
    for mode in ("random_smooth", "oracle_smooth"):
        repair = RiskTriggeredObservationRepair(
            mode, 3, 3, smooth_alpha=0.5)
        first = np.arange(12, dtype=np.float32).reshape(3, 2, 2) + 1
        second = first + 2
        repair.transform(
            observation(first), np.zeros(3),
            trigger_mask=np.zeros((3, 2), dtype=bool))
        trigger = np.zeros((3, 2), dtype=bool)
        trigger[1, 0] = True
        output, stats = repair.transform(
            observation(second), np.zeros(3), trigger_mask=trigger)
        actual = extracted(output)
        np.testing.assert_allclose(actual[1, 0], first[1, 0] + 1)
        np.testing.assert_allclose(actual[0], second[0])
        assert stats["intervention_rate"] == 1.0 / 6.0


def test_external_trigger_is_required_and_shape_checked():
    repair = RiskTriggeredObservationRepair("random_smooth", 3, 3)
    obs = observation(np.ones((3, 2, 2), dtype=np.float32))
    try:
        repair.transform(obs, np.zeros(3))
        raise AssertionError("missing trigger_mask should fail")
    except ValueError as exc:
        assert "requires trigger_mask" in str(exc)


def test_high_risk_first_step_seeds_history_without_fabrication():
    repair = RiskTriggeredObservationRepair(
        "risk_smooth", 3, 3, smooth_alpha=0.5)
    first = np.arange(12, dtype=np.float32).reshape(3, 2, 2) + 1
    first_output, first_stats = repair.transform(
        observation(first), np.ones(3))
    np.testing.assert_allclose(extracted(first_output), first)
    assert first_stats["intervention_rate"] == 0.0
    second = first + 2
    second_output, second_stats = repair.transform(
        observation(second), np.ones(3))
    np.testing.assert_allclose(extracted(second_output), first + 1)
    assert second_stats["intervention_rate"] == 1.0


if __name__ == "__main__":
    test_mask_hold_fills_only_missing_slots_after_history()
    test_risk_hold_does_not_change_clean_low_risk_observation()
    test_velocity_extrapolates_only_triggered_agent()
    test_first_step_missing_has_no_fabricated_repair()
    test_always_smooth_intervenes_everywhere_after_history()
    test_external_trigger_controls_random_and_oracle_modes()
    test_external_trigger_is_required_and_shape_checked()
    test_high_risk_first_step_seeds_history_without_fabrication()
    print("observation repair tests passed")
