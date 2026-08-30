import numpy as np

from onpolicy.envs.mpe.perturbations import (
    TeammatePositionCorruptor,
    teammate_position_slice,
)


def observations():
    return np.arange(3 * 18, dtype=np.float32).reshape(3, 18)


def test_simple_spread_three_agent_slice():
    position_slice = teammate_position_slice(3, 3)
    assert position_slice.start == 10
    assert position_slice.stop == 14


def test_noise_changes_only_teammate_positions_and_is_reproducible():
    obs = observations()
    a = TeammatePositionCorruptor("noise", 0.2, 3, 3, seed=7)
    b = TeammatePositionCorruptor("noise", 0.2, 3, 3, seed=7)
    out_a, out_b = a.transform(obs), b.transform(obs)
    np.testing.assert_array_equal(out_a, out_b)
    np.testing.assert_array_equal(out_a[:, :10], obs[:, :10])
    np.testing.assert_array_equal(out_a[:, 14:], obs[:, 14:])
    assert not np.array_equal(out_a[:, 10:14], obs[:, 10:14])


def test_full_mask_zeros_each_teammate_vector_only():
    obs = observations()
    corruptor = TeammatePositionCorruptor("mask", 1.0, 3, 3, seed=3)
    out = corruptor.transform(obs)
    np.testing.assert_array_equal(out[:, 10:14], 0.0)
    np.testing.assert_array_equal(out[:, :10], obs[:, :10])
    np.testing.assert_array_equal(out[:, 14:], obs[:, 14:])


def test_delay_uses_old_block_and_reset_prevents_episode_leakage():
    corruptor = TeammatePositionCorruptor("delay", 2, 3, 3, seed=1)
    obs0 = observations()
    obs1, obs2 = obs0 + 100, obs0 + 200
    out0 = corruptor.transform(obs0)
    out1 = corruptor.transform(obs1)
    out2 = corruptor.transform(obs2)
    np.testing.assert_array_equal(out0[:, 10:14], obs0[:, 10:14])
    np.testing.assert_array_equal(out1[:, 10:14], obs0[:, 10:14])
    np.testing.assert_array_equal(out2[:, 10:14], obs0[:, 10:14])
    corruptor.reset()
    fresh = corruptor.transform(obs2)
    np.testing.assert_array_equal(fresh[:, 10:14], obs2[:, 10:14])
