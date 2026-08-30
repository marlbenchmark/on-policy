import numpy as np
import torch
from gym import spaces

from cstm_test_utils import make_args
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.cstm_mappo.algorithm.dual_policy_actor import B1Actor


def test_disabled_b1_is_exactly_b0():
    torch.manual_seed(3)
    obs_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
    action_space = spaces.Discrete(5)
    b0_args = make_args(algorithm_name="rmappo")
    b0 = R_Actor(b0_args, obs_space, action_space)
    b1_args = make_args(cstm_use_teammate_policy=False)
    b1 = B1Actor(b1_args, obs_space, action_space, b1_args.num_agents)
    result = b1.load_state_dict(b0.state_dict(), strict=False)
    assert not result.unexpected_keys

    obs = np.random.randn(10, 7).astype(np.float32)
    states = np.random.randn(10, 1, b0_args.hidden_size).astype(np.float32)
    masks = np.ones((10, 1), dtype=np.float32)
    with torch.no_grad():
        b0_out = b0(obs, states, masks, deterministic=True)
        b1_out = b1(obs, states, masks, deterministic=True)
    for expected, actual in zip(b0_out, b1_out):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_enabled_b1_starts_from_b0_policy():
    torch.manual_seed(7)
    obs_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
    action_space = spaces.Discrete(5)
    args = make_args()
    b0 = R_Actor(make_args(algorithm_name="rmappo"), obs_space, action_space)
    b1 = B1Actor(args, obs_space, action_space, args.num_agents)
    b1.load_state_dict(b0.state_dict(), strict=False)
    obs = np.random.randn(5, 7).astype(np.float32)
    states = np.zeros((5, 1, args.hidden_size), dtype=np.float32)
    masks = np.ones((5, 1), dtype=np.float32)
    with torch.no_grad():
        b0_out = b0(obs, states, masks, deterministic=True)
        b1_out = b1(obs, states, masks, deterministic=True)
    for expected, actual in zip(b0_out, b1_out):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
