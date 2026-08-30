import inspect

import numpy as np
import torch
from gym import spaces

from cstm_test_utils import make_args
from onpolicy.algorithms.cstm_mappo.algorithm.dual_policy_actor import B1Actor


def test_targets_are_not_actor_inputs():
    args = make_args()
    actor = B1Actor(args, spaces.Box(-1, 1, shape=(7,), dtype=np.float32),
                    spaces.Discrete(5), args.num_agents)
    parameters = inspect.signature(actor.forward).parameters
    assert "teammate_actions" not in parameters
    assert "share_obs" not in parameters

    obs = np.random.randn(4, 7).astype(np.float32)
    states = np.zeros((4, 1, args.hidden_size), dtype=np.float32)
    masks = np.ones((4, 1), dtype=np.float32)
    fake_targets_a = np.zeros((4, 2, 1), dtype=np.int64)
    fake_targets_b = np.full((4, 2, 1), 4, dtype=np.int64)
    with torch.no_grad():
        out_a = actor(obs, states, masks, deterministic=True)[:2]
        # Changing privileged labels cannot affect an interface that never sees them.
        assert not np.array_equal(fake_targets_a, fake_targets_b)
        out_b = actor(obs, states, masks, deterministic=True)[:2]
    for first, second in zip(out_a, out_b):
        torch.testing.assert_close(first, second)
