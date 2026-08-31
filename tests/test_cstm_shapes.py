import numpy as np
import torch
from gym import spaces

from cstm_test_utils import make_args
from onpolicy.algorithms.cstm_mappo.algorithm.dual_policy_actor import B1Actor
from onpolicy.utils.cstm_buffer import CSTMReplayBuffer


def test_teammate_target_order_and_shape():
    actions = np.array([[[0], [1], [2]], [[3], [4], [0]]])
    targets, masks = CSTMReplayBuffer.build_teammate_targets(actions)
    assert targets.shape == (2, 3, 2, 1)
    assert masks.shape == targets.shape
    np.testing.assert_array_equal(targets[0, 0, :, 0], [1, 2])
    np.testing.assert_array_equal(targets[0, 1, :, 0], [0, 2])
    np.testing.assert_array_equal(targets[0, 2, :, 0], [0, 1])


def test_actor_and_decoder_shapes():
    args = make_args()
    actor = B1Actor(args, spaces.Box(-1, 1, shape=(7,), dtype=np.float32),
                    spaces.Discrete(5), args.num_agents)
    batch = 6
    obs = np.random.randn(batch, 7).astype(np.float32)
    states = np.zeros((batch, 1, args.hidden_size), dtype=np.float32)
    masks = np.ones((batch, 1), dtype=np.float32)
    actions, log_probs, next_states = actor(obs, states, masks)
    logits, _ = actor.teammate_predictions(obs, states, masks)
    assert actions.shape == (batch, 1)
    assert log_probs.shape == (batch, 1)
    assert next_states.shape == (batch, 1, args.hidden_size)
    assert logits.shape == (batch, args.num_agents - 1, 5)
    assert torch.isfinite(logits).all()
