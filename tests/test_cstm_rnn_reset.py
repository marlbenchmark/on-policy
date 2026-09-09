import numpy as np
import torch
from gym import spaces

from cstm_test_utils import make_args
from onpolicy.algorithms.cstm_mappo.algorithm.dual_policy_actor import B1Actor


def test_zero_mask_resets_actor_history():
    args = make_args()
    actor = B1Actor(args, spaces.Box(-1, 1, shape=(7,), dtype=np.float32),
                    spaces.Discrete(5), args.num_agents)
    obs = np.random.randn(4, 7).astype(np.float32)
    state_a = np.random.randn(4, 1, args.hidden_size).astype(np.float32)
    state_b = np.random.randn(4, 1, args.hidden_size).astype(np.float32)
    reset_masks = np.zeros((4, 1), dtype=np.float32)
    with torch.no_grad():
        logits_a, next_a = actor.teammate_predictions(obs, state_a, reset_masks)
        logits_b, next_b = actor.teammate_predictions(obs, state_b, reset_masks)
    torch.testing.assert_close(logits_a, logits_b)
    torch.testing.assert_close(next_a, next_b)
