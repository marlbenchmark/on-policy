import io

import numpy as np
import torch
from gym import spaces

from cstm_test_utils import make_args
from onpolicy.algorithms.cstm_mappo.algorithm.dual_policy_actor import B1Actor


def test_actor_save_restore_preserves_outputs():
    args = make_args()
    obs_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
    action_space = spaces.Discrete(5)
    actor = B1Actor(args, obs_space, action_space, args.num_agents)
    obs = np.random.randn(6, 7).astype(np.float32)
    states = np.zeros((6, 1, args.hidden_size), dtype=np.float32)
    masks = np.ones((6, 1), dtype=np.float32)
    with torch.no_grad():
        before = actor(obs, states, masks, deterministic=True)
        logits_before, _ = actor.teammate_predictions(obs, states, masks)
    stream = io.BytesIO()
    torch.save(actor.state_dict(), stream)
    stream.seek(0)
    restored = B1Actor(args, obs_space, action_space, args.num_agents)
    restored.load_state_dict(torch.load(stream))
    with torch.no_grad():
        after = restored(obs, states, masks, deterministic=True)
        logits_after, _ = restored.teammate_predictions(obs, states, masks)
    for expected, actual in zip(before, after):
        torch.testing.assert_close(expected, actual)
    torch.testing.assert_close(logits_before, logits_after)
