import numpy as np
from gym import spaces
from .multi_discrete import MultiDiscrete as LegacyMultiDiscrete


# --------------------------------------------------------------------------- #
# PettingZoo parallel-env factories                                            #
# --------------------------------------------------------------------------- #

def _make_simple_spread(args):
    from pettingzoo.mpe import simple_spread_v3
    return simple_spread_v3.parallel_env(
        N=args.num_agents,
        max_cycles=args.episode_length,
        continuous_actions=False,
    )


def _make_simple_speaker_listener(args):
    from pettingzoo.mpe import simple_speaker_listener_v4
    return simple_speaker_listener_v4.parallel_env(
        max_cycles=args.episode_length,
        continuous_actions=False,
    )


def _make_simple_reference(args):
    from pettingzoo.mpe import simple_reference_v3
    return simple_reference_v3.parallel_env(
        max_cycles=args.episode_length,
        continuous_actions=False,
    )


def _make_pistonball(args):
    from pettingzoo.butterfly import pistonball_v6
    return pistonball_v6.parallel_env(
        n_pistons=args.num_agents,
        max_cycles=args.episode_length,
        continuous=False,
    )


_FACTORIES = {
    'simple_spread': _make_simple_spread,
    'simple_speaker_listener': _make_simple_speaker_listener,
    'simple_reference': _make_simple_reference,
    'pistonball': _make_pistonball,
}


# --------------------------------------------------------------------------- #
# Action-space compatibility shim                                              #
# --------------------------------------------------------------------------- #

def _wrap_action_space(pz_space):
    """Convert a PettingZoo/Gymnasium MultiDiscrete to LegacyMultiDiscrete.

    The runner accesses `.shape` (int) and `.high[i]` on MultiDiscrete spaces,
    which is the interface of the legacy custom class.  Discrete spaces are
    returned unchanged because the runner only needs `.n` and that attribute
    exists on both gym and gymnasium Discrete.
    """
    if pz_space.__class__.__name__ == 'MultiDiscrete':
        return LegacyMultiDiscrete([[0, int(n) - 1] for n in pz_space.nvec])
    return pz_space


# --------------------------------------------------------------------------- #
# Wrapper                                                                      #
# --------------------------------------------------------------------------- #

class _PZWrapper:
    """Adapts a PettingZoo parallel env to the legacy MultiAgentEnv interface.

    The legacy interface expected by the runners:
      - env.n                     int, number of agents
      - env.observation_space     list of gym.spaces.Box, one per agent
      - env.share_observation_space  list of gym.spaces.Box, one per agent
      - env.action_space          list of spaces, one per agent
      - env.seed(seed)            store seed for next reset()
      - env.reset()               -> list of np.ndarray, one per agent
      - env.step(action_n)        -> (obs_n, reward_n, done_n, info_n)
        where reward_n[i] = [scalar], info_n[i] = {'individual_reward': scalar}
      - env.close()
    """

    def __init__(self, pz_env):
        self._env = pz_env
        self._seed = None

        # Initialise the environment so observation/action spaces are populated.
        pz_env.reset()

        # Stable agent ordering guarantees runner index == agent position.
        self._agents = sorted(pz_env.possible_agents)
        self.n = len(self._agents)

        raw_obs_spaces = [pz_env.observation_space(a) for a in self._agents]
        raw_act_spaces = [pz_env.action_space(a) for a in self._agents]

        # Flatten multi-dimensional observations (e.g. pistonball images) to
        # 1-D so the runner's reshape/concatenate calls work uniformly.
        self.observation_space = [
            spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(int(np.prod(s.shape)),), dtype=np.float32,
            )
            for s in raw_obs_spaces
        ]

        # Convert gymnasium MultiDiscrete → LegacyMultiDiscrete.
        self.action_space = [_wrap_action_space(s) for s in raw_act_spaces]

        # Shared observation = concatenation of all agents' observations.
        share_obs_dim = sum(int(np.prod(s.shape)) for s in raw_obs_spaces)
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(share_obs_dim,), dtype=np.float32,
            )
            for _ in range(self.n)
        ]

    # ---------------------------------------------------------------------- #

    def seed(self, seed):
        self._seed = seed

    def reset(self):
        obs_dict, _ = self._env.reset(seed=self._seed)
        # Consume the seed once so subsequent within-run resets are not
        # deterministically identical (mirrors np.random.seed behaviour of
        # the old bundled MPE).
        self._seed = None
        return [
            obs_dict[a].flatten().astype(np.float32)
            for a in self._agents
        ]

    def step(self, action_n):
        actions = {
            a: self._decode_action(action_n[i], self.action_space[i])
            for i, a in enumerate(self._agents)
        }
        obs_dict, rew_dict, term_dict, trunc_dict, _ = self._env.step(actions)

        obs_n, reward_n, done_n, info_n = [], [], [], []
        for i, a in enumerate(self._agents):
            obs = obs_dict.get(a, np.zeros(self.observation_space[i].shape, dtype=np.float32))
            obs_n.append(obs.flatten().astype(np.float32))
            r = float(rew_dict.get(a, 0.0))
            reward_n.append([r])
            done_n.append(bool(term_dict.get(a, True) or trunc_dict.get(a, True)))
            info_n.append({'individual_reward': r})

        return obs_n, reward_n, done_n, info_n

    def close(self):
        self._env.close()

    # ---------------------------------------------------------------------- #

    def _decode_action(self, action, action_space):
        """Convert runner one-hot (or concatenated one-hots) → PettingZoo integer action."""
        if isinstance(action_space, LegacyMultiDiscrete):
            # action is concatenated one-hot vectors, one per sub-space.
            result = []
            offset = 0
            for hi in action_space.high:
                size = int(hi) + 1
                result.append(int(np.argmax(action[offset:offset + size])))
                offset += size
            return np.array(result, dtype=np.int32)
        # Discrete: one-hot vector → integer index.
        return int(np.argmax(action))


# --------------------------------------------------------------------------- #
# Public factory (preserves the original call signature)                       #
# --------------------------------------------------------------------------- #

def MPEEnv(args):
    scenario = args.scenario_name
    if scenario not in _FACTORIES:
        raise ValueError(
            f"Unsupported scenario: {scenario!r}. "
            f"Supported: {', '.join(_FACTORIES)}"
        )
    return _PZWrapper(_FACTORIES[scenario](args))
