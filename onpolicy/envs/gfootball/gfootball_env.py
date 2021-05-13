"""
    MARL environment for google football
"""

import numpy as np
import gym
import gfootball.env.football_env as football_env
from gfootball.env import _process_representation_wrappers
from gfootball.env import _process_reward_wrappers
from gfootball.env import config
from gfootball.env import wrappers


class GoogleFootballEnv(object):
    def __init__(self,
                 num_of_left_agents,
                 num_of_right_agents=0,
                 env_name="test_example_multiagent",
                 stacked=False,
                 representation='extracted',
                 rewards='scoring',
                 write_goal_dumps=False,
                 write_full_episode_dumps=False,
                 render=False,
                 write_video=False,
                 dump_frequency=1,
                 extra_players=None,
                 channel_dimensions=(96, 72),
                 other_config_options={}) -> None:

        assert num_of_left_agents >= 0
        assert num_of_right_agents >= 0
        assert num_of_left_agents + num_of_right_agents != 0

        # config the environment

        scenario_config = config.Config({'level': env_name}).ScenarioConfig()
        players = [('agent:left_players=%d,right_players=%d' %
                    (num_of_left_agents, num_of_right_agents))]

        if extra_players is not None:
            players.extend(extra_players)

        config_values = {
            'dump_full_episodes': write_full_episode_dumps,
            'dump_scores': write_goal_dumps,
            'players': players,
            'level': env_name,
            'tracesdir': "/tmp/gfootball_log",
            'write_video': write_video,
        }
        config_values.update(other_config_options)
        c = config.Config(config_values)
        self._env = football_env.FootballEnv(c)

        if dump_frequency > 1:
            self._env = wrappers.PeriodicDumpWriter(self._env, dump_frequency,
                                                    render)
        elif render:
            self._env.render()

        # _apply_output_wrappers 在只有一个agent时非要加 wrapper

        self._env = _process_reward_wrappers(self._env, rewards)
        self._env = _process_representation_wrappers(self._env, representation,
                                                     channel_dimensions)

        if stacked:
            self._env = wrappers.FrameStack(self._env, 4)
            self._env = wrappers.GetStateWrapper(self._env)

        self._action_space = self._env.action_space \
        if num_of_left_agents + num_of_right_agents == 1 else \
        gym.spaces.Discrete(self._env.action_space.nvec[0])

        self._observation_space = None if representation == "raw" else gym.spaces.Box(
            low=self._env.observation_space.low[0],
            high=self._env.observation_space.high[0],
            dtype=self._env.observation_space.dtype)

        self._num_left = num_of_left_agents
        self._num_right = num_of_right_agents

        self._left_keys = tuple("agent_%d" % i for i in range(self._num_left))
        self._right_keys = tuple("agent_%d" % i
                                 for i in range(self._num_right))

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def seed(self, seed=None):
        return self._env.seed(seed)

    def reset(self):
        return self._raw2dict(self._env.reset())

    def _raw2dict(self, raw):
        return {
            'left':
            {self._left_keys[i]: raw[i]
             for i in range(self._num_left)},
            'right': {
                self._right_keys[i]: raw[self._num_left + i]
                for i in range(self._num_right)
            }
        }

    def step(self, action_dict):

        #虽然理论上讲迭代器不保证访问次序？
        actions_left = [action_dict['left'][key] for key in self._left_keys]
        actions_right = [action_dict['right'][key] for key in self._right_keys]
        o, r, d, i = self._env.step(actions_left + actions_right)

        return self._raw2dict(o), self._raw2dict(r), d, i

    def random_step(self):
        o, r, d, i = self._env.step([
            self.action_space.sample()
            for i in range(self._num_left + self._num_right)
        ])
        return self._raw2dict(o), self._raw2dict(r), d, i


if __name__ == "__main__":
    e = GoogleFootballEnv(num_of_left_agents=2,
                          num_of_right_agents=2,
                          env_name='5_vs_5',
                          representation="raw")
    o = e.reset()
    while True:
        o, _, done, _ = e.random_step()
        assert (o['left']['agent_0']['left_team'] == o['left']['agent_1']
                ['left_team']).all()
        assert (o['right']['agent_0']['right_team'] == o['right']['agent_1']
                ['right_team']).all()

        assert not (o['left']['agent_0']['left_team'] +
               o['right']['agent_0']['right_team']).any()

        if done:
            break
