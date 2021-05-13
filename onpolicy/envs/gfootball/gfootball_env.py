"""
    MARL environment for google football
"""

import numpy as np
import gym
import gfootball.env as env


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

        self._env = env.create_environment(
            env_name=env_name,
            stacked=stacked,
            representation=representation,
            rewards=rewards,
            write_goal_dumps=write_goal_dumps,
            write_full_episode_dumps=write_full_episode_dumps,
            render=render,
            write_video=write_video,
            dump_frequency=dump_frequency,
            logdir="/tmp/gfootball_log",
            extra_players=extra_players,
            number_of_left_players_agent_controls=num_of_left_agents,
            number_of_right_players_agent_controls=num_of_right_agents,
            channel_dimensions=channel_dimensions,
            other_config_options=other_config_options)

        if num_of_left_agents + num_of_right_agents == 1:
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space
        else:
            self._action_space = gym.spaces.Discrete(
                self._env.action_space.nvec[0])
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
        return self._raw2obj(self._env.reset())

    def _raw2obj(self, raw):
        if self._num_left == 0:
            obj = {
                'left': {},
                'right': {
                    self._right_keys[i]:
                    raw if self._num_right == 1 else raw[i]
                    for i in range(self._num_right)
                }
            }
        elif self._num_right == 0:
            obj = {
                'left': {
                    self._left_keys[i]: raw if self._num_left == 1 else raw[i]
                    for i in range(self._num_left)
                },
                'right': {}
            }
        else:
            obj = {
                'left':
                {self._left_keys[i]: raw[i]
                 for i in range(self._num_left)},
                'right': {
                    self._right_keys[i]: raw[self._num_left + i]
                    for i in range(self._num_right)
                }
            }

        return obj

    def step(self, action_dict):

        #虽然理论上讲迭代器不保证访问次序？
        actions_left = [action_dict['left'][key] for key in self._left_keys]
        actions_right = [action_dict['right'][key] for key in self._right_keys]
        o, r, d, i = self._env.step(actions_left + actions_right)

        return self._raw2obj(o), self._raw2obj(r), d, i

    def random_step(self):
        o, r, d, i = self._env.step([
            self.action_space.sample()
            for i in range(self._num_left + self._num_right)
        ])
        return self._raw2obj(o), self._raw2obj(r), d, i


if __name__ == "__main__":
    e = GoogleFootballEnv(num_of_left_agents=4,
                          env_name='5_vs_5',
                          representation="raw")
    o = e.reset()
    print(o['left']['agent_0']['active'], o['left']['agent_1']['active'],
          o['left']['agent_2']['active'], o['left']['agent_3']['active'])

    print(o['left']['agent_0'].keys())
    exit()
    while True:
        o, _, done, _ = e.random_step()
        print(o['left']['agent_0']['active'], o['left']['agent_1']['active'],
              o['left']['agent_2']['active'], o['left']['agent_3']['active'])
        if done:
            break
