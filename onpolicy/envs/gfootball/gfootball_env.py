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

        self._action_space = gym.spaces.Discrete(
            self._env.action_space.nvec[0])

        self._observation_space = None if representation == "raw" else gym.spaces.Box(
            low=self._env.observation_space.low[0],
            high=self._env.observation_space.high[0],
            dtype=self._env.observation_space.dtype)

        self._num_left = num_of_left_agents
        self._num_right = num_of_right_agents

        self._share_observation_space = gym.spaces.Box(
            low=np.concatenate([
                self._observation_space.low
                for i in range(self._num_left + self._num_right)
            ],
                         axis=-1),
            high=np.concatenate([
                self._observation_space.high
                for i in range(self._num_left + self._num_right)
            ],
                          axis=-1),
            dtype=self._observation_space.dtype)


    @property
    def action_space(self):
        return [
            self._action_space for i in range(self._num_left + self._num_right)
        ]

    @property
    def observation_space(self):
        return [
            self._observation_space
            for i in range(self._num_left + self._num_right)
        ]

    @property
    def share_observation_space(self):
        return [
            self._share_observation_space
            for i in range(self._num_left + self._num_right)
        ]

    def seed(self, seed=None):
        return self._env.seed(seed)

    def reset(self):
        return self._env.reset()

    def step(self, actions):
        return self._env.step(actions)

    @property
    def num_of_left_agents(self):
        return self._num_left

    @property
    def num_of_right_agents(self):
        return self._num_right

    def random_step(self):
        return self._env.step([
            self._action_space.sample()
            for i in range(self._num_left + self._num_right)
        ])


if __name__ == "__main__":
    e = GoogleFootballEnv(num_of_left_agents=2,
                          num_of_right_agents=2,
                          env_name='5_vs_5',
                          representation="simple115v2")
    o = e.reset()
    _,_,_,infos=e.random_step()
    print(infos['score_reward'])
