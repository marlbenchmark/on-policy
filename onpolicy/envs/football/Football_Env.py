import random

import gfootball.env as football_env
from gym import spaces
import numpy as np


class FootballEnv(object):
    '''Wrapper to make Google Research Football environment compatible'''

    def __init__(self, args):
        self.num_agents = args.num_agents
        self.scenario_name = args.scenario_name
        
        # make env
        if not (args.use_render and args.save_videos):
            self.env = football_env.create_environment(
                env_name=args.scenario_name,
                stacked=args.use_stacked_frames,
                representation=args.representation,
                rewards=args.rewards,
                number_of_left_players_agent_controls=args.num_agents,
                number_of_right_players_agent_controls=0,
                channel_dimensions=(args.smm_width, args.smm_height),
                render=(args.use_render and args.save_gifs)
            )
        else:
            # render env and save videos
            self.env = football_env.create_environment(
                env_name=args.scenario_name,
                stacked=args.use_stacked_frames,
                representation=args.representation,
                rewards=args.rewards,
                number_of_left_players_agent_controls=args.num_agents,
                number_of_right_players_agent_controls=0,
                channel_dimensions=(args.smm_width, args.smm_height),
                # video related params
                write_full_episode_dumps=True,
                render=True,
                write_video=True,
                dump_frequency=1,
                logdir=args.video_dir
            )
            
        self.max_steps = self.env.unwrapped.observation()[0]["steps_left"]
        self.remove_redundancy = args.remove_redundancy
        self.zero_feature = args.zero_feature
        self.share_reward = args.share_reward
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        if self.num_agents == 1:
            self.action_space.append(self.env.action_space)
            self.observation_space.append(self.env.observation_space)
            self.share_observation_space.append(self.env.observation_space)
        else:
            for idx in range(self.num_agents):
                self.action_space.append(spaces.Discrete(
                    n=self.env.action_space[idx].n
                ))
                self.observation_space.append(spaces.Box(
                    low=self.env.observation_space.low[idx],
                    high=self.env.observation_space.high[idx],
                    shape=self.env.observation_space.shape[1:],
                    dtype=self.env.observation_space.dtype
                ))
                self.share_observation_space.append(spaces.Box(
                    low=self.env.observation_space.low[idx],
                    high=self.env.observation_space.high[idx],
                    shape=self.env.observation_space.shape[1:],
                    dtype=self.env.observation_space.dtype
                ))


    def reset(self):
        obs = self.env.reset()
        obs = self._obs_wrapper(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._obs_wrapper(obs)
        reward = reward.reshape(self.num_agents, 1)
        if self.share_reward:
            global_reward = np.sum(reward)
            reward = [[global_reward]] * self.num_agents

        done = np.array([done] * self.num_agents)
        info = self._info_wrapper(info)
        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.close()

    def _obs_wrapper(self, obs):
        if self.num_agents == 1:
            return obs[np.newaxis, :]
        else:
            return obs

    def _info_wrapper(self, info):
        state = self.env.unwrapped.observation()
        info.update(state[0])
        info["max_steps"] = self.max_steps
        info["active"] = np.array([state[i]["active"] for i in range(self.num_agents)])
        info["designated"] = np.array([state[i]["designated"] for i in range(self.num_agents)])
        info["sticky_actions"] = np.stack([state[i]["sticky_actions"] for i in range(self.num_agents)])
        return info
