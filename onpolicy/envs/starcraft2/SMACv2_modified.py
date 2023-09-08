from .StarCraft2v2.wrapper import StarCraftCapabilityEnvWrapper # modified smac v2 environment
import random
from gym.spaces import Discrete
import numpy as np

class SMACv2(StarCraftCapabilityEnvWrapper):
    def __init__(self, **kwargs):
        super(SMACv2, self).__init__(obs_last_action=False, **kwargs)
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        
        self.n_agents = self.env.n_agents

        for i in range(self.env.n_agents):
            self.action_space.append(Discrete(self.env.n_actions))
            self.observation_space.append(self.env.get_obs_size())
            self.share_observation_space.append(self.env.get_state_size())

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        obs, state = super().reset()
        avail_actions = [self.get_avail_agent_actions(i) for i in range(self.env.n_agents)]
        return obs, state, avail_actions

    def step(self, actions):
        reward, terminated, info = super().step(actions)
        local_obs = self.get_obs()
        global_state = np.array([self.env.get_state_agent(agent_id) for agent_id in range(self.env.n_agents)])
        rewards = [[reward]] * self.env.n_agents

        dones = []

        for i in range(self.env.n_agents):
            if terminated:
                dones.append(True)
            else:
                dones.append(self.env.death_tracker_ally[i])

        infos = [info] * self.env.n_agents
        avail_actions = [self.get_avail_agent_actions(i) for i in range(self.env.n_agents)]
        
        bad_transition = True if self.env._episode_steps >= self.env.episode_limit else False
        for info in infos:
            info['bad_transition'] = bad_transition
            info["battles_won"] = self.env.battles_won
            info["battles_game"] = self.env.battles_game
            info["battles_draw"] = self.env.timeouts
            info["restarts"] = self.env.force_restarts
            info["won"] = self.env.win_counted
        return local_obs, global_state, rewards, dones, infos, avail_actions
