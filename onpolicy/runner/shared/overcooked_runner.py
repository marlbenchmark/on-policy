import time
import wandb
import copy
import os
import numpy as np
import itertools
from itertools import chain
import torch
import imageio
import warnings
import functools
from onpolicy.runner.shared.base_runner import Runner
from pathlib import Path
from collections import defaultdict, deque
from tqdm import tqdm
from collections import defaultdict, deque
from typing import Dict
# from icecream import ic
from scipy.stats import rankdata


def _t2n(x):
    return x.detach().cpu().numpy()


class OvercookedRunner(Runner):
    """
    A wrapper to start the RL agent training algorithm.
    """

    def __init__(self, config):
        super(OvercookedRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                obs = np.stack(obs)
                total_num_steps += (self.n_rollout_threads)
                # self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)
                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Layout {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.layout_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

                env_infos = defaultdict(list)
                if self.env_name == "Overcooked":
                    for info in infos:
                        env_infos['ep_sparse_r_by_agent0'].append(info['episode']['ep_sparse_r_by_agent'][0])
                        env_infos['ep_sparse_r_by_agent1'].append(info['episode']['ep_sparse_r_by_agent'][1])
                        env_infos['ep_shaped_r_by_agent0'].append(info['episode']['ep_shaped_r_by_agent'][0])
                        env_infos['ep_shaped_r_by_agent1'].append(info['episode']['ep_shaped_r_by_agent'][1])
                        env_infos['ep_sparse_r'].append(info['episode']['ep_sparse_r'])
                        env_infos['ep_shaped_r'].append(info['episode']['ep_shaped_r'])

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        obs = np.stack(obs)

        # replay buffer
        if self.use_centralized_V:
            share_obs = share_obs
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = share_obs
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                           masks)

    def restore(self):
        if self.use_single_network:
            policy_model_state_dict = torch.load(str(self.model_dir) + '/model.pt', map_location=self.device)
            self.policy.model.load_state_dict(policy_model_state_dict)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir), map_location=self.device)
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not (self.all_args.use_render or self.all_args.use_eval):
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt', map_location=self.device)
                self.policy.critic.load_state_dict(policy_critic_state_dict)

    def save(self, step=0):
        # if self.use_single_network:
        if False:
            policy_model = self.trainer.policy.model
            torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_periodic_{}.pt".format(step))
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_periodic_{}.pt".format(step))
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_periodic_{}.pt".format(step))

    def load(self,step=0):
        torch.load(str(self.save_dir) + "/actor_periodic_{}.pt".format(step))
        self.trainer.policy.actor.load_state_dict(torch.load(str(self.save_dir) + "/actor_periodic_{}.pt".format(step)))
        torch.load(str(self.save_dir) + "/actor_periodic_{}.pt".format(step))
        self.trainer.policy.critic.load_state_dict(torch.load(str(self.save_dir) + "/critic_periodic_{}.pt".format(step)))


    @torch.no_grad()
    def eval(self, total_num_steps):
        self.load()
        eval_env_infos = defaultdict(list)
        eval_average_episode_rewards = []
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_obs = np.stack(eval_obs)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                   np.concatenate(eval_rnn_states),
                                                                   np.concatenate(eval_masks),
                                                                   deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            eval_obs = np.stack(eval_obs)
            eval_average_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        for eval_info in eval_infos:
            eval_env_infos['eval_ep_sparse_r_by_agent0'].append(eval_info['episode']['ep_sparse_r_by_agent'][0])
            eval_env_infos['eval_ep_sparse_r_by_agent1'].append(eval_info['episode']['ep_sparse_r_by_agent'][1])
            eval_env_infos['eval_ep_shaped_r_by_agent0'].append(eval_info['episode']['ep_shaped_r_by_agent'][0])
            eval_env_infos['eval_ep_shaped_r_by_agent1'].append(eval_info['episode']['ep_shaped_r_by_agent'][1])
            eval_env_infos['eval_ep_sparse_r'].append(eval_info['episode']['ep_sparse_r'])
            eval_env_infos['eval_ep_shaped_r'].append(eval_info['episode']['ep_shaped_r'])

        eval_env_infos['eval_average_episode_rewards'] = np.sum(eval_average_episode_rewards, axis=0)
        print("eval average sparse rewards: " + str(np.mean(eval_env_infos['eval_ep_sparse_r'])))

        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        envs = self.envs
        obs, share_obs, available_actions = envs.reset()
        obs = np.stack(obs)

        for episode in range(self.all_args.render_episodes):
            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []
            trajectory = []
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                             np.concatenate(rnn_states),
                                                             np.concatenate(masks),
                                                             deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions)
                obs = np.stack(obs)

                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            # for info in infos:
            #     ic(info['episode']['ep_sparse_r_by_agent'][0])
            #     ic(info['episode']['ep_sparse_r_by_agent'][1])
            #     ic(info['episode']['ep_shaped_r_by_agent'][0])
            #     ic(info['episode']['ep_shaped_r_by_agent'][1])
            #     ic(info['episode']['ep_sparse_r'])
            #     ic(info['episode']['ep_shaped_r'])

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))