import time
import numpy as np
from functools import reduce
import torch
import wandb
from onpolicy.runner.semi_shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def eval_only(self):
        """
        This method is used for evaluation only, without training.
        It initializes the environment and runs the evaluation loop.
        """
        self.warmup()
        self.eval(0)

    def run(self):

        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for ut in self.unique_unit_types:
                    self.trainer[ut].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                data = obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

            self.compute()
            train_infos = self.train()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                print(f"\n Map {self.all_args.map_name} Algo {self.algorithm_name} Exp {self.experiment_name} updates {episode}/{episodes} episodes, total num timesteps {total_num_steps}/{self.num_env_steps}, FPS {int(total_num_steps / (end - start))}.\n")

                battles_won = []
                battles_game = []
                incre_battles_won = []
                incre_battles_game = []
                for i, info in enumerate(infos):
                    if 'battles_won' in info[0].keys():
                        battles_won.append(info[0]['battles_won'])
                        incre_battles_won.append(info[0]['battles_won'] - last_battles_won[i])
                    if 'battles_game' in info[0].keys():
                        battles_game.append(info[0]['battles_game'])
                        incre_battles_game.append(info[0]['battles_game'] - last_battles_game[i])

                incre_win_rate = np.sum(incre_battles_won) / np.sum(incre_battles_game) if np.sum(incre_battles_game) > 0 else 0.0
                print(f"incre win rate is {incre_win_rate}.")
                if self.use_wandb:
                    wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                last_battles_game = battles_game
                last_battles_won = battles_won

                # Example dead ratio (averaged across types)
                dead_ratios = [1 - self.buffer[ut].active_masks.sum() / (len(self.type_to_agents[ut]) * reduce(lambda x, y: x * y, list(self.buffer[ut].active_masks.shape))) for ut in self.unique_unit_types]
                train_infos[0]['dead_ratio'] = np.mean(dead_ratios)  # Aggregate for logging
                self.log_train(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        if not self.use_centralized_V:
            share_obs = obs
        for ut in self.unique_unit_types:
            agent_indices = self.type_to_agents[ut]
            batch_size = self.n_rollout_threads * len(agent_indices) # * self.num_agents_per_type[ut]
            self.buffer[ut].share_obs[0] = share_obs[:, agent_indices].copy() #.reshape(batch_size, -1)
            self.buffer[ut].obs[0] = obs[:, agent_indices].copy() # .reshape(self.n_rollout_threads * len(agent_indices), -1)
            self.buffer[ut].available_actions[0] = available_actions[:, agent_indices].copy() # .reshape(self.n_rollout_threads * len(agent_indices), -1)

    @torch.no_grad()
    def collect(self, step):
        values = np.zeros((self.n_rollout_threads, self.num_agents, 1))
        actions = np.zeros((self.n_rollout_threads, self.num_agents, 1))  # Adjust dim as needed
        action_log_probs = np.zeros((self.n_rollout_threads, self.num_agents, 1))
        rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size))
        rnn_states_critic = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size))

        for ut in self.unique_unit_types:
            self.trainer[ut].prep_rollout()
            agent_indices = self.type_to_agents[ut]
            num_agents_ut = len(agent_indices)
            batch_size = self.n_rollout_threads * num_agents_ut

            batched_share_obs = np.concatenate(self.buffer[ut].share_obs[step]) # .reshape(batch_size, -1)
            batched_obs = np.concatenate(self.buffer[ut].obs[step]) # .reshape(batch_size, -1)
            batched_rnn_states = np.concatenate(self.buffer[ut].rnn_states[step]) # .reshape(batch_size, -1)
            batched_rnn_states_critic = np.concatenate(self.buffer[ut].rnn_states_critic[step]) # .reshape(batch_size, -1)
            batched_masks = np.concatenate(self.buffer[ut].masks[step])
            batched_available_actions = np.concatenate(self.buffer[ut].available_actions[step]) # .reshape(batch_size, -1)

            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[ut].policy.get_actions(
                batched_share_obs, batched_obs, batched_rnn_states, batched_rnn_states_critic, batched_masks, batched_available_actions
            )

            # Unbatch and place in full arrays
            ut_values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            ut_actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
            ut_action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
            ut_rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
            ut_rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

            for i, idx in enumerate(agent_indices):
                values[:, idx] = ut_values[:, i]
                actions[:, idx] = ut_actions[:, i]
                action_log_probs[:, idx] = ut_action_log_probs[:, i]
                rnn_states[:, idx] = ut_rnn_states[:, i]
                rnn_states_critic[:, idx] = ut_rnn_states_critic[:, i]

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        for ut in self.unique_unit_types:
            agent_indices = self.type_to_agents[ut]
            num_agents_ut = len(agent_indices)
            batch_size = self.n_rollout_threads * num_agents_ut

            batched_share_obs = share_obs[:, agent_indices]# .reshape(batch_size, -1)
            batched_obs = obs[:, agent_indices]# .reshape(batch_size, -1)
            batched_rnn_states = rnn_states[:, agent_indices]# .reshape(batch_size, -1)
            batched_rnn_states_critic = rnn_states_critic[:, agent_indices]# .reshape(batch_size, -1)
            batched_actions = actions[:, agent_indices]# .reshape(batch_size, -1)
            batched_action_log_probs = action_log_probs[:, agent_indices]# .reshape(batch_size, -1)
            batched_values = values[:, agent_indices]# .reshape(batch_size, -1)
            batched_rewards = rewards[:, agent_indices]# .reshape(batch_size, -1)
            batched_masks = masks[:, agent_indices]# .reshape(batch_size, -1)
            batched_bad_masks = np.array(bad_masks)[:, agent_indices]# .reshape(batch_size, -1)
            batched_active_masks = active_masks[:, agent_indices]# .reshape(batch_size, -1)
            batched_available_actions = available_actions[:, agent_indices]# .reshape(batch_size, -1)

            self.buffer[ut].insert(batched_share_obs, batched_obs, batched_rnn_states, batched_rnn_states_critic,
                                   batched_actions, batched_action_log_probs, batched_values, batched_rewards,
                                   batched_masks, batched_bad_masks, batched_active_masks, batched_available_actions)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0
        eval_episode_rewards = [[] for _ in range(self.n_eval_rollout_threads)]
        one_episode_rewards = [[] for _ in range(self.n_eval_rollout_threads)]

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1),
            dtype=np.float32
        )

        while True:
            # Collect actions per unit type
            eval_actions = np.zeros((self.n_eval_rollout_threads, self.num_agents, 1))  # Adjust action dim as needed
            for ut in self.unique_unit_types:
                self.trainer[ut].prep_rollout()
                agent_indices = self.type_to_agents[ut]
                num_agents_ut = len(agent_indices)
                batch_size = self.n_eval_rollout_threads * num_agents_ut

                batched_obs = np.concatenate(eval_obs[:, agent_indices]) # .reshape(batch_size, -1)
                batched_rnn_states = np.concatenate(eval_rnn_states[:, agent_indices]) # .reshape(batch_size, -1)
                batched_masks = np.concatenate(eval_masks[:, agent_indices]) #.reshape(batch_size, -1)
                batched_available_actions = np.concatenate(eval_available_actions[:, agent_indices]) # .reshape(batch_size, -1)

                # Deterministic action selection for evaluation
                actions, _ = self.trainer[ut].policy.act(
                    batched_obs,
                    batched_rnn_states,
                    batched_masks,
                    batched_available_actions,
                    deterministic=True
                )

                ut_actions = np.array(np.split(_t2n(actions), self.n_eval_rollout_threads))
                for i, idx in enumerate(agent_indices):
                    eval_actions[:, idx] = ut_actions[:, i]

            # Step environment
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)

            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1),
                dtype=np.float32
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1),
                dtype=np.float32
            )

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []
                    if eval_infos[eval_i][0].get('won', False):
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards_flat = np.concatenate(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards_flat}
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won / eval_episode if eval_episode > 0 else 0.0
                print(f"eval win rate is {eval_win_rate}.")
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
