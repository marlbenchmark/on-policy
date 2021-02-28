    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class HanabiRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for Hanabi. See parent class for details."""
    def __init__(self, config):
        super(HanabiRunner, self).__init__(config)
        self.true_total_num_steps = 0
    
    def run(self):
        self.turn_obs = np.zeros((self.n_rollout_threads,*self.buffer.obs.shape[2:]), dtype=np.float32)
        self.turn_share_obs = np.zeros((self.n_rollout_threads,*self.buffer.share_obs.shape[2:]), dtype=np.float32)
        self.turn_available_actions = np.zeros((self.n_rollout_threads,*self.buffer.available_actions.shape[2:]), dtype=np.float32)
        self.turn_values = np.zeros((self.n_rollout_threads,*self.buffer.value_preds.shape[2:]), dtype=np.float32)
        self.turn_actions = np.zeros((self.n_rollout_threads,*self.buffer.actions.shape[2:]), dtype=np.float32)       
        self.turn_action_log_probs = np.zeros((self.n_rollout_threads,*self.buffer.action_log_probs.shape[2:]), dtype=np.float32)
        self.turn_rnn_states = np.zeros((self.n_rollout_threads,*self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        self.turn_rnn_states_critic = np.zeros_like(self.turn_rnn_states)
        self.turn_masks = np.ones((self.n_rollout_threads,*self.buffer.masks.shape[2:]), dtype=np.float32)
        self.turn_active_masks = np.ones_like(self.turn_masks)
        self.turn_bad_masks = np.ones_like(self.turn_masks)
        self.turn_rewards = np.zeros((self.n_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        self.turn_rewards_since_last_action = np.zeros_like(self.turn_rewards)

        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            self.scores = []
            for step in range(self.episode_length):
                self.reset_choose = np.zeros(self.n_rollout_threads) == 1.0
                # Sample actions
                self.collect(step) 

                if step == 0 and episode > 0:
                    # deal with the data of the last index in buffer
                    self.buffer.share_obs[-1] = self.turn_share_obs.copy()
                    self.buffer.obs[-1] = self.turn_obs.copy()
                    self.buffer.available_actions[-1] = self.turn_available_actions.copy()
                    self.buffer.active_masks[-1] = self.turn_active_masks.copy()

                    # deal with rewards
                    # 1. shift all rewards
                    self.buffer.rewards[0:self.episode_length-1] = self.buffer.rewards[1:]
                    # 2. last step rewards
                    self.buffer.rewards[-1] = self.turn_rewards.copy()

                    # compute return and update network
                    self.compute()
                    train_infos = self.train()

                # insert turn data into buffer
                self.buffer.chooseinsert(self.turn_share_obs,
                                        self.turn_obs,
                                        self.turn_rnn_states,
                                        self.turn_rnn_states_critic,
                                        self.turn_actions,
                                        self.turn_action_log_probs,
                                        self.turn_values,
                                        self.turn_rewards,
                                        self.turn_masks,
                                        self.turn_bad_masks,
                                        self.turn_active_masks,
                                        self.turn_available_actions)
                # env reset
                obs, share_obs, available_actions = self.envs.reset(self.reset_choose)
                share_obs = share_obs if self.use_centralized_V else obs

                self.use_obs[self.reset_choose] = obs[self.reset_choose]
                self.use_share_obs[self.reset_choose] = share_obs[self.reset_choose]
                self.use_available_actions[self.reset_choose] = available_actions[self.reset_choose]
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0 and episode > 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.hanabi_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "Hanabi":
                    average_score = np.mean(self.scores) if len(self.scores) > 0 else 0.0
                    print("average score is {}.".format(average_score))
                    if self.use_wandb:
                        wandb.log({'average_score': average_score}, step = self.true_total_num_steps)
                    else:
                        self.writter.add_scalars('average_score', {'average_score': average_score}, self.true_total_num_steps)

                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                
                self.log_train(train_infos, self.true_total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(self.true_total_num_steps)

    def warmup(self):
        # reset env
        self.reset_choose = np.ones(self.n_rollout_threads) == 1.0
        obs, share_obs, available_actions = self.envs.reset(self.reset_choose)

        share_obs = share_obs if self.use_centralized_V else obs

        # replay buffer
        self.use_obs = obs.copy()
        self.use_share_obs = share_obs.copy()
        self.use_available_actions = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        for current_agent_id in range(self.num_agents):
            env_actions = np.ones((self.n_rollout_threads, *self.buffer.actions.shape[3:]), dtype=np.float32)*(-1.0)
            choose = np.any(self.use_available_actions == 1, axis=1)
            if ~np.any(choose):
                self.reset_choose = np.ones(self.n_rollout_threads) == 1.0
                break
            
            self.trainer.prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer.policy.get_actions(self.use_share_obs[choose],
                                                self.use_obs[choose],
                                                self.turn_rnn_states[choose, current_agent_id],
                                                self.turn_rnn_states_critic[choose, current_agent_id],
                                                self.turn_masks[choose, current_agent_id],
                                                self.use_available_actions[choose])
            
            self.turn_obs[choose, current_agent_id] = self.use_obs[choose].copy()
            self.turn_share_obs[choose, current_agent_id] = self.use_share_obs[choose].copy()
            self.turn_available_actions[choose, current_agent_id] = self.use_available_actions[choose].copy()
            self.turn_values[choose, current_agent_id] = _t2n(value)
            self.turn_actions[choose, current_agent_id] = _t2n(action)
            env_actions[choose] = _t2n(action)
            self.turn_action_log_probs[choose, current_agent_id] = _t2n(action_log_prob)
            self.turn_rnn_states[choose, current_agent_id] = _t2n(rnn_state)
            self.turn_rnn_states_critic[choose, current_agent_id] = _t2n(rnn_state_critic)

            obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(env_actions)
            
            self.true_total_num_steps += (choose==True).sum()
            share_obs = share_obs if self.use_centralized_V else obs

            # truly used value
            self.use_obs = obs.copy()
            self.use_share_obs = share_obs.copy()
            self.use_available_actions = available_actions.copy()

            # rearrange reward
            # reward of step 0 will be thrown away.
            self.turn_rewards[choose, current_agent_id] = self.turn_rewards_since_last_action[choose, current_agent_id].copy()
            self.turn_rewards_since_last_action[choose, current_agent_id] = 0.0
            self.turn_rewards_since_last_action[choose] += rewards[choose]

            # done==True env

            # deal with reset_choose
            self.reset_choose[dones == True] = np.ones((dones == True).sum(), dtype=bool)

            # deal with all agents
            self.use_available_actions[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.available_actions.shape[3:]), dtype=np.float32)
            self.turn_masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)
            self.turn_rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            self.turn_rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

            # deal with the current agent
            self.turn_active_masks[dones == True, current_agent_id] = np.ones(((dones == True).sum(), 1), dtype=np.float32)

            # deal with the left agents
            left_agent_id = current_agent_id + 1
            left_agents_num = self.num_agents - left_agent_id
            self.turn_active_masks[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
            
            self.turn_rewards[dones == True, left_agent_id:] = self.turn_rewards_since_last_action[dones == True, left_agent_id:]
            self.turn_rewards_since_last_action[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
            
            # other variables use what at last time, action will be useless.
            self.turn_values[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
            self.turn_obs[dones == True, left_agent_id:] = 0
            self.turn_share_obs[dones == True, left_agent_id:] = 0

            # done==False env
            # deal with current agent
            self.turn_masks[dones == False, current_agent_id] = np.ones(((dones == False).sum(), 1), dtype=np.float32)
            self.turn_active_masks[dones == False, current_agent_id] = np.ones(((dones == False).sum(), 1), dtype=np.float32)

            # done==None
            # pass

            for done, info in zip(dones, infos):
                if done:
                    if 'score' in info.keys():
                        self.scores.append(info['score'])
            
   
    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.chooseafter_update()
        return train_infos
  
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_envs = self.eval_envs

        eval_scores = []

        eval_finish = False
        eval_reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
        
        eval_obs, eval_share_obs, eval_available_actions = eval_envs.reset(eval_reset_choose)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            if eval_finish:
                break
            for agent_id in range(self.num_agents):
                eval_actions = np.ones((self.n_eval_rollout_threads, 1), dtype=np.float32) * (-1.0)
                eval_choose = np.any(eval_available_actions == 1, axis=1)

                if ~np.any(eval_choose):
                    eval_finish = True
                    break

                self.trainer.prep_rollout()
                eval_action, eval_rnn_state = self.trainer.policy.act(eval_obs[eval_choose],
                                                                eval_rnn_states[eval_choose, agent_id],
                                                                eval_masks[eval_choose, agent_id],
                                                                eval_available_actions[eval_choose],
                                                                deterministic=True)

                eval_actions[eval_choose] = _t2n(eval_action)
                eval_rnn_states[eval_choose, agent_id] = _t2n(eval_rnn_state)

                # Obser reward and next obs
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_envs.step(eval_actions)
                
                eval_available_actions[eval_dones == True] = np.zeros(((eval_dones == True).sum(), *self.buffer.available_actions.shape[3:]), dtype=np.float32)

                for eval_done, eval_info in zip(eval_dones, eval_infos):
                    if eval_done:
                        if 'score' in eval_info.keys():
                            eval_scores.append(eval_info['score'])

        eval_average_score = np.mean(eval_scores)
        print("eval average score is {}.".format(eval_average_score))
        if self.use_wandb:
            wandb.log({'eval_average_score': eval_average_score}, step=total_num_steps)
        else:
            self.writter.add_scalars('eval_average_score', {'eval_average_score': eval_average_score}, total_num_steps)

    
    @torch.no_grad()
    def eval_100k(self, eval_games=100000):
        eval_envs = self.eval_envs
        trials = int(eval_games/self.n_eval_rollout_threads)

        eval_scores = []
        for trial in range(trials):
            print("trail is {}".format(trial))
            eval_finish = False
            eval_reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
            
            eval_obs, eval_share_obs, eval_available_actions = eval_envs.reset(eval_reset_choose)

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            while True:
                if eval_finish:
                    break
                for agent_id in range(self.num_agents):
                    eval_actions = np.ones((self.n_eval_rollout_threads, 1), dtype=np.float32) * (-1.0)
                    eval_choose = np.any(eval_available_actions == 1, axis=1)

                    if ~np.any(eval_choose):
                        eval_finish = True
                        break

                    self.trainer.prep_rollout()
                    eval_action, eval_rnn_state = self.trainer.policy.act(eval_obs[eval_choose],
                                                                    eval_rnn_states[eval_choose, agent_id],
                                                                    eval_masks[eval_choose, agent_id],
                                                                    eval_available_actions[eval_choose],
                                                                    deterministic=True)

                    eval_actions[eval_choose] = _t2n(eval_action)
                    eval_rnn_states[eval_choose, agent_id] = _t2n(eval_rnn_state)

                    # Obser reward and next obs
                    eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_envs.step(eval_actions)
                    
                    eval_available_actions[eval_dones == True] = np.zeros(((eval_dones == True).sum(), *self.buffer.available_actions.shape[3:]), dtype=np.float32)

                    for eval_done, eval_info in zip(eval_dones, eval_infos):
                        if eval_done:
                            if 'score' in eval_info.keys():
                                eval_scores.append(eval_info['score'])

        eval_average_score = np.mean(eval_scores)
        print("eval average score is {}.".format(eval_average_score))