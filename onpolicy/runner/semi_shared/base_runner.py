import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # Get unit types
        print(dir(self.envs))
        # self.unit_type_ids = [agent.unit_type for agent in self.envs.agents]
        self.unit_type_ids = [0] * 3 + [1] * 5 # TODO: Remove this cursed workaround
        self.unique_unit_types = list(set(self.unit_type_ids))
        self.num_unit_types = len(self.unique_unit_types)
        self.agent_to_type = {i: self.unit_type_ids[i] for i in range(self.num_agents)}
        self.num_agents_per_type = {ut: self.unit_type_ids.count(ut) for ut in self.unique_unit_types}
        self.type_to_agents = {ut: [i for i in range(self.num_agents) if self.unit_type_ids[i] == ut] for ut in self.unique_unit_types}

        # Parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # Interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # Dir
        self.model_dir = self.all_args.model_dir
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # Policy selection (example with r_mappo; adapt as needed)
        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        self.policy = {}
        self.trainer = {}
        self.buffer = {}
        for ut in self.unique_unit_types:
            share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
            po = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], device=self.device)
            self.policy[ut] = po
            tr = TrainAlgo(self.all_args, po, device=self.device)
            self.trainer[ut] = tr
            bu = SharedReplayBuffer(self.all_args, self.num_agents_per_type[ut], self.envs.observation_space[0], share_observation_space, self.envs.action_space[0])
            self.buffer[ut] = bu

        if self.model_dir is not None:
            self.restore()

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for ut in self.unique_unit_types:
            self.trainer[ut].prep_rollout()
            # Batch next values for agents of this type
            agent_indices = self.type_to_agents[ut]
            batch_size = len(agent_indices) * self.n_rollout_threads
            next_values = self.trainer[ut].policy.get_values(
                np.concatenate(self.buffer[ut].share_obs[-1]), #.reshape(batch_size, -1),
                np.concatenate(self.buffer[ut].rnn_states_critic[-1]), #.reshape(batch_size, -1),
                np.concatenate(self.buffer[ut].masks[-1])
            )
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))

            self.buffer[ut].compute_returns(next_values, self.trainer[ut].value_normalizer)

    def train(self):
        train_infos = []
        for ut in self.unique_unit_types:
            self.trainer[ut].prep_training()
            # Batch training for all agents of this type
            train_info = self.trainer[ut].train(self.buffer[ut])
            train_infos.append(train_info)
            self.buffer[ut].after_update()
        return train_infos

    def save(self):
        for ut in self.unique_unit_types:
            torch.save(self.trainer[ut].policy.actor.state_dict(), f"{self.save_dir}/actor_unit_type{ut}.pt")
            torch.save(self.trainer[ut].policy.critic.state_dict(), f"{self.save_dir}/critic_unit_type{ut}.pt")

    def restore(self):
        for ut in self.unique_unit_types:
            actor_state = torch.load(f"{self.model_dir}/actor_unit_type{ut}.pt")
            self.policy[ut].actor.load_state_dict(actor_state)
            critic_state = torch.load(f"{self.model_dir}/critic_unit_type{ut}.pt")
            self.policy[ut].critic.load_state_dict(critic_state)

    def log_train(self, train_infos, total_num_steps):
        for idx, info in enumerate(train_infos):
            ut = self.unique_unit_types[idx]
            for k, v in info.items():
                key = f"unit_type{ut}/{k}"
                if self.use_wandb:
                    wandb.log({key: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(key, {key: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
