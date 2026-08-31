import numpy as np
import torch

from onpolicy.utils.shared_buffer import SharedReplayBuffer, _cast, _flatten


class CSTMReplayBuffer(SharedReplayBuffer):
    """Shared MAPPO buffer with privileged teammate-action training labels."""

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        if act_space.__class__.__name__ != "Discrete":
            raise NotImplementedError("B1 currently supports Discrete actions only")
        super().__init__(args, num_agents, obs_space, cent_obs_space, act_space)
        self.teammate_actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents,
             num_agents - 1, 1), dtype=np.int64)
        self.teammate_active_masks = np.ones_like(
            self.teammate_actions, dtype=np.float32)

    @staticmethod
    def build_teammate_targets(actions, active_masks=None):
        """Return labels ordered by ascending teammate id for every focal agent."""
        num_agents = actions.shape[1]
        targets = np.stack(
            [np.delete(actions, agent_id, axis=1)
             for agent_id in range(num_agents)], axis=1).astype(np.int64)
        if active_masks is None:
            masks = np.ones_like(targets, dtype=np.float32)
        else:
            masks = np.stack(
                [np.delete(active_masks, agent_id, axis=1)
                 for agent_id in range(num_agents)], axis=1).astype(np.float32)
        return targets, masks

    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic,
               actions, action_log_probs, value_preds, rewards, masks,
               bad_masks=None, active_masks=None, available_actions=None,
               teammate_actions=None, teammate_active_masks=None):
        if teammate_actions is None:
            raise ValueError("CSTM buffer requires teammate action labels")
        step = self.step
        self.teammate_actions[step] = teammate_actions.copy()
        if teammate_active_masks is not None:
            self.teammate_active_masks[step] = teammate_active_masks.copy()
        super().insert(
            share_obs, obs, rnn_states_actor, rnn_states_critic, actions,
            action_log_probs, value_preds, rewards, masks, bad_masks,
            active_masks, available_actions)

    def recurrent_generator(self, advantages, num_mini_batch,
                            data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        if batch_size % data_chunk_length != 0:
            raise ValueError("rollout batch must be divisible by data_chunk_length")
        data_chunks = batch_size // data_chunk_length
        if data_chunks < num_mini_batch:
            raise ValueError("fewer recurrent chunks than mini-batches")
        mini_batch_size = data_chunks // num_mini_batch
        sampler = torch.randperm(data_chunks).numpy()

        share_obs = _cast(self.share_obs[:-1])
        obs = _cast(self.obs[:-1])
        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        teammate_actions = self.teammate_actions.transpose(1, 2, 0, 3, 4).reshape(
            -1, *self.teammate_actions.shape[3:])
        teammate_active_masks = self.teammate_active_masks.transpose(
            1, 2, 0, 3, 4).reshape(-1, *self.teammate_active_masks.shape[3:])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(
            -1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(
            1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_critic.shape[3:])
        available_actions = (_cast(self.available_actions[:-1])
                             if self.available_actions is not None else None)

        for batch_id in range(num_mini_batch):
            indices = sampler[batch_id * mini_batch_size:
                              (batch_id + 1) * mini_batch_size]
            sequences = [[] for _ in range(12)]
            rnn_batch, critic_rnn_batch = [], []
            available_batch = []
            for index in indices:
                start = index * data_chunk_length
                end = start + data_chunk_length
                arrays = (share_obs, obs, actions, value_preds, returns, masks,
                          active_masks, action_log_probs, advantages,
                          teammate_actions, teammate_active_masks)
                for bucket, array in zip(sequences[:11], arrays):
                    bucket.append(array[start:end])
                if available_actions is not None:
                    available_batch.append(available_actions[start:end])
                rnn_batch.append(rnn_states[start])
                critic_rnn_batch.append(rnn_states_critic[start])

            L, N = data_chunk_length, len(indices)
            flat = [_flatten(L, N, np.stack(bucket, axis=1))
                    for bucket in sequences[:11]]
            share_obs_batch, obs_batch, actions_batch, value_preds_batch, \
                return_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, adv_targ, teammate_actions_batch, \
                teammate_active_masks_batch = flat
            available_actions_batch = (
                _flatten(L, N, np.stack(available_batch, axis=1))
                if available_actions is not None else None)
            rnn_states_batch = np.stack(rnn_batch).reshape(
                N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(critic_rnn_batch).reshape(
                N, *self.rnn_states_critic.shape[3:])

            yield (share_obs_batch, obs_batch, rnn_states_batch,
                   rnn_states_critic_batch, actions_batch, value_preds_batch,
                   return_batch, masks_batch, active_masks_batch,
                   old_action_log_probs_batch, adv_targ,
                   available_actions_batch, teammate_actions_batch,
                   teammate_active_masks_batch)
