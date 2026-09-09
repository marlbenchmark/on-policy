import numpy as np
import torch

from onpolicy.utils.shared_buffer import SharedReplayBuffer, _cast, _flatten


class SelectiveReplayBuffer(SharedReplayBuffer):
    """MAPPO buffer carrying independent frozen B0 and B2 RNN states."""

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        super().__init__(args, num_agents, obs_space, cent_obs_space, act_space)
        self.b0_rnn_states = np.zeros_like(self.rnn_states)

    def insert(self, share_obs, obs, b2_rnn_states, b0_rnn_states,
               rnn_states_critic, actions, action_log_probs, value_preds,
               rewards, masks, bad_masks=None, active_masks=None,
               available_actions=None):
        step = self.step
        self.b0_rnn_states[step + 1] = b0_rnn_states.copy()
        super().insert(
            share_obs, obs, b2_rnn_states, rnn_states_critic, actions,
            action_log_probs, value_preds, rewards, masks, bad_masks,
            active_masks, available_actions)

    def after_update(self):
        super().after_update()
        self.b0_rnn_states[0] = self.b0_rnn_states[-1].copy()

    def recurrent_generator(self, advantages, num_mini_batch,
                            data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        if batch_size % data_chunk_length != 0:
            raise ValueError("rollout batch must divide into complete RNN chunks")
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
        b2_states = self.rnn_states[:-1].transpose(
            1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        b0_states = self.b0_rnn_states[:-1].transpose(
            1, 2, 0, 3, 4).reshape(-1, *self.b0_rnn_states.shape[3:])
        critic_states = self.rnn_states_critic[:-1].transpose(
            1, 2, 0, 3, 4).reshape(
                -1, *self.rnn_states_critic.shape[3:])
        available_actions = (
            _cast(self.available_actions[:-1])
            if self.available_actions is not None else None)

        for batch_id in range(num_mini_batch):
            indices = sampler[batch_id * mini_batch_size:
                              (batch_id + 1) * mini_batch_size]
            sequences = [[] for _ in range(9)]
            b2_batch, b0_batch, critic_batch, available_batch = [], [], [], []
            arrays = (share_obs, obs, actions, value_preds, returns, masks,
                      active_masks, action_log_probs, advantages)
            for index in indices:
                start = index * data_chunk_length
                stop = start + data_chunk_length
                for bucket, array in zip(sequences, arrays):
                    bucket.append(array[start:stop])
                if available_actions is not None:
                    available_batch.append(available_actions[start:stop])
                b2_batch.append(b2_states[start])
                b0_batch.append(b0_states[start])
                critic_batch.append(critic_states[start])

            length, count = data_chunk_length, len(indices)
            flat = [_flatten(length, count, np.stack(bucket, axis=1))
                    for bucket in sequences]
            (share_obs_batch, obs_batch, actions_batch, value_preds_batch,
             return_batch, masks_batch, active_masks_batch,
             old_action_log_probs_batch, adv_targ) = flat
            available_actions_batch = (
                _flatten(length, count, np.stack(available_batch, axis=1))
                if available_actions is not None else None)
            b2_rnn_states_batch = np.stack(b2_batch).reshape(
                count, *self.rnn_states.shape[3:])
            b0_rnn_states_batch = np.stack(b0_batch).reshape(
                count, *self.b0_rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(critic_batch).reshape(
                count, *self.rnn_states_critic.shape[3:])
            yield (share_obs_batch, obs_batch, b2_rnn_states_batch,
                   b0_rnn_states_batch, rnn_states_critic_batch,
                   actions_batch, value_preds_batch, return_batch,
                   masks_batch, active_masks_batch,
                   old_action_log_probs_batch, adv_targ,
                   available_actions_batch)
