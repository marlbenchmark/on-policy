import numpy as np

from onpolicy.runner.shared.mpe_runner import MPERunner
from onpolicy.utils.cstm_buffer import CSTMReplayBuffer


class CSTMMPErunner(MPERunner):
    """MPE runner that constructs B1 supervision after action collection."""

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, \
            rnn_states, rnn_states_critic = data

        # The labels are stored only after actions have been sampled.  They are
        # never arguments to get_actions(), act(), or the actor forward pass.
        current_active_masks = self.buffer.active_masks[self.buffer.step]
        teammate_actions, teammate_active_masks = \
            CSTMReplayBuffer.build_teammate_targets(
                actions, current_active_masks)

        rnn_states[dones] = np.zeros(
            ((dones).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        rnn_states_critic[dones] = np.zeros(
            ((dones).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
                        dtype=np.float32)
        masks[dones] = 0.0

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(
                self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs, obs, rnn_states, rnn_states_critic, actions,
            action_log_probs, values, rewards, masks,
            teammate_actions=teammate_actions,
            teammate_active_masks=teammate_active_masks)
