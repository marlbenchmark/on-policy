from pathlib import Path

import torch

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic
from onpolicy.algorithms.utils.util import check
from onpolicy.algorithms.selective_mappo.selective_actor import SelectiveActor
from onpolicy.utils.util import update_linear_schedule


class SelectivePolicy:
    """Policy wrapper for a frozen B0/B2 mixture and trainable selector."""

    def __init__(self, args, obs_space, cent_obs_space, act_space,
                 device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.actor = SelectiveActor(
            args, obs_space, act_space, args.num_agents, device)
        self.critic = R_Critic(args, cent_obs_space, device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.selector_parameters, lr=self.lr, eps=args.opti_eps,
            weight_decay=args.weight_decay)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=args.opti_eps,
            weight_decay=args.weight_decay)

    def load_initial_checkpoints(self, b2_model_dir, b0_model_dir):
        b2_model_dir = Path(b2_model_dir)
        b0_model_dir = Path(b0_model_dir)
        actor_state = torch.load(
            b2_model_dir / "actor.pt", map_location=self.device)
        if any(key.startswith("b2_actor.") for key in actor_state):
            self.actor.load_state_dict(actor_state)
        else:
            self.actor.b2_actor.load_state_dict(actor_state)
            self.actor.b0_actor.load_state_dict(torch.load(
                b0_model_dir / "actor.pt", map_location=self.device))
        self.critic.load_state_dict(torch.load(
            b2_model_dir / "critic.pt", map_location=self.device))

    def lr_decay(self, episode, episodes):
        update_linear_schedule(
            self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(
            self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, b2_rnn_states, b0_rnn_states,
                    rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        actions, action_log_probs, next_b2, next_b0, diagnostics = self.actor(
            obs, b2_rnn_states, b0_rnn_states, masks, available_actions,
            deterministic)
        values, next_critic = self.critic(
            cent_obs, rnn_states_critic, masks)
        return (values, actions, action_log_probs, next_b2, next_b0,
                next_critic, diagnostics)

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, b2_rnn_states, b0_rnn_states,
                         rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        action_log_probs, dist_entropy, diagnostics = \
            self.actor.evaluate_actions(
                obs, b2_rnn_states, b0_rnn_states, action, masks,
                available_actions, active_masks)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy, diagnostics

    @torch.no_grad()
    def act(self, obs, b2_rnn_states, b0_rnn_states, masks,
            available_actions=None, deterministic=False):
        actions, _, next_b2, next_b0, diagnostics = self.actor(
            obs, b2_rnn_states, b0_rnn_states, masks, available_actions,
            deterministic)
        return actions, next_b2, next_b0, diagnostics
