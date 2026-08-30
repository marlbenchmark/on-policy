import torch
import torch.nn as nn

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.utils.util import check
from onpolicy.algorithms.cstm_mappo.algorithm.teammate_model import TeammateModel


class B1Actor(R_Actor):
    """RMAPPO actor augmented with a single-head teammate representation.

    ``act`` remains the original RMAPPO action head.  The B1 path fuses the
    local recurrent feature with ``z`` before that head.  Fusion starts as
    ``[I, 0]`` so loading a B0 actor is an exact, stable initialization.
    """

    def __init__(self, args, obs_space, action_space, num_agents,
                 device=torch.device("cpu")):
        if action_space.__class__.__name__ != "Discrete":
            raise NotImplementedError("B1 currently supports Discrete actions only")
        super().__init__(args, obs_space, action_space, device)
        self.use_teammate_policy = args.cstm_use_teammate_policy
        self.teammate_model = TeammateModel(
            self.hidden_size, args.cstm_latent_dim, num_agents - 1,
            action_space.n, self._use_orthogonal)
        self.tm_fusion = nn.Linear(self.hidden_size + args.cstm_latent_dim,
                                   self.hidden_size)
        with torch.no_grad():
            self.tm_fusion.weight.zero_()
            self.tm_fusion.bias.zero_()
            self.tm_fusion.weight[:, :self.hidden_size].copy_(
                torch.eye(self.hidden_size))
        self.to(device)

    def _features(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        features = self.base(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            features, rnn_states = self.rnn(features, rnn_states, masks)
        return features, rnn_states

    def _teammate_features(self, local_features):
        latent, logits = self.teammate_model(local_features)
        if self.use_teammate_policy:
            policy_features = self.tm_fusion(torch.cat((local_features, latent), dim=-1))
        else:
            policy_features = local_features
        return policy_features, logits

    def forward(self, obs, rnn_states, masks, available_actions=None,
                deterministic=False):
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        local_features, rnn_states = self._features(obs, rnn_states, masks)
        policy_features, _ = self._teammate_features(local_features)
        actions, action_log_probs = self.act(
            policy_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions_with_teammates(self, obs, rnn_states, action, masks,
                                        available_actions=None, active_masks=None):
        action = check(action).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        local_features, _ = self._features(obs, rnn_states, masks)
        policy_features, teammate_logits = self._teammate_features(local_features)
        action_log_probs, dist_entropy = self.act.evaluate_actions(
            policy_features, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None)
        return action_log_probs, dist_entropy, teammate_logits

    def evaluate_actions(self, obs, rnn_states, action, masks,
                         available_actions=None, active_masks=None):
        action_log_probs, dist_entropy, _ = self.evaluate_actions_with_teammates(
            obs, rnn_states, action, masks, available_actions, active_masks)
        return action_log_probs, dist_entropy

    @torch.no_grad()
    def teammate_predictions(self, obs, rnn_states, masks):
        local_features, rnn_states = self._features(obs, rnn_states, masks)
        _, logits = self.teammate_model(local_features)
        return logits, rnn_states
