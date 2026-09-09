import torch
import torch.nn as nn

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.utils.util import check
from onpolicy.algorithms.cstm_mappo.algorithm.teammate_model import TeammateModel


class B1Actor(R_Actor):
    """RMAPPO actor augmented with a teammate representation.

    ``act`` remains the original RMAPPO action head.  The B1 path fuses the
    local recurrent feature with ``z`` before that head.  Fusion starts as
    ``[I, 0]`` so loading a B0 actor is an exact, stable initialization. B2
    retains this path and adds ensemble disagreement through a zero-initialized
    residual adapter.
    """

    def __init__(self, args, obs_space, action_space, num_agents,
                 device=torch.device("cpu")):
        if action_space.__class__.__name__ != "Discrete":
            raise NotImplementedError("B1 currently supports Discrete actions only")
        super().__init__(args, obs_space, action_space, device)
        self.use_teammate_policy = args.cstm_use_teammate_policy
        self.num_heads = args.cstm_num_heads
        self.use_uncertainty_feature = args.cstm_use_uncertainty_feature
        self.use_separate_detector = args.cstm_use_separate_detector
        self.uncertainty_override = None
        policy_prior_scale = (
            0.0 if self.use_separate_detector
            else args.cstm_random_prior_scale)
        self.teammate_model = TeammateModel(
            self.hidden_size, args.cstm_latent_dim, num_agents - 1,
            action_space.n, self._use_orthogonal, args.cstm_num_heads,
            policy_prior_scale)
        self.uncertainty_detector = None
        if self.use_separate_detector:
            self.uncertainty_detector = TeammateModel(
                self.hidden_size, args.cstm_latent_dim, num_agents - 1,
                action_space.n, self._use_orthogonal, args.cstm_num_heads,
                args.cstm_random_prior_scale)
        self.tm_fusion = nn.Linear(self.hidden_size + args.cstm_latent_dim,
                                   self.hidden_size)
        with torch.no_grad():
            self.tm_fusion.weight.zero_()
            self.tm_fusion.bias.zero_()
            self.tm_fusion.weight[:, :self.hidden_size].copy_(
                torch.eye(self.hidden_size))
        if self.num_heads > 1:
            self.uncertainty_adapter = nn.Linear(1, self.hidden_size)
            with torch.no_grad():
                self.uncertainty_adapter.weight.zero_()
                self.uncertainty_adapter.bias.zero_()
        else:
            self.uncertainty_adapter = None
        self.to(device)

    def set_uncertainty_override(self, value=None):
        """Override only the uncertainty value consumed by the policy.

        Predictions and reported diagnostics remain unchanged. ``None`` uses
        predicted uncertainty; a scalar supports inference-only ablations.
        """
        if value is None:
            self.uncertainty_override = None
            return
        value = float(value)
        if not torch.isfinite(torch.tensor(value)) or value < 0.0:
            raise ValueError(
                "uncertainty override must be finite and nonnegative")
        self.uncertainty_override = value

    def _features(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        features = self.base(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            features, rnn_states = self.rnn(features, rnn_states, masks)
        return features, rnn_states

    def _teammate_features(self, local_features):
        latent, head_logits, mean_probs, disagreement = \
            self.teammate_model(local_features)
        if self.use_teammate_policy:
            policy_features = self.tm_fusion(torch.cat((local_features, latent), dim=-1))
            if self.use_uncertainty_feature and self.uncertainty_adapter is not None:
                mean_uncertainty = disagreement.mean(dim=-1, keepdim=True)
                if self.uncertainty_override is not None:
                    mean_uncertainty = torch.full_like(
                        mean_uncertainty, self.uncertainty_override)
                policy_features = policy_features + self.uncertainty_adapter(
                    mean_uncertainty)
        else:
            policy_features = local_features
        return policy_features, head_logits, mean_probs, disagreement

    def forward(self, obs, rnn_states, masks, available_actions=None,
                deterministic=False):
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        local_features, rnn_states = self._features(obs, rnn_states, masks)
        policy_features, _, _, _ = self._teammate_features(local_features)
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
        policy_features, teammate_logits, _, teammate_uncertainty = \
            self._teammate_features(local_features)
        action_log_probs, dist_entropy = self.act.evaluate_actions(
            policy_features, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None)
        return (action_log_probs, dist_entropy, teammate_logits,
                teammate_uncertainty)

    def evaluate_actions(self, obs, rnn_states, action, masks,
                         available_actions=None, active_masks=None):
        action_log_probs, dist_entropy, _, _ = self.evaluate_actions_with_teammates(
            obs, rnn_states, action, masks, available_actions, active_masks)
        return action_log_probs, dist_entropy

    @torch.no_grad()
    def teammate_predictions(self, obs, rnn_states, masks):
        local_features, rnn_states = self._features(obs, rnn_states, masks)
        _, _, mean_probs, _ = self.teammate_model(local_features)
        return mean_probs, rnn_states

    @torch.no_grad()
    def teammate_diagnostics(self, obs, rnn_states, masks):
        return self.detector_outputs(obs, rnn_states, masks)

    def teammate_outputs(self, obs, rnn_states, masks):
        """Return policy-branch predictions for clean auxiliary training."""
        local_features, rnn_states = self._features(obs, rnn_states, masks)
        _, head_logits, mean_probs, disagreement = self.teammate_model(
            local_features)
        return head_logits, mean_probs, disagreement, rnn_states

    def detector_outputs(self, obs, rnn_states, masks):
        """Return detector predictions without a gradient path to policy features."""
        if self.uncertainty_detector is None:
            return self.teammate_outputs(obs, rnn_states, masks)
        local_features, rnn_states = self._features(obs, rnn_states, masks)
        _, head_logits, mean_probs, disagreement = \
            self.uncertainty_detector(local_features.detach())
        return head_logits, mean_probs, disagreement, rnn_states
