import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.cstm_mappo.algorithm.dual_policy_actor import B1Actor
from onpolicy.algorithms.utils.util import check


class SelectiveActor(nn.Module):
    """Frozen B0/B2 actors with a trainable monotonic mixture selector."""

    def __init__(self, args, obs_space, action_space, num_agents,
                 device=torch.device("cpu")):
        super().__init__()
        if action_space.__class__.__name__ != "Discrete":
            raise NotImplementedError(
                "selective_mappo currently supports Discrete actions only")
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_dim = action_space.n
        self.hidden_size = args.hidden_size
        self.num_agents = num_agents

        self.b0_actor = R_Actor(args, obs_space, action_space, device)
        b2_args = args
        self.b2_actor = B1Actor(
            b2_args, obs_space, action_space, num_agents, device)
        if self.b2_actor.uncertainty_detector is None:
            raise ValueError(
                "selective_mappo requires the separate frozen detector")

        selector_input_dim = (
            args.hidden_size + args.cstm_latent_dim + 1)
        selector_hidden = args.cstm_selector_hidden_dim
        self.selector_value = nn.Sequential(
            nn.Linear(selector_input_dim, selector_hidden),
            nn.Tanh(),
            nn.Linear(selector_hidden, 1),
        )
        with torch.no_grad():
            self.selector_value[-1].weight.zero_()
            self.selector_value[-1].bias.zero_()
        initial_scale = float(args.cstm_selector_initial_scale)
        if initial_scale <= 0:
            raise ValueError("selector initial scale must be positive")
        raw_scale = math.log(math.expm1(initial_scale)) \
            if initial_scale < 20 else initial_scale
        self.raw_uncertainty_scale = nn.Parameter(torch.tensor(raw_scale))
        self.selector_bias = nn.Parameter(torch.tensor(
            initial_scale * float(args.cstm_selector_initial_threshold)))
        self.eps = 1e-8

        for module in (self.b0_actor, self.b2_actor):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()
        self.to(device)

    @property
    def selector_parameters(self):
        return (list(self.selector_value.parameters())
                + [self.raw_uncertainty_scale, self.selector_bias])

    @property
    def uncertainty_scale(self):
        return F.softplus(self.raw_uncertainty_scale)

    def selector_gate(self, selector_context, detector_risk):
        """B2 probability, monotone non-increasing in detector risk."""
        selector_logit = (self.selector_value(selector_context)
                           + self.selector_bias
                           - self.uncertainty_scale * detector_risk)
        return torch.sigmoid(selector_logit)

    def load_pretrained(self, b0_actor_path, b2_actor_path, map_location=None):
        self.b0_actor.load_state_dict(torch.load(
            b0_actor_path, map_location=map_location))
        self.b2_actor.load_state_dict(torch.load(
            b2_actor_path, map_location=map_location))
        self.b0_actor.eval()
        self.b2_actor.eval()

    def train(self, mode=True):
        super().train(mode)
        # Frozen branches must remain inference-only even while selector trains.
        self.b0_actor.eval()
        self.b2_actor.eval()
        return self

    def _b0_features(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        features = self.b0_actor.base(obs)
        if (self.b0_actor._use_naive_recurrent_policy
                or self.b0_actor._use_recurrent_policy):
            features, rnn_states = self.b0_actor.rnn(
                features, rnn_states, masks)
        return features, rnn_states

    def _distribution_outputs(self, obs, b2_rnn_states, b0_rnn_states,
                              masks, available_actions=None):
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        b0_features, next_b0_states = self._b0_features(
            obs, b0_rnn_states, masks)
        b2_local_features, next_b2_states = self.b2_actor._features(
            obs, b2_rnn_states, masks)
        b2_policy_features, _, _, _ = self.b2_actor._teammate_features(
            b2_local_features)
        detector_latent, _, _, detector_uncertainty = \
            self.b2_actor.uncertainty_detector(b2_local_features.detach())

        b0_probs = self.b0_actor.act.get_probs(
            b0_features, available_actions)
        b2_probs = self.b2_actor.act.get_probs(
            b2_policy_features, available_actions)
        risk = detector_uncertainty.mean(dim=-1, keepdim=True)
        action_divergence = 0.5 * torch.abs(
            b0_probs - b2_probs).sum(dim=-1, keepdim=True)
        selector_context = torch.cat((
            b2_local_features.detach(), detector_latent.detach(),
            action_divergence.detach()), dim=-1)
        b2_gate = self.selector_gate(selector_context, risk.detach())
        mixed_probs = (b2_gate * b2_probs
                       + (1.0 - b2_gate) * b0_probs)
        mixed_probs = mixed_probs.clamp_min(self.eps)
        mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)
        diagnostics = {
            "b2_gate": b2_gate,
            "fallback_probability": 1.0 - b2_gate,
            "detector_risk": risk,
            "action_divergence": action_divergence,
            "uncertainty_scale": self.uncertainty_scale,
        }
        return (mixed_probs, next_b2_states, next_b0_states, diagnostics,
                b0_probs, b2_probs)

    def forward(self, obs, b2_rnn_states, b0_rnn_states, masks,
                available_actions=None, deterministic=False):
        mixed_probs, next_b2, next_b0, diagnostics, _, _ = \
            self._distribution_outputs(
                obs, b2_rnn_states, b0_rnn_states, masks,
                available_actions)
        distribution = torch.distributions.Categorical(probs=mixed_probs)
        actions = (mixed_probs.argmax(dim=-1) if deterministic
                   else distribution.sample()).unsqueeze(-1)
        action_log_probs = distribution.log_prob(
            actions.squeeze(-1)).unsqueeze(-1)
        return (actions, action_log_probs, next_b2, next_b0,
                diagnostics)

    def evaluate_actions(self, obs, b2_rnn_states, b0_rnn_states, action,
                         masks, available_actions=None, active_masks=None):
        action = check(action).long().to(self.tpdv["device"])
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        mixed_probs, _, _, diagnostics, _, _ = self._distribution_outputs(
            obs, b2_rnn_states, b0_rnn_states, masks, available_actions)
        distribution = torch.distributions.Categorical(probs=mixed_probs)
        action_log_probs = distribution.log_prob(
            action.squeeze(-1)).unsqueeze(-1)
        entropy = distribution.entropy().unsqueeze(-1)
        if active_masks is not None:
            dist_entropy = (entropy * active_masks).sum() \
                / active_masks.sum().clamp_min(1.0)
        else:
            dist_entropy = entropy.mean()
        return action_log_probs, dist_entropy, diagnostics
