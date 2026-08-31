import numpy as np
import torch
import torch.nn as nn

from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_gard_norm


class Selective_MAPPO(R_MAPPO):
    """PPO training for only the selector and critic; B0/B2 stay frozen."""

    def __init__(self, args, policy, device=torch.device("cpu")):
        super().__init__(args, policy, device)
        self.fallback_cost = args.cstm_selector_fallback_cost
        self.coverage_coef = args.cstm_selector_coverage_coef
        self.coverage_target = args.cstm_selector_coverage_target
        self.selector_entropy_coef = args.cstm_selector_entropy_coef
        if self.fallback_cost < 0 or self.coverage_coef < 0:
            raise ValueError("selector regularization coefficients must be nonnegative")
        if not 0 <= self.coverage_target <= 1:
            raise ValueError("selector coverage target must be in [0, 1]")

    @staticmethod
    def _correlation(x, y):
        x = x.reshape(-1) - x.mean()
        y = y.reshape(-1) - y.mean()
        denominator = torch.sqrt(x.square().sum() * y.square().sum())
        if denominator <= 1e-8:
            return torch.zeros((), dtype=x.dtype, device=x.device)
        return (x * y).sum() / denominator

    def ppo_update(self, sample, update_actor=True):
        (share_obs_batch, obs_batch, b2_rnn_states_batch,
         b0_rnn_states_batch, rnn_states_critic_batch, actions_batch,
         value_preds_batch, return_batch, masks_batch, active_masks_batch,
         old_action_log_probs_batch, adv_targ,
         available_actions_batch) = sample

        old_action_log_probs_batch = check(
            old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy, diagnostics = \
            self.policy.evaluate_actions(
                share_obs_batch, obs_batch, b2_rnn_states_batch,
                b0_rnn_states_batch, rnn_states_critic_batch, actions_batch,
                masks_batch, available_actions_batch, active_masks_batch)
        imp_weights = torch.exp(
            action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(
            imp_weights, 1.0 - self.clip_param,
            1.0 + self.clip_param) * adv_targ
        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch).sum() \
                / active_masks_batch.sum().clamp_min(1.0)
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1,
                keepdim=True).mean()

        b2_gate = diagnostics["b2_gate"]
        fallback_probability = diagnostics["fallback_probability"]
        gate_entropy = -(
            b2_gate.clamp_min(1e-8).log() * b2_gate
            + fallback_probability.clamp_min(1e-8).log()
            * fallback_probability).mean()
        fallback_penalty = fallback_probability.mean()
        coverage_penalty = (
            b2_gate.mean() - self.coverage_target).square()
        selector_loss = (
            policy_action_loss
            + self.fallback_cost * fallback_penalty
            + self.coverage_coef * coverage_penalty
            - self.selector_entropy_coef * gate_entropy)

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (selector_loss - self.entropy_coef * dist_entropy).backward()
        selector_parameters = self.policy.actor.selector_parameters
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                selector_parameters, self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(selector_parameters)
        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(
                self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        risk = diagnostics["detector_risk"]
        gate_risk_correlation = self._correlation(b2_gate.detach(), risk)
        extra = {
            "selector_loss": selector_loss.detach(),
            "mean_b2_gate": b2_gate.detach().mean(),
            "gate_std": b2_gate.detach().std(unbiased=False),
            "soft_fallback_rate": fallback_penalty.detach(),
            "selector_entropy": gate_entropy.detach(),
            "mean_detector_risk": risk.detach().mean(),
            "mean_action_divergence": diagnostics[
                "action_divergence"].detach().mean(),
            "gate_risk_correlation": gate_risk_correlation.detach(),
            "uncertainty_scale": diagnostics[
                "uncertainty_scale"].detach(),
        }
        return (value_loss, critic_grad_norm, policy_action_loss,
                dist_entropy, actor_grad_norm, imp_weights, extra)

    def train(self, buffer, update_actor=True):
        if self._use_popart or self._use_valuenorm:
            advantages = (buffer.returns[:-1]
                          - self.value_normalizer.denormalize(
                              buffer.value_preds[:-1]))
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        valid_advantages = advantages.copy()
        valid_advantages[buffer.active_masks[:-1] == 0.0] = np.nan
        advantages = ((advantages - np.nanmean(valid_advantages))
                      / (np.nanstd(valid_advantages) + 1e-5))

        keys = ("value_loss", "policy_loss", "dist_entropy",
                "actor_grad_norm", "critic_grad_norm", "ratio",
                "selector_loss", "mean_b2_gate", "gate_std",
                "soft_fallback_rate", "selector_entropy",
                "mean_detector_risk", "mean_action_divergence",
                "gate_risk_correlation", "uncertainty_scale")
        train_info = {key: 0.0 for key in keys}
        updates = 0
        for _ in range(self.ppo_epoch):
            generator = buffer.recurrent_generator(
                advantages, self.num_mini_batch, self.data_chunk_length)
            for sample in generator:
                (value_loss, critic_grad_norm, policy_loss, dist_entropy,
                 actor_grad_norm, imp_weights, extra) = self.ppo_update(
                    sample, update_actor)
                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += float(actor_grad_norm)
                train_info["critic_grad_norm"] += float(critic_grad_norm)
                train_info["ratio"] += imp_weights.mean().item()
                for key, value in extra.items():
                    train_info[key] += float(value)
                updates += 1
        for key in train_info:
            train_info[key] /= max(1, updates)
        return train_info
