import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_gard_norm


class CSTM_MAPPO(R_MAPPO):
    """B1 trainer: PPO plus single-head teammate-action prediction."""

    def __init__(self, args, policy, device=torch.device("cpu")):
        super().__init__(args, policy, device)
        if not self._use_recurrent_policy:
            raise ValueError("cstm_mappo B1 requires --use_recurrent_policy")
        self.aux_coef = args.cstm_aux_coef

    def ppo_update(self, sample, update_actor=True):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, \
            actions_batch, value_preds_batch, return_batch, masks_batch, \
            active_masks_batch, old_action_log_probs_batch, adv_targ, \
            available_actions_batch, teammate_actions_batch, \
            teammate_active_masks_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        teammate_actions_batch = check(teammate_actions_batch).long().to(self.device)
        teammate_active_masks_batch = check(teammate_active_masks_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy, teammate_logits = \
            self.policy.evaluate_actions_with_teammates(
                share_obs_batch, obs_batch, rnn_states_batch,
                rnn_states_critic_batch, actions_batch, masks_batch,
                available_actions_batch, active_masks_batch)

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch).sum() / active_masks_batch.sum().clamp_min(1.0)
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        targets = teammate_actions_batch.squeeze(-1)
        target_masks = teammate_active_masks_batch.squeeze(-1)
        per_target_loss = F.cross_entropy(
            teammate_logits.reshape(-1, teammate_logits.shape[-1]),
            targets.reshape(-1), reduction="none").reshape_as(target_masks)
        valid_count = target_masks.sum().clamp_min(1.0)
        team_prediction_loss = (per_target_loss * target_masks).sum() / valid_count

        self.policy.actor_optimizer.zero_grad()
        actor_loss = (policy_action_loss - dist_entropy * self.entropy_coef
                      + self.aux_coef * team_prediction_loss)
        if update_actor:
            actor_loss.backward()
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        with torch.no_grad():
            predictions = teammate_logits.argmax(dim=-1)
            accuracy = ((predictions == targets).float() * target_masks).sum() / valid_count
            valid_targets = targets[target_masks.bool()]
            if valid_targets.numel():
                counts = torch.bincount(
                    valid_targets, minlength=teammate_logits.shape[-1])
                majority_accuracy = counts.max().float() / valid_targets.numel()
            else:
                majority_accuracy = torch.zeros((), device=self.device)
            recalls = []
            for action_id in range(teammate_logits.shape[-1]):
                class_mask = (targets == action_id).float() * target_masks
                denom = class_mask.sum()
                recall = (((predictions == action_id).float() * class_mask).sum()
                          / denom.clamp_min(1.0))
                recalls.append(recall)

        metrics = {
            "value_loss": value_loss,
            "critic_grad_norm": critic_grad_norm,
            "policy_loss": policy_action_loss,
            "dist_entropy": dist_entropy,
            "actor_grad_norm": actor_grad_norm,
            "ratio": imp_weights.mean(),
            "team_prediction_loss": team_prediction_loss,
            "team_action_accuracy": accuracy,
            "majority_action_accuracy": majority_accuracy,
        }
        metrics.update({f"team_action_recall_{i}": value
                        for i, value in enumerate(recalls)})
        return metrics

    def train(self, buffer, update_actor=True):
        if self._use_popart or self._use_valuenorm:
            advantages = (buffer.returns[:-1]
                          - self.value_normalizer.denormalize(buffer.value_preds[:-1]))
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        advantages = ((advantages - np.nanmean(advantages_copy))
                      / (np.nanstd(advantages_copy) + 1e-5))

        totals = {}
        updates = 0
        for _ in range(self.ppo_epoch):
            generator = buffer.recurrent_generator(
                advantages, self.num_mini_batch, self.data_chunk_length)
            for sample in generator:
                metrics = self.ppo_update(sample, update_actor)
                for key, value in metrics.items():
                    scalar = value.item() if torch.is_tensor(value) else float(value)
                    totals[key] = totals.get(key, 0.0) + scalar
                updates += 1
        return {key: value / updates for key, value in totals.items()}
