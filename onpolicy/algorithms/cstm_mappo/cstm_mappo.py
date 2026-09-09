import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_gard_norm


class CSTM_MAPPO(R_MAPPO):
    """B1/B2 trainer: PPO plus bootstrap teammate-action prediction."""

    def __init__(self, args, policy, device=torch.device("cpu")):
        super().__init__(args, policy, device)
        if not self._use_recurrent_policy:
            raise ValueError("CSTM teammate modelling requires recurrent policy")
        self.aux_coef = args.cstm_aux_coef
        self.detector_aux_coef = args.cstm_detector_aux_coef
        self.use_separate_detector = args.cstm_use_separate_detector
        self.detector_only = args.cstm_detector_only
        self.num_heads = args.cstm_num_heads
        self.bootstrap_prob = args.cstm_bootstrap_prob
        if not 0 < self.bootstrap_prob <= 1:
            raise ValueError("cstm_bootstrap_prob must be in (0, 1]")
        self.ood_rank_coef = args.cstm_ood_rank_coef
        self.ood_rank_margin = args.cstm_ood_rank_margin
        self.ood_noise_std = args.cstm_ood_noise_std
        self.ood_mask_prob = args.cstm_ood_mask_prob
        self.ood_delay_prob = args.cstm_ood_delay_prob
        self.uncertainty_cal_coef = args.cstm_uncertainty_cal_coef
        self.uncertainty_target_scale = args.cstm_uncertainty_target_scale
        self.uncertainty_corr_coef = args.cstm_uncertainty_corr_coef
        self.num_agents = args.num_agents
        self.num_landmarks = args.num_landmarks
        if self.ood_rank_coef < 0 or self.ood_rank_margin < 0:
            raise ValueError("OOD rank coefficient and margin must be non-negative")
        if self.ood_noise_std < 0:
            raise ValueError("cstm_ood_noise_std must be non-negative")
        if not 0 <= self.ood_mask_prob <= 1:
            raise ValueError("cstm_ood_mask_prob must be in [0, 1]")
        if not 0 <= self.ood_delay_prob <= 1:
            raise ValueError("cstm_ood_delay_prob must be in [0, 1]")
        if self.uncertainty_cal_coef < 0:
            raise ValueError("cstm_uncertainty_cal_coef must be non-negative")
        if self.uncertainty_corr_coef < 0:
            raise ValueError("cstm_uncertainty_corr_coef must be non-negative")
        if self.detector_aux_coef < 0:
            raise ValueError("cstm_detector_aux_coef must be non-negative")
        if self.use_separate_detector and self.num_heads < 2:
            raise ValueError("separate uncertainty detector requires multiple heads")
        if self.detector_only and not self.use_separate_detector:
            raise ValueError("detector-only training requires a separate detector")
        if not 0 < self.uncertainty_target_scale <= 1:
            raise ValueError("cstm_uncertainty_target_scale must be in (0, 1]")

    def _corrupt_teammate_positions(self, obs, masks):
        """Training-only augmentation of simple_spread teammate positions."""
        corrupted = check(obs).to(**self.tpdv).clone()
        start = 4 + 2 * self.num_landmarks
        stop = start + 2 * (self.num_agents - 1)
        if corrupted.shape[-1] < stop:
            raise ValueError("observation is too short for OOD augmentation")
        block = corrupted[..., start:stop].reshape(
            -1, self.num_agents - 1, 2)
        if self.ood_noise_std > 0:
            block = block + torch.randn_like(block) * self.ood_noise_std
        if self.ood_mask_prob > 0:
            drop = torch.rand(
                block.shape[0], block.shape[1], 1,
                dtype=block.dtype, device=block.device) < self.ood_mask_prob
            block = torch.where(drop, torch.zeros_like(block), block)
        if self.ood_delay_prob > 0:
            sequence_length = self.data_chunk_length
            if corrupted.shape[0] % sequence_length != 0:
                raise ValueError("OOD delay augmentation requires full RNN chunks")
            sequence_count = corrupted.shape[0] // sequence_length
            block_sequence = block.reshape(
                sequence_length, sequence_count, self.num_agents - 1, 2)
            delayed = torch.cat(
                (block_sequence[:1], block_sequence[:-1]), dim=0)
            sequence_masks = check(masks).to(**self.tpdv).reshape(
                sequence_length, sequence_count, -1)
            can_delay = torch.ones(
                sequence_length, sequence_count, 1, 1,
                dtype=torch.bool, device=block.device)
            can_delay[0] = False
            can_delay[1:] &= sequence_masks[1:, :, :1].unsqueeze(-1) > 0
            choose_delay = torch.rand(
                sequence_length, sequence_count, 1, 1,
                device=block.device) < self.ood_delay_prob
            block_sequence = torch.where(
                choose_delay & can_delay, delayed, block_sequence)
            block = block_sequence.reshape(-1, self.num_agents - 1, 2)
        corrupted[..., start:stop] = block.reshape(-1, stop - start)
        return corrupted

    @staticmethod
    def _correlation(x, y):
        x = x - x.mean()
        y = y - y.mean()
        denominator = torch.sqrt((x.square().sum()) * (y.square().sum()))
        if denominator <= 1e-8:
            return torch.zeros((), dtype=x.dtype, device=x.device)
        return (x * y).sum() / denominator

    def ppo_update(self, sample, update_actor=True):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, \
            actions_batch, value_preds_batch, return_batch, masks_batch, \
            active_masks_batch, old_action_log_probs_batch, adv_targ, \
            available_actions_batch, teammate_actions_batch, \
            teammate_active_masks_batch, teammate_bootstrap_masks_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        teammate_actions_batch = check(teammate_actions_batch).long().to(self.device)
        teammate_active_masks_batch = check(teammate_active_masks_batch).to(**self.tpdv)
        teammate_bootstrap_masks_batch = check(
            teammate_bootstrap_masks_batch).to(**self.tpdv).squeeze(-1)

        values, action_log_probs, dist_entropy, teammate_logits, \
            teammate_uncertainty = \
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
        batch_size, num_heads, num_teammates, action_dim = \
            teammate_logits.shape
        expanded_targets = targets.unsqueeze(1).expand(-1, num_heads, -1)
        per_target_loss = F.cross_entropy(
            teammate_logits.reshape(-1, action_dim),
            expanded_targets.reshape(-1), reduction="none").reshape(
                batch_size, num_heads, num_teammates)
        bootstrap_masks = teammate_bootstrap_masks_batch
        expected_bootstrap_shape = (batch_size, num_heads, num_teammates)
        if tuple(bootstrap_masks.shape) != expected_bootstrap_shape:
            raise ValueError(
                "bootstrap mask shape {} does not match {}".format(
                    tuple(bootstrap_masks.shape), expected_bootstrap_shape))
        bootstrap_masks = bootstrap_masks * target_masks.unsqueeze(1)
        supervised_count = bootstrap_masks.sum().clamp_min(1.0)
        team_prediction_loss = (
            per_target_loss * bootstrap_masks).sum() / supervised_count
        valid_count = target_masks.sum().clamp_min(1.0)

        detector_logits = teammate_logits
        detector_mean_probs = F.softmax(detector_logits, dim=-1).mean(dim=1)
        detector_uncertainty = teammate_uncertainty
        detector_prediction_loss = torch.zeros((), device=self.device)
        if self.use_separate_detector:
            detector_logits, detector_mean_probs, detector_uncertainty, _ = \
                self.policy.actor.detector_outputs(
                    obs_batch, rnn_states_batch, masks_batch)
            if tuple(detector_logits.shape) != tuple(teammate_logits.shape):
                raise ValueError("detector and policy ensemble shapes must match")
            detector_per_target_loss = F.cross_entropy(
                detector_logits.reshape(-1, action_dim),
                expanded_targets.reshape(-1), reduction="none").reshape(
                    batch_size, num_heads, num_teammates)
            detector_prediction_loss = (
                detector_per_target_loss * bootstrap_masks
            ).sum() / supervised_count

        ood_rank_loss = torch.zeros((), device=self.device)
        ood_harmful_fraction = torch.zeros((), device=self.device)
        ood_clean_uncertainty = torch.zeros((), device=self.device)
        ood_corrupt_uncertainty = torch.zeros((), device=self.device)
        ood_corrupt_accuracy = torch.zeros((), device=self.device)
        uncertainty_calibration_loss = torch.zeros((), device=self.device)
        uncertainty_correlation_loss = torch.zeros((), device=self.device)
        clean_uncertainty_risk_correlation = torch.zeros((), device=self.device)
        corrupt_uncertainty_risk_correlation = torch.zeros((), device=self.device)
        if self.ood_rank_coef > 0 and num_heads > 1:
            corrupted_obs = self._corrupt_teammate_positions(
                obs_batch, masks_batch)
            _, corrupt_mean_probs, corrupt_uncertainty, _ = \
                self.policy.actor.detector_outputs(
                    corrupted_obs, rnn_states_batch, masks_batch)
            clean_mean_probs = detector_mean_probs
            clean_true_probs = clean_mean_probs.gather(
                -1, targets.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
            corrupt_true_probs = corrupt_mean_probs.gather(
                -1, targets.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
            clean_nll = -clean_true_probs.log()
            corrupt_nll = -corrupt_true_probs.log()
            harmful = (
                (corrupt_nll.detach() > clean_nll.detach() + 0.05).float()
                * target_masks)
            harmful_count = harmful.sum()
            if harmful_count > 0:
                rank_violation = F.relu(
                    self.ood_rank_margin
                    - (corrupt_uncertainty - detector_uncertainty))
                ood_rank_loss = (
                    rank_violation * harmful).sum() / harmful_count
                ood_clean_uncertainty = (
                    detector_uncertainty * harmful).sum() / harmful_count
                ood_corrupt_uncertainty = (
                    corrupt_uncertainty * harmful).sum() / harmful_count
            ood_harmful_fraction = harmful.sum() / valid_count
            corrupt_predictions = corrupt_mean_probs.argmax(dim=-1)
            ood_corrupt_accuracy = (
                (corrupt_predictions == targets).float() * target_masks
            ).sum() / valid_count
            if self.uncertainty_cal_coef > 0:
                clean_risk_target = self.uncertainty_target_scale * (
                    1.0 - clean_true_probs.detach())
                corrupt_risk_target = self.uncertainty_target_scale * (
                    1.0 - corrupt_true_probs.detach())
                clean_calibration = (
                    (detector_uncertainty - clean_risk_target).square()
                    * target_masks).sum() / valid_count
                corrupt_calibration = (
                    (corrupt_uncertainty - corrupt_risk_target).square()
                    * target_masks).sum() / valid_count
                uncertainty_calibration_loss = 0.5 * (
                    clean_calibration + corrupt_calibration)
            if self.uncertainty_corr_coef > 0:
                valid = target_masks.bool()
                clean_risk = (1.0 - clean_true_probs.detach())[valid]
                corrupt_risk = (1.0 - corrupt_true_probs.detach())[valid]
                clean_uncertainty_risk_correlation = self._correlation(
                    detector_uncertainty[valid], clean_risk)
                corrupt_uncertainty_risk_correlation = self._correlation(
                    corrupt_uncertainty[valid], corrupt_risk)
                uncertainty_correlation_loss = 1.0 - 0.5 * (
                    clean_uncertainty_risk_correlation
                    + corrupt_uncertainty_risk_correlation)

        self.policy.actor_optimizer.zero_grad()
        if self.policy.detector_optimizer is not None:
            self.policy.detector_optimizer.zero_grad()
        policy_actor_loss = (
            policy_action_loss - dist_entropy * self.entropy_coef
            + self.aux_coef * team_prediction_loss)
        uncertainty_objective = (
            self.ood_rank_coef * ood_rank_loss
            + self.uncertainty_cal_coef * uncertainty_calibration_loss
            + self.uncertainty_corr_coef * uncertainty_correlation_loss)
        detector_loss = (
            self.detector_aux_coef * detector_prediction_loss
            + uncertainty_objective)
        actor_loss = policy_actor_loss
        if not self.use_separate_detector:
            actor_loss = actor_loss + uncertainty_objective
        if update_actor:
            if not self.detector_only:
                actor_loss.backward()
            if self.policy.detector_optimizer is not None:
                detector_loss.backward()
        actor_grad_norm = torch.zeros((), device=self.device)
        if not self.detector_only:
            if self._use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.actor_parameters, self.max_grad_norm)
            else:
                actor_grad_norm = get_gard_norm(self.policy.actor_parameters)
        detector_grad_norm = torch.zeros((), device=self.device)
        if self.policy.detector_parameters:
            if self._use_max_grad_norm:
                detector_grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.detector_parameters, self.max_grad_norm)
            else:
                detector_grad_norm = get_gard_norm(
                    self.policy.detector_parameters)
        if not self.detector_only:
            self.policy.actor_optimizer.step()
        if self.policy.detector_optimizer is not None:
            self.policy.detector_optimizer.step()

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        critic_grad_norm = torch.zeros((), device=self.device)
        if not self.detector_only:
            (value_loss * self.value_loss_coef).backward()
            if self._use_max_grad_norm:
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), self.max_grad_norm)
            else:
                critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
            self.policy.critic_optimizer.step()

        with torch.no_grad():
            head_probs = F.softmax(detector_logits, dim=-1)
            mean_probs = head_probs.mean(dim=1)
            eps = torch.finfo(head_probs.dtype).eps
            predictive_entropy = -(
                mean_probs * mean_probs.clamp_min(eps).log()).sum(dim=-1)
            per_head_entropy = -(
                head_probs * head_probs.clamp_min(eps).log()).sum(dim=-1)
            normalized_predictive_entropy = predictive_entropy / np.log(action_dim)
            normalized_head_entropy = per_head_entropy / np.log(action_dim)
            mean_confidence = mean_probs.max(dim=-1).values
            predictions = mean_probs.argmax(dim=-1)
            accuracy = ((predictions == targets).float() * target_masks).sum() / valid_count
            valid_targets = targets[target_masks.bool()]
            if valid_targets.numel():
                counts = torch.bincount(
                    valid_targets, minlength=action_dim)
                majority_accuracy = counts.max().float() / valid_targets.numel()
            else:
                majority_accuracy = torch.zeros((), device=self.device)
            recalls = []
            for action_id in range(action_dim):
                class_mask = (targets == action_id).float() * target_masks
                denom = class_mask.sum()
                recall = (((predictions == action_id).float() * class_mask).sum()
                          / denom.clamp_min(1.0))
                recalls.append(recall)

            head_predictions = detector_logits.argmax(dim=-1)
            per_head_accuracies = []
            for head in range(num_heads):
                head_accuracy = (
                    (head_predictions[:, head] == targets).float()
                    * target_masks).sum() / valid_count
                per_head_accuracies.append(head_accuracy)

            action_votes = F.one_hot(
                head_predictions, num_classes=action_dim).sum(dim=1)
            majority_vote_fraction = action_votes.max(dim=-1).values.float() / num_heads
            head_action_disagreement = (
                (1.0 - majority_vote_fraction) * target_masks
            ).sum() / valid_count
            mean_uncertainty = (
                detector_uncertainty * target_masks).sum() / valid_count
            valid_uncertainty = detector_uncertainty[target_masks.bool()]
            uncertainty_std = (valid_uncertainty.std(unbiased=False)
                               if valid_uncertainty.numel()
                               else torch.zeros((), device=self.device))
            prediction_errors = (predictions != targets).float()[target_masks.bool()]
            uncertainty_error_correlation = (
                self._correlation(valid_uncertainty, prediction_errors)
                if valid_uncertainty.numel() > 1
                else torch.zeros((), device=self.device))
            mean_predictive_entropy = (
                normalized_predictive_entropy * target_masks
            ).sum() / valid_count
            mean_head_entropy = (
                normalized_head_entropy * target_masks.unsqueeze(1)
            ).sum() / (valid_count * num_heads)
            mean_prediction_confidence = (
                mean_confidence * target_masks).sum() / valid_count

            pairwise_js_values = []
            for left in range(num_heads):
                for right in range(left + 1, num_heads):
                    pair_mean = 0.5 * (
                        head_probs[:, left] + head_probs[:, right])
                    pair_entropy = -(
                        pair_mean * pair_mean.clamp_min(eps).log()).sum(-1)
                    component_entropy = 0.5 * (
                        per_head_entropy[:, left]
                        + per_head_entropy[:, right])
                    pairwise_js_values.append(
                        (pair_entropy - component_entropy) / np.log(action_dim))
            if pairwise_js_values:
                pairwise_js = torch.stack(pairwise_js_values).mean(dim=0)
                mean_pairwise_js = (
                    pairwise_js * target_masks).sum() / valid_count
            else:
                mean_pairwise_js = torch.zeros((), device=self.device)

        metrics = {
            "value_loss": value_loss,
            "critic_grad_norm": critic_grad_norm,
            "policy_loss": policy_action_loss,
            "dist_entropy": dist_entropy,
            "actor_grad_norm": actor_grad_norm,
            "detector_grad_norm": detector_grad_norm,
            "ratio": imp_weights.mean(),
            "team_prediction_loss": team_prediction_loss,
            "detector_prediction_loss": detector_prediction_loss.detach(),
            "detector_loss": detector_loss.detach(),
            "team_action_accuracy": accuracy,
            "majority_action_accuracy": majority_accuracy,
            "mean_uncertainty": mean_uncertainty,
            "uncertainty_std": uncertainty_std,
            "head_action_disagreement": head_action_disagreement,
            "uncertainty_error_correlation": uncertainty_error_correlation,
            "mean_predictive_entropy": mean_predictive_entropy,
            "mean_head_entropy": mean_head_entropy,
            "mean_prediction_confidence": mean_prediction_confidence,
            "mean_pairwise_js": mean_pairwise_js,
            "ood_rank_loss": ood_rank_loss.detach(),
            "ood_harmful_fraction": ood_harmful_fraction.detach(),
            "ood_clean_uncertainty": ood_clean_uncertainty.detach(),
            "ood_corrupt_uncertainty": ood_corrupt_uncertainty.detach(),
            "ood_corrupt_accuracy": ood_corrupt_accuracy.detach(),
            "uncertainty_calibration_loss":
                uncertainty_calibration_loss.detach(),
            "uncertainty_correlation_loss":
                uncertainty_correlation_loss.detach(),
            "clean_uncertainty_risk_correlation":
                clean_uncertainty_risk_correlation.detach(),
            "corrupt_uncertainty_risk_correlation":
                corrupt_uncertainty_risk_correlation.detach(),
            "bootstrap_coverage": bootstrap_masks.sum() /
                                  target_masks.unsqueeze(1).expand_as(
                                      bootstrap_masks).sum().clamp_min(1.0),
        }
        metrics.update({f"team_action_recall_{i}": value
                        for i, value in enumerate(recalls)})
        metrics.update({f"team_head_{i}_accuracy": value
                        for i, value in enumerate(per_head_accuracies)})
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
