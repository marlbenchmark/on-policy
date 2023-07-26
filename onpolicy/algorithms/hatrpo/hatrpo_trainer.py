import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.algorithms.utils.popart_hatrpo import PopArt
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

class HATRPO():
    """
    Trainer class for MATRPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (HATRPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self.kl_threshold = args.kl_threshold
        self.ls_step = args.ls_step
        self.accept_ratio = args.accept_ratio

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None


    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart or self._use_valuenorm:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def flat_grad(self, grads):
        grad_flatten = []
        for grad in grads:
            if grad is None:
                continue
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten)
        return grad_flatten

    def flat_hessian(self, hessians):
        hessians_flatten = []
        for hessian in hessians:
            if hessian is None:
                continue
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten).data
        return hessians_flatten

    def flat_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten

    def update_model(self, model, new_params):
        index = 0
        for params in model.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length

    def kl_approx(self, q, p):
        r = torch.exp(p - q)
        kl = r - 1 - p + q
        return kl

    def kl_divergence(self, obs, rnn_states, action, masks, available_actions, active_masks, new_actor, old_actor):
        _, _, mu, std, probs = new_actor.evaluate_actions(obs, rnn_states, action, masks, available_actions, active_masks)
        _, _, mu_old, std_old, probs_old = old_actor.evaluate_actions(obs, rnn_states, action, masks, available_actions, active_masks)
        if mu.grad_fn==None:
            probs_old=probs_old.detach()
            kl= self.kl_approx(probs_old,probs)
        else:
            logstd = torch.log(std)
            mu_old = mu_old.detach()
            std_old = std_old.detach()
            logstd_old = torch.log(std_old)
            # kl divergence between old policy and new policy : D( pi_old || pi_new )
            # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
            # be careful of calculating KL-divergence. It is not symmetric metric
            kl =  logstd - logstd_old  + (std_old.pow(2) + (mu_old - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
        
        if len(kl.shape)>1:
            kl=kl.sum(1, keepdim=True)
        return kl

    # from openai baseline code
    # https://github.com/openai/baselines/blob/master/baselines/common/cg.py
    def conjugate_gradient(self, actor, obs, rnn_states, action, masks, available_actions, active_masks, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size()).to(device=self.device)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = self.fisher_vector_product(actor, obs, rnn_states, action, masks, available_actions, active_masks, p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def fisher_vector_product(self, actor, obs, rnn_states, action, masks, available_actions, active_masks, p):
        p.detach()
        kl = self.kl_divergence(obs, rnn_states, action, masks, available_actions, active_masks, new_actor=actor, old_actor=actor)
        kl = kl.mean()
        kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True, allow_unused=True)
        kl_grad = self.flat_grad(kl_grad)  # check kl_grad == 0
        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters(), allow_unused=True)
        kl_hessian_p = self.flat_hessian(kl_hessian_p)
        return kl_hessian_p + 0.1 * p

    def trpo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, factor_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy, action_mu, action_std, _ = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        # actor update
        ratio = torch.prod(torch.exp(action_log_probs - old_action_log_probs_batch),dim=-1,keepdim=True)
        if self._use_policy_active_masks:
            loss = (torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True) *
                           active_masks_batch).sum() / active_masks_batch.sum()
        else:
            loss = torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True).mean()
        
        loss_grad = torch.autograd.grad(loss, self.policy.actor.parameters(), allow_unused=True)
        loss_grad = self.flat_grad(loss_grad)

        step_dir = self.conjugate_gradient(self.policy.actor, 
                                      obs_batch, 
                                      rnn_states_batch, 
                                      actions_batch, 
                                      masks_batch, 
                                      available_actions_batch, 
                                      active_masks_batch, 
                                      loss_grad.data, 
                                      nsteps=10)
        
        loss = loss.data.cpu().numpy()
        params = self.flat_params(self.policy.actor)

        fvp = self.fisher_vector_product(self.policy.actor,
                                    obs_batch, 
                                    rnn_states_batch, 
                                    actions_batch, 
                                    masks_batch, 
                                    available_actions_batch, 
                                    active_masks_batch, 
                                    step_dir)
        shs = 0.5 * (step_dir * fvp).sum(0, keepdim=True)
        step_size = 1 / torch.sqrt(shs / self.kl_threshold)[0]
        full_step = step_size * step_dir

        old_actor = R_Actor(self.policy.args, 
                            self.policy.obs_space,  
                            self.policy.act_space, 
                            self.device)
        self.update_model(old_actor, params)
        expected_improve = (loss_grad * full_step).sum(0, keepdim=True)
        expected_improve = expected_improve.data.cpu().numpy()
        
        # Backtracking line search
        flag = False
        fraction = 1
        for i in range(self.ls_step):
            new_params = params + fraction * full_step
            self.update_model(self.policy.actor, new_params)
            values, action_log_probs, dist_entropy, action_mu, action_std, _ = self.policy.evaluate_actions(share_obs_batch,
                                                                                obs_batch, 
                                                                                rnn_states_batch, 
                                                                                rnn_states_critic_batch, 
                                                                                actions_batch, 
                                                                                masks_batch, 
                                                                                available_actions_batch,
                                                                                active_masks_batch)

            ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
            if self._use_policy_active_masks:
                new_loss = (torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True) *
                            active_masks_batch).sum() / active_masks_batch.sum()
            else:
                new_loss = torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True).mean()

            new_loss = new_loss.data.cpu().numpy()
            loss_improve = new_loss - loss
            
            kl = self.kl_divergence(obs_batch, 
                               rnn_states_batch, 
                               actions_batch, 
                               masks_batch, 
                               available_actions_batch, 
                               active_masks_batch,
                               new_actor=self.policy.actor,
                               old_actor=old_actor)
            kl = kl.mean()

            if kl < self.kl_threshold and (loss_improve / expected_improve) > self.accept_ratio and loss_improve.item()>0:
                flag = True
                break
            expected_improve *= 0.5
            fraction *= 0.5

        if not flag:
            params = self.flat_params(old_actor)
            self.update_model(self.policy.actor, params)
            print('policy update does not impove the surrogate')

        return value_loss, critic_grad_norm, kl, loss_improve, expected_improve, dist_entropy, ratio

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['kl'] = 0
        train_info['dist_entropy'] = 0
        train_info['loss_improve'] = 0
        train_info['expected_improve'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0


        if self._use_recurrent_policy:
            data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
        elif self._use_naive_recurrent:
            data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
        else:
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

        for sample in data_generator:

            value_loss, critic_grad_norm, kl, loss_improve, expected_improve, dist_entropy, imp_weights \
                = self.trpo_update(sample, update_actor)

            train_info['value_loss'] += value_loss.item()
            train_info['kl'] += kl
            train_info['loss_improve'] += loss_improve.item()
            train_info['expected_improve'] += expected_improve
            train_info['dist_entropy'] += dist_entropy.item()
            train_info['critic_grad_norm'] += critic_grad_norm
            train_info['ratio'] += imp_weights.mean()

        num_updates = self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
