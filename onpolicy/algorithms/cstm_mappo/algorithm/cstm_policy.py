import torch

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic
from onpolicy.algorithms.cstm_mappo.algorithm.dual_policy_actor import B1Actor


class CSTMPolicy(R_MAPPOPolicy):
    def __init__(self, args, obs_space, cent_obs_space, act_space,
                 device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = B1Actor(args, obs_space, act_space, args.num_agents, device)
        self.critic = R_Critic(args, cent_obs_space, device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, eps=self.opti_eps,
            weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps,
            weight_decay=self.weight_decay)

    def evaluate_actions_with_teammates(
            self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action,
            masks, available_actions=None, active_masks=None):
        action_log_probs, dist_entropy, teammate_logits, teammate_uncertainty = \
            self.actor.evaluate_actions_with_teammates(
                obs, rnn_states_actor, action, masks, available_actions,
                active_masks)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return (values, action_log_probs, dist_entropy, teammate_logits,
                teammate_uncertainty)
