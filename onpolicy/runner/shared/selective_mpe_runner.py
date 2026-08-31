import numpy as np

from onpolicy.envs.mpe.perturbations import TeammatePositionCorruptor
from onpolicy.runner.shared.mpe_runner import MPERunner, _t2n


class SelectiveMPERunner(MPERunner):
    """Rollout runner for frozen B0/B2 branches and a learned selector."""

    def __init__(self, config):
        super().__init__(config)
        args = self.all_args
        if args.scenario_name != "simple_spread":
            raise ValueError(
                "B4 corruption training currently supports simple_spread")
        if not 0.0 <= args.cstm_selector_clean_probability <= 1.0:
            raise ValueError("selector clean probability must be in [0, 1]")
        if not (args.cstm_selector_noise_levels
                and args.cstm_selector_mask_levels
                and args.cstm_selector_delay_levels):
            raise ValueError("selector corruption level lists cannot be empty")
        rng = np.random.RandomState(args.seed + 15401)
        corruption_types = ("noise", "mask", "delay")
        self.training_conditions = []
        self.corruptors = []
        for rank in range(self.n_rollout_threads):
            if rng.random_sample() < args.cstm_selector_clean_probability:
                corruption_type, level = "clean", 0.0
            else:
                corruption_type = corruption_types[rng.randint(3)]
                if corruption_type == "noise":
                    level = rng.choice(args.cstm_selector_noise_levels)
                elif corruption_type == "mask":
                    level = rng.choice(args.cstm_selector_mask_levels)
                else:
                    level = rng.choice(args.cstm_selector_delay_levels)
            self.training_conditions.append((corruption_type, float(level)))
            self.corruptors.append(TeammatePositionCorruptor(
                corruption_type, level, self.num_agents, args.num_landmarks,
                seed=args.seed * 100003 + rank * 9176))

    def warmup(self):
        super().warmup()
        self.buffer.b0_rnn_states[0].fill(0.0)

    def _corrupt_observations(self, observations):
        return np.stack([
            corruptor.transform(observation)
            for corruptor, observation in zip(self.corruptors, observations)
        ]).astype(np.float32)

    @staticmethod
    def _actions_to_env(actions, action_space):
        if action_space.__class__.__name__ != "Discrete":
            raise NotImplementedError(
                "selective_mappo currently supports Discrete actions only")
        return np.squeeze(np.eye(action_space.n)[actions], 2)

    def collect(self, step):
        self.trainer.prep_rollout()
        actor_obs = self._corrupt_observations(self.buffer.obs[step])
        # PPO must later evaluate exactly the observations used at rollout.
        self.buffer.obs[step] = actor_obs.copy()
        outputs = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(actor_obs),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.b0_rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]))
        (value, action, action_log_prob, b2_rnn_states, b0_rnn_states,
         rnn_states_critic, diagnostics) = outputs
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(
            _t2n(action_log_prob), self.n_rollout_threads))
        b2_rnn_states = np.array(np.split(
            _t2n(b2_rnn_states), self.n_rollout_threads))
        b0_rnn_states = np.array(np.split(
            _t2n(b0_rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(
            _t2n(rnn_states_critic), self.n_rollout_threads))
        actions_env = self._actions_to_env(
            actions, self.envs.action_space[0])
        self.last_rollout_diagnostics = {
            key: float(value.detach().mean())
            for key, value in diagnostics.items()}
        return (values, actions, action_log_probs, b2_rnn_states,
                b0_rnn_states, rnn_states_critic, actions_env)

    def insert(self, data):
        (obs, rewards, dones, infos, values, actions, action_log_probs,
         b2_rnn_states, b0_rnn_states, rnn_states_critic) = data
        b2_rnn_states[dones] = 0.0
        b0_rnn_states[dones] = 0.0
        rnn_states_critic[dones] = 0.0
        for rank, done_row in enumerate(dones):
            if np.any(done_row):
                self.corruptors[rank].reset()
        masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones] = 0.0
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(
                self.num_agents, axis=1)
        else:
            share_obs = obs
        self.buffer.insert(
            share_obs, obs, b2_rnn_states, b0_rnn_states,
            rnn_states_critic, actions, action_log_probs, values, rewards,
            masks)

    def run(self):
        # MPERunner.run expects the original collect tuple. Keep its loop here
        # explicit so both branch states reach insert without ambiguity.
        import time
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length \
            // self.n_rollout_threads
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            for step in range(self.episode_length):
                collected = self.collect(step)
                (values, actions, action_log_probs, b2_states, b0_states,
                 critic_states, actions_env) = collected
                obs, rewards, dones, infos = self.envs.step(actions_env)
                self.insert((
                    obs, rewards, dones, infos, values, actions,
                    action_log_probs, b2_states, b0_states, critic_states))
            self.compute()
            train_infos = self.train()
            total_num_steps = ((episode + 1) * self.episode_length
                               * self.n_rollout_threads)
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()
            if episode % self.log_interval == 0 or episode == episodes - 1:
                elapsed = max(1e-6, time.time() - start)
                print("\n Scenario {} Algo {} Exp {} updates {}/{}, "
                      "timesteps {}/{}, FPS {}.\n".format(
                          self.all_args.scenario_name, self.algorithm_name,
                          self.experiment_name, episode, episodes,
                          total_num_steps, self.num_env_steps,
                          int(total_num_steps / elapsed)))
                train_infos["average_episode_rewards"] = (
                    np.mean(self.buffer.rewards) * self.episode_length)
                for key, value in getattr(
                        self, "last_rollout_diagnostics", {}).items():
                    train_infos["rollout_" + key] = value
                print("average episode rewards is {}".format(
                    train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
