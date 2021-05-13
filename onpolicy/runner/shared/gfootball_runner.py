from onpolicy.runner.shared.base_runner import Runner

import time
import numpy as np
import wandb
import torch


def _t2n(x):
    return x.detach().cpu().numpy()


class GfootballRunner(Runner):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps
                       ) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(
                    step)


                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)


                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.algorithm_name,
                            self.experiment_name, episode, episodes,
                            total_num_steps, self.num_env_steps,
                            int(total_num_steps / (end - start))))

                train_infos["average_episode_rewards"] = np.mean(
                    self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(
                    train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        self.envs.reset()

    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.chooseafter_update()
        return train_infos

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.squeeze(np.split(_t2n(action), self.n_rollout_threads),
                             axis=-1)
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(
            rnn_states.shape[1:],
            dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            rnn_states_critic.shape[1:],
            dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
                        dtype=np.float32)
        masks[dones == True] = np.zeros(masks.shape[1:], dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents,
                                                            axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           np.expand_dims(actions, axis=-1), action_log_probs,
                           values, np.expand_dims(rewards, axis=-1), masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1),
                             dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True)
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[
                    0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(
                        self.eval_envs.action_space[0].high[i] +
                        1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate(
                            (eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[
                    0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(
                    np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones
                  == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32)
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1),
                dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(
            np.array(eval_episode_rewards), axis=0)
        print("eval average episode rewards of agent: " +
              str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)


if __name__ == "__main__":
    from onpolicy.config import get_config
    from onpolicy.envs.gfootball.gfootball_env import GoogleFootballEnv
    from onpolicy.envs.env_wrappers import SubprocVecEnv

    args=get_config().parse_known_args()[0]
    config = {
        'all_args':
        args,
        'envs':
        SubprocVecEnv([
            lambda: GoogleFootballEnv(num_of_left_agents=3,
                                      env_name='test_example_multiagent',
                                      representation="simple115v2")
            for i in range(args.n_rollout_threads)
        ]),
        'eval_envs':
        None,
        'device':
        None,
        'num_agents':
        3,
        'run_dir':
        '.'
    }
    config['all_args'].use_wandb = False
    runner = GfootballRunner(config)
    runner.run()
