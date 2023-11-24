import gym, tqdm
import time
import numpy as np
from onpolicy.envs.overcooked.overcooked_ai_py.utils import mean_and_std_err, append_dictionaries
from onpolicy.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from onpolicy.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, EVENT_TYPES
from onpolicy.envs.overcooked.overcooked_ai_py.mdp.overcooked_trajectory import TIMESTEP_TRAJ_KEYS, EPISODE_TRAJ_KEYS, DEFAULT_TRAJ_KEYS
from onpolicy.envs.overcooked.overcooked_ai_py.planning.planners import MediumLevelActionManager, MotionPlanner, NO_COUNTERS_PARAMS
from onpolicy.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import imageio
import os
import pickle
from collections import defaultdict
from onpolicy.envs.overcooked.script_agent import SCRIPT_AGENTS

DEFAULT_ENV_PARAMS = {
    "horizon": 400
}

MAX_HORIZON = 1e10


class OvercookedEnv(object):
    """
    An environment wrapper for the OvercookedGridworld Markov Decision Process.

    The environment keeps track of the current state of the agent, updates
    it as the agent takes actions, and provides rewards to the agent.

    E.g. of how to instantiate OvercookedEnv:
    > mdp = OvercookedGridworld(...)
    > env = OvercookedEnv.from_mdp(mdp, horizon=400)
    """

    #########################
    # INSTANTIATION METHODS #
    #########################

    def __init__(self, mdp_generator_fn, start_state_fn=None, horizon=MAX_HORIZON, mlam_params=NO_COUNTERS_PARAMS,
                 info_level=0, num_mdp=1, initial_info={}):
        """
        mdp_generator_fn (callable):    A no-argument function that returns a OvercookedGridworld instance
        start_state_fn (callable):      Function that returns start state for the MDP, called at each environment reset
        horizon (int):                  Number of steps before the environment returns done=True
        mlam_params (dict):             params for MediumLevelActionManager
        info_level (int):               Change amount of logging
        num_mdp (int):                  the number of mdp if we are using a list of mdps
        initial_info (dict):            the initial outside information feed into the generator function

        TODO: Potentially make changes based on this discussion
        https://github.com/HumanCompatibleAI/overcooked_ai/pull/22#discussion_r416786847
        """
        assert callable(mdp_generator_fn), "OvercookedEnv takes in a OvercookedGridworld generator function. " \
                                           "If trying to instantiate directly from a OvercookedGridworld " \
                                           "instance, use the OvercookedEnv.from_mdp method"
        self.num_mdp = num_mdp
        self.variable_mdp = num_mdp == 1
        self.mdp_generator_fn = mdp_generator_fn
        self.horizon = horizon
        self._mlam = None
        self._mp = None
        self.mlam_params = mlam_params
        self.start_state_fn = start_state_fn
        self.info_level = info_level
        self.reset(outside_info=initial_info)
        if self.horizon >= MAX_HORIZON and self.info_level > 0:
            print("Environment has (near-)infinite horizon and no terminal states. \
                Reduce info level of OvercookedEnv to not see this message.")

    @property
    def mlam(self):
        if self._mlam is None:
            if self.info_level > 0:
                print("Computing MediumLevelActionManager")
            self._mlam = MediumLevelActionManager.from_pickle_or_compute(self.mdp, self.mlam_params,
                                                                         force_compute=False)
        return self._mlam

    @property
    def mp(self):
        if self._mp is None:
            if self._mlam is not None:
                self._mp = self.mlam.motion_planner
            else:
                if self.info_level > 0:
                    print("Computing MotionPlanner")
                self._mp = MotionPlanner.from_pickle_or_compute(self.mdp, self.mlam_params["counter_goals"],
                                                                force_compute=False)
        return self._mp

    @staticmethod
    def from_mdp(mdp, start_state_fn=None, horizon=MAX_HORIZON, mlam_params=NO_COUNTERS_PARAMS, info_level=0):
        """
        Create an OvercookedEnv directly from a OvercookedGridworld mdp
        rather than a mdp generating function.
        """
        assert isinstance(mdp, OvercookedGridworld)
        mdp_generator_fn = lambda _ignored: mdp
        return OvercookedEnv(
            mdp_generator_fn=mdp_generator_fn,
            start_state_fn=start_state_fn,
            horizon=horizon,
            mlam_params=mlam_params,
            info_level=info_level,
            num_mdp=1
        )

    #####################
    # BASIC CLASS UTILS #
    #####################

    @property
    def env_params(self):
        """
        Env params should be though of as all of the params of an env WITHOUT the mdp.
        Alone, env_params is not sufficent to recreate a copy of the Env instance, but it is
        together with mdp_params (which is sufficient to build a copy of the Mdp instance).
        """
        return {
            "start_state_fn": self.start_state_fn,
            "horizon": self.horizon,
            "info_level": self.info_level,
            "_variable_mdp": self.variable_mdp
        }

    def copy(self):
        # TODO: Add testing for checking that these util methods are up to date?
        return OvercookedEnv(
            mdp_generator_fn=self.mdp_generator_fn,
            start_state_fn=self.start_state_fn,
            horizon=self.horizon,
            info_level=self.info_level,
            num_mdp=self.num_mdp
        )

    #############################
    # ENV VISUALIZATION METHODS #
    #############################

    def __repr__(self):
        """
        Standard way to view the state of an environment programatically
        is just to print the Env object
        """
        return self.mdp.state_string(self.state)

    def display_states(self, *states):
        old_state = self.state
        for s in states:
            self.state = s
            print(self)
        self.state = old_state

    def print_state_transition(self, a_t, r_t, env_info, fname=None, display_phi=False):
        """
        Terminal graphics visualization of a state transition.
        """
        # TODO: turn this into a "formatting action probs" function and add action symbols too
        action_probs = [None if "action_probs" not in agent_info.keys() else list(agent_info["action_probs"]) for
                        agent_info in env_info["agent_infos"]]

        action_probs = [None if player_action_probs is None else [round(p, 2) for p in player_action_probs[0]] for
                        player_action_probs in action_probs]

        if display_phi:
            state_potential_str = "\nState potential = " + str(env_info["phi_s_prime"]) + "\t"
            potential_diff_str = "Δ potential = " + str(
                0.99 * env_info["phi_s_prime"] - env_info["phi_s"]) + "\n"  # Assuming gamma 0.99
        else:
            state_potential_str = ""
            potential_diff_str = ""

        output_string = "Timestep: {}\nJoint action taken: {} \t Reward: {} + shaping_factor * {}\nAction probs by index: {} {} {}\n{}\n".format(
            self.state.timestep,
            tuple(Action.ACTION_TO_CHAR[a] for a in a_t),
            r_t,
            env_info["shaped_r_by_agent"],
            action_probs,
            state_potential_str,
            potential_diff_str,
            self)

        if fname is None:
            print(output_string)
        else:
            f = open(fname, 'a')
            print(output_string, file=f)
            f.close()

    ###################
    # BASIC ENV LOGIC #
    ###################

    def step(self, joint_action, joint_agent_action_info=None, display_phi=False):
        """Performs a joint action, updating the environment state
        and providing a reward.

        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        assert not self.is_done()
        if joint_agent_action_info is None: joint_agent_action_info = [{}, {}]
        next_state, mdp_infos = self.mdp.get_state_transition(self.state, joint_action, display_phi, self.mp)

        # Update game_stats
        self._update_game_stats(mdp_infos)

        # Update state and done
        self.state = next_state
        done = self.is_done()
        env_info = self._prepare_info_dict(joint_agent_action_info, mdp_infos)

        if done: self._add_episode_info(env_info)

        timestep_sparse_reward = sum(mdp_infos["sparse_reward_by_agent"])
        return next_state, timestep_sparse_reward, done, env_info

    def lossless_state_encoding_mdp(self, state):
        """
        Wrapper of the mdp's lossless_encoding
        """
        orgin=self.mdp.lossless_state_encoding(state, self.horizon)
        orgin=list(orgin)
        for player in range(len(orgin)):
            orgin[player] = np.transpose(orgin[player], (2,0,1))
        orgin=tuple(orgin)
        return orgin

    def featurize_state_mdp(self, state, num_pots=2):
        """
        Wrapper of the mdp's featurize_state
        """
        return self.mdp.featurize_state(state, self.mlam, num_pots=num_pots)

    def reset(self, regen_mdp=True, outside_info={}):
        """
        Resets the environment. Does NOT reset the agent.
        Args:
            regen_mdp (bool): gives the option of not re-generating mdp on the reset,
                                which is particularly helpful with reproducing results on variable mdp
            outside_info (dict): the outside information that will be fed into the scheduling_fn (if used), which will
                                 in turn generate a new set of mdp_params that is used to regenerate mdp.
                                 Please note that, if you intend to use this arguments throughout the run,
                                 you need to have a "initial_info" dictionary with the same keys in the "env_params"
        """
        if regen_mdp:
            self.mdp = self.mdp_generator_fn(outside_info)
            self._mlam = None
            self._mp = None
        if self.start_state_fn is None:
            self.state = self.mdp.get_standard_start_state()
        elif type(self.start_state_fn) in [float, int]:
            p = np.random.uniform(0, 1)
            if p <= self.start_state_fn:
                self.state = self.mdp.get_random_start_state()
            else:
                self.state = self.mdp.get_standard_start_state()
        else:
            self.state = self.start_state_fn()
        self.state = self.state.deepcopy()

        events_dict = {k: [[] for _ in range(self.mdp.num_players)] for k in EVENT_TYPES}
        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": np.array([0] * self.mdp.num_players),
            "cumulative_shaped_rewards_by_agent": np.array([0] * self.mdp.num_players),
            "cumulative_category_rewards_by_agent": np.zeros((self.mdp.num_players, 28))
        }
        self.game_stats = {**events_dict, **rewards_dict}
        return self.state

    def is_done(self):
        """Whether the episode is over."""
        return self.state.timestep >= self.horizon or self.mdp.is_terminal(self.state)

    def potential(self, mlam, state=None, gamma=0.99):
        """
        Return the potential of the environment's current state, if no state is provided
        Otherwise return the potential of `state`
        args:
            mlam (MediumLevelActionManager): the mlam of self.mdp
            state (OvercookedState): the current state we are evaluating the potential on
            gamma (float): discount rate
        """
        state = state if state else self.state
        return self.mdp.potential_function(state, mp=mlam.motion_planner, gamma=gamma)

    def _prepare_info_dict(self, joint_agent_action_info, mdp_infos):
        """
        The normal timestep info dict will contain infos specifc to each agent's action taken,
        and reward shaping information.
        """
        # Get the agent action info, that could contain info about action probs, or other
        # custom user defined information
        env_info = {"agent_infos": [joint_agent_action_info[agent_idx] for agent_idx in range(self.mdp.num_players)]}
        # TODO: This can be further simplified by having all the mdp_infos copied over to the env_infos automatically
        env_info["sparse_r_by_agent"] = mdp_infos["sparse_reward_by_agent"]
        env_info["shaped_r_by_agent"] = mdp_infos["shaped_reward_by_agent"]
        # env_info["shaped_info_by_agent"] = mdp_infos["shaped_info_by_agent"]
        env_info["phi_s"] = mdp_infos["phi_s"] if "phi_s" in mdp_infos else None
        env_info["phi_s_prime"] = mdp_infos["phi_s_prime"] if "phi_s_prime" in mdp_infos else None
        return env_info

    def _add_episode_info(self, env_info):
        env_info["episode"] = {
            "ep_game_stats": self.game_stats,
            "ep_sparse_r": sum(self.game_stats["cumulative_sparse_rewards_by_agent"]),
            "ep_shaped_r": sum(self.game_stats["cumulative_shaped_rewards_by_agent"]),
            "ep_sparse_r_by_agent": self.game_stats["cumulative_sparse_rewards_by_agent"],
            "ep_shaped_r_by_agent": self.game_stats["cumulative_shaped_rewards_by_agent"],
            "ep_category_r_by_agent": self.game_stats["cumulative_category_rewards_by_agent"],
            "ep_length": self.state.timestep
        }
        return env_info

    def vectorize_shaped_info(self, shaped_info_by_agent):
        def vectorize(d: dict):
            return np.array([v for k, v in d.items()])

        shaped_info_by_agent = np.stack([vectorize(shaped_info) for shaped_info in shaped_info_by_agent])
        return shaped_info_by_agent

    def _update_game_stats(self, infos):
        """
        Update the game stats dict based on the events of the current step
        NOTE: the timer ticks after events are logged, so there can be events from time 0 to time self.horizon - 1
        """
        self.game_stats["cumulative_sparse_rewards_by_agent"] += np.array(infos["sparse_reward_by_agent"])
        self.game_stats["cumulative_shaped_rewards_by_agent"] += np.array(infos["shaped_reward_by_agent"])
        # self.game_stats["cumulative_category_rewards_by_agent"] += self.vectorize_shaped_info(infos["shaped_info_by_agent"])

        for event_type, bool_list_by_agent in infos["event_infos"].items():
            # For each event type, store the timestep if it occurred
            event_occurred_by_idx = [int(x) for x in bool_list_by_agent]
            for idx, event_by_agent in enumerate(event_occurred_by_idx):
                if event_by_agent:
                    self.game_stats[event_type][idx].append(self.state.timestep)

    ####################
    # TRAJECTORY LOGIC #
    ####################

    def execute_plan(self, start_state, joint_action_plan, display=False):
        """Executes action_plan (a list of joint actions) from a start
        state in the mdp and returns the resulting state."""
        self.state = start_state
        done = False
        if display: print("Starting state\n{}".format(self))
        for joint_action in joint_action_plan:
            self.step(joint_action)
            done = self.is_done()
            if display: print(self)
            if done: break
        successor_state = self.state
        self.reset(False)
        return successor_state, done

    def run_agents(self, agent_pair, include_final_state=False, display=False, dir=None, display_phi=False,
                   display_until=np.Inf):
        """
        Trajectory returned will a list of state-action pairs (s_t, joint_a_t, r_t, done_t, info_t).
        """
        assert self.state.timestep == 0, "Did not reset environment before running agents"
        trajectory = []
        done = False
        # default is to not print to file
        fname = None

        if dir != None:
            fname = dir + '/roll_out_' + str(time.time()) + '.txt'
            f = open(fname, 'w+')
            print(self, file=f)
            f.close()
        while not done:
            s_t = self.state

            # Getting actions and action infos (optional) for both agents
            joint_action_and_infos = agent_pair.joint_action(s_t)
            a_t, a_info_t = zip(*joint_action_and_infos)
            assert all(a in Action.ALL_ACTIONS for a in a_t)
            assert all(type(a_info) is dict for a_info in a_info_t)

            s_tp1, r_t, done, info = self.step(a_t, a_info_t, display_phi)
            trajectory.append((s_t, a_t, r_t, done, info))

            if display and self.state.timestep < display_until:
                self.print_state_transition(a_t, r_t, info, fname, display_phi)

        assert len(trajectory) == self.state.timestep, "{} vs {}".format(len(trajectory), self.state.timestep)

        # Add final state
        if include_final_state:
            trajectory.append((s_tp1, (None, None), 0, True, None))

        total_sparse = sum(self.game_stats["cumulative_sparse_rewards_by_agent"])
        total_shaped = sum(self.game_stats["cumulative_shaped_rewards_by_agent"])
        return np.array(trajectory, dtype=object), self.state.timestep, total_sparse, total_shaped

    def get_rollouts(self, agent_pair, num_games, display=False, dir=None, final_state=False, display_phi=False,
                     display_until=np.Inf, metadata_fn=None, metadata_info_fn=None, info=True):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed
        trajectories.

        Returning excessive information to be able to convert trajectories to any required format
        (baselines, stable_baselines, etc)

        metadata_fn returns some metadata information computed at the end of each trajectory based on
        some of the trajectory data.

        NOTE: this is the standard trajectories format used throughout the codebase
        """
        trajectories = {k: [] for k in DEFAULT_TRAJ_KEYS}
        metadata_fn = (lambda x: {}) if metadata_fn is None else metadata_fn
        metadata_info_fn = (lambda x: "") if metadata_info_fn is None else metadata_info_fn
        range_iterator = tqdm.trange(num_games, desc="", leave=True) if info else range(num_games)
        for i in range_iterator:
            agent_pair.set_mdp(self.mdp)

            rollout_info = self.run_agents(agent_pair, display=display, dir=dir, include_final_state=final_state,
                                           display_phi=display_phi, display_until=display_until)
            trajectory, time_taken, tot_rews_sparse, _tot_rews_shaped = rollout_info
            obs, actions, rews, dones, infos = trajectory.T[0], trajectory.T[1], trajectory.T[2], trajectory.T[3], \
            trajectory.T[4]
            trajectories["ep_states"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_infos"].append(infos)
            trajectories["ep_returns"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)
            trajectories["mdp_params"].append(self.mdp.mdp_params)
            trajectories["env_params"].append(self.env_params)
            trajectories["metadatas"].append(metadata_fn(rollout_info))

            # we do not need to regenerate MDP if we are trying to generate a series of rollouts using the same MDP
            # Basically, the FALSE here means that we are using the same layout and starting positions
            # (if regen_mdp == True, resetting will call mdp_gen_fn to generate another layout & starting position)
            self.reset(regen_mdp=False)
            agent_pair.reset()

            if info:
                mu, se = mean_and_std_err(trajectories["ep_returns"])
                description = "Avg rew: {:.2f} (std: {:.2f}, se: {:.2f}); avg len: {:.2f}; ".format(
                    mu, np.std(trajectories["ep_returns"]), se, np.mean(trajectories["ep_lengths"]))
                description += metadata_info_fn(trajectories["metadatas"])
                range_iterator.set_description(description)
                range_iterator.refresh()

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}

        # Merging all metadata dictionaries, assumes same keys throughout all
        trajectories["metadatas"] = append_dictionaries(trajectories["metadatas"])

        # TODO: should probably transfer check methods over to Env class
        from onpolicy.envs.overcooked.overcooked_ai_py.agents.benchmarking import AgentEvaluator
        AgentEvaluator.check_trajectories(trajectories, verbose=info)
        return trajectories

    ####################
    # TRAJECTORY UTILS #
    ####################

    @staticmethod
    def get_discounted_rewards(trajectories, gamma):
        rews = trajectories["ep_rewards"]
        horizon = rews.shape[1]
        return OvercookedEnv._get_discounted_rewards_with_horizon(rews, gamma, horizon)

    @staticmethod
    def _get_discounted_rewards_with_horizon(rewards_matrix, gamma, horizon):
        rewards_matrix = np.array(rewards_matrix)
        discount_array = [gamma ** i for i in range(horizon)]
        rewards_matrix = rewards_matrix[:, :horizon]
        discounted_rews = np.sum(rewards_matrix * discount_array, axis=1)
        return discounted_rews

    @staticmethod
    def get_agent_infos_for_trajectories(trajectories, agent_idx):
        """
        Returns a dictionary of the form
        {
            "[agent_info_0]": [ [episode_values], [], ... ],
            "[agent_info_1]": [ [], [], ... ],
            ...
        }
        with as keys the keys returned by the agent in it's agent_info dictionary

        NOTE: deprecated
        """
        agent_infos = []
        for traj_idx in range(len(trajectories["ep_lengths"])):
            ep_infos = trajectories["ep_infos"][traj_idx]
            traj_agent_infos = [step_info["agent_infos"][agent_idx] for step_info in ep_infos]

            # Append all dictionaries together
            traj_agent_infos = append_dictionaries(traj_agent_infos)
            agent_infos.append(traj_agent_infos)

        # Append all dictionaries together once again
        agent_infos = append_dictionaries(agent_infos)
        agent_infos = {k: np.array(v) for k, v in agent_infos.items()}
        return agent_infos

    @staticmethod
    def proportion_stuck_time(trajectories, agent_idx, stuck_time=3):
        """
        Simple util for calculating a guess for the proportion of time in the trajectories
        during which the agent with the desired agent index was stuck.

        NOTE: deprecated
        """
        stuck_matrix = []
        for traj_idx in range(len(trajectories["ep_lengths"])):
            stuck_matrix.append([])
            obs = trajectories["ep_states"][traj_idx]
            for traj_timestep in range(stuck_time, trajectories["ep_lengths"][traj_idx]):
                if traj_timestep >= stuck_time:
                    recent_states = obs[traj_timestep - stuck_time: traj_timestep + 1]
                    recent_player_pos_and_or = [s.players[agent_idx].pos_and_or for s in recent_states]

                    if len({item for item in recent_player_pos_and_or}) == 1:
                        # If there is only one item in the last stuck_time steps, then we classify the agent as stuck
                        stuck_matrix[traj_idx].append(True)
                    else:
                        stuck_matrix[traj_idx].append(False)
                else:
                    stuck_matrix[traj_idx].append(False)
        return stuck_matrix


class Overcooked(gym.Env):
    """
    Wrapper for the Env class above that is SOMEWHAT compatible with the standard gym API.

    NOTE: Observations returned are in a dictionary format with various information that is
    necessary to be able to handle the multi-agent nature of the environment. There are probably
    better ways to handle this, but we found this to work with minor modifications to OpenAI Baselines.

    NOTE: The index of the main agent in the mdp is randomized at each reset of the environment, and
    is kept track of by the self.agent_idx attribute. This means that it is necessary to pass on this
    information in the output to know for which agent index featurizations should be made for other agents.

    For example, say one is training A0 paired with A1, and A1 takes a custom state featurization.
    Then in the runner.py loop in OpenAI Baselines, we will get the lossless encodings of the state,
    and the true Overcooked state. When we encode the true state to feed to A1, we also need to know
    what agent index it has in the environment (as encodings will be index dependent).
    """
    env_name = "Overcooked-v0"

    def __init__(self, all_args, run_dir, baselines_reproducible=False, featurize_type=("ppo", "ppo"), stuck_time=4,
                 rank=None):
        """
        base_env: OvercookedEnv
        featurize_fn(mdp, state): fn used to featurize states returned in the 'both_agent_obs' field
        """
        if baselines_reproducible:
            # NOTE:
            # This will cause all agent indices to be chosen in sync across simulation
            # envs (for each update, all envs will have index 0 or index 1).
            # This is to prevent the randomness of choosing agent indexes
            # from leaking when using subprocess-vec-env in baselines (which
            # seeding does not reach) i.e. having different results for different
            # runs with the same seed.
            # The effect of this should be negligible, as all other randomness is
            # controlled by the actual run seeds
            np.random.seed(0)
        self.all_args = all_args
        self.agent_idx = 0
        self._initial_reward_shaping_factor = all_args.initial_reward_shaping_factor
        self.reward_shaping_factor = all_args.reward_shaping_factor
        self.reward_shaping_horizon = all_args.reward_shaping_horizon
        self.use_phi = all_args.use_phi
        self.use_hsp = all_args.use_hsp
        self.random_index = all_args.random_index
        self.store_traj = getattr(all_args, "store_traj", False)
        self.rank = rank
        if self.use_hsp:
            self.w0 = self.string2array(all_args.w0)
            self.w1 = self.string2array(all_args.w1)
            print(self.w0, self.w1)
        self.use_render = all_args.use_render
        self.num_agents = all_args.num_agents
        self.layout_name = all_args.layout_name
        self.episode_length = all_args.episode_length
        self.random_start_prob = getattr(all_args, "random_start_prob", 0.)
        self.stuck_time = stuck_time
        self.history_sa = []
        self.traj_num = 0
        self.step_count = 0
        self.run_dir = run_dir
        if getattr(all_args, "stage", 1) == 1:
            rew_shaping_params = {
                "PLACEMENT_IN_POT_REW": 0,
                "DISH_PICKUP_REWARD": 3,
                "SOUP_PICKUP_REWARD": 5,
                "PICKUP_TOMATO_REWARD": 0,
                "DISH_DISP_DISTANCE_REW": 0,
                "POT_DISTANCE_REW": 0,
                "SOUP_DISTANCE_REW": 0,
                "USEFUL_TOMATO_PICKUP": 0,
                "FOLLOW_TOMATO": 0,
                "PLACE_FIRST_TOMATO": 0,
            }
        else:
            if self.layout_name == "distant_tomato":
                rew_shaping_params = {
                    "PLACEMENT_IN_POT_REW": 0,
                    "DISH_PICKUP_REWARD": 3,
                    "SOUP_PICKUP_REWARD": 5,
                    "PICKUP_TOMATO_REWARD": 0,
                    "DISH_DISP_DISTANCE_REW": 0,
                    "POT_DISTANCE_REW": 0,
                    "SOUP_DISTANCE_REW": 0,
                    "USEFUL_TOMATO_PICKUP": 10,
                    "FOLLOW_TOMATO": 5,
                    "PLACE_FIRST_TOMATO": -10,
                }
            else:
                rew_shaping_params = {
                    "PLACEMENT_IN_POT_REW": 0,
                    "DISH_PICKUP_REWARD": 3,
                    "SOUP_PICKUP_REWARD": 5,
                    "PICKUP_TOMATO_REWARD": 0,
                    "DISH_DISP_DISTANCE_REW": 0,
                    "POT_DISTANCE_REW": 0,
                    "SOUP_DISTANCE_REW": 0,
                    "USEFUL_TOMATO_PICKUP": 0,
                    "FOLLOW_TOMATO": 0,
                    "PLACE_FIRST_TOMATO": 0,
                }
        self.base_mdp = OvercookedGridworld.from_layout_name(all_args.layout_name,
                                                             rew_shaping_params=rew_shaping_params)
        self.base_env = OvercookedEnv.from_mdp(self.base_mdp, horizon=self.episode_length,
                                               start_state_fn=self.random_start_prob)
        self.use_agent_policy_id = dict(all_args._get_kwargs()).get("use_agent_policy_id",
                                                                    False)  # Add policy id for loaded policy
        self.agent_policy_id = [-1. for _ in range(self.num_agents)]
        self.featurize_fn_ppo = lambda state: self.base_env.lossless_state_encoding_mdp(state)  # Encoding obs for PPO
        self.featurize_fn_bc = lambda state: self.base_env.featurize_state_mdp(state)  # Encoding obs for BC
        self.featurize_fn_mapping = {
            "ppo": self.featurize_fn_ppo,
            "bc": self.featurize_fn_bc
        }
        self.reset_featurize_type(featurize_type=featurize_type)  # default agents are both ppo

        if self.all_args.algorithm_name == "population":
            assert not self.random_index
            self.script_agent = [None, None]
            for player_idx, policy_name in enumerate([all_args.agent0_policy_name, all_args.agent1_policy_name]):
                if policy_name.startswith("script:"):
                    self.script_agent[player_idx] = SCRIPT_AGENTS[policy_name[7:]]()
                    self.script_agent[player_idx].reset(self.base_env.mdp, self.base_env.state, player_idx)
        else:
            self.script_agent = [None, None]

    def reset_featurize_type(self, featurize_type=("ppo", "ppo")):
        assert len(featurize_type) == 2
        self.featurize_type = featurize_type
        self.featurize_fn = lambda state: [self.featurize_fn_mapping[f](state)[i] * (255 if f == "ppo" else 1) for i, f
                                           in enumerate(self.featurize_type)]

        # reset observation_space, share_observation_space and action_space
        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []
        self._setup_observation_space()
        for i in range(2):
            self.observation_space.append(self._observation_space(featurize_type[i]))
            self.action_space.append(gym.spaces.Discrete(len(Action.ALL_ACTIONS)))
            self.share_observation_space.append(self._setup_share_observation_space())

    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            # No annealing if horizon is zero
            return start_v
        else:
            off_t = curr_t - start_t
            # Calculate the new value based on linear annealing formula
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v

    def onehot2idx(self, onehot):
        idx = []
        for a in onehot:
            idx.append(np.argmax(a))
        return idx

    def string2array(self, weight):
        w = []
        for s in weight.split(','):
            w.append(float(s))
        return np.array(w).astype(np.float32)

    def _action_convertor(self, action):
        return [a[0] for a in list(action)]

    def _observation_space(self, featurize_type):
        return {
            "ppo": self.ppo_observation_space,
            "bc": self.bc_observation_space
        }[featurize_type]

    def _setup_observation_space(self):
        dummy_state = self.base_env.mdp.get_standard_start_state()

        # ppo observation
        featurize_fn_ppo = lambda state: self.base_env.lossless_state_encoding_mdp(state)
        obs_shape = featurize_fn_ppo(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        self.ppo_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

        # bc observation
        featurize_fn_bc = lambda state: self.base_env.featurize_state_mdp(state)
        obs_shape = featurize_fn_bc(dummy_state)[0].shape
        high = np.ones(obs_shape) * 100
        low = np.ones(obs_shape) * -100
        self.bc_observation_space = gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

    def _setup_share_observation_space(self):
        dummy_state = self.base_env.mdp.get_standard_start_state()
        share_obs_shape = self.featurize_fn_ppo(dummy_state)[0].shape
        if self.use_agent_policy_id:
            share_obs_shape = [share_obs_shape[0], share_obs_shape[1], share_obs_shape[2] + 1]
        share_obs_shape = [share_obs_shape[0], share_obs_shape[1], share_obs_shape[2] * self.num_agents]
        high = np.ones(share_obs_shape) * float("inf")
        low = np.ones(share_obs_shape) * 0

        return gym.spaces.Box(np.float32(low), np.float32(high), dtype=np.float32)

    def _set_agent_policy_id(self, agent_policy_id):
        self.agent_policy_id = agent_policy_id

    def _gen_share_observation(self, state):
        share_obs = list(self.featurize_fn_ppo(state))
        if self.agent_idx == 1:
            share_obs = [share_obs[1], share_obs[0]]
        if self.use_agent_policy_id:
            for a in range(self.num_agents):
                share_obs[a] = np.concatenate(
                    [share_obs[a], np.ones((*share_obs[a].shape[:2], 1), dtype=np.float32) * self.agent_policy_id[a]],
                    axis=-1)
        share_obs0 = np.concatenate([share_obs[0], share_obs[1]], axis=-1) * 255
        share_obs1 = np.concatenate([share_obs[1], share_obs[0]], axis=-1) * 255
        return np.stack([share_obs0, share_obs1], axis=0)

    def step(self, action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format

        returns:
            observation: formatted to be standard input for self.agent_idx's policy

        main_agent_index:
            While existing other agent like planning or human model, use an index to fix the main RL-policy agent.
            Default False for multi-agent training.
        """
        self.step_count += 1
        action = self._action_convertor(action)
        assert all(self.action_space[0].contains(a) for a in action), "%r (%s) invalid" % (action, type(action))

        agent_action, other_agent_action = [Action.INDEX_TO_ACTION[a] for a in action]

        joint_action = [agent_action, other_agent_action]

        for a in range(self.num_agents):
            if self.script_agent[a] is not None:
                joint_action[a] = self.script_agent[a].step(self.base_env.mdp, self.base_env.state, a)
        joint_action = tuple(joint_action)

        if self.agent_idx == 1:
            joint_action = (other_agent_action, agent_action)

        if self.stuck_time > 0:
            self.history_sa[-1][1] = joint_action

        if self.store_traj:
            self.traj_to_store.append(info["shaped_info_by_agnet"])
            self.traj_to_store.append(joint_action)


        next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=False)

        dense_reward = info["shaped_r_by_agent"]
        shaped_reward_p0 = sparse_reward + self.reward_shaping_factor * dense_reward[0]
        shaped_reward_p1 = sparse_reward + self.reward_shaping_factor * dense_reward[1]


        if self.store_traj:
            self.traj_to_store.append(self.base_env.state.to_dict())

        reward = [[shaped_reward_p0], [shaped_reward_p1]]

        if self.agent_idx == 1:
            reward = [[shaped_reward_p1], [shaped_reward_p0]]

        self.history_sa = self.history_sa[1:] + [[next_state, None], ]

        # stuck
        stuck_info = []
        for agent_id in range(2):
            stuck, history_a = self.is_stuck(agent_id)
            if stuck:
                assert any([a not in history_a for a in Direction.ALL_DIRECTIONS]), history_a
                history_a_idxes = [Action.ACTION_TO_INDEX[a] for a in history_a]
                stuck_info.append([True, history_a_idxes])
            else:
                stuck_info.append([False, []])

        info["stuck"] = stuck_info

        # can_begin_cook_soup
        # can_begin_cook_info = [self.mdp.can_begin_cook_soup(next_state, agent_id) for agent_id in
        #                        range(self.num_agents)]
        # info["can_begin_cook"] = can_begin_cook_info

        if self.use_render:
            state = self.base_env.state
            self.traj["ep_states"][0].append(state)
            self.traj["ep_actions"][0].append(joint_action)
            self.traj["ep_rewards"][0].append(sparse_reward)
            self.traj["ep_dones"][0].append(done)
            self.traj["ep_infos"][0].append(info)
            if done:
                self.traj['ep_returns'].append(info['episode']['ep_sparse_r'])
                self.traj["mdp_params"].append(self.base_mdp.mdp_params)
                self.traj["env_params"].append(self.base_env.env_params)
                self.render()

        ob_p0, ob_p1 = self.featurize_fn(next_state)

        both_agents_ob = (ob_p0, ob_p1)
        if self.agent_idx == 1:
            both_agents_ob = (ob_p1, ob_p0)

        share_obs = self._gen_share_observation(self.base_env.state)
        done = [done, done]
        available_actions = np.ones((2, len(Action.ALL_ACTIONS)), dtype=np.uint8)

        return both_agents_ob, share_obs, reward, done, info, available_actions

    def anneal_reward_shaping_factor(self, timesteps):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        new_factor = self._anneal(self._initial_reward_shaping_factor, timesteps, self.reward_shaping_horizon)
        self.set_reward_shaping_factor(new_factor)

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

    def reset(self, reset_choose=True):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        if reset_choose:
            self.traj_num += 1
            self.step_count = 0
            self.base_env.reset()
            self.cumulative_shaped_info = [defaultdict(int), defaultdict(int)]

        if self.random_index:
            self.agent_idx = np.random.choice([0, 1])

        for a in range(self.num_agents):
            if self.script_agent[a] is not None:
                self.script_agent[a].reset(self.base_env.mdp, self.base_env.state, a)

        self.mdp = self.base_env.mdp
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.stuck_time > 0:
            self.history_sa = [None for _ in range(self.stuck_time - 1)] + [[self.base_env.state, None]]

        both_agents_ob = (ob_p0, ob_p1)
        if self.agent_idx == 1:
            both_agents_ob = (ob_p1, ob_p0)

        if self.use_render:
            self.init_traj()

        if self.store_traj:
            self.traj_to_store = []
            self.traj_to_store.append(self.base_env.state.to_dict())

        share_obs = self._gen_share_observation(self.base_env.state)
        available_actions = np.ones((2, len(Action.ALL_ACTIONS)), dtype=np.uint8)

        return both_agents_ob, share_obs, available_actions

    def is_stuck(self, agent_id):
        if self.stuck_time == 0 or None in self.history_sa:
            return False, []
        history_s = [sa[0] for sa in self.history_sa]
        history_a = [sa[1][agent_id] for sa in self.history_sa[:-1]]  # last action is None
        player_s = [s.players[agent_id] for s in history_s]
        pos_and_ors = [p.pos_and_or for p in player_s]
        cur_po = pos_and_ors[-1]
        if all([po[0] == cur_po[0] and po[1] == cur_po[1] for po in pos_and_ors]):
            return True, history_a
        return False, []

    def init_traj(self):
        self.traj = {k: [] for k in DEFAULT_TRAJ_KEYS}
        for key in TIMESTEP_TRAJ_KEYS:
            self.traj[key].append([])

    def render(self):
        try:
            save_dir = f'{self.run_dir}/gifs/{self.layout_name}/traj_num_{self.traj_num}'
            save_dir = os.path.expanduser(save_dir)
            StateVisualizer().display_rendered_trajectory(self.traj,
                                                          img_directory_path=save_dir,
                                                          ipython_display=False)
            for img_path in os.listdir(save_dir):
                img_path = save_dir + '/' + img_path
            imgs = []
            imgs_dir = os.listdir(save_dir)
            imgs_dir = sorted(imgs_dir, key=lambda x: int(x.split('.')[0]))
            for img_path in imgs_dir:
                img_path = save_dir + '/' + img_path
                imgs.append(imageio.imread(img_path))
            imageio.mimsave(save_dir + f'/reward_{self.traj["ep_returns"][0]}.gif', imgs, duration=0.05)
            imgs_dir = os.listdir(save_dir)
            for img_path in imgs_dir:
                img_path = save_dir + '/' + img_path
                if 'png' in img_path:
                    os.remove(img_path)
        except Exception as e:
            print('failed to render traj: ', e)

    def _store_trajectory(self):
        if not os.path.exists(f'{self.run_dir}/trajs/{self.layout_name}/'):
            os.makedirs(f'{self.run_dir}/trajs/{self.layout_name}/')
        save_dir = f'{self.run_dir}/trajs/{self.layout_name}/traj_{self.rank}_{self.traj_num}.pkl'
        pickle.dump(self.traj_to_store, open(save_dir, 'wb'))