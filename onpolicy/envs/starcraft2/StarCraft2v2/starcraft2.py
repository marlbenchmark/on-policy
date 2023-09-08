from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smacv2.env.multiagentenv import MultiAgentEnv

from smacv2.env.starcraft2.maps import get_map_params


import atexit
from warnings import warn
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging
from pysc2.lib.units import Neutral, Protoss, Terran, Zerg

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


EPS = 1e-7


class StarCraft2Env(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(
        self,
        map_name="8m",
        step_mul=8,
        move_amount=2,
        difficulty="7",
        game_version=None,
        seed=None,
        continuing_episode=False,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=False,
        obs_pathing_grid=False,
        obs_terrain_height=False,
        obs_instead_of_state=False,
        obs_timestep_number=False,
        obs_own_pos=False,
        obs_starcraft=True,
        conic_fov=False,
        num_fov_actions=12,
        state_last_action=True,
        state_timestep_number=False,
        reward_sparse=False,
        reward_only_positive=True,
        reward_death_value=10,
        reward_win=200,
        reward_defeat=0,
        reward_negative_scale=0.5,
        reward_scale=True,
        reward_scale_rate=20,
        use_unit_ranges=False,
        min_attack_range=2,
        kill_unit_step_mul=2,
        fully_observable=False,
        capability_config={},
        replay_dir="",
        replay_prefix="",
        window_size_x=1920,
        window_size_y=1200,
        heuristic_ai=False,
        heuristic_rest=False,
        debug=False,
        state_agent_id=True,
    ):
        """
        Create a StarCraftC2Env environment.

        Parameters
        ----------
        map_name : str, optional
            The name of the SC2 map to play (default is "8m"). The full list
            can be found by running bin/map_list.
        step_mul : int, optional
            How many game steps per agent step (default is 8). None
            indicates to use the default map step_mul.
        move_amount : float, optional
            How far away units are ordered to move per step (default is 2).
        difficulty : str, optional
            The difficulty of built-in computer AI bot (default is "7").
        game_version : str, optional
            StarCraft II game version (default is None). None indicates the
            latest version.
        seed : int, optional
            Random seed used during game initialisation. This allows to
        continuing_episode : bool, optional
            Whether to consider episodes continuing or finished after time
            limit is reached (default is False).
        obs_all_health : bool, optional
            Agents receive the health of all units (in the sight range) as part
            of observations (default is True).
        obs_own_health : bool, optional
            Agents receive their own health as a part of observations (default
            is False). This flag is ignored when obs_all_health == True.
        obs_last_action : bool, optional
            Agents receive the last actions of all units (in the sight range)
            as part of observations (default is False).
        obs_pathing_grid : bool, optional
            Whether observations include pathing values surrounding the agent
            (default is False).
        obs_terrain_height : bool, optional
            Whether observations include terrain height values surrounding the
            agent (default is False).
        obs_instead_of_state : bool, optional
            Use combination of all agents' observations as the global state
            (default is False).
        obs_timestep_number : bool, optional
            Whether observations include the current timestep of the episode
            (default is False).
        state_last_action : bool, optional
            Include the last actions of all agents as part of the global state
            (default is True).
        state_timestep_number : bool, optional
            Whether the state include the current timestep of the episode
            (default is False).
        reward_sparse : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). Whe rest of reward parameters are ignored if True.
        reward_only_positive : bool, optional
            Reward is always positive (default is True).
        reward_death_value : float, optional
            The amount of reward received for killing an enemy unit (default
            is 10). This is also the negative penalty for having an allied unit
            killed if reward_only_positive == False.
        reward_win : float, optional
            The reward for winning in an episode (default is 200).
        reward_defeat : float, optional
            The reward for loosing in an episode (default is 0). This value
            should be nonpositive.
        reward_negative_scale : float, optional
            Scaling factor for negative rewards (default is 0.5). This
            parameter is ignored when reward_only_positive == True.
        reward_scale : bool, optional
            Whether or not to scale the reward (default is True).
        reward_scale_rate : float, optional
            Reward scale rate (default is 20). When reward_scale == True, the
            reward received by the agents is divided by (max_reward /
            reward_scale_rate), where max_reward is the maximum possible
            reward per episode without considering the shield regeneration
            of Protoss units.
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        window_size_x : int, optional
            The length of StarCraft II window size (default is 1920).
        window_size_y: int, optional
            The height of StarCraft II window size (default is 1200).
        heuristic_ai: bool, optional
            Whether or not to use a non-learning heuristic AI (default False).
        heuristic_rest: bool, optional
            At any moment, restrict the actions of the heuristic AI to be
            chosen from actions available to RL agents (default is False).
            Ignored if heuristic_ai == False.
        debug: bool, optional
            Log messages about observations, state, actions and rewards for
            debugging purposes (default is False).
        """
        # Map arguments
        self.map_name = map_name
        map_params = get_map_params(self.map_name)
        self.map_params = map_params
        self.episode_limit = map_params["limit"]
        self._move_amount = move_amount
        self._step_mul = step_mul
        self._kill_unit_step_mul = kill_unit_step_mul
        self.difficulty = difficulty

        # Observations and state
        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_instead_of_state = obs_instead_of_state
        self.obs_last_action = obs_last_action
        self.obs_pathing_grid = obs_pathing_grid
        self.obs_terrain_height = obs_terrain_height
        self.obs_timestep_number = obs_timestep_number
        self.obs_starcraft = obs_starcraft
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9
        self.state_agent_id = state_agent_id

        # Rewards args
        self.reward_sparse = reward_sparse
        self.reward_only_positive = reward_only_positive
        self.reward_negative_scale = reward_negative_scale
        self.reward_death_value = reward_death_value
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat
        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate

        # Meta MARL
        self.capability_config = capability_config
        self.fully_observable = fully_observable
        self.stochastic_attack = "attack" in self.capability_config
        self.stochastic_health = "health" in self.capability_config
        self.replace_teammates = "team_gen" in self.capability_config
        self.obs_own_pos = obs_own_pos
        self.mask_enemies = "enemy_mask" in self.capability_config
        if self.stochastic_attack:
            self.zero_pad_stochastic_attack = not self.capability_config[
                "attack"
            ]["observe"]
            self.observe_attack_probs = self.capability_config["attack"][
                "observe"
            ]
        if self.stochastic_health:
            self.zero_pad_health = not self.capability_config["health"][
                "observe"
            ]
            self.observe_teammate_health = self.capability_config["health"][
                "observe"
            ]
        if self.replace_teammates:
            self.zero_pad_unit_types = not self.capability_config["team_gen"][
                "observe"
            ]
            self.observe_teammate_types = self.capability_config["team_gen"][
                "observe"
            ]
        self.n_agents = (
            map_params["n_agents"]
            if not self.replace_teammates
            else self.capability_config["team_gen"]["n_units"]
        )
        self.n_enemies = (
            map_params["n_enemies"]
            if not self.replace_teammates
            else self.capability_config["team_gen"]["n_enemies"]
        )
        self.random_start = "start_positions" in self.capability_config
        self.conic_fov = conic_fov
        self.n_fov_actions = num_fov_actions if self.conic_fov else 0
        self.conic_fov_angle = (
            (2 * np.pi) / self.n_fov_actions if self.conic_fov else 0
        )
        # Other
        self.game_version = game_version
        self.continuing_episode = continuing_episode
        self._seed = seed
        self.heuristic_ai = heuristic_ai
        self.heuristic_rest = heuristic_rest
        self.debug = debug
        self.window_size = (window_size_x, window_size_y)
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix
        self.use_unit_ranges = use_unit_ranges
        self.min_attack_range = min_attack_range

        # Actions
        self.n_actions_move = 4

        self.n_actions_no_attack = self.n_actions_move + self.n_fov_actions + 2
        self.n_actions = self.n_actions_no_attack + self.n_enemies

        # Map info
        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        # NOTE: The map_type, which is used to initialise the unit
        # type ids, the unit_type_bits and the races, are still properties of the
        # map. This means even the 10gen_{race} maps are limited to the
        # unit types statically defined in the unit type id assignment.
        # Lifting this restriction shouldn't be too much work, I've just
        # not done it.
        self.unit_type_bits = map_params["unit_type_bits"]
        self.map_type = map_params["map_type"]
        self._unit_types = None

        self.max_reward = (
            self.n_enemies * self.reward_death_value + self.reward_win
        )

        # create lists containing the names of attributes returned in states
        self.ally_state_attr_names = [
            "visible",
            "distance",
            "rel_x",
            "rel_y",
            "energy/cooldown",
            "center_x",
            "center_y",
            "health",
        ]
        self.enemy_state_attr_names = [
            "available", 
            "distance", 
            "rel_x", 
            "rel_y", 
            "visible", 
            "center_x", 
            "center_y", 
            "health", ]

        if self.shield_bits_ally > 0:
            self.ally_state_attr_names += ["shield"]
        if self.shield_bits_enemy > 0:
            self.enemy_state_attr_names += ["shield"]
        if self.conic_fov:
            self.ally_state_attr_names += ["fov_x", "fov_y"]

        self.capability_attr_names = []
        if "attack" in self.capability_config:
            self.capability_attr_names += ["attack_probability"]
        if "health" in self.capability_config:
            self.capability_attr_names += ["total_health"]
        if self.unit_type_bits > 0:
            bit_attr_names = [
                "type_{}".format(bit) for bit in range(self.unit_type_bits)
            ]
            self.capability_attr_names += bit_attr_names
            self.enemy_state_attr_names += bit_attr_names

        self.agents = {}
        self.enemies = {}
        self.unit_name_to_id_map = {}
        self.id_to_unit_name_map = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._obs = None
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None
        self.agent_attack_probabilities = np.zeros(self.n_agents)
        self.agent_health_levels = np.zeros(self.n_agents)
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.fov_directions = np.zeros((self.n_agents, 2))
        self.fov_directions[:, 0] = 1.0
        self.canonical_fov_directions = np.array(
            [
                (
                    np.cos(2 * np.pi * (i / self.n_fov_actions)),
                    np.sin(2 * np.pi * (i / self.n_fov_actions)),
                )
                for i in range(self.n_fov_actions)
            ]
        )
        self.new_unit_positions = np.zeros((self.n_agents, 2))
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.init_positions = np.zeros((self.n_agents, 2))
        self._min_unit_type = 0
        self.marine_id = self.marauder_id = self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0
        self.stalker_id = self.colossus_id = self.zealot_id = 0
        self.max_distance_x = 0
        self.max_distance_y = 0
        self.map_x = 0
        self.map_y = 0
        self.reward = 0
        self.renderer = None
        self.terrain_height = None
        self.pathing_grid = None
        self.state_feature_names = self.build_state_feature_names()
        self.obs_feature_names = self.build_obs_feature_names()
        self._run_config = None
        self._sc2_proc = None
        self._controller = None
        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())

    def _only_one_meta_marl_flag_on(self):
        """Function that checks that either all the meta marl flags are off,
        or at most one has been enabled."""
        if self.stochastic_attack:
            return not self.stochastic_health and not self.replace_teammates
        else:
            return not self.replace_teammates or not self.stochastic_health

    def _launch(self):
        """Launch the StarCraft II game."""
        self._run_config = run_configs.get(version=self.game_version)
        self.version = self._run_config.version
        _map = maps.get(self.map_name)

        # Setting up the interface
        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
        self._sc2_proc = self._run_config.start(
            window_size=self.window_size, want_rgb=False
        )
        self._controller = self._sc2_proc.controller

        # Request to create the game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path),
            ),
            realtime=False,
            random_seed=self._seed,
        )
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(
            type=sc_pb.Computer,
            race=races[self._bot_race],
            difficulty=difficulties[self.difficulty],
        )
        self._controller.create_game(create)

        join = sc_pb.RequestJoinGame(
            race=races[self._agent_race], options=interface_options
        )
        self._controller.join_game(join)

        game_info = self._controller.game_info()
        map_info = game_info.start_raw
        self.map_play_area_min = map_info.playable_area.p0
        self.map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = (
            self.map_play_area_max.x - self.map_play_area_min.x
        )
        self.max_distance_y = (
            self.map_play_area_max.y - self.map_play_area_min.y
        )
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y

        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8)
            )
            self.pathing_grid = np.transpose(
                np.array(
                    [
                        [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                        for row in vals
                    ],
                    dtype=bool,
                )
            )
        else:
            self.pathing_grid = np.invert(
                np.flip(
                    np.transpose(
                        np.array(
                            list(map_info.pathing_grid.data), dtype=np.bool
                        ).reshape(self.map_x, self.map_y)
                    ),
                    axis=1,
                )
            )

        self.terrain_height = (
            np.flip(
                np.transpose(
                    np.array(list(map_info.terrain_height.data)).reshape(
                        self.map_x, self.map_y
                    )
                ),
                1,
            )
            / 255
        )

    def reset(self, episode_config={}):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        self.episode_config = episode_config
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()

        # Information kept for counting the reward
        self.agent_attack_probabilities = episode_config.get("attack", {}).get(
            "item", None
        )
        self.agent_health_levels = episode_config.get("health", {}).get(
            "item", None
        )
        self.enemy_mask = episode_config.get("enemy_mask", {}).get(
            "item", None
        )
        self.ally_start_positions = episode_config.get(
            "ally_start_positions", {}
        ).get("item", None)
        self.enemy_start_positions = episode_config.get(
            "enemy_start_positions", {}
        ).get("item", None)
        self.mask_enemies = self.enemy_mask is not None
        ally_team = episode_config.get("team_gen", {}).get("ally_team", None)
        enemy_team = episode_config.get("team_gen", {}).get("enemy_team", None)
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.fov_directions = np.zeros((self.n_agents, 2))
        self.fov_directions[:, 0] = 1.0
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False
        if self.debug:
            logging.debug(
                f"Attack Probabilities: {self.agent_attack_probabilities}"
            )
            logging.debug(f"Health Levels: {self.agent_health_levels}")
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        try:
            self._obs = self._controller.observe()
            self.init_units(
                ally_team, enemy_team, episode_config=episode_config
            )
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(
                    60, "*"
                )
            )
        
        global_state = np.array([self.get_state_agent(agent_id) for agent_id in range(self.n_agents)])
        return self.get_obs(), global_state # self.get_state()

    def _restart(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        """
        try:
            self._kill_all_units()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

    def full_restart(self):
        """Full restart. Closes the SC2 process and launches a new one."""
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1

    def _kill_units_below_health_level(self):
        units_to_kill = []
        for al_id, al_unit in self.agents.items():
            if (
                al_unit.health / al_unit.health_max
                < self.agent_health_levels[al_id]
            ) and not self.death_tracker_ally[al_id]:
                units_to_kill.append(al_unit.tag)
        self._kill_units(units_to_kill)

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions_int):
            if not self.heuristic_ai:
                sc_action = self.get_agent_action(a_id, action)
            else:
                sc_action, action_num = self.get_agent_action_heuristic(
                    a_id, action
                )
                actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)
        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)

        try:

            if self.conic_fov:
                self.render_fovs()
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            if not self.stochastic_health:
                self._controller.step(self._step_mul)
            else:
                self._controller.step(
                    self._step_mul - self._kill_unit_step_mul
                )
                self._kill_units_below_health_level()
                self._controller.step(self._kill_unit_step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return 0, True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for _al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1
        for _e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1

        info["dead_allies"] = dead_allies
        info["dead_enemies"] = dead_enemies

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, "-"))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        self.reward = reward

        return reward, terminated, info

    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert (
            avail_actions[action] == 1
        ), "Agent {} cannot perform action {}".format(a_id, action)

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            self.new_unit_positions[a_id] = np.array(
                [x, y + self._move_amount]
            )
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            self.new_unit_positions[a_id] = np.array(
                [x, y - self._move_amount]
            )
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            self.new_unit_positions[a_id] = np.array(
                [x + self._move_amount, y]
            )

            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            self.new_unit_positions[a_id] = np.array(
                [x - self._move_amount, y]
            )
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))
        elif self.conic_fov and action in range(6, 6 + self.n_fov_actions):
            self.fov_directions[a_id] = self.canonical_fov_directions[
                action - 6
            ]
            cmd = None
        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            if (
                self.map_type in ["MMM", "terran_gen"]
                and unit.unit_type == self.medivac_id
            ):
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
                target_unit = self.enemies[target_id]
                action_name = "attack"

            if self.stochastic_attack:
                p = np.random.default_rng().uniform()
                if p > self.agent_attack_probabilities[a_id]:
                    if self.debug:
                        logging.debug(
                            f"Agent {a_id} {action_name}s {target_id}, but fails"
                        )
                    return None
            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )

            if self.debug:
                logging.debug(
                    "Agent {} {}s unit # {}".format(
                        a_id, action_name, target_id
                    )
                )
        if cmd:
            sc_action = sc_pb.Action(
                action_raw=r_pb.ActionRaw(unit_command=cmd)
            )
            return sc_action
        return None

    def get_agent_action_heuristic(self, a_id, action):
        unit = self.get_unit_by_id(a_id)
        tag = unit.tag

        target = self.heuristic_targets[a_id]
        if unit.unit_type == self.medivac_id:
            if (
                target is None
                or self.agents[target].health == 0
                or self.agents[target].health == self.agents[target].health_max
            ):
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for al_id, al_unit in self.agents.items():
                    if al_unit.unit_type == self.medivac_id:
                        continue
                    if (
                        al_unit.health != 0
                        and al_unit.health != al_unit.health_max
                    ):
                        dist = self.distance(
                            unit.pos.x,
                            unit.pos.y,
                            al_unit.pos.x,
                            al_unit.pos.y,
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_id = al_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions["heal"]
            target_tag = self.agents[self.heuristic_targets[a_id]].tag
        else:
            if target is None or self.enemies[target].health == 0:
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for e_id, e_unit in self.enemies.items():
                    if (
                        unit.unit_type == self.marauder_id
                        and e_unit.unit_type == self.medivac_id
                    ):
                        continue
                    if e_unit.health > 0:
                        dist = self.distance(
                            unit.pos.x, unit.pos.y, e_unit.pos.x, e_unit.pos.y
                        )
                        if dist < min_dist:
                            min_dist = dist
                            min_id = e_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions["attack"]
            target_tag = self.enemies[self.heuristic_targets[a_id]].tag

        action_num = self.heuristic_targets[a_id] + self.n_actions_no_attack

        # Check if the action is available
        if (
            self.heuristic_rest
            and self.get_avail_agent_actions(a_id)[action_num] == 0
        ):

            # Move towards the target rather than attacking/healing
            if unit.unit_type == self.medivac_id:
                target_unit = self.agents[self.heuristic_targets[a_id]]
            else:
                target_unit = self.enemies[self.heuristic_targets[a_id]]

            delta_x = target_unit.pos.x - unit.pos.x
            delta_y = target_unit.pos.y - unit.pos.y

            if abs(delta_x) > abs(delta_y):  # east or west
                if delta_x > 0:  # east
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x + self._move_amount, y=unit.pos.y
                    )
                    action_num = 4
                else:  # west
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x - self._move_amount, y=unit.pos.y
                    )
                    action_num = 5
            else:  # north or south
                if delta_y > 0:  # north
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y + self._move_amount
                    )
                    action_num = 2
                else:  # south
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y - self._move_amount
                    )
                    action_num = 3

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=target_pos,
                unit_tags=[tag],
                queue_command=False,
            )
        else:
            # Attack/heal the target
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action, action_num

    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        assert (
            not self.stochastic_health or self.reward_only_positive
        ), "Different Health Levels are currently only compatible with positive rewards"
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                    self.previous_ally_units[al_id].health
                    + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += neg_scale * (
                        prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                    self.previous_enemy_units[e_id].health
                    + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:
            reward = max(delta_enemy + delta_deaths, 0)  # shield regeneration
        else:
            reward = delta_enemy + delta_deaths - delta_ally

        return reward

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, agent_id):
        """Returns the shooting range for an agent."""
        if self.use_unit_ranges:
            attack_range_map = {
                self.stalker_id: 6,
                self.zealot_id: 0.1,
                self.colossus_id: 7,
                self.zergling_id: 0.1,
                self.baneling_id: 0.25,
                self.hydralisk_id: 5,
                self.marine_id: 5,
                self.marauder_id: 6,
                self.medivac_id: 4,
            }
            unit = self.agents[agent_id]
            return max(attack_range_map[unit.unit_type], self.min_attack_range)
        else:
            return 6

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent."""
        # get the unit
        if self.use_unit_ranges:
            sight_range_map = {
                self.stalker_id: 10,
                self.zealot_id: 9,
                self.colossus_id: 10,
                self.zergling_id: 8,
                self.baneling_id: 8,
                self.hydralisk_id: 9,
                self.marine_id: 9,
                self.marauder_id: 10,
                self.medivac_id: 11,
            }
            unit = self.agents[agent_id]
            return sight_range_map[unit.unit_type]
        else:
            return 9

    def unit_max_cooldown(self, unit):
        """Returns the maximal cooldown for a unit."""
        switcher = {
            self.marine_id: 15,
            self.marauder_id: 25,
            self.medivac_id: 200,  # max energy
            self.stalker_id: 35,
            self.zealot_id: 22,
            self.colossus_id: 24,
            self.hydralisk_id: 10,
            self.zergling_id: 11,
            self.baneling_id: 1,
        }
        return switcher.get(unit.unit_type, 15)

    def save_replay(self):
        """Save a replay."""
        prefix = self.replay_prefix or self.map_name
        replay_dir = self.replay_dir or ""
        replay_path = self._run_config.save_replay(
            self._controller.save_replay(),
            replay_dir=replay_dir,
            prefix=prefix,
        )
        logging.info("Replay saved at: %s" % replay_path)

    def unit_max_shield(self, unit):
        """Returns maximal shield for a given unit."""
        if unit.unit_type == 74 or unit.unit_type == self.stalker_id:
            return 80  # Protoss's Stalker
        elif unit.unit_type == 73 or unit.unit_type == self.zealot_id:
            return 50  # Protoss's Zealot
        elif unit.unit_type == 4 or unit.unit_type == self.colossus_id:
            return 150  # Protoss's Colossus
        else:
            raise Exception("Maximum shield not recognised")

    def build_state_feature_names(self):
        """Return the state feature names."""
        if self.obs_instead_of_state:
            raise NotImplementedError

        feature_names = []

        # Ally features.
        for al_id in range(self.n_agents):
            feature_names.append(f"ally_health_{al_id}")
            feature_names.append(f"ally_cooldown_{al_id}")
            feature_names.append(f"ally_relative_x_{al_id}")
            feature_names.append(f"ally_relative_y_{al_id}")

            if self.shield_bits_ally > 0:
                feature_names.append(f"ally_shield_{al_id}")

            if self.stochastic_attack:
                feature_names.append(f"ally_attack_prob_{al_id}")

            if self.stochastic_health:
                feature_names.append(f"ally_health_level_{al_id}")

            if self.conic_fov:
                feature_names.append(f"ally_fov_x_{al_id}")
                feature_names.append(f"ally_fov_y_{al_id}")

            if self.unit_type_bits > 0:
                for bit in range(self.unit_type_bits):
                    feature_names.append(f"ally_unit_type_{al_id}_bit_{bit}")

        # Enemy features.
        for e_id in range(self.n_enemies):
            feature_names.append(f"enemy_health_{e_id}")
            feature_names.append(f"enemy_relative_x_{e_id}")
            feature_names.append(f"enemy_relative_y_{e_id}")

            if self.shield_bits_enemy > 0:
                feature_names.append(f"enemy_shield_{e_id}")

            if self.unit_type_bits > 0:
                for bit in range(self.unit_type_bits):
                    feature_names.append(f"enemy_unit_type_{e_id}_bit_{bit}")

        if self.state_last_action:
            for al_id in range(self.n_agents):
                for action_idx in range(self.n_actions):
                    feature_names.append(
                        f"ally_last_action_{al_id}_action_{action_idx}"
                    )

        if self.state_timestep_number:
            feature_names.append("timestep")

        return feature_names

    def get_state_feature_names(self):
        return self.state_feature_names

    def build_obs_feature_names(self):
        """Return the observations feature names."""
        feature_names = []

        # Movement features.
        feature_names.extend(
            [
                "move_action_north",
                "move_action_south",
                "move_action_east",
                "move_action_west",
            ]
        )
        if self.obs_pathing_grid:
            feature_names.extend(
                [f"pathing_grid_{n}" for n in range(self.n_obs_pathing)]
            )
        if self.obs_terrain_height:
            feature_names.extend(
                [f"terrain_height_{n}" for n in range(self.n_obs_height)]
            )

        # Enemy features.
        for e_id in range(self.n_enemies):
            feature_names.extend(
                [
                    f"enemy_shootable_{e_id}",
                    f"enemy_distance_{e_id}",
                    f"enemy_relative_x_{e_id}",
                    f"enemy_relative_y_{e_id}",
                ]
            )
            if self.obs_all_health:
                feature_names.append(f"enemy_health_{e_id}")
            if self.obs_all_health and self.shield_bits_enemy > 0:
                feature_names.append(f"enemy_shield_{e_id}")
            if self.unit_type_bits > 0:
                feature_names.extend(
                    [
                        f"enemy_unit_type_{e_id}_bit_{bit}"
                        for bit in range(self.unit_type_bits)
                    ]
                )

        # Ally features.
        # From the perspective of agent 0.
        al_ids = [al_id for al_id in range(self.n_agents) if al_id != 0]
        for al_id in al_ids:
            feature_names.extend(
                [
                    f"ally_visible_{al_id}",
                    f"ally_distance_{al_id}",
                    f"ally_relative_x_{al_id}",
                    f"ally_relative_y_{al_id}",
                ]
            )
            if self.obs_all_health:
                feature_names.append(f"ally_health_{al_id}")
                if self.shield_bits_ally > 0:
                    feature_names.append(f"ally_shield_{al_id}")
            if self.stochastic_attack and (
                self.observe_attack_probs or self.zero_pad_stochastic_attack
            ):
                feature_names.append(f"ally_attack_prob_{al_id}")
            if self.stochastic_health and (
                self.observe_teammate_health or self.zero_pad_health
            ):
                feature_names.append(f"ally_health_level_{al_id}")
            if self.unit_type_bits > 0 and (
                (not self.replace_teammates or self.observe_teammate_types)
                or self.zero_pad_unit_types
            ):
                feature_names.extend(
                    [
                        f"ally_unit_type_{al_id}_bit_{bit}"
                        for bit in range(self.unit_type_bits)
                    ]
                )
            if self.obs_last_action:
                feature_names.extend(
                    [
                        f"ally_last_action_{al_id}_action_{action}"
                        for action in range(self.n_actions)
                    ]
                )

        # Own features.
        if self.obs_own_health:
            feature_names.append("own_health")
            if self.shield_bits_ally > 0:
                feature_names.append("own_shield")
        if self.stochastic_attack:
            feature_names.append("own_attack_prob")
        if self.stochastic_health:
            feature_names.append("own_health_level")
        if self.obs_own_pos:
            feature_names.extend(["own_pos_x", "own_pos_y"])
        if self.conic_fov:
            feature_names.extend(["own_fov_x", "own_fov_y"])
        if self.unit_type_bits > 0:
            feature_names.extend(
                [
                    f"own_unit_type_bit_{bit}"
                    for bit in range(self.unit_type_bits)
                ]
            )
        if not self.obs_starcraft:
            feature_names = []

        if self.obs_timestep_number:
            feature_names.append("timestep")

        return feature_names

    def get_obs_feature_names(self):
        return self.obs_feature_names

    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount / 2

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if self.check_bounds(x, y) and self.pathing_grid[x, y]:
            return True

        return False

    def get_surrounding_points(self, unit, include_self=False):
        """Returns the surrounding points of the unit in 8 directions."""
        x = int(unit.pos.x)
        y = int(unit.pos.y)

        ma = self._move_amount

        points = [
            (x, y + 2 * ma),
            (x, y - 2 * ma),
            (x + 2 * ma, y),
            (x - 2 * ma, y),
            (x + ma, y + ma),
            (x - ma, y - ma),
            (x + ma, y - ma),
            (x - ma, y + ma),
        ]

        if include_self:
            points.append((x, y))

        return points

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return 0 <= x < self.map_x and 0 <= y < self.map_y

    def get_surrounding_pathing(self, unit):
        """Returns pathing values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=False)
        vals = [
            self.pathing_grid[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def get_surrounding_height(self, unit):
        """Returns height values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=True)
        vals = [
            self.terrain_height[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def _compute_health(self, agent_id, unit):
        """Each agent has a health bar with max health
        `health_max` and current health `health`. We set a level
        `health_level` between `0` and `1` where the agent dies if its
        proportional health (`health / health_max`) is below that level.
        This function rescales health to take into account this death level.

        In the proportional health scale we have something that looks like this:

        -------------------------------------------------------------
        0                                                            1
                  ^ health_level            ^ proportional_health
        And so we compute
            (proportional_health - health_level) / (1 - health_level)
        """
        proportional_health = unit.health / unit.health_max
        health_level = self.agent_health_levels[agent_id]
        return (1.0 / (1 - health_level)) * (
            proportional_health - health_level
        )

    def render_fovs(self):
        lines_to_render = []
        for agent_id in range(self.n_agents):
            if not self.death_tracker_ally[agent_id]:
                lines_to_render.extend(self.agent_cone(agent_id))
        debug_command = d_pb.DebugCommand(
            draw=d_pb.DebugDraw(lines=lines_to_render)
        )
        self._controller.debug(debug_command)

    def agent_cone(self, agent_id):
        fov_direction = self.fov_directions[agent_id]
        c, s = np.cos(self.conic_fov_angle / 2), np.sin(
            self.conic_fov_angle / 2
        )
        sight_range = self.unit_sight_range(agent_id)
        rot = np.array([[c, -s], [s, c]])
        neg_rot = np.array([[c, s], [-s, c]])
        start_pos = self.new_unit_positions[agent_id]
        init_pos = sc_common.Point(
            x=start_pos[0],
            y=start_pos[1],
            z=self.get_unit_by_id(agent_id).pos.z,
        )
        upper_cone_end = start_pos + (rot @ fov_direction) * sight_range
        lower_cone_end = start_pos + (neg_rot @ fov_direction) * sight_range
        lines = [
            d_pb.DebugLine(
                line=d_pb.Line(
                    p0=init_pos,
                    p1=sc_common.Point(
                        x=upper_cone_end[0],
                        y=upper_cone_end[1],
                        z=init_pos.z,
                    ),
                )
            ),
            d_pb.DebugLine(
                line=d_pb.Line(
                    p0=init_pos,
                    p1=sc_common.Point(
                        x=lower_cone_end[0],
                        y=lower_cone_end[1],
                        z=init_pos.z,
                    ),
                )
            ),
        ]
        return lines

    def is_position_in_cone(self, agent_id, pos, range="sight_range"):
        ally_pos = self.get_unit_by_id(agent_id).pos
        distance = self.distance(ally_pos.x, ally_pos.y, pos.x, pos.y)
        # position is in this agent's cone if it is not outside the sight
        # range and has the correct angle
        if range == "sight_range":
            unit_range = self.unit_sight_range(agent_id)
        elif range == "shoot_range":
            unit_range = self.unit_shoot_range(agent_id)
        else:
            raise Exception("Range argument not recognised")
        if distance > unit_range:
            return False
        x_diff = pos.x - ally_pos.x
        x_diff = max(x_diff, EPS) if x_diff > 0 else min(x_diff, -EPS)
        obj_angle = np.arctan((pos.y - ally_pos.y) / x_diff)
        x = self.fov_directions[agent_id][0]
        x = max(x, EPS) if x_diff > 0 else min(x, -EPS)
        fov_angle = np.arctan(self.fov_directions[agent_id][1] / x)
        return np.abs(obj_angle - fov_angle) < self.conic_fov_angle / 2

    def get_obs_agent(self, agent_id, fully_observable=False):
        """Returns observation for agent_id. The observation is composed of:

        - agent movement features (where it can move to, height information
            and pathing grid)
        - enemy features (available_to_attack, health, relative_x, relative_y,
            shield, unit_type)
        - ally features (visible, distance, relative_x, relative_y, shield,
            unit_type)
        - agent unit features (health, shield, unit_type)

        All of this information is flattened and concatenated into a list,
        in the aforementioned order. To know the sizes of each of the
        features inside the final list of features, take a look at the
        functions ``get_obs_move_feats_size()``,
        ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
        ``get_obs_own_feats_size()``.

        The size of the observation vector may vary, depending on the
        environment configuration and type of units present in the map.
        For instance, non-Protoss units will not have shields, movement
        features may or may not include terrain height and pathing grid,
        unit_type is not included if there is only one type of unit in the
        map etc.).

        NOTE: Agents should have access only to their local observations
        during decentralised execution.

        fully_observable: -- ignores sight range for a particular unit.
        For Debugging purposes ONLY -- not a fair observation.
        """
        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        if (
            unit.health > 0 and self.obs_starcraft
        ):  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features. Do not need similar for looking
            # around because this is always possible
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                    ind : ind + self.n_obs_pathing  # noqa
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)
                enemy_visible = (
                    self.is_position_in_cone(agent_id, e_unit.pos)
                    if self.conic_fov
                    else dist < sight_range
                )
                if (enemy_visible and e_unit.health > 0) or (
                    e_unit.health > 0 and fully_observable
                ):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[
                        self.n_actions_no_attack + e_id
                    ]  # available
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (
                        e_x - x
                    ) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (
                        e_y - y
                    ) / sight_range  # relative Y
                    show_enemy = (
                        self.mask_enemies
                        and not self.enemy_mask[agent_id][e_id]
                    ) or not self.mask_enemies
                    ind = 4
                    if self.obs_all_health and show_enemy:
                        enemy_feats[e_id, ind] = (
                            e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (
                                e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0 and show_enemy:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type

            # Ally features
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)
                ally_visible = (
                    self.is_position_in_cone(agent_id, al_unit.pos)
                    if self.conic_fov
                    else dist < sight_range
                )
                if (ally_visible and al_unit.health > 0) or (
                    al_unit.health > 0 and fully_observable
                ):  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        if not self.stochastic_health:
                            ally_feats[i, ind] = (
                                al_unit.health / al_unit.health_max
                            )  # health
                            ind += 1
                        elif self.observe_teammate_health:
                            ally_feats[i, ind] = self._compute_health(
                                agent_id=al_id, unit=al_unit
                            )
                            ind += 1
                        elif self.zero_pad_health:
                            ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                al_unit.shield / max_shield
                            )  # shield
                            ind += 1
                    if self.stochastic_attack and self.observe_attack_probs:
                        ally_feats[i, ind] = self.agent_attack_probabilities[
                            al_id
                        ]
                        ind += 1
                    elif (
                        self.stochastic_attack
                        and self.zero_pad_stochastic_attack
                    ):
                        ind += 1

                    if self.stochastic_health and self.observe_teammate_health:
                        ally_feats[i, ind] = self.agent_health_levels[al_id]
                        ind += 1
                    elif self.stochastic_health and self.zero_pad_health:
                        ind += 1
                    if self.unit_type_bits > 0 and (
                        not self.replace_teammates
                        or self.observe_teammate_types
                    ):
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits
                    elif self.unit_type_bits > 0 and self.zero_pad_unit_types:
                        ind += self.unit_type_bits
                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                if not self.stochastic_health:
                    own_feats[ind] = unit.health / unit.health_max
                else:
                    own_feats[ind] = self._compute_health(agent_id, unit)
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.stochastic_attack:
                own_feats[ind] = self.agent_attack_probabilities[agent_id]
                ind += 1
            if self.stochastic_health:
                own_feats[ind] = self.agent_health_levels[agent_id]
                ind += 1
            if self.obs_own_pos:
                own_feats[ind] = x / self.map_x
                own_feats[ind + 1] = y / self.map_y
                ind += 2
            if self.conic_fov:
                own_feats[ind : ind + 2] = self.fov_directions[agent_id]
                ind += 2
            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1
        if self.obs_starcraft:
            agent_obs = np.concatenate(
                (
                    move_feats.flatten(),
                    enemy_feats.flatten(),
                    ally_feats.flatten(),
                    own_feats.flatten(),
                )
            )

        if self.obs_timestep_number:
            if self.obs_starcraft:
                agent_obs = np.append(
                    agent_obs, self._episode_steps / self.episode_limit
                )
            else:
                agent_obs = np.zeros(1, dtype=np.float32)
                agent_obs[:] = self._episode_steps / self.episode_limit

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug(
                "Avail. actions {}".format(
                    self.get_avail_agent_actions(agent_id)
                )
            )
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        return agent_obs

    def get_state_agent(self, agent_id):
        """Returns the global state.
        NOTE: This function should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat

        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_enemy_num_attributes()
        ally_feats_dim = self.get_ally_num_attributes()
        own_feats_dim = self.get_state_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros((self.n_enemies, enemy_feats_dim), dtype=np.float32)
        ally_feats = np.zeros(((self.n_agents - 1), ally_feats_dim), dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features. Do not need similar for looking
            # around because this is always possible
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                    ind : ind + self.n_obs_pathing  # noqa
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)
                enemy_visible = (
                    self.is_position_in_cone(agent_id, e_unit.pos)
                    if self.conic_fov
                    else dist < sight_range
                )
                if e_unit.health > 0 or (
                    e_unit.health > 0 and fully_observable
                ):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[
                        self.n_actions_no_attack + e_id
                    ]  # available
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (
                        e_x - x
                    ) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (
                        e_y - y
                    ) / sight_range  # relative Y
                    show_enemy = (
                        self.mask_enemies
                        and not self.enemy_mask[agent_id][e_id]
                    ) or not self.mask_enemies
                    enemy_feats[e_id, 4] = enemy_visible  # visible
                    ind = 5

                    enemy_feats[e_id, ind] = (e_x - center_x) / self.max_distance_x  # center X
                    enemy_feats[e_id, ind+1] = (e_y - center_y) / self.max_distance_y  # center Y
                    ind += 2

                    if self.obs_all_health and show_enemy:
                        enemy_feats[e_id, ind] = (
                            e_unit.health / e_unit.health_max
                        )  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (
                                e_unit.shield / max_shield
                            )  # shield
                            ind += 1

                    if self.unit_type_bits > 0 and show_enemy:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type


            # Ally features
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)
                ally_visible = (
                    self.is_position_in_cone(agent_id, al_unit.pos)
                    if self.conic_fov
                    else dist < sight_range
                )
                if al_unit.health > 0:  # visible and alive
                    ally_feats[i, 0] = ally_visible  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    max_cd = self.unit_max_cooldown(al_unit)
                    if (
                        self.map_type in ["MMM", "terran_gen"]
                        and al_unit.unit_type == self.medivac_id
                    ):
                        ally_feats[i, 4] = al_unit.energy / max_cd  # energy
                    else:
                        ally_feats[i, 4] = (
                            al_unit.weapon_cooldown / max_cd
                        )  # cooldown

                    ind = 5
                    ally_feats[i, ind] = (al_x - center_x) / self.max_distance_x  # center X
                    ally_feats[i, ind+1] = (al_y - center_y) / self.max_distance_y  # center Y
                    ind += 2

                    if self.obs_all_health:
                        if not self.stochastic_health:
                            ally_feats[i, ind] = (
                                al_unit.health / al_unit.health_max
                            )  # health
                            ind += 1
                        elif self.observe_teammate_health:
                            ally_feats[i, ind] = self._compute_health(
                                agent_id=al_id, unit=al_unit
                            )
                            ind += 1
                        elif self.zero_pad_health:
                            ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (
                                al_unit.shield / max_shield
                            )  # shield
                            ind += 1
                    if self.stochastic_attack and self.observe_attack_probs:
                        ally_feats[i, ind] = self.agent_attack_probabilities[
                            al_id
                        ]
                        ind += 1
                    elif (
                        self.stochastic_attack
                        and self.zero_pad_stochastic_attack
                    ):
                        ind += 1

                    if self.stochastic_health and self.observe_teammate_health:
                        ally_feats[i, ind] = self.agent_health_levels[al_id]
                        ind += 1
                    elif self.stochastic_health and self.zero_pad_health:
                        ind += 1
                    if self.unit_type_bits > 0 and (
                        not self.replace_teammates
                        or self.observe_teammate_types
                    ):
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits
                    elif self.unit_type_bits > 0 and self.zero_pad_unit_types:
                        ind += self.unit_type_bits
                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                if not self.stochastic_health:
                    own_feats[ind] = unit.health / unit.health_max
                else:
                    own_feats[ind] = self._compute_health(agent_id, unit)
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            own_feats[ind] = (x - center_x) / self.max_distance_x  # center X
            own_feats[ind+1] = (y - center_y) / self.max_distance_y  # center Y
            ind += 2

            if self.stochastic_attack:
                own_feats[ind] = self.agent_attack_probabilities[agent_id]
                ind += 1
            if self.stochastic_health:
                own_feats[ind] = self.agent_health_levels[agent_id]
                ind += 1
            if self.obs_own_pos:
                own_feats[ind] = x / self.map_x
                own_feats[ind + 1] = y / self.map_y
                ind += 2
            if self.conic_fov:
                own_feats[ind : ind + 2] = self.fov_directions[agent_id]
                ind += 2
            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1

        state = np.concatenate((ally_feats.flatten(), 
                                enemy_feats.flatten(),
                                move_feats.flatten(),
                                own_feats.flatten()))

        # Agent id features
        if self.state_agent_id:
            agent_id_feats[agent_id] = 1.
            state = np.append(state, agent_id_feats.flatten())

        if self.state_timestep_number:
            state = np.append(state, self._episode_steps / self.episode_limit)

        return state

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [
            self.get_obs_agent(i, fully_observable=self.fully_observable)
            for i in range(self.n_agents)
        ]
        return agents_obs

    def get_capabilities_agent(self, agent_id):
        unit = self.get_unit_by_id(agent_id)
        cap_feats = np.zeros(self.get_cap_size(), dtype=np.float32)

        ind = 0
        if self.stochastic_attack:
            cap_feats[ind] = self.agent_attack_probabilities[agent_id]
            ind += 1
        if self.stochastic_health:
            cap_feats[ind] = self.agent_health_levels[agent_id]
            ind += 1
        if self.unit_type_bits > 0:
            type_id = self.get_unit_type_id(unit, True)
            cap_feats[ind + type_id] = 1

        return cap_feats

    def get_capabilities(self):
        """Returns all agent capabilities in a list."""
        agents_cap = [
            self.get_capabilities_agent(i) for i in range(self.n_agents)
        ]
        agents_cap = np.concatenate(agents_cap, axis=0).astype(np.float32)
        return agents_cap

    def get_state(self):
        """Returns the global state.
        NOTE: This function should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat

        state_dict = self.get_state_dict()

        state = np.append(
            state_dict["allies"].flatten(), state_dict["enemies"].flatten()
        )
        if "last_action" in state_dict:
            state = np.append(state, state_dict["last_action"].flatten())
        if "timestep" in state_dict:
            state = np.append(state, state_dict["timestep"])
        

        state = state.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(state_dict["allies"]))
            logging.debug("Enemy state {}".format(state_dict["enemies"]))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))

        return state

    def get_ally_num_attributes(self):
        return len(self.ally_state_attr_names) + len(
            self.capability_attr_names
        )

    def get_enemy_num_attributes(self):
        return len(self.enemy_state_attr_names)

    def get_state_dict(self):
        """Returns the global state as a dictionary.

        - allies: numpy array containing agents and their attributes
        - enemies: numpy array containing enemies and their attributes
        - last_action: numpy array of previous actions for each agent
        - timestep: current no. of steps divided by total no. of steps

        NOTE: This function should not be used during decentralised execution.
        """

        # number of features equals the number of attribute names
        nf_al = self.get_ally_num_attributes()
        nf_en = self.get_enemy_num_attributes()

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)
                if not self.stochastic_health:
                    ally_state[al_id, 0] = (
                        al_unit.health / al_unit.health_max
                    )  # health
                else:
                    ally_state[al_id, 0] = self._compute_health(al_id, al_unit)
                if (
                    self.map_type in ["MMM", "terran_gen"]
                    and al_unit.unit_type == self.medivac_id
                ):
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[al_id, 1] = (
                        al_unit.weapon_cooldown / max_cd
                    )  # cooldown
                ally_state[al_id, 2] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                ind = 4
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_state[al_id, ind] = (
                        al_unit.shield / max_shield
                    )  # shield
                    ind += 1

                if self.stochastic_attack:
                    ally_state[al_id, ind] = self.agent_attack_probabilities[
                        al_id
                    ]
                    ind += 1
                if self.stochastic_health:
                    ally_state[al_id, ind] = self.agent_health_levels[al_id]
                    ind += 1
                if self.conic_fov:
                    ally_state[al_id, ind : ind + 2] = self.fov_directions[
                        al_id
                    ]
                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_state[al_id, type_id - self.unit_type_bits] = 1

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = (
                    e_unit.health / e_unit.health_max
                )  # health
                enemy_state[e_id, 1] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, 3] = e_unit.shield / max_shield  # shield

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, type_id - self.unit_type_bits] = 1

        state = {"allies": ally_state, "enemies": enemy_state}

        if self.state_last_action:
            state["last_action"] = self.last_action
        if self.state_timestep_number:
            state["timestep"] = self._episode_steps / self.episode_limit

        return state

    def get_obs_enemy_feats_size(self):
        """Returns the dimensions of the matrix containing enemy features.
        Size is n_enemies x n_features.
        """
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        return self.n_enemies, nf_en

    def get_obs_ally_feats_size(self):
        """Returns the dimensions of the matrix containing ally features.
        Size is n_allies x n_features.
        """
        nf_al = 4
        nf_cap = self.get_obs_ally_capability_size()

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_actions

        return self.n_agents - 1, nf_al + nf_cap

    def get_obs_own_feats_size(self):
        """
        Returns the size of the vector containing the agents' own features.
        """
        own_feats = self.get_cap_size()
        if self.obs_own_health and self.obs_starcraft:
            own_feats += 1 + self.shield_bits_ally
        if self.conic_fov and self.obs_starcraft:
            own_feats += 2
        if self.obs_own_pos and self.obs_starcraft:
            own_feats += 2
        return own_feats

    def get_state_own_feats_size(self):
        own_feats = self.get_obs_own_feats_size() + 2
        return own_feats

    def get_obs_move_feats_size(self):
        """Returns the size of the vector containing the agents's movement-
        related features.
        """
        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        return move_feats

    def get_obs_ally_capability_size(self):
        """Returns the size of capabilities observed by teammates."""
        cap_feats = self.unit_type_bits
        if self.stochastic_attack and (
            self.zero_pad_stochastic_attack or self.observe_attack_probs
        ):
            cap_feats += 1
        if self.stochastic_health and (
            self.observe_teammate_health or self.zero_pad_health
        ):
            cap_feats += 1

        return cap_feats

    def get_cap_size(self):
        """Returns the size of the own capabilities of the agent."""
        cap_feats = 0
        if self.stochastic_attack:
            cap_feats += 1
        if self.stochastic_health:
            cap_feats += 1
        if self.unit_type_bits > 0:
            cap_feats += self.unit_type_bits

        return cap_feats

    def get_obs_size(self):
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()
        n_allies, n_ally_feats = self.get_obs_ally_feats_size()

        enemy_feats = n_enemies * n_enemy_feats
        ally_feats = n_allies * n_ally_feats

        all_feats = move_feats + enemy_feats + ally_feats + own_feats

        timestep_feats = 0
        if self.obs_timestep_number:
            timestep_feats = 1
            all_feats += timestep_feats

        return [all_feats, [n_allies, n_ally_feats], [n_enemies, n_enemy_feats], [1, move_feats], [1, own_feats+timestep_feats]]


    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size()[0] * self.n_agents
            
        own_feats = self.get_state_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemy_feats = self.get_enemy_num_attributes()
        n_ally_feats = self.get_ally_num_attributes()

        enemy_feats = self.n_enemies * n_enemy_feats
        n_allies = self.n_agents - 1
        ally_feats = n_allies * n_ally_feats

        all_feats = move_feats + enemy_feats + ally_feats + own_feats

        agent_id_feats = 0
        timestep_feats = 0

        if self.state_agent_id:
            agent_id_feats = self.n_agents
            all_feats += agent_id_feats

        if self.state_timestep_number:
            timestep_feats = 1
            all_feats += timestep_feats

        return [all_feats, [n_allies, n_ally_feats], [self.n_enemies, n_enemy_feats], [1, move_feats], [1, own_feats+agent_id_feats+timestep_feats]]

        
        # nf_al = self.get_ally_num_attributes()
        # nf_en = self.get_enemy_num_attributes()

        # enemy_state = self.n_enemies * nf_en
        # ally_state = self.n_agents * nf_al

        # size = enemy_state + ally_state

        # if self.state_last_action:
        #     size += self.n_agents * self.n_actions
        # if self.state_timestep_number:
        #     size += 1

        # return size

    def get_visibility_matrix(self):
        """Returns a boolean numpy array of dimensions
        (n_agents, n_agents + n_enemies) indicating which units
        are visible to each agent.
        """
        arr = np.zeros(
            (self.n_agents, self.n_agents + self.n_enemies),
            dtype=np.bool,
        )

        for agent_id in range(self.n_agents):
            current_agent = self.get_unit_by_id(agent_id)
            if current_agent.health > 0:  # it agent not dead
                x = current_agent.pos.x
                y = current_agent.pos.y
                sight_range = self.unit_sight_range(agent_id)

                # Enemies
                for e_id, e_unit in self.enemies.items():
                    e_x = e_unit.pos.x
                    e_y = e_unit.pos.y
                    dist = self.distance(x, y, e_x, e_y)

                    if dist < sight_range and e_unit.health > 0:
                        # visible and alive
                        arr[agent_id, self.n_agents + e_id] = 1

                # The matrix for allies is filled symmetrically
                al_ids = [
                    al_id for al_id in range(self.n_agents) if al_id > agent_id
                ]
                for _, al_id in enumerate(al_ids):
                    al_unit = self.get_unit_by_id(al_id)
                    al_x = al_unit.pos.x
                    al_y = al_unit.pos.y
                    dist = self.distance(x, y, al_x, al_y)

                    if dist < sight_range and al_unit.health > 0:
                        # visible and alive
                        arr[agent_id, al_id] = arr[al_id, agent_id] = 1

        return arr

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""

        if self.map_type == "protoss_gen":
            if unit.unit_type in (self.stalker_id, Protoss.Stalker):
                return 0
            if unit.unit_type in (self.zealot_id, Protoss.Zealot):
                return 1
            if unit.unit_type in (self.colossus_id, Protoss.Colossus):
                return 2
            raise AttributeError()
        if self.map_type == "terran_gen":
            if unit.unit_type in (self.marine_id, Terran.Marine):
                return 0
            if unit.unit_type in (self.marauder_id, Terran.Marauder):
                return 1
            if unit.unit_type in (self.medivac_id, Terran.Medivac):
                return 2
            raise AttributeError()

        if self.map_type == "zerg_gen":
            if unit.unit_type in (self.zergling_id, Zerg.Zergling):
                return 0
            if unit.unit_type in (self.hydralisk_id, Zerg.Hydralisk):
                return 1
            if unit.unit_type in (self.baneling_id, Zerg.Baneling):
                return 2
            raise AttributeError()

        # Old stuff
        if ally:  # use new SC2 unit types
            type_id = unit.unit_type - self._min_unit_type

        if self.map_type == "stalkers_and_zealots":
            # id(Stalker) = 74, id(Zealot) = 73
            type_id = unit.unit_type - 73
        elif self.map_type == "colossi_stalkers_zealots":
            # id(Stalker) = 74, id(Zealot) = 73, id(Colossus) = 4
            if unit.unit_type == 4:
                type_id = 0
            elif unit.unit_type == 74:
                type_id = 1
            else:
                type_id = 2
        elif self.map_type == "bane":
            if unit.unit_type == 9:
                type_id = 0
            else:
                type_id = 1
        elif self.map_type == "MMM":
            if unit.unit_type == 51:
                type_id = 0
            elif unit.unit_type == 48:
                type_id = 1
            else:
                type_id = 2

        return type_id

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            if self.conic_fov:
                avail_actions[6 : 6 + self.n_fov_actions] = [
                    1
                ] * self.n_fov_actions

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()
            if (
                self.map_type in ("MMM", "terran_gen")
                and unit.unit_type == self.medivac_id
            ):
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if t_unit.unit_type != self.medivac_id
                ]
            # should we only be able to target people in the cone?
            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(
                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                    )
                    can_shoot = (
                        dist <= shoot_range
                        if not self.conic_fov
                        else self.is_position_in_cone(
                            agent_id, t_unit.pos, range="shoot_range"
                        )
                    )
                    if can_shoot:
                        avail_actions[t_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def close(self):
        """Close StarCraft II."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self._sc2_proc:
            self._sc2_proc.close()

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def render(self, mode="human"):
        if self.renderer is None:
            from smacv2.env.starcraft2.render import StarCraft2Renderer

            self.renderer = StarCraft2Renderer(self, mode)
        assert (
            mode == self.renderer.mode
        ), "mode must be consistent across render calls"
        return self.renderer.render(mode)

    def _kill_units(self, unit_tags):
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=unit_tags))
        ]
        self._controller.debug(debug_command)

    def _kill_all_units(self):
        """Kill all units on the map. Steps controller and so can throw
        exceptions"""
        units = [unit.tag for unit in self._obs.observation.raw_data.units]
        self._kill_units(units)
        # check the units are dead
        units = len(self._obs.observation.raw_data.units)
        while len(self._obs.observation.raw_data.units) > 0:
            self._controller.step(2)
            self._obs = self._controller.observe()

    def _create_new_team(self, team, episode_config, ally):
        # unit_names = {
        #     self.id_to_unit_name_map[unit.unit_type]
        #     for unit in self.agents.values()
        # }
        # It's important to set the number of agents and enemies
        # because we use that to identify whether all the units have
        # been created successfully

        # TODO hardcoding init location. change this later for new maps
        if not self.random_start:
            if ally:
                init_pos = [sc_common.Point2D(x=8, y=16)] * self.n_agents
            else:
                init_pos = [sc_common.Point2D(x=24, y=16)] * self.n_enemies
        else:
            if ally:
                init_pos = [
                    sc_common.Point2D(
                        x=self.ally_start_positions[i][0],
                        y=self.ally_start_positions[i][1],
                    )
                    for i in range(self.ally_start_positions.shape[0])
                ]
            else:
                init_pos = [
                    sc_common.Point2D(
                        x=self.enemy_start_positions[i][0],
                        y=self.enemy_start_positions[i][1],
                    )
                    for i in range(self.enemy_start_positions.shape[0])
                ]
        debug_command = []
        for unit_id, unit in enumerate(team):
            unit_type = self._convert_unit_name_to_unit_type(unit, ally=ally)
            owner = 1 if ally else 2
            debug_command.append(
                d_pb.DebugCommand(
                    create_unit=d_pb.DebugCreateUnit(
                        unit_type=unit_type,
                        owner=owner,
                        pos=init_pos[unit_id],
                        quantity=1,
                    )
                )
            )
        self._controller.debug(debug_command)

    def _convert_unit_name_to_unit_type(self, unit_name, ally=True):
        if ally:
            return self.ally_unit_map[unit_name]
        else:
            return self.enemy_unit_map[unit_name]

    def init_units(self, ally_team, enemy_team, episode_config={}):
        """Initialise the units."""
        if ally_team and enemy_team:
            # can use any value for min unit type because
            # it is hardcoded based on the version
            self._init_ally_unit_types(0)
            self._create_new_team(ally_team, episode_config, ally=True)
            self._create_new_team(enemy_team, episode_config, ally=False)
            try:
                self._controller.step(1)
                self._obs = self._controller.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset(episode_config=episode_config)
        while True:
            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}

            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 1
            ]
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug:
                    logging.debug(
                        "Unit {} is {}, x = {}, y = {}".format(
                            len(self.agents),
                            self.agents[i].unit_type,
                            self.agents[i].pos.x,
                            self.agents[i].pos.y,
                        )
                    )

            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2:
                    self.enemies[len(self.enemies)] = unit
                    if self._episode_count == 0:
                        self.max_reward += unit.health_max + unit.shield_max

            if self._episode_count == 0 and not ally_team:
                min_unit_type = min(
                    unit.unit_type for unit in self.agents.values()
                )
                self._init_ally_unit_types(min_unit_type)

            all_agents_created = len(self.agents) == self.n_agents
            all_enemies_created = len(self.enemies) == self.n_enemies

            self._unit_types = [
                unit.unit_type for unit in ally_units_sorted
            ] + [
                unit.unit_type
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 2
            ]

            # TODO move this to the start
            if all_agents_created and all_enemies_created:  # all good
                return

            try:
                self._controller.step(1)
                self._obs = self._controller.observe()
            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset(episode_config=episode_config)

    def get_unit_types(self):
        if self._unit_types is None:
            warn(
                "unit types have not been initialized yet, please call"
                "env.reset() to populate this and call t1286he method again."
            )

        return self._unit_types

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if (
            n_ally_alive == 0
            and n_enemy_alive > 0
            or self.only_medivac_left(ally=True)
        ):
            return -1  # lost
        if (
            n_ally_alive > 0
            and n_enemy_alive == 0
            or self.only_medivac_left(ally=False)
        ):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def _register_unit_mapping(self, unit_name, unit_type_id):
        self.id_to_unit_name_map[unit_type_id] = unit_name
        self.unit_name_to_id_map[unit_name] = unit_type_id

    def _init_ally_unit_types(self, min_unit_type):
        """Initialise ally unit types. Should be called once from the
        init_units function.
        """

        self._min_unit_type = min_unit_type

        if "10gen_" in self.map_name:
            num_rl_units = 9
            self._min_unit_type = (
                len(self._controller.data().units) - num_rl_units
            )

            self.baneling_id = self._min_unit_type
            self.colossus_id = self._min_unit_type + 1
            self.hydralisk_id = self._min_unit_type + 2
            self.marauder_id = self._min_unit_type + 3
            self.marine_id = self._min_unit_type + 4
            self.medivac_id = self._min_unit_type + 5
            self.stalker_id = self._min_unit_type + 6
            self.zealot_id = self._min_unit_type + 7
            self.zergling_id = self._min_unit_type + 8

            self.ally_unit_map = {
                "baneling": self.baneling_id,
                "colossus": self.colossus_id,
                "hydralisk": self.hydralisk_id,
                "marauder": self.marauder_id,
                "marine": self.marine_id,
                "medivac": self.medivac_id,
                "stalker": self.stalker_id,
                "zealot": self.zealot_id,
                "zergling": self.zergling_id,
            }
            self.enemy_unit_map = {
                "baneling": Zerg.Baneling,
                "colossus": Protoss.Colossus,
                "hydralisk": Zerg.Hydralisk,
                "marauder": Terran.Marauder,
                "marine": Terran.Marine,
                "medivac": Terran.Medivac,
                "stalker": Protoss.Stalker,
                "zealot": Protoss.Zealot,
                "zergling": Zerg.Zergling,
            }

        else:
            if self.map_type == "marines":
                self.marine_id = min_unit_type
                self._register_unit_mapping("marine", min_unit_type)
            elif self.map_type == "stalkers_and_zealots":
                self.stalker_id = min_unit_type
                self._register_unit_mapping("stalker", min_unit_type)
                self.zealot_id = min_unit_type + 1
                self._register_unit_mapping("zealot", min_unit_type + 1)
            elif self.map_type == "colossi_stalkers_zealots":
                self.colossus_id = min_unit_type
                self._register_unit_mapping("colossus", min_unit_type)
                self.stalker_id = min_unit_type + 1
                self._register_unit_mapping("stalker", min_unit_type + 1)
                self.zealot_id = min_unit_type + 2
                self._register_unit_mapping("zealot", min_unit_type + 2)
            elif self.map_type == "MMM":
                self.marauder_id = min_unit_type
                self._register_unit_mapping("marauder", min_unit_type)
                self.marine_id = min_unit_type + 1
                self._register_unit_mapping("marine", min_unit_type + 1)
                self.medivac_id = min_unit_type + 2
                self._register_unit_mapping("medivac", min_unit_type + 2)
            elif self.map_type == "zealots":
                self.zealot_id = min_unit_type
                self._register_unit_mapping("zealot", min_unit_type)
            elif self.map_type == "hydralisks":
                self.hydralisk_id = min_unit_type
                self._register_unit_mapping("hydralisk", min_unit_type)
            elif self.map_type == "stalkers":
                self.stalker_id = min_unit_type
                self._register_unit_mapping("stalker", min_unit_type)
            elif self.map_type == "colossus":
                self.colossus_id = min_unit_type
                self._register_unit_mapping("colossus", min_unit_type)
            elif self.map_type == "bane":
                self.baneling_id = min_unit_type
                self._register_unit_mapping("baneling", min_unit_type)
                self.zergling_id = min_unit_type + 1
                self._register_unit_mapping("zergling", min_unit_type + 1)

    def only_medivac_left(self, ally):
        """Check if only Medivac units are left."""
        if self.map_type != "MMM" and self.map_type != "terran_gen":
            return False

        if ally:
            units_alive = [
                a
                for a in self.agents.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [
                a
                for a in self.enemies.values()
                if (a.health > 0 and a.unit_type != Terran.Medivac)
            ]
            if len(units_alive) == 0:
                return True
            return False

    def get_unit_by_id(self, a_id):
        """Get unit by ID."""
        return self.agents[a_id]

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats

    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["agent_features"] = (
            self.ally_state_attr_names + self.capability_attr_names
        )
        env_info["enemy_features"] = self.enemy_state_attr_names
        return env_info