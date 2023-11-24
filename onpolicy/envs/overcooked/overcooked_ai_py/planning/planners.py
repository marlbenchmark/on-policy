import itertools
import os
import pickle
import time

import numpy as np

from onpolicy.envs.overcooked.overcooked_ai_py.data.planners import (
    PLANNERS_DIR,
    load_saved_action_manager,
    load_saved_motion_planner,
)
from onpolicy.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from onpolicy.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
    EVENT_TYPES,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
)
from onpolicy.envs.overcooked.overcooked_ai_py.planning.search import Graph, NotConnectedError
from onpolicy.envs.overcooked.overcooked_ai_py.utils import manhattan_distance

# Run planning logic with additional checks and
# computation to prevent or identify possible minor errors
SAFE_RUN = False

NO_COUNTERS_PARAMS = {
    "start_orientations": False,
    "wait_allowed": False,
    "counter_goals": [],
    "counter_drop": [],
    "counter_pickup": [],
    "same_motion_goals": True,
}

NO_COUNTERS_START_OR_PARAMS = {
    "start_orientations": True,
    "wait_allowed": False,
    "counter_goals": [],
    "counter_drop": [],
    "counter_pickup": [],
    "same_motion_goals": True,
}


class MotionPlanner(object):
    """A planner that computes optimal plans for a single agent to
    arrive at goal positions and orientations in an OvercookedGridworld.

    Args:
        mdp (OvercookedGridworld): gridworld of interest
        counter_goals (list): list of positions of counters we will consider
                              as valid motion goals
    """

    def __init__(self, mdp, counter_goals=[]):
        self.mdp = mdp

        # If positions facing counters should be
        # allowed as motion goals
        self.counter_goals = counter_goals

        # Graph problem that solves shortest path problem
        # between any position & orientation start-goal pair
        self.graph_problem = self._graph_from_grid()
        self.motion_goals_for_pos = self._get_goal_dict()

        self.all_plans = self._populate_all_plans()

    def save_to_file(self, filename):
        with open(filename, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_file(filename):
        return load_saved_motion_planner(filename)

    @staticmethod
    def from_pickle_or_compute(
        mdp,
        counter_goals,
        custom_filename=None,
        force_compute=False,
        info=False,
    ):
        assert isinstance(mdp, OvercookedGridworld)

        filename = (
            custom_filename
            if custom_filename is not None
            else mdp.layout_name + "_mp.pkl"
        )

        if force_compute:
            return MotionPlanner.compute_mp(filename, mdp, counter_goals)

        try:
            mp = MotionPlanner.from_file(filename)

            if mp.counter_goals != counter_goals or mp.mdp != mdp:
                if info:
                    print(
                        "motion planner with different counter goal or mdp found, computing from scratch"
                    )
                return MotionPlanner.compute_mp(filename, mdp, counter_goals)

        except (
            FileNotFoundError,
            ModuleNotFoundError,
            EOFError,
            AttributeError,
        ) as e:
            if info:
                print("Recomputing motion planner due to:", e)
            return MotionPlanner.compute_mp(filename, mdp, counter_goals)

        if info:
            print(
                "Loaded MotionPlanner from {}".format(
                    os.path.join(PLANNERS_DIR, filename)
                )
            )
        return mp

    @staticmethod
    def compute_mp(filename, mdp, counter_goals):
        final_filepath = os.path.join(PLANNERS_DIR, filename)
        print(
            "Computing MotionPlanner to be saved in {}".format(final_filepath)
        )
        start_time = time.time()
        mp = MotionPlanner(mdp, counter_goals)
        print(
            "It took {} seconds to create mp".format(time.time() - start_time)
        )
        mp.save_to_file(final_filepath)
        return mp

    def get_plan(self, start_pos_and_or, goal_pos_and_or):
        """
        Returns pre-computed plan from initial agent position
        and orientation to a goal position and orientation.

        Args:
            start_pos_and_or (tuple): starting (pos, or) tuple
            goal_pos_and_or (tuple): goal (pos, or) tuple
        """
        plan_key = (start_pos_and_or, goal_pos_and_or)
        action_plan, pos_and_or_path, plan_cost = self.all_plans[plan_key]
        return action_plan, pos_and_or_path, plan_cost

    def get_gridworld_distance(self, start_pos_and_or, goal_pos_and_or):
        """Number of actions necessary to go from starting position
        and orientations to goal position and orientation (not including
        interaction action)"""
        assert self.is_valid_motion_start_goal_pair(
            start_pos_and_or, goal_pos_and_or
        ), "Goal position and orientation were not a valid motion goal"
        _, _, plan_cost = self.get_plan(start_pos_and_or, goal_pos_and_or)
        # Removing interaction cost
        return plan_cost - 1

    def get_gridworld_pos_distance(self, pos1, pos2):
        """Minimum (over possible orientations) number of actions necessary
        to go from starting position to goal position (not including
        interaction action)."""
        # NOTE: currently unused, pretty bad code. If used in future, clean up
        min_cost = np.Inf
        for d1, d2 in itertools.product(Direction.ALL_DIRECTIONS, repeat=2):
            start = (pos1, d1)
            end = (pos2, d2)
            if self.is_valid_motion_start_goal_pair(start, end):
                plan_cost = self.get_gridworld_distance(start, end)
                if plan_cost < min_cost:
                    min_cost = plan_cost
        return min_cost

    def _populate_all_plans(self):
        """Pre-computes all valid plans from any valid pos_or to any valid motion_goal"""
        all_plans = {}
        valid_pos_and_ors = (
            self.mdp.get_valid_player_positions_and_orientations()
        )
        valid_motion_goals = filter(
            self.is_valid_motion_goal, valid_pos_and_ors
        )
        for start_motion_state, goal_motion_state in itertools.product(
            valid_pos_and_ors, valid_motion_goals
        ):
            if not self.is_valid_motion_start_goal_pair(
                start_motion_state, goal_motion_state
            ):
                continue
            action_plan, pos_and_or_path, plan_cost = self._compute_plan(
                start_motion_state, goal_motion_state
            )
            plan_key = (start_motion_state, goal_motion_state)
            all_plans[plan_key] = (action_plan, pos_and_or_path, plan_cost)
        return all_plans

    def is_valid_motion_start_goal_pair(
        self, start_pos_and_or, goal_pos_and_or
    ):
        if not self.is_valid_motion_goal(goal_pos_and_or):
            return False
        # the valid motion start goal needs to be in the same connected component
        if not self.positions_are_connected(start_pos_and_or, goal_pos_and_or):
            return False
        return True

    def is_valid_motion_goal(self, goal_pos_and_or):
        """Checks that desired single-agent goal state (position and orientation)
        is reachable and is facing a terrain feature"""
        goal_position, goal_orientation = goal_pos_and_or
        if goal_position not in self.mdp.get_valid_player_positions():
            return False

        # Restricting goals to be facing a terrain feature
        pos_of_facing_terrain = Action.move_in_direction(
            goal_position, goal_orientation
        )
        facing_terrain_type = self.mdp.get_terrain_type_at_pos(
            pos_of_facing_terrain
        )
        if facing_terrain_type == " " or (
            facing_terrain_type == "X"
            and pos_of_facing_terrain not in self.counter_goals
        ):
            return False
        return True

    def _compute_plan(self, start_motion_state, goal_motion_state):
        """Computes optimal action plan for single agent movement

        Args:
            start_motion_state (tuple): starting positions and orientations
            goal_motion_state (tuple): goal positions and orientations
        """
        assert self.is_valid_motion_start_goal_pair(
            start_motion_state, goal_motion_state
        )
        positions_plan = self._get_position_plan_from_graph(
            start_motion_state, goal_motion_state
        )
        (
            action_plan,
            pos_and_or_path,
            plan_length,
        ) = self.action_plan_from_positions(
            positions_plan, start_motion_state, goal_motion_state
        )
        return action_plan, pos_and_or_path, plan_length

    def positions_are_connected(self, start_pos_and_or, goal_pos_and_or):
        return self.graph_problem.are_in_same_cc(
            start_pos_and_or, goal_pos_and_or
        )

    def _get_position_plan_from_graph(self, start_node, end_node):
        """Recovers positions to be reached by agent after the start node to reach the end node"""
        node_path = self.graph_problem.get_node_path(start_node, end_node)
        assert node_path[0] == start_node and node_path[-1] == end_node
        positions_plan = [state_node[0] for state_node in node_path[1:]]
        return positions_plan

    def action_plan_from_positions(
        self, position_list, start_motion_state, goal_motion_state
    ):
        """
        Recovers an action plan reaches the goal motion position and orientation, and executes
        and interact action.

        Args:
            position_list (list): list of positions to be reached after the starting position
                                  (does not include starting position, but includes ending position)
            start_motion_state (tuple): starting position and orientation
            goal_motion_state (tuple): goal position and orientation

        Returns:
            action_plan (list): list of actions to reach goal state
            pos_and_or_path (list): list of (pos, or) pairs visited during plan execution
                                    (not including start, but including goal)
        """
        goal_position, goal_orientation = goal_motion_state
        action_plan, pos_and_or_path = [], []
        position_to_go = list(position_list)
        curr_pos, curr_or = start_motion_state

        # Get agent to goal position
        while position_to_go and curr_pos != goal_position:
            next_pos = position_to_go.pop(0)
            action = Action.determine_action_for_change_in_pos(
                curr_pos, next_pos
            )
            action_plan.append(action)
            curr_or = action if action != Action.STAY else curr_or
            pos_and_or_path.append((next_pos, curr_or))
            curr_pos = next_pos

        # Fix agent orientation if necessary
        if curr_or != goal_orientation:
            new_pos, _ = self.mdp._move_if_direction(
                curr_pos, curr_or, goal_orientation
            )
            assert new_pos == goal_position
            action_plan.append(goal_orientation)
            pos_and_or_path.append((goal_position, goal_orientation))

        # Add interact action
        action_plan.append(Action.INTERACT)
        pos_and_or_path.append((goal_position, goal_orientation))

        return action_plan, pos_and_or_path, len(action_plan)

    def _graph_from_grid(self):
        """Creates a graph adjacency matrix from an Overcooked MDP class."""
        state_decoder = {}
        for state_index, motion_state in enumerate(
            self.mdp.get_valid_player_positions_and_orientations()
        ):
            state_decoder[state_index] = motion_state

        pos_encoder = {
            motion_state: state_index
            for state_index, motion_state in state_decoder.items()
        }
        num_graph_nodes = len(state_decoder)

        adjacency_matrix = np.zeros((num_graph_nodes, num_graph_nodes))
        for state_index, start_motion_state in state_decoder.items():
            for (
                action,
                successor_motion_state,
            ) in self._get_valid_successor_motion_states(start_motion_state):
                adj_pos_index = pos_encoder[successor_motion_state]
                adjacency_matrix[state_index][
                    adj_pos_index
                ] = self._graph_action_cost(action)

        return Graph(adjacency_matrix, pos_encoder, state_decoder)

    def _graph_action_cost(self, action):
        """Returns cost of a single-agent action"""
        assert action in Action.ALL_ACTIONS
        return 1

    def _get_valid_successor_motion_states(self, start_motion_state):
        """Get valid motion states one action away from the starting motion state."""
        start_position, start_orientation = start_motion_state
        return [
            (
                action,
                self.mdp._move_if_direction(
                    start_position, start_orientation, action
                ),
            )
            for action in Action.ALL_ACTIONS
        ]

    def min_cost_between_features(
        self, pos_list1, pos_list2, manhattan_if_fail=False
    ):
        """
        Determines the minimum number of timesteps necessary for a player to go from any
        terrain feature in list1 to any feature in list2 and perform an interact action
        """
        min_dist = np.Inf
        min_manhattan = np.Inf
        for pos1, pos2 in itertools.product(pos_list1, pos_list2):
            for mg1, mg2 in itertools.product(
                self.motion_goals_for_pos[pos1],
                self.motion_goals_for_pos[pos2],
            ):
                if not self.is_valid_motion_start_goal_pair(mg1, mg2):
                    if manhattan_if_fail:
                        pos0, pos1 = mg1[0], mg2[0]
                        curr_man_dist = manhattan_distance(pos0, pos1)
                        if curr_man_dist < min_manhattan:
                            min_manhattan = curr_man_dist
                    continue
                curr_dist = self.get_gridworld_distance(mg1, mg2)
                if curr_dist < min_dist:
                    min_dist = curr_dist

        # +1 to account for interaction action
        if manhattan_if_fail and min_dist == np.Inf:
            min_dist = min_manhattan
        min_cost = min_dist + 1
        return min_cost

    def min_cost_to_feature(
        self,
        start_pos_and_or,
        feature_pos_list,
        with_argmin=False,
        debug=False,
    ):
        """
        Determines the minimum number of timesteps necessary for a player to go from the starting
        position and orientation to any feature in feature_pos_list and perform an interact action
        """
        start_pos = start_pos_and_or[0]
        assert self.mdp.get_terrain_type_at_pos(start_pos) != "X"
        min_dist = np.Inf
        best_feature = None
        for feature_pos in feature_pos_list:
            for feature_goal in self.motion_goals_for_pos[feature_pos]:
                if not self.is_valid_motion_start_goal_pair(
                    start_pos_and_or, feature_goal
                ):
                    continue
                curr_dist = self.get_gridworld_distance(
                    start_pos_and_or, feature_goal
                )
                if curr_dist < min_dist:
                    best_feature = feature_pos
                    min_dist = curr_dist
        # +1 to account for interaction action
        min_cost = min_dist + 1
        if with_argmin:
            # assert best_feature is not None, "{} vs {}".format(start_pos_and_or, feature_pos_list)
            return min_cost, best_feature
        return min_cost

    def _get_goal_dict(self):
        """Creates a dictionary of all possible goal states for all possible
        terrain features that the agent might want to interact with."""
        terrain_feature_locations = []
        for terrain_type, pos_list in self.mdp.terrain_pos_dict.items():
            if terrain_type != " ":
                terrain_feature_locations += pos_list
        return {
            feature_pos: self._get_possible_motion_goals_for_feature(
                feature_pos
            )
            for feature_pos in terrain_feature_locations
        }

    def _get_possible_motion_goals_for_feature(self, goal_pos):
        """Returns a list of possible goal positions (and orientations)
        that could be used for motion planning to get to goal_pos"""
        goals = []
        valid_positions = self.mdp.get_valid_player_positions()
        for d in Direction.ALL_DIRECTIONS:
            adjacent_pos = Action.move_in_direction(goal_pos, d)
            if adjacent_pos in valid_positions:
                goal_orientation = Direction.OPPOSITE_DIRECTIONS[d]
                motion_goal = (adjacent_pos, goal_orientation)
                goals.append(motion_goal)
        return goals


class JointMotionPlanner(object):
    """A planner that computes optimal plans for a two agents to
    arrive at goal positions and orientations in a OvercookedGridworld.

    Args:
        mdp (OvercookedGridworld): gridworld of interest
    """

    def __init__(self, mdp, params, debug=False):
        self.mdp = mdp

        # Whether starting orientations should be accounted for
        # when solving all motion problems
        # (increases number of plans by a factor of 4)
        # but removes additional fudge factor <= 1 for each
        # joint motion plan
        self.debug = debug
        self.start_orientations = params["start_orientations"]

        # Enable both agents to have the same motion goal
        self.same_motion_goals = params["same_motion_goals"]

        # Single agent motion planner
        self.motion_planner = MotionPlanner(
            mdp, counter_goals=params["counter_goals"]
        )

        # Graph problem that returns optimal paths from
        # starting positions to goal positions (without
        # accounting for orientations)
        self.joint_graph_problem = self._joint_graph_from_grid()
        self.all_plans = self._populate_all_plans()

    def get_low_level_action_plan(self, start_jm_state, goal_jm_state):
        """
        Returns pre-computed plan from initial joint motion state
        to a goal joint motion state.

        Args:
            start_jm_state (tuple): starting pos & orients ((pos1, or1), (pos2, or2))
            goal_jm_state (tuple): goal pos & orients ((pos1, or1), (pos2, or2))

        Returns:
            joint_action_plan (list): joint actions to be executed to reach end_jm_state
            end_jm_state (tuple): the pair of (pos, or) tuples corresponding
                to the ending timestep (this will usually be different from
                goal_jm_state, as one agent will end before other).
            plan_lengths (tuple): lengths for each agent's plan
        """
        assert self.is_valid_joint_motion_pair(
            start_jm_state, goal_jm_state
        ), "start: {} \t end: {} was not a valid motion goal pair".format(
            start_jm_state, goal_jm_state
        )

        if self.start_orientations:
            plan_key = (start_jm_state, goal_jm_state)
        else:
            starting_positions = tuple(
                player_pos_and_or[0] for player_pos_and_or in start_jm_state
            )
            goal_positions = tuple(
                player_pos_and_or[0] for player_pos_and_or in goal_jm_state
            )
            # If beginning positions are equal to end positions, the pre-stored
            # plan (not dependent on initial positions) will likely return a
            # wrong answer, so we compute it from scratch.
            #
            # This is because we only compute plans with starting orientations
            # (North, North), so if one of the two agents starts at location X
            # with orientation East it's goal is to get to location X with
            # orientation North. The precomputed plan will just tell that agent
            # that it is already at the goal, so no actions (or just 'interact')
            # are necessary.
            #
            # We also compute the plan for any shared motion goal with SAFE_RUN,
            # as there are some minor edge cases that could not be accounted for
            # but I expect should not make a difference in nearly all scenarios
            if any(
                [s == g for s, g in zip(starting_positions, goal_positions)]
            ) or (SAFE_RUN and goal_positions[0] == goal_positions[1]):
                return self._obtain_plan(start_jm_state, goal_jm_state)

            dummy_orientation = Direction.NORTH
            dummy_start_jm_state = tuple(
                (pos, dummy_orientation) for pos in starting_positions
            )
            plan_key = (dummy_start_jm_state, goal_jm_state)

        if plan_key not in self.all_plans:
            num_player = len(goal_jm_state)
            return [], None, [np.inf] * num_player
        joint_action_plan, end_jm_state, plan_lengths = self.all_plans[
            plan_key
        ]
        return joint_action_plan, end_jm_state, plan_lengths

    def _populate_all_plans(self):
        """Pre-compute all valid plans"""
        all_plans = {}

        # Joint states are valid if players are not in same location
        if self.start_orientations:
            valid_joint_start_states = (
                self.mdp.get_valid_joint_player_positions_and_orientations()
            )
        else:
            valid_joint_start_states = (
                self.mdp.get_valid_joint_player_positions()
            )

        valid_player_states = (
            self.mdp.get_valid_player_positions_and_orientations()
        )
        possible_joint_goal_states = list(
            itertools.product(valid_player_states, repeat=2)
        )
        valid_joint_goal_states = list(
            filter(self.is_valid_joint_motion_goal, possible_joint_goal_states)
        )

        if self.debug:
            print(
                "Number of plans being pre-calculated: ",
                len(valid_joint_start_states) * len(valid_joint_goal_states),
            )
        for joint_start_state, joint_goal_state in itertools.product(
            valid_joint_start_states, valid_joint_goal_states
        ):
            # If orientations not present, joint_start_state just includes positions.
            if not self.start_orientations:
                dummy_orientation = Direction.NORTH
                joint_start_state = tuple(
                    (pos, dummy_orientation) for pos in joint_start_state
                )

            # If either start-end states are not connected, skip to next plan
            if not self.is_valid_jm_start_goal_pair(
                joint_start_state, joint_goal_state
            ):
                continue

            # Note: we might fail to get the plan, just due to the nature of the layouts
            joint_action_list, end_statuses, plan_lengths = self._obtain_plan(
                joint_start_state, joint_goal_state
            )
            if end_statuses is None:
                continue
            plan_key = (joint_start_state, joint_goal_state)
            all_plans[plan_key] = (
                joint_action_list,
                end_statuses,
                plan_lengths,
            )
        return all_plans

    def is_valid_jm_start_goal_pair(self, joint_start_state, joint_goal_state):
        """Checks if the combination of joint start state and joint goal state is valid"""
        if not self.is_valid_joint_motion_goal(joint_goal_state):
            return False
        check_valid_fn = self.motion_planner.is_valid_motion_start_goal_pair
        return all(
            [
                check_valid_fn(joint_start_state[i], joint_goal_state[i])
                for i in range(2)
            ]
        )

    def _obtain_plan(self, joint_start_state, joint_goal_state):
        """Either use motion planner or actually compute a joint plan"""
        # Try using MotionPlanner plans and join them together
        (
            action_plans,
            pos_and_or_paths,
            plan_lengths,
        ) = self._get_plans_from_single_planner(
            joint_start_state, joint_goal_state
        )

        # Check if individual plans conflict
        have_conflict = self.plans_have_conflict(
            joint_start_state, joint_goal_state, pos_and_or_paths, plan_lengths
        )

        # If there is no conflict, the joint plan computed by joining single agent MotionPlanner plans is optimal
        if not have_conflict:
            (
                joint_action_plan,
                end_pos_and_orientations,
            ) = self._join_single_agent_action_plans(
                joint_start_state,
                action_plans,
                pos_and_or_paths,
                min(plan_lengths),
            )
            return joint_action_plan, end_pos_and_orientations, plan_lengths

        # If there is a conflict in the single motion plan and the agents have the same goal,
        # the graph problem can't be used either as it can't handle same goal state: we compute
        # manually what the best way to handle the conflict is
        elif self._agents_are_in_same_position(joint_goal_state):
            (
                joint_action_plan,
                end_pos_and_orientations,
                plan_lengths,
            ) = self._handle_path_conflict_with_same_goal(
                joint_start_state,
                joint_goal_state,
                action_plans,
                pos_and_or_paths,
            )
            return joint_action_plan, end_pos_and_orientations, plan_lengths

        # If there is a conflict, and the agents have different goals, we can use solve the joint graph problem
        return self._compute_plan_from_joint_graph(
            joint_start_state, joint_goal_state
        )

    def _get_plans_from_single_planner(
        self, joint_start_state, joint_goal_state
    ):
        """
        Get individual action plans for each agent from the MotionPlanner to get each agent
        independently to their goal state. NOTE: these plans might conflict
        """
        single_agent_motion_plans = [
            self.motion_planner.get_plan(start, goal)
            for start, goal in zip(joint_start_state, joint_goal_state)
        ]
        action_plans, pos_and_or_paths = [], []
        for action_plan, pos_and_or_path, _ in single_agent_motion_plans:
            action_plans.append(action_plan)
            pos_and_or_paths.append(pos_and_or_path)
        plan_lengths = tuple(len(p) for p in action_plans)
        assert all(
            [plan_lengths[i] == len(pos_and_or_paths[i]) for i in range(2)]
        )
        return action_plans, pos_and_or_paths, plan_lengths

    def plans_have_conflict(
        self,
        joint_start_state,
        joint_goal_state,
        pos_and_or_paths,
        plan_lengths,
    ):
        """Check if the sequence of pos_and_or_paths for the two agents conflict"""
        min_length = min(plan_lengths)
        prev_positions = tuple(s[0] for s in joint_start_state)
        for t in range(min_length):
            curr_pos_or0, curr_pos_or1 = (
                pos_and_or_paths[0][t],
                pos_and_or_paths[1][t],
            )
            curr_positions = (curr_pos_or0[0], curr_pos_or1[0])
            if self.mdp.is_transition_collision(
                prev_positions, curr_positions
            ):
                return True
            prev_positions = curr_positions
        return False

    def _join_single_agent_action_plans(
        self, joint_start_state, action_plans, pos_and_or_paths, finishing_time
    ):
        """Returns the joint action plan and end joint state obtained by joining the individual action plans"""
        assert finishing_time > 0
        end_joint_state = (
            pos_and_or_paths[0][finishing_time - 1],
            pos_and_or_paths[1][finishing_time - 1],
        )
        joint_action_plan = list(
            zip(
                *[
                    action_plans[0][:finishing_time],
                    action_plans[1][:finishing_time],
                ]
            )
        )
        return joint_action_plan, end_joint_state

    def _handle_path_conflict_with_same_goal(
        self,
        joint_start_state,
        joint_goal_state,
        action_plans,
        pos_and_or_paths,
    ):
        """Assumes that optimal path in case two agents have the same goal and their paths conflict
        is for one of the agents to wait. Checks resulting plans if either agent waits, and selects the
        shortest cost among the two."""

        (
            joint_plan0,
            end_pos_and_or0,
            plan_lengths0,
        ) = self._handle_conflict_with_same_goal_idx(
            joint_start_state,
            joint_goal_state,
            action_plans,
            pos_and_or_paths,
            wait_agent_idx=0,
        )

        (
            joint_plan1,
            end_pos_and_or1,
            plan_lengths1,
        ) = self._handle_conflict_with_same_goal_idx(
            joint_start_state,
            joint_goal_state,
            action_plans,
            pos_and_or_paths,
            wait_agent_idx=1,
        )

        assert any([joint_plan0 is not None, joint_plan1 is not None])

        best_plan_idx = np.argmin([min(plan_lengths0), min(plan_lengths1)])
        solutions = [
            (joint_plan0, end_pos_and_or0, plan_lengths0),
            (joint_plan1, end_pos_and_or1, plan_lengths1),
        ]
        return solutions[best_plan_idx]

    def _handle_conflict_with_same_goal_idx(
        self,
        joint_start_state,
        joint_goal_state,
        action_plans,
        pos_and_or_paths,
        wait_agent_idx,
    ):
        """
        Determines what is the best joint plan if whenether there is a conflict between the two agents' actions,
        the agent with index `wait_agent_idx` waits one turn.

        If the agent that is assigned to wait is "in front" of the non-waiting agent, this could result
        in an endless conflict. In this case, we return infinite finishing times.
        """
        idx0, idx1 = 0, 0
        prev_positions = [
            start_pos_and_or[0] for start_pos_and_or in joint_start_state
        ]
        curr_pos_or0, curr_pos_or1 = joint_start_state

        agent0_plan_original, agent1_plan_original = action_plans

        joint_plan = []
        # While either agent hasn't finished their plan
        while idx0 != len(agent0_plan_original) and idx1 != len(
            agent1_plan_original
        ):
            next_pos_or0, next_pos_or1 = (
                pos_and_or_paths[0][idx0],
                pos_and_or_paths[1][idx1],
            )
            next_positions = (next_pos_or0[0], next_pos_or1[0])

            # If agents collide, let the waiting agent wait and the non-waiting
            # agent take a step
            if self.mdp.is_transition_collision(
                prev_positions, next_positions
            ):
                if wait_agent_idx == 0:
                    curr_pos_or0 = (
                        curr_pos_or0  # Agent 0 will wait, stays the same
                    )
                    curr_pos_or1 = next_pos_or1
                    curr_joint_action = [
                        Action.STAY,
                        agent1_plan_original[idx1],
                    ]
                    idx1 += 1
                elif wait_agent_idx == 1:
                    curr_pos_or0 = next_pos_or0
                    curr_pos_or1 = (
                        curr_pos_or1  # Agent 1 will wait, stays the same
                    )
                    curr_joint_action = [
                        agent0_plan_original[idx0],
                        Action.STAY,
                    ]
                    idx0 += 1

                curr_positions = (curr_pos_or0[0], curr_pos_or1[0])

                # If one agent waiting causes other to crash into it, return None
                if self._agents_are_in_same_position(
                    (curr_pos_or0, curr_pos_or1)
                ):
                    return None, None, [np.Inf, np.Inf]

            else:
                curr_pos_or0, curr_pos_or1 = next_pos_or0, next_pos_or1
                curr_positions = next_positions
                curr_joint_action = [
                    agent0_plan_original[idx0],
                    agent1_plan_original[idx1],
                ]
                idx0 += 1
                idx1 += 1

            joint_plan.append(curr_joint_action)
            prev_positions = curr_positions

        assert idx0 != idx1, "No conflict found"

        end_pos_and_or = (curr_pos_or0, curr_pos_or1)
        finishing_times = (
            (np.Inf, idx1) if wait_agent_idx == 0 else (idx0, np.Inf)
        )
        return joint_plan, end_pos_and_or, finishing_times

    def is_valid_joint_motion_goal(self, joint_goal_state):
        """Checks whether the goal joint positions and orientations are a valid goal"""
        if not self.same_motion_goals and self._agents_are_in_same_position(
            joint_goal_state
        ):
            return False
        multi_cc_map = (
            len(self.motion_planner.graph_problem.connected_components) > 1
        )
        players_in_same_cc = self.motion_planner.graph_problem.are_in_same_cc(
            joint_goal_state[0], joint_goal_state[1]
        )
        if multi_cc_map and players_in_same_cc:
            return False
        return all(
            [
                self.motion_planner.is_valid_motion_goal(player_state)
                for player_state in joint_goal_state
            ]
        )

    def is_valid_joint_motion_pair(self, joint_start_state, joint_goal_state):
        if not self.is_valid_joint_motion_goal(joint_goal_state):
            return False
        return all(
            [
                self.motion_planner.is_valid_motion_start_goal_pair(
                    joint_start_state[i], joint_goal_state[i]
                )
                for i in range(2)
            ]
        )

    def _agents_are_in_same_position(self, joint_motion_state):
        agent_positions = [
            player_pos_and_or[0] for player_pos_and_or in joint_motion_state
        ]
        return len(agent_positions) != len(set(agent_positions))

    def _compute_plan_from_joint_graph(
        self, joint_start_state, joint_goal_state
    ):
        """Compute joint action plan for two agents to achieve a
        certain position and orientation with the joint motion graph

        Args:
            joint_start_state: pair of start (pos, or)
            joint_goal_state: pair of goal (pos, or)
        """
        assert self.is_valid_joint_motion_pair(
            joint_start_state, joint_goal_state
        ), joint_goal_state
        # Solve shortest-path graph problem
        start_positions = list(zip(*joint_start_state))[0]
        goal_positions = list(zip(*joint_goal_state))[0]
        try:
            joint_positions_node_path = self.joint_graph_problem.get_node_path(
                start_positions, goal_positions
            )[1:]
        except NotConnectedError:
            # The cost will be infinite if there is no path
            num_player = len(goal_positions)
            return [], None, [np.inf] * num_player
        (
            joint_actions_list,
            end_pos_and_orientations,
            finishing_times,
        ) = self.joint_action_plan_from_positions(
            joint_positions_node_path, joint_start_state, joint_goal_state
        )
        return joint_actions_list, end_pos_and_orientations, finishing_times

    def joint_action_plan_from_positions(
        self, joint_positions, joint_start_state, joint_goal_state
    ):
        """
        Finds an action plan and it's cost, such that at least one of the agent goal states is achieved

        Args:
            joint_positions (list): list of joint positions to be reached after the starting position
                                    (does not include starting position, but includes ending position)
            joint_start_state (tuple): pair of starting positions and orientations
            joint_goal_state (tuple): pair of goal positions and orientations
        """
        action_plans = []
        for i in range(2):
            agent_position_sequence = [
                joint_position[i] for joint_position in joint_positions
            ]
            action_plan, _, _ = self.motion_planner.action_plan_from_positions(
                agent_position_sequence,
                joint_start_state[i],
                joint_goal_state[i],
            )
            action_plans.append(action_plan)

        finishing_times = tuple(len(plan) for plan in action_plans)
        trimmed_action_plans = self._fix_plan_lengths(action_plans)
        joint_action_plan = list(zip(*trimmed_action_plans))
        end_pos_and_orientations = self._rollout_end_pos_and_or(
            joint_start_state, joint_action_plan
        )
        return joint_action_plan, end_pos_and_orientations, finishing_times

    def _fix_plan_lengths(self, plans):
        """Truncates the longer plan when shorter plan ends"""
        plans = list(plans)
        finishing_times = [len(p) for p in plans]
        delta_length = max(finishing_times) - min(finishing_times)
        if delta_length != 0:
            index_long_plan = np.argmax(finishing_times)
            plans[index_long_plan] = plans[index_long_plan][
                : min(finishing_times)
            ]
        return plans

    def _rollout_end_pos_and_or(self, joint_start_state, joint_action_plan):
        """Execute plan in environment to determine ending positions and orientations"""
        # Assumes that final pos and orientations only depend on initial ones
        # (not on objects and other aspects of state).
        # Also assumes can't deliver more than two orders in one motion goal
        # (otherwise Environment will terminate)
        from onpolicy.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

        dummy_state = OvercookedState.from_players_pos_and_or(
            joint_start_state, all_orders=self.mdp.start_all_orders
        )
        env = OvercookedEnv.from_mdp(
            self.mdp, horizon=200, info_level=int(self.debug)
        )  # Plans should be shorter than 200 timesteps, or something is likely wrong
        successor_state, is_done = env.execute_plan(
            dummy_state, joint_action_plan
        )
        assert not is_done
        return successor_state.players_pos_and_or

    def _joint_graph_from_grid(self):
        """Creates a graph instance from the mdp instance. Each graph node encodes a pair of positions"""
        state_decoder = {}
        # Valid positions pairs, not including ones with both players in same spot
        valid_joint_positions = self.mdp.get_valid_joint_player_positions()
        for state_index, joint_pos in enumerate(valid_joint_positions):
            state_decoder[state_index] = joint_pos

        state_encoder = {v: k for k, v in state_decoder.items()}
        num_graph_nodes = len(state_decoder)

        adjacency_matrix = np.zeros((num_graph_nodes, num_graph_nodes))
        for start_state_index, start_joint_positions in state_decoder.items():
            for (
                joint_action,
                successor_jm_state,
            ) in self._get_valid_successor_joint_positions(
                start_joint_positions
            ).items():
                successor_node_index = state_encoder[successor_jm_state]

                this_action_cost = self._graph_joint_action_cost(joint_action)
                current_cost = adjacency_matrix[start_state_index][
                    successor_node_index
                ]

                if current_cost == 0 or this_action_cost < current_cost:
                    adjacency_matrix[start_state_index][
                        successor_node_index
                    ] = this_action_cost

        return Graph(adjacency_matrix, state_encoder, state_decoder)

    def _graph_joint_action_cost(self, joint_action):
        """The cost used in the graph shortest-path problem for a certain joint-action"""
        num_of_non_stay_actions = len(
            [a for a in joint_action if a != Action.STAY]
        )
        # NOTE: Removing the possibility of having 0 cost joint_actions
        if num_of_non_stay_actions == 0:
            return 1
        return num_of_non_stay_actions

    def _get_valid_successor_joint_positions(self, starting_positions):
        """Get all joint positions that can be reached by a joint action.
        NOTE: this DOES NOT include joint positions with superimposed agents.
        """
        successor_joint_positions = {}
        joint_motion_actions = itertools.product(
            Action.MOTION_ACTIONS, Action.MOTION_ACTIONS
        )

        # Under assumption that orientation doesn't matter
        dummy_orientation = Direction.NORTH
        dummy_player_states = [
            PlayerState(pos, dummy_orientation) for pos in starting_positions
        ]
        for joint_action in joint_motion_actions:
            new_positions, _ = self.mdp.compute_new_positions_and_orientations(
                dummy_player_states, joint_action
            )
            successor_joint_positions[joint_action] = new_positions
        return successor_joint_positions

    def derive_state(self, start_state, end_pos_and_ors, action_plans):
        """
        Given a start state, end position and orientations, and an action plan, recovers
        the resulting state without executing the entire plan.
        """
        if len(action_plans) == 0:
            return start_state

        end_state = start_state.deepcopy()
        end_players = []
        for player, end_pos_and_or in zip(end_state.players, end_pos_and_ors):
            new_player = player.deepcopy()
            position, orientation = end_pos_and_or
            new_player.update_pos_and_or(position, orientation)
            end_players.append(new_player)

        end_state.players = tuple(end_players)

        # Resolve environment effects for t - 1 turns
        plan_length = len(action_plans)
        assert plan_length > 0
        for _ in range(plan_length - 1):
            self.mdp.step_environment_effects(end_state)

        # Interacts
        last_joint_action = tuple(
            a if a == Action.INTERACT else Action.STAY
            for a in action_plans[-1]
        )

        events_dict = {
            k: [[] for _ in range(self.mdp.num_players)] for k in EVENT_TYPES
        }
        self.mdp.resolve_interacts(end_state, last_joint_action, events_dict)
        self.mdp.resolve_movement(end_state, last_joint_action)
        self.mdp.step_environment_effects(end_state)
        return end_state


class MediumLevelActionManager(object):
    """
    Manager for medium level actions (specific joint motion goals).
    Determines available medium level actions for each state.

    Args:
        mdp (OvercookedGridWorld): gridworld of interest
        mlam_params (dictionary): parameters for the medium level action manager
    """

    def __init__(self, mdp, mlam_params):
        self.mdp = mdp

        self.params = mlam_params
        self.wait_allowed = mlam_params["wait_allowed"]
        self.counter_drop = mlam_params["counter_drop"]
        self.counter_pickup = mlam_params["counter_pickup"]

        self.joint_motion_planner = JointMotionPlanner(mdp, mlam_params)
        self.motion_planner = self.joint_motion_planner.motion_planner

    def save_to_file(self, filename):
        with open(filename, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_file(filename):
        return load_saved_action_manager(filename)

    @staticmethod
    def from_pickle_or_compute(
        mdp, mlam_params, custom_filename=None, force_compute=False, info=False
    ):
        assert isinstance(mdp, OvercookedGridworld)

        filename = (
            custom_filename
            if custom_filename is not None
            else mdp.layout_name + "_am.pkl"
        )

        if force_compute:
            return MediumLevelActionManager.compute_mlam(
                filename, mdp, mlam_params, info=info
            )

        try:
            mlam = MediumLevelActionManager.from_file(filename)

            if mlam.params != mlam_params or mlam.mdp != mdp:
                if info:
                    print(
                        "medium level action manager with different params or mdp found, computing from scratch"
                    )
                return MediumLevelActionManager.compute_mlam(
                    filename, mdp, mlam_params, info=info
                )

        except (
            FileNotFoundError,
            ModuleNotFoundError,
            EOFError,
            AttributeError,
        ) as e:
            if info:
                print("Recomputing planner due to:", e)
            return MediumLevelActionManager.compute_mlam(
                filename, mdp, mlam_params, info=info
            )

        if info:
            print(
                "Loaded MediumLevelActionManager from {}".format(
                    os.path.join(PLANNERS_DIR, filename)
                )
            )
        return mlam

    @staticmethod
    def compute_mlam(filename, mdp, mlam_params, info=False):
        final_filepath = os.path.join(PLANNERS_DIR, filename)
        if info:
            print(
                "Computing MediumLevelActionManager to be saved in {}".format(
                    final_filepath
                )
            )
        start_time = time.time()
        mlam = MediumLevelActionManager(mdp, mlam_params=mlam_params)
        if info:
            print(
                "It took {} seconds to create mlam".format(
                    time.time() - start_time
                )
            )
        mlam.save_to_file(final_filepath)
        return mlam

    def joint_ml_actions(self, state):
        """Determine all possible joint medium level actions for a certain state"""
        agent1_actions, agent2_actions = tuple(
            self.get_medium_level_actions(state, player)
            for player in state.players
        )
        joint_ml_actions = list(
            itertools.product(agent1_actions, agent2_actions)
        )

        # ml actions are nothing but specific joint motion goals
        valid_joint_ml_actions = list(
            filter(
                lambda a: self.is_valid_ml_action(state, a), joint_ml_actions
            )
        )

        # HACK: Could cause things to break.
        # Necessary to prevent states without successors (due to no counters being allowed and no wait actions)
        # causing A* to not find a solution
        if len(valid_joint_ml_actions) == 0:
            agent1_actions, agent2_actions = tuple(
                self.get_medium_level_actions(
                    state, player, waiting_substitute=True
                )
                for player in state.players
            )
            joint_ml_actions = list(
                itertools.product(agent1_actions, agent2_actions)
            )
            valid_joint_ml_actions = list(
                filter(
                    lambda a: self.is_valid_ml_action(state, a),
                    joint_ml_actions,
                )
            )
            if len(valid_joint_ml_actions) == 0:
                print(
                    "WARNING: Found state without valid actions even after adding waiting substitute actions. State: {}".format(
                        state
                    )
                )
        return valid_joint_ml_actions

    def is_valid_ml_action(self, state, ml_action):
        return self.joint_motion_planner.is_valid_jm_start_goal_pair(
            state.players_pos_and_or, ml_action
        )

    def get_medium_level_actions(
        self, state, player, waiting_substitute=False
    ):
        """
        Determine valid medium level actions for a player.

        Args:
            state (OvercookedState): current state
            player (PlayerState): the player's current state
            waiting_substitute (bool): add a substitute action that takes the place of
                                       a waiting action (going to closest feature)

        Returns:
            player_actions (list): possible motion goals (pairs of goal positions and orientations)
        """
        player_actions = []
        counter_pickup_objects = self.mdp.get_counter_objects_dict(
            state, self.counter_pickup
        )
        if not player.has_object():
            onion_pickup = self.pickup_onion_actions(counter_pickup_objects)
            tomato_pickup = self.pickup_tomato_actions(counter_pickup_objects)
            dish_pickup = self.pickup_dish_actions(counter_pickup_objects)
            soup_pickup = self.pickup_counter_soup_actions(
                counter_pickup_objects
            )

            pot_states_dict = self.mdp.get_pot_states(state)
            start_cooking = self.start_cooking_actions(pot_states_dict)
            player_actions.extend(
                onion_pickup
                + tomato_pickup
                + dish_pickup
                + soup_pickup
                + start_cooking
            )

        else:
            player_object = player.get_object()
            pot_states_dict = self.mdp.get_pot_states(state)

            # No matter the object, we can place it on a counter
            if len(self.counter_drop) > 0:
                player_actions.extend(self.place_obj_on_counter_actions(state))

            if player_object.name == "soup":
                player_actions.extend(self.deliver_soup_actions())
            elif player_object.name == "onion":
                player_actions.extend(
                    self.put_onion_in_pot_actions(pot_states_dict)
                )
            elif player_object.name == "tomato":
                player_actions.extend(
                    self.put_tomato_in_pot_actions(pot_states_dict)
                )
            elif player_object.name == "dish":
                # Not considering all pots (only ones close to ready) to reduce computation
                # NOTE: could try to calculate which pots are eligible, but would probably take
                # a lot of compute
                player_actions.extend(
                    self.pickup_soup_with_dish_actions(
                        pot_states_dict, only_nearly_ready=False
                    )
                )
            else:
                raise ValueError("Unrecognized object")

        if self.wait_allowed:
            player_actions.extend(self.wait_actions(player))

        if waiting_substitute:
            # Trying to mimic a "WAIT" action by adding the closest allowed feature to the avaliable actions
            # This is because motion plans that aren't facing terrain features (non counter, non empty spots)
            # are not considered valid
            player_actions.extend(self.go_to_closest_feature_actions(player))

        is_valid_goal_given_start = (
            lambda goal: self.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, goal
            )
        )
        player_actions = list(
            filter(is_valid_goal_given_start, player_actions)
        )
        return player_actions

    def pickup_onion_actions(self, counter_objects, only_use_dispensers=False):
        """If only_use_dispensers is True, then only take onions from the dispensers"""
        onion_pickup_locations = self.mdp.get_onion_dispenser_locations()
        if not only_use_dispensers:
            onion_pickup_locations += counter_objects["onion"]
        return self._get_ml_actions_for_positions(onion_pickup_locations)

    def pickup_tomato_actions(self, counter_objects):
        tomato_dispenser_locations = self.mdp.get_tomato_dispenser_locations()
        tomato_pickup_locations = (
            tomato_dispenser_locations + counter_objects["tomato"]
        )
        return self._get_ml_actions_for_positions(tomato_pickup_locations)

    def pickup_dish_actions(self, counter_objects, only_use_dispensers=False):
        """If only_use_dispensers is True, then only take dishes from the dispensers"""
        dish_pickup_locations = self.mdp.get_dish_dispenser_locations()
        if not only_use_dispensers:
            dish_pickup_locations += counter_objects["dish"]
        return self._get_ml_actions_for_positions(dish_pickup_locations)

    def pickup_counter_soup_actions(self, counter_objects):
        soup_pickup_locations = counter_objects["soup"]
        return self._get_ml_actions_for_positions(soup_pickup_locations)

    def start_cooking_actions(self, pot_states_dict):
        """This is for start cooking a pot that is cookable"""
        cookable_pots_location = self.mdp.get_partially_full_pots(
            pot_states_dict
        ) + self.mdp.get_full_but_not_cooking_pots(pot_states_dict)
        return self._get_ml_actions_for_positions(cookable_pots_location)

    def place_obj_on_counter_actions(self, state):
        all_empty_counters = set(self.mdp.get_empty_counter_locations(state))
        valid_empty_counters = [
            c_pos for c_pos in self.counter_drop if c_pos in all_empty_counters
        ]
        return self._get_ml_actions_for_positions(valid_empty_counters)

    def deliver_soup_actions(self):
        serving_locations = self.mdp.get_serving_locations()
        return self._get_ml_actions_for_positions(serving_locations)

    def put_onion_in_pot_actions(self, pot_states_dict):
        partially_full_onion_pots = self.mdp.get_partially_full_pots(
            pot_states_dict
        )
        fillable_pots = partially_full_onion_pots + pot_states_dict["empty"]
        return self._get_ml_actions_for_positions(fillable_pots)

    def put_tomato_in_pot_actions(self, pot_states_dict):
        partially_full_onion_pots = self.mdp.get_partially_full_pots(
            pot_states_dict
        )
        fillable_pots = partially_full_onion_pots + pot_states_dict["empty"]
        return self._get_ml_actions_for_positions(fillable_pots)

    def pickup_soup_with_dish_actions(
        self, pot_states_dict, only_nearly_ready=False
    ):
        ready_pot_locations = pot_states_dict["ready"]
        nearly_ready_pot_locations = pot_states_dict["cooking"]
        if not only_nearly_ready:
            partially_full_pots = self.mdp.get_partially_full_pots(
                pot_states_dict
            )
            nearly_ready_pot_locations = (
                nearly_ready_pot_locations
                + pot_states_dict["empty"]
                + partially_full_pots
            )
        return self._get_ml_actions_for_positions(
            ready_pot_locations + nearly_ready_pot_locations
        )

    def go_to_closest_feature_actions(self, player):
        feature_locations = (
            self.mdp.get_onion_dispenser_locations()
            + self.mdp.get_tomato_dispenser_locations()
            + self.mdp.get_pot_locations()
            + self.mdp.get_dish_dispenser_locations()
        )
        closest_feature_pos = self.motion_planner.min_cost_to_feature(
            player.pos_and_or, feature_locations, with_argmin=True
        )[1]
        return self._get_ml_actions_for_positions([closest_feature_pos])

    def go_to_closest_feature_or_counter_to_goal(
        self, goal_pos_and_or, goal_location
    ):
        """Instead of going to goal_pos_and_or, go to the closest feature or counter to this goal, that ISN'T the goal itself"""
        valid_locations = (
            self.mdp.get_onion_dispenser_locations()
            + self.mdp.get_tomato_dispenser_locations()
            + self.mdp.get_pot_locations()
            + self.mdp.get_dish_dispenser_locations()
            + self.counter_drop
        )
        valid_locations.remove(goal_location)
        closest_non_goal_feature_pos = self.motion_planner.min_cost_to_feature(
            goal_pos_and_or, valid_locations, with_argmin=True
        )[1]
        return self._get_ml_actions_for_positions(
            [closest_non_goal_feature_pos]
        )

    def wait_actions(self, player):
        waiting_motion_goal = (player.position, player.orientation)
        return [waiting_motion_goal]

    def _get_ml_actions_for_positions(self, positions_list):
        """Determine what are the ml actions (joint motion goals) for a list of positions

        Args:
            positions_list (list): list of target terrain feature positions
        """
        possible_motion_goals = []
        for pos in positions_list:
            # All possible ways to reach the target feature
            for (
                motion_goal
            ) in self.joint_motion_planner.motion_planner.motion_goals_for_pos[
                pos
            ]:
                possible_motion_goals.append(motion_goal)
        return possible_motion_goals


# # Deprecated, since agent-level dynamic planning is no longer used
# class MediumLevelPlanner(object):
#     """
#     A planner that computes optimal plans for two agents to deliver a certain number of dishes
#     in an OvercookedGridworld using medium level actions (single motion goals) in the corresponding
#     A* search problem.
#     """
#
#     def __init__(self, mdp, mlp_params, ml_action_manager=None):
#         self.mdp = mdp
#         self.params = mlp_params
#         self.ml_action_manager = ml_action_manager if ml_action_manager else MediumLevelActionManager(mdp, mlp_params)
#         self.jmp = self.ml_action_manager.joint_motion_planner
#         self.mp = self.jmp.motion_planner
#
#     @staticmethod
#     def from_action_manager_file(filename):
#         mlp_action_manager = load_saved_action_manager(filename)
#         mdp = mlp_action_manager.mdp
#         params = mlp_action_manager.params
#         return MediumLevelPlanner(mdp, params, mlp_action_manager)
#
#     @staticmethod
#     def from_pickle_or_compute(mdp, mlp_params, custom_filename=None, force_compute=False, info=True):
#         assert isinstance(mdp, OvercookedGridworld)
#
#         filename = custom_filename if custom_filename is not None else mdp.layout_name + "_am.pkl"
#
#         if force_compute:
#             return MediumLevelPlanner.compute_mlp(filename, mdp, mlp_params)
#
#         try:
#             mlp = MediumLevelPlanner.from_action_manager_file(filename)
#
#             if mlp.ml_action_manager.params != mlp_params or mlp.mdp != mdp:
#                 print("Mlp with different params or mdp found, computing from scratch")
#                 return MediumLevelPlanner.compute_mlp(filename, mdp, mlp_params)
#
#         except (FileNotFoundError, ModuleNotFoundError, EOFError, AttributeError) as e:
#             print("Recomputing planner due to:", e)
#             return MediumLevelPlanner.compute_mlp(filename, mdp, mlp_params)
#
#         if info:
#             print("Loaded MediumLevelPlanner from {}".format(os.path.join(PLANNERS_DIR, filename)))
#         return mlp
#
#     @staticmethod
#     def compute_mlp(filename, mdp, mlp_params):
#         final_filepath = os.path.join(PLANNERS_DIR, filename)
#         print("Computing MediumLevelPlanner to be saved in {}".format(final_filepath))
#         start_time = time.time()
#         mlp = MediumLevelPlanner(mdp, mlp_params=mlp_params)
#         print("It took {} seconds to create mlp".format(time.time() - start_time))
#         mlp.ml_action_manager.save_to_file(final_filepath)
#         return mlp
#
# Deprecated.
# def get_successor_states(self, start_state):
#     """Successor states for medium-level actions are defined as
#     the first state in the corresponding motion plan in which
#     one of the two agents' subgoals is satisfied.
#
#     Returns: list of
#         joint_motion_goal: ((pos1, or1), (pos2, or2)) specifying the
#                             motion plan goal for both agents
#
#         successor_state:   OvercookedState corresponding to state
#                            arrived at after executing part of the motion plan
#                            (until one of the agents arrives at his goal status)
#
#         plan_length:       Time passed until arrival to the successor state
#     """
#     if self.mdp.is_terminal(start_state):
#         return []
#
#     start_jm_state = start_state.players_pos_and_or
#     successor_states = []
#     for goal_jm_state in self.ml_action_manager.joint_ml_actions(start_state):
#         joint_motion_action_plans, end_pos_and_ors, plan_costs = self.jmp.get_low_level_action_plan(start_jm_state, goal_jm_state)
#         end_state = self.jmp.derive_state(start_state, end_pos_and_ors, joint_motion_action_plans)
#
#         if SAFE_RUN:
#             from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
#             assert end_pos_and_ors[0] == goal_jm_state[0] or end_pos_and_ors[1] == goal_jm_state[1]
#             s_prime, _ = OvercookedEnv.execute_plan(self.mdp, start_state, joint_motion_action_plans, display=False)
#             assert end_state == s_prime,  [self.mdp.state_string(s_prime), self.mdp.state_string(end_state)]
#
#         successor_states.append((goal_jm_state, end_state, min(plan_costs)))
#     return successor_states

# Deprecated.
# def get_successor_states_fixed_other(self, start_state, other_agent, other_agent_idx):
#     """
#     Get the successor states of a given start state, assuming that the other agent is fixed and will act according to the passed in model
#     """
#     if self.mdp.is_terminal(start_state):
#         return []
#
#     player = start_state.players[1 - other_agent_idx]
#     ml_actions = self.ml_action_manager.get_medium_level_actions(start_state, player)
#
#     if len(ml_actions) == 0:
#         ml_actions = self.ml_action_manager.get_medium_level_actions(start_state, player, waiting_substitute=True)
#
#     successor_high_level_states = []
#     for ml_action in ml_actions:
#         action_plan, end_state, cost = self.get_embedded_low_level_action_plan(start_state, ml_action, other_agent, other_agent_idx)
#
#         if not self.mdp.is_terminal(end_state):
#             # Adding interact action and deriving last state
#             other_agent_action, _ = other_agent.action(end_state)
#             last_joint_action = (Action.INTERACT, other_agent_action) if other_agent_idx == 1 else (other_agent_action, Action.INTERACT)
#             action_plan = action_plan + (last_joint_action,)
#             cost = cost + 1
#
#             end_state, _ = self.embedded_mdp_step(end_state, Action.INTERACT, other_agent_action, other_agent.agent_index)
#
#         successor_high_level_states.append((action_plan, end_state, cost))
#     return successor_high_level_states

# Deprecated. because no longer used
# def check_heuristic_consistency(self, curr_heuristic_val, prev_heuristic_val, actual_edge_cost):
#     delta_h = curr_heuristic_val - prev_heuristic_val
#     assert actual_edge_cost >= delta_h, \
#         "Heuristic was not consistent. \n Prev h: {}, Curr h: {}, Actual cost: {}, Δh: {}" \
#         .format(prev_heuristic_val, curr_heuristic_val, actual_edge_cost, delta_h)
#
# def embedded_mdp_succ_fn(self, state, other_agent):
#     other_agent_action, _ = other_agent.action(state)
#
#     successors = []
#     for a in Action.ALL_ACTIONS:
#         successor_state, joint_action = self.embedded_mdp_step(state, a, other_agent_action, other_agent.agent_index)
#         cost = 1
#         successors.append((joint_action, successor_state, cost))
#     return successors
#
# def embedded_mdp_step(self, state, action, other_agent_action, other_agent_index):
#     if other_agent_index == 0:
#         joint_action = (other_agent_action, action)
#     else:
#         joint_action = (action, other_agent_action)
#     if not self.mdp.is_terminal(state):
#         results, _ = self.mdp.get_state_transition(state, joint_action)
#         successor_state = results
#     else:
#         print("Tried to find successor of terminal")
#         assert False, "state {} \t action {}".format(state, action)
#         successor_state = state
#     return successor_state, joint_action

# Deprecated due to Heuristic
# def get_low_level_action_plan(self, start_state, h_fn, delivery_horizon=4, debug=False, goal_info=False):
#     """
#     Get a plan of joint-actions executable in the environment that will lead to a goal number of deliveries
#
#     Args:
#         state (OvercookedState): starting state
#         h_fn: heuristic function
#
#     Returns:
#         full_joint_action_plan (list): joint actions to reach goal
#     """
#     start_state = start_state.deepcopy()
#     ml_plan, cost = self.get_ml_plan(start_state, h_fn, delivery_horizon=delivery_horizon, debug=debug)
#
#     full_joint_action_plan = self.get_low_level_plan_from_ml_plan(
#         start_state, ml_plan, h_fn, debug=debug, goal_info=goal_info
#     )
#     assert cost == len(full_joint_action_plan), "A* cost {} but full joint action plan cost {}".format(cost, len(full_joint_action_plan))
#
#     if debug: print("Found plan with cost {}".format(cost))
#     return full_joint_action_plan

# Deprecated due to Heuristic
# def get_low_level_plan_from_ml_plan(self, start_state, ml_plan, heuristic_fn, debug=False, goal_info=False):
#     t = 0
#     full_joint_action_plan = []
#     curr_state = start_state
#     curr_motion_state = start_state.players_pos_and_or
#     prev_h = heuristic_fn(start_state, t, debug=False)
#
#     if len(ml_plan) > 0 and goal_info:
#         print("First motion goal: ", ml_plan[0][0])
#
#     if not clean and debug:
#         print("Start state")
#         OvercookedEnv.print_state(self.mdp, start_state)
#
#     for joint_motion_goal, goal_state in ml_plan:
#         joint_action_plan, end_motion_state, plan_costs = \
#             self.ml_action_manager.joint_motion_planner.get_low_level_action_plan(curr_motion_state, joint_motion_goal)
#         curr_plan_cost = min(plan_costs)
#         full_joint_action_plan.extend(joint_action_plan)
#         t += 1
#
#         if not clean and debug:
#             print(t)
#             OvercookedEnv.print_state(self.mdp, goal_state)
#
#         if not clean and SAFE_RUN:
#             s_prime, _ = OvercookedEnv.execute_plan(self.mdp, curr_state, joint_action_plan)
#             assert s_prime == goal_state
#
#         curr_h = heuristic_fn(goal_state, t, debug=False)
#         self.check_heuristic_consistency(curr_h, prev_h, curr_plan_cost)
#         curr_motion_state, prev_h, curr_state = end_motion_state, curr_h, goal_state
#     return full_joint_action_plan


# Deprecated due to Heuristic
# def get_ml_plan(self, start_state, h_fn, delivery_horizon=4, debug=False):
#     """
#     Solves A* Search problem to find optimal sequence of medium level actions
#     to reach the goal number of deliveries
#
#     Returns:
#         ml_plan (list): plan not including starting state in form
#             [(joint_action, successor_state), ..., (joint_action, goal_state)]
#         cost (int): A* Search cost
#     """
#     start_state = start_state.deepcopy()
#
#     expand_fn = lambda state: self.get_successor_states(state)
#     goal_fn = lambda state: state.delivery_rew >= DELIVERY_REW_THRES
#     heuristic_fn = lambda state: h_fn(state)
#
#     search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, debug=debug)
#     ml_plan, cost = search_problem.A_star_graph_search(info=True)
#     return ml_plan[1:], cost

# Deprecated
# def get_embedded_low_level_action_plan(self, state, goal_pos_and_or, other_agent, other_agent_idx):
#     """Find action plan for a specific motion goal with A* considering the other agent"""
#     other_agent.set_agent_index(other_agent_idx)
#     agent_idx = 1 - other_agent_idx
#
#     expand_fn = lambda state: self.embedded_mdp_succ_fn(state, other_agent)
#     # FIXME
#     goal_fn = lambda state: state.players[agent_idx].pos_and_or == goal_pos_and_or or state.delivery_rew >= DELIVERY_REW_THRES
#     heuristic_fn = lambda state: sum(pos_distance(state.players[agent_idx].position, goal_pos_and_or[0]))
#
#     search_problem = SearchTree(state, goal_fn, expand_fn, heuristic_fn)
#     state_action_plan, cost = search_problem.A_star_graph_search(info=False)
#     action_plan, state_plan = zip(*state_action_plan)
#     action_plan = action_plan[1:]
#     end_state = state_plan[-1]
#     return action_plan, end_state, cost


# Deprecated.
# class HighLevelAction:
#     """A high level action is given by a set of subsequent motion goals"""
#
#     def __init__(self, motion_goals):
#         self.motion_goals = motion_goals
#
#     def _check_valid(self):
#         for goal in self.motion_goals:
#             assert len(goal) == 2
#             pos, orient = goal
#             assert orient in Direction.ALL_DIRECTIONS
#             assert type(pos) is tuple
#             assert len(pos) == 2
#
#     def __getitem__(self, i):
#         """Get ith motion goal of the HL Action"""
#         return self.motion_goals[i]
#
#
# class HighLevelActionManager(object):
#     """
#     Manager for high level actions. Determines available high level actions
#     for each state and player.
#     """
#
#     def __init__(self, medium_level_planner):
#         self.mdp = medium_level_planner.mdp
#
#         self.wait_allowed = medium_level_planner.params['wait_allowed']
#         self.counter_drop = medium_level_planner.params["counter_drop"]
#         self.counter_pickup = medium_level_planner.params["counter_pickup"]
#
#         self.mlp = medium_level_planner
#         self.ml_action_manager = medium_level_planner.ml_action_manager
#         self.mp = medium_level_planner.mp
#
#     def joint_hl_actions(self, state):
#         hl_actions_a0, hl_actions_a1 = tuple(self.get_high_level_actions(state, player) for player in state.players)
#         joint_hl_actions = list(itertools.product(hl_actions_a0, hl_actions_a1))
#
#         assert self.mlp.params["same_motion_goals"]
#         valid_joint_hl_actions = joint_hl_actions
#
#         if len(valid_joint_hl_actions) == 0:
#             print("WARNING: found a state without high level successors")
#         return valid_joint_hl_actions
#
#     def get_high_level_actions(self, state, player):
#         player_hl_actions = []
#         counter_pickup_objects = self.mdp.get_counter_objects_dict(state, self.counter_pickup)
#         if player.has_object():
#             place_obj_ml_actions = self.ml_action_manager.get_medium_level_actions(state, player)
#
#             # HACK to prevent some states not having successors due to lack of waiting actions
#             if len(place_obj_ml_actions) == 0:
#                 place_obj_ml_actions = self.ml_action_manager.get_medium_level_actions(state, player, waiting_substitute=True)
#
#             place_obj_hl_actions = [HighLevelAction([ml_action]) for ml_action in place_obj_ml_actions]
#             player_hl_actions.extend(place_obj_hl_actions)
#         else:
#             pot_states_dict = self.mdp.get_pot_states(state)
#             player_hl_actions.extend(self.get_onion_and_put_in_pot(state, counter_pickup_objects, pot_states_dict))
#             player_hl_actions.extend(self.get_tomato_and_put_in_pot(state, counter_pickup_objects, pot_states_dict))
#             player_hl_actions.extend(self.get_dish_and_soup_and_serve(state, counter_pickup_objects, pot_states_dict))
#             player_hl_actions.extend(self.start_cooking(state, pot_states_dict))
#         return player_hl_actions
#
#     def get_dish_and_soup_and_serve(self, state, counter_objects, pot_states_dict):
#         """Get all sequences of medium-level actions (hl actions) that involve a player getting a dish,
#         going to a pot and picking up a soup, and delivering the soup."""
#         dish_pickup_actions = self.ml_action_manager.pickup_dish_actions(counter_objects)
#         pickup_soup_actions = self.ml_action_manager.pickup_soup_with_dish_actions(pot_states_dict)
#         deliver_soup_actions = self.ml_action_manager.deliver_soup_actions()
#         hl_level_actions = list(itertools.product(dish_pickup_actions, pickup_soup_actions, deliver_soup_actions))
#         return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]
#
#     def get_onion_and_put_in_pot(self, state, counter_objects, pot_states_dict):
#         """Get all sequences of medium-level actions (hl actions) that involve a player getting an onion
#         from a dispenser and placing it in a pot."""
#         onion_pickup_actions = self.ml_action_manager.pickup_onion_actions(counter_objects)
#         put_in_pot_actions = self.ml_action_manager.put_onion_in_pot_actions(pot_states_dict)
#         hl_level_actions = list(itertools.product(onion_pickup_actions, put_in_pot_actions))
#         return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]
#
#     def get_tomato_and_put_in_pot(self, state, counter_objects, pot_states_dict):
#         """Get all sequences of medium-level actions (hl actions) that involve a player getting an tomato
#         from a dispenser and placing it in a pot."""
#         tomato_pickup_actions = self.ml_action_manager.pickup_tomato_actions(counter_objects)
#         put_in_pot_actions = self.ml_action_manager.put_tomato_in_pot_actions(pot_states_dict)
#         hl_level_actions = list(itertools.product(tomato_pickup_actions, put_in_pot_actions))
#         return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]
#
#     def start_cooking(self, state, pot_states_dict):
#         """Go to a pot that is not empty and start cooking. Currently, because high level action requires 2 goals,
#         we are going to repeat the same goal twice"""
#         start_cooking = self.ml_action_manager.start_cooking_actions(pot_states_dict)
#         hl_level_actions = [(pot, pot) for pot in start_cooking]
#         return [HighLevelAction(hl_action_list) for hl_action_list in hl_level_actions]
#
#
#
# class HighLevelPlanner(object):
#     """A planner that computes optimal plans for two agents to
#     deliver a certain number of dishes in an OvercookedGridworld
#     using high level actions in the corresponding A* search problems
#     """
#
#     def __init__(self, hl_action_manager):
#         self.hl_action_manager = hl_action_manager
#         self.mlp = self.hl_action_manager.mlp
#         self.jmp = self.mlp.ml_action_manager.joint_motion_planner
#         self.mp = self.jmp.motion_planner
#         self.mdp = self.mlp.mdp
#
#     def get_successor_states(self, start_state):
#         """Determines successor states for high-level actions"""
#         successor_states = []
#
#         if self.mdp.is_terminal(start_state):
#             return successor_states
#
#         for joint_hl_action in self.hl_action_manager.joint_hl_actions(start_state):
#             _, end_state, hl_action_cost = self.perform_hl_action(joint_hl_action, start_state)
#
#             successor_states.append((joint_hl_action, end_state, hl_action_cost))
#         return successor_states
#
#     def perform_hl_action(self, joint_hl_action, curr_state):
#         """Determines the end state for a high level action, and the corresponding low level action plan and cost.
#         Will return Nones if a pot exploded throughout the execution of the action"""
#         full_plan = []
#         motion_goal_indices = (0, 0)
#         total_cost = 0
#         while not self.at_least_one_finished_hl_action(joint_hl_action, motion_goal_indices):
#             curr_jm_goal = tuple(joint_hl_action[i].motion_goals[motion_goal_indices[i]] for i in range(2))
#             joint_motion_action_plans, end_pos_and_ors, plan_costs = \
#                 self.jmp.get_low_level_action_plan(curr_state.players_pos_and_or, curr_jm_goal)
#             curr_state = self.jmp.derive_state(curr_state, end_pos_and_ors, joint_motion_action_plans)
#             motion_goal_indices = self._advance_motion_goal_indices(motion_goal_indices, plan_costs)
#             total_cost += min(plan_costs)
#             full_plan.extend(joint_motion_action_plans)
#         return full_plan, curr_state, total_cost
#
#     def at_least_one_finished_hl_action(self, joint_hl_action, motion_goal_indices):
#         """Returns whether either agent has reached the end of the motion goal list it was supposed
#         to perform to finish it's high level action"""
#         return any([len(joint_hl_action[i].motion_goals) == motion_goal_indices[i] for i in range(2)])
#
#     def get_low_level_action_plan(self, start_state, h_fn, debug=False):
#         """
#         Get a plan of joint-actions executable in the environment that will lead to a goal number of deliveries
#         by performaing an A* search in high-level action space
#
#         Args:
#             state (OvercookedState): starting state
#
#         Returns:
#             full_joint_action_plan (list): joint actions to reach goal
#             cost (int): a cost in number of timesteps to reach the goal
#         """
#         full_joint_low_level_action_plan = []
#         hl_plan, cost = self.get_hl_plan(start_state, h_fn)
#         curr_state = start_state
#         prev_h = h_fn(start_state, debug=False)
#         total_cost = 0
#         for joint_hl_action, curr_goal_state in hl_plan:
#             assert all([type(a) is HighLevelAction for a in joint_hl_action])
#             hl_action_plan, curr_state, hl_action_cost = self.perform_hl_action(joint_hl_action, curr_state)
#             full_joint_low_level_action_plan.extend(hl_action_plan)
#             total_cost += hl_action_cost
#             assert curr_state == curr_goal_state
#
#             curr_h = h_fn(curr_state, debug=False)
#             self.mlp.check_heuristic_consistency(curr_h, prev_h, total_cost)
#             prev_h = curr_h
#         assert total_cost == cost == len(full_joint_low_level_action_plan), "{} vs {} vs {}"\
#             .format(total_cost, cost, len(full_joint_low_level_action_plan))
#         return full_joint_low_level_action_plan, cost
#
#     # Deprecated due to Heuristic
#     # def get_hl_plan(self, start_state, h_fn, debug=False):
#     #     expand_fn = lambda state: self.get_successor_states(state)
#     #     goal_fn = lambda state: state.delivery_rew >= DELIVERY_REW_THRES
#     #     heuristic_fn = lambda state: h_fn(state)
#     #
#     #     search_problem = SearchTree(start_state, goal_fn, expand_fn, heuristic_fn, debug=debug)
#     #     hl_plan, cost = search_problem.A_star_graph_search(info=True)
#     #     return hl_plan[1:], cost
#
#     def _advance_motion_goal_indices(self, curr_plan_indices, plan_lengths):
#         """Advance indices for agents current motion goals
#         based on who finished their motion goal this round"""
#         idx0, idx1 = curr_plan_indices
#         if plan_lengths[0] == plan_lengths[1]:
#             return idx0 + 1, idx1 + 1
#
#         who_finished = np.argmin(plan_lengths)
#         if who_finished == 0:
#             return idx0 + 1, idx1
#         elif who_finished == 1:
#             return idx0, idx1 + 1

# # Deprecated.
# class Heuristic(object):
#
#     def __init__(self, mp):
#         self.motion_planner = mp
#         self.mdp = mp.mdp
#         self.heuristic_cost_dict = self._calculate_heuristic_costs()
#
#     def hard_heuristic(self, state, goal_deliveries, time=0, debug=False):
#         # NOTE: does not support tomatoes – currently deprecated as harder heuristic
#         # does not seem worth the additional computational time
#
#         """
#         From a state, we can calculate exactly how many:
#         - soup deliveries we need
#         - dishes to pots we need
#         - onion to pots we need
#
#         We then determine if there are any soups/dishes/onions
#         in transit (on counters or on players) than can be
#         brought to their destinations faster than starting off from
#         a dispenser of the same type. If so, we consider fulfilling
#         all demand from these positions.
#
#         After all in-transit objects are considered, we consider the
#         costs required to fulfill all the rest of the demand, that is
#         given by:
#         - pot-delivery trips
#         - dish-pot trips
#         - onion-pot trips
#
#         The total cost is obtained by determining an optimistic time
#         cost for each of these trip types
#         """
#         forward_cost = 0
#
#         # Obtaining useful quantities
#         objects_dict = state.unowned_objects_by_type
#         player_objects = state.player_objects_by_type
#         pot_states_dict = self.mdp.get_pot_states(state)
#         min_pot_delivery_cost = self.heuristic_cost_dict['pot-delivery']
#         min_dish_to_pot_cost = self.heuristic_cost_dict['dish-pot']
#         min_onion_to_pot_cost = self.heuristic_cost_dict['onion-pot']
#
#         pot_locations = self.mdp.get_pot_locations()
#         full_soups_in_pots = pot_states_dict['cooking'] + pot_states_dict['ready']
#         partially_full_soups = self.mdp.get_partially_full_pots(pot_states_dict)
#         num_onions_in_partially_full_pots = sum([state.get_object(loc).state[1] for loc in partially_full_soups])
#
#         # Calculating costs
#         num_deliveries_to_go = goal_deliveries - state.num_delivered
#
#         # SOUP COSTS
#         total_num_soups_needed = max([0, num_deliveries_to_go])
#
#         soups_on_counters = [soup_obj for soup_obj in objects_dict['soup'] if soup_obj.position not in pot_locations]
#         soups_in_transit = player_objects['soup'] + soups_on_counters
#         soup_delivery_locations = self.mdp.get_serving_locations()
#
#         num_soups_better_than_pot, total_better_than_pot_soup_cost = \
#             self.get_costs_better_than_dispenser(soups_in_transit, soup_delivery_locations, min_pot_delivery_cost, total_num_soups_needed, state)
#
#         min_pot_to_delivery_trips = max([0, total_num_soups_needed - num_soups_better_than_pot])
#         pot_to_delivery_costs = min_pot_delivery_cost * min_pot_to_delivery_trips
#
#         forward_cost += total_better_than_pot_soup_cost
#         forward_cost += pot_to_delivery_costs
#
#         # DISH COSTS
#         total_num_dishes_needed = max([0, min_pot_to_delivery_trips])
#         dishes_on_counters = objects_dict['dish']
#         dishes_in_transit = player_objects['dish'] + dishes_on_counters
#
#         num_dishes_better_than_disp, total_better_than_disp_dish_cost = \
#             self.get_costs_better_than_dispenser(dishes_in_transit, pot_locations, min_dish_to_pot_cost, total_num_dishes_needed, state)
#
#         min_dish_to_pot_trips = max([0, min_pot_to_delivery_trips - num_dishes_better_than_disp])
#         dish_to_pot_costs = min_dish_to_pot_cost * min_dish_to_pot_trips
#
#         forward_cost += total_better_than_disp_dish_cost
#         forward_cost += dish_to_pot_costs
#
#         # START COOKING COSTS, each to be filled pots will require 1 INTERACT to start cooking
#         num_pots_to_be_filled = min_pot_to_delivery_trips - len(full_soups_in_pots)
#         """Note that this is still assuming every soup requires 3 ingredients"""
#         forward_cost += num_pots_to_be_filled
#
#         # ONION COSTS
#         total_num_onions_needed = num_pots_to_be_filled * 3 - num_onions_in_partially_full_pots
#         onions_on_counters = objects_dict['onion']
#         onions_in_transit = player_objects['onion'] + onions_on_counters
#
#         num_onions_better_than_disp, total_better_than_disp_onion_cost = \
#             self.get_costs_better_than_dispenser(onions_in_transit, pot_locations, min_onion_to_pot_cost, total_num_onions_needed, state)
#
#         min_onion_to_pot_trips = max([0, total_num_onions_needed - num_onions_better_than_disp])
#         onion_to_pot_costs = min_onion_to_pot_cost * min_onion_to_pot_trips
#
#         forward_cost += total_better_than_disp_onion_cost
#         forward_cost += onion_to_pot_costs
#
#         # Going to closest feature costs
#         # NOTE: as implemented makes heuristic inconsistent
#         # for player in state.players:
#         #     if not player.has_object():
#         #         counter_objects = soups_on_counters + dishes_on_counters + onions_on_counters
#         #         possible_features = counter_objects + pot_locations + self.mdp.get_dish_dispenser_locations() + self.mdp.get_onion_dispenser_locations()
#         #         forward_cost += self.action_manager.min_cost_to_feature(player.pos_and_or, possible_features)
#
#         heuristic_cost = forward_cost / 2
#
#         if not clean and debug:
#             env = OvercookedEnv.from_mdp(self.mdp)
#             env.state = state
#             print("\n" + "#"*35)
#             print("Current state: (ml timestep {})\n".format(time))
#
#             print("# in transit: \t\t Soups {} \t Dishes {} \t Onions {}".format(
#                 len(soups_in_transit), len(dishes_in_transit), len(onions_in_transit)
#             ))
#
#             # NOTE Possible improvement: consider cost of dish delivery too when considering if a
#             # transit soup is better than dispenser equivalent
#             print("# better than disp: \t Soups {} \t Dishes {} \t Onions {}".format(
#                 num_soups_better_than_pot, num_dishes_better_than_disp, num_onions_better_than_disp
#             ))
#
#             print("# of trips: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
#                 min_pot_to_delivery_trips, min_dish_to_pot_trips, min_onion_to_pot_trips
#             ))
#
#             print("Trip costs: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
#                 pot_to_delivery_costs, dish_to_pot_costs, onion_to_pot_costs
#             ))
#
#             print(str(env) + "HEURISTIC: {}".format(heuristic_cost))
#
#         return heuristic_cost
#
#     def get_costs_better_than_dispenser(self, possible_objects, target_locations, baseline_cost, num_needed, state):
#         """
#         Computes the number of objects whose minimum cost to any of the target locations is smaller than
#         the baseline cost (clipping it if greater than the number needed). It also calculates a lower
#         bound on the cost of using such objects.
#         """
#         costs_from_transit_locations = []
#         for obj in possible_objects:
#             obj_pos = obj.position
#             if obj_pos in state.player_positions:
#                 # If object is being carried by a player
#                 player = [p for p in state.players if p.position == obj_pos][0]
#                 # NOTE: not sure if this -1 is justified.
#                 # Made things work better in practice for greedy heuristic based agents.
#                 # For now this function is just used from there. Consider removing later if
#                 # greedy heuristic agents end up not being used.
#                 min_cost = self.motion_planner.min_cost_to_feature(player.pos_and_or, target_locations) - 1
#             else:
#                 # If object is on a counter
#                 min_cost = self.motion_planner.min_cost_between_features([obj_pos], target_locations)
#             costs_from_transit_locations.append(min_cost)
#
#         costs_better_than_dispenser = [cost for cost in costs_from_transit_locations if cost <= baseline_cost]
#         better_than_dispenser_total_cost = sum(np.sort(costs_better_than_dispenser)[:num_needed])
#         return len(costs_better_than_dispenser), better_than_dispenser_total_cost
#
#     def _calculate_heuristic_costs(self, debug=False):
#         """Pre-computes the costs between common trip types for this mdp"""
#         pot_locations = self.mdp.get_pot_locations()
#         delivery_locations = self.mdp.get_serving_locations()
#         dish_locations = self.mdp.get_dish_dispenser_locations()
#         onion_locations = self.mdp.get_onion_dispenser_locations()
#         tomato_locations = self.mdp.get_tomato_dispenser_locations()
#
#         heuristic_cost_dict = {
#             'pot-delivery': self.motion_planner.min_cost_between_features(pot_locations, delivery_locations, manhattan_if_fail=True),
#             'pot-cooking': 20, # this assume cooking time is always 20 timesteps
#             'dish-pot': self.motion_planner.min_cost_between_features(dish_locations, pot_locations, manhattan_if_fail=True)
#         }
#
#         onion_pot_cost = self.motion_planner.min_cost_between_features(onion_locations, pot_locations, manhattan_if_fail=True)
#         tomato_pot_cost = self.motion_planner.min_cost_between_features(tomato_locations, pot_locations, manhattan_if_fail=True)
#
#         if debug: print("Heuristic cost dict", heuristic_cost_dict)
#         assert onion_pot_cost != np.inf or tomato_pot_cost != np.inf
#         if onion_pot_cost != np.inf:
#             heuristic_cost_dict['onion-pot'] = onion_pot_cost
#         if tomato_pot_cost != np.inf:
#             heuristic_cost_dict['tomato-pot'] = tomato_pot_cost
#
#         return heuristic_cost_dict
#
#     # Deprecated. This is out of date with the current MDP, but is no longer needed, so deprecated
#     def simple_heuristic(self, state, time=0, debug=False):
#         """Simpler heuristic that tends to run faster than current one"""
#         # NOTE: State should be modified to have an order list w.r.t. which
#         # one can calculate the heuristic
#
#         objects_dict = state.unowned_objects_by_type
#         player_objects = state.player_objects_by_type
#         pot_states_scores_dict = self.mdp.get_pot_states_scores(state)
#         max_recipe_value = self.mdp.max_recipe_value(state)
#         num_deliveries_to_go = (DELIVERY_REW_THRES - state.delivery_rew)//max_recipe_value
#         num_full_soups_in_pots = sum(pot_states_scores_dict['cooking'] + pot_states_scores_dict['ready'])//max_recipe_value
#
#         pot_states_dict = self.mdp.get_pot_states(state)
#         partially_full_soups = self.mdp.get_partially_full_pots(pot_states_dict)
#         num_items_in_partially_full_pots = sum([len(state.get_object(loc).ingredients) for loc in partially_full_soups])
#
#         soups_in_transit = player_objects['soup']
#         dishes_in_transit = objects_dict['dish'] + player_objects['dish']
#         onions_in_transit = objects_dict['onion'] + player_objects['onion']
#         tomatoes_in_transit = objects_dict['tomato'] + player_objects['tomato']
#
#         num_pot_to_delivery = max([0, num_deliveries_to_go - len(soups_in_transit)])
#         num_dish_to_pot = max([0, num_pot_to_delivery - len(dishes_in_transit)])
#
#         # FIXME: the following logic might need to be discussed, when incoporating tomatoes
#         num_pots_to_be_filled = num_pot_to_delivery - num_full_soups_in_pots
#         num_onions_needed_for_pots = num_pots_to_be_filled * 3 - len(onions_in_transit) - num_items_in_partially_full_pots
#         num_tomatoes_needed_for_pots = 0
#         num_onion_to_pot = max([0, num_onions_needed_for_pots])
#         num_tomato_to_pot = max([0, num_tomatoes_needed_for_pots])
#
#         pot_to_delivery_costs = (self.heuristic_cost_dict['pot-delivery'] + self.heuristic_cost_dict['pot-cooking']) \
#                                 * num_pot_to_delivery
#         dish_to_pot_costs = self.heuristic_cost_dict['dish-pot'] * num_dish_to_pot
#
#         items_to_pot_costs = []
#         # FIXME: might want to change this for anything beyond 3-onion soup
#         if 'onion-pot' in self.heuristic_cost_dict.keys():
#             onion_to_pot_costs = self.heuristic_cost_dict['onion-pot'] * num_onion_to_pot
#             items_to_pot_costs.append(onion_to_pot_costs)
#         if 'tomato-pot' in self.heuristic_cost_dict.keys():
#             tomato_to_pot_costs = self.heuristic_cost_dict['tomato-pot'] * num_tomato_to_pot
#             items_to_pot_costs.append(tomato_to_pot_costs)
#
#         # NOTE: doesn't take into account that a combination of the two might actually be more advantageous.
#         # Might cause heuristic to be inadmissable in some edge cases.
#         # FIXME: only onion for now
#         items_to_pot_cost = onion_to_pot_costs
#
#         # num_pot_to_delivery added to account for the additional "INTERACT" to start soup cooking
#         heuristic_cost = (pot_to_delivery_costs + dish_to_pot_costs + num_pot_to_delivery + items_to_pot_cost) / 2
#
#         if not clean and debug:
#             env = OvercookedEnv.from_mdp(self.mdp)
#             env.state = state
#             print("\n" + "#" * 35)
#             print("Current state: (ml timestep {})\n".format(time))
#
#             print("# in transit: \t\t Soups {} \t Dishes {} \t Onions {}".format(
#                 len(soups_in_transit), len(dishes_in_transit), len(onions_in_transit)
#             ))
#
#             print("Trip costs: \t\t pot-del {} \t dish-pot {} \t onion-pot {}".format(
#                 pot_to_delivery_costs, dish_to_pot_costs, onion_to_pot_costs
#             ))
#
#             print(str(env) + "HEURISTIC: {}".format(heuristic_cost))
#         if heuristic_cost < 15:
#             print(heuristic_cost, (pot_to_delivery_costs, dish_to_pot_costs, num_pot_to_delivery, items_to_pot_cost))
#             print(self.mdp.state_string(state))
#         return heuristic_cost
