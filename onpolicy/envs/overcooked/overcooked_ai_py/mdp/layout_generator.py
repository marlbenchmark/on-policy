import copy
import random

import numpy as np

from onpolicy.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from onpolicy.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe
from onpolicy.envs.overcooked.overcooked_ai_py.utils import rnd_int_uniform, rnd_uniform

EMPTY = " "
COUNTER = "X"
ONION_DISPENSER = "O"
TOMATO_DISPENSER = "T"
POT = "P"
DISH_DISPENSER = "D"
SERVING_LOC = "S"

CODE_TO_TYPE = {
    0: EMPTY,
    1: COUNTER,
    2: ONION_DISPENSER,
    3: TOMATO_DISPENSER,
    4: POT,
    5: DISH_DISPENSER,
    6: SERVING_LOC,
}
TYPE_TO_CODE = {v: k for k, v in CODE_TO_TYPE.items()}


def mdp_fn_random_choice(mdp_fn_choices):
    assert type(mdp_fn_choices) is list and len(mdp_fn_choices) > 0
    return random.choice(mdp_fn_choices)


"""
size_bounds: (min_layout_size, max_layout_size)
prop_empty: (min, max) proportion of empty space in generated layout
prop_feats: (min, max) proportion of counters with features on them
"""

DEFAULT_MDP_GEN_PARAMS = {
    "inner_shape": (5, 4),
    "prop_empty": 0.95,
    "prop_feats": 0.1,
    "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
    "recipe_values": [20],
    "recipe_times": [20],
    "display": False,
}


def DEFAILT_PARAMS_SCHEDULE_FN(outside_information):
    mdp_default_gen_params = {
        "inner_shape": (5, 4),
        "prop_empty": 0.95,
        "prop_feats": 0.1,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "recipe_values": [20],
        "recipe_times": [20],
        "display": False,
    }
    return mdp_default_gen_params


class MDPParamsGenerator(object):
    def __init__(self, params_schedule_fn):
        """
        params_schedule_fn (callable): the function to produce a set of mdp_params for a specific layout
        """
        assert callable(
            params_schedule_fn
        ), "params scheduling function must be a callable"
        self.params_schedule_fn = params_schedule_fn

    @staticmethod
    def from_fixed_param(mdp_params_always):
        # s naive schedule function that always return the same set of parameter
        naive_schedule_fn = lambda _ignored: mdp_params_always
        return MDPParamsGenerator(naive_schedule_fn)

    def generate(self, outside_information={}):
        """
        generate a set of mdp_params that can be used to generate a mdp
        outside_information (dict): passing in outside information
        """
        assert type(outside_information) is dict
        mdp_params = self.params_schedule_fn(outside_information)
        return mdp_params


DEFAULT_FEATURE_TYPES = (
    POT,
    ONION_DISPENSER,
    DISH_DISPENSER,
    SERVING_LOC,
)  # NOTE: TOMATO_DISPENSER is disabled by default


class LayoutGenerator(object):
    # NOTE: This class hasn't been tested extensively.

    def __init__(self, mdp_params_generator, outer_shape=(5, 4)):
        """
        Defines a layout generator that will return OvercoookedGridworld instances
        using mdp_params_generator
        """
        self.mdp_params_generator = mdp_params_generator
        self.outer_shape = outer_shape

    @staticmethod
    def mdp_gen_fn_from_dict(
        mdp_params, outer_shape=None, mdp_params_schedule_fn=None
    ):
        """
        mdp_params: one set of fixed mdp parameter used by the enviroment
        outer_shape: outer shape of the environment
        mdp_params_schedule_fn: the schedule for varying mdp params
        """
        # if outer_shape is not defined, we have to be using one of the defualt layout from names bank
        if outer_shape is None:
            assert type(mdp_params) is dict and "layout_name" in mdp_params
            mdp = OvercookedGridworld.from_layout_name(**mdp_params)
            mdp_fn = lambda _ignored: mdp
        else:
            # there is no schedule, we are using the same set of mdp_params all the time
            if mdp_params_schedule_fn is None:
                assert mdp_params is not None
                mdp_pg = MDPParamsGenerator.from_fixed_param(
                    mdp_params_always=mdp_params
                )
            else:
                assert mdp_params is None, (
                    "please remove the mdp_params from the variable, "
                    "because mdp_params_schedule_fn exist and we will "
                    "always use the schedule_fn if it exist"
                )
                mdp_pg = MDPParamsGenerator(
                    params_schedule_fn=mdp_params_schedule_fn
                )
            lg = LayoutGenerator(mdp_pg, outer_shape)
            mdp_fn = lg.generate_padded_mdp
        return mdp_fn

    def generate_padded_mdp(self, outside_information={}):
        """
        Return a PADDED MDP with mdp params specified in self.mdp_params
        """
        mdp_gen_params = self.mdp_params_generator.generate(
            outside_information
        )

        outer_shape = self.outer_shape
        if (
            "layout_name" in mdp_gen_params.keys()
            and mdp_gen_params["layout_name"] is not None
        ):
            mdp = OvercookedGridworld.from_layout_name(**mdp_gen_params)
            mdp_generator_fn = lambda: self.padded_mdp(mdp)
        else:
            required_keys = [
                "inner_shape",
                "prop_empty",
                "prop_feats",
                "display",
            ]
            # with generate_all_orders key start_all_orders will be generated inside make_new_layout method
            if not mdp_gen_params.get("generate_all_orders"):
                required_keys.append("start_all_orders")
            missing_keys = [
                k for k in required_keys if k not in mdp_gen_params.keys()
            ]
            if len(missing_keys) != 0:
                print("missing keys dict", mdp_gen_params)
            assert (
                len(missing_keys) == 0
            ), "These keys were missing from the mdp_params: {}".format(
                missing_keys
            )
            inner_shape = mdp_gen_params["inner_shape"]
            assert (
                inner_shape[0] <= outer_shape[0]
                and inner_shape[1] <= outer_shape[1]
            ), "inner_shape cannot fit into the outershap"
            layout_generator = LayoutGenerator(
                self.mdp_params_generator, outer_shape=self.outer_shape
            )

            if "feature_types" not in mdp_gen_params:
                mdp_gen_params["feature_types"] = DEFAULT_FEATURE_TYPES

            mdp_generator_fn = lambda: layout_generator.make_new_layout(
                mdp_gen_params
            )
        return mdp_generator_fn()

    @staticmethod
    def create_base_params(mdp_gen_params):
        assert mdp_gen_params.get("start_all_orders") or mdp_gen_params.get(
            "generate_all_orders"
        )
        mdp_gen_params = LayoutGenerator.add_generated_mdp_params_orders(
            mdp_gen_params
        )
        recipe_params = {
            "start_all_orders": mdp_gen_params["start_all_orders"]
        }
        if mdp_gen_params.get("start_bonus_orders"):
            recipe_params["start_bonus_orders"] = mdp_gen_params[
                "start_bonus_orders"
            ]
        if "recipe_values" in mdp_gen_params:
            recipe_params["recipe_values"] = mdp_gen_params["recipe_values"]
        if "recipe_times" in mdp_gen_params:
            recipe_params["recipe_times"] = mdp_gen_params["recipe_times"]
        return recipe_params

    @staticmethod
    def add_generated_mdp_params_orders(mdp_params):
        """
        adds generated parameters (i.e. generated orders) to mdp_params,
        returns onchanged copy of mdp_params when there is no "generate_all_orders" and "generate_bonus_orders" keys inside mdp_params
        """
        mdp_params = copy.deepcopy(mdp_params)
        if mdp_params.get("generate_all_orders"):
            all_orders_kwargs = copy.deepcopy(
                mdp_params["generate_all_orders"]
            )

            if all_orders_kwargs.get("recipes"):
                all_orders_kwargs["recipes"] = [
                    Recipe.from_dict(r) for r in all_orders_kwargs["recipes"]
                ]

            all_recipes = Recipe.generate_random_recipes(**all_orders_kwargs)
            mdp_params["start_all_orders"] = [r.to_dict() for r in all_recipes]
        else:
            Recipe.configure({})
            all_recipes = Recipe.ALL_RECIPES

        if mdp_params.get("generate_bonus_orders"):
            bonus_orders_kwargs = copy.deepcopy(
                mdp_params["generate_bonus_orders"]
            )

            if not bonus_orders_kwargs.get("recipes"):
                bonus_orders_kwargs["recipes"] = all_recipes

            bonus_recipes = Recipe.generate_random_recipes(
                **bonus_orders_kwargs
            )
            mdp_params["start_bonus_orders"] = [
                r.to_dict() for r in bonus_recipes
            ]
        return mdp_params

    def padded_mdp(self, mdp, display=False):
        """Returns a padded MDP from an MDP"""
        grid = Grid.from_mdp(mdp)
        padded_grid = self.embed_grid(grid)

        start_positions = self.get_random_starting_positions(padded_grid)
        mdp_grid = self.padded_grid_to_layout_grid(
            padded_grid, start_positions, display=display
        )
        return OvercookedGridworld.from_grid(mdp_grid)

    def make_new_layout(self, mdp_gen_params):
        return self.make_disjoint_sets_layout(
            inner_shape=mdp_gen_params["inner_shape"],
            prop_empty=mdp_gen_params["prop_empty"],
            prop_features=mdp_gen_params["prop_feats"],
            base_param=LayoutGenerator.create_base_params(mdp_gen_params),
            feature_types=mdp_gen_params["feature_types"],
            display=mdp_gen_params["display"],
        )

    def make_disjoint_sets_layout(
        self,
        inner_shape,
        prop_empty,
        prop_features,
        base_param,
        feature_types=DEFAULT_FEATURE_TYPES,
        display=True,
    ):
        grid = Grid(inner_shape)
        self.dig_space_with_disjoint_sets(grid, prop_empty)
        self.add_features(grid, prop_features, feature_types)

        padded_grid = self.embed_grid(grid)
        start_positions = self.get_random_starting_positions(padded_grid)
        mdp_grid = self.padded_grid_to_layout_grid(
            padded_grid, start_positions, display=display
        )
        return OvercookedGridworld.from_grid(mdp_grid, base_param)

    def padded_grid_to_layout_grid(
        self, padded_grid, start_positions, display=False
    ):
        if display:
            print("Generated layout")
            print(padded_grid)

        # Start formatting to actual OvercookedGridworld input type
        mdp_grid = padded_grid.convert_to_string()

        for i, pos in enumerate(start_positions):
            x, y = pos
            mdp_grid[y][x] = str(i + 1)

        return mdp_grid

    def embed_grid(self, grid):
        """Randomly embeds a smaller grid in a grid of size self.outer_shape"""
        # Check that smaller grid fits
        assert all(grid.shape <= self.outer_shape)

        padded_grid = Grid(self.outer_shape)
        x_leeway, y_leeway = self.outer_shape - grid.shape
        starting_x = np.random.randint(0, x_leeway) if x_leeway else 0
        starting_y = np.random.randint(0, y_leeway) if y_leeway else 0

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                item = grid.terrain_at_loc((x, y))
                # Abstraction violation
                padded_grid.mtx[x + starting_x][y + starting_y] = item

        return padded_grid

    def dig_space_with_disjoint_sets(self, grid, prop_empty):
        dsets = DisjointSets([])
        while not (
            grid.proportion_empty() > prop_empty and dsets.num_sets == 1
        ):
            valid_dig_location = False
            while not valid_dig_location:
                loc = grid.get_random_interior_location()
                valid_dig_location = grid.is_valid_dig_location(loc)

            grid.dig(loc)
            dsets.add_singleton(loc)

            for neighbour in grid.get_near_locations(loc):
                if dsets.contains(neighbour):
                    dsets.union(neighbour, loc)

    def make_fringe_expansion_layout(self, shape, prop_empty=0.1):
        grid = Grid(shape)
        self.dig_space_with_fringe_expansion(grid, prop_empty)
        self.add_features(grid)
        # print(grid)

    def dig_space_with_fringe_expansion(self, grid, prop_empty=0.1):
        starting_location = grid.get_random_interior_location()
        fringe = Fringe(grid)
        fringe.add(starting_location)

        while grid.proportion_empty() < prop_empty:
            curr_location = fringe.pop()
            grid.dig(curr_location)

            for location in grid.get_near_locations(curr_location):
                if grid.is_valid_dig_location(location):
                    fringe.add(location)

    def add_features(
        self, grid, prop_features=0, feature_types=DEFAULT_FEATURE_TYPES
    ):
        """
        Places one round of basic features and then adds random features
        until prop_features of valid locations are filled"""

        valid_locations = grid.valid_feature_locations()
        np.random.shuffle(valid_locations)
        assert len(valid_locations) > len(feature_types)

        num_features_placed = 0
        for location in valid_locations:
            current_prop = num_features_placed / len(valid_locations)
            if num_features_placed < len(feature_types):
                grid.add_feature(location, feature_types[num_features_placed])
            elif current_prop >= prop_features:
                break
            else:
                random_feature = np.random.choice(feature_types)
                grid.add_feature(location, random_feature)
            num_features_placed += 1

    def get_random_starting_positions(self, grid, divider_x=None):
        pos0 = grid.get_random_empty_location()
        pos1 = grid.get_random_empty_location()
        # NOTE: Assuming more than 1 empty location, hacky code
        while pos0 == pos1:
            pos0 = grid.get_random_empty_location()
        return pos0, pos1


class Grid(object):
    def __init__(self, shape):
        assert len(shape) == 2, "Grid must be 2 dimensional"
        grid = (np.ones(shape) * TYPE_TO_CODE[COUNTER]).astype(np.int)
        self.mtx = grid
        self.shape = np.array(shape)
        self.width = shape[0]
        self.height = shape[1]

    @staticmethod
    def from_mdp(mdp):
        terrain_matrix = np.array(mdp.terrain_mtx)
        mdp_grid = Grid((terrain_matrix.shape[1], terrain_matrix.shape[0]))
        for y in range(terrain_matrix.shape[0]):
            for x in range(terrain_matrix.shape[1]):
                feature = terrain_matrix[y][x]
                mdp_grid.mtx[x][y] = TYPE_TO_CODE[feature]
        return mdp_grid

    def terrain_at_loc(self, location):
        x, y = location
        return self.mtx[x][y]

    def dig(self, location):
        assert self.is_valid_dig_location(location)
        self.change_location(location, EMPTY)

    def add_feature(self, location, feature_string):
        assert self.is_valid_feature_location(location)
        self.change_location(location, feature_string)

    def change_location(self, location, feature_string):
        x, y = location
        self.mtx[x][y] = TYPE_TO_CODE[feature_string]

    def proportion_empty(self):
        flattened_grid = self.mtx.flatten()
        num_eligible = len(flattened_grid) - 2 * sum(self.shape) + 4
        num_empty = sum(
            [1 for x in flattened_grid if x == TYPE_TO_CODE[EMPTY]]
        )
        return float(num_empty) / num_eligible

    def get_near_locations(self, location):
        """Get neighbouring locations to the passed in location"""
        near_locations = []
        for d in Direction.ALL_DIRECTIONS:
            new_location = Action.move_in_direction(location, d)
            if self.is_in_bounds(new_location):
                near_locations.append(new_location)
        return near_locations

    def is_in_bounds(self, location):
        x, y = location
        return x >= 0 and y >= 0 and x < self.shape[0] and y < self.shape[1]

    def is_valid_dig_location(self, location):
        x, y = location

        # If already empty
        if self.location_is_empty(location):
            return False

        # If one of the edges of the map, or outside the map
        if (
            x <= 0
            or y <= 0
            or x >= self.shape[0] - 1
            or y >= self.shape[1] - 1
        ):
            return False
        return True

    def valid_feature_locations(self):
        valid_locations = []
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                location = (x, y)
                if self.is_valid_feature_location(location):
                    valid_locations.append(location)
        return np.array(valid_locations)

    def is_valid_feature_location(self, location):
        x, y = location

        # If is empty or has a feature on it
        if not self.mtx[x][y] == TYPE_TO_CODE[COUNTER]:
            return False

        # If outside the map
        if not self.is_in_bounds(location):
            return False

        # If location is next to at least one empty square
        if any(
            [
                loc
                for loc in self.get_near_locations(location)
                if CODE_TO_TYPE[self.terrain_at_loc(loc)] == EMPTY
            ]
        ):
            return True
        else:
            return False

    def location_is_empty(self, location):
        x, y = location
        return self.mtx[x][y] == TYPE_TO_CODE[EMPTY]

    def get_random_interior_location(self):
        rand_x = np.random.randint(low=1, high=self.shape[0] - 1)
        rand_y = np.random.randint(low=1, high=self.shape[1] - 1)
        return rand_x, rand_y

    def get_random_empty_location(self):
        is_empty = False
        while not is_empty:
            loc = self.get_random_interior_location()
            is_empty = self.location_is_empty(loc)
        return loc

    def convert_to_string(self):
        rows = []
        for y in range(self.shape[1]):
            column = []
            for x in range(self.shape[0]):
                column.append(CODE_TO_TYPE[self.mtx[x][y]])
            rows.append(column)
        string_grid = np.array(rows)
        assert np.array_equal(
            string_grid.T.shape, self.shape
        ), "{} vs {}".format(string_grid.shape, self.shape)
        return string_grid

    def __repr__(self):
        s = ""
        for y in range(self.shape[1]):
            for x in range(self.shape[0]):
                s += CODE_TO_TYPE[self.mtx[x][y]]
                s += " "
            s += "\n"
        return s


class Fringe(object):
    def __init__(self, grid):
        self.fringe_list = []
        self.distribution = []
        self.grid = grid

    def add(self, item):
        if item not in self.fringe_list:
            self.fringe_list.append(item)
            self.update_probs()

    def pop(self):
        assert len(self.fringe_list) > 0
        choice_idx = np.random.choice(
            len(self.fringe_list), p=self.distribution
        )
        removed_pos = self.fringe_list.pop(choice_idx)
        self.update_probs()
        return removed_pos

    def update_probs(self):
        self.distribution = np.ones(len(self.fringe_list)) / len(
            self.fringe_list
        )


class DisjointSets(object):
    """A simple implementation of the Disjoint Sets data structure.

    Implements path compression but not union-by-rank.

    Taken from https://github.com/HumanCompatibleAI/planner-inference
    """

    def __init__(self, elements):
        self.num_elements = len(elements)
        self.num_sets = len(elements)
        self.parents = {element: element for element in elements}

    def is_connected(self):
        return self.num_sets == 1

    def get_num_elements(self):
        return self.num_elements

    def contains(self, element):
        return element in self.parents

    def add_singleton(self, element):
        assert not self.contains(element)
        self.num_elements += 1
        self.num_sets += 1
        self.parents[element] = element

    def find(self, element):
        parent = self.parents[element]
        if element == parent:
            return parent

        result = self.find(parent)
        self.parents[element] = result
        return result

    def union(self, e1, e2):
        p1, p2 = map(self.find, (e1, e2))
        if p1 != p2:
            self.num_sets -= 1
            self.parents[p1] = p2
