import os
import pickle

from onpolicy.envs.overcooked.overcooked_ai_py.static import PLANNERS_DIR
from onpolicy.envs.overcooked.overcooked_ai_py.utils import load_dict_from_file


def load_saved_action_manager(filename):
    with open(os.path.join(PLANNERS_DIR, filename), "rb") as f:
        mlp_action_manager = pickle.load(f)
        return mlp_action_manager


def load_saved_motion_planner(filename):
    with open(os.path.join(PLANNERS_DIR, filename), "rb") as f:
        motion_planner = pickle.load(f)
        return motion_planner
