import numpy as np

"""
NOTE: Currently under construction...


TODO: stretch goal of taking object-oriented approach to trajectories by creating Trajectory class. 
This would require changes both throughout this repo and overcooked-ai repo, so it's blue sky goal for now


This file's utility functions represents a primitive first-step towards treating trajectories as first class objects


The standard format for Overcooked trajectories is:
    trajs = {
        # With shape (n_episodes, game_len), where game_len might vary across games:
        "ep_states":    [ [traj_1_states], [traj_2_states], ... ],                          # Individual trajectory states
        "ep_actions":   [ [traj_1_joint_actions], [traj_2_joint_actions], ... ],            # Trajectory joint actions, by agent
        "ep_rewards":   [ [traj_1_timestep_rewards], [traj_2_timestep_rewards], ... ],      # (Sparse) reward values by timestep
        "ep_dones":     [ [traj_1_timestep_dones], [traj_2_timestep_dones], ... ],          # Done values (should be all 0s except last one for each traj) TODO: add this to traj checks
        "ep_infos":     [ [traj_1_timestep_infos], [traj_2_traj_1_timestep_infos], ... ],   # Info dictionaries

        # With shape (n_episodes, ):
        "ep_returns":   [ cumulative_traj1_reward, cumulative_traj2_reward, ... ],          # Sum of sparse rewards across each episode
        "ep_lengths":   [ traj1_length, traj2_length, ... ],                                # Lengths (in env timesteps) of each episode
        "mdp_params":   [ traj1_mdp_params, traj2_mdp_params, ... ],                        # Custom Mdp params to for each episode
        "env_params":   [ traj1_env_params, traj2_env_params, ... ],                        # Custom Env params for each episode

        # Custom metadata key value pairs
        "metadatas":   [{custom metadata key:value pairs for traj 1}, {...}, ...]          # Each metadata dictionary is of similar format to the trajectories dictionary
    }
"""

TIMESTEP_TRAJ_KEYS = set(
    ["ep_states", "ep_actions", "ep_rewards", "ep_dones", "ep_infos"]
)
EPISODE_TRAJ_KEYS = set(
    ["ep_returns", "ep_lengths", "mdp_params", "env_params"]
)
DEFAULT_TRAJ_KEYS = set(
    list(TIMESTEP_TRAJ_KEYS) + list(EPISODE_TRAJ_KEYS) + ["metadatas"]
)


def get_empty_trajectory():
    return {k: [] if k != "metadatas" else {} for k in DEFAULT_TRAJ_KEYS}


def append_trajectories(traj_one, traj_two):
    # Note: Drops metadatas for now
    if not traj_one and not traj_two:
        return {}
    if not traj_one:
        traj_one = get_empty_trajectory()
    if not traj_two:
        traj_two = get_empty_trajectory()

    if (
        set(traj_one.keys()) != DEFAULT_TRAJ_KEYS
        or set(traj_two.keys()) != DEFAULT_TRAJ_KEYS
    ):
        raise ValueError("Trajectory key mismatch!")

    appended_traj = {"metadatas": {}}
    for k in traj_one:
        if k != "metadatas":
            traj_one_value = traj_one[k]
            traj_two_value = traj_two[k]
            assert type(traj_one_value) == type(
                traj_two_value
            ), "mismatched trajectory types!"

            if type(traj_one_value) == list:
                appended_traj[k] = traj_one_value + traj_two_value
            else:
                appended_traj[k] = np.concatenate(
                    [traj_one_value, traj_two_value], axis=0
                )

    return appended_traj
