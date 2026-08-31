from collections import deque

import numpy as np


def teammate_position_slice(num_agents, num_landmarks, dim_p=2):
    """Return the simple_spread observation slice containing teammate positions.

    simple_spread observations are ordered as own velocity, own position,
    landmark-relative positions, teammate-relative positions, communication.
    """
    if num_agents < 2:
        raise ValueError("teammate corruption requires at least two agents")
    if num_landmarks < 1:
        raise ValueError("num_landmarks must be positive")
    start = 2 * dim_p + num_landmarks * dim_p
    stop = start + (num_agents - 1) * dim_p
    return slice(start, stop)


class TeammatePositionCorruptor:
    """Apply test-time corruption only to teammate-relative positions."""

    VALID_TYPES = ("clean", "noise", "mask", "delay")

    def __init__(self, corruption_type, level, num_agents, num_landmarks,
                 seed=0, dim_p=2):
        if corruption_type not in self.VALID_TYPES:
            raise ValueError("unknown corruption type: {}".format(corruption_type))
        if level < 0:
            raise ValueError("corruption level must be non-negative")
        if corruption_type == "mask" and level > 1:
            raise ValueError("mask probability must be in [0, 1]")
        if corruption_type == "delay" and int(level) != level:
            raise ValueError("delay must be an integer number of steps")
        self.corruption_type = corruption_type
        self.level = int(level) if corruption_type == "delay" else float(level)
        self.num_agents = num_agents
        self.num_teammates = num_agents - 1
        self.dim_p = dim_p
        self.position_slice = teammate_position_slice(
            num_agents, num_landmarks, dim_p)
        self.rng = np.random.RandomState(seed)
        self.history = deque(maxlen=max(1, int(self.level) + 1))

    def reset(self):
        self.history.clear()

    def transform(self, observations):
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[0] != self.num_agents:
            raise ValueError(
                "expected observations [num_agents, obs_dim], got {}".format(
                    observations.shape))
        if observations.shape[1] < self.position_slice.stop:
            raise ValueError("observation is too short for teammate positions")

        result = observations.copy()
        block = observations[:, self.position_slice].reshape(
            self.num_agents, self.num_teammates, self.dim_p)

        if self.corruption_type == "clean" or self.level == 0:
            return result
        if self.corruption_type == "noise":
            corrupted = block + self.rng.normal(
                0.0, self.level, size=block.shape).astype(np.float32)
        elif self.corruption_type == "mask":
            # One Bernoulli decision per perceived teammate, shared by x/y.
            keep = (self.rng.random_sample(
                (self.num_agents, self.num_teammates, 1)) >= self.level)
            corrupted = block * keep.astype(np.float32)
        else:
            self.history.append(block.copy())
            index = max(0, len(self.history) - 1 - self.level)
            corrupted = self.history[index]

        result[:, self.position_slice] = corrupted.reshape(
            self.num_agents, -1)
        return result
