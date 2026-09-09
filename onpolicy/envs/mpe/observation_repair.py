"""Stateful test-time repair for corrupted simple_spread teammate positions."""

import numpy as np

from onpolicy.envs.mpe.perturbations import teammate_position_slice


class RiskTriggeredObservationRepair:
    """Repair teammate-position slots using only information available online.

    ``mask_hold`` only fills exact zero-vector masks.  The other modes also
    intervene for every teammate slot of an agent when its frozen detector risk
    exceeds ``risk_threshold``.  No ground-truth corruption labels are used.
    """

    VALID_MODES = ("mask_hold", "risk_hold", "risk_velocity", "risk_smooth",
                   "always_smooth", "random_smooth", "oracle_smooth")

    def __init__(self, mode, num_agents, num_landmarks, risk_threshold=0.03,
                 smooth_alpha=0.35, max_velocity=0.20, zero_tolerance=1e-8):
        if mode not in self.VALID_MODES:
            raise ValueError("unknown repair mode: {}".format(mode))
        if not 0.0 <= smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be in [0, 1]")
        if max_velocity < 0.0:
            raise ValueError("max_velocity must be non-negative")
        self.mode = mode
        self.num_agents = num_agents
        self.num_teammates = num_agents - 1
        self.position_slice = teammate_position_slice(
            num_agents, num_landmarks)
        self.risk_threshold = float(risk_threshold)
        self.smooth_alpha = float(smooth_alpha)
        self.max_velocity = float(max_velocity)
        self.zero_tolerance = float(zero_tolerance)
        self.reset()

    def reset(self):
        shape = (self.num_agents, self.num_teammates, 2)
        self.last = np.zeros(shape, dtype=np.float32)
        self.velocity = np.zeros(shape, dtype=np.float32)
        self.valid = np.zeros(shape[:-1], dtype=bool)

    def transform(self, observations, risk, trigger_mask=None):
        observations = np.asarray(observations, dtype=np.float32)
        risk = np.asarray(risk, dtype=np.float32).reshape(self.num_agents)
        if observations.ndim != 2 or observations.shape[0] != self.num_agents:
            raise ValueError("expected [num_agents, obs_dim], got {}".format(
                observations.shape))
        result = observations.copy()
        incoming = observations[:, self.position_slice].reshape(
            self.num_agents, self.num_teammates, 2).copy()
        missing = np.linalg.norm(incoming, axis=-1) <= self.zero_tolerance
        if self.mode == "mask_hold":
            requested = missing
        elif self.mode == "always_smooth":
            requested = np.ones_like(missing, dtype=bool)
        elif self.mode in ("random_smooth", "oracle_smooth"):
            if trigger_mask is None:
                raise ValueError("{} requires trigger_mask".format(self.mode))
            trigger_mask = np.asarray(trigger_mask, dtype=bool)
            if trigger_mask.shape != missing.shape:
                raise ValueError("trigger_mask shape {} does not match {}".format(
                    trigger_mask.shape, missing.shape))
            requested = missing | trigger_mask
        else:
            requested = missing | (risk[:, None] > self.risk_threshold)
        was_valid = self.valid.copy()
        intervene = requested & was_valid

        clipped_velocity = np.clip(
            self.velocity, -self.max_velocity, self.max_velocity)
        if self.mode == "risk_velocity":
            estimate = self.last + clipped_velocity
        else:
            estimate = self.last
        repaired = incoming.copy()
        if self.mode in ("risk_smooth", "always_smooth", "random_smooth",
                          "oracle_smooth"):
            blended = (self.smooth_alpha * incoming
                       + (1.0 - self.smooth_alpha) * estimate)
            repaired[intervene] = blended[intervene]
            repaired[missing & self.valid] = estimate[missing & self.valid]
        else:
            repaired[intervene] = estimate[intervene]

        initialize = (~was_valid) & (~missing)
        if np.any(initialize):
            self.velocity[initialize] = 0.0
            self.last[initialize] = incoming[initialize]
            self.valid[initialize] = True

        accepted_clean = (~requested) & (~missing) & was_valid
        if np.any(accepted_clean):
            delta = np.clip(
                incoming - self.last, -self.max_velocity, self.max_velocity)
            self.velocity[accepted_clean] = delta[accepted_clean]
            self.last[accepted_clean] = incoming[accepted_clean]

        advanced = intervene & (self.mode in (
            "risk_velocity", "risk_smooth", "always_smooth",
            "random_smooth", "oracle_smooth"))
        if np.any(advanced):
            self.last[advanced] = repaired[advanced]

        result[:, self.position_slice] = repaired.reshape(
            self.num_agents, -1)
        stats = {
            "requested_rate": float(requested.mean()),
            "intervention_rate": float(intervene.mean()),
            "missing_rate": float(missing.mean()),
            "mean_abs_change": float(np.abs(repaired - incoming).mean()),
        }
        return result, stats
