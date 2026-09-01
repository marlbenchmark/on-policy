import numpy as np

from onpolicy.envs.mpe.perturbations import TeammatePositionCorruptor
from onpolicy.runner.shared.cstm_mpe_runner import CSTMMPErunner
from onpolicy.runner.shared.mpe_runner import MPERunner


def build_corruption_training(args, n_rollout_threads, num_agents):
    """Build a deterministic corruption mixture across rollout workers.

    Each worker keeps one sampled corruption condition for the full run.  The
    mixture is therefore represented across parallel workers rather than being
    resampled every episode.  Corruptor state (notably delay history) is reset
    at episode boundaries.
    """
    if args.scenario_name != "simple_spread":
        raise ValueError("corruption training currently supports simple_spread")
    clean_probability = args.cstm_selector_clean_probability
    if not 0.0 <= clean_probability <= 1.0:
        raise ValueError("clean probability must be in [0, 1]")
    if not (args.cstm_selector_noise_levels
            and args.cstm_selector_mask_levels
            and args.cstm_selector_delay_levels):
        raise ValueError("corruption level lists cannot be empty")
    rng = np.random.RandomState(args.seed + 15401)
    corruption_types = ("noise", "mask", "delay")
    conditions = []
    corruptors = []
    for rank in range(n_rollout_threads):
        if rng.random_sample() < clean_probability:
            corruption_type, level = "clean", 0.0
        else:
            corruption_type = corruption_types[rng.randint(3)]
            if corruption_type == "noise":
                level = rng.choice(args.cstm_selector_noise_levels)
            elif corruption_type == "mask":
                level = rng.choice(args.cstm_selector_mask_levels)
            else:
                level = rng.choice(args.cstm_selector_delay_levels)
        conditions.append((corruption_type, float(level)))
        corruptors.append(TeammatePositionCorruptor(
            corruption_type, level, num_agents, args.num_landmarks,
            seed=args.seed * 100003 + rank * 9176))
    return conditions, corruptors


class CorruptionTrainingMixin:
    """Apply B4's actor-observation corruption mix to a trainable policy."""

    def __init__(self, config):
        super().__init__(config)
        self.training_conditions, self.corruptors = build_corruption_training(
            self.all_args, self.n_rollout_threads, self.num_agents)

    def _corrupt_observations(self, observations):
        return np.stack([
            corruptor.transform(observation)
            for corruptor, observation in zip(self.corruptors, observations)
        ]).astype(np.float32)

    def collect(self, step):
        actor_obs = self._corrupt_observations(self.buffer.obs[step])
        # PPO log-probabilities must be recomputed from exactly the local
        # observations used during rollout.  ``share_obs`` remains untouched,
        # so the centralized critic continues to receive the clean state.
        self.buffer.obs[step] = actor_obs.copy()
        return super().collect(step)

    def insert(self, data):
        dones = data[2]
        for rank, done_row in enumerate(dones):
            if np.any(done_row):
                self.corruptors[rank].reset()
        return super().insert(data)


class RobustMPERunner(CorruptionTrainingMixin, MPERunner):
    pass


class RobustCSTMMPErunner(CorruptionTrainingMixin, CSTMMPErunner):
    pass
