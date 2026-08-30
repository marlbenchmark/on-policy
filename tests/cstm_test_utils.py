from argparse import Namespace


def make_args(**overrides):
    values = dict(
        hidden_size=16,
        gain=0.01,
        use_orthogonal=True,
        use_policy_active_masks=True,
        use_naive_recurrent_policy=False,
        use_recurrent_policy=True,
        recurrent_N=1,
        use_feature_normalization=True,
        use_ReLU=True,
        stacked_frames=1,
        use_stacked_frames=False,
        layer_N=1,
        algorithm_name="cstm_mappo",
        cstm_latent_dim=8,
        cstm_aux_coef=0.1,
        cstm_use_teammate_policy=True,
        num_agents=3,
        episode_length=4,
        n_rollout_threads=2,
        gamma=0.99,
        gae_lambda=0.95,
        use_gae=True,
        use_popart=False,
        use_valuenorm=False,
        use_proper_time_limits=False,
    )
    values.update(overrides)
    return Namespace(**values)
