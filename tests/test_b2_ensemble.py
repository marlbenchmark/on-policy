import torch
import torch.nn.functional as F
import numpy as np
from gym import spaces

from cstm_test_utils import make_args
from onpolicy.algorithms.cstm_mappo.algorithm.dual_policy_actor import B1Actor
from onpolicy.algorithms.cstm_mappo.algorithm.teammate_model import TeammateModel
from onpolicy.utils.cstm_buffer import CSTMReplayBuffer


def test_single_head_has_zero_js_and_preserves_b1_shapes():
    model = TeammateModel(16, 8, 2, 5, num_heads=1)
    latent, logits, mean_probs, disagreement = model(torch.randn(7, 16))
    assert latent.shape == (7, 8)
    assert logits.shape == (7, 1, 2, 5)
    assert mean_probs.shape == (7, 2, 5)
    assert disagreement.shape == (7, 2)
    torch.testing.assert_close(disagreement, torch.zeros_like(disagreement))


def test_three_heads_are_distinct_and_js_is_normalized():
    torch.manual_seed(31)
    model = TeammateModel(16, 8, 2, 5, num_heads=3)
    _, logits, mean_probs, disagreement = model(torch.randn(32, 16))
    assert logits.shape == (32, 3, 2, 5)
    assert not torch.allclose(logits[:, 0], logits[:, 1])
    assert not torch.allclose(logits[:, 1], logits[:, 2])
    torch.testing.assert_close(mean_probs.sum(-1), torch.ones(32, 2))
    assert torch.all(disagreement >= 0)
    assert torch.all(disagreement <= 1)
    assert disagreement.mean() > 0


def test_randomized_priors_are_frozen_distinct_and_checkpointed():
    torch.manual_seed(41)
    model = TeammateModel(
        16, 8, 2, 5, num_heads=3, random_prior_scale=0.5)
    assert len(model.prior_decoders) == 3
    assert all(not parameter.requires_grad
               for parameter in model.prior_decoders.parameters())
    features = torch.randn(16, 16)
    with torch.no_grad():
        _, logits, _, disagreement = model(features)
    assert not torch.allclose(logits[:, 0], logits[:, 1])
    assert disagreement.mean() > 0

    restored = TeammateModel(
        16, 8, 2, 5, num_heads=3, random_prior_scale=0.5)
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        _, restored_logits, _, restored_disagreement = restored(features)
    torch.testing.assert_close(restored_logits, logits)
    torch.testing.assert_close(restored_disagreement, disagreement)


def test_bootstrap_heads_receive_different_subsets():
    rng = np.random.RandomState(19)
    active_masks = np.ones((128, 3, 2, 1), dtype=np.float32)
    masks = CSTMReplayBuffer.build_bootstrap_masks(
        active_masks, 3, 0.8, rng)
    assert masks.shape == (128, 3, 3, 2, 1)
    assert not np.array_equal(masks[:, :, 0], masks[:, :, 1])
    assert not np.array_equal(masks[:, :, 1], masks[:, :, 2])
    coverage = masks.mean().item()
    assert 0.70 < coverage < 0.90
    assert np.all(masks.sum(axis=(0, 1, 3, 4)) > 0)


def test_bootstrap_masks_are_stable_after_collection():
    rng = np.random.RandomState(43)
    active_masks = np.ones((4, 3, 2, 1), dtype=np.float32)
    stored = CSTMReplayBuffer.build_bootstrap_masks(
        active_masks, 3, 0.5, rng)
    first_epoch = stored.copy()
    second_epoch = stored.copy()
    np.testing.assert_array_equal(first_epoch, second_epoch)


def test_b2_actor_diagnostic_shapes_and_save_restore():
    torch.manual_seed(37)
    args = make_args(
        algorithm_name="ua_rep_mappo", cstm_num_heads=3,
        cstm_use_uncertainty_feature=True)
    obs_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
    action_space = spaces.Discrete(5)
    actor = B1Actor(args, obs_space, action_space, args.num_agents)
    obs = np.random.randn(6, 7).astype(np.float32)
    states = np.zeros((6, 1, args.hidden_size), dtype=np.float32)
    masks = np.ones((6, 1), dtype=np.float32)
    with torch.no_grad():
        before = actor(obs, states, masks, deterministic=True)
        heads, mean_probs, uncertainty, next_states = \
            actor.teammate_diagnostics(obs, states, masks)
    assert heads.shape == (6, 3, 2, 5)
    assert mean_probs.shape == (6, 2, 5)
    assert uncertainty.shape == (6, 2)
    assert next_states.shape == (6, 1, args.hidden_size)
    assert torch.isfinite(uncertainty).all()

    restored = B1Actor(args, obs_space, action_space, args.num_agents)
    restored.load_state_dict(actor.state_dict())
    with torch.no_grad():
        after = restored(obs, states, masks, deterministic=True)
        restored_diagnostics = restored.teammate_diagnostics(
            obs, states, masks)
    for expected, actual in zip(before, after):
        torch.testing.assert_close(expected, actual)
    for expected, actual in zip(
            (heads, mean_probs, uncertainty, next_states),
            restored_diagnostics):
        torch.testing.assert_close(expected, actual)


def test_teammate_outputs_support_uncertainty_gradients():
    torch.manual_seed(47)
    args = make_args(
        algorithm_name="ua_rep_mappo", cstm_num_heads=3,
        cstm_random_prior_scale=0.5,
        cstm_use_uncertainty_feature=True)
    actor = B1Actor(
        args, spaces.Box(-1, 1, shape=(18,), dtype=np.float32),
        spaces.Discrete(5), args.num_agents)
    obs = np.random.randn(8, 18).astype(np.float32)
    states = np.zeros((8, 1, args.hidden_size), dtype=np.float32)
    masks = np.ones((8, 1), dtype=np.float32)
    _, _, uncertainty, _ = actor.teammate_outputs(obs, states, masks)
    uncertainty.mean().backward()
    assert actor.teammate_model.decoder[2].weight.grad is not None
    assert all(parameter.grad is None
               for parameter in actor.teammate_model.prior_decoders.parameters())


def test_uncertainty_override_changes_only_policy_input():
    torch.manual_seed(53)
    args = make_args(
        algorithm_name="ua_rep_mappo", cstm_num_heads=3,
        cstm_use_uncertainty_feature=True)
    actor = B1Actor(
        args, spaces.Box(-1, 1, shape=(18,), dtype=np.float32),
        spaces.Discrete(5), args.num_agents)
    with torch.no_grad():
        actor.uncertainty_adapter.weight.fill_(1.0)
        actor.uncertainty_adapter.bias.zero_()
        features = torch.randn(8, args.hidden_size)
        predicted, heads, mean_probs, disagreement = \
            actor._teammate_features(features)
        actor.set_uncertainty_override(0.0)
        zeroed, zero_heads, zero_probs, zero_disagreement = \
            actor._teammate_features(features)
        actor.set_uncertainty_override(0.125)
        fixed, fixed_heads, fixed_probs, fixed_disagreement = \
            actor._teammate_features(features)
        actor.set_uncertainty_override(None)
        restored, _, _, _ = actor._teammate_features(features)

    assert not torch.allclose(predicted, zeroed)
    torch.testing.assert_close(fixed - zeroed, torch.full_like(fixed, 0.125))
    torch.testing.assert_close(restored, predicted)
    torch.testing.assert_close(zero_heads, heads)
    torch.testing.assert_close(fixed_heads, heads)
    torch.testing.assert_close(zero_probs, mean_probs)
    torch.testing.assert_close(fixed_probs, mean_probs)
    torch.testing.assert_close(zero_disagreement, disagreement)
    torch.testing.assert_close(fixed_disagreement, disagreement)


def test_three_heads_overfit_fixed_minibatch_without_collapsing_parameters():
    torch.manual_seed(23)
    model = TeammateModel(16, 8, 2, 5, num_heads=3)
    features = torch.randn(256, 16)
    targets = torch.stack((
        (features[:, 0] > 0).long(),
        2 + (features[:, 1] > 0).long(),
    ), dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    generator = torch.Generator().manual_seed(101)
    with torch.no_grad():
        _, initial_logits, _, _ = model(features)
        initial_loss = F.cross_entropy(
            initial_logits.flatten(0, 2),
            targets.unsqueeze(1).expand(-1, 3, -1).flatten()).item()
    for _ in range(250):
        _, logits, _, _ = model(features)
        expanded_targets = targets.unsqueeze(1).expand(-1, 3, -1)
        per_item = F.cross_entropy(
            logits.flatten(0, 2), expanded_targets.flatten(),
            reduction="none").reshape(256, 3, 2)
        masks = (torch.rand(256, 3, 2, generator=generator) < 0.8).float()
        loss = (per_item * masks).sum() / masks.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        _, logits, mean_probs, disagreement = model(features)
        final_loss = F.cross_entropy(
            logits.flatten(0, 2),
            targets.unsqueeze(1).expand(-1, 3, -1).flatten()).item()
        accuracy = (mean_probs.argmax(-1) == targets).float().mean().item()
    assert final_loss < initial_loss * 0.2
    assert accuracy > 0.95
    assert torch.isfinite(disagreement).all()
    assert not torch.equal(
        model.decoder[0].weight, model.extra_decoders[0][0].weight)
