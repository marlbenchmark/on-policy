import copy

import numpy as np
import torch
from gym import spaces

from cstm_test_utils import make_args
from onpolicy.algorithms.selective_mappo.selective_actor import SelectiveActor


def _args(**overrides):
    values = dict(
        algorithm_name="selective_mappo",
        cstm_num_heads=3,
        cstm_use_separate_detector=True,
        cstm_random_prior_scale=0.5,
        cstm_use_uncertainty_feature=True,
        cstm_selector_hidden_dim=16,
        cstm_selector_initial_scale=200.0,
        cstm_selector_initial_threshold=0.03,
    )
    values.update(overrides)
    return make_args(**values)


def _actor():
    return SelectiveActor(
        _args(), spaces.Box(-1, 1, shape=(18,), dtype=np.float32),
        spaces.Discrete(5), 3)


def test_selector_is_monotonic_in_risk_for_fixed_context():
    torch.manual_seed(1)
    actor = _actor()
    context = torch.randn(7, actor.hidden_size + 8 + 1)
    low = actor.selector_gate(context, torch.full((7, 1), 0.01))
    high = actor.selector_gate(context, torch.full((7, 1), 0.05))
    assert torch.all(low > high)
    assert actor.uncertainty_scale.item() > 0


def test_only_selector_parameters_are_trainable():
    actor = _actor()
    assert actor.selector_parameters
    assert all(parameter.requires_grad for parameter in actor.selector_parameters)
    assert not any(parameter.requires_grad
                   for parameter in actor.b0_actor.parameters())
    assert not any(parameter.requires_grad
                   for parameter in actor.b2_actor.parameters())


def test_mixture_probabilities_and_log_prob_are_exact():
    torch.manual_seed(3)
    actor = _actor()
    obs = np.random.randn(6, 18).astype(np.float32)
    states = np.zeros((6, 1, 16), dtype=np.float32)
    masks = np.ones((6, 1), dtype=np.float32)
    mixed, _, _, diagnostics, b0_probs, b2_probs = \
        actor._distribution_outputs(obs, states, states.copy(), masks)
    expected = (diagnostics["b2_gate"] * b2_probs
                + diagnostics["fallback_probability"] * b0_probs)
    torch.testing.assert_close(mixed, expected)
    torch.testing.assert_close(
        mixed.sum(-1), torch.ones(mixed.shape[0]))
    actions = mixed.argmax(-1, keepdim=True)
    log_probs, _, _ = actor.evaluate_actions(
        obs, states, states.copy(), actions, masks)
    expected_log_probs = mixed.gather(-1, actions).log()
    torch.testing.assert_close(log_probs, expected_log_probs)


def test_selector_update_cannot_change_frozen_branch_tensors():
    torch.manual_seed(5)
    actor = _actor()
    before_b0 = copy.deepcopy(actor.b0_actor.state_dict())
    before_b2 = copy.deepcopy(actor.b2_actor.state_dict())
    optimizer = torch.optim.Adam(actor.selector_parameters, lr=1e-2)
    context = torch.randn(12, actor.hidden_size + 8 + 1)
    loss = actor.selector_gate(
        context, torch.rand(12, 1) * 0.05).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for key, value in actor.b0_actor.state_dict().items():
        torch.testing.assert_close(value, before_b0[key], rtol=0, atol=0)
    for key, value in actor.b2_actor.state_dict().items():
        torch.testing.assert_close(value, before_b2[key], rtol=0, atol=0)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("SELECTIVE_DIRECT_TESTS_PASSED", len(tests))
