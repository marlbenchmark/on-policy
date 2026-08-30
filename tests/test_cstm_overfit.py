import torch
import torch.nn.functional as F

from onpolicy.algorithms.cstm_mappo.algorithm.teammate_model import TeammateModel


def test_single_head_overfits_fixed_minibatch():
    torch.manual_seed(11)
    model = TeammateModel(16, 8, num_teammates=2, action_dim=5)
    features = torch.randn(256, 16)
    targets = torch.stack((
        (features[:, 0] > 0).long(),
        2 + (features[:, 1] > 0).long(),
    ), dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    with torch.no_grad():
        _, initial_logits = model(features)
        initial_loss = F.cross_entropy(initial_logits.flatten(0, 1),
                                       targets.flatten()).item()
    for _ in range(250):
        _, logits = model(features)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        _, logits = model(features)
        final_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten()).item()
        accuracy = (logits.argmax(-1) == targets).float().mean().item()
        majority = torch.bincount(targets.flatten(), minlength=5).max().item() / targets.numel()
    assert final_loss < initial_loss * 0.2
    assert accuracy > 0.95
    assert accuracy > majority
