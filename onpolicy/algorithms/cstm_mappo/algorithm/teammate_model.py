import torch
import torch.nn as nn

from onpolicy.algorithms.utils.util import init


class TeammateModel(nn.Module):
    """LIAM-style latent and a single teammate-action decoder.

    The input is the recurrent actor feature, hence it is computed only from
    the controlled agent's local observation history.  Targets are accepted
    by the trainer, never by this module's inference path.
    """

    def __init__(self, hidden_size, latent_dim, num_teammates, action_dim,
                 use_orthogonal=True):
        super().__init__()
        if num_teammates < 1:
            raise ValueError("CSTM requires at least two agents")
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        def init_(module, gain=1.0):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.num_teammates = num_teammates
        self.action_dim = action_dim
        self.latent = init_(nn.Linear(hidden_size, latent_dim))
        self.decoder = nn.Sequential(
            init_(nn.Linear(latent_dim, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, num_teammates * action_dim), gain=0.01),
        )

    def forward(self, actor_features):
        latent = torch.tanh(self.latent(actor_features))
        logits = self.decoder(latent)
        logits = logits.reshape(-1, self.num_teammates, self.action_dim)
        return latent, logits
