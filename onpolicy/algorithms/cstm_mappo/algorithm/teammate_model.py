import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.utils.util import init


class TeammateModel(nn.Module):
    """LIAM-style latent with one or more teammate-action decoders.

    The input is the recurrent actor feature, hence it is computed only from
    the controlled agent's local observation history.  Targets are accepted
    by the trainer, never by this module's inference path.  Multiple decoder
    heads form the B2 bootstrap ensemble; a single head preserves B1.
    """

    def __init__(self, hidden_size, latent_dim, num_teammates, action_dim,
                 use_orthogonal=True, num_heads=1, random_prior_scale=0.0):
        super().__init__()
        if num_teammates < 1:
            raise ValueError("CSTM requires at least two agents")
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        def init_(module, gain=1.0):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.num_teammates = num_teammates
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.random_prior_scale = float(random_prior_scale)
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if self.random_prior_scale < 0:
            raise ValueError("random_prior_scale must be non-negative")
        self.latent = init_(nn.Linear(hidden_size, latent_dim))

        def make_decoder(final_gain=0.01):
            return nn.Sequential(
                init_(nn.Linear(latent_dim, hidden_size)),
                nn.ReLU(),
                init_(nn.Linear(
                    hidden_size, num_teammates * action_dim), gain=final_gain),
            )

        # Keep this name for exact B1 checkpoint compatibility.
        self.decoder = make_decoder()
        self.extra_decoders = nn.ModuleList(
            [make_decoder() for _ in range(num_heads - 1)])
        self.prior_decoders = nn.ModuleList()
        if num_heads > 1 and self.random_prior_scale > 0:
            # Fixed randomized prior functions preserve epistemic diversity
            # away from supervised data while leaving B1/legacy B2 unchanged.
            self.prior_decoders.extend(
                make_decoder(final_gain=1.0) for _ in range(num_heads))
            for parameter in self.prior_decoders.parameters():
                parameter.requires_grad_(False)

    def forward(self, actor_features):
        latent = torch.tanh(self.latent(actor_features))
        decoders = [self.decoder] + list(self.extra_decoders)
        logits = []
        for head, decoder in enumerate(decoders):
            value = decoder(latent)
            if self.prior_decoders:
                value = value + self.random_prior_scale * \
                    self.prior_decoders[head](latent.detach())
            logits.append(value.reshape(
                -1, self.num_teammates, self.action_dim))
        head_logits = torch.stack(logits, dim=1)
        head_probs = F.softmax(head_logits, dim=-1)
        mean_probs = head_probs.mean(dim=1)

        if self.num_heads == 1:
            disagreement = torch.zeros(
                mean_probs.shape[:-1], dtype=mean_probs.dtype,
                device=mean_probs.device)
        else:
            eps = torch.finfo(head_probs.dtype).eps
            mean_entropy = -(mean_probs * mean_probs.clamp_min(eps).log()).sum(-1)
            head_entropy = -(head_probs * head_probs.clamp_min(eps).log()).sum(-1)
            disagreement = (
                mean_entropy - head_entropy.mean(dim=1)
            ) / math.log(self.action_dim)
            disagreement = disagreement.clamp_min(0.0)
        return latent, head_logits, mean_probs, disagreement
