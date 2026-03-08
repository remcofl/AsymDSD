from functools import partial

import torch
from torch import nn


class KoLeoLoss(nn.Module):
    def __init__(self, input_is_normalized: bool = False, eps=1e-6):
        super().__init__()
        self.pdist = nn.PairwiseDistance(p=2)
        self.norm = (
            nn.Identity()
            if input_is_normalized
            else partial(nn.functional.normalize, p=2, dim=-1, eps=eps)
        )
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm(x)  # (B, C, F)
        # Dot product between batches for all C.
        # Note there is no product between different crops.
        cosine_sim = torch.einsum("acf,bcf->cab", x, x)
        # Set diagonal to -1 (max dist)
        cosine_sim.diagonal(dim1=1, dim2=2).fill_(-1)

        # Get best match for each embedding
        idx = cosine_sim.max(dim=-1).indices  # (C, B)
        idx = idx.transpose(0, 1).unsqueeze(-1).expand_as(x)  # (B, C, F)
        x_match = torch.gather(x, 0, idx)

        min_dist = self.pdist(x, x_match)  #
        loss = -torch.log(min_dist + self.eps).mean()
        return loss
