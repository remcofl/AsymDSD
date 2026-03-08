# Modified from https://github.com/charlesq34/pointnet2/blob/42926632a3c33461aebfbee2d829098b30a23aaa/utils/pointnet_util.py

import torch
import torch.nn as nn

from . import ActivationLayer, NormalizationLayer, TransposeBatchNorm1d


def dist_squared(points_1, points_2):
    # Alternative cdist ** 2
    dist = -2 * torch.matmul(points_1, points_2.transpose(1, 2))
    dist += torch.sum(points_1**2, dim=-1, keepdim=True)
    dist += torch.sum(points_2**2, dim=-1).unsqueeze(1)
    return dist


class PointUpsampling(nn.Module):
    def __init__(
        self,
        *dims: int,
        norm_layer: NormalizationLayer = TransposeBatchNorm1d,
        act_layer: ActivationLayer = nn.GELU,
        bias: bool = False,
        top_k: int = 3,
    ):
        super().__init__()
        self.top_k = top_k
        layers = []

        # Standard MLP without the last layer
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            layers.append(norm_layer(dims[i + 1]))
            layers.append(act_layer())

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        super_xyz: torch.Tensor,
        super_point_features: torch.Tensor,
        xyz: torch.Tensor,
        point_features: torch.Tensor,
    ):
        B, N, C = xyz.shape
        S = super_xyz.shape[1]
        F = super_point_features.shape[-1]

        if S == 1:
            interp_points = super_point_features.expand(-1, N, -1)
        else:
            dists = dist_squared(xyz, super_xyz)
            dists, idx = torch.topk(
                dists, self.top_k, dim=-1, largest=False, sorted=False
            )

            dists.clamp_(min=0)
            weight = torch.reciprocal(dists + torch.finfo(dists.dtype).eps)
            weight /= weight.sum(dim=2, keepdim=True)

            # No gather, as batch_indices does not require grad.
            batch_indices = (
                torch.arange(B, dtype=torch.long, device=xyz.device)
                .view(B, 1, 1)
                .expand(B, idx.shape[1], self.top_k)
            )

            gathered_features = super_point_features[batch_indices, idx]

            interp_points = torch.sum(gathered_features * weight.unsqueeze(-1), dim=2)

        new_points = (
            torch.cat([point_features, interp_points], dim=-1)
            if point_features is not None
            else interp_points
        )

        return self.mlp(new_points)
