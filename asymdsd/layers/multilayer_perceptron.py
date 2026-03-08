from dataclasses import dataclass

from torch import nn

from .activation import ActivationLayer, is_gated_activation
from .normalization import NormalizationLayer


@dataclass
class MLPConfig:
    dims: list[int]
    norm_layer: NormalizationLayer | None = None
    act_layer: ActivationLayer = nn.GELU
    dropout_p: float = 0.0
    bias: bool = True


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 384,
        out_dim: int | None = None,
        *,
        norm_layer: NormalizationLayer | None = None,
        act_layer: ActivationLayer = nn.GELU,
        dropout_p: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_dim = out_dim or in_dim
        norm_layer = norm_layer or nn.Identity

        # When gated, the first linear layer has twice the number of dim.
        l1_hidden_dim = 2 * hidden_dim if is_gated_activation(act_layer) else hidden_dim

        self.norm = norm_layer(hidden_dim)
        self.linear_1 = nn.Linear(in_dim, l1_hidden_dim, bias=bias)
        self.linear_2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.norm(x)  # TODO: Move after dropout
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


# TODO: Make this the default MLP, and remove old one.
class MLPVarLen(nn.Module):
    def __init__(
        self,
        *dims: int,
        norm_layer: NormalizationLayer | None = None,
        act_layer: ActivationLayer = nn.GELU,
        dropout_p: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        n_layers = len(dims) - 1
        if n_layers < 1:
            raise ValueError(f"Expected at least 2 dimensions, got {len(dims)}")

        norm_layer = norm_layer or nn.Identity

        layers = []
        if n_layers > 1:
            for i in range(n_layers - 1):
                next_dim = (
                    2 * dims[i + 1] if is_gated_activation(act_layer) else dims[i + 1]
                )

                layers += [
                    nn.Linear(dims[i], next_dim, bias=bias),
                    norm_layer(dims[i + 1]),  # TODO: Move after dropout
                    act_layer(),
                    nn.Dropout(dropout_p),
                ]

        layers += [nn.Linear(dims[-2], dims[-1], bias=bias)]  # TODO: Consider False

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
