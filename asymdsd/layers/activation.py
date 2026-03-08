import torch
from torch import nn

ActivationLayerStd = (
    type[nn.ReLU]
    | type[nn.LeakyReLU]
    | type[nn.GELU]
    | type[nn.SiLU]
    | type[nn.Tanh]
    | type[nn.Identity]
)


class GLU(nn.Module):
    def __init__(self, act_layer: ActivationLayerStd) -> None:
        super().__init__()
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, g = x.chunk(2, dim=-1)
        x = self.act(g) * x
        return x


class GEGLU(GLU):
    def __init__(self) -> None:
        super().__init__(nn.GELU)


class SwiGLU(GLU):
    def __init__(self) -> None:
        super().__init__(nn.SiLU)


ActivationLayer = ActivationLayerStd | type[GEGLU] | type[SwiGLU]


def is_gated_activation(act_layer: ActivationLayer) -> bool:
    return issubclass(act_layer, GLU)
