from typing import TYPE_CHECKING, Callable, Sequence, TypeVar, Union

if TYPE_CHECKING:
    from .scheduling import Schedule

from pathlib import Path

import torch
from torch import nn
from torch.optim.optimizer import ParamsT as Params

T = TypeVar("T")

PathLike = Union[str, Path]
OptionalTensor = torch.Tensor | None
OptionalListTensor = list[torch.Tensor] | None
OneOrSequence_T = T | Sequence[T]
LayerFn = nn.Module | Callable[[torch.Tensor], torch.Tensor]
FloatMayCall = Union[float, "Schedule", Callable[[int], float]]
