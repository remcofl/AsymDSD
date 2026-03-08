import inspect
from functools import wraps
from typing import Sequence

import torch
from torch import nn


def init_lazy_defaults(func):
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        for key, val in bound.arguments.items():
            if hasattr(val, "_lazy_init"):
                args = val.lazy_get_init_args().as_dict()
                # Take the original class and not the lazy wrapper
                obj = val.__class__.__bases__[1](**args)
                bound.arguments[key] = obj

        return func(*bound.args, **bound.kwargs)

    return wrapper


def set_cuda_float32_matmul_from_env_var():
    import os

    if "CUDA_MATMUL_TF32" in os.environ:
        setting = os.environ["CUDA_MATMUL_TF32"]
        torch.set_float32_matmul_precision(setting)
        # Note: cuDNN backend still uses float32


def xyz_view(points: torch.Tensor) -> torch.Tensor:
    return points[..., :3]  # Not a copy, but a view


def gather_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # -1 corresponds to num_masks
    var_shape = (x.shape[0], -1, x.shape[-1])
    return x[mask].reshape(var_shape)


def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    # lengths: (B, N)
    # mask: (B, max_length)
    mask = torch.arange(max_length, device=lengths.device).unsqueeze(
        0
    ) < lengths.unsqueeze(-1)
    return mask


def sequentialize_transform(
    transform: nn.Module | Sequence[nn.Module],
) -> nn.Module:
    if isinstance(transform, Sequence):
        transform = nn.Sequential(*transform)
    return transform


def compute_decay_fractional_update(
    decay: float, update_size: int | float, original_update_size: int | float
) -> float:
    return decay ** (update_size / original_update_size)


def compile_model(
    model: nn.Module,
    cache_size_limit: int = 16,
    suppress_errors: bool = True,
    **torch_compile_kwargs,
) -> nn.Module:
    if not torch_compile_kwargs.get("disable", False):
        # If OS does not support, it should avoid compile call at all.
        model = torch.compile(model, **torch_compile_kwargs)  # type: ignore

    from torch._dynamo import config as dynamo_config

    dynamo_config.cache_size_limit = cache_size_limit
    dynamo_config.suppress_errors = suppress_errors

    return model
