from abc import ABC, abstractmethod
from typing import Callable

from jsonargparse import lazy_instance
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from .common_types import Params
from .scheduling import CosineAnnealingWarmupSchedule
from .utils import init_lazy_defaults
from .weight_decay import WeightDecayScheduler

_DEFAULT_LR_SCHEDULE = lazy_instance(
    CosineAnnealingWarmupSchedule,
    base_value=5e-3,
    final_value=1e-7,
)


class OptimizerSpec(ABC):
    @init_lazy_defaults
    def __init__(
        self,
        lr: float | Callable[[int], float] = _DEFAULT_LR_SCHEDULE,
        weight_decay: float | Callable[[int], float] = 5e-2,
    ) -> None:
        self.lr = lr
        self.wd = weight_decay

        self.base_lr = 1.0 if callable(lr) else lr
        self.initial_wd = weight_decay(0) if callable(weight_decay) else weight_decay

    def get_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler | None:
        return LambdaLR(optimizer, self.lr) if callable(self.lr) else None

    def get_wd_scheduler(self, optimizer: Optimizer) -> WeightDecayScheduler | None:
        return WeightDecayScheduler(optimizer, self.wd) if callable(self.wd) else None

    @abstractmethod
    def get_optim(self, params: Params, lr_multiplier: float) -> Optimizer:
        pass

    @property
    @abstractmethod
    def optimizerCls(self) -> type[Optimizer]:
        pass


class AdamWSpec(OptimizerSpec):
    def __init__(
        self, betas: tuple[float, float] = (0.9, 0.999), **optim_kwargs
    ) -> None:
        super().__init__(**optim_kwargs)
        self.betas = betas

    def get_optim(self, params: Params, lr_multiplier: float = 1.0) -> Optimizer:
        return AdamW(
            params,
            lr=self.base_lr * lr_multiplier,
            weight_decay=self.initial_wd,  # Is initial weight decay
            betas=self.betas,
        )

    @property
    def optimizerCls(self):
        return AdamW


class SGDSpec(OptimizerSpec):
    def __init__(self, momentum: float = 0.9, **optim_kwargs) -> None:
        super().__init__(**optim_kwargs)
        self.momentum = momentum

    def get_optim(self, param: Params, lr_multiplier: float = 1.0) -> Optimizer:
        return SGD(
            param,
            lr=self.base_lr * lr_multiplier,
            weight_decay=self.initial_wd,
            momentum=self.momentum,
        )

    @property
    def optimizerCls(self):
        return SGD
