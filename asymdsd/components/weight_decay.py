from typing import Callable

import torch

from .scheduling import Scheduler


class WeightDecayScheduler(Scheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        weight_decay: float | Callable[[int], float] = 1e-3,
    ) -> None:
        super().__init__(weight_decay=weight_decay)
        self.optimizer = optimizer

        self._update_weight_decay()

    @property
    def last_weight_decay(self) -> float:
        return self.value["weight_decay"]

    def _update_weight_decay(self) -> None:
        weight_decay = self.value["weight_decay"]
        for group in self.optimizer.param_groups:
            group["weight_decay"] = weight_decay

    def step(self) -> None:
        super().step()
        self._update_weight_decay()
