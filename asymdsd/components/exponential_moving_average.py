import copy

import torch
from torch import nn


class EMA:
    def __init__(
        self,
        source_model: nn.Module | nn.ModuleDict,
        target_model: nn.Module | nn.ModuleDict | None,
    ) -> None:
        self.source_model = source_model
        self.target_model = target_model

        if self.target_model is None:
            self.target_model = copy.deepcopy(source_model)

        self.target_model.requires_grad_(False)

        # When ModuleDict, only perform EMA on modules that are in both source and target.
        if isinstance(self.source_model, nn.ModuleDict):
            if not isinstance(self.target_model, nn.ModuleDict):
                raise TypeError(
                    "target_model must be ModuleDict when source_model is ModuleDict"
                )
            keys = set(self.source_model.keys()) & set(self.target_model.keys())
            self.source_model = nn.ModuleList([self.source_model[k] for k in keys])
            self.ema_model = nn.ModuleList([self.target_model[k] for k in keys])
        else:
            self.ema_model = self.target_model

    def init_weights(self) -> None:
        # Should be called after target model is initialized
        for src, ema in zip(
            self.source_model.parameters(), self.ema_model.parameters()
        ):
            # Similar, but in-place of datach().clone()
            ema.data.copy_(src.data)

    @torch.no_grad()
    def update_parameters(self, decay: float = 0.0) -> None:
        if decay >= 1.0:
            return
        # Could also copy buffers if desired
        for src, ema in zip(
            self.source_model.parameters(), self.ema_model.parameters()
        ):
            ema.data = decay * ema.data + (1 - decay) * src.data
