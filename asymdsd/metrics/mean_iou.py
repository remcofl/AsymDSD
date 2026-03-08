from typing import Any

import torch
from torchmetrics.functional.segmentation.mean_iou import (
    _mean_iou_update,
)
from torchmetrics.metric import Metric
from typing_extensions import Literal


# Custom metric based on MeanIoU from torchmetrics
class MeanIoU(Metric):
    instance_score: torch.Tensor
    class_score: torch.Tensor
    num_batches: torch.Tensor
    instances_per_class: torch.Tensor
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        num_segmentation_classes: int,
        num_instance_classes: int | None = None,
        instance_mean: bool = True,
        instance_class_mean: bool = True,
        input_format: Literal["one-hot", "index"] = "one-hot",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if instance_class_mean and num_instance_classes is None:
            raise ValueError(
                "Expected argument `num_object_classes` to be a positive integer, but got None."
            )

        self.num_instance_classes = num_instance_classes
        self.num_segmentation_classes = num_segmentation_classes
        self.instance_mean = instance_mean
        self.instance_class_mean = instance_class_mean
        self.input_format = input_format

        if instance_mean:
            self.add_state(
                "instance_score",
                default=torch.zeros(1),
                dist_reduce_fx="sum",
            )
        if instance_class_mean:
            self.add_state(
                "class_score",
                default=torch.zeros(num_instance_classes),  # type: ignore
                dist_reduce_fx="sum",
            )
            self.add_state(
                "instances_per_class",
                default=torch.zeros(num_instance_classes),  # type: ignore
                dist_reduce_fx="sum",
            )

        self.add_state("num_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        instance_classes: torch.Tensor | None,
    ) -> None:
        intersection, union = _mean_iou_update(
            preds,
            target,
            self.num_segmentation_classes,
            include_background=True,
            input_format=self.input_format,  # type: ignore
        )
        iou = intersection / union
        # Where union is 0, iou is nan. These segmentation classes should not contribute to the score.
        # Therefore, we can conviently use the nanmean.
        score = iou.nanmean(dim=1)

        self.instance_score += score.mean()
        self.num_batches += 1

        if instance_classes is not None:
            self.class_score.scatter_add_(0, instance_classes, score)
            self.instances_per_class.scatter_add_(
                0, instance_classes, torch.ones_like(self.instances_per_class)
            )

    def compute(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return (
            self.instance_score / self.num_batches if self.instance_mean else None,
            (self.class_score / self.instances_per_class).mean()
            if self.instance_class_mean
            else None,
        )
