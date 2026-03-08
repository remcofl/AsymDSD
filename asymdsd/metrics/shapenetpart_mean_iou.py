from typing import Any

import torch
from torchmetrics.functional.segmentation.mean_iou import (
    _mean_iou_update,
)
from torchmetrics.metric import Metric


# Custom metric based on MeanIoU from torchmetrics
class ShapeNetPartMeanIoU(Metric):
    instance_score: torch.Tensor
    class_score: torch.Tensor
    num_instances: torch.Tensor
    instances_per_class: torch.Tensor
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    parts_mask: torch.Tensor

    INSTANCE_CLASS_TO_SEGMENTATION_CLASS = [
        [0, 1, 2, 3],
        [4, 5],
        [6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23],
        [24, 25, 26, 27],
        [28, 29],
        [30, 31, 32, 33, 34, 35],
        [36, 37],
        [38, 39, 40],
        [41, 42, 43],
        [44, 45, 46],
        [47, 48, 49],
    ]
    NUM_PARTS_PER_CLS = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]

    def __init__(
        self,
        num_segmentation_classes: int,
        num_instance_classes: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_instance_classes = num_instance_classes
        self.num_segmentation_classes = num_segmentation_classes

        self.add_state(
            "instance_score",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )

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
        # Create a mask tensor of shape (num_instance_classes, num_segmentation_classes)
        mask = torch.zeros(
            (num_instance_classes, num_segmentation_classes),  # type: ignore
            dtype=torch.bool,
        )

        for instance_class, segmentation_classes in enumerate(
            ShapeNetPartMeanIoU.INSTANCE_CLASS_TO_SEGMENTATION_CLASS
        ):
            mask[instance_class, segmentation_classes] = True

        self.register_buffer("parts_mask", mask)

        self.add_state("num_instances", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        instance_classes: torch.Tensor,
    ) -> None:
        parts_mask = self.parts_mask[instance_classes]
        preds = preds.masked_fill(~parts_mask.unsqueeze(-1), float("-inf"))
        preds = preds.argmax(dim=1)

        intersection, union = _mean_iou_update(
            preds,
            target,
            self.num_segmentation_classes,
            include_background=True,  # type: ignore
        )
        preds = torch.nn.functional.one_hot(
            preds, num_classes=self.num_segmentation_classes
        ).movedim(-1, 1)
        target = torch.nn.functional.one_hot(
            target, num_classes=self.num_segmentation_classes
        ).movedim(-1, 1)

        reduce_axis = list(range(2, preds.ndim))
        intersection = torch.sum(preds & target, dim=reduce_axis)
        target_sum = torch.sum(target, dim=reduce_axis)
        pred_sum = torch.sum(preds, dim=reduce_axis)
        union = target_sum + pred_sum - intersection

        iou = intersection / union
        iou[iou.isnan() & parts_mask] = 1.0
        iou[~parts_mask] = torch.nan

        # Where union is 0, iou is nan. These segmentation classes should not contribute to the score.
        # Therefore, we can conviently use the nanmean.
        score = iou.nanmean(dim=1)

        self.instance_score += (
            score.sum()
        )  # Not mean, because this is numerically less stable.
        self.class_score.scatter_add_(0, instance_classes, score)
        self.instances_per_class.scatter_add_(
            0, instance_classes, torch.ones_like(self.instances_per_class)
        )
        self.num_instances += preds.shape[0]

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.instance_score / self.num_instances,
            (self.class_score / self.instances_per_class).mean(),
        )
