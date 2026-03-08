import copy
from enum import StrEnum, auto
from typing import Any

import lightning as L
import torch
import torchmetrics
from jsonargparse import lazy_instance
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from torch import nn

from ..components import *
from ..components.checkpointing_utils import load_module_from_checkpoint
from ..components.scheduling import Schedule
from ..components.utils import (
    init_lazy_defaults,
    lengths_to_mask,
    sequentialize_transform,
)
from ..data import PCFieldKey, SupervisedPCDataModule
from ..defaults import *
from ..layers import *
from ..layers.patchify import PatchPoints
from ..layers.tokenization import *
from ..loggers import get_default_logger
from .point_encoder import (
    DEFAULT_POINT_ENCODER,
    PointEncoder,
    PointEncoderOutput,
)

logger = get_default_logger()

DEFAULT_CLS_HEAD_CONFIG = lazy_instance(
    MLPConfig,
    dims=[256, 256],
    dropout_p=0.5,
    norm_layer=nn.BatchNorm1d,
    bias=False,
)


class ClassificationHeadType(StrEnum):
    LINEAR = auto()
    MLP = auto()


class NeuralClassifier(L.LightningModule):
    DEFAULT_BATCH_SIZE = 32

    @init_lazy_defaults
    def __init__(
        self,
        point_encoder: PointEncoder = DEFAULT_POINT_ENCODER,
        encoder_ckpt_path: str | None = None,
        encoder_choice: EncoderBranch | str = EncoderBranch.TEACHER,
        freeze_encoder: bool | int = False,
        map_avg_pooling: bool = True,
        map_max_pooling: bool = False,
        map_cls_token: bool = False,
        map_attn_pooling: bool | int = False,
        classification_head_type: ClassificationHeadType = ClassificationHeadType.LINEAR,
        mlp_head_config: MLPConfig | None = DEFAULT_CLS_HEAD_CONFIG,
        num_classes: int | None = None,
        subsampling_transform: SubsamplingTransform | None = None,
        aug_transform: AugmentationTransform | None = DEFAULT_AUG_TRANSFORM,
        norm_transform: NormalizationTransform | None = DEFAULT_NORM_TRANSFORM,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_epochs: int | None = 100,
        max_steps: int | None = None,
        steps_per_epoch: int | None = None,
        optimizer: OptimizerSpec = DEFAULT_CLASSIFIER_OPTIMIZER,
        label_smoothing: float = 0.1,
        init_weight_scale: float = 0.02,
        soft_init_head: bool = False,
        voting: int | None = None,
        voting_augmentations: AugmentationTransform | None = None,
        classifier_name: str = "neural",
        top_k_metrics: int | list[int] = [1, 3],
    ) -> None:
        super().__init__()
        self.max_epochs = max_epochs if max_epochs and max_epochs > 0 else None
        self.max_steps = max_steps if max_steps and max_steps > 0 else None
        if max_steps is None and max_epochs is None:
            raise ValueError("Either max_epochs or max_steps must be specified.")

        self.steps_per_epoch = steps_per_epoch

        if not (
            map_avg_pooling
            or map_max_pooling
            or map_cls_token
            or map_attn_pooling is not False
        ):
            raise ValueError(
                "At least one of map_avg_pooling, map_max_pooling, or map_cls_token must be True"
            )

        if map_cls_token and point_encoder.cls_token is None:
            map_cls_token = False
            logger.warning(
                "map_cls_token is True, but encoder does not have a cls token. Setting map_cls_token to False."
            )

        if (
            classification_head_type == ClassificationHeadType.MLP
            and mlp_head_config is None
        ):
            raise ValueError(
                "mlp_head_config must be specified when classification_mode is MLP."
            )

        self.map_avg_pooling = map_avg_pooling
        self.map_max_pooling = map_max_pooling
        self.map_cls_token = map_cls_token
        self.map_attn_pooling = map_attn_pooling

        self.classification_head_type = classification_head_type
        self.mlp_head_config = mlp_head_config
        self.num_classes = num_classes
        self.voting = voting

        self.encoder_ckpt_path = encoder_ckpt_path
        self.classifier_name = classifier_name

        self.batch_size = batch_size
        self.optimizer_spec = copy.deepcopy(optimizer)
        self.init_weight_scale = init_weight_scale
        self.soft_init_head = soft_init_head

        self.subsampling_transform = subsampling_transform or IdentityPassThrough()
        self.aug_transform: nn.Module = (
            sequentialize_transform(aug_transform)
            if aug_transform
            else IdentityMultiArg()
        )
        self.norm_transform: nn.Module = norm_transform or IdentityMultiArg()
        self.voting_augmentations: nn.Module = (
            sequentialize_transform(voting_augmentations)
            if voting_augmentations
            else IdentityMultiArg()
        )

        self.point_encoder = point_encoder
        self.encoder_choice = encoder_choice

        # Freeze encoder
        if isinstance(freeze_encoder, bool):
            freeze_encoder = -1 if freeze_encoder else 0
        self.freeze_encoder = freeze_encoder

        # Requires encoder to have embed_dim
        self.embed_dim = self.point_encoder.embed_dim

        self.classification_head: MLPVarLen = None  # type: ignore
        self.cls_token: TrainableToken = None  # type: ignore
        self.attention: nn.Module = None  # type: ignore

        self.top_acc_train = nn.ModuleDict()
        self.top_acc_val = nn.ModuleDict()
        self.voting_acc_val: torchmetrics.Metric = None  # type: ignore
        self.mean_acc_val: torchmetrics.Metric = None  # type: ignore

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.schedules = {
            "lr": self.optimizer_spec.lr,
            "wd": self.optimizer_spec.wd,
        }

        self.loaded_from_checkpoint = False

        # Convert report_topk to list if it's an integer
        self.topk = [top_k_metrics] if isinstance(top_k_metrics, int) else top_k_metrics

    def init_weights(self):
        std = self.init_weight_scale

        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if not self.encoder_ckpt_path:
            self.point_encoder.apply(_init_weights)
            if self.point_encoder.cls_token is not None:
                nn.init.trunc_normal_(self.point_encoder.cls_token, std=std)

        if self.soft_init_head:
            self.classification_head.apply(_init_weights)  # Can consider kaiming_normal

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=std)

    def _init_classification_head(self, num_classes: int):
        input_dim = (
            int(self.map_cls_token)
            + int(self.map_avg_pooling)
            + int(self.map_max_pooling)
            + int(self.map_attn_pooling is not False)
        ) * self.embed_dim

        if self.classification_head_type == ClassificationHeadType.LINEAR:
            self.classification_head = MLPVarLen(input_dim, num_classes, bias=True)
        elif self.classification_head_type == ClassificationHeadType.MLP:
            cfg: MLPConfig = self.mlp_head_config  # type: ignore
            self.classification_head = MLPVarLen(
                *([input_dim] + cfg.dims + [num_classes]),
                norm_layer=cfg.norm_layer,
                act_layer=cfg.act_layer,
                dropout_p=cfg.dropout_p,
                bias=cfg.bias,
            )

        if self.map_attn_pooling is not False:
            num_heads = (
                self.map_attn_pooling if isinstance(self.map_attn_pooling, int) else 1
            )
            self.cls_token = TrainableToken(embed_dim=self.embed_dim)

            self.attention = nn.MultiheadAttention(
                self.embed_dim,
                num_heads=num_heads,
                # dropout=cfg.dropout_p,
                bias=cfg.bias,
                batch_first=True,
            )

    def _init_metrics(self, num_classes: int):
        accuracy_kwargs = {
            "task": "multiclass",
            "num_classes": num_classes,
            "average": "micro",
        }

        # Initialize topk accuracy metrics for training and validation based on report_topk
        for k in self.topk:
            self.top_acc_train[str(k)] = torchmetrics.Accuracy(
                top_k=k, **accuracy_kwargs
            )
            self.top_acc_val[str(k)] = torchmetrics.Accuracy(top_k=k, **accuracy_kwargs)

        # Voting and mean accuracy always use top_k=1
        if self.voting:
            self.voting_acc_val = torchmetrics.Accuracy(top_k=1, **accuracy_kwargs)
        accuracy_kwargs["average"] = "macro"
        self.mean_acc_val = torchmetrics.Accuracy(top_k=1, **accuracy_kwargs)

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                experiment = logger.experiment
                for k in self.topk:
                    experiment.define_metric(
                        f"{self.benchmark}/val/{self.classifier_name}/top{k}_acc",
                        summary="last,max",
                    )
                if self.voting:
                    experiment.define_metric(
                        f"{self.benchmark}/val/{self.classifier_name}/voting_acc",
                        summary="last,max",
                    )

    def setup(
        self,
        stage: str | None = None,
        datamodule: SupervisedPCDataModule | None = None,
    ):
        if (
            self.steps_per_epoch is None or self.num_classes is None
        ) and datamodule is None:
            try:
                datamodule = self.trainer.datamodule  # type: ignore
            except AttributeError:
                raise ValueError(
                    "steps_per_epoch and num_classes must be specified if not using 'PointCloudData'"
                )

        self.benchmark = datamodule.name if datamodule.name != "" else "benchmark"  # type: ignore

        if self.num_classes is None:
            self.num_classes = datamodule.num_classes[PCFieldKey.CLOUD_LABEL]  # type: ignore

        self._init_classification_head(self.num_classes)

        if stage == "fit":
            if self.steps_per_epoch is None:
                self.steps_per_epoch = datamodule.len_train_dataset // self.batch_size  # type: ignore

            real_schedule: list[Schedule] = [
                s for s in self.schedules.values() if isinstance(s, Schedule)
            ]

            for schedule in real_schedule:
                max_epochs = self.max_epochs or (self.max_steps / self.steps_per_epoch)  # type: ignore
                schedule.set_default_max_epochs(max_epochs)  # type: ignore
                schedule.set_steps_per_epoch(self.steps_per_epoch)

        if stage == "fit" or stage == "test":
            self._init_metrics(self.num_classes)

        if self.encoder_ckpt_path:
            load_module_from_checkpoint(
                self.encoder_ckpt_path,
                module=self.point_encoder,
                device=self.device,
                key_prefix=[
                    "point_encoder",
                    "encoder",
                    "_encoder",
                    f"{str(self.encoder_choice)}.point_encoder",
                ],
                replace_key_part={
                    "attn_module": "self_attn",
                    "ffn_module": "ffn",
                },
            )

    def forward(self, x: PatchPoints) -> torch.Tensor:
        embedding: PointEncoderOutput = self.point_encoder(x)

        if not self.map_cls_token and embedding.cls_features is not None:
            all_embeddings = torch.cat(
                (embedding.cls_features.unsqueeze(1), embedding.patch_features), dim=1
            )
        else:
            all_embeddings = embedding.patch_features

        features = []
        if self.map_cls_token:
            features.append(embedding.cls_features)
        if self.map_avg_pooling:
            features.append(all_embeddings.mean(dim=1))
        if self.map_max_pooling:
            features.append(all_embeddings.amax(dim=1))
        if self.map_attn_pooling:
            patch_features = embedding.patch_features
            cls_token = self.cls_token.expand(patch_features.shape[0], 1, -1)
            x, _ = self.attention(cls_token, patch_features, patch_features)
            features.append(x.squeeze(1))  # type: ignore

        x = torch.cat(features, dim=-1)  # type: ignore
        x = self.classification_head(x)

        return x  # type: ignore

    def on_train_epoch_start(self) -> None:
        if (
            self.freeze_encoder == -1
            or self.trainer.current_epoch < self.freeze_encoder
        ):
            self.point_encoder.freeze()
        else:
            self.point_encoder.unfreeze()

    def forward_full(
        self, batch: dict[str, Any], augment_data: bool = False
    ) -> torch.Tensor:
        patch_points = PatchPoints(
            points=batch[PCFieldKey.POINTS],
            num_points=batch.get("num_points"),
            patches_idx=batch.get("patches_idx"),
            centers_idx=batch.get("centers_idx"),
        )

        points = patch_points.points
        num_points = patch_points.num_points

        points, num_points = self.subsampling_transform(points, num_points)

        mask = (
            lengths_to_mask(num_points, points.size(1))
            if num_points is not None
            else None
        )

        if augment_data:
            points = self.aug_transform(points)
        points = self.norm_transform(points, mask=mask)

        patch_points.points = points
        patch_points.num_points = num_points

        pred_logits = self(patch_points)
        return pred_logits

    def training_step(self, batch: dict[str, Any], batch_idx: int | None = None):
        pred_logits = self.forward_full(batch, augment_data=True)
        target_indices = batch[PCFieldKey.CLOUD_LABEL]

        loss = self.ce_loss(pred_logits, target_indices)

        # Update all topk accuracy metrics
        for metric in self.top_acc_train.values():
            metric(pred_logits, target_indices)

        return {"loss": loss}

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int | None = None,
        dataloader_idx: int = 0,
    ):
        patch_points = PatchPoints(
            points=batch[PCFieldKey.POINTS],
            num_points=batch.get("num_points"),
            patches_idx=batch.get("patches_idx"),
            centers_idx=batch.get("centers_idx"),
        )
        target_indices = batch[PCFieldKey.CLOUD_LABEL]

        points = patch_points.points
        num_points = patch_points.num_points

        points_sample, num_points_sample = self.subsampling_transform(
            points, num_points
        )

        mask = (
            lengths_to_mask(num_points_sample, points_sample.size(1))
            if num_points_sample is not None
            else None
        )
        points_sample: torch.Tensor = self.norm_transform(points_sample, mask=mask)
        # No augmentation for validation

        patch_points.points = points_sample
        patch_points.num_points = num_points_sample
        pred_logits = self(patch_points)

        loss = self.ce_loss(pred_logits, target_indices)

        # Update all topk accuracy metrics
        for metric in self.top_acc_val.values():
            metric(pred_logits, target_indices)

        self.mean_acc_val(pred_logits, target_indices)

        pred_indices = torch.argmax(pred_logits, dim=-1)

        if self.voting:
            for _ in range(self.voting - 1):
                points_sample, num_points_sample = self.subsampling_transform(
                    points, num_points
                )
                mask = (
                    lengths_to_mask(num_points_sample, points_sample.size(1))
                    if num_points_sample is not None
                    else None
                )
                points_sample = self.voting_augmentations(points_sample)
                points_sample = self.norm_transform(points_sample, mask=mask)

                patch_points.points = points_sample
                patch_points.num_points = num_points_sample

                pred_logits += self(patch_points)

            self.voting_acc_val(torch.argmax(pred_logits, dim=-1), target_indices)

        return {
            "loss": loss,
            "pred_indices": pred_indices,
            "target_indices": target_indices,
        }

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int | None = None,
        dataloader_idx: int = 0,
    ):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch: dict[str, Any]) -> Any:
        pred_logits = self.forward_full(batch, augment_data=True)
        return {"pred_indices": pred_logits.argmax(dim=1), "pred_logits": pred_logits}

    def on_fit_start(self) -> None:
        # This is after checkpoint loading
        if not self.loaded_from_checkpoint:
            self.init_weights()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self.log(
            f"{self.benchmark}/train/{self.classifier_name}/loss",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Log all topk metrics
        log_dict = {}
        for k, metric in self.top_acc_train.items():
            log_dict[f"{self.benchmark}/train/{self.classifier_name}/top{k}_acc"] = (
                metric
            )

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        log_dict = {
            f"{self.benchmark}/val/{self.classifier_name}/loss": outputs["loss"],
            f"{self.benchmark}/val/{self.classifier_name}/mean_acc": self.mean_acc_val,
        }

        # Add all topk metrics to log_dict
        for k, metric in self.top_acc_val.items():
            log_dict[f"{self.benchmark}/val/{self.classifier_name}/top{k}_acc"] = metric

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.voting:
            self.log(
                f"{self.benchmark}/val/{self.classifier_name}/voting_acc",
                self.voting_acc_val,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # When the encoder is always frozen, don't save encoder.
        if self.freeze_encoder == -1:
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith("point_encoder"):
                    del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.loaded_from_checkpoint = True

        if self.freeze_encoder == -1:
            # Always frozen, so must load from SSRL checkpoint
            # The checkpoint connector does not support unstrict loading,
            # therefore add keys from current state point_encoder.
            ckpt_state_dict: dict[str, Any] = checkpoint["state_dict"]
            point_encoder_state_dict = self.state_dict()

            for k in list(point_encoder_state_dict.keys()):
                if k.startswith("point_encoder"):
                    ckpt_state_dict[k] = point_encoder_state_dict[k]

        if self.voting_augmentations is not None:
            ckpt_state_dict: dict[str, Any] = checkpoint["state_dict"]
            va_state_dict = self.voting_augmentations.state_dict()
            for k in list(va_state_dict.keys()):
                ckpt_state_dict[f"voting_augmentations.{k}"] = va_state_dict[k]

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Any | None
    ) -> None:
        # Needs to overwrite to support scheduler that is not LRScheduler
        if metric is None:
            scheduler.step()  # type: ignore[call-arg]
        else:
            scheduler.step(metric)  # Also works for wd_schedule

    def configure_optimizers(self):
        # TODO: Make util function for this.
        # lr_multiplier = self.batch_size / NeuralClassifier.DEFAULT_BATCH_SIZE
        lr_multiplier = 1.0

        if self.freeze_encoder == -1:
            # Only give classification head parameters as the encoder will remain frozen.
            parameters = self.classification_head.parameters()
        else:
            parameters = self.parameters()

        optimizer = self.optimizer_spec.get_optim(parameters, lr_multiplier)
        lr_scheduler = self.optimizer_spec.get_lr_scheduler(optimizer)
        weight_decay_scheduler = self.optimizer_spec.get_wd_scheduler(optimizer)

        optimizers = [optimizer]
        schedules = []

        if lr_scheduler is not None:
            schedules.append(
                {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "name": "lr_schedule",
                }
            )

        if weight_decay_scheduler is not None:
            schedules.append(
                {
                    "scheduler": weight_decay_scheduler,
                    "interval": "step",
                    "name": "wd_schedule",
                }
            )

        return optimizers, schedules
