import copy
from typing import Any, Iterable

import lightning as L
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from asymdsd import AsymDSD
from asymdsd.components import EncoderBranch, OptimizerSpec
from asymdsd.data import SupervisedPCDataModule
from asymdsd.defaults import DEFAULT_CLASSIFIER_OPTIMIZER
from asymdsd.layers import TransformerEncoder
from asymdsd.models import PointEncoder
from asymdsd.models.neural_classifier import (
    DEFAULT_CLS_HEAD_CONFIG,
    ClassificationHeadType,
    MLPConfig,
    NeuralClassifier,
)


class NeuralClassifierEval(L.Callback):
    def __init__(
        self,
        datamodule: SupervisedPCDataModule,
        classifier_name: str = "neural",
        eval_run_interval: int | list[int] = 1,
        encoder_choice: EncoderBranch | str = EncoderBranch.TEACHER,
        max_epochs: int = 100,
        limit_train_batches: int | None = None,
        eval_last_num_epochs: int = 1,
        precision: str = "16-mixed",
        callbacks: list[L.Callback] | None = None,
        pre_empty_cache: bool = False,
        freeze_encoder: bool | int = True,
        map_avg_pooling: bool = True,
        map_max_pooling: bool = False,
        map_cls_token: bool = False,
        map_attn_pooling: bool | int = False,
        classification_head_type: ClassificationHeadType = ClassificationHeadType.LINEAR,
        mlp_head_config: MLPConfig | None = DEFAULT_CLS_HEAD_CONFIG,
        num_classes: int | None = None,
        optimizer: OptimizerSpec = DEFAULT_CLASSIFIER_OPTIMIZER,
        drop_path_p: float = 0.0,
        label_smoothing: float = 0.1,
        init_weight_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.classifier_name = classifier_name

        if isinstance(encoder_choice, str):
            encoder_choice = EncoderBranch(encoder_choice)
        self.encoder_choice = encoder_choice

        self.eval_run_interval = eval_run_interval

        self.max_epochs = max_epochs
        self.eval_last_num_epochs = eval_last_num_epochs
        self.batch_size = datamodule.batch_size
        self.limit_train_batches = limit_train_batches
        self.precision = precision
        self.callbacks = callbacks
        self.empty_cache = pre_empty_cache

        if isinstance(freeze_encoder, bool):
            freeze_encoder = -1 if freeze_encoder else 0
        self.freeze_encoder = freeze_encoder

        self.drop_path_p = drop_path_p
        # classifier_kwargs.pop("point_encoder", None)
        # classifier_kwargs.pop("encoder_ckpt_path", None)
        # self.classifier_kwargs = classifier_kwargs

        self.fabric = L.Fabric(precision=self.precision)  # type: ignore

        self.trainer: L.Trainer = None  # type: ignore
        self._datamodule = datamodule
        self.classifier: NeuralClassifier = None  # type: ignore
        self.optimizer: torch.optim.Optimizer = None  # type: ignore

        # Jsonargparse has problems with finding kwargs...
        self.map_avg_pooling = map_avg_pooling
        self.map_max_pooling = map_max_pooling
        self.map_cls_token = map_cls_token
        self.map_attn_pooling = map_attn_pooling
        self.classification_head_type = classification_head_type
        self.mlp_head_config = mlp_head_config
        self.num_classes = num_classes
        self.optimizer_spec = optimizer
        self.label_smoothing = label_smoothing
        self.init_weight_scale = init_weight_scale

    def setup(
        self, trainer: L.Trainer, pl_module: AsymDSD, stage: str | None = None
    ) -> None:
        self.asymdsd = pl_module

        self._datamodule.prepare_data()
        self._datamodule.setup(stage=stage)  # type: ignore
        self.benchmark_name = (
            self._datamodule.name if self._datamodule.name != "" else "benchmark"
        )

        train_dataloader = self._datamodule.train_dataloader()
        val_dataloader = self._datamodule.val_dataloader()
        (
            self.train_dataloader,
            self.val_dataloader,
        ) = self.fabric.setup_dataloaders(train_dataloader, val_dataloader)  # type: ignore

        self._setup_callbacks(trainer)

    def _init_classifier(self):
        # No (deep)copy is made
        point_encoder = self._get_encoder(self.asymdsd)
        original_encoder = point_encoder.encoder

        if (
            self.freeze_encoder != -1
            or self.drop_path_p != original_encoder.config.drop_path_p
        ):
            point_encoder = copy.deepcopy(point_encoder)
            # point_encoder.encoder.config.drop_path_p = self.drop_path_p
            encoder_config = point_encoder.encoder.original_config
            encoder_config.drop_path_p = self.drop_path_p

            new_encoder = TransformerEncoder(encoder_config)
            point_encoder.encoder = new_encoder
            # 2.07 MiB at this point?
            state_dict = original_encoder.state_dict()
            new_encoder.load_state_dict(state_dict)

        aug_transform = self.asymdsd.aug_transform
        norm_transform = self.asymdsd.norm_transform

        self.classifier = NeuralClassifier(
            point_encoder=point_encoder,
            freeze_encoder=self.freeze_encoder,
            aug_transform=aug_transform,  # type: ignore
            norm_transform=norm_transform,  # type: ignore
            classifier_name=self.classifier_name,
            max_epochs=self.max_epochs,
            map_avg_pooling=self.map_avg_pooling,
            map_max_pooling=self.map_max_pooling,
            map_cls_token=self.map_cls_token,
            map_attn_pooling=self.map_attn_pooling,
            classification_head_type=self.classification_head_type,
            mlp_head_config=self.mlp_head_config,
            num_classes=self.num_classes,
            batch_size=self.batch_size,
            optimizer=self.optimizer_spec,
            label_smoothing=self.label_smoothing,
            init_weight_scale=self.init_weight_scale,
        )
        self.classifier.setup(stage="fit", datamodule=self._datamodule)
        optimizers, schedules = self.classifier.configure_optimizers()

        self.optimizer = optimizers[0]
        self.lr_schedule = schedules[0]["scheduler"]
        self.wd_schedule = schedules[1]["scheduler"] if len(schedules) > 1 else None

        self.fabric = L.Fabric(precision=self.precision)  # type: ignore

        self.classifier, self.optimizer = self.fabric.setup(
            self.classifier, self.optimizer
        )

    def _get_encoder(self, asymdsd_module: AsymDSD) -> PointEncoder:  # type: ignore
        if self.encoder_choice == EncoderBranch.TEACHER:
            return asymdsd_module.teacher.point_encoder
        elif self.encoder_choice == EncoderBranch.STUDENT:
            return asymdsd_module.student.point_encoder

    def _wrap_progress_bar(
        self, dataloader: DataLoader, desc: str | None = None
    ) -> Iterable:
        return tqdm(dataloader, leave=False, desc=desc)

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: AsymDSD) -> None:
        validation_epoch = pl_module.validation_epoch
        if isinstance(self.eval_run_interval, list):
            if validation_epoch in self.eval_run_interval:
                self._run_evaluation(trainer)
        else:
            if validation_epoch % self.eval_run_interval == self.eval_run_interval - 1:
                self._run_evaluation(trainer)

    def _run_evaluation(self, trainer: L.Trainer) -> None:
        if self.empty_cache:
            torch.cuda.empty_cache()
        self._init_classifier()
        self._fit()
        self._log_metrics()
        self._restore_encoder()
        self._del_references()

        if self.empty_cache:
            torch.cuda.empty_cache()

    def _fit(self):
        for epoch in range(self.max_epochs):
            self.fabric.current_epoch = epoch  # type: ignore
            self.classifier.on_train_epoch_start()
            train_dataloader = self._wrap_progress_bar(
                self.train_dataloader,
                desc=f"Fitting {self.classifier_name} Classifier on {self.benchmark_name} - Epoch {epoch}",
            )
            self.classifier.train()

            with torch.enable_grad():
                for batch_idx, batch in enumerate(train_dataloader):
                    self.optimizer.zero_grad()
                    if (
                        self.limit_train_batches is not None
                        and batch_idx >= self.limit_train_batches
                    ):
                        break
                    with self.fabric.autocast():
                        out = self.classifier.training_step(batch)
                    self.fabric.backward(out["loss"])

                    self.optimizer.step()
                    self.lr_schedule.step()
                    if self.wd_schedule is not None:
                        self.wd_schedule.step()

            if epoch >= self.max_epochs - self.eval_last_num_epochs:
                # Calculate validation metrics only for the last eval_last_num_epochs epochs
                self._evaluate()

        self.classifier.on_validation_epoch_start()

    def _evaluate(self):
        self._on_validation_start_callbacks(self.trainer)

        val_dataloader = self._wrap_progress_bar(
            self.val_dataloader,
            desc=f"Evaluating {self.classifier_name} Classifier on {self.benchmark_name}",
        )
        self.classifier.eval()

        for batch_idx, batch in enumerate(val_dataloader):
            outputs = self.classifier.validation_step(batch)
            self._on_validation_batch_end_callbacks(
                self.trainer, self.classifier, outputs, batch, batch_idx
            )

        self._on_validation_end_callbacks(self.trainer, self.classifier)

    def _restore_encoder(self):
        self.classifier.point_encoder.unfreeze()
        if self.encoder_choice == EncoderBranch.TEACHER:
            self.classifier.point_encoder.requires_grad_(False)

    def _del_references(self):
        del self.classifier
        del self.optimizer
        del self.fabric

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     self._log_metrics()

    def _log_metrics(self) -> None:
        log_dict = {
            f"{self.benchmark_name}/val/{self.classifier_name}/top{k}_acc": metric.compute()
            for k, metric in self.classifier.top_acc_val.items()
        }
        self.asymdsd.log_dict(
            log_dict,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _setup_callbacks(self, trainer: L.Trainer):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.setup(trainer, self.classifier, stage="fit")

    def _on_validation_start_callbacks(self, trainer: L.Trainer):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_validation_start(self, self.classifier)  # type: ignore

    def _on_validation_batch_end_callbacks(
        self,
        trainer: L.Trainer,
        pl_module: NeuralClassifier,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_validation_batch_end(
                    trainer, self.classifier, outputs, batch, batch_idx
                )

    def _on_validation_end_callbacks(
        self, trainer: L.Trainer, pl_module: NeuralClassifier
    ):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_validation_end(trainer, self.classifier)

    @property
    def datamodule(self):
        return self._datamodule
