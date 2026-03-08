from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

import numpy as np
import zarr
from torch.utils.data import ConcatDataset, DataLoader

from ..components.common_types import PathLike
from ..loggers import get_default_logger
from .data_module import (
    DatasetSplit,
    PointCloudDataModule,
    SupervisedPCDataModule,
)
from .dataset_builder import ClassLabels, DatasetBuilder, PCFieldKey
from .dataset_utils import compose_transform
from .dataset_zarr import ZarrDataset, create_zarr_ds
from .multi_crop import MultiCropConfig, PointMultiCrop
from .patchify import PatchifyModule
from .pc_transforms import AugmentationTransform
from .transforms import (
    CropSampleArrays,
    FarthestPointSampleArrays,
    PadArrays,
    UniformSampleArrays,
)

logger = get_default_logger()

SplitMap = dict[DatasetSplit, str | list[str]]


class SubsampleMode(StrEnum):
    UNIFORM = auto()
    FPS = auto()


def _prepare_zarr_ds(
    path: PathLike,
    dataset_builder: DatasetBuilder | None = None,
    num_workers: int | None = None,
) -> None:
    try:
        group = zarr.open_group(str(path), mode="r")
        # Check if the dataset is complete
        if group.attrs["complete"]:
            logger.info(f"Found complete dataset at {path}: {group.attrs['name']}.")

    except Exception:
        logger.info(
            f"No complete dataset is found at {path}.\n"
            "Calling create_zarr_ds to prepare the dataset..."
        )

        if dataset_builder is None:
            raise ValueError(
                f"dataset_builder must be provided for dataset to be created and saved at {path}"
            )

        create_zarr_ds(
            dataset_save_path=path,
            dataset_builder=dataset_builder,
            num_workers=num_workers,
        )


@dataclass
class DatasetConfig:
    dataset_path: PathLike
    split_map: SplitMap | None = None
    dataset_builder: DatasetBuilder | None = None
    num_workers_create_ds: int | None = None


@dataclass
class MultiPatchify:
    global_patchify: PatchifyModule
    local_patchify: PatchifyModule | None = None


SampleArraysTransform = (
    CropSampleArrays | FarthestPointSampleArrays | UniformSampleArrays
)


class UnsupervisedZarrPCDataModule(PointCloudDataModule):
    def __init__(
        self,
        dataset: PathLike | DatasetConfig | list[PathLike | DatasetConfig],
        name: str | None = None,
        batch_size: int = 32,
        max_num_points: int | None = 1024,
        subsample_transform: SampleArraysTransform | None = None,
        augmentation_transform: AugmentationTransform | None = None,
        patchify: PatchifyModule | MultiPatchify | None = None,
        multi_crop_config: MultiCropConfig | None = None,
        num_workers_train: int = 0,
        num_workers_val_test: int = 0,
        include_labels: bool = False,
        pin_memory: bool = False,
        seed: int | None = None,
    ):
        super().__init__(
            name=name,
            batch_size=batch_size,
            num_workers_train=num_workers_train,
            num_workers_val_test=num_workers_val_test,
            pin_memory=pin_memory,
            seed=seed,
        )
        if not isinstance(dataset, list):
            dataset = [dataset]

        self.ds_configs: list[DatasetConfig] = []

        for config in dataset:
            if isinstance(config, PathLike):
                config = DatasetConfig(config)
            config.dataset_path = Path(config.dataset_path).expanduser().resolve()
            self.ds_configs.append(config)

        self.max_num_points = max_num_points
        self.augmentation_transform = compose_transform(
            augmentation_transform, seed=self.seed
        )

        self.multi_crop_cfg = multi_crop_config

        if self.max_num_points is None and self.multi_crop_cfg is None:
            raise ValueError(
                "max_num_points must be provided if multi-crop is disabled."
            )

        self.perform_patchify = patchify is not None
        self.perform_multi_crop = self.multi_crop_cfg is not None
        self.has_local_crops = (
            self.perform_multi_crop and self.multi_crop_cfg.local_cfg is not None  # type: ignore
        )

        if self.perform_patchify:
            if isinstance(patchify, PatchifyModule):
                self.patchify = patchify
            else:
                self.patchify = MultiPatchify(
                    global_patchify=patchify.global_patchify,  # type: ignore
                    local_patchify=patchify.local_patchify,  # type: ignore
                )
            if self.has_local_crops and self.patchify.local_patchify is None:  # type: ignore
                raise ValueError(
                    "Local patchify is required for multi crop with local crops."
                )
            self.patchify.global_patchify.set_seed(self.seed)  # type: ignore
            if self.has_local_crops:
                self.patchify.local_patchify.set_seed(self.seed)  # type: ignore

        if self.multi_crop_cfg is not None:
            self.multi_crop = PointMultiCrop(self.multi_crop_cfg, seed=self.seed)

        max_num_points_global: int = (  # type: ignore
            self.max_num_points or self.multi_crop_cfg.global_cfg.num_points_range[1]  # type: ignore
        )
        max_num_points_local: int = (  # type: ignore
            self.multi_crop_cfg.local_cfg.num_points_range[1]  # type: ignore
            if self.has_local_crops
            else None
        )

        if subsample_transform is None:
            self.subsample = UniformSampleArrays(
                sample_size=max_num_points_global, axis=0
            )
        else:
            self.subsample = subsample_transform

        self.subsample.set_seed(self.seed)

        self.pad_global = PadArrays(
            pad_to_length=max_num_points_global,
            axis=0,
            output_arr_len_key="num_points",
        )

        self.pad_local = PadArrays(
            pad_to_length=max_num_points_local,
            axis=0,
            output_arr_len_key="num_points",
        )

        self.include_labels = include_labels

    def prepare_data(self) -> None:
        for cfg in self.ds_configs:
            _prepare_zarr_ds(
                cfg.dataset_path,
                dataset_builder=cfg.dataset_builder,
                num_workers=cfg.num_workers_create_ds,
            )

    def _collate_crops(
        self, x: dict[str, list[dict[str, np.ndarray]]]
    ) -> dict[str, dict[str, np.ndarray | list[np.ndarray]]]:
        collated = {
            crop_type_key: {
                feature_key: [
                    np.stack(arr, axis=0)
                    for arr in [
                        [crop[feature_key][i] for crop in crops_batch_list]
                        for i in range(len(crops_batch_list[0][feature_key]))
                    ]
                ]
                if isinstance(crops_batch_list[0][feature_key], list)
                else np.stack([crop[feature_key] for crop in crops_batch_list], axis=0)
                for feature_key in crops_batch_list[0].keys()
            }
            for crop_type_key, crops_batch_list in x.items()
        }
        return collated

    def setup(self, stage: str | None = None) -> None:
        datasets = []

        for cfg in self.ds_configs:
            data_root = zarr.open_group(str(cfg.dataset_path), mode="r")

            split_map = cfg.split_map

            if split_map is None:
                list_splits = data_root.attrs["splits"]
                split_map = {split: split for split in list_splits}
                if DatasetSplit.TRAIN not in split_map:
                    raise ValueError(
                        "split_map must be provided if TRAIN split is not found in dataset."
                    )

            if DatasetSplit.TRAIN in split_map:
                split = split_map[DatasetSplit.TRAIN]

                dataset = ZarrDataset(
                    cfg.dataset_path,
                    split,
                    array_keys=[PCFieldKey.POINTS],  # type: ignore
                    attr_keys=[PCFieldKey.CLOUD_LABEL] if self.include_labels else None,
                )
                # TODO: Consider adding support for FEATURES
                # TODO: Consider pre-normalization

                if self.multi_crop_cfg is not None:
                    dataset.map(
                        self._map_multi_crop_process,
                        input_columns=[PCFieldKey.POINTS],
                        remove_columns=[PCFieldKey.POINTS],
                        input_as_positional_args=True,
                    )
                else:
                    dataset.map(self.subsample)

                    dataset.map(
                        self.augmentation_transform,
                        input_columns=[PCFieldKey.POINTS],
                        output_columns=[PCFieldKey.POINTS],
                    )

                    if self.perform_patchify:
                        dataset.map(
                            self.patchify.global_patchify,  # type: ignore
                            input_columns=[PCFieldKey.POINTS],
                        )

                    dataset.map(
                        self.pad_global,
                        input_columns=[PCFieldKey.POINTS],
                        input_as_positional_args=False,
                    )

                datasets.append(dataset)

        self.dataset[DatasetSplit.TRAIN] = ConcatDataset(datasets)

    def val_dataloader(self) -> DataLoader | Iterable | None:
        yield None

    @property
    def len_train_dataset(self) -> int:
        return len(self.dataset[DatasetSplit.TRAIN])  # type: ignore

    def _map_multi_crop_process(self, points: np.ndarray) -> dict:
        x = self.multi_crop(points)
        global_crops = x["global_crops"]
        local_crops = x.get("local_crops")

        for crop in global_crops:
            crop["points"] = self.augmentation_transform(crop["points"])
            if self.perform_patchify:
                crop.update(self.patchify.global_patchify(crop["points"]))  # type: ignore
            crop.update(self.pad_global({"points": crop["points"]}))

        if self.has_local_crops:
            for crop in local_crops:  # type: ignore
                crop["points"] = self.augmentation_transform(crop["points"])
                if self.perform_patchify:
                    crop.update(self.patchify.local_patchify(crop["points"]))  # type: ignore
                crop.update(self.pad_local({"points": crop["points"]}))

        return self._collate_crops(x)


class SupervisedZarrPCDataModule(SupervisedPCDataModule):
    def __init__(
        self,
        dataset: PathLike | DatasetConfig,
        name: str | None = None,
        batch_size: int = 32,
        max_num_points: int = 1024,
        subsample_mode: SubsampleMode = SubsampleMode.UNIFORM,
        augmentation_transform: AugmentationTransform | None = None,
        patchify: PatchifyModule | None = None,
        supervision_key: str | PCFieldKey | list[PCFieldKey] | None = None,
        load_features: bool = False,
        deterministic_val_data: bool = True,
        num_workers_train: int = 0,
        num_workers_val_test: int = 0,
        dataset_builder: DatasetBuilder | None = None,
        split_map: SplitMap | None = None,
        num_workers_create_ds: int | None = None,
        pin_memory: bool = False,
        seed: int | None = None,
    ):
        super().__init__(
            name=name,
            batch_size=batch_size,
            num_workers_train=num_workers_train,
            num_workers_val_test=num_workers_val_test,
            pin_memory=pin_memory,
            seed=seed,
        )
        if isinstance(dataset, PathLike):
            self.ds_config = DatasetConfig(dataset)
        self.ds_config.dataset_path = (
            Path(self.ds_config.dataset_path).expanduser().resolve()
        )
        self.dataset_path = self.ds_config.dataset_path
        self.split_map = split_map or self.ds_config.split_map
        self.ds_config.dataset_builder = (
            dataset_builder or self.ds_config.dataset_builder
        )
        self.ds_config.num_workers_create_ds = (
            num_workers_create_ds or self.ds_config.num_workers_create_ds
        )

        self.max_num_points = max_num_points
        self.supervision_key = supervision_key
        self.load_features = load_features

        self.patchify: PatchifyModule = patchify  # type: ignore
        if self.patchify is not None:
            self.patchify.set_seed(self.seed)

            self.patchify_val = deepcopy(self.patchify)
            self.patchify_val.deterministic = deterministic_val_data
        else:
            self.patchify_val = None

        self.augmentation_transform = compose_transform(
            augmentation_transform, seed=self.seed
        )

        if subsample_mode == SubsampleMode.UNIFORM:
            self.subsample = UniformSampleArrays(
                sample_size=self.max_num_points,
                axis=0,
                seed=self.seed,
            )
        else:
            self.subsample = FarthestPointSampleArrays(
                sample_size=self.max_num_points,
                axis=-2,
                seed=self.seed,
            )

        self.subsample_val = deepcopy(self.subsample)
        self.subsample_val.deterministic = deterministic_val_data

        self.pad = PadArrays(
            pad_to_length=self.max_num_points,
            axis=0,
            output_arr_len_key="num_points",
        )

    def prepare_data(self) -> None:
        _prepare_zarr_ds(
            self.dataset_path,
            dataset_builder=self.ds_config.dataset_builder,
            num_workers=self.ds_config.num_workers_create_ds,
        )

    def _init_class_labels(self) -> None:
        label_names: dict[str, list[str]] = self.data_root.attrs["label_names"]

        # Which key to use for supervision
        self.supervision_key = (
            self.supervision_key
            if self.supervision_key is not None
            else PCFieldKey(list(label_names.keys())[0])  # Simply get the first key
        )

        self.supervision_key = (
            [self.supervision_key]
            if isinstance(self.supervision_key, PCFieldKey)
            or isinstance(self.supervision_key, str)
            else self.supervision_key
        )

        self.replace_cloud_label = None

        self.class_labels = {}
        for key in self.supervision_key:
            if key in label_names:
                if (
                    isinstance(key, str)
                    and key != PCFieldKey.CLOUD_LABEL
                    and key != PCFieldKey.SEMANTIC_LABELS
                    and key != PCFieldKey.INSTANCE_LABELS
                ):
                    replace_key = PCFieldKey.CLOUD_LABEL
                    self.replace_cloud_label = key
                else:
                    replace_key = key
                self.class_labels[replace_key] = ClassLabels(label_names[key])


        self._num_classes = {}
        self._label_names = {}
        self._label_int2str = {}

        for key, class_labels in self.class_labels.items():
            class_labels: ClassLabels
            self._num_classes[key] = class_labels.num_classes
            self._label_names[key] = class_labels.label_names
            self._label_int2str[key] = class_labels.int2str

    def _init_keys(self) -> None:
        ds_attr_keys = self.data_root.attrs["attr_keys"]
        ds_array_keys = self.data_root.attrs["array_keys"]

        # Always select points
        self.array_keys: list[str] = [PCFieldKey.POINTS]
        self.attr_keys: list[str] = []

        if PCFieldKey.FEATURES in ds_array_keys and self.load_features:
            self.array_keys.append(PCFieldKey.FEATURES)

        for key in self.supervision_key: # type: ignore
            if key in ds_array_keys:
                self.array_keys.append(key)
            elif key in ds_attr_keys:
                self.attr_keys.append(key)

        if not all(key in ds_array_keys for key in self.array_keys):
            raise ValueError(f"Array keys {self.array_keys} not found in dataset.")
        if not all(key in ds_attr_keys for key in self.attr_keys):
            raise ValueError(f"Attribute keys {self.attr_keys} not found in dataset.")

    def setup(self, stage: str | None = None) -> None:
        self.data_root = zarr.open_group(str(self.dataset_path), mode="r")
        if self._name is None:
            self._name = self.data_root.attrs["name"]

        self._init_class_labels()
        self._init_keys()

        if self.split_map is None:
            list_splits = self.data_root.attrs["splits"]
            self.split_map = {split: split for split in list_splits}
            if DatasetSplit.TRAIN not in self.split_map:
                logger.warning(
                    f"There is no TRAIN split in the provided dataset at {self.dataset_path}. "
                    f"Provide a split_map to specify the splits."
                )

        for dataset_split, split in self.split_map.items():
            dataset = ZarrDataset(
                self.dataset_path,
                split,
                attr_keys=self.attr_keys,
                array_keys=self.array_keys,
            )

            self._apply_map_dataset(dataset, dataset_split)

            self.dataset[dataset_split] = dataset

    def _apply_map_dataset(
        self, dataset: ZarrDataset, dataset_split: DatasetSplit
    ) -> None:
        # Subsample point cloud (all arrays in dataset) if above num_points
        # Also pads if below num_points for collation
        dataset.map(
            self.subsample
            if dataset_split == DatasetSplit.TRAIN
            else self.subsample_val,
            input_columns=self.array_keys,
            input_as_positional_args=False,
        )

        if dataset_split == DatasetSplit.TRAIN:
            dataset.map(
                self.augmentation_transform,
                input_columns=[PCFieldKey.POINTS],
                output_columns=[PCFieldKey.POINTS],
            )

        if self.patchify is not None:
            dataset.map(
                self.patchify  # type: ignore
                if dataset_split == DatasetSplit.TRAIN
                else self.patchify_val,
                input_columns=[PCFieldKey.POINTS],
            )

        dataset.map(
            self.pad,
            input_columns=self.array_keys,
            input_as_positional_args=False,
        )

        if self.replace_cloud_label is not None:
            dataset.map(
                lambda x: x,
                input_columns=[self.replace_cloud_label],
                output_columns=[PCFieldKey.CLOUD_LABEL],
                remove_columns=[self.replace_cloud_label],
            )

    @property
    def len_train_dataset(self) -> int:
        return len(self.dataset["train"])  # type: ignore

    @property
    def num_classes(self) -> dict[str | PCFieldKey, int]:
        return self._num_classes

    @property
    def label_names(self) -> dict[str | PCFieldKey, list[str]]:
        return self._label_names

    @property
    def label_int2str(self) -> dict[str | PCFieldKey, Callable[[int], str]]:
        return self._label_int2str