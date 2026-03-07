import json
import zipfile
from collections.abc import Iterable, Iterator
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from asymdsd.components.common_types import PathLike
from asymdsd.data import (
    ClassLabels,
    DataField,
    DatasetBuilder,
    FieldType,
    PCFieldKey,
)
from asymdsd.data.transforms import DecodeMesh, SampleSurfacePoints

from .label_names import LABEL_NAMES


@lru_cache(1)
def open_zipfile_buffer(data_path):
    return zipfile.ZipFile(data_path, "r")


class Future3DBuilder(DatasetBuilder):
    DATA_FILES = [f"3D-FUTURE-model-part{i}.zip" for i in range(1, 5)]
    MODEL_JSON = "3D-FUTURE-model-part1/model_info.json"
    SPLITS = ["train"]
    FILE_FORMAT = "obj"

    LABEL_NAMES = LABEL_NAMES

    def __init__(
        self,
        data_path: PathLike | None = None,
        num_pre_sample_points: int = 16384,
        seed: int | None = None,
    ):
        if data_path is None:
            data_path = Path(__file__).parent / "data"

        self._set_info(
            name="3D-FUTURE",
            data_path=data_path,
            splits=Future3DBuilder.SPLITS,
            data_fields=[
                DataField(
                    key=PCFieldKey.POINTS,
                    key_type=FieldType.ARRAY,
                ),
                DataField(
                    key=PCFieldKey.CLOUD_LABEL,
                    key_type=FieldType.STRING_LABEL,
                ),
            ],
            class_labels={
                PCFieldKey.CLOUD_LABEL: ClassLabels(Future3DBuilder.LABEL_NAMES)
            },
        )

        self.num_pre_sample_points = num_pre_sample_points
        self.decode_mesh = DecodeMesh(format=Future3DBuilder.FILE_FORMAT)
        self.sample_surface = SampleSurfacePoints(
            num_points=num_pre_sample_points,
            seed=seed,
        )

    def process_instance(self, args: tuple[str, dict[str, Any]]
                         ) -> dict[str, str | np.ndarray]:
        path, meta_data = args
        path_parts = path.split("/")
        zip_file = open_zipfile_buffer(self.data_path / f"{path_parts[0]}.zip")

        mesh_binary = zip_file.read(path)
        mesh = self.decode_mesh(mesh_binary)
        points = self.sample_surface(mesh)
        points = points[:, [2, 0, 1]]

        name = path_parts[-2]
        label = meta_data["category"]
        label = "Other" if label is None else label

        return {
            "name": name,
            PCFieldKey.POINTS: points,
            PCFieldKey.CLOUD_LABEL: label,
        }

    def iterate_data(
        self, split: str, num_workers: int | None = 1
    ) -> Iterable[dict[str, str | np.ndarray]]:
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}")

        return _DataIterator(
            self,
            split=split,
            num_workers=num_workers,
        )


class _DataIterator:
    def __init__(
        self,
        builder: Future3DBuilder,
        split: str,
        num_workers: int | None = 1,
    ):
        self.builder = builder
        self.num_workers = num_workers or 0

        paths = []
        for data_file in builder.DATA_FILES:
            datapath = builder.data_path / data_file
            with zipfile.ZipFile(datapath, "r") as zip_file:
                paths.extend([
                    path
                    for path in zip_file.namelist()
                    if path.endswith('normalized_model.obj') and not path.startswith('__MACOSX')
                ])

        with zipfile.ZipFile(builder.data_path / builder.DATA_FILES[0], "r") as zip_file:
            with zip_file.open(builder.MODEL_JSON) as json_file:
                meta_data = json.load(json_file)

        meta_data = {entry["model_id"]: entry for entry in meta_data if entry.get("model_id")}
        self.data_tuple = [(path, meta_data[path.split("/")[-2]]) for path in paths]

        self.results_len = len(self.data_tuple)

    def __iter__(self) -> Iterator[dict[str, str | np.ndarray]]:
        if self.num_workers > 0:
            with Pool(self.num_workers) as pool:
                self.results = pool.imap_unordered(
                    self.builder.process_instance, self.data_tuple, chunksize=4
                )
                for result in self.results:
                    yield result
        else:
            for data_tuple in self.data_tuple:
                yield self.builder.process_instance(data_tuple)

    def __len__(self):
        return self.results_len
