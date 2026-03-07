import zipfile
from collections.abc import Iterable, Iterator
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

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


class Toys4KBuilder(DatasetBuilder):
    DATA_FILE = "toys4k_point_clouds.zip"
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
            data_path = Path(__file__).parent / "data" / Toys4KBuilder.DATA_FILE

        self._set_info(
            name="Toys4K",
            data_path=data_path,
            splits=Toys4KBuilder.SPLITS,
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
                PCFieldKey.CLOUD_LABEL: ClassLabels(Toys4KBuilder.LABEL_NAMES)
            },
        )

        self.num_pre_sample_points = num_pre_sample_points
        self.decode_mesh = DecodeMesh(format=Toys4KBuilder.FILE_FORMAT)
        self.sample_surface = SampleSurfacePoints(
            num_points=num_pre_sample_points,
            seed=seed,
        )

    def process_instance(self, path: str) -> dict[str, str | np.ndarray]:
        zip_file = open_zipfile_buffer(self.data_path)
        mesh_binary = zip_file.read(path)
        mesh = self.decode_mesh(mesh_binary)
        
        points = self.sample_surface(mesh)
        points = points[:, [2, 0, 1]]

        path_parts = path.split("/")

        name = path_parts[-2]
        label = path_parts[-3]

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
        builder: Toys4KBuilder,
        split: str,
        num_workers: int | None = 1,
    ):
        self.builder = builder
        self.num_workers = num_workers or 0

        self.paths = [
            path
            for path in zipfile.ZipFile(builder.data_path, "r").namelist()
            if path.endswith(builder.FILE_FORMAT)
        ]
        self.results_len = len(self.paths)

    def __iter__(self) -> Iterator[dict[str, str | np.ndarray]]:
        if self.num_workers > 0:
            with Pool(self.num_workers) as pool:
                self.results = pool.imap_unordered(
                    self.builder.process_instance, self.paths, chunksize=4
                )
                for result in self.results:
                    yield result
        else:
            for path in self.paths:
                yield self.builder.process_instance(path)

    def __len__(self):
        return self.results_len
