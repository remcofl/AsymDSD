import os
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from functools import lru_cache
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Union

import numpy as np

from asymdsd.components.common_types import PathLike
from asymdsd.data import (
    ClassLabels,
    DataField,
    DatasetBuilder,
    FieldType,
    PCFieldKey,
)

from .label_names import LABEL_NAMES


@lru_cache(1)
def open_zipfile_buffer(data_path):
    return zipfile.ZipFile(data_path, "r")


def fix_area_5_error(data_path: Union[str, Path]):
    print("Fixing error in Area_5 of S3DIS dataset. This can take a while.")

    error_file = (
        "Stanford3dDataset_v1.2/Area_5/office_19/Annotations/ceiling_1.txt"
    )
    row_index = 323473
    column_index = 5
    correct_value = "131.000000"

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the entire zip archive to the temporary directory
        with zipfile.ZipFile(data_path, "r") as zip_file:
            zip_file.extractall(temp_dir)

        # Fix the error file in place
        error_file_path = Path(temp_dir) / error_file
        with open(error_file_path, "r") as file:
            lines = file.readlines()
            columns = lines[row_index].split()
            columns[column_index] = correct_value
            lines[row_index] = " ".join(columns) + "\n"
        with open(error_file_path, "w") as file:
            file.writelines(lines)

        # Compress the temporary directory back into a zip archive
        with zipfile.ZipFile(
            data_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as zip_file:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    joined_path = os.path.join(root, file)
                    zip_file.write(
                        joined_path, os.path.relpath(joined_path, temp_dir)
                    )


class S3DISObjectsBuilder(DatasetBuilder):
    DATA_FILE = "Stanford3dDataset_v1.2.zip"
    SPLITS = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
    FILE_FORMAT = "txt"

    LABEL_NAMES = LABEL_NAMES

    def __init__(
        self,
        data_path: PathLike | None = None,
        num_pre_sample_points: int = 16384,
        min_num_points: int = 2048,
        fix_original_archive: bool = False,
        seed: int | None = None,
    ):
        if data_path is None:
            data_path = Path(__file__).parent / "data" / S3DISObjectsBuilder.DATA_FILE

        self._set_info(
            name="S3DIS",
            data_path=data_path,
            splits=S3DISObjectsBuilder.SPLITS,
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
                PCFieldKey.CLOUD_LABEL: ClassLabels(
                    S3DISObjectsBuilder.LABEL_NAMES
                ),
            },
        )

        self.num_pre_sample_points = num_pre_sample_points
        self.min_num_points = min_num_points
        self.np_rng = np.random.default_rng(seed)
        self.fix_original_archive = fix_original_archive

    def process_instance(
        self, annotations_path: str
    ) -> dict[str, str | np.ndarray] | None:
        zip_file = open_zipfile_buffer(self.data_path)
    
        with zip_file.open(annotations_path) as file:
            str_label = annotations_path.split("/")[-1].split("_")[0]
            points = np.loadtxt(file, delimiter=" ", dtype=np.float32)
        
        # Random sample of points if more than num_pre_sample_points
        if len(points) > self.num_pre_sample_points:
            points = self.np_rng.choice(
                points, self.num_pre_sample_points, replace=False
            )
        if len(points) < self.min_num_points:
            # print(f"Warning: {annotations_path} has less than {self.min_num_points} points: {len(points)}")
            return None

        path_parts = annotations_path.split("/")

        name = f"{path_parts[2]}_{path_parts[-1].rsplit('.', 1)[0]}"

        return {
            "name": name,
            PCFieldKey.POINTS: points[:, :3],
            PCFieldKey.CLOUD_LABEL: str_label,
        }

    def iterate_data(
        self, split: str, num_workers: int = 1
    ) -> Iterable[dict[str, str | np.ndarray] | None]:
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}")
        if split == "Area_5" and self.fix_original_archive:
            fix_area_5_error(self.data_path)

        return _DataIterator(
            self,
            split=split,
            num_workers=num_workers,
        )


class _DataIterator:
    def __init__(
        self,
        builder: S3DISObjectsBuilder,
        split: str,
        num_workers: int | None = 0,
    ):
        self.builder = builder
        self.num_workers = num_workers or 0

        name_list = zipfile.ZipFile(builder.data_path, "r").namelist()

        self.annotations_paths = [
            path
            for path in name_list
            if path.endswith(builder.FILE_FORMAT)
            and path.split("/")[-2] == "Annotations"
            and path.split("/")[-4] == split
        ]

        self.results_len = len(self.annotations_paths)

    def __iter__(self) -> Iterator[dict[str, str | np.ndarray] | None]:
        if self.num_workers > 0:
            with Pool(self.num_workers) as pool:
                self.results = pool.imap_unordered(
                    self.builder.process_instance, self.annotations_paths, chunksize=1
                )
                for result in self.results:
                    yield result
        else:
            for path in self.annotations_paths:
                yield self.builder.process_instance(path)

    def __len__(self):
        return self.results_len
