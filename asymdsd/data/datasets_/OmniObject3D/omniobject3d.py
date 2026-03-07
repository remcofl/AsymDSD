import zipfile
from collections.abc import Iterable, Iterator
from functools import lru_cache
from pathlib import Path

import h5py
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


class OmniObject3DBuilder(DatasetBuilder):
    DATA_FILE = "OmniObject3D.zip"
    SPLITS = ["train"]
    EXPECTED_NUM_ITERS = 5911

    LABEL_NAMES = LABEL_NAMES

    def __init__(
        self,
        data_path: PathLike | None = None,
    ):
        if data_path is None:
            data_path = Path(__file__).parent / "data" / OmniObject3DBuilder.DATA_FILE

        self._set_info(
            name="OmniObject3D",
            data_path=data_path,
            splits=OmniObject3DBuilder.SPLITS,
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
                PCFieldKey.CLOUD_LABEL: ClassLabels(OmniObject3DBuilder.LABEL_NAMES)
            },
        )

    def iterate_data(
        self, split: str, num_workers: int | None = 1
    ) -> Iterable[dict[str, str | np.ndarray]]:
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}")

        return _DataIterator(self)


class _DataIterator:
    def __init__(
        self,
        builder: OmniObject3DBuilder,
    ):
        self.builder = builder

        self.zip_file = zipfile.ZipFile(builder.data_path, "r")
        self.paths = self.zip_file.namelist()

    def __iter__(self) -> Iterator[dict[str, str | np.ndarray]]:
        for path in self.paths:
            h5f = h5py.File(self.zip_file.open(path), "r")

            label = path.split("/")[-1].rsplit("_", 1)[0]

            data: np.ndarray = h5f["data"][:]  # type: ignore

            for index in range(len(data)):  # type: ignore
                yield {
                    "name": f"{label}_{index}",
                    PCFieldKey.POINTS: np.array(
                        data[index][:, [2, 0, 1]], dtype=np.float32
                    ),
                    PCFieldKey.CLOUD_LABEL: label,
                }

    def __len__(self):
        return OmniObject3DBuilder.EXPECTED_NUM_ITERS
