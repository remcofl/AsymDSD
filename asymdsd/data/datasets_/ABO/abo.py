import tarfile
from collections.abc import Iterable, Iterator
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from asymdsd.components.common_types import PathLike
from asymdsd.data import (
    DataField,
    DatasetBuilder,
    FieldType,
    PCFieldKey,
)
from asymdsd.data.transforms import DecodeMesh, SampleSurfacePoints


@lru_cache(1)
def open_tarfile_buffer(data_path):
    return tarfile.open(data_path, "r")


class ABOBuilder(DatasetBuilder):
    DATA_FILE = "abo-3dmodels.tar"
    ABO_LISTINGS_FILE = "abo-listings.tar"
    SPLITS = ["train"]
    FILE_FORMAT = "glb"

    def __init__(
        self,
        data_path: PathLike | None = None,
        num_pre_sample_points: int = 16384,
        seed: int | None = None,
    ):
        if data_path is None:
            data_path = Path(__file__).parent / "data" / ABOBuilder.DATA_FILE

        self._set_info(
            name="Amazon Berkeley Objects",
            data_path=data_path,
            splits=ABOBuilder.SPLITS,
            data_fields=[
                DataField(
                    key=PCFieldKey.POINTS,
                    key_type=FieldType.ARRAY,
                ),
            ],
        )

        self.num_pre_sample_points = num_pre_sample_points
        self.decode_mesh = DecodeMesh(format=ABOBuilder.FILE_FORMAT)
        self.sample_surface = SampleSurfacePoints(
            num_points=num_pre_sample_points,
            seed=seed,
        )

    def process_instance(
        self, object_member: tarfile.TarInfo
    ) -> dict[str, str | np.ndarray]:
        tar = open_tarfile_buffer(self.data_path / self.DATA_FILE)
        with tar.extractfile(object_member) as object_file:  # type: ignore
            mesh_binary = object_file.read()

        mesh = self.decode_mesh(mesh_binary)
        points = self.sample_surface(mesh)
        points = points[:, [2, 0, 1]]

        name = object_member.name.split("/")[-1].split(".")[0]

        return {
            "name": name,
            PCFieldKey.POINTS: points,
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
        builder: ABOBuilder,
        split: str,
        num_workers: int | None = 1,
    ):
        self.builder = builder
        self.num_workers = num_workers or 0

        # Index the large 3D data tar file for fast non-sequential access among workers
        with tarfile.open(builder.data_path / builder.DATA_FILE, "r") as tar:
            self.object_files = [
                member for member in tar.getmembers() if member.name.endswith(".glb")
            ]

        self.results_len = len(self.object_files)

    def __iter__(self) -> Iterator[dict[str, str | np.ndarray]]:
        if self.num_workers > 0:
            with Pool(self.num_workers, maxtasksperchild=10) as pool:
                self.results = pool.imap_unordered(
                    self.builder.process_instance, self.object_files, chunksize=4
                )
                for result in self.results:
                    yield result
        else:
            for object_file in self.object_files:
                yield self.builder.process_instance(object_file)

    def __len__(self):
        return self.results_len
