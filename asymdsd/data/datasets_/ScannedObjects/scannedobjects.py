import zipfile
from collections.abc import Iterable, Iterator
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


class ScannedObjectsBuilder(DatasetBuilder):
    DATA_FILE = "ScannedObjects"
    SPLITS = ["train"]
    FILE_FORMAT = "obj"
    MESH_FILE = "meshes/model.obj"

    def __init__(
        self,
        data_path: PathLike | None = None,
        num_pre_sample_points: int = 16384,
        seed: int | None = None,
    ):
        if data_path is None:
            data_path = Path(__file__).parent / "data" / ScannedObjectsBuilder.DATA_FILE

        self._set_info(
            name="ScannedObjects",
            data_path=data_path,
            splits=ScannedObjectsBuilder.SPLITS,
            data_fields=[
                DataField(
                    key=PCFieldKey.POINTS,
                    key_type=FieldType.ARRAY,
                ),
            ],
        )

        self.num_pre_sample_points = num_pre_sample_points
        self.decode_mesh = DecodeMesh(format=ScannedObjectsBuilder.FILE_FORMAT)
        self.sample_surface = SampleSurfacePoints(
            num_points=num_pre_sample_points,
            seed=seed,
        )

    def process_instance(self, path: Path) -> dict[str, str | np.ndarray]:
        zip_file = zipfile.ZipFile(path, "r")
        mesh_binary = zip_file.read(ScannedObjectsBuilder.MESH_FILE)

        mesh = self.decode_mesh(mesh_binary)
        points = self.sample_surface(mesh)
        name = path.stem

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
        builder: ScannedObjectsBuilder,
        split: str,
        num_workers: int | None = 1,
    ):
        self.builder = builder
        self.num_workers = num_workers or 0

        # List all zip files in builder.data_path
        self.paths = list(Path(builder.data_path).glob("*.zip"))
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
