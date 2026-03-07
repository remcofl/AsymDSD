import os
import zipfile
from collections.abc import Iterable, Iterator
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.io as sio

from asymdsd.components.common_types import PathLike
from asymdsd.data import (
    ClassLabels,
    DataField,
    DatasetBuilder,
    FieldType,
    PCFieldKey,
)

from .label_names import LABEL_NAMES
from .rgbd_to_pc import extract_object_points, read_3d_points


@lru_cache(1)
def open_zipfile_buffer(data_path):
    return zipfile.ZipFile(data_path, "r")


class SunRGBDBuilder(DatasetBuilder):
    DATA_FILE = "SUNRGBD.zip"
    TOOLBOX_FILE = "SUNRGBDtoolbox.zip"
    BB_FILE = "SUNRGBDMeta3DBB_v2.mat"

    SPLITS = ["train", "test"]
    FILE_FORMAT = "off"

    LABEL_NAMES = LABEL_NAMES

    def __init__(
        self,
        data_path: PathLike | None = None,
        num_pre_sample_points: int = 16384,
        min_num_points: int = 2048,
        seed: int | None = None,
    ):
        if data_path is None:
            data_path = Path(__file__).parent / "data"

        self._set_info(
            name="Sun RGB-D",
            data_path=data_path,
            splits=SunRGBDBuilder.SPLITS,
            data_fields=[
                DataField(
                    key=PCFieldKey.POINTS,
                    key_type=FieldType.ARRAY,
                ),
                # DataField(
                #     key=PCFieldKey.CLOUD_LABEL,
                #     key_type=FieldType.STRING_LABEL,
                # ),
            ],
            # class_labels={
            #     PCFieldKey.CLOUD_LABEL: ClassLabels(SunRGBDBuilder.LABEL_NAMES)
            # },
        )

        # self.labels = []
        # self.class_labels_ = ClassLabels(SunRGBDBuilder.LABEL_NAMES)
        self.num_pre_sample_points = num_pre_sample_points
        self.min_num_points = min_num_points
        self.np_rng = np.random.default_rng(seed)

    def process_instance(self, meta_data) -> list[dict[str, str | np.ndarray] | None]:
        zip_file = open_zipfile_buffer(self.data_path / self.DATA_FILE)
        points3d = read_3d_points(meta_data, zip_file)
        points3d = points3d[~np.isnan(points3d[:, 0])]

        base_name = meta_data["sequenceName"][0][8:].replace("/", "_")

        instances = []
        for j, box3D in enumerate(meta_data["groundtruth3DBB"][0]):
            centroid = box3D["centroid"][0]
            coeffs = np.abs(box3D["coeffs"][0])
            bbox = np.concatenate((centroid, coeffs))
            points = extract_object_points(points3d, bbox).astype(np.float32)

            if len(points) < self.min_num_points:
                instances.append(None)
                continue
            if len(points) > self.num_pre_sample_points:
                points = self.np_rng.choice(
                    points, self.num_pre_sample_points, replace=False
                )

            name = f"{base_name}_{j}"
            instances.append(
                {
                    "name": name,
                    PCFieldKey.POINTS: points,
                    # PCFieldKey.CLOUD_LABEL: label,
                }
            )
        return instances

    def iterate_data(
        self, split: str, num_workers: int | None = 1
    ) -> Iterable[dict[str, str | np.ndarray] | None]:
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
        builder: SunRGBDBuilder,
        split: str,
        num_workers: int | None = 1,
    ):
        self.builder = builder
        self.num_workers = num_workers or 0

        with zipfile.ZipFile(
            Path(self.builder.data_path) / self.builder.TOOLBOX_FILE, "r"
        ) as zip_file:
            with zip_file.open(
                "SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat"
            ) as allsplit_file:
                split_map = sio.loadmat(allsplit_file)

        # This leaves much of the data unused.
        num_paths = len(split_map[f"all{split}"][0])
        paths = {split_map[f"all{split}"][0][i][0][16:] for i in range(num_paths)}

        # Load metadata
        meta_data = sio.loadmat(Path(self.builder.data_path) / self.builder.BB_FILE)
        SUNRGBDMeta = meta_data["SUNRGBDMeta"][0]

        self.meta_data = []
        for id in range(len(SUNRGBDMeta)):
            data = SUNRGBDMeta[id]
            depthpath = data["depthpath"][0]
            depthpath = depthpath[16:]
            filepath = os.path.dirname(os.path.dirname(depthpath))

            if filepath in paths and data["groundtruth3DBB"].size > 0:
                self.meta_data.append(data)

        self.results_len = sum(
            len(meta_data["groundtruth3DBB"][0]) for meta_data in self.meta_data
        )

    def __iter__(self) -> Iterator[dict[str, str | np.ndarray] | None]:
        if self.num_workers > 0:
            with Pool(self.num_workers) as pool:
                self.results = pool.imap_unordered(
                    self.builder.process_instance, self.meta_data, chunksize=4
                )
                for result in self.results:
                    for instance in result:
                        yield instance
        else:
            for path in self.meta_data:
                for instance in self.builder.process_instance(path):
                    yield instance

    def __len__(self):
        return self.results_len
