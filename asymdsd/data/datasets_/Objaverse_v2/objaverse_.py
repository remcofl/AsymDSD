import os
import pickle
import queue
import random
import threading
import zipfile
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache, partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import objaverse
import open3d as o3d
import zarr
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

from asymdsd.components.common_types import PathLike
from asymdsd.data import (
    ClassLabels,
    DataField,
    DatasetBuilder,
    FieldType,
    PCFieldKey,
)

from .lvis_label_names import LABEL_NAMES


@lru_cache(1)
def open_zipfile_buffer(data_path):
    return zipfile.ZipFile(data_path, "r")


class ObjaverseV2Builder(DatasetBuilder):
    SPLITS = ["1_0", "xl"]
    REPO_ID = "tiange/Cap3D"
    SUBFOLDER = "PointCloud_zips"
    INFO_FILE = "compressed_files_info.pkl"
    CHUNK_SIZE = 1000
    FILE_FORMAT = "ply"

    LABEL_NAMES = LABEL_NAMES

    def __init__(
        self,
        data_path: PathLike | None = None,
        skip_download_and_process: bool = False,
        lvis_annotations: bool = False,
        lvis_few_shot: int | None = None,
        lvis_few_shot_seed: int | list[int] | None = None,
    ):
        if data_path is None:
            data_path = Path(__file__).parent / "data"

        self._set_info(
            name="Objaverse v2",
            data_path=data_path,
            splits=ObjaverseV2Builder.SPLITS,
            data_fields=[
                DataField(
                    key=PCFieldKey.POINTS,
                    key_type=FieldType.ARRAY,
                ),
                DataField(
                    key=PCFieldKey.CLOUD_LABEL,
                    key_type=FieldType.STRING_LABEL,
                ),
                # Could add colors
            ],
            class_labels={
                PCFieldKey.CLOUD_LABEL: ClassLabels(ObjaverseV2Builder.LABEL_NAMES)
            },
        )

        self.skip_download_and_process = skip_download_and_process
        self.lvis_annotations = lvis_annotations
        self.lvis_few_shot = lvis_few_shot

        if lvis_few_shot is not None:
            if isinstance(lvis_few_shot_seed, int):
                self.lvis_few_shot_seed = [lvis_few_shot_seed]
            elif isinstance(lvis_few_shot_seed, list):
                self.lvis_few_shot_seed = lvis_few_shot_seed
            else:
                self.lvis_few_shot_seed = [0]

    @classmethod
    def process_instance(
        cls, args: dict[str, int | str], data_root: str, zip_path: str
    ) -> None:
        uid = args["uid"]
        chunk = args["chunk"]
        split = args["split"]
        write_path = f"{data_root}/{split}/{chunk:03d}"

        array_path = f"{uid}/points"
        store = zarr.DirectoryStore(write_path)
        points_group = zarr.open_group(store, mode="a")

        if array_path in points_group:
            return

        zip_ref = open_zipfile_buffer(zip_path)
        ply_path = f"Cap3D_pcs/{uid}.ply"
        with zip_ref.open(ply_path) as f:
            mesh_binary = f.read()

        pcd = o3d.io.read_point_cloud_from_bytes(
            mesh_binary,
            "mem::xyz",
        )  # Does not support colors

        points = np.asarray(pcd.points, dtype="float32")

        # Way slower, but supports colors
        # decode_mesh = DecodeMesh(format=ObjaverseV2Builder.FILE_FORMAT)
        # mesh = decode_mesh(mesh_binary)
        # points = np.array(mesh.vertices, dtype="float32")

        points_group[array_path] = zarr.array(
            points,
            dtype=np.float32,
        )

    def download_and_process(
        self,
        meta_data: dict[str, dict[str, int | str]],
        num_workers: int,
    ) -> None:
        meta_data = deepcopy(meta_data)

        all_files = list_repo_files(
            repo_id=ObjaverseV2Builder.REPO_ID, repo_type="dataset"
        )
        zip_files = sorted(
            [
                f
                for f in all_files
                if f.endswith(".zip") and f.startswith(f"{self.SUBFOLDER}/")
            ]
        )

        if len(zip_files) == 0:
            raise ValueError(
                f"No zip files found in {ObjaverseV2Builder.REPO_ID}/{self.SUBFOLDER}"
            )

        chunk_groups = list(self.group_1_0.groups()) + list(self.group_xl.groups())

        for _, chunk_group in tqdm(
            chunk_groups, desc="Scanning to find remaining objects"
        ):
            processed_uids = list(chunk_group.keys())
            # Remove processed uids from meta_data
            for uid in processed_uids:
                meta_data.pop(uid, None)

        grouped_meta_data = defaultdict(list)
        for data in meta_data.values():
            subset = data["subset"]
            data.pop("subset")
            grouped_meta_data[subset].append(data)

        # Final result
        arg_meta_data = dict(grouped_meta_data)

        process_zip_files = {}
        for subset in arg_meta_data.keys():
            file = f"{self.SUBFOLDER}/{subset}.zip"
            if file not in zip_files:
                print(f"Warning: {file} not found in zip files. Skipping this file.")
            else:
                process_zip_files[subset] = file

        # === Shared queue ===
        zip_queue = queue.Queue(maxsize=1)

        # === Downloader thread ===
        def downloader():
            for subset, zip_filename in process_zip_files.items():
                print(f"[Downloader] Downloading {zip_filename}...")
                local_path = hf_hub_download(
                    repo_id=ObjaverseV2Builder.REPO_ID,
                    filename=zip_filename,
                    cache_dir=self.data_path,
                    repo_type="dataset",
                )
                print(f"[Downloader] Enqueuing {zip_filename}")
                zip_queue.put((subset, local_path))  # blocks if full
            zip_queue.put(None)  # Sentinel

        # === Processor thread ===
        def process_zip(zip_path: str, arg_data: list[dict[str, int | str]]) -> None:
            print(f"[Processor] Processing {os.path.basename(zip_path)}")

            process_fn = partial(
                ObjaverseV2Builder.process_instance,
                data_root=str(Path(self.root.store.path)),  # type: ignore
                zip_path=zip_path,
            )
            needed_groups = set((data["split"], data["chunk"]) for data in arg_data)
            for split, chunk in needed_groups:
                self.root.require_group(f"{split}/{chunk:03d}")

            with Pool(num_workers) as pool:
                for _ in tqdm(
                    pool.imap_unordered(process_fn, arg_data, chunksize=100),
                    total=len(arg_data),
                    desc=f"Processing {os.path.basename(zip_path)}",
                ):
                    pass
            print(f"[Processor] Done — deleting {os.path.basename(zip_path)}")

            # This deletes both the symlink and the real blob
            real_path = os.path.realpath(zip_path)
            print(f"[Cleanup] Removing symlink: {zip_path}")
            os.remove(zip_path)
            if os.path.exists(real_path) and real_path != zip_path:
                print(f"[Cleanup] Removing blob: {real_path}")
                os.remove(real_path)

        threading.Thread(target=downloader, daemon=True).start()

        while True:
            res = zip_queue.get()
            if res is None:
                break
            subset, zip_path = res
            process_zip(zip_path, arg_meta_data[subset])

    def build(
        self, dataset_save_path: PathLike, num_workers: int | None = None
    ) -> None:
        dataset_save_path = Path(dataset_save_path).expanduser().resolve()
        num_workers = num_workers or 1

        # Open zarr group and download data
        sync_path = dataset_save_path / ".zarr.lock"
        sync = zarr.ProcessSynchronizer(str(sync_path))

        self.root = zarr.open_group(str(dataset_save_path), mode="a", synchronizer=sync)
        self.group_1_0 = self.root.require_group("1_0")
        self.group_xl = self.root.require_group("xl")

        if "complete" in self.root.attrs:
            print("Dataset is already complete.")
            return

        info_file = hf_hub_download(
            repo_id=ObjaverseV2Builder.REPO_ID,
            filename=f"{self.SUBFOLDER}/{ObjaverseV2Builder.INFO_FILE}",
            cache_dir=self.data_path,
            repo_type="dataset",
        )

        # Load the info file
        with open(info_file, "rb") as f:
            info: dict[str, list[str]] = pickle.load(f)

        # i = -1
        # meta_data = {
        #     uid: {
        #         "chunk": (i := i + 1) // ObjaverseV2Builder.CHUNK_SIZE,
        #         "uid": uid,
        #         "subset": subset,
        #         "split": "1_0" if int(subset.split("_")[-1]) < 10 else "xl",
        #     }
        #     for subset, uids in info.items()
        #     for uid in uids
        # }

        counters = {"1_0": 0, "xl": 0}
        meta_data = {}

        for subset, uids in info.items():
            split = "1_0" if int(subset.split("_")[-1]) < 10 else "xl"
            for uid in uids:
                chunk = counters[split] // ObjaverseV2Builder.CHUNK_SIZE
                meta_data[uid] = {
                    "chunk": chunk,
                    "uid": uid,
                    "subset": subset,
                    "split": split,
                }
                counters[split] += 1

        if not self.skip_download_and_process:
            self.download_and_process(meta_data, num_workers=num_workers)

        if self.lvis_annotations and self.lvis_few_shot is not None:
            lvis = objaverse.load_lvis_annotations()
            lvis_group = self.root.require_group("lvis")

            str2int = self.class_labels[PCFieldKey.CLOUD_LABEL].str2int  # type: ignore

            path2labels = {}
            labels2paths = defaultdict(list)
            num_missing_objects = 0
            for str_label, uids in lvis.items():
                int_label = str2int(str_label)
                for uid in uids:
                    object_meta_data = meta_data.get(uid)
                    if object_meta_data is None:
                        num_missing_objects += 1
                        continue
                    chunk = object_meta_data["chunk"]
                    split = object_meta_data["split"]

                    path = f"{split}/{chunk:03d}/{uid}"
                    path2labels[path] = int_label
                    labels2paths[str_label].append(path)

            if num_missing_objects > 0:
                print(
                    f"Warning: {num_missing_objects} objects are missing from the dataset."
                )
            lvis_group.attrs[PCFieldKey.CLOUD_LABEL] = path2labels
            lvis_group.attrs["paths"] = list(path2labels.keys())

            # lvis_group.attrs["label_names"] = self.root.attrs["label_names"]

            label_names = {
                PCFieldKey.CLOUD_LABEL: self.class_labels[  # type: ignore
                    PCFieldKey.CLOUD_LABEL
                ].label_names
            }

        if self.lvis_few_shot is not None:
            labels2paths = dict(labels2paths)
            # Gather all labels with more than lvis_few_shot instances
            labels2paths_sub = {
                label: paths
                for label, paths in labels2paths.items()
                if len(paths) >= self.lvis_few_shot
            }

            lvis_class_labels = ClassLabels(list(labels2paths_sub.keys()))
            lvis_str2int = lvis_class_labels.str2int

            label_key = f"lvis_s{self.lvis_few_shot}"
            # Process for each seed
            for seed in self.lvis_few_shot_seed:
                # Initialize random generator with current seed
                rng = random.Random(seed)

                train_labels = {}
                test_labels = {}

                for str_label, paths in labels2paths_sub.items():
                    int_label = lvis_str2int(str_label)
                    # Create a copy to avoid modifying the original paths
                    paths_copy = paths.copy()
                    # Shuffle with the current seed's RNG
                    rng.shuffle(paths_copy)

                    train_paths = paths_copy[: self.lvis_few_shot]
                    test_paths = paths_copy[self.lvis_few_shot :]

                    for path in train_paths:
                        train_labels[path] = int_label
                    for path in test_paths:
                        test_labels[path] = int_label

                # Create groups with seed in the name

                lvis_fewshot_train_group = self.root.require_group(
                    f"lvis_train_s{self.lvis_few_shot}f{seed}"
                )
                lvis_fewshot_test_group = self.root.require_group(
                    f"lvis_test_s{self.lvis_few_shot}f{seed}"
                )

                lvis_fewshot_train_group.attrs[label_key] = train_labels
                lvis_fewshot_test_group.attrs[label_key] = test_labels
                lvis_fewshot_train_group.attrs["paths"] = list(train_labels.keys())
                lvis_fewshot_test_group.attrs["paths"] = list(test_labels.keys())

            # Add to label_names for each seed
        label_names[label_key] = lvis_class_labels.label_names  # type: ignore

        self.root.attrs["label_names"] = label_names

        # Find all array paths and store them in a list
        for split_group in [self.group_1_0, self.group_xl]:
            array_paths = []
            for chunk in tqdm(split_group, desc="Finding array paths"):
                for item_group_key in split_group[chunk].group_keys():  # type: ignore
                    array_paths.append(f"train/{chunk}/{item_group_key}")
            split_group.attrs["paths"] = array_paths

        self.data_fields.append(
            DataField(key=label_key, key_type=FieldType.STRING_LABEL)
        )

        self.root.attrs["name"] = self.name
        self.root.attrs["splits"] = self.splits
        self.root.attrs["attr_keys"] = [
            field.key
            for field in self.data_fields
            if field.key_type == FieldType.STRING_LABEL
            or field.key_type == FieldType.INT_LABEL
        ]
        self.root.attrs["array_keys"] = [
            field.key for field in self.data_fields if field.key_type == FieldType.ARRAY
        ]

        self.root.attrs["complete"] = True
        print("Dataset build complete.")

    def iterate_data(self, split: str, num_workers: int | None = 1):
        return iter([])
