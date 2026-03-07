import os
import tempfile
import threading
import time
import urllib.request
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

import numpy as np
import objaverse
import psutil
import trimesh
import zarr
from tqdm import tqdm
from trimesh.parent import Geometry

from asymdsd.components.common_types import PathLike
from asymdsd.data import (
    DataField,
    DatasetBuilder,
    FieldType,
    PCFieldKey,
)
from asymdsd.data.transforms import SampleSurfacePoints


class MemoryExceededError(Exception):
    """Custom exception to handle memory limit exceedance."""

    pass


class ObjaverseBuilder(DatasetBuilder):
    SPLITS = ["train"]
    FILE_FORMAT = "glb"
    HF_URL = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"

    def __init__(
        self,
        data_path: PathLike | None = None,
        num_pre_sample_points: int = 16384,
        min_num_points: int = 2048,
        memory_limit_per_process_mb: int = 8192,
        processing_timeout_s: int = 60,
        skip_data_processing: bool = False,
        point_unique_tolerance: float = 1e-6,
        seed: int | None = None,
    ):
        if data_path is None:
            data_path = Path(__file__).parent / "data"

        self._set_info(
            name="Objaverse 1.0",
            data_path=data_path,
            splits=ObjaverseBuilder.SPLITS,
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
        )

        self.num_pre_sample_points = num_pre_sample_points
        self.min_num_points = min_num_points
        self.seed = seed

        self.memory_limit_per_process_mb = memory_limit_per_process_mb
        self.processing_timeout_s = processing_timeout_s
        self.skip_data_processing = skip_data_processing
        self.point_unique_tolerance = point_unique_tolerance

        self.np_rng = np.random.default_rng(seed)

    @classmethod
    def extract_points(
        cls, geometry: Geometry, num_pre_sample_points: int, seed: int | None = None
    ) -> np.ndarray | None:
        if isinstance(geometry, trimesh.PointCloud) or isinstance(
            geometry, trimesh.path.Path3D
        ):
            # If the mesh is a point cloud, we can use the points directly.
            # Similarly, for a path, the vertices are gathered.
            return geometry.vertices
        elif isinstance(geometry, trimesh.Trimesh):
            return SampleSurfacePoints(
                num_points=num_pre_sample_points,
                seed=seed,
            )(geometry)
        else:
            print(f"Unsupported geometry type: {type(geometry)}")
            pass

    @classmethod
    def process_instance(
        cls,
        args: dict[str, Any],
        data_root: str,
        min_num_points: int = 2048,
        num_pre_sample_points: int = 16384,
        memory_limit_mb: int = 8192,
        timeout_s: int = 60,
        point_unique_tolerance: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        uid = args["uid"]
        object_path = args["object_path"]
        chunk = args["chunk"]
        write_path = f"{data_root}/train/{chunk:03d}"

        array_path = f"{uid}/points"
        store = zarr.DirectoryStore(write_path)
        points_group = zarr.open_group(store, mode="a")

        if array_path in points_group:
            return

        try:
            # Download first, without timeout
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = os.path.join(tmp_dir, "object.glb")
                hf_url = f"{ObjaverseBuilder.HF_URL}/{object_path}"
                urllib.request.urlretrieve(hf_url, local_path)

                def process():
                    try:
                        geometry = trimesh.load_mesh(
                            local_path, ObjaverseBuilder.FILE_FORMAT
                        )

                        if isinstance(geometry, trimesh.Scene):
                            # In case of a scene, we need to concatenate all geometries into a single mesh
                            # Trimesh, Path2D, Path3D, PointCloud
                            if isinstance(
                                list(geometry.geometry.values())[0], trimesh.PointCloud
                            ):
                                geometry_list: list[Geometry] = geometry.dump(  # type: ignore
                                    concatenate=False
                                )

                                points = []
                                for pc in geometry_list:  # type: ignore
                                    new_points = ObjaverseBuilder.extract_points(
                                        pc, num_pre_sample_points, seed
                                    )
                                    points.append(new_points)
                                points = np.concatenate(points)
                            else:
                                geometry = geometry.dump(concatenate=True)
                                points = ObjaverseBuilder.extract_points(
                                    geometry, num_pre_sample_points, seed
                                )
                        else:
                            points = ObjaverseBuilder.extract_points(
                                geometry, num_pre_sample_points, seed
                            )

                        if points is None or points.shape[0] < min_num_points:
                            print(
                                f"Skipping {chunk:03d}/{uid} due to insufficient points."
                            )
                            return

                        if points.shape[0] > num_pre_sample_points:
                            selected_indices = np.random.choice(
                                points.shape[0], num_pre_sample_points, replace=False
                            )
                            points = points[selected_indices]

                        if np.any(np.isnan(points)):
                            print(f"Skipping {chunk:03d}/{uid} due to present NaNs.")
                            return

                        tol = (
                            np.max((np.max(points, axis=0) - np.min(points, axis=0)))
                            * point_unique_tolerance
                        )
                        if (
                            tol == 0
                            or np.unique(np.round(points / tol) * tol, axis=0).shape[0]
                            < min_num_points
                        ):
                            print(f"Too few unique points in {chunk:03d}/{uid}")
                            return

                        points = points[:, [2, 0, 1]]  # Convert to Z-up

                        points_group[array_path] = zarr.array(
                            points,
                            dtype=np.float32,
                            # compressor=compressor,
                        )
                    except Exception as e:
                        print(f"Error processing {local_path}: {e}")

                def monitor_memory():
                    """Monitors memory usage, including subprocesses, and exits if it exceeds the threshold."""
                    # Note does not always work, likely due to other threads/processes
                    process = psutil.Process(os.getpid())  # Get the main process
                    while True:
                        mem_usage = process.memory_info().rss  # Main process memory
                        for child in process.children(
                            recursive=True
                        ):  # Include subprocesses
                            mem_usage += child.memory_info().rss

                        mem_usage_mb = mem_usage / (1024 * 1024)  # Convert to MB
                        if mem_usage_mb > memory_limit_mb:
                            raise MemoryExceededError(
                                f"Memory limit exceeded for object {uid}: {mem_usage_mb:.2f}MB"
                            )
                        time.sleep(0.5)

                thread = threading.Thread(target=process)
                monitor_thread = threading.Thread(target=monitor_memory, daemon=True)

                thread.start()
                monitor_thread.start()

                thread.join(timeout=timeout_s)

                if thread.is_alive():
                    print(f"Timeout while loading object {uid}")
                    return None  # Skip this task

        except MemoryExceededError:
            print(f"Memory limit exceeded for object {uid}")
            return None  # Gracefully return

        except Exception:
            print(f"Unexpected error processing {uid}")
            return None  # Skip this task

    def build(
        self, dataset_save_path: PathLike, num_workers: int | None = None
    ) -> None:
        dataset_save_path = Path(dataset_save_path).expanduser().resolve()
        num_workers = num_workers or 0

        # Open zarr group and download data
        self.root = zarr.open_group(str(dataset_save_path), mode="a")
        self.train_group = self.root.require_group("train")

        if "complete" in self.root.attrs:
            print("Dataset is already complete.")
            return

        if not self.skip_data_processing:
            meta_data = objaverse._load_object_paths()
            # Assign uids to chunks
            chunk_size = 1000

            meta_data = {
                kv_pair[0]: {
                    "chunk": i // chunk_size,
                    "uid": kv_pair[0],
                    "object_path": kv_pair[1],
                }
                for i, kv_pair in enumerate(meta_data.items())
            }

            chunk_groups = list(self.train_group.groups())

            for _, chunk_group in tqdm(
                chunk_groups, desc="Scanning to find remaining objects"
            ):
                processed_uids = list(chunk_group.keys())
                # Remove processed uids from meta_data
                for uid in processed_uids:
                    meta_data.pop(uid, None)

            meta_data = list(meta_data.values())

            self.np_rng.shuffle(meta_data)

            process_fn = partial(
                ObjaverseBuilder.process_instance,
                data_root=str(Path(self.root.store.path)),  # type: ignore
                min_num_points=self.min_num_points,
                num_pre_sample_points=self.num_pre_sample_points,
                seed=self.seed,
                memory_limit_mb=self.memory_limit_per_process_mb,
                timeout_s=self.processing_timeout_s,
                point_unique_tolerance=self.point_unique_tolerance,
            )
            if num_workers > 0:
                total_items = len(meta_data)
                pbar = tqdm(total=total_items, desc="Processing dataset")

                with Pool(num_workers, maxtasksperchild=1) as pool:
                    for _ in pool.imap_unordered(process_fn, meta_data, chunksize=1):
                        pbar.update(1)  # Update tqdm for each processed item

                pbar.close()  # Close tqdm at the end

            else:
                # Maybe delete this case altogether due to the memory leak issue
                for args in tqdm(meta_data, desc="Processing dataset"):
                    process_fn(args)

        # Find all array paths and store them in a list
        array_paths = []
        split_group = self.train_group
        for chunk in tqdm(split_group, desc="Finding array paths"):
            for item_group_key in split_group[chunk].group_keys():  # type: ignore
                array_paths.append(f"train/{chunk}/{item_group_key}")
        split_group.attrs["paths"] = array_paths

        self.root.attrs["name"] = self.name
        self.root.attrs["splits"] = self.splits
        self.root.attrs["array_keys"] = [
            field.key for field in self.data_fields if field.key_type == FieldType.ARRAY
        ]

        self.root.attrs["complete"] = True

    def iterate_data(self, split: str, num_workers: int | None = 1):
        return iter([])
