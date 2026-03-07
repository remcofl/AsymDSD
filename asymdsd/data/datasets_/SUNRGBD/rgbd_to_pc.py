# Based on the original matlab code from the SUNRGBD dataset,
# the function read_3d_pts_general is used to convert the depth map to 3D points using camera intrinsics.
# The function read3dPoints reads the depth map and converts it to a 3D point cloud.

from zipfile import ZipFile

import cv2
import numpy as np


def imread_from_zip(zip_file: ZipFile, path: str):
    img_bytes = zip_file.read(path)
    img_buffer = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)


def read_3d_pts_general(depthInpaint, K, depthInpaintsize):
    """Converts depth map to 3D points using camera intrinsics"""
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    invalid = depthInpaint == 0

    # Generate 3D points
    x, y = np.meshgrid(np.arange(depthInpaintsize[1]), np.arange(depthInpaintsize[0]))
    x3 = (x - cx) * depthInpaint / fx
    y3 = (y - cy) * depthInpaint / fy
    z3 = depthInpaint

    points3dMatrix = np.stack((x3, z3, -y3), axis=-1)
    points3dMatrix[invalid] = np.nan
    points3d = points3dMatrix.reshape(-1, 3)
    points3d[invalid.ravel(), :] = np.nan

    return points3d


def read_3d_points(meta_data, zip_file):
    """Reads the depth map and converts it to 3D point cloud."""
    depth_path = meta_data["depthpath"][0]
    depthVis = imread_from_zip(zip_file, depth_path[17:])
    depthInpaint = (
        np.bitwise_or(
            np.right_shift(depthVis, 3), np.left_shift(depthVis, 16 - 3)
        ).astype(np.float32)
        / 1000.0
    )
    depthInpaint[depthInpaint > 8] = 8
    points3d = read_3d_pts_general(depthInpaint, meta_data["K"], depthInpaint.shape)
    points3d = (np.dot(meta_data["Rtilt"], points3d.T)).T
    return points3d


def extract_object_points(points3d, bbox):
    """Extracts points within the given 3D bounding box."""
    x_min, x_max = bbox[0] - bbox[3], bbox[0] + bbox[3]
    y_min, y_max = bbox[1] - bbox[4], bbox[1] + bbox[4]
    z_min, z_max = bbox[2] - bbox[5], bbox[2] + bbox[5]
    mask = (
        (points3d[:, 0] >= x_min)
        & (points3d[:, 0] <= x_max)
        & (points3d[:, 1] >= y_min)
        & (points3d[:, 1] <= y_max)
        & (points3d[:, 2] >= z_min)
        & (points3d[:, 2] <= z_max)
    )
    return points3d[mask]
