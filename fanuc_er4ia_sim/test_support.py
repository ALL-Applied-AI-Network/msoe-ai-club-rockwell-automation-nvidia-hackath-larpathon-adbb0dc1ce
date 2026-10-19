"""Construct a calibrated synthetic camera for host-side regression tests."""

import numpy as np
import scene_config as C
from backproject import CameraModel


def camera_model(cfg):
    eye = np.array(cfg["eye"], float)
    axis = np.array(cfg["look_at"], float) - eye
    axis /= np.linalg.norm(axis)
    right = np.cross(axis, cfg["up"])
    right /= np.linalg.norm(right)
    up = np.cross(right, axis)
    matrix = np.eye(4)
    matrix[:3, :3] = [right, up, -axis]
    matrix[3, :3] = eye
    w, h = cfg["resolution"]
    return CameraModel(
        dict(
            resolution=(w, h),
            focal_length_mm=cfg["focal_length"],
            horizontal_aperture_mm=cfg["horiz_aperture"],
            vertical_aperture_mm=cfg["horiz_aperture"] * h / w,
            camera_to_world=matrix.tolist(),
            table_top_z=C.TABLE_TOP_Z + C.SURFACE_DZ,
        )
    )
