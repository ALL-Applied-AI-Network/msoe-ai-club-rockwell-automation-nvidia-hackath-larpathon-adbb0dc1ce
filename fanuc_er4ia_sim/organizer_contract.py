"""Host-side validation of the organizer observation and TCP action contract."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


RGB_CAMERAS = ("table", "side", "wrist")
DEPTH_SHAPES = {"table": (400, 640), "wrist": (480, 848)}
TCP_LIMITS = {
    "x": (150.0, 650.0),
    "y": (-450.0, 450.0),
    "z": (-334.0, 300.0),
    "orientation": (-360.0, 360.0),
    "dt": (0.1, 1.0),
}


def _numeric_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and numeric")
    return array


def validate_observation(
    observation: dict,
    *,
    cameras: Iterable[str] = RGB_CAMERAS,
    image_size: tuple[int, int] = (1920, 1080),
    include_depth: bool = False,
) -> dict:
    """Validate the documented local client payload; return normalized metadata."""
    required = {"t", "status", "trial", "queued", "joints", "tcp_pose", "images"}
    missing = sorted(required - set(observation))
    if missing:
        raise ValueError(f"observation missing fields: {missing}")
    for name in ("t", "trial", "queued"):
        if not isinstance(observation[name], (int, float)) or not math.isfinite(
            float(observation[name])
        ):
            raise ValueError(f"observation {name} must be finite and numeric")
    if not isinstance(observation["status"], str):
        raise ValueError("observation status must be a string")
    joints = _numeric_array(observation["joints"], (7,), "joints")
    if float(joints[6]) not in (0.0, 1.0):
        raise ValueError("observed gripper state must be binary")
    _numeric_array(observation["tcp_pose"], (6,), "tcp_pose")
    requested = tuple(cameras)
    if any(camera not in RGB_CAMERAS for camera in requested):
        raise ValueError("unknown RGB camera")
    width, height = image_size
    if not (16 <= width <= 1920 and 16 <= height <= 1080):
        raise ValueError("requested RGB size exceeds local client limits")
    images = observation["images"]
    if not isinstance(images, dict) or set(images) != set(requested):
        raise ValueError("images must contain exactly the requested RGB cameras")
    for camera in requested:
        image = np.asarray(images[camera])
        if image.shape != (height, width, 3) or image.dtype != np.uint8:
            raise ValueError(f"{camera} RGB must be uint8 HxWx3 at requested size")
    if include_depth:
        depth = observation.get("depth")
        if not isinstance(depth, dict) or set(depth) != set(DEPTH_SHAPES):
            raise ValueError("depth must contain exactly table and wrist")
        for camera, shape in DEPTH_SHAPES.items():
            array = np.asarray(depth[camera])
            if array.shape != shape or array.dtype != np.uint16:
                raise ValueError(f"{camera} depth must be uint16 with shape {shape}")
    return {
        "valid": True,
        "rgb_size": [width, height],
        "rgb_cameras": list(requested),
        "depth_validated": include_depth,
        "depth_units": "millimetres_along_optical_axis",
        "zero_depth_is_invalid": True,
    }


def validate_tcp_actions(actions: object, *, dt_seconds: float) -> np.ndarray:
    """Strictly validate absolute UF0 TCP rows [x,y,z,w,p,r,gripper]."""
    rows = np.asarray(actions, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 7 or not 1 <= rows.shape[0] <= 100:
        raise ValueError("TCP actions must have shape (N, 7), with 1 <= N <= 100")
    if not np.isfinite(rows).all():
        raise ValueError("TCP actions must be finite")
    if not TCP_LIMITS["dt"][0] <= float(dt_seconds) <= TCP_LIMITS["dt"][1]:
        raise ValueError("TCP action dt is outside [0.1, 1.0] seconds")
    bounds = (TCP_LIMITS["x"], TCP_LIMITS["y"], TCP_LIMITS["z"])
    for column, (minimum, maximum) in enumerate(bounds):
        if ((rows[:, column] < minimum) | (rows[:, column] > maximum)).any():
            raise ValueError(f"TCP translation column {column} outside documented limits")
    if (np.abs(rows[:, 3:6]) > 360.0).any():
        raise ValueError("TCP orientation outside documented limits")
    if not np.isin(rows[:, 6], (0.0, 1.0)).all():
        raise ValueError("commanded gripper values must be exactly binary")
    return rows


def hold_still_action(observation: dict) -> np.ndarray:
    """Create a numeric one-row hold proposal from the current observed pose."""
    pose = _numeric_array(observation["tcp_pose"], (6,), "tcp_pose").astype(float)
    joints = _numeric_array(observation["joints"], (7,), "joints")
    gripper = float(joints[6])
    if gripper not in (0.0, 1.0):
        raise ValueError("cannot construct hold with nonbinary gripper observation")
    return np.concatenate((pose, [gripper])).reshape(1, 7)


ORGANIZER_CONTRACT_AUDIT = {
    "placement_top_level": "rollouts",
    "placement_value": "[x_mm, y_mm, optional_yaw_deg] in robot frame",
    "submission_rollouts": 10,
    "policy_reset": "at_start_of_every_rollout",
    "tcp_action": "absolute_uf0_[x_y_z_w_p_r_binary_gripper]",
    "straight_down_orientation": [-180.0, 0.0, "R"],
    "rgb_native_maximum": {"table": [1920, 1080], "side": [1920, 1080], "wrist": [1280, 720]},
    "depth_shapes_hw": {key: list(value) for key, value in DEPTH_SHAPES.items()},
    "server_side_semantics": "unresolved_beyond_local_parser",
}
