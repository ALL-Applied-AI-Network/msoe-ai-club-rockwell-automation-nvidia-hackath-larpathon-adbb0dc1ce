"""Public, host-safe geometry and calibration contracts for perception.

This module contains task geometry, not scene instances or object poses.  The
simulation workspace below is explicitly scoped to the calibrated simulator;
hardware deployments must provide a separately surveyed polygon and matching
camera calibration rather than silently reusing these coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np


CUBE_EDGE_M = 0.010
CUBE_BOUNDING_RADIUS_M = CUBE_EDGE_M / math.sqrt(2.0)

WORKSPACE_CONFIGURATION_VERSION = "part-source-workspace-v1"
SIMULATION_COORDINATE_FRAME = "fanuc_er4ia_robot_base_world"
SIMULATION_CALIBRATION_VERSION = "fanuc-er4ia-table-camera-v05-sim"
SIMULATION_CALIBRATION_COMPATIBILITY_ID = "fanuc-er4ia-phase4b-sim-v1"
SIMULATION_SOURCE_WORKSPACE_POLYGON_M = (
    (0.245, -0.235),
    (0.505, -0.235),
    (0.505, 0.245),
    (0.245, 0.245),
)
SIMULATION_TABLE_CAMERA_CALIBRATION = {
    "camera": "table",
    "resolution": [1920, 1080],
    "focal_length_mm": 11.215800285339355,
    "horizontal_aperture_mm": 20.954999923706055,
    "vertical_aperture_mm": 11.787187576293945,
    "camera_to_world": [
        [-0.018651004245830327, 0.9989789034382659, 0.041149611491507025, 0.0],
        [-0.7943405646546062, -0.03979944657748935, 0.6061675275007125, 0.0],
        [0.6071863036867872, -0.02138117249838115, 0.7942717658822899, 0.0],
        [0.7770000100135803, -0.04259999841451645, 1.7023999691009521, 1.0],
    ],
    "table_top_z": 0.8949,
    "robot_base_xy": [0.0, 0.0],
}

_CAMERA_HASH_FIELDS = (
    "camera",
    "resolution",
    "focal_length_mm",
    "horizontal_aperture_mm",
    "vertical_aperture_mm",
    "camera_to_world",
    "table_top_z",
    "coordinate_frame",
    "calibration_version",
    "calibration_compatibility_id",
)


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def camera_calibration_sha256(camera: dict) -> str:
    """Hash the public camera geometry and its explicit compatibility metadata."""
    missing = [field for field in _CAMERA_HASH_FIELDS if field not in camera]
    if missing:
        raise ValueError(f"camera calibration missing required fields: {missing}")
    return _canonical_sha256({field: camera[field] for field in _CAMERA_HASH_FIELDS})


def decorate_camera_calibration(
    camera: dict,
    *,
    coordinate_frame: str = SIMULATION_COORDINATE_FRAME,
    calibration_version: str = SIMULATION_CALIBRATION_VERSION,
    compatibility_id: str = SIMULATION_CALIBRATION_COMPATIBILITY_ID,
) -> dict:
    """Return a camera record with a verifiable public frame/calibration identity."""
    result = dict(camera)
    result.setdefault("camera", "table")
    result["coordinate_frame"] = coordinate_frame
    result["calibration_version"] = calibration_version
    result["calibration_compatibility_id"] = compatibility_id
    result["calibration_sha256"] = camera_calibration_sha256(result)
    return result


def simulation_table_camera_calibration() -> dict:
    """Return the surveyed Isaac camera calibration used by Phase 4B captures."""
    return decorate_camera_calibration(json.loads(json.dumps(SIMULATION_TABLE_CAMERA_CALIBRATION)))


def workspace_configuration_sha256(workspace: dict) -> str:
    payload = {key: value for key, value in workspace.items() if key != "configuration_sha256"}
    return _canonical_sha256(payload)


def simulation_source_workspace(camera: dict) -> dict:
    """Bind the public simulation source polygon to one exact camera calibration."""
    calibration_hash = camera_calibration_sha256(camera)
    if camera.get("calibration_sha256") != calibration_hash:
        raise ValueError("camera calibration_sha256 is missing or inconsistent")
    result = {
        "configuration_version": WORKSPACE_CONFIGURATION_VERSION,
        "deployment_scope": "simulation_only_recalibrate_for_real_robot",
        "coordinate_frame": SIMULATION_COORDINATE_FRAME,
        "polygon_vertices_xy_m": [list(point) for point in SIMULATION_SOURCE_WORKSPACE_POLYGON_M],
        "table_plane": {
            "coordinate_frame": SIMULATION_COORDINATE_FRAME,
            "normal": [0.0, 0.0, 1.0],
            "z_m": float(camera["table_top_z"]),
        },
        "calibration_compatibility_id": SIMULATION_CALIBRATION_COMPATIBILITY_ID,
        "compatible_calibration_version": SIMULATION_CALIBRATION_VERSION,
        "compatible_calibration_sha256": calibration_hash,
        "real_robot_recalibration_required": True,
    }
    result["configuration_sha256"] = workspace_configuration_sha256(result)
    return result


def _validated_polygon(workspace: dict) -> np.ndarray:
    try:
        polygon = np.asarray(workspace["polygon_vertices_xy_m"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("workspace polygon is missing or malformed") from exc
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("workspace polygon must contain at least three XY vertices")
    if not np.all(np.isfinite(polygon)):
        raise ValueError("workspace polygon contains a non-finite coordinate")
    edges = np.roll(polygon, -1, axis=0) - polygon
    if np.any(np.linalg.norm(edges, axis=1) <= 1e-12):
        raise ValueError("workspace polygon contains duplicate adjacent vertices")
    next_edges = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if np.any(cross <= 1e-12):
        raise ValueError("workspace polygon must be strictly convex and counter-clockwise")
    return polygon


def validate_workspace_compatibility(workspace: dict | None, camera: dict) -> np.ndarray:
    """Fail closed unless workspace, frame, table plane, and camera identity agree."""
    if workspace is None:
        raise ValueError("public part-source workspace configuration is required")
    if workspace.get("configuration_version") != WORKSPACE_CONFIGURATION_VERSION:
        raise ValueError("unsupported workspace configuration version")
    expected_workspace_hash = workspace_configuration_sha256(workspace)
    if workspace.get("configuration_sha256") != expected_workspace_hash:
        raise ValueError("workspace configuration SHA-256 is missing or inconsistent")
    camera_hash = camera_calibration_sha256(camera)
    if camera.get("calibration_sha256") != camera_hash:
        raise ValueError("camera calibration SHA-256 is missing or inconsistent")
    frame = workspace.get("coordinate_frame")
    if not frame or frame != camera.get("coordinate_frame"):
        raise ValueError("camera/workspace coordinate frame mismatch")
    if workspace.get("calibration_compatibility_id") != camera.get("calibration_compatibility_id"):
        raise ValueError("camera/workspace compatibility identifier mismatch")
    if workspace.get("compatible_calibration_version") != camera.get("calibration_version"):
        raise ValueError("camera/workspace calibration version mismatch")
    if workspace.get("compatible_calibration_sha256") != camera_hash:
        raise ValueError("workspace is not bound to this camera calibration SHA-256")
    plane = workspace.get("table_plane", {})
    if plane.get("coordinate_frame") != frame or plane.get("normal") != [0.0, 0.0, 1.0]:
        raise ValueError("workspace table plane metadata is invalid")
    if not math.isclose(float(plane.get("z_m", math.nan)), float(camera["table_top_z"]), abs_tol=1e-12):
        raise ValueError("camera/workspace table plane mismatch")
    return _validated_polygon(workspace)


def _edge_signed_distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = points[:, None, :] - polygon[None, :, :]
    cross = edges[None, :, 0] * offsets[:, :, 1] - edges[None, :, 1] * offsets[:, :, 0]
    return cross / np.linalg.norm(edges, axis=1)[None, :]


def source_workspace_containment(
    xy_m,
    workspace: dict,
    *,
    conservative_radius_m: float = CUBE_BOUNDING_RADIUS_M,
    yaw_deg: float | None = None,
    exact_oriented_footprint: bool = False,
) -> dict:
    """Return signed containment for a full cube footprint in a convex polygon.

    The detector deliberately uses the bounding circle.  Exact square corners
    are available only when a caller explicitly justifies reliable yaw.
    """
    polygon = _validated_polygon(workspace)
    center = np.asarray(xy_m, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("candidate workspace position must be a finite XY pair")
    if exact_oriented_footprint:
        if yaw_deg is None or not math.isfinite(float(yaw_deg)):
            raise ValueError("exact oriented footprint requires finite yaw")
        half = CUBE_EDGE_M / 2.0
        corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        angle = math.radians(float(yaw_deg))
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        footprint = corners @ rotation.T + center
        margin = float(np.min(_edge_signed_distances(footprint, polygon)))
        model = "exact_oriented_square"
        radius = None
    else:
        radius = float(conservative_radius_m)
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("conservative footprint radius must be finite and non-negative")
        footprint = None
        margin = float(np.min(_edge_signed_distances(center, polygon)) - radius)
        model = "conservative_bounding_circle"
    return {
        "footprint_model": model,
        "conservative_radius_m": radius,
        "oriented_footprint_xy_m": None if footprint is None else footprint.tolist(),
        "signed_containment_margin_m": margin,
        "signed_containment_margin_mm": margin * 1000.0,
        "passed": margin >= -1e-12,
    }
