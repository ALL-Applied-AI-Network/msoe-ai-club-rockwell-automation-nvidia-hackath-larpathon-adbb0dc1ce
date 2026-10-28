"""Host-safe design, validation, hashing, and accounting for Phase 4B.

This module contains no Isaac imports.  It deterministically defines all twelve
scenes before capture and freezes the Phase 4A detector/configuration before any
image is processed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import numpy as np

from backproject import CameraModel
import deterministic_cv as detector
import scene_config as C
from perception_geometry import (
    SIMULATION_SOURCE_WORKSPACE_POLYGON_M,
    simulation_source_workspace,
    simulation_table_camera_calibration,
)


PHASE_VERSION = "phase4b_v1"
DESIGN_CANONICALIZATION_VERSION = "phase4b-design-canonical-v2"
CANONICAL_FLOAT_DECIMALS = 12
SEEDS = tuple(range(4100, 4112))
RESOLUTION = (1920, 1080)
COLOURS = ("red", "red", "blue", "blue", "yellow", "yellow")
DEFECT_CLASS = dict(C.DEFECT_CLASS_BY_COLOUR)

USABLE_X_M = (
    min(point[0] for point in SIMULATION_SOURCE_WORKSPACE_POLYGON_M),
    max(point[0] for point in SIMULATION_SOURCE_WORKSPACE_POLYGON_M),
)
USABLE_Y_M = (
    min(point[1] for point in SIMULATION_SOURCE_WORKSPACE_POLYGON_M),
    max(point[1] for point in SIMULATION_SOURCE_WORKSPACE_POLYGON_M),
)
MINIMUM_EDGE_CLEARANCE_M = 0.020
ZONE_CLEARANCE_M = 0.010
CAMERA_MARGIN_PX = 35
REACH_RADIUS_M = 0.525
LIGHTING_SCALE_RANGE = (0.90, 1.10)
CHALLENGE_CAP_CLEARANCE_RANGE_M = (0.020, 0.038)
GENERATION_RESERVE_M = 0.005
# Eight attempts bound worst-case runtime while giving each requested seed seven
# reproducible fallback layouts after its original attempt-zero design.
MAX_GENERATION_ATTEMPTS = 8

DIAGNOSTIC_TRANSFORMS = {
    "brightness_low": {"kind": "brightness", "factor": 0.85},
    "brightness_high": {"kind": "brightness", "factor": 1.15},
    "gamma_low": {"kind": "gamma", "gamma": 0.85},
    "gamma_high": {"kind": "gamma", "gamma": 1.15},
    "gaussian_blur": {"kind": "gaussian_blur", "kernel": 3, "sigma": 0.8},
    "gaussian_noise": {"kind": "gaussian_noise", "sigma": 3.0},
}

DETECTOR_SOURCE_FILES = (
    "deterministic_cv.py",
    "cube_pose.py",
    "backproject.py",
    "perception_geometry.py",
)

PHASE4B_RUNTIME_FILES = (
    "backproject.py",
    "cell.py",
    "cube_pose.py",
    "deterministic_cv.py",
    "deterministic_cv_evaluate.py",
    "fk.py",
    "gripper_geometry.py",
    "ik.py",
    "oracle_motion_smoke.py",
    "perception_geometry.py",
    "phase4b_capture.py",
    "phase4b_design.py",
    "phase4b_process.py",
    "run_phase4b_randomized_perception.sh",
    "run_phase4b_lifecycle_smoke.sh",
    "scene_config.py",
    "requirements.txt",
    "assets/gripper/smc_mhzl2_16d2.json",
    "assets/gripper/photo_finger_detail.json",
    "assets/usd_assets/cube.usda",
    "assets/usd_assets/vial.usda",
    "assets/usd_assets/vial_cap.usda",
)


def simulation_camera_calibration() -> dict:
    return simulation_table_camera_calibration()


def _camera_model() -> CameraModel:
    return CameraModel(simulation_camera_calibration())


def _segment(obj: dict) -> tuple[np.ndarray, np.ndarray, float]:
    centre = np.asarray(obj["xy_m"], float)
    if obj["category"] != "vial":
        return centre, centre, float(obj["footprint_radius_m"])
    axis = np.array([math.cos(math.radians(obj["yaw_deg"])), math.sin(math.radians(obj["yaw_deg"]))])
    half = C.VIAL_LENGTH / 2
    return centre - axis * half, centre + axis * half, C.VIAL_RADIUS


def _point_segment_distance(point, start, end) -> float:
    point, start, end = map(lambda value: np.asarray(value, float), (point, start, end))
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-15:
        return float(np.linalg.norm(point - start))
    amount = float(np.clip(np.dot(point - start, delta) / denominator, 0, 1))
    return float(np.linalg.norm(point - (start + amount * delta)))


def _orient(a, b, c) -> float:
    first = np.asarray(b, float) - np.asarray(a, float)
    second = np.asarray(c, float) - np.asarray(a, float)
    return float(first[0] * second[1] - first[1] * second[0])


def _segments_intersect(a, b, c, d) -> bool:
    if np.linalg.norm(np.asarray(b) - a) <= 1e-12:
        return _point_segment_distance(a, c, d) <= 1e-12
    if np.linalg.norm(np.asarray(d) - c) <= 1e-12:
        return _point_segment_distance(c, a, b) <= 1e-12
    return _orient(a, b, c) * _orient(a, b, d) <= 0 and _orient(c, d, a) * _orient(c, d, b) <= 0


def _segment_distance(a, b, c, d) -> float:
    first_point = np.linalg.norm(np.asarray(b) - a) <= 1e-12
    second_point = np.linalg.norm(np.asarray(d) - c) <= 1e-12
    if first_point and second_point:
        return float(np.linalg.norm(np.asarray(a) - c))
    if first_point:
        return _point_segment_distance(a, c, d)
    if second_point:
        return _point_segment_distance(c, a, b)
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def object_clearance(first: dict, second: dict) -> float:
    a, b, ar = _segment(first)
    c, d, cr = _segment(second)
    return _segment_distance(a, b, c, d) - ar - cr


def capsule_yaw_deg(rotation: np.ndarray) -> float:
    """Return the vial long-axis yaw in the canonical [-90, 90) range.

    Vials are authored with their long axis on local +Z.  A capsule is
    symmetric under a 180-degree rotation, so normalizing modulo 180 keeps the
    initial, pre-release, settled, validation, and reporting conventions equal.
    """
    yaw = math.degrees(math.atan2(float(rotation[1, 2]), float(rotation[0, 2])))
    return (yaw + 90.0) % 180.0 - 90.0


def angular_difference_deg(first: float, second: float, modulo: float = 360.0) -> float:
    return (float(second) - float(first) + modulo / 2.0) % modulo - modulo / 2.0


def _point_rectangle_distance(point, lo, hi) -> float:
    point = np.asarray(point, float)
    return float(np.linalg.norm(np.maximum(np.maximum(np.asarray(lo) - point, point - np.asarray(hi)), 0)))


def _segment_rectangle_distance(start, end, lo, hi) -> float:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if np.linalg.norm(np.asarray(end) - start) <= 1e-12:
        return _point_rectangle_distance(start, lo, hi)
    if np.all(start >= lo) and np.all(start <= hi) or np.all(end >= lo) and np.all(end <= hi):
        return 0.0
    corners = [lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    if any(_segments_intersect(start, end, a, b) for a, b in edges):
        return 0.0
    return min(
        _point_rectangle_distance(start, lo, hi),
        _point_rectangle_distance(end, lo, hi),
        *(_point_segment_distance(corner, start, end) for corner in corners),
    )


def _outside_zones(obj: dict) -> bool:
    start, end, radius = _segment(obj)
    half = np.asarray(C.DEFECT_ZONE_SIZE) / 2 + ZONE_CLEARANCE_M
    return all(
        _segment_rectangle_distance(start, end, np.asarray(center) - half, np.asarray(center) + half) > radius
        for center in C.DEFECT_ZONE_CENTERS.values()
    )


def _within_workspace(obj: dict) -> bool:
    start, end, radius = _segment(obj)
    lo = np.array([USABLE_X_M[0], USABLE_Y_M[0]]) + radius
    hi = np.array([USABLE_X_M[1], USABLE_Y_M[1]]) - radius
    if not (np.all(start >= lo) and np.all(start <= hi) and np.all(end >= lo) and np.all(end <= hi)):
        return False
    return max(np.linalg.norm(start), np.linalg.norm(end)) + radius <= REACH_RADIUS_M


def workspace_margins(obj: dict) -> dict:
    start, end, radius = _segment(obj)
    lo = np.array([USABLE_X_M[0], USABLE_Y_M[0]]) + radius
    hi = np.array([USABLE_X_M[1], USABLE_Y_M[1]]) - radius
    boundary = min(
        *(float(value) for value in start - lo),
        *(float(value) for value in hi - start),
        *(float(value) for value in end - lo),
        *(float(value) for value in hi - end),
    )
    reach = REACH_RADIUS_M - (max(float(np.linalg.norm(start)), float(np.linalg.norm(end))) + radius)
    return {
        "boundary_margin_m": boundary,
        "reach_margin_m": reach,
        "minimum_margin_m": min(boundary, reach),
    }


def destination_zone_margins(obj: dict) -> dict:
    start, end, radius = _segment(obj)
    half = np.asarray(C.DEFECT_ZONE_SIZE) / 2 + ZONE_CLEARANCE_M
    margins = {
        colour: _segment_rectangle_distance(
            start,
            end,
            np.asarray(center) - half,
            np.asarray(center) + half,
        ) - radius
        for colour, center in sorted(C.DEFECT_ZONE_CENTERS.items())
    }
    return {"by_zone_m": margins, "minimum_margin_m": min(margins.values())}


def _camera_contained(obj: dict) -> bool:
    cam = _camera_model()
    start, end, radius = _segment(obj)
    height = {"cube": C.CUBE_EDGE, "cap": C.CAP_HEIGHT, "vial": C.VIAL_RADIUS * 2}[obj["category"]]
    points = []
    for centre in (start, end):
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            points.append(centre + radius * np.array([np.cos(angle), np.sin(angle)]))
    for point in points:
        uv = cam.project([point[0], point[1], cam.table_z + height])
        if uv is None or not (
            CAMERA_MARGIN_PX <= uv[0] <= cam.W - CAMERA_MARGIN_PX
            and CAMERA_MARGIN_PX <= uv[1] <= cam.H - CAMERA_MARGIN_PX
        ):
            return False
    return True


def _valid_against(candidate: dict, placed: list[dict], reserve_m: float = 0.0) -> bool:
    workspace = workspace_margins(candidate)
    zones = destination_zone_margins(candidate)
    return (
        workspace["minimum_margin_m"] >= reserve_m
        and zones["minimum_margin_m"] >= reserve_m
        and _camera_contained(candidate)
        and all(
            object_clearance(candidate, other) >= MINIMUM_EDGE_CLEARANCE_M + reserve_m
            for other in placed
        )
    )


def _make_object(name: str, category: str, xy, yaw: float, colour=None) -> dict:
    radius = {"cube": C.CUBE_EDGE / math.sqrt(2), "cap": C.CAP_RADIUS, "vial": C.VIAL_RADIUS}[category]
    result = {
        "name": name,
        "category": category,
        "xy_m": [float(xy[0]), float(xy[1])],
        "yaw_deg": float(yaw),
        "footprint_radius_m": float(radius),
    }
    if colour:
        result.update(colour=colour, defect_class=DEFECT_CLASS[colour])
    return result


def _random_candidate(rng: random.Random, name: str, category: str, colour=None) -> dict:
    xy = (rng.uniform(*USABLE_X_M), rng.uniform(*USABLE_Y_M))
    yaw = rng.uniform(-45, 45) if category == "cube" else rng.uniform(-180, 180)
    return _make_object(name, category, xy, yaw, colour)


def derive_attempt_seed(top_level_seed: int, attempt_index: int) -> int:
    if top_level_seed not in SEEDS:
        raise ValueError(f"Phase 4B top-level seed must be one of {SEEDS[0]}..{SEEDS[-1]}")
    if not 0 <= attempt_index < MAX_GENERATION_ATTEMPTS:
        raise ValueError(f"attempt_index must be in 0..{MAX_GENERATION_ATTEMPTS - 1}")
    if attempt_index == 0:
        return top_level_seed
    material = f"{PHASE_VERSION}|{top_level_seed}|{attempt_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def generate_scene(
    top_level_seed: int,
    attempt_index: int = 0,
    generation_reserve_m: float = GENERATION_RESERVE_M,
) -> dict:
    derived_seed = derive_attempt_seed(top_level_seed, attempt_index)
    rng = random.Random(derived_seed)
    placed = []
    for index, colour in enumerate(COLOURS):
        for _ in range(20000):
            candidate = _random_candidate(rng, f"SortCube{index}", "cube", colour)
            if _valid_against(candidate, placed, generation_reserve_m):
                placed.append(candidate)
                break
        else:
            raise RuntimeError(f"seed {top_level_seed} attempt {attempt_index}: cannot place cube {index}")

    # One cap is deliberately near a yellow cube, but still preserves the same
    # documented edge clearance and all other constraints.
    yellow = placed[4 + rng.randrange(2)]
    for _ in range(20000):
        angle = rng.uniform(-math.pi, math.pi)
        clearance = rng.uniform(*CHALLENGE_CAP_CLEARANCE_RANGE_M)
        distance = yellow["footprint_radius_m"] + C.CAP_RADIUS + clearance
        xy = np.asarray(yellow["xy_m"]) + distance * np.array([math.cos(angle), math.sin(angle)])
        candidate = _make_object("SortCap0", "cap", xy, rng.uniform(-180, 180))
        if _valid_against(candidate, placed, generation_reserve_m):
            placed.append(candidate)
            challenge = {
                "yellow_cube": yellow["name"],
                "cap": candidate["name"],
                # Store the sampled source value.  Actual geometry is validated
                # independently below instead of being inferred from this field.
                "sampled_edge_clearance_m": float(clearance),
            }
            break
    else:
        raise RuntimeError(f"seed {top_level_seed} attempt {attempt_index}: cannot place challenge cap")

    for index in (1,):
        for _ in range(20000):
            candidate = _random_candidate(rng, f"SortCap{index}", "cap")
            if _valid_against(candidate, placed, generation_reserve_m):
                placed.append(candidate)
                break
        else:
            raise RuntimeError(f"seed {top_level_seed} attempt {attempt_index}: cannot place cap {index}")

    for index in range(2):
        for _ in range(30000):
            candidate = _random_candidate(rng, f"SortVial{index}", "vial")
            if _valid_against(candidate, placed, generation_reserve_m):
                placed.append(candidate)
                break
        else:
            raise RuntimeError(f"seed {top_level_seed} attempt {attempt_index}: cannot place vial {index}")

    lighting = {
        name: rng.uniform(*LIGHTING_SCALE_RANGE)
        for name in ("DomeLight", "KeyLight", "FillLight", "RimLight")
    }
    scene = {
        "seed": top_level_seed,
        "top_level_seed": top_level_seed,
        "attempt_index": attempt_index,
        "derived_seed": derived_seed,
        "design_canonicalization": DESIGN_CANONICALIZATION_VERSION,
        "objects": placed,
        "lighting_intensity_scales": lighting,
        "yellow_cap_challenge": challenge,
        "constraints": constraint_specification(),
    }
    failures = validate_scene_design(scene)
    if failures:
        raise RuntimeError(f"seed {top_level_seed} attempt {attempt_index}: generated invalid scene: {failures}")
    scene["design_sha256"] = scene_design_sha256(scene)
    return scene


def _canonicalize(value):
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("canonical Phase 4B designs cannot contain NaN or infinity")
        quantum = Decimal(1).scaleb(-CANONICAL_FLOAT_DECIMALS)
        quantized = Decimal(str(number)).quantize(quantum, rounding=ROUND_HALF_EVEN)
        scaled = int(quantized.scaleb(CANONICAL_FLOAT_DECIMALS))
        return {"$phase4b_float12": 0 if scaled == 0 else scaled}
    if isinstance(value, np.integer):
        return int(value)
    return value


def canonical_design_bytes(scene: dict) -> bytes:
    canonical = {key: value for key, value in scene.items() if key != "design_sha256"}
    payload = {
        "canonicalization": DESIGN_CANONICALIZATION_VERSION,
        "float_decimal_places": CANONICAL_FLOAT_DECIMALS,
        "scene": _canonicalize(canonical),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def scene_design_sha256(scene: dict) -> str:
    return hashlib.sha256(canonical_design_bytes(scene)).hexdigest()


def constraint_specification() -> dict:
    return {
        "usable_x_m": list(USABLE_X_M),
        "usable_y_m": list(USABLE_Y_M),
        "minimum_edge_clearance_m": MINIMUM_EDGE_CLEARANCE_M,
        "destination_zone_clearance_m": ZONE_CLEARANCE_M,
        "camera_margin_px": CAMERA_MARGIN_PX,
        "reach_radius_m": REACH_RADIUS_M,
        "lighting_scale_range": list(LIGHTING_SCALE_RANGE),
        "challenge_cap_clearance_range_m": list(CHALLENGE_CAP_CLEARANCE_RANGE_M),
        "generation_reserve_m": GENERATION_RESERVE_M,
        "maximum_generation_attempts": MAX_GENERATION_ATTEMPTS,
        "resolution": list(RESOLUTION),
    }


def validate_scene_design(scene: dict) -> list[str]:
    objects = scene.get("objects", [])
    failures = []
    categories = Counter(obj.get("category") for obj in objects)
    colours = Counter(obj.get("colour") for obj in objects if obj.get("category") == "cube")
    if categories != {"cube": 6, "cap": 2, "vial": 2}:
        failures.append(f"object_counts:{dict(categories)}")
    if colours != {"red": 2, "blue": 2, "yellow": 2}:
        failures.append(f"colour_counts:{dict(colours)}")
    if len({obj.get("name") for obj in objects}) != 10:
        failures.append("object_names_not_unique")
    for obj in objects:
        if not _within_workspace(obj): failures.append(f"workspace:{obj['name']}")
        if not _outside_zones(obj): failures.append(f"destination_zone:{obj['name']}")
        if not _camera_contained(obj): failures.append(f"camera:{obj['name']}")
    for index, first in enumerate(objects):
        for second in objects[index + 1:]:
            if object_clearance(first, second) < MINIMUM_EDGE_CLEARANCE_M - 1e-9:
                failures.append(f"clearance:{first['name']}:{second['name']}")
    scales = scene.get("lighting_intensity_scales", {})
    if set(scales) != {"DomeLight", "KeyLight", "FillLight", "RimLight"} or any(
        not LIGHTING_SCALE_RANGE[0] <= value <= LIGHTING_SCALE_RANGE[1] for value in scales.values()
    ):
        failures.append("lighting_range")
    return failures


def _quaternion_angle_deg(first, second) -> float | None:
    if first is None or second is None:
        return None
    a = np.asarray(first, float)
    b = np.asarray(second, float)
    if a.shape != (4,) or b.shape != (4,):
        return None
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-15:
        return None
    dot = float(np.clip(abs(np.dot(a, b)) / denominator, -1.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def geometry_report(
    scene: dict,
    *,
    root_poses: dict | None = None,
    release_scene: dict | None = None,
    release_root_poses: dict | None = None,
) -> dict:
    """Return fail-closed geometry evidence using the shared capsule model."""
    objects = scene.get("objects", [])
    release_objects = {
        obj["name"]: obj for obj in (release_scene or {}).get("objects", [])
    }
    root_poses = root_poses or {}
    release_root_poses = release_root_poses or {}
    rows = []
    for obj in objects:
        start, end, radius = _segment(obj)
        workspace = workspace_margins(obj)
        zones = destination_zone_margins(obj)
        prior = release_objects.get(obj["name"])
        root = root_poses.get(obj["name"])
        prior_root = release_root_poses.get(obj["name"])
        motion = None
        if prior is not None:
            motion = {
                "geometric_center_translation_m": float(
                    np.linalg.norm(np.asarray(obj["xy_m"], float) - np.asarray(prior["xy_m"], float))
                ),
                "yaw_rotation_deg": abs(
                    angular_difference_deg(
                        prior["yaw_deg"],
                        obj["yaw_deg"],
                        180.0 if obj["category"] == "vial" else 360.0,
                    )
                ),
                "root_translation_m": None,
                "root_rotation_deg": None,
            }
            if root is not None and prior_root is not None:
                motion["root_translation_m"] = float(
                    np.linalg.norm(
                        np.asarray(root["position"], float)
                        - np.asarray(prior_root["position"], float)
                    )
                )
                motion["root_rotation_deg"] = _quaternion_angle_deg(
                    prior_root.get("orientation"), root.get("orientation")
                )
        rows.append(
            {
                "name": obj["name"],
                "category": obj["category"],
                "root_pose": root,
                "geometric_center_xy_m": [float(value) for value in obj["xy_m"]],
                "long_axis_yaw_deg": float(obj["yaw_deg"]),
                "long_axis_endpoints_xy_m": [start.tolist(), end.tolist()],
                "footprint": {
                    "kind": "capsule" if obj["category"] == "vial" else "disc_bound",
                    "radius_m": float(radius),
                    "length_m": float(C.VIAL_LENGTH) if obj["category"] == "vial" else 0.0,
                },
                "workspace": workspace,
                "destination_zones": zones,
                "release_to_settle": motion,
            }
        )
    pairs = []
    for index, first in enumerate(objects):
        for second in objects[index + 1:]:
            clearance = object_clearance(first, second)
            pairs.append(
                {
                    "first": first["name"],
                    "second": second["name"],
                    "actual_edge_clearance_m": clearance,
                    "required_edge_clearance_m": MINIMUM_EDGE_CLEARANCE_M,
                    "signed_clearance_margin_m": clearance - MINIMUM_EDGE_CLEARANCE_M,
                    "passed": clearance >= MINIMUM_EDGE_CLEARANCE_M - 1e-9,
                }
            )
    failures = validate_scene_design(scene)
    return {
        "geometry_convention": {
            "vial_xy": "geometric_center",
            "vial_long_axis": "rotated_local_positive_z",
            "vial_yaw_canonical_range_deg": "[-90,90)",
            "vial_radius_m": C.VIAL_RADIUS,
            "vial_length_m": C.VIAL_LENGTH,
        },
        "objects": rows,
        "pairwise_clearances": pairs,
        "required_edge_clearance_m": MINIMUM_EDGE_CLEARANCE_M,
        "failure_reasons": failures,
        "passed": not failures,
    }


def generation_attempt_record(
    top_level_seed: int,
    attempt_index: int,
    scene: dict,
    initial_geometry: dict | None,
    settled_geometry: dict | None,
    rejection_reasons: list[str],
) -> dict:
    return {
        "top_level_seed": int(top_level_seed),
        "attempt_index": int(attempt_index),
        "derived_seed": int(scene["derived_seed"]),
        "design_sha256": scene["design_sha256"],
        "canonicalization": DESIGN_CANONICALIZATION_VERSION,
        "initial_constraints": initial_geometry,
        "settled_constraints": settled_geometry,
        "rejection_reasons": list(rejection_reasons),
        "accepted": not rejection_reasons,
        "detector_consulted": False,
    }


def generate_batch_design(seeds=SEEDS) -> dict:
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or any(seed not in SEEDS for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("batch seeds must be a non-empty unique subset of Phase 4B top-level seeds")
    return {
        "phase_version": PHASE_VERSION,
        "design_canonicalization": DESIGN_CANONICALIZATION_VERSION,
        "canonical_float_decimal_places": CANONICAL_FLOAT_DECIMALS,
        "seeds": list(seeds),
        "scene_count": len(seeds),
        "scenes": [generate_scene(seed, 0) for seed in seeds],
        "constraints": constraint_specification(),
        "diagnostic_transforms": DIAGNOSTIC_TRANSFORMS,
    }


def detector_configuration() -> dict:
    return {
        key: getattr(detector, key)
        for key in (
            "MIN_COMPONENT_AREA_1920", "MAX_COMPONENT_AREA_1920",
            "MIN_EXPECTED_SIZE_AGREEMENT", "MIN_FOOTPRINT_AGREEMENT",
            "MIN_SOLIDITY", "MIN_POSE_IOU", "MIN_COLOR_CONFIDENCE",
            "MIN_CLASSIFICATION_CONFIDENCE", "MIN_POSE_CONFIDENCE",
            "DEDUPLICATION_DISTANCE_M",
        )
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_detector_record(root: str | Path) -> dict:
    root = Path(root)
    files = {name: sha256_file(root / name) for name in DETECTOR_SOURCE_FILES}
    source_sha = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    config = detector_configuration()
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    workspace = simulation_source_workspace(simulation_camera_calibration())
    identity = {
        "source_sha256": source_sha,
        "appearance_configuration_sha256": config_sha,
        "workspace_configuration_sha256": workspace["configuration_sha256"],
    }
    combined = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "source_files": files,
        "source_sha256": source_sha,
        "configuration": config,
        # Preserve this key for readers of the Phase 4B v1 record.  Its value
        # remains the appearance-only configuration identity.
        "configuration_sha256": config_sha,
        "appearance_configuration_sha256": config_sha,
        "workspace_configuration": workspace,
        "workspace_configuration_sha256": workspace["configuration_sha256"],
        "combined_sha256": combined,
    }


def runtime_manifest(root: str | Path) -> dict:
    root = Path(root)
    rows = []
    for relative in PHASE4B_RUNTIME_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "phase_version": PHASE_VERSION,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "external_read_only_asset": "/job/in/assets/isaac-6.0/Assets/Isaac/6.0/Isaac/Robots/Fanuc/er4ia/er4ia.usd",
        "container_image": "nvcr.io/nvidia/isaac-sim:6.0.0",
    }


def record_failed_seed(seed: int, error: str) -> dict:
    return {"seed": seed, "capture_status": "generation_failed", "stability_outcome": False, "error": str(error), "robot_commands_issued": False}


def capture_is_authorized(stability_report: dict) -> bool:
    return bool(
        stability_report.get("scene_stable_before_commands") is True
        and stability_report.get("consecutive_stable_window_count", 0) >= 2
        and not stability_report.get("timed_out", False)
    )


def aggregate_evaluations(scene_records: list[dict], expected_seeds=SEEDS) -> dict:
    expected = list(expected_seeds)
    seen = [row["seed"] for row in scene_records]
    if sorted(seen) != sorted(expected) or len(seen) != len(set(seen)):
        raise ValueError(f"scene accounting must contain every seed exactly once: expected={expected}, seen={seen}")
    generated = [row for row in scene_records if row.get("capture_status") == "captured"]
    stabilized = [row for row in generated if row.get("stability_outcome") is True]
    evaluations = [row.get("evaluation") for row in stabilized if row.get("evaluation")]
    matches = [match for evaluation in evaluations for match in evaluation.get("matched_cubes", [])]
    detected = sum(evaluation.get("detected_cube_count", 0) for evaluation in evaluations)
    false_positive = sum(evaluation.get("false_positive_count", 0) for evaluation in evaluations)
    missed = sum(evaluation.get("missed_cube_count", 0) for evaluation in evaluations)
    errors = [match["xy_error_mm"] for match in matches]
    correct_colors = sum(match.get("color_correct") is True for match in matches)
    correct_classes = sum(match.get("class_correct") is True for match in matches)
    distractors = [item for evaluation in evaluations for item in evaluation.get("distractor_rejection", [])]
    accepted_distractors = sum(item.get("accepted_as_cube") is True for item in distractors)
    inventory_passes = sum(evaluation.get("benchmark_passed") is True for evaluation in evaluations)
    failures = {
        str(row["seed"]): row.get("error") or row.get("evaluation", {}).get("success_targets")
        for row in scene_records
        if row.get("capture_status") != "captured" or not row.get("evaluation", {}).get("benchmark_passed", False)
    }
    scene_count = len(expected)
    expected_cubes = scene_count * 6
    expected_distractors = scene_count * 4
    criteria = {
        "all_requested_scenes_stabilized": len(stabilized) == scene_count,
        "all_requested_complete_inventories": inventory_passes == scene_count,
        "every_scene_exactly_2_2_2": all(e.get("counts_by_color") == {"red": 2, "blue": 2, "yellow": 2} for e in evaluations) and len(evaluations) == scene_count,
        "zero_caps_or_vials_accepted": accepted_distractors == 0 and len(distractors) == expected_distractors,
        "all_cubes_correct_colors_classes": len(matches) == expected_cubes and correct_colors == expected_cubes and correct_classes == expected_cubes,
        "aggregate_median_xy_error_at_most_3_mm": bool(errors) and float(np.median(errors)) <= 3.0,
        "maximum_xy_error_at_most_5_mm": bool(errors) and float(np.max(errors)) <= 5.0,
    }
    return {
        "expected_seeds": expected,
        "accounted_seeds": seen,
        "scenes_generated": len(generated),
        "scenes_stabilized": len(stabilized),
        "scenes_passing_complete_inventory": inventory_passes,
        "total_accepted_cubes": detected,
        "total_cube_precision": (len(matches) / detected) if detected else 0.0,
        "total_cube_recall": len(matches) / expected_cubes,
        "color_accuracy": correct_colors / len(matches) if matches else 0.0,
        "defect_class_accuracy": correct_classes / len(matches) if matches else 0.0,
        "cap_vial_false_positive_rate": accepted_distractors / len(distractors) if distractors else 0.0,
        "xy_error_mm": {
            "count": len(errors),
            "median": float(np.median(errors)) if errors else None,
            "p95": float(np.percentile(errors, 95)) if errors else None,
            "maximum": float(np.max(errors)) if errors else None,
        },
        "failure_categories_by_seed": failures,
        "go_no_go_criteria": criteria,
        "go": all(criteria.values()),
        "false_positive_count": false_positive,
        "missed_cube_count": missed,
    }


def resource_estimate() -> dict:
    return {
        "simulation_app_initializations": 1,
        "capture_scene_count": 12,
        "expected_capture_and_settle_minutes": {"minimum": 10, "planning": 18},
        "expected_detection_evaluation_minutes": {"authoritative": 3, "diagnostic_transforms": 14},
        "expected_total_runtime_minutes": 35,
        "estimated_output_disk_gb": 0.35,
        "estimated_transfer_gb": 0.35,
        "recommended_yaad": {"memory_gb": 48, "cpus": 8, "gpu": True, "max_minutes": 45, "nodes": 1},
    }


def proposed_yaad_request() -> dict:
    return {
        "name": "fanuc-er4ia-phase4b-randomized-deterministic-perception",
        "command": "set -euo pipefail\ncd /job/in/project\nexport ISAAC=/isaac-sim\nexport FANUC_ROBOT_USD=/job/in/assets/isaac-6.0/Assets/Isaac/6.0/Isaac/Robots/Fanuc/er4ia/er4ia.usd\nbash run_phase4b_randomized_perception.sh /job/out/phase4b_v1",
        "memory_gb": 48,
        "cpus": 8,
        "gpu": True,
        "max_minutes": 45,
        "input_path": "shared/fanuc-stack",
        "image": "nvcr.io/nvidia/isaac-sim:6.0.0",
        "nodes": 1,
        "env": {"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"},
    }


def smoke_resource_estimate() -> dict:
    return {
        "simulation_app_initializations": 1,
        "top_level_seeds": [4100, 4104],
        "maximum_attempts_per_seed": MAX_GENERATION_ATTEMPTS,
        "expected_runtime_minutes": {"nominal": 8, "planning": 15},
        "estimated_output_disk_gb": 0.08,
        "estimated_transfer_gb": 0.08,
        "recommended_yaad": {
            "memory_gb": 48,
            "cpus": 8,
            "gpu": True,
            "max_minutes": 30,
            "nodes": 1,
        },
    }


def proposed_smoke_yaad_request() -> dict:
    return {
        "name": "fanuc-er4ia-phase4b-two-seed-workspace-gate-smoke",
        "command": (
            "set -euo pipefail\n"
            "cd /job/in/project\n"
            "export ISAAC=/isaac-sim\n"
            "export FANUC_ROBOT_USD=/job/in/assets/isaac-6.0/Assets/Isaac/6.0/Isaac/Robots/Fanuc/er4ia/er4ia.usd\n"
            "RUN=/job/out/phase4b_workspace_gate_smoke_v2\n"
            "bash run_phase4b_lifecycle_smoke.sh \"$RUN\"\n"
            "\"$ISAAC/python.sh\" phase4b_process.py validate-output --run \"$RUN\"\n"
            "\"$ISAAC/python.sh\" phase4b_process.py validate-output --run \"$RUN\" --require-smoke-go"
        ),
        "memory_gb": 48,
        "cpus": 8,
        "gpu": True,
        "max_minutes": 30,
        "input_path": "shared/fanuc-stack",
        "image": "nvcr.io/nvidia/isaac-sim:6.0.0",
        "nodes": 1,
        "env": {"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"},
    }


def write_preparation_bundle(output: str | Path) -> dict:
    root = Path(__file__).resolve().parent
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    values = {
        "batch_design_preview.json": generate_batch_design(),
        "frozen_detector.json": frozen_detector_record(root),
        "runtime_dependency_manifest.json": runtime_manifest(root),
        "resource_estimate.json": resource_estimate(),
        "proposed_yaad_request.json": proposed_yaad_request(),
        "proposed_output_validation.json": {
            "structural_command": "$ISAAC/python.sh phase4b_process.py validate-output --run /job/out/phase4b_v1",
            "strict_go_command": "$ISAAC/python.sh phase4b_process.py validate-output --run /job/out/phase4b_v1 --require-go",
        },
    }
    for name, value in values.items():
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return {name: sha256_file(output / name) for name in values}


def write_smoke_preparation_bundle(output: str | Path) -> dict:
    root = Path(__file__).resolve().parent
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    values = {
        "smoke_design_preview.json": generate_batch_design((4100, 4104)),
        "canonical_hash_fixture_4100_4111.json": {
            "canonicalization": DESIGN_CANONICALIZATION_VERSION,
            "float_decimal_places": CANONICAL_FLOAT_DECIMALS,
            "hashes": {
                str(seed): generate_scene(seed)["design_sha256"] for seed in SEEDS
            },
        },
        "frozen_detector.json": frozen_detector_record(root),
        "runtime_dependency_manifest.json": runtime_manifest(root),
        "smoke_resource_estimate.json": smoke_resource_estimate(),
        "proposed_smoke_yaad_request.json": proposed_smoke_yaad_request(),
        "proposed_smoke_output_validation.json": {
            "structural_command": (
                "$ISAAC/python.sh phase4b_process.py validate-output "
                "--run /job/out/phase4b_workspace_gate_smoke_v2"
            ),
            "strict_smoke_go_command": (
                "$ISAAC/python.sh phase4b_process.py validate-output "
                "--run /job/out/phase4b_workspace_gate_smoke_v2 --require-smoke-go"
            ),
        },
    }
    for name, value in values.items():
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return {name: sha256_file(output / name) for name in values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-dir")
    group.add_argument("--prepare-smoke-dir")
    args = parser.parse_args()
    if args.prepare_smoke_dir:
        result = write_smoke_preparation_bundle(args.prepare_smoke_dir)
    else:
        result = write_preparation_bundle(args.prepare_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
