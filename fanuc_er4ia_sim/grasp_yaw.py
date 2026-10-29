"""Fail-closed RGB-only grasp-yaw estimation for accepted cube detections.

This module is deliberately separate from the frozen Phase 4B detector.  Its
only inputs are an RGB image, public camera/workspace calibration, and the
detector's already accepted cube records.  Yaw is always modulo 90 degrees.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from backproject import CameraModel
from cube_pose import silhouette, square_yaw
from deterministic_cv import (
    MIN_CLASSIFICATION_CONFIDENCE,
    MIN_COLOR_CONFIDENCE,
    MIN_EXPECTED_SIZE_AGREEMENT,
    MIN_FOOTPRINT_AGREEMENT,
    MIN_POSE_CONFIDENCE,
    MIN_POSE_IOU,
)
from perception_geometry import (
    camera_calibration_sha256,
    validate_workspace_compatibility,
    workspace_configuration_sha256,
)


EVIDENCE_SOURCE = "rgb_projected_cube_silhouette_profile_v1"
YAW_GRID_STEP_DEG = 0.25
AUTHORIZATION_UNCERTAINTY_LIMIT_DEG = 5.0
MIN_PHOTOMETRIC_CONTRAST = 0.08
CENTRE_OFFSETS_PX = (-0.5, 0.0, 0.5)
IOU_EVIDENCE_PIXELS = 3.0


def normalize_yaw_mod90(degrees: float) -> float:
    """Normalize a physically equivalent square yaw to [-45, 45)."""
    if not math.isfinite(float(degrees)):
        raise ValueError("yaw must be finite")
    return square_yaw(float(degrees))


def angular_distance_mod90(first: float, second: float) -> float:
    return abs(normalize_yaw_mod90(float(first) - float(second)))


def scaled_camera_workspace(
    camera: dict, workspace: dict, resolution: tuple[int, int]
) -> tuple[dict, dict]:
    """Scale only raster sampling while preserving the physical camera model."""
    validate_workspace_compatibility(workspace, camera)
    width, height = map(int, resolution)
    if width <= 0 or height <= 0:
        raise ValueError("resolution must be positive")
    old_width, old_height = map(int, camera["resolution"])
    if width * old_height != height * old_width:
        raise ValueError("resolution scaling must preserve aspect ratio")
    scaled_camera = copy.deepcopy(camera)
    scaled_camera["resolution"] = [width, height]
    suffix = f"{width}x{height}"
    scaled_camera["calibration_version"] = (
        f"{camera['calibration_version']}-raster-{suffix}"
    )
    scaled_camera["calibration_compatibility_id"] = (
        f"{camera['calibration_compatibility_id']}-raster-{suffix}"
    )
    scaled_camera["calibration_sha256"] = camera_calibration_sha256(scaled_camera)
    scaled_workspace = copy.deepcopy(workspace)
    scaled_workspace["compatible_calibration_version"] = scaled_camera[
        "calibration_version"
    ]
    scaled_workspace["calibration_compatibility_id"] = scaled_camera[
        "calibration_compatibility_id"
    ]
    scaled_workspace["compatible_calibration_sha256"] = scaled_camera[
        "calibration_sha256"
    ]
    scaled_workspace["configuration_sha256"] = workspace_configuration_sha256(
        scaled_workspace
    )
    validate_workspace_compatibility(scaled_workspace, scaled_camera)
    return scaled_camera, scaled_workspace


def _mask_iou(first: np.ndarray, second: np.ndarray) -> tuple[float, int]:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / max(1, union), union


def _observed_component(image: np.ndarray, detection: dict):
    polygon = np.asarray(detection.get("contour_polygon", []), np.int32)
    if polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3:
        raise ValueError("accepted detection is missing its contour polygon")
    pad = 7
    lower = np.maximum(polygon.min(axis=0) - pad, 0)
    upper = np.minimum(polygon.max(axis=0) + pad + 1, image.shape[1::-1])
    x0, y0 = map(int, lower)
    x1, y1 = map(int, upper)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("accepted detection contour is outside the image")
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    local = polygon - [x0, y0]
    cv2.fillPoly(mask, [local.reshape((-1, 1, 2))], 1)
    crop = image[y0:y1, x0:x1]
    ring = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - mask
    inside_pixels = crop[mask > 0].astype(float)
    ring_pixels = crop[ring > 0].astype(float)
    if not len(inside_pixels) or not len(ring_pixels):
        contrast = 0.0
    else:
        contrast = float(
            np.linalg.norm(inside_pixels.mean(axis=0) - ring_pixels.mean(axis=0))
            / (255.0 * math.sqrt(3.0))
        )
    return mask.astype(bool), (x0, y0), contrast


def _candidate_centres(cam: CameraModel, xy) -> list[np.ndarray]:
    xy = np.asarray(xy, float)
    uv = cam.project([*xy, cam.table_z + 0.005])
    centres = []
    for dv in CENTRE_OFFSETS_PX:
        for du in CENTRE_OFFSETS_PX:
            point = cam.to_plane(
                uv[0] + du, uv[1] + dv, z=cam.table_z + 0.005
            )
            centres.append(np.asarray(point[:2], float))
    return centres


def estimate_detection_yaw(
    image: np.ndarray,
    cam: CameraModel,
    workspace: dict,
    detection: dict,
) -> dict:
    """Estimate one cube yaw and an image-supported modulo-90 uncertainty."""
    validate_workspace_compatibility(workspace, cam.j)
    if image is None or image.shape != (cam.H, cam.W, 3):
        raise ValueError("yaw estimator image does not match camera calibration")
    if detection.get("accepted") is not True or detection.get("category") != "cube":
        raise ValueError("yaw estimator accepts only validated cube detections")
    observed, origin, contrast = _observed_component(image, detection)
    centres = _candidate_centres(cam, detection["world_xy_m"])
    angles = np.arange(-45.0, 45.0, YAW_GRID_STEP_DEG)
    scores = np.zeros(len(angles), float)
    unions = np.ones(len(angles), int)
    winning_centres = []
    for index, angle in enumerate(angles):
        best_score, best_union, best_centre = -1.0, 1, centres[0]
        for centre in centres:
            predicted = silhouette(
                cam, centre, angle, observed.shape, origin=origin
            ).astype(bool)
            score, union = _mask_iou(predicted, observed)
            if score > best_score + 1e-15:
                best_score, best_union, best_centre = score, union, centre
        scores[index] = best_score
        unions[index] = best_union
        winning_centres.append(best_centre)
    best_index = int(np.argmax(scores))
    best_yaw = float(angles[best_index])
    best_score = float(scores[best_index])
    best_union = int(unions[best_index])
    score_tolerance = float(IOU_EVIDENCE_PIXELS / max(1, best_union))
    support = scores >= best_score - score_tolerance - 1e-12
    support_distances = np.array(
        [angular_distance_mod90(angle, best_yaw) for angle in angles]
    )
    uncertainty = float(
        support_distances[support].max() + YAW_GRID_STEP_DEG / 2.0
    )
    competing = scores[
        support_distances > AUTHORIZATION_UNCERTAINTY_LIMIT_DEG + 1e-12
    ]
    competing_score = float(competing.max()) if len(competing) else 0.0
    competing_margin = float(best_score - competing_score)
    reasons = []
    if contrast < MIN_PHOTOMETRIC_CONTRAST:
        reasons.append("insufficient_photometric_contrast")
    if best_score < 0.70:
        reasons.append("projected_silhouette_evidence_too_weak")
    if uncertainty > AUTHORIZATION_UNCERTAINTY_LIMIT_DEG + 1e-12:
        reasons.append("yaw_uncertainty_exceeds_5_deg")
    if competing_margin <= score_tolerance + 1e-12:
        reasons.append("competing_orientation_not_separated")
    valid = not reasons
    separation = np.clip(competing_margin / max(score_tolerance, 1e-12), 0.0, 2.0) / 2.0
    confidence = float(
        np.clip(
            best_score
            * separation
            * max(0.0, 1.0 - uncertainty / 45.0)
            * min(1.0, contrast / max(MIN_PHOTOMETRIC_CONTRAST, 1e-12)),
            0.0,
            1.0,
        )
    )
    return {
        "detection_id": detection["detection_id"],
        "color": detection["color"],
        "input_world_xy_m": list(map(float, detection["world_xy_m"])),
        "estimated_yaw_mod_90_deg": normalize_yaw_mod90(best_yaw),
        "executable_yaw_mod_90_deg": normalize_yaw_mod90(best_yaw) if valid else None,
        "yaw_valid": valid,
        "confidence": confidence,
        "competing_orientation_margin": competing_margin,
        "competing_orientation_score": competing_score,
        "score_tolerance_from_image_evidence": score_tolerance,
        "estimated_uncertainty_deg": uncertainty,
        "authorization_uncertainty_limit_deg": AUTHORIZATION_UNCERTAINTY_LIMIT_DEG,
        "best_silhouette_iou": best_score,
        "best_union_pixel_count": best_union,
        "photometric_contrast": contrast,
        "evidence_source": EVIDENCE_SOURCE,
        "validity_reasons": ["supported_yaw_within_5_deg"] if valid else [],
        "invalidity_reasons": reasons,
        "fitted_world_xy_m": winning_centres[best_index].tolist(),
        "grid_step_deg": YAW_GRID_STEP_DEG,
    }


def validate_executable_inventory(detections: dict) -> list[dict]:
    """Recheck the frozen detector inventory before yaw estimation."""
    rows = detections.get("accepted_detections", [])
    failures = []
    if detections.get("inventory_valid") is not True or len(rows) != 6:
        failures.append("detector_inventory_not_exactly_six")
    ids = [row.get("detection_id") for row in rows]
    if None in ids or len(set(ids)) != len(ids):
        failures.append("detection_ids_not_unique")
    if Counter(row.get("color") for row in rows) != Counter(
        {"red": 2, "blue": 2, "yellow": 2}
    ):
        failures.append("inventory_not_two_two_two")
    for row in rows:
        containment = row.get("workspace_containment", {})
        checks = (
            row.get("accepted") is True,
            containment.get("passed") is True,
            float(row.get("color_confidence", 0)) >= MIN_COLOR_CONFIDENCE,
            float(row.get("classification_confidence", 0))
            >= MIN_CLASSIFICATION_CONFIDENCE,
            float(row.get("pose_confidence", 0)) >= MIN_POSE_CONFIDENCE,
            float(row.get("pose_fit_iou", 0)) >= MIN_POSE_IOU,
            float(row.get("expected_size_agreement", 0))
            >= MIN_EXPECTED_SIZE_AGREEMENT,
            float(row.get("footprint_agreement", 0)) >= MIN_FOOTPRINT_AGREEMENT,
        )
        if not all(checks):
            failures.append(f"detection_not_executable:{row.get('detection_id')}")
    if failures:
        raise ValueError(";".join(failures))
    return rows


def estimate_inventory_yaws(
    image: np.ndarray, cam: CameraModel, workspace: dict, detections: dict
) -> dict:
    rows = validate_executable_inventory(detections)
    estimates = [
        estimate_detection_yaw(image, cam, workspace, detection)
        for detection in rows
    ]
    valid_count = sum(row["yaw_valid"] for row in estimates)
    return {
        "schema_version": 1,
        "inference_source": "deterministic_grasp_yaw",
        "evidence_source": EVIDENCE_SOURCE,
        "inputs_allowed": [
            "rgb_image",
            "camera_calibration",
            "public_workspace_configuration",
            "accepted_deterministic_cv_detections",
        ],
        "ground_truth_used": False,
        "simulator_state_used": False,
        "oracle_plan_used": False,
        "object_names_used": False,
        "authorization_uncertainty_limit_deg": AUTHORIZATION_UNCERTAINTY_LIMIT_DEG,
        "detection_count": len(estimates),
        "valid_yaw_count": valid_count,
        "invalid_yaw_count": len(estimates) - valid_count,
        "all_six_yaws_authorized": len(estimates) == 6 and valid_count == 6,
        "yaw_estimates": estimates,
    }


def run_yaw_estimator(
    image_path: str | Path,
    camera_path: str | Path,
    workspace_path: str | Path,
    detections_path: str | Path,
    output_path: str | Path,
) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    cam = CameraModel(str(camera_path))
    workspace = json.loads(Path(workspace_path).read_text(encoding="utf-8"))
    detections = json.loads(Path(detections_path).read_text(encoding="utf-8"))
    result = estimate_inventory_yaws(image, cam, workspace, detections)
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_yaw_estimator(
        args.image, args.camera, args.workspace, args.detections, args.output
    )
    print(
        json.dumps(
            {
                "valid_yaw_count": result["valid_yaw_count"],
                "all_six_yaws_authorized": result["all_six_yaws_authorized"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
