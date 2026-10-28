"""Deterministic, offline Dataset 3 cube perception.

Inference is intentionally limited to an RGB image and the captured camera
calibration.  Simulator state, object names, oracle plans, and ground truth are
not accepted by this module.  Evaluation lives in deterministic_cv_evaluate.py.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from backproject import CameraModel
from cube_pose import refine_pose, silhouette, square_yaw
from perception_geometry import CUBE_EDGE_M


INFERENCE_SOURCE = "deterministic_cv"
COLOURS = ("red", "blue", "yellow")
DEFECT_CLASS = {
    "red": "Defect Class A",
    "blue": "Defect Class B",
    "yellow": "Defect Class C",
}

# Candidate generation is deliberately permissive.  Acceptance is controlled
# by calibrated projected geometry and pose-fit quality below.
MIN_COMPONENT_AREA_1920 = 35.0
MAX_COMPONENT_AREA_1920 = 1800.0
MIN_EXPECTED_SIZE_AGREEMENT = 0.58
MIN_FOOTPRINT_AGREEMENT = 0.54
MIN_SOLIDITY = 0.72
MIN_POSE_IOU = 0.72
MIN_COLOR_CONFIDENCE = 0.68
MIN_CLASSIFICATION_CONFIDENCE = 0.72
MIN_POSE_CONFIDENCE = 0.70
DEDUPLICATION_DISTANCE_M = 0.006


def _colour_mask(hsv: np.ndarray, colour: str) -> np.ndarray:
    """Return a candidate mask, including red's OpenCV hue wraparound."""
    h, s, v = cv2.split(hsv)
    if colour == "red":
        hue = (h <= 12) | (h >= 168)
    elif colour == "blue":
        hue = (h >= 94) & (h <= 136)
    elif colour == "yellow":
        hue = (h >= 13) & (h <= 43)
    else:
        raise ValueError(f"unsupported colour: {colour}")
    mask = (hue & (s >= 48) & (v >= 35)).astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _hue_confidence(pixels: np.ndarray, colour: str) -> float:
    if not len(pixels):
        return 0.0
    h = pixels[:, 0].astype(float)
    s = pixels[:, 1].astype(float) / 255.0
    v = pixels[:, 2].astype(float) / 255.0
    if colour == "red":
        distance = np.minimum(h, 180.0 - h)
        hue_score = np.clip(1.0 - distance / 18.0, 0.0, 1.0)
    elif colour == "blue":
        hue_score = np.clip(1.0 - np.abs(h - 115.0) / 28.0, 0.0, 1.0)
    else:
        hue_score = np.clip(1.0 - np.abs(h - 28.0) / 22.0, 0.0, 1.0)
    score = 0.62 * hue_score + 0.25 * s + 0.13 * np.minimum(1.0, v * 1.4)
    return float(np.clip(np.mean(score), 0.0, 1.0))


def _component_contour(component: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else np.empty((0, 1, 2), np.int32)


def _shape_metrics(component: np.ndarray, contour: np.ndarray) -> dict:
    area = float(np.count_nonzero(component))
    if not len(contour):
        return dict(area_px=area, solidity=0.0, extent=0.0, aspect_ratio=0.0,
                    circularity=0.0, footprint_agreement=0.0)
    x, y, w, h = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    aspect = min(rw, rh) / max(rw, rh) if max(rw, rh) else 0.0
    solidity = min(1.0, area / max(1.0, hull_area))
    extent = area / max(1.0, float(w * h))
    circularity = 4.0 * np.pi * area / max(1.0, perimeter * perimeter)
    # Perspective can stretch a projected cube, so this is a soft square gate.
    footprint = float(np.clip(0.55 * aspect + 0.45 * min(1.0, extent / 0.72), 0, 1))
    return dict(
        area_px=area,
        solidity=float(solidity),
        extent=float(extent),
        aspect_ratio=float(aspect),
        circularity=float(circularity),
        footprint_agreement=footprint,
    )


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum()))


def _yaw_evidence(component: np.ndarray, cam: CameraModel, xy: np.ndarray, yaw: float) -> tuple[float, float]:
    """Return best sampled IoU and separation from another plausible yaw."""
    ys, xs = np.nonzero(component)
    if not len(xs):
        return 0.0, 0.0
    pad = 4
    x0, x1 = max(0, int(xs.min()) - pad), min(component.shape[1], int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(component.shape[0], int(ys.max()) + pad + 1)
    observed = component[y0:y1, x0:x1] > 0
    samples = []
    for angle in np.arange(-45.0, 45.0, 2.0):
        predicted = silhouette(cam, xy, angle, observed.shape, origin=(x0, y0)) > 0
        samples.append((angle, _mask_iou(predicted, observed)))
    best = max(samples, key=lambda item: item[1])
    alternatives = [score for angle, score in samples if abs(square_yaw(angle - best[0])) >= 8.0]
    second = max(alternatives, default=0.0)
    return float(best[1]), float(max(0.0, best[1] - second))


def _candidate_record(
    image: np.ndarray,
    hsv: np.ndarray,
    cam: CameraModel,
    colour: str,
    label_image: np.ndarray,
    label: int,
    stats: np.ndarray,
    centroid: np.ndarray,
) -> dict:
    component = (label_image == label).astype(np.uint8)
    contour = _component_contour(component)
    shape = _shape_metrics(component, contour)
    x, y, w, h, area = [int(v) for v in stats[label]]
    u, v = [float(q) for q in centroid[label]]
    reasons = []
    scale2 = (cam.W / 1920.0) ** 2
    if not MIN_COMPONENT_AREA_1920 * scale2 <= area <= MAX_COMPONENT_AREA_1920 * scale2:
        reasons.append("component_area_out_of_candidate_range")

    initial = cam.to_plane(u, v, z=cam.table_z + CUBE_EDGE_M / 2)
    xy = None if initial is None else initial[:2]
    expected_area = 0.0
    coarse_size = 0.0
    if xy is None:
        reasons.append("centroid_ray_does_not_intersect_cube_centre_plane")
    else:
        expected = silhouette(cam, xy, 0.0, image.shape[:2])
        expected_area = float(np.count_nonzero(expected))
        ratio = area / max(1.0, expected_area)
        coarse_size = float(min(ratio, 1.0 / max(ratio, 1e-9)))

    fitted_xy = xy
    yaw = 0.0
    pose_iou = 0.0
    # Avoid expensive fitting for obvious zone markings, caps, or tiny noise.
    if xy is not None and not reasons and coarse_size >= 0.30:
        fitted_xy, yaw, pose_iou = refine_pose(
            image, cam, {"colour": colour}, xy, 0.0, search_half_width=45
        )
        fitted_mask = silhouette(cam, fitted_xy, yaw, image.shape[:2])
        fitted_area = float(np.count_nonzero(fitted_mask))
        ratio = area / max(1.0, fitted_area)
        size_agreement = float(min(ratio, 1.0 / max(ratio, 1e-9)))
    else:
        size_agreement = coarse_size

    pixels = hsv[component > 0]
    color_confidence = _hue_confidence(pixels, colour)
    contour_quality = float(np.clip(0.5 * shape["solidity"] + 0.5 * shape["extent"] / 0.75, 0, 1))
    classification_confidence = float(
        np.clip(0.62 * color_confidence + 0.20 * size_agreement + 0.18 * shape["footprint_agreement"], 0, 1)
    )
    pose_confidence = float(np.clip(0.68 * pose_iou + 0.20 * size_agreement + 0.12 * contour_quality, 0, 1))
    localization_confidence = float(np.clip(0.55 * pose_iou + 0.30 * size_agreement + 0.15 * shape["solidity"], 0, 1))

    if size_agreement < MIN_EXPECTED_SIZE_AGREEMENT:
        reasons.append("projected_size_inconsistent_with_10_mm_cube")
    if shape["footprint_agreement"] < MIN_FOOTPRINT_AGREEMENT:
        reasons.append("footprint_not_square_enough")
    if shape["solidity"] < MIN_SOLIDITY:
        reasons.append("component_not_solid_enough")
    if pose_iou < MIN_POSE_IOU:
        reasons.append("calibrated_cube_pose_fit_iou_too_low")
    if color_confidence < MIN_COLOR_CONFIDENCE:
        reasons.append("colour_confidence_too_low")
    if classification_confidence < MIN_CLASSIFICATION_CONFIDENCE:
        reasons.append("classification_confidence_too_low")
    if pose_confidence < MIN_POSE_CONFIDENCE:
        reasons.append("pose_confidence_too_low")

    yaw_best_iou, yaw_margin = (0.0, 0.0)
    if fitted_xy is not None and pose_iou >= MIN_POSE_IOU:
        yaw_best_iou, yaw_margin = _yaw_evidence(component, cam, fitted_xy, yaw)
    yaw_valid = bool(pose_iou >= 0.76 and yaw_best_iou >= 0.70 and yaw_margin >= 0.015)

    polygon = contour.reshape(-1, 2).tolist() if len(contour) else []
    diagnostic_label = "rejected_colour_component"
    if (
        colour == "yellow"
        and shape["area_px"] >= 250.0 * scale2
        and shape["circularity"] >= 0.75
        and size_agreement < MIN_EXPECTED_SIZE_AGREEMENT
    ):
        diagnostic_label = "yellow_circular_oversize_candidate"
    elif shape["footprint_agreement"] < MIN_FOOTPRINT_AGREEMENT:
        diagnostic_label = "non_cube_geometry"
    return {
        "candidate_id": f"candidate-{colour}-{label:03d}",
        "category_hypothesis": "cube",
        "color": colour,
        "defect_class": DEFECT_CLASS[colour],
        "pixel_centroid": [u, v],
        "bounding_box_xywh": [x, y, w, h],
        "contour_polygon": polygon,
        "contour_point_count": len(polygon),
        "world_xy_m": None if fitted_xy is None else [float(q) for q in fitted_xy],
        "world_z_m": None if fitted_xy is None else float(cam.table_z + CUBE_EDGE_M / 2),
        "table_height_assumption_m": CUBE_EDGE_M / 2,
        "yaw_mod_90_deg": float(square_yaw(yaw)) if yaw_valid else None,
        "yaw_valid": yaw_valid,
        "yaw_evidence_iou": yaw_best_iou,
        "yaw_evidence_margin": yaw_margin,
        "color_confidence": color_confidence,
        "classification_confidence": classification_confidence,
        "pose_confidence": pose_confidence,
        "localization_confidence": localization_confidence,
        "pose_fit_iou": float(pose_iou),
        "expected_projected_area_px": expected_area,
        "expected_size_agreement": size_agreement,
        "contour_quality": contour_quality,
        **shape,
        "accepted": not reasons,
        "acceptance_reasons": [
            "colour_threshold_passed",
            "projected_10_mm_size_passed",
            "square_footprint_passed",
            "calibrated_pose_fit_passed",
        ] if not reasons else [],
        "rejection_reasons": reasons,
        "diagnostic_label": "accepted_cube" if not reasons else diagnostic_label,
        "inference_source": INFERENCE_SOURCE,
    }


def suppress_duplicates(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the highest-quality spatial hypothesis without count forcing."""
    ranked = sorted(
        [c for c in candidates if c["accepted"]],
        key=lambda c: (c["pose_confidence"], c["classification_confidence"]),
        reverse=True,
    )
    kept, rejected = [], []
    for candidate in ranked:
        xy = np.asarray(candidate["world_xy_m"], float)
        duplicate = next(
            (other for other in kept if np.linalg.norm(xy - other["world_xy_m"]) < DEDUPLICATION_DISTANCE_M),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
            continue
        candidate["accepted"] = False
        candidate["acceptance_reasons"] = []
        candidate["rejection_reasons"].append(f"duplicate_of_{duplicate['candidate_id']}")
        rejected.append(candidate)
    return kept, rejected


def finalize_inventory(candidates: list[dict]) -> dict:
    """Apply de-duplication and the non-negotiable six/2-2-2 inventory gate."""
    accepted, duplicate_rejections = suppress_duplicates(candidates)
    rejected = [c for c in candidates if not c["accepted"]]
    for candidate in duplicate_rejections:
        if candidate not in rejected:
            rejected.append(candidate)
    accepted.sort(key=lambda c: (COLOURS.index(c["color"]), c["pixel_centroid"][0]))
    for index, detection in enumerate(accepted, 1):
        detection["detection_id"] = f"det-cv-{index:03d}"
        detection["category"] = "cube"

    counts = Counter(c["color"] for c in accepted)
    inventory_reasons = []
    if len(accepted) != 6:
        inventory_reasons.append(f"accepted_cube_count_{len(accepted)}_expected_6")
    if dict(counts) != {"red": 2, "blue": 2, "yellow": 2}:
        inventory_reasons.append(f"colour_counts_{dict(counts)}_expected_2_each")
    valid = not inventory_reasons
    return {
        "schema_version": 1,
        "inference_source": INFERENCE_SOURCE,
        "inputs_allowed": ["rgb_image", "camera_calibration"],
        "ground_truth_used": False,
        "simulator_state_used": False,
        "oracle_plan_used": False,
        "object_names_used": False,
        "cube_edge_m": CUBE_EDGE_M,
        "inventory_valid": valid,
        "inventory_failure_reasons": inventory_reasons,
        "accepted_cube_count": len(accepted),
        "counts_by_color": {colour: counts.get(colour, 0) for colour in COLOURS},
        "accepted_detections": accepted,
        "executable_cube_inventory": accepted if valid else [],
        "rejected_candidates": rejected,
        "candidate_count": len(candidates),
        "diagnostic_notes": [
            "Clear vials are rejected before candidacy because they do not satisfy any saturated cube-colour mask.",
            "Yellow circular components remain in rejected_candidates with geometry-based reasons.",
        ],
        "thresholds": {
            "minimum_expected_size_agreement": MIN_EXPECTED_SIZE_AGREEMENT,
            "minimum_footprint_agreement": MIN_FOOTPRINT_AGREEMENT,
            "minimum_solidity": MIN_SOLIDITY,
            "minimum_pose_fit_iou": MIN_POSE_IOU,
            "minimum_color_confidence": MIN_COLOR_CONFIDENCE,
            "minimum_classification_confidence": MIN_CLASSIFICATION_CONFIDENCE,
            "minimum_pose_confidence": MIN_POSE_CONFIDENCE,
            "deduplication_distance_m": DEDUPLICATION_DISTANCE_M,
        },
    }


def detect_image(image: np.ndarray, cam: CameraModel) -> dict:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("detector requires a BGR RGB-image array")
    if (image.shape[1], image.shape[0]) != (cam.W, cam.H):
        raise ValueError("image resolution does not match camera calibration")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidates = []
    for colour in COLOURS:
        mask = _colour_mask(hsv, colour)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        for label in range(1, count):
            candidates.append(
                _candidate_record(image, hsv, cam, colour, labels, label, stats, centroids)
            )
    return finalize_inventory(candidates)


def _draw_candidate(image: np.ndarray, candidate: dict, accepted: bool) -> None:
    x, y, w, h = candidate["bounding_box_xywh"]
    colour = (40, 210, 40) if accepted else (30, 80, 240)
    cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
    label = candidate.get("detection_id", candidate["candidate_id"])
    short_reasons = {
        "component_area_out_of_candidate_range": "area",
        "projected_size_inconsistent_with_10_mm_cube": "size",
        "footprint_not_square_enough": "footprint",
        "component_not_solid_enough": "solidity",
        "calibrated_cube_pose_fit_iou_too_low": "pose-IoU",
        "colour_confidence_too_low": "colour-conf",
        "classification_confidence_too_low": "class-conf",
        "pose_confidence_too_low": "pose-conf",
    }
    reason = "PASS" if accepted else short_reasons.get(
        candidate["rejection_reasons"][0], candidate["rejection_reasons"][0]
    )
    cv2.putText(image, f"{label} {candidate['color']} {reason}", (x, max(16, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)


def write_overlays(image: np.ndarray, result: dict, accepted_path: Path, rejected_path: Path) -> None:
    accepted_overlay = image.copy()
    rejected_overlay = image.copy()
    for candidate in result["accepted_detections"]:
        _draw_candidate(accepted_overlay, candidate, True)
    scale2 = (image.shape[1] / 1920.0) ** 2
    displayed_rejections = [
        candidate for candidate in result["rejected_candidates"]
        if MIN_COMPONENT_AREA_1920 * scale2 <= candidate["area_px"] <= MAX_COMPONENT_AREA_1920 * scale2
    ]
    for candidate in displayed_rejections:
        _draw_candidate(rejected_overlay, candidate, False)
    status = "VALID 6/6" if result["inventory_valid"] else "INVENTORY FAILED"
    cv2.putText(accepted_overlay, f"Offline deterministic CV: {status}", (32, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 220, 30) if result["inventory_valid"] else (20, 40, 240), 2)
    cv2.putText(rejected_overlay, "Offline deterministic CV: rejected candidates", (32, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 40, 240), 2)
    cv2.putText(
        rejected_overlay,
        f"Showing {len(displayed_rejections)}/{len(result['rejected_candidates'])}; obvious background/noise retained in JSON",
        (32, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 40, 240), 2,
    )
    if not cv2.imwrite(str(accepted_path), accepted_overlay):
        raise OSError(f"failed to write {accepted_path}")
    if not cv2.imwrite(str(rejected_path), rejected_overlay):
        raise OSError(f"failed to write {rejected_path}")


def run_detector(image_path: str | Path, camera_path: str | Path, output_path: str | Path,
                 overlay_path: str | Path | None = None,
                 rejections_path: str | Path | None = None) -> dict:
    """Run inference.  This is the complete detector I/O boundary."""
    image_path, camera_path, output_path = map(Path, (image_path, camera_path, output_path))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    result = detect_image(image, CameraModel(str(camera_path)))
    result["offline_analysis_artifact"] = True
    result["image_file"] = image_path.name
    result["camera_file"] = camera_path.name
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    if overlay_path and rejections_path:
        write_overlays(image, result, Path(overlay_path), Path(rejections_path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--rejections", required=True)
    args = parser.parse_args()
    result = run_detector(args.image, args.camera, args.output, args.overlay, args.rejections)
    print(json.dumps({"inventory_valid": result["inventory_valid"],
                      "accepted_cube_count": result["accepted_cube_count"],
                      "counts_by_color": result["counts_by_color"]}, indent=2))


if __name__ == "__main__":
    main()
