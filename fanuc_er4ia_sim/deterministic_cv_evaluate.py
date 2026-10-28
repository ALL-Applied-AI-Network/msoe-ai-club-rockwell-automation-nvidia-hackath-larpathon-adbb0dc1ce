"""Evaluator-only ground-truth comparison for deterministic_cv predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from cube_pose import square_yaw


MATCH_LIMIT_M = 0.020
TARGET_MEDIAN_XY_MM = 3.0
TARGET_MAX_XY_MM = 5.0


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def _point_segment_distance(point: np.ndarray, centre: np.ndarray, yaw_deg: float, length: float) -> float:
    axis = np.array([np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))])
    start, end = centre - axis * length / 2, centre + axis * length / 2
    delta = end - start
    t = np.clip(np.dot(point - start, delta) / max(1e-12, np.dot(delta, delta)), 0, 1)
    return float(np.linalg.norm(point - (start + t * delta)))


def evaluate_predictions(predictions: dict, ground_truth: dict, settled_state: dict | None = None) -> dict:
    detections = predictions.get("accepted_detections", [])
    cubes = [o for o in ground_truth.get("objects", []) if o.get("category") == "cube"]
    distractors = [o for o in ground_truth.get("objects", []) if o.get("category") in {"cap", "vial"}]
    pred_xy = np.array([d["world_xy_m"] for d in detections], float).reshape((-1, 2))
    cube_xy = np.array([o["xyz"][:2] for o in cubes], float).reshape((-1, 2))
    cost = np.linalg.norm(pred_xy[:, None, :] - cube_xy[None, :, :], axis=2) if len(detections) and len(cubes) else np.empty((len(detections), len(cubes)))
    rows, cols = linear_sum_assignment(cost) if cost.size else (np.array([], int), np.array([], int))
    matched_rows, matched_cols, matches = set(), set(), []
    for row, col in zip(rows, cols):
        if cost[row, col] > MATCH_LIMIT_M:
            continue
        matched_rows.add(int(row)); matched_cols.add(int(col))
        detection, truth = detections[row], cubes[col]
        error_mm = float(cost[row, col] * 1000.0)
        yaw_error = None
        if detection.get("yaw_valid") and detection.get("yaw_mod_90_deg") is not None and truth.get("yaw_deg") is not None:
            yaw_error = abs(square_yaw(detection["yaw_mod_90_deg"] - truth["yaw_deg"]))
        matches.append({
            "detection_id": detection["detection_id"],
            "ground_truth_name": truth["name"],
            "predicted_color": detection["color"],
            "ground_truth_color": truth["colour"],
            "predicted_defect_class": detection["defect_class"],
            "xy_error_mm": error_mm,
            "yaw_error_mod_90_deg": yaw_error,
            "color_correct": detection["color"] == truth["colour"],
            "class_correct": detection["defect_class"] == {
                "red": "Defect Class A", "blue": "Defect Class B", "yellow": "Defect Class C"
            }[truth["colour"]],
        })

    false_positives = [detections[i]["detection_id"] for i in range(len(detections)) if i not in matched_rows]
    missed = [cubes[i]["name"] for i in range(len(cubes)) if i not in matched_cols]
    distractor_results = []
    for obj in distractors:
        centre = np.asarray(obj["xyz"][:2], float)
        accepted = []
        for detection in detections:
            point = np.asarray(detection["world_xy_m"], float)
            if obj["category"] == "cap":
                distance = float(np.linalg.norm(point - centre))
                threshold = max(0.020, float(obj.get("size", 0)) * 0.75)
            else:
                distance = _point_segment_distance(point, centre, obj.get("yaw_deg", 0), obj.get("length", 0.14))
                threshold = max(0.014, float(obj.get("size", 0)) * 0.75)
            if distance <= threshold:
                accepted.append(detection["detection_id"])
        distractor_results.append({
            "ground_truth_name": obj["name"],
            "category": obj["category"],
            "accepted_as_cube": bool(accepted),
            "nearby_detection_ids": accepted,
        })

    errors = [m["xy_error_mm"] for m in matches]
    yaw_errors = [m["yaw_error_mod_90_deg"] for m in matches if m["yaw_error_mod_90_deg"] is not None]
    correct_colors = sum(m["color_correct"] for m in matches)
    correct_classes = sum(m["class_correct"] for m in matches)
    counts = Counter(d["color"] for d in detections)
    median_error = float(np.median(errors)) if errors else None
    max_error = float(np.max(errors)) if errors else None
    targets = {
        "exactly_six_accepted_cubes": len(detections) == 6,
        "exactly_two_each_color": all(counts[c] == 2 for c in ("red", "blue", "yellow")),
        "zero_accepted_caps_or_vials": not any(r["accepted_as_cube"] for r in distractor_results),
        "six_of_six_correct_colors_and_classes": len(matches) == 6 and correct_colors == 6 and correct_classes == 6,
        "median_xy_error_at_most_3_mm": median_error is not None and median_error <= TARGET_MEDIAN_XY_MM,
        "maximum_xy_error_at_most_5_mm": max_error is not None and max_error <= TARGET_MAX_XY_MM,
    }
    return {
        "schema_version": 1,
        "evaluation_source": "offline_ground_truth_evaluator",
        "predictions_inference_source": predictions.get("inference_source"),
        "predictions_written_before_evaluation": True,
        "settled_state_used_for_matching": False,
        "detected_cube_count": len(detections),
        "counts_by_color": {c: counts[c] for c in ("red", "blue", "yellow")},
        "counts_by_class": dict(Counter(d["defect_class"] for d in detections)),
        "false_positive_detection_ids": false_positives,
        "false_positive_count": len(false_positives),
        "missed_cube_names": missed,
        "missed_cube_count": len(missed),
        "matched_cubes": matches,
        "distractor_rejection": distractor_results,
        "accepted_cap_count": sum(r["category"] == "cap" and r["accepted_as_cube"] for r in distractor_results),
        "accepted_vial_count": sum(r["category"] == "vial" and r["accepted_as_cube"] for r in distractor_results),
        "color_accuracy": correct_colors / len(matches) if matches else 0.0,
        "defect_class_accuracy": correct_classes / len(matches) if matches else 0.0,
        "xy_error_mm": _distribution(errors),
        "yaw_error_mod_90_deg": _distribution(yaw_errors),
        "confidence_distributions": {
            key: _distribution([float(d[key]) for d in detections])
            for key in ("color_confidence", "classification_confidence", "pose_confidence", "localization_confidence")
        },
        "pose_fit_iou_distribution": _distribution([float(d["pose_fit_iou"]) for d in detections]),
        "success_targets": targets,
        "benchmark_passed": all(targets.values()),
        "offline_analysis_artifact": True,
    }


def run_evaluator(predictions_path: str | Path, ground_truth_path: str | Path,
                  output_path: str | Path, settled_state_path: str | Path | None = None) -> dict:
    predictions = json.loads(Path(predictions_path).read_text())
    ground_truth = json.loads(Path(ground_truth_path).read_text())
    settled = json.loads(Path(settled_state_path).read_text()) if settled_state_path else None
    result = evaluate_predictions(predictions, ground_truth, settled)
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--settled-state")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_evaluator(args.predictions, args.ground_truth, args.output, args.settled_state)
    print(json.dumps({"benchmark_passed": result["benchmark_passed"],
                      "detected_cube_count": result["detected_cube_count"],
                      "xy_error_mm": result["xy_error_mm"]}, indent=2))


if __name__ == "__main__":
    main()
