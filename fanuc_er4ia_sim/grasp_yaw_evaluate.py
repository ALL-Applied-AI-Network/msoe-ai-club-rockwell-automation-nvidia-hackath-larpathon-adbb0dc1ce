"""Evaluator-only modulo-90 grasp-yaw accuracy reporting."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from grasp_yaw import (
    AUTHORIZATION_UNCERTAINTY_LIMIT_DEG,
    angular_distance_mod90,
    normalize_yaw_mod90,
)


MATCH_LIMIT_M = 0.020


def _distribution(values) -> dict:
    values = list(map(float, values))
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def evaluate_yaw_predictions(predictions: dict, ground_truth: dict) -> dict:
    estimates = predictions.get("yaw_estimates", [])
    cubes = [row for row in ground_truth.get("objects", []) if row.get("category") == "cube"]
    pred_xy = np.asarray([row["input_world_xy_m"] for row in estimates], float).reshape((-1, 2))
    truth_xy = np.asarray([row["xyz"][:2] for row in cubes], float).reshape((-1, 2))
    cost = (
        np.linalg.norm(pred_xy[:, None] - truth_xy[None], axis=2)
        if len(estimates) and len(cubes)
        else np.empty((len(estimates), len(cubes)))
    )
    row_indices, column_indices = (
        linear_sum_assignment(cost) if cost.size else (np.array([], int), np.array([], int))
    )
    matches = []
    for row_index, column_index in zip(row_indices, column_indices):
        if cost[row_index, column_index] > MATCH_LIMIT_M:
            continue
        estimate = estimates[int(row_index)]
        truth = cubes[int(column_index)]
        error = angular_distance_mod90(
            estimate["estimated_yaw_mod_90_deg"], truth["yaw_deg"]
        )
        false_valid = bool(
            estimate["yaw_valid"]
            and error > AUTHORIZATION_UNCERTAINTY_LIMIT_DEG + 1e-12
        )
        matches.append(
            {
                "detection_id": estimate["detection_id"],
                "ground_truth_name": truth["name"],
                "color": estimate["color"],
                "yaw_valid": estimate["yaw_valid"],
                "predicted_yaw_mod_90_deg": estimate[
                    "estimated_yaw_mod_90_deg"
                ],
                "ground_truth_yaw_mod_90_deg": normalize_yaw_mod90(
                    truth["yaw_deg"]
                ),
                "yaw_error_mod_90_deg": error,
                "estimated_uncertainty_deg": estimate[
                    "estimated_uncertainty_deg"
                ],
                "confidence": estimate["confidence"],
                "false_valid_above_5_deg": false_valid,
            }
        )
    valid = [row for row in matches if row["yaw_valid"]]
    valid_errors = [row["yaw_error_mod_90_deg"] for row in valid]
    confidence_bins = []
    for lower, upper in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.000001)):
        rows = [row for row in valid if lower <= row["confidence"] < upper]
        confidence_bins.append(
            {
                "range": [lower, min(1.0, upper)],
                "count": len(rows),
                "yaw_error_deg": _distribution(
                    row["yaw_error_mod_90_deg"] for row in rows
                ),
            }
        )
    by_color = {}
    for color in ("red", "blue", "yellow"):
        rows = [row for row in matches if row["color"] == color]
        by_color[color] = {
            "total": len(rows),
            "valid": sum(row["yaw_valid"] for row in rows),
            "false_valid": sum(row["false_valid_above_5_deg"] for row in rows),
            "valid_error_deg": _distribution(
                row["yaw_error_mod_90_deg"] for row in rows if row["yaw_valid"]
            ),
        }
    return {
        "schema_version": 1,
        "evaluation_source": "evaluator_only_ground_truth",
        "predictions_written_before_evaluation": True,
        "prediction_truth_flags": {
            "ground_truth_used": predictions.get("ground_truth_used"),
            "simulator_state_used": predictions.get("simulator_state_used"),
            "oracle_plan_used": predictions.get("oracle_plan_used"),
            "object_names_used": predictions.get("object_names_used"),
        },
        "matched_cube_count": len(matches),
        "valid_yaw_count": len(valid),
        "invalid_or_ambiguous_count": len(matches) - len(valid),
        "false_valid_count_above_5_deg": sum(
            row["false_valid_above_5_deg"] for row in matches
        ),
        "valid_yaw_error_deg": _distribution(valid_errors),
        "all_six_simultaneously_safe": bool(
            len(matches) == 6
            and len(valid) == 6
            and not any(row["false_valid_above_5_deg"] for row in matches)
        ),
        "per_color": by_color,
        "confidence_error_calibration": confidence_bins,
        "matches": matches,
    }


def aggregate_yaw_evaluations(evaluations: dict[str, dict]) -> dict:
    rows = [row for evaluation in evaluations.values() for row in evaluation["matches"]]
    valid = [row for row in rows if row["yaw_valid"]]
    by_color = defaultdict(lambda: {"total": 0, "valid": 0, "false_valid": 0})
    for row in rows:
        entry = by_color[row["color"]]
        entry["total"] += 1
        entry["valid"] += int(row["yaw_valid"])
        entry["false_valid"] += int(row["false_valid_above_5_deg"])
    safe_scenes = sorted(
        seed for seed, result in evaluations.items() if result["all_six_simultaneously_safe"]
    )
    confidence_bins = []
    for lower, upper in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.000001)):
        selected = [row for row in valid if lower <= row["confidence"] < upper]
        confidence_bins.append({
            "range": [lower, min(1.0, upper)],
            "count": len(selected),
            "false_valid_count_above_5_deg": sum(
                row["false_valid_above_5_deg"] for row in selected
            ),
            "yaw_error_deg": _distribution(
                row["yaw_error_mod_90_deg"] for row in selected
            ),
            "estimated_uncertainty_deg": _distribution(
                row["estimated_uncertainty_deg"] for row in selected
            ),
        })
    return {
        "scene_count": len(evaluations),
        "cube_count": len(rows),
        "valid_yaw_count": len(valid),
        "invalid_or_ambiguous_count": len(rows) - len(valid),
        "coverage": len(valid) / len(rows) if rows else 0.0,
        "false_valid_count_above_5_deg": sum(
            row["false_valid_above_5_deg"] for row in rows
        ),
        "valid_yaw_error_deg": _distribution(
            row["yaw_error_mod_90_deg"] for row in valid
        ),
        "valid_estimated_uncertainty_deg": _distribution(
            row["estimated_uncertainty_deg"] for row in valid
        ),
        "confidence_error_calibration": confidence_bins,
        "per_color": dict(by_color),
        "per_scene_valid_counts": {
            seed: result["valid_yaw_count"] for seed, result in evaluations.items()
        },
        "scenes_with_all_six_simultaneously_safe": safe_scenes,
        "any_scene_all_six_safe": bool(safe_scenes),
        "every_scene_all_six_safe": len(safe_scenes) == len(evaluations),
    }


def run_evaluator(predictions_path, ground_truth_path, output_path) -> dict:
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    ground_truth = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    result = evaluate_yaw_predictions(predictions, ground_truth)
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_evaluator(args.predictions, args.ground_truth, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
