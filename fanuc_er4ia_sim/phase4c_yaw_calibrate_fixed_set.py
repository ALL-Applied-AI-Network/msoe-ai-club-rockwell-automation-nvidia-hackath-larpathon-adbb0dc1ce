"""Freeze the fixed Phase 4C-Y 4100--4105 calibration result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


CALIBRATION_SEEDS = tuple(range(4100, 4106))
ALPHA = 1.0 / 7.0
AUTHORIZATION_LIMIT_DEG = 5.0
PHYSICAL_LIMIT_DEG = 9.407
DETECTOR_SOURCE_SHA256 = "4649ea76007b86938c97848fecee4ff0ee1080b44f335ae7b76aa8234c2c7216"
YAW_SOURCE_SHA256 = "b3f18409df1fa66ef23549b261a439cfea3b17c32db16a0fdad5e06631475f84"
EXPECTED_EVENTS = [
    "allowed_inputs_hashed",
    "detector_inference_started",
    "detector_predictions_fsynced",
    "yaw_inference_started",
    "yaw_predictions_fsynced",
    "predictions_frozen_before_truth",
    "evaluator_truth_opened",
    "evaluation_fsynced",
    "uncertainty_analysis_fsynced",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def durable_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    directories = {
        4100: root / "phase4c_yaw_seed4100_clean_reevaluation_v1",
        4101: root / "phase4c_yaw_seed4101_clean_reevaluation_v1",
        4102: root / "phase4c_yaw_seed4102_clean_reevaluation_v1",
        4103: root / "phase4c_yaw_seed4103_clean_reevaluation_v1",
        4104: root / "phase4c_yaw_seed4104_clean_reevaluation_v2",
        4105: root / "phase4c_yaw_seed4105_clean_reevaluation_v1",
    }
    if tuple(sorted(directories)) != CALIBRATION_SEEDS:
        raise RuntimeError("fixed calibration seed set changed")

    all_rows: list[dict] = []
    seed_records: list[dict] = []
    input_hashes = {}
    for seed in CALIBRATION_SEEDS:
        directory = directories[seed]
        paths = {
            "detections": directory / "deterministic_cv_detections.json",
            "yaw": directory / "grasp_yaw_predictions.json",
            "evaluation": directory / "grasp_yaw_evaluation.json",
            "ordering": directory / "ordering_provenance_report.json",
        }
        input_hashes[str(seed)] = {
            name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in paths.items()
        }
        detections = json.loads(paths["detections"].read_text(encoding="utf-8"))
        yaw = json.loads(paths["yaw"].read_text(encoding="utf-8"))
        evaluation = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
        ordering = json.loads(paths["ordering"].read_text(encoding="utf-8"))
        events = [row["event"] for row in ordering["sequence"]]
        if events != EXPECTED_EVENTS:
            raise RuntimeError(f"seed {seed}: inference/truth ordering mismatch")
        if not ordering.get("predictions_fsynced_before_truth_opened"):
            raise RuntimeError(f"seed {seed}: predictions not frozen before truth")
        if ordering.get("forbidden_inference_arguments"):
            raise RuntimeError(f"seed {seed}: forbidden inference argument")
        if ordering.get("truth_state_oracle_object_names_supplied_to_inference"):
            raise RuntimeError(f"seed {seed}: forbidden truth/state/oracle/name input")
        source_hashes = ordering["inputs_hashed_before_inference"]
        if source_hashes["frozen_detector_source"]["sha256"] != DETECTOR_SOURCE_SHA256:
            raise RuntimeError(f"seed {seed}: detector source identity mismatch")
        if source_hashes["frozen_yaw_estimator_source"]["sha256"] != YAW_SOURCE_SHA256:
            raise RuntimeError(f"seed {seed}: yaw source identity mismatch")
        if not detections.get("inventory_valid") or detections.get("accepted_cube_count") != 6:
            raise RuntimeError(f"seed {seed}: detector inventory invalid")
        if detections.get("counts_by_color") != {"red": 2, "blue": 2, "yellow": 2}:
            raise RuntimeError(f"seed {seed}: detector colour inventory invalid")

        by_detection = {
            row["detection_id"]: row for row in detections["accepted_detections"]
        }
        by_yaw = {row["detection_id"]: row for row in yaw["yaw_estimates"]}
        rows = []
        for match in evaluation["matches"]:
            detection_id = match["detection_id"]
            detection = by_detection[detection_id]
            estimate = by_yaw[detection_id]
            error = float(match["yaw_error_mod_90_deg"])
            spread = float(estimate["estimated_uncertainty_deg"])
            residual = error - spread
            row = {
                "seed": seed,
                "detection_id": detection_id,
                "color": detection["color"],
                "category": detection["category"],
                "defect_class": detection.get("defect_class"),
                "detector_world_xy_m": detection["world_xy_m"],
                "fitted_world_xy_m": estimate["fitted_world_xy_m"],
                "predicted_yaw_mod_90_deg": estimate["estimated_yaw_mod_90_deg"],
                "image_hypothesis_spread_deg": spread,
                "evaluator_only_true_yaw_mod_90_deg": match["ground_truth_yaw_mod_90_deg"],
                "evaluator_only_error_mod_90_deg": error,
                "signed_residual_error_minus_spread_deg": residual,
                "positive_residual_deg": max(0.0, residual),
                "yaw_valid": bool(estimate["yaw_valid"]),
                "ambiguity_or_invalidity_reasons": estimate["invalidity_reasons"],
                "validity_reasons": estimate["validity_reasons"],
                "competing_orientation_margin": estimate["competing_orientation_margin"],
                "false_valid_above_5_deg": bool(match["false_valid_above_5_deg"]),
            }
            rows.append(row)
            all_rows.append(row)
        if len(rows) != 6:
            raise RuntimeError(f"seed {seed}: expected six evaluated cubes")
        score = max(row["signed_residual_error_minus_spread_deg"] for row in rows)
        seed_records.append(
            {
                "seed": seed,
                "score_seed_max_signed_residual_deg": score,
                "valid_yaw_count": sum(row["yaw_valid"] for row in rows),
                "false_valid_count": sum(row["false_valid_above_5_deg"] for row in rows),
                "all_six_executable_before_calibration": all(row["yaw_valid"] for row in rows),
            }
        )

    ordered_scores = sorted(row["score_seed_max_signed_residual_deg"] for row in seed_records)
    rank = math.ceil((len(CALIBRATION_SEEDS) + 1) * (1 - ALPHA))
    if rank != 6:
        raise RuntimeError(f"unexpected conformal rank {rank}")
    allowance = max(0.0, ordered_scores[rank - 1])
    for row in all_rows:
        row["frozen_allowance_deg"] = allowance
        row["calibrated_authorization_bound_deg"] = (
            row["image_hypothesis_spread_deg"] + allowance
        )
        row["bound_within_5_deg"] = (
            row["calibrated_authorization_bound_deg"] <= AUTHORIZATION_LIMIT_DEG
        )
        row["would_be_authorized_by_frozen_rule"] = bool(
            row["yaw_valid"]
            and row["bound_within_5_deg"]
            and not row["ambiguity_or_invalidity_reasons"]
        )
    for seed_record in seed_records:
        rows = [row for row in all_rows if row["seed"] == seed_record["seed"]]
        seed_record["bounds_over_5_deg"] = sum(not row["bound_within_5_deg"] for row in rows)
        seed_record["scene_authorized_by_frozen_rule"] = all(
            row["would_be_authorized_by_frozen_rule"] for row in rows
        )

    rule = {
        "schema_version": 1,
        "calibration_seed_set": list(CALIBRATION_SEEDS),
        "validation_seed_set_untouched": list(range(4106, 4112)),
        "confidence": {"alpha": ALPHA, "coverage_lower_bound": 6.0 / 7.0},
        "residual_formula": "modulo_90_yaw_error_deg - image_hypothesis_spread_deg",
        "seed_score_formula": "maximum signed residual among the seed's six cubes",
        "finite_sample_rank": rank,
        "allowance_deg": allowance,
        "authorization_bound_formula": "image_hypothesis_spread_deg + frozen_allowance_deg",
        "authorization_limit_deg": AUTHORIZATION_LIMIT_DEG,
        "physical_limit_deg": PHYSICAL_LIMIT_DEG,
        "ambiguity_rule": (
            "yaw estimator must mark the estimate valid and report no competing-orientation "
            "or other invalidity reason; frozen estimator behavior is not retuned"
        ),
        "authorization_requires": [
            "six valid anonymous cube detections",
            "exact 2/2/2 colour inventory",
            "all six yaw predictions executable",
            "all six calibrated authorization bounds at most 5 degrees",
            "no competing-orientation ambiguity",
            "no truth/state/oracle/object-name access during inference",
        ],
        "detector_source_sha256": DETECTOR_SOURCE_SHA256,
        "yaw_estimator_source_sha256": YAW_SOURCE_SHA256,
        "motion_authorized": False,
    }
    rule["frozen_rule_canonical_sha256"] = canonical_sha256(rule)
    errors_all = [row["evaluator_only_error_mod_90_deg"] for row in all_rows]
    valid_rows = [row for row in all_rows if row["yaw_valid"]]
    errors_valid = [row["evaluator_only_error_mod_90_deg"] for row in valid_rows]
    coverage = {
        "cube_count": len(all_rows),
        "valid_yaw_count": len(valid_rows),
        "valid_yaw_fraction": len(valid_rows) / len(all_rows),
        "scenes_with_all_six_executable": sum(
            row["all_six_executable_before_calibration"] for row in seed_records
        ),
        "false_valid_count": sum(row["false_valid_above_5_deg"] for row in all_rows),
        "all_cube_error_deg": {
            "median": percentile(errors_all, 50),
            "p95": percentile(errors_all, 95),
            "maximum": max(errors_all),
        },
        "valid_cube_error_deg": {
            "median": percentile(errors_valid, 50),
            "p95": percentile(errors_valid, 95),
            "maximum": max(errors_valid),
        },
        "bounds_over_5_deg": sum(not row["bound_within_5_deg"] for row in all_rows),
        "scenes_authorized_by_frozen_rule": sum(
            row["scene_authorized_by_frozen_rule"] for row in seed_records
        ),
        "frozen_rule_would_authorize_every_calibration_scene": all(
            row["scene_authorized_by_frozen_rule"] for row in seed_records
        ),
    }
    durable_json(output / "calibration_rule.json", rule)
    durable_json(output / "calibration_cube_rows.json", all_rows)
    report = {
        "schema_version": 1,
        "study": "phase4c-yaw-fixed-calibration-4100-4105-v1",
        "input_artifacts": input_hashes,
        "seed_scores": seed_records,
        "ordered_seed_scores_deg": ordered_scores,
        "allowance_deg": allowance,
        "coverage": coverage,
        "calibration_rule_sha256": sha256(output / "calibration_rule.json"),
        "calibration_cube_rows_sha256": sha256(output / "calibration_cube_rows.json"),
        "validation_seeds_examined": False,
        "motion_authorized": False,
    }
    durable_json(output / "calibration_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
