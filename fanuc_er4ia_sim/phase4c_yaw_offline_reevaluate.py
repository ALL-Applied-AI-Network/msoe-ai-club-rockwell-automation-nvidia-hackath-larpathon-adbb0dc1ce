"""Run truth-blind detector/yaw inference before opening evaluator truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def durable_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def fsync_existing(path: Path) -> None:
    with path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def append_event(path: Path, sequence: list[dict], event: str, **fields) -> None:
    row = {"sequence": len(sequence) + 1, "event": event, "utc": utc_now(), **fields}
    sequence.append(row)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_checked(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"subprocess_failed:{completed.returncode}:{Path(command[1]).name}"
        )


def uncertainty_analysis(predictions: dict, evaluation: dict, seed: int = 4100) -> dict:
    estimates = {row["detection_id"]: row for row in predictions["yaw_estimates"]}
    rows = []
    for match in evaluation["matches"]:
        spread = float(estimates[match["detection_id"]]["estimated_uncertainty_deg"])
        error = float(match["yaw_error_mod_90_deg"])
        rows.append(
            {
                "detection_id": match["detection_id"],
                "image_hypothesis_spread_deg": spread,
                "empirical_evaluator_error_deg": error,
                "positive_residual_error_minus_spread_deg": max(0.0, error - spread),
                "spread_upper_bounded_this_error": spread + 1e-12 >= error,
            }
        )
    maximum_residual = max(
        (row["positive_residual_error_minus_spread_deg"] for row in rows), default=0.0
    )
    provisional_allowance = math.ceil(maximum_residual * 4.0) / 4.0
    return {
        "schema_version": 1,
        "authorization_limit_deg_unchanged": 5.0,
        "reported_estimated_uncertainty_semantics": (
            "image-hypothesis support spread from silhouette-IoU evidence; "
            "not a proven conservative error bound"
        ),
        f"seed_{seed}_rows": rows,
        "spread_undercovered_count": sum(
            not row["spread_upper_bounded_this_error"] for row in rows
        ),
        "maximum_positive_residual_deg": maximum_residual,
        f"provisional_seed{seed}_only_allowance_deg": provisional_allowance,
        "provisional_rule_not_authorized_for_motion": True,
        "proposed_future_rule": {
            "formula": (
                "authorization_uncertainty_bound_deg = "
                "image_hypothesis_spread_deg + frozen_4k_calibration_allowance_deg"
            ),
            "calibration_set_fixed_seeds": [4100, 4101, 4102, 4103, 4104, 4105],
            "disjoint_validation_set_fixed_seeds": [4106, 4107, 4108, 4109, 4110, 4111],
            "allowance_method": (
                "one-sided finite-sample conformal upper quantile of positive "
                "evaluator error minus image-hypothesis spread"
            ),
            "authorization_requires_bound_at_most_deg": 5.0,
            "existing_appearance_and_ambiguity_gates_retained": True,
        },
        "calibration_status": (
            f"insufficient: seed {seed} alone cannot calibrate or validate a "
            "conservative 4K authorization allowance"
        ),
        "motion_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--detector-script", type=Path, required=True)
    parser.add_argument("--yaw-script", type=Path, required=True)
    parser.add_argument("--evaluator-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--compare-detections", type=Path)
    parser.add_argument("--compare-yaw", type=Path)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sequence_path = output / "ordering_events.jsonl"
    sequence: list[dict] = []
    inputs = {
        name: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in (
            ("rgb_image", args.image),
            ("camera_calibration", args.camera),
            ("public_workspace", args.workspace),
            ("frozen_detector_source", args.detector_script),
            ("frozen_yaw_estimator_source", args.yaw_script),
        )
    }
    append_event(sequence_path, sequence, "allowed_inputs_hashed")

    detections = output / "deterministic_cv_detections.json"
    detector_command = [
        sys.executable,
        str(args.detector_script.resolve()),
        "--image",
        str(args.image.resolve()),
        "--camera",
        str(args.camera.resolve()),
        "--workspace",
        str(args.workspace.resolve()),
        "--output",
        str(detections),
        "--overlay",
        str(output / "deterministic_cv_overlay.png"),
        "--rejections",
        str(output / "deterministic_cv_rejections.png"),
    ]
    append_event(sequence_path, sequence, "detector_inference_started")
    run_checked(detector_command, args.detector_script.resolve().parent)
    fsync_existing(detections)
    detection_hash = sha256(detections)
    append_event(
        sequence_path,
        sequence,
        "detector_predictions_fsynced",
        sha256=detection_hash,
    )

    yaw_predictions = output / "grasp_yaw_predictions.json"
    yaw_command = [
        sys.executable,
        str(args.yaw_script.resolve()),
        "--image",
        str(args.image.resolve()),
        "--camera",
        str(args.camera.resolve()),
        "--workspace",
        str(args.workspace.resolve()),
        "--detections",
        str(detections),
        "--output",
        str(yaw_predictions),
    ]
    append_event(sequence_path, sequence, "yaw_inference_started")
    run_checked(yaw_command, args.yaw_script.resolve().parent)
    fsync_existing(yaw_predictions)
    yaw_hash = sha256(yaw_predictions)
    append_event(
        sequence_path,
        sequence,
        "yaw_predictions_fsynced",
        sha256=yaw_hash,
    )

    comparisons = {}
    for name, generated, prior in (
        ("detector_predictions", detections, args.compare_detections),
        ("yaw_predictions", yaw_predictions, args.compare_yaw),
    ):
        if prior:
            comparisons[name] = {
                "prior_sha256": sha256(prior),
                "generated_sha256": sha256(generated),
                "byte_identical": prior.read_bytes() == generated.read_bytes(),
            }
    append_event(sequence_path, sequence, "predictions_frozen_before_truth")

    # This is deliberately the first access to evaluator truth in this process.
    truth_hash = sha256(args.ground_truth)
    append_event(
        sequence_path, sequence, "evaluator_truth_opened", sha256=truth_hash
    )
    evaluation_path = output / "grasp_yaw_evaluation.json"
    evaluator_command = [
        sys.executable,
        str(args.evaluator_script.resolve()),
        "--predictions",
        str(yaw_predictions),
        "--ground-truth",
        str(args.ground_truth.resolve()),
        "--output",
        str(evaluation_path),
    ]
    run_checked(evaluator_command, args.evaluator_script.resolve().parent)
    fsync_existing(evaluation_path)
    append_event(
        sequence_path,
        sequence,
        "evaluation_fsynced",
        sha256=sha256(evaluation_path),
    )

    predictions = json.loads(yaw_predictions.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    uncertainty_path = output / "yaw_uncertainty_analysis.json"
    durable_json(
        uncertainty_path, uncertainty_analysis(predictions, evaluation, args.seed)
    )
    append_event(sequence_path, sequence, "uncertainty_analysis_fsynced")

    forbidden_terms = ("ground-truth", "ground_truth", "scene_state", "oracle", "object_name")
    inference_arguments = detector_command[2:] + yaw_command[2:]
    forbidden_inference_arguments = [
        value
        for value in inference_arguments
        if any(term in value.lower() for term in forbidden_terms)
    ]
    report = {
        "schema_version": 1,
        "study": f"phase4c-yaw-seed{args.seed}-clean-offline-reevaluation-v1",
        "seed": args.seed,
        "inputs_hashed_before_inference": inputs,
        "detector_prediction_sha256": detection_hash,
        "yaw_prediction_sha256": yaw_hash,
        "ground_truth_sha256_after_predictions_frozen": truth_hash,
        "comparisons_with_prior_offline_outputs": comparisons,
        "inference_commands": {
            "detector": detector_command,
            "yaw": yaw_command,
        },
        "forbidden_inference_arguments": forbidden_inference_arguments,
        "truth_state_oracle_object_names_supplied_to_inference": False,
        "predictions_fsynced_before_truth_opened": True,
        "sequence": sequence,
        "motion_authorized": False,
    }
    durable_json(output / "ordering_provenance_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
