"""Truth-blind structural validator for the four missing calibration captures."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from phase4b_design import capture_is_authorized
from perception_geometry import validate_workspace_compatibility


EXPECTED_SEEDS = (4101, 4102, 4103, 4105)
EXPECTED_RESOLUTION = (3840, 2160)
BASE_EVENTS = {
    "shell_wrapper_started",
    "python_process_started",
    "before_project_imports",
    "after_project_imports",
    "before_isaac_import",
    "after_isaac_import",
    "before_simulation_app",
    "after_simulation_app",
    "before_simulation_app_close",
    "simulation_app_close_invoked",
    "wrapper_exit_after_python",
}


def read_json(path: Path, failures: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"missing_or_invalid:{path}:{type(exc).__name__}")
        return {}


def progress_rows(path: Path, failures: list[str]) -> list[dict]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        failures.append(f"progress_log_missing:{type(exc).__name__}")
        return rows
    for number, line in enumerate(lines, 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            failures.append(f"progress_invalid_json_line:{number}")
    return rows


def cleanup_completed(cleanup: dict) -> bool:
    actions = {row.get("action"): row for row in cleanup.get("actions", [])}
    return bool(
        actions
        and actions.get("release_camera", {}).get("ran") is True
        and actions.get("clear_world", {}).get("ran") is True
    )


def validate_capture_set(run: Path) -> dict:
    run = Path(run).resolve()
    failures: list[str] = []
    progress = progress_rows(run / "diagnostic_progress.jsonl", failures)
    events = {row.get("event") for row in progress}
    required_events = set(BASE_EVENTS)
    for seed in EXPECTED_SEEDS:
        required_events.update(
            {
                f"before_seed_{seed}_stage",
                f"after_seed_{seed}_stage",
                "before_physics_release",
                "stability_window_started",
                "stability_window_completed",
                "before_camera_creation",
                "after_camera_creation",
                "before_replicator_sync",
                "after_replicator_sync",
                "image_written",
            }
        )
    missing = sorted(required_events - events)
    failures.extend(f"missing_progress_marker:{event}" for event in missing)

    batch = read_json(run / "batch_capture_report.json", failures)
    rows = batch.get("records", []) if isinstance(batch, dict) else []
    records = {int(row["seed"]): row for row in rows if "seed" in row}
    if len(rows) != len(records) or tuple(sorted(records)) != EXPECTED_SEEDS:
        failures.append(
            f"batch_must_contain_exactly_seeds:{list(EXPECTED_SEEDS)}:got:{sorted(records)}"
        )

    scenes = {}
    stage_ids = []
    for seed in EXPECTED_SEEDS:
        scene = run / f"seed_{seed}"
        record = records.get(seed, {})
        if record.get("capture_status") != "captured":
            failures.append(f"seed_{seed}:not_captured")
        if record.get("accepted_attempt_index") != 0:
            failures.append(f"seed_{seed}:attempt_zero_not_accepted")
        if record.get("accepted_derived_seed") != seed:
            failures.append(f"seed_{seed}:derived_seed_mismatch")
        if record.get("robot_commands_issued") is not False:
            failures.append(f"seed_{seed}:robot_commands_issued_not_false")

        image = cv2.imread(str(scene / "world_camera.png"), cv2.IMREAD_COLOR)
        if image is None:
            failures.append(f"seed_{seed}:image_missing_or_undecodable")
            image_row = None
        else:
            image_row = {
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "minimum": int(image.min()),
                "maximum": int(image.max()),
                "peak_to_peak": int(np.ptp(image)),
                "nonzero_count": int(np.count_nonzero(image)),
            }
            if image.shape != (EXPECTED_RESOLUTION[1], EXPECTED_RESOLUTION[0], 3):
                failures.append(f"seed_{seed}:unexpected_image_shape:{list(image.shape)}")
            if np.ptp(image) == 0 or np.count_nonzero(image) == 0:
                failures.append(f"seed_{seed}:image_black_or_constant")

        camera = read_json(scene / "camera.json", failures)
        workspace = read_json(scene / "workspace.json", failures)
        lifecycle = read_json(scene / "camera_render_lifecycle.json", failures)
        stability = read_json(scene / "stability_report.json", failures)
        geometry = read_json(scene / "settled_geometry.json", failures)
        resolution = read_json(scene / "capture_resolution_metadata.json", failures)
        cleanup = read_json(
            scene / "attempts" / "attempt_00" / "cleanup_report.json", failures
        )
        try:
            validate_workspace_compatibility(workspace, camera)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"seed_{seed}:camera_workspace_incompatible:{exc}")
        camera_row = lifecycle.get("camera", {})
        render_row = lifecycle.get("render_product", {})
        if not (
            camera_row.get("valid")
            and camera_row.get("type") == "Camera"
            and camera_row.get("owned_by_current_stage")
        ):
            failures.append(f"seed_{seed}:current_stage_camera_invalid")
        if not (render_row.get("valid") and render_row.get("owned_by_current_stage")):
            failures.append(f"seed_{seed}:current_stage_render_product_invalid")
        if lifecycle.get("timeline_playing") is not True:
            failures.append(f"seed_{seed}:timeline_not_playing")
        lights = lifecycle.get("lights", {})
        if len(lights) != 4 or not all(row.get("valid") for row in lights.values()):
            failures.append(f"seed_{seed}:four_valid_lights_not_recorded")
        attempts = lifecycle.get("capture_attempts", [])
        if not attempts or not any(row.get("accepted") for row in attempts):
            failures.append(f"seed_{seed}:no_valid_synchronized_capture")
        if not capture_is_authorized(stability):
            failures.append(f"seed_{seed}:stability_gate_not_passed")
        if geometry.get("passed") is not True:
            failures.append(f"seed_{seed}:settled_geometry_gate_not_passed")
        if not cleanup_completed(cleanup):
            failures.append(f"seed_{seed}:cleanup_not_completed")
        expected_metadata = {
            "source_design_resolution": [1920, 1080],
            "requested_capture_resolution": [3840, 2160],
            "actual_decoded_output_resolution": [3840, 2160],
        }
        for key, expected in expected_metadata.items():
            if resolution.get(key) != expected:
                failures.append(f"seed_{seed}:resolution_metadata_mismatch:{key}")
        stage_id = lifecycle.get("stage_identifier")
        stage_ids.append(stage_id)
        scenes[str(seed)] = {
            "attempt_index": record.get("accepted_attempt_index"),
            "derived_seed": record.get("accepted_derived_seed"),
            "design_sha256": record.get("scene_generation_parameters", {}).get(
                "design_sha256"
            ),
            "stage_identifier": stage_id,
            "image": image_row,
            "stability_authorized": capture_is_authorized(stability),
            "geometry_passed": geometry.get("passed"),
            "cleanup_completed": cleanup_completed(cleanup),
            "ground_truth_opened_by_validator": False,
        }

    if len(stage_ids) == len(EXPECTED_SEEDS) and (
        None in stage_ids or len(set(stage_ids)) != len(EXPECTED_SEEDS)
    ):
        failures.append("fresh_stages_not_distinct")
    wrapper_rows = [row for row in progress if row.get("event") == "wrapper_exit_after_python"]
    wrapper_exit = wrapper_rows[-1] if wrapper_rows else {}
    process_ok = bool(
        wrapper_exit.get("python_exit_status") == 0
        and wrapper_exit.get("process_terminated_normally") is True
    )
    if wrapper_rows and not process_ok:
        failures.append("python_process_did_not_terminate_normally")
    try:
        shutdown_observed = "Simulation App Shutting Down" in (
            run / "diagnostic_console.log"
        ).read_text(encoding="utf-8", errors="replace")
    except OSError:
        shutdown_observed = False
    if not shutdown_observed:
        failures.append("simulation_app_shutdown_not_observed")

    return {
        "schema_version": 1,
        "study": "phase4c-yaw-calibration-missing-four-4k-capture-v1",
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_resolution": list(EXPECTED_RESOLUTION),
        "one_isaac_process": True,
        "fresh_stage_per_seed": len(set(stage_ids)) == len(EXPECTED_SEEDS),
        "passed": not failures,
        "failures": failures,
        "missing_required_events": missing,
        "scenes": scenes,
        "close_invoked": "simulation_app_close_invoked" in events,
        "close_return_observed": "after_simulation_app_close" in events,
        "shutdown_observed_in_console": shutdown_observed,
        "wrapper_python_exit_status": wrapper_exit.get("python_exit_status"),
        "process_terminated_normally": process_ok,
        "ground_truth_used": False,
        "detector_inference_run": False,
        "yaw_inference_run": False,
        "motion_or_controller_commands_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = validate_capture_set(args.run)
    output = args.run / "phase4c_yaw_calibration_capture_validation.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    progress = args.run / "diagnostic_progress.jsonl"
    with progress.open("a", encoding="utf-8") as stream:
        row = {
            "event": "validation_completed",
            "utc": datetime.now(timezone.utc).isoformat(),
            "passed": result["passed"],
        }
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
