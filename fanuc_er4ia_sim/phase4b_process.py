"""Process, aggregate, and validate a captured Phase 4B batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from deterministic_cv import run_detector
from deterministic_cv_evaluate import run_evaluator
from phase4b_design import (
    DIAGNOSTIC_TRANSFORMS,
    PHASE_VERSION,
    SEEDS,
    aggregate_evaluations,
    frozen_detector_record,
    sha256_file,
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _assert_frozen(root: Path, expected: dict) -> None:
    current = frozen_detector_record(root)
    if current != expected:
        raise RuntimeError(
            "Frozen Phase 4A detector/configuration changed during Phase 4B: "
            f"expected={expected['combined_sha256']} current={current['combined_sha256']}"
        )


def _transform(image: np.ndarray, specification: dict, seed: int, variant_index: int) -> np.ndarray:
    kind = specification["kind"]
    if kind == "brightness":
        return np.clip(image.astype(np.float32) * specification["factor"], 0, 255).astype(np.uint8)
    if kind == "gamma":
        normalized = image.astype(np.float32) / 255.0
        return np.clip(np.power(normalized, specification["gamma"]) * 255, 0, 255).astype(np.uint8)
    if kind == "gaussian_blur":
        kernel = int(specification["kernel"])
        return cv2.GaussianBlur(image, (kernel, kernel), specification["sigma"])
    if kind == "gaussian_noise":
        rng = np.random.default_rng(seed * 100 + variant_index)
        noise = rng.normal(0, specification["sigma"], image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown diagnostic transform {kind}")


def _record_for_aggregate(capture: dict, evaluation=None, error=None) -> dict:
    row = {
        "seed": capture["seed"],
        "capture_status": capture.get("capture_status"),
        "stability_outcome": capture.get("stability_outcome", False),
    }
    if evaluation is not None:
        row["evaluation"] = evaluation
    if error:
        row["error"] = error
    elif capture.get("error"):
        row["error"] = capture["error"]
    return row


def process_batch(run: Path) -> dict:
    root = Path(__file__).resolve().parent
    frozen = _read(run / "frozen_detector.json")
    _assert_frozen(root, frozen)
    design = _read(run / "batch_design.json")
    expected_seeds = tuple(design["seeds"])
    capture_rows = _read(run / "batch_capture_report.json")["records"]
    by_seed = {row["seed"]: row for row in capture_rows}
    if sorted(by_seed) != sorted(expected_seeds) or len(capture_rows) != len(expected_seeds):
        raise RuntimeError("capture report does not account for every seed exactly once")

    # Pass 1: every authoritative image is processed before any evaluator truth
    # is opened.  This is the inference/evaluation separation boundary.
    detection_status = {}
    for seed in expected_seeds:
        capture = by_seed[seed]
        scene_dir = run / f"seed_{seed}"
        if capture.get("capture_status") != "captured" or not capture.get("stability_outcome"):
            detection_status[seed] = {"status": "skipped_capture_failure", "error": capture.get("error")}
            continue
        try:
            _assert_frozen(root, frozen)
            result = run_detector(
                scene_dir / "world_camera.png",
                scene_dir / "camera.json",
                scene_dir / "workspace.json",
                scene_dir / "deterministic_cv_detections.json",
                scene_dir / "deterministic_cv_overlay.png",
                scene_dir / "deterministic_cv_rejections.png",
            )
            if any(result.get(key) is not False for key in ("ground_truth_used", "simulator_state_used", "oracle_plan_used")):
                raise RuntimeError("detector provenance flags violate inference isolation")
            if result.get("workspace_configuration_sha256") != frozen.get("workspace_configuration_sha256"):
                raise RuntimeError("runtime workspace configuration differs from frozen identity")
            detection_status[seed] = {"status": "processed", "inventory_valid": result["inventory_valid"]}
        except Exception as exc:
            detection_status[seed] = {"status": "detection_failed", "error": f"{type(exc).__name__}: {exc}"}
    _write(run / "authoritative_detection_pass.json", {str(k): v for k, v in detection_status.items()})

    # Pass 2: truth is opened only after the complete inference pass.
    authoritative_records = []
    for seed in expected_seeds:
        capture = by_seed[seed]
        scene_dir = run / f"seed_{seed}"
        status = detection_status[seed]
        if status["status"] != "processed":
            authoritative_records.append(_record_for_aggregate(capture, error=status.get("error")))
            continue
        try:
            evaluation = run_evaluator(
                scene_dir / "deterministic_cv_detections.json",
                scene_dir / "ground_truth.json",
                scene_dir / "deterministic_cv_evaluation.json",
                scene_dir / "settled_scene_state.json",
            )
            authoritative_records.append(_record_for_aggregate(capture, evaluation=evaluation))
        except Exception as exc:
            authoritative_records.append(_record_for_aggregate(capture, error=f"{type(exc).__name__}: {exc}"))
    authoritative = aggregate_evaluations(authoritative_records, expected_seeds)
    _write(run / "aggregate_evaluation.json", authoritative)

    # Diagnostic images and detector outputs are physically separated from the
    # authoritative scene directory. All diagnostic inference completes before
    # any transformed-image evaluator is invoked.
    diagnostic_status = {name: {} for name in DIAGNOSTIC_TRANSFORMS}
    for variant_index, (variant, specification) in enumerate(DIAGNOSTIC_TRANSFORMS.items()):
        for seed in expected_seeds:
            capture = by_seed[seed]
            if capture.get("capture_status") != "captured" or not capture.get("stability_outcome"):
                diagnostic_status[variant][seed] = {"status": "skipped_capture_failure"}
                continue
            scene_dir = run / f"seed_{seed}"
            diagnostic_dir = scene_dir / "diagnostics" / variant
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            try:
                image = cv2.imread(str(scene_dir / "world_camera.png"), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(scene_dir / "world_camera.png")
                transformed = _transform(image, specification, seed, variant_index)
                if not cv2.imwrite(str(diagnostic_dir / "world_camera.png"), transformed):
                    raise OSError("diagnostic image write failed")
                _assert_frozen(root, frozen)
                run_detector(
                    diagnostic_dir / "world_camera.png",
                    scene_dir / "camera.json",
                    scene_dir / "workspace.json",
                    diagnostic_dir / "deterministic_cv_detections.json",
                )
                diagnostic_status[variant][seed] = {"status": "processed"}
            except Exception as exc:
                diagnostic_status[variant][seed] = {"status": "detection_failed", "error": f"{type(exc).__name__}: {exc}"}

    diagnostic_summary = {}
    for variant, statuses in diagnostic_status.items():
        records = []
        for seed in expected_seeds:
            capture = by_seed[seed]
            status = statuses[seed]
            if status["status"] != "processed":
                records.append(_record_for_aggregate(capture, error=status.get("error")))
                continue
            scene_dir = run / f"seed_{seed}"
            diagnostic_dir = scene_dir / "diagnostics" / variant
            try:
                evaluation = run_evaluator(
                    diagnostic_dir / "deterministic_cv_detections.json",
                    scene_dir / "ground_truth.json",
                    diagnostic_dir / "deterministic_cv_evaluation.json",
                    scene_dir / "settled_scene_state.json",
                )
                records.append(_record_for_aggregate(capture, evaluation=evaluation))
            except Exception as exc:
                records.append(_record_for_aggregate(capture, error=f"{type(exc).__name__}: {exc}"))
        diagnostic_summary[variant] = {
            "transformation": DIAGNOSTIC_TRANSFORMS[variant],
            "aggregate": aggregate_evaluations(records, expected_seeds),
        }
    _write(run / "photometric_diagnostics.json", diagnostic_summary)
    _write(run / "diagnostic_detection_pass.json", {
        variant: {str(seed): value for seed, value in statuses.items()}
        for variant, statuses in diagnostic_status.items()
    })

    final_frozen = frozen_detector_record(root)
    if final_frozen != frozen:
        raise RuntimeError("detector/configuration hash changed during batch")
    artifacts = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.name != "batch_manifest.json":
            artifacts.append({
                "path": path.relative_to(run).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    manifest = {
        "phase_version": PHASE_VERSION,
        "seeds": list(expected_seeds),
        "scene_generation": design,
        "capture_records": capture_rows,
        "frozen_detector": frozen,
        "authoritative_aggregate": authoritative,
        "photometric_diagnostics": diagnostic_summary,
        "artifact_count_excluding_manifest": len(artifacts),
        "artifact_total_bytes_excluding_manifest": sum(row["size_bytes"] for row in artifacts),
        "artifacts": artifacts,
    }
    _write(run / "batch_manifest.json", manifest)
    return manifest


def validate_output(run: Path, require_go=False, require_smoke_go=False) -> dict:
    manifest = _read(run / "batch_manifest.json")
    failures = []
    if manifest.get("phase_version") != PHASE_VERSION:
        failures.append("phase_version")
    seeds = manifest.get("seeds", [])
    if not seeds or len(seeds) != len(set(seeds)) or any(seed not in SEEDS for seed in seeds):
        failures.append("seed_accounting")
    paths = [row["path"] for row in manifest.get("artifacts", [])]
    if len(paths) != len(set(paths)) or len(paths) != manifest.get("artifact_count_excluding_manifest"):
        failures.append("artifact_accounting")
    for row in manifest.get("artifacts", []):
        path = run / row["path"]
        if not path.is_file() or path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            failures.append(f"artifact_integrity:{row['path']}")
    if require_go and not manifest.get("authoritative_aggregate", {}).get("go"):
        failures.append("phase4b_no_go")
    if require_smoke_go:
        if seeds != [4100, 4104]:
            failures.append("smoke_seed_set")
        records = manifest.get("capture_records", [])
        if len(records) != 2 or any(row.get("capture_status") != "captured" for row in records):
            failures.append("smoke_capture_status")
        stage_ids = [row.get("stage_identifier") for row in records]
        if any(not value for value in stage_ids) or len(set(stage_ids)) != len(stage_ids):
            failures.append("smoke_distinct_stage_ownership")
        for seed in seeds:
            lifecycle_path = run / f"seed_{seed}" / "camera_render_lifecycle.json"
            if not lifecycle_path.is_file():
                failures.append(f"smoke_lifecycle_missing:{seed}")
                continue
            lifecycle = _read(lifecycle_path)
            if not lifecycle.get("camera", {}).get("owned_by_current_stage"):
                failures.append(f"smoke_camera_ownership:{seed}")
            if not lifecycle.get("render_product", {}).get("owned_by_current_stage"):
                failures.append(f"smoke_render_ownership:{seed}")
            if not lifecycle.get("annotator", {}).get("owned_by_current_stage"):
                failures.append(f"smoke_annotator_ownership:{seed}")
            accepted = [row for row in lifecycle.get("capture_attempts", []) if row.get("accepted")]
            if len(accepted) != 1 or accepted[0].get("peak_to_peak", 0) <= 0:
                failures.append(f"smoke_frame_content:{seed}")
            detections_path = run / f"seed_{seed}" / "deterministic_cv_detections.json"
            if not detections_path.is_file():
                failures.append(f"smoke_detections_missing:{seed}")
                continue
            detections = _read(detections_path)
            if not detections.get("public_workspace_configuration_used"):
                failures.append(f"smoke_workspace_not_used:{seed}")
            if detections.get("accepted_cube_count") != 6:
                failures.append(f"smoke_cube_count:{seed}")
            if detections.get("counts_by_color") != {"red": 2, "blue": 2, "yellow": 2}:
                failures.append(f"smoke_color_counts:{seed}")
            if any(
                row.get("workspace_containment", {}).get("passed") is not True
                for row in detections.get("accepted_detections", [])
            ):
                failures.append(f"smoke_accepted_outside_workspace:{seed}")
            workspace_red_rejections = [
                row for row in detections.get("rejected_candidates", [])
                if row.get("color") == "red"
                and row.get("rejection_reasons") == [
                    "outside_calibrated_source_workspace"
                ]
            ]
            if not workspace_red_rejections:
                failures.append(f"smoke_sensor_nose_workspace_rejection:{seed}")
        if not manifest.get("authoritative_aggregate", {}).get("go"):
            failures.append("phase4b_smoke_no_go")
    result = {
        "valid": not failures,
        "failures": failures,
        "go": manifest.get("authoritative_aggregate", {}).get("go", False),
        "accounted_seed_count": len(manifest.get("seeds", [])),
        "artifact_count": len(paths),
    }
    if failures:
        raise SystemExit("Phase 4B output validation failed: " + "; ".join(failures))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    process = sub.add_parser("process")
    process.add_argument("--run", required=True, type=Path)
    validate = sub.add_parser("validate-output")
    validate.add_argument("--run", required=True, type=Path)
    validate.add_argument("--require-go", action="store_true")
    validate.add_argument("--require-smoke-go", action="store_true")
    args = parser.parse_args()
    if args.command == "process":
        result = process_batch(args.run.resolve())
        print(json.dumps(result["authoritative_aggregate"], indent=2))
    else:
        print(json.dumps(validate_output(
            args.run.resolve(), args.require_go, args.require_smoke_go
        ), indent=2))


if __name__ == "__main__":
    main()
