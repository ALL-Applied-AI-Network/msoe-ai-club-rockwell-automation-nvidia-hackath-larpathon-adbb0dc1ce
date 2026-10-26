"""Validate a completed capture using output artifacts and its text log only."""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2


EXPECTED_RESOLUTION = [640, 360]
EXPECTED_OBJECT_COUNT = 10


def load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/job/out")
    parser.add_argument("--log", default="/job/out/capture_job.log")
    args = parser.parse_args()

    out = Path(args.out)
    log_path = Path(args.log)
    paths = {
        "image": out / "world_camera.png",
        "camera": out / "camera.json",
        "ground_truth": out / "ground_truth.json",
        "scene_state": out / "scene_state.json",
        "log": log_path,
    }
    checks = {}
    details = {"paths": {name: str(path) for name, path in paths.items()}}

    checks["all_required_files_exist"] = all(
        path.is_file() and path.stat().st_size > 0 for path in paths.values()
    )
    if not checks["all_required_files_exist"]:
        missing = [
            str(path)
            for path in paths.values()
            if not path.is_file() or path.stat().st_size == 0
        ]
        details["missing_or_empty"] = missing
    else:
        image = cv2.imread(str(paths["image"]), cv2.IMREAD_COLOR)
        camera = load_json(paths["camera"])
        ground_truth = load_json(paths["ground_truth"])
        scene_state = load_json(paths["scene_state"])
        log = paths["log"].read_text(encoding="utf-8", errors="replace")

        image_nonempty = image is not None and image.size > 0
        image_min = int(image.min()) if image_nonempty else 0
        image_max = int(image.max()) if image_nonempty else 0
        image_shape = list(image.shape) if image_nonempty else []
        details["image"] = {
            "shape": image_shape,
            "min": image_min,
            "max": image_max,
            "bytes": paths["image"].stat().st_size,
        }
        details["camera"] = camera
        details["ground_truth_objects"] = len(ground_truth.get("objects", []))
        details["scene_state_objects"] = len(scene_state.get("objects", {}))

        checks["image_decodes"] = image_nonempty
        checks["image_is_640x360"] = image_nonempty and image_shape[:2] == [360, 640]
        checks["image_nonblack"] = image_nonempty and image_max > 0
        checks["image_nonconstant"] = image_nonempty and image_min < image_max
        checks["camera_is_table"] = camera.get("camera") == "table"
        checks["camera_reports_640x360"] = camera.get("resolution") == EXPECTED_RESOLUTION
        checks["ground_truth_has_10_objects"] = (
            details["ground_truth_objects"] == EXPECTED_OBJECT_COUNT
        )
        checks["scene_state_has_10_objects"] = (
            details["scene_state_objects"] == EXPECTED_OBJECT_COUNT
        )

        setup_pattern = re.compile(
            r"^\[capture\] setup .*camera=/World/TableCam "
            r"camera_valid=True camera_type=Camera .*"
            r"render_product_valid=True timeline_playing=True "
            r"resolution=\(640, 360\)$",
            re.MULTILINE,
        )
        rgb_pattern = re.compile(
            r"^\[capture\] RGB ready after ([1-9][0-9]*) rendered steps$",
            re.MULTILINE,
        )
        setup_match = setup_pattern.search(log)
        rgb_match = rgb_pattern.search(log)
        checks["log_camera_prim_valid_and_typed"] = setup_match is not None
        checks["log_render_product_valid"] = setup_match is not None
        checks["log_timeline_playing"] = setup_match is not None
        checks["log_rgb_became_available"] = rgb_match is not None
        details["rgb_ready_after_steps"] = (
            int(rgb_match.group(1)) if rgb_match is not None else None
        )

    failures = [name for name, passed in checks.items() if not passed]
    report = {
        "validation": "capture_outputs_only",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "details": details,
    }
    report_path = out / "capture_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
