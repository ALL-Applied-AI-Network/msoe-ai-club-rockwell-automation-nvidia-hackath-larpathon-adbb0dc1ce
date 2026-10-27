"""Validate required Phase 3A outputs and record optional video status."""

import argparse
import json
from pathlib import Path


REQUIRED_MOTION_FILES = (
    "execution_report.json",
    "execution_events.json",
    "before_table_camera.png",
    "after_table_camera.png",
    "after_table_camera_annotated.png",
    "execution_progress.log",
)
VIDEO_STATUS_FILE = "video_postprocessing_report.json"


def validate_required_motion_outputs(run):
    """Require successful motion evidence independently of optional video."""
    run = Path(run)
    missing = [name for name in REQUIRED_MOTION_FILES if not (run / name).is_file()]
    side_frames = run / "side_frames"
    if not side_frames.is_dir() or not any(side_frames.glob("*.png")):
        missing.append("side_frames/*.png")
    if missing:
        raise RuntimeError(f"Missing required smoke outputs: {missing}")

    report = json.loads((run / "execution_report.json").read_text(encoding="utf-8"))
    if not report.get("smoke_test_success"):
        reason = report.get("abort_reason") or report.get("abort_conditions")
        raise RuntimeError(f"Oracle smoke motion failed: {reason}")
    return report


def record_video_status(run, *, status, error=None, output="side_execution.mp4"):
    """Write optional video outcome without changing the execution report."""
    if status not in {"success", "unavailable", "failed"}:
        raise ValueError(f"Unsupported video status: {status}")
    run = Path(run)
    output_path = run / output
    if status == "success" and not output_path.is_file():
        raise RuntimeError(f"Video reported success but is missing: {output_path}")
    payload = {
        "video_requested": True,
        "video_status": status,
        "video_error": error,
        "video_output": output if status == "success" and output_path.is_file() else None,
    }
    (run / VIDEO_STATUS_FILE).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-motion")
    validate.add_argument("--run", required=True)
    video = subparsers.add_parser("record-video")
    video.add_argument("--run", required=True)
    video.add_argument(
        "--status", choices=("success", "unavailable", "failed"), required=True
    )
    video.add_argument("--error")
    video.add_argument("--output", default="side_execution.mp4")
    args = parser.parse_args(argv)

    if args.command == "validate-motion":
        validate_required_motion_outputs(args.run)
        print("[oracle-smoke] required motion outputs PASS")
    else:
        payload = record_video_status(
            args.run, status=args.status, error=args.error, output=args.output
        )
        print(f"[oracle-smoke] video status: {payload['video_status']}")


if __name__ == "__main__":
    main()
