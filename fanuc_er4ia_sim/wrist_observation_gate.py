"""Transmission-free verification gate for stationary native wrist observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from camera_calibration_contract import (
    CameraCalibration,
    RgbDepthRegistration,
    validate_camera_calibration,
    validate_registration,
)
from grasp_clearance import ClearanceMeasurements, authorize_grasp_clearance
from native_observation import NativeObservation
from wrist_cube_verification import WristCubeEvidence, verify_wrist_cube


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class WristGateEvidence:
    candidate_count: int
    detection_id: str | None
    colour: str | None
    wrist_xy_mm: tuple[float, float]
    measured_yaw_deg: float | None
    yaw_uncertainty_deg: float | None
    closing_axis_uncertainty_mm: float
    depth_valid_fraction: float | None
    measured_cube_top_height_mm: float | None
    expected_cube_top_height_mm: float | None
    table_plane_residual_mm: float | None
    approach_corridor_clear: bool
    centering_error_mm: float
    depth_used: bool


def verify_wrist_observations(
    observations: list[NativeObservation],
    *,
    selected_table_detection: dict,
    expected_yaw_deg: float | None,
    evidence: WristGateEvidence,
    wrist_rgb_calibration: CameraCalibration | None,
    wrist_depth_calibration: CameraCalibration | None = None,
    registration: RgbDepthRegistration | None = None,
    clearance_measurements: ClearanceMeasurements | None = None,
    maximum_frame_span_seconds: float = 0.5,
    maximum_joint_delta_deg: float = 0.05,
    maximum_tcp_delta: float = 0.10,
    minimum_mean_frame_delta: float = 0.01,
    maximum_mean_frame_delta: float = 12.0,
) -> dict:
    reasons: list[str] = []
    if len(observations) < 2:
        reasons.append("insufficient_repeated_wrist_frames")
    if selected_table_detection.get("category") != "cube" or not selected_table_detection.get("detection_id"):
        reasons.append("selected_table_detection_invalid")
    if expected_yaw_deg is None or not math.isfinite(float(expected_yaw_deg)):
        reasons.append("expected_yaw_missing")
    try:
        rgb_calibration = validate_camera_calibration(
            wrist_rgb_calibration, expected_stream="wrist_rgb"
        )
    except ValueError as error:
        rgb_calibration = None
        reasons.append(str(error))
    depth_calibration = None
    if evidence.depth_used:
        try:
            depth_calibration = validate_camera_calibration(
                wrist_depth_calibration,
                expected_stream="wrist_depth",
                expected_compatibility_id=None if rgb_calibration is None else rgb_calibration.compatibility_id,
            )
            if rgb_calibration is None:
                raise ValueError("wrist RGB calibration is required before depth registration")
            validate_registration(
                registration,
                rgb_calibration=rgb_calibration,
                depth_calibration=depth_calibration,
            )
        except ValueError as error:
            reasons.append(str(error))

    frame_hashes: list[str] = []
    if observations:
        trial_hash = observations[0].trial_sha256
        times = [item.timestamp for item in observations]
        if any(item.trial_sha256 != trial_hash for item in observations):
            reasons.append("cross_trial_wrist_observations")
        if any(second <= first for first, second in zip(times, times[1:])):
            reasons.append("wrist_timestamps_not_strictly_increasing")
        if max(times) - min(times) > maximum_frame_span_seconds:
            reasons.append("wrist_observation_bundle_stale")
        base_joints = observations[0].joints[:6]
        base_tcp = observations[0].tcp_pose
        for item in observations:
            if "wrist" not in item.rgb:
                reasons.append("wrist_rgb_missing")
                continue
            record = next(
                record for record in item.provenance if record.camera == "wrist" and record.stream == "rgb"
            )
            frame_hashes.append(record.content_sha256)
            if np.max(np.abs(item.joints[:6] - base_joints)) > maximum_joint_delta_deg:
                reasons.append("robot_not_stationary_joint_delta")
            if np.max(np.abs(item.tcp_pose - base_tcp)) > maximum_tcp_delta:
                reasons.append("robot_not_stationary_tcp_delta")
        if len(frame_hashes) != len(set(frame_hashes)):
            reasons.append("repeated_wrist_frame_hash")
        for first, second in zip(observations, observations[1:]):
            if "wrist" in first.rgb and "wrist" in second.rgb:
                delta = float(
                    np.mean(
                        np.abs(first.rgb["wrist"].astype(np.int16) - second.rgb["wrist"].astype(np.int16))
                    )
                )
                if delta < minimum_mean_frame_delta:
                    reasons.append("repeated_wrist_frames_frozen")
                if delta > maximum_mean_frame_delta:
                    reasons.append("repeated_wrist_frames_inconsistent")

    verification = None
    required_numeric = (
        evidence.measured_yaw_deg,
        evidence.yaw_uncertainty_deg,
        evidence.depth_valid_fraction,
        evidence.measured_cube_top_height_mm,
        evidence.expected_cube_top_height_mm,
        evidence.table_plane_residual_mm,
    )
    if any(value is None or not math.isfinite(float(value)) for value in required_numeric):
        reasons.append("wrist_yaw_or_depth_evidence_missing")
    elif expected_yaw_deg is not None:
        verification = verify_wrist_cube(
            WristCubeEvidence(
                candidate_count=evidence.candidate_count,
                table_detection_id=selected_table_detection.get("detection_id"),
                wrist_detection_id=evidence.detection_id,
                expected_colour=selected_table_detection.get("colour"),
                wrist_colour=evidence.colour,
                table_xy_mm=tuple(selected_table_detection.get("xy_mm", (math.nan, math.nan))),
                wrist_xy_mm=evidence.wrist_xy_mm,
                expected_yaw_deg=float(expected_yaw_deg),
                measured_yaw_deg=float(evidence.measured_yaw_deg),
                yaw_uncertainty_deg=float(evidence.yaw_uncertainty_deg),
                closing_axis_uncertainty_mm=evidence.closing_axis_uncertainty_mm,
                depth_valid_fraction=float(evidence.depth_valid_fraction),
                measured_cube_top_height_mm=float(evidence.measured_cube_top_height_mm),
                expected_cube_top_height_mm=float(evidence.expected_cube_top_height_mm),
                table_plane_residual_mm=float(evidence.table_plane_residual_mm),
                approach_corridor_clear=evidence.approach_corridor_clear,
            )
        )
        reasons.extend(verification["reasons"])

    measurements = clearance_measurements or ClearanceMeasurements()
    if evidence.measured_yaw_deg is None or evidence.yaw_uncertainty_deg is None or expected_yaw_deg is None:
        clearance = {"authorized": False, "reasons": ["yaw_evidence_missing"]}
    else:
        clearance = authorize_grasp_clearance(
            float(evidence.measured_yaw_deg) - float(expected_yaw_deg),
            float(evidence.yaw_uncertainty_deg),
            evidence.centering_error_mm,
            measurements,
        )
    reasons.extend(clearance["reasons"])
    reasons = list(dict.fromkeys(reasons))
    report = {
        "passed": not reasons,
        "reasons": reasons,
        "selected_detection_id": selected_table_detection.get("detection_id"),
        "observation_sha256": [item.observation_sha256 for item in observations],
        "wrist_frame_sha256": frame_hashes,
        "wrist_rgb_calibration_sha256": None if rgb_calibration is None else rgb_calibration.integrity_sha256(),
        "wrist_depth_calibration_sha256": None if depth_calibration is None else depth_calibration.integrity_sha256(),
        "verification": verification,
        "clearance": clearance,
        "evidence": asdict(evidence),
        "action": None,
        "robot_motion_authorized": False,
    }
    report["report_sha256"] = _hash(report)
    return report
