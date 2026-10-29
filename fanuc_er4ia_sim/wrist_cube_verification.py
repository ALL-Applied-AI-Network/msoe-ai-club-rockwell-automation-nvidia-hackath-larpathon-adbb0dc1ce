"""Fail-closed fusion gates for a wrist RGB-depth cube verification view."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


def square_yaw_error_deg(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 45.0) % 90.0 - 45.0)


@dataclass(frozen=True)
class VerificationThresholds:
    maximum_xy_disagreement_mm: float = 1.0
    maximum_combined_yaw_error_deg: float = 3.0
    maximum_yaw_uncertainty_deg: float = 1.0
    maximum_closing_axis_uncertainty_mm: float = 0.25
    minimum_depth_valid_fraction: float = 0.90
    maximum_cube_top_height_error_mm: float = 2.0
    maximum_table_plane_residual_mm: float = 1.0


@dataclass(frozen=True)
class WristCubeEvidence:
    candidate_count: int
    table_detection_id: str | None
    wrist_detection_id: str | None
    expected_colour: str
    wrist_colour: str | None
    table_xy_mm: tuple[float, float]
    wrist_xy_mm: tuple[float, float]
    expected_yaw_deg: float
    measured_yaw_deg: float
    yaw_uncertainty_deg: float
    closing_axis_uncertainty_mm: float
    depth_valid_fraction: float
    measured_cube_top_height_mm: float
    expected_cube_top_height_mm: float
    table_plane_residual_mm: float
    approach_corridor_clear: bool


def verify_wrist_cube(
    evidence: WristCubeEvidence,
    thresholds: VerificationThresholds = VerificationThresholds(),
) -> dict:
    numeric = (
        *evidence.table_xy_mm,
        *evidence.wrist_xy_mm,
        evidence.expected_yaw_deg,
        evidence.measured_yaw_deg,
        evidence.yaw_uncertainty_deg,
        evidence.closing_axis_uncertainty_mm,
        evidence.depth_valid_fraction,
        evidence.measured_cube_top_height_mm,
        evidence.expected_cube_top_height_mm,
        evidence.table_plane_residual_mm,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("wrist verification evidence must be finite")
    reasons: list[str] = []
    if evidence.candidate_count != 1:
        reasons.append("candidate_not_unique")
    if not evidence.table_detection_id or evidence.table_detection_id != evidence.wrist_detection_id:
        reasons.append("detection_identity_mismatch")
    if evidence.wrist_colour != evidence.expected_colour:
        reasons.append("colour_mismatch")
    xy_disagreement = math.dist(evidence.table_xy_mm, evidence.wrist_xy_mm)
    if xy_disagreement > thresholds.maximum_xy_disagreement_mm:
        reasons.append("table_wrist_xy_disagreement_exceeds_limit")
    yaw_error = square_yaw_error_deg(evidence.expected_yaw_deg, evidence.measured_yaw_deg)
    uncertainty = abs(evidence.yaw_uncertainty_deg)
    if uncertainty > thresholds.maximum_yaw_uncertainty_deg:
        reasons.append("yaw_uncertainty_exceeds_limit")
    if yaw_error + uncertainty > thresholds.maximum_combined_yaw_error_deg:
        reasons.append("combined_yaw_error_exceeds_limit")
    if abs(evidence.closing_axis_uncertainty_mm) > thresholds.maximum_closing_axis_uncertainty_mm:
        reasons.append("closing_axis_uncertainty_exceeds_limit")
    if evidence.depth_valid_fraction < thresholds.minimum_depth_valid_fraction:
        reasons.append("insufficient_valid_depth")
    height_error = abs(evidence.measured_cube_top_height_mm - evidence.expected_cube_top_height_mm)
    if height_error > thresholds.maximum_cube_top_height_error_mm:
        reasons.append("cube_top_height_error_exceeds_limit")
    if abs(evidence.table_plane_residual_mm) > thresholds.maximum_table_plane_residual_mm:
        reasons.append("table_plane_residual_exceeds_limit")
    if not evidence.approach_corridor_clear:
        reasons.append("approach_corridor_not_clear")
    return {
        "authorized": not reasons,
        "reasons": reasons,
        "xy_disagreement_mm": xy_disagreement,
        "yaw_error_deg": yaw_error,
        "combined_yaw_bound_deg": yaw_error + uncertainty,
        "cube_top_height_error_mm": height_error,
        "thresholds": asdict(thresholds),
    }
