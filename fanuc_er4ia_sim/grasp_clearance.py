"""Pure fail-closed clearance calculations for a square in parallel jaws."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


def normalize_square_yaw_deg(angle_deg: float) -> float:
    if not math.isfinite(float(angle_deg)):
        raise ValueError("yaw must be finite")
    return (float(angle_deg) + 45.0) % 90.0 - 45.0


def presented_square_width_mm(edge_mm: float, relative_yaw_deg: float) -> float:
    if not math.isfinite(float(edge_mm)) or edge_mm <= 0:
        raise ValueError("square edge must be positive and finite")
    angle = math.radians(normalize_square_yaw_deg(relative_yaw_deg))
    return float(edge_mm) * (abs(math.cos(angle)) + abs(math.sin(angle)))


def opening_clearance_mm(
    relative_yaw_deg: float, *, edge_mm: float = 10.0, opening_mm: float = 12.0
) -> dict:
    if not math.isfinite(float(opening_mm)) or opening_mm <= 0:
        raise ValueError("opening must be positive and finite")
    width = presented_square_width_mm(edge_mm, relative_yaw_deg)
    total = float(opening_mm) - width
    return {
        "relative_yaw_deg": normalize_square_yaw_deg(relative_yaw_deg),
        "presented_width_mm": width,
        "total_clearance_mm": total,
        "per_side_clearance_mm": total / 2.0,
    }


@dataclass(frozen=True)
class ClearanceMeasurements:
    cube_edge_mm: float = 10.0
    open_gap_mm: float = 12.0
    dimensional_tolerance_per_side_mm: float | None = None
    finger_tolerance_per_side_mm: float | None = None
    version: str = "unverified-csi-measurements"


@dataclass(frozen=True)
class ClearancePolicy:
    maximum_combined_yaw_error_deg: float = 3.0
    maximum_yaw_uncertainty_deg: float = 1.0
    maximum_centering_error_mm: float = 0.25
    minimum_residual_per_side_mm: float = 0.25


def authorize_grasp_clearance(
    yaw_error_deg: float,
    yaw_uncertainty_deg: float,
    centering_error_mm: float,
    measurements: ClearanceMeasurements,
    policy: ClearancePolicy = ClearancePolicy(),
) -> dict:
    """Return an explicit authorization result; unknown tolerances always fail."""
    values = (yaw_error_deg, yaw_uncertainty_deg, centering_error_mm)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("yaw and centering inputs must be finite")
    reasons: list[str] = []
    if measurements.dimensional_tolerance_per_side_mm is None:
        reasons.append("dimensional_tolerance_unverified")
    if measurements.finger_tolerance_per_side_mm is None:
        reasons.append("finger_tolerance_unverified")
    uncertainty = abs(float(yaw_uncertainty_deg))
    combined = abs(normalize_square_yaw_deg(yaw_error_deg)) + uncertainty
    if uncertainty > policy.maximum_yaw_uncertainty_deg:
        reasons.append("yaw_uncertainty_exceeds_limit")
    if combined > policy.maximum_combined_yaw_error_deg:
        reasons.append("combined_yaw_error_exceeds_limit")
    centering = abs(float(centering_error_mm))
    if centering > policy.maximum_centering_error_mm:
        reasons.append("centering_error_exceeds_limit")
    clearance = opening_clearance_mm(
        combined,
        edge_mm=measurements.cube_edge_mm,
        opening_mm=measurements.open_gap_mm,
    )
    tolerance_budget = None
    residual = None
    if (
        measurements.dimensional_tolerance_per_side_mm is not None
        and measurements.finger_tolerance_per_side_mm is not None
    ):
        tolerance_budget = (
            abs(float(measurements.dimensional_tolerance_per_side_mm))
            + abs(float(measurements.finger_tolerance_per_side_mm))
            + centering
        )
        residual = clearance["per_side_clearance_mm"] - tolerance_budget
        if residual < policy.minimum_residual_per_side_mm:
            reasons.append("residual_per_side_clearance_below_limit")
    return {
        "authorized": not reasons,
        "reasons": reasons,
        "combined_yaw_bound_deg": combined,
        "clearance": clearance,
        "tolerance_budget_per_side_mm": tolerance_budget,
        "residual_per_side_clearance_mm": residual,
        "measurements": asdict(measurements),
        "policy": asdict(policy),
        "rounded_edge_credit_mm": 0.0,
        "hook_geometry_credit_mm": 0.0,
    }
