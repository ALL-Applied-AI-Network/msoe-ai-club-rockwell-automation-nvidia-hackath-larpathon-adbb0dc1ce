"""Explicit Phase 5B deployment prerequisites; this phase never authorizes motion."""

from __future__ import annotations

from dataclasses import dataclass, field


REQUIRED_VERIFICATIONS = (
    "real_table_calibration",
    "real_wrist_rgb_calibration",
    "real_wrist_depth_calibration",
    "surveyed_table_plane",
    "source_workspace",
    "bin_centres_and_footprints",
    "tcp_uframe_utool",
    "tcp_r_sign_and_gripper_offset",
    "real_open_gap_measurement",
    "finger_parallelism_installation_tolerance",
    "wrist_xy_yaw_repeatability",
    "bin_release_height",
    "action_schema_compatibility",
    "organizer_placement_schema",
    "inventory_and_identifiers",
)


@dataclass(frozen=True)
class DeploymentReadiness:
    verifications: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in REQUIRED_VERIFICATIONS}
    )
    evidence_references: dict[str, str] = field(default_factory=dict)

    def report(self) -> dict:
        unknown = sorted(set(self.verifications) - set(REQUIRED_VERIFICATIONS))
        if unknown:
            raise ValueError(f"unknown readiness fields: {unknown}")
        completed = {
            name: self.verifications.get(name) is True for name in REQUIRED_VERIFICATIONS
        }
        missing = [name for name, ready in completed.items() if not ready]
        return {
            "phase": "5B_host_only",
            "verifications": completed,
            "missing": missing,
            "all_requirements_verified": not missing,
            "robot_motion_authorized": False,
            "authorization_reason": "Phase 5B implements host-only contracts and cannot authorize motion",
            "evidence_references": dict(self.evidence_references),
        }


PRODUCTION_READINESS = DeploymentReadiness()
