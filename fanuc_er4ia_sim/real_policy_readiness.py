"""Fail-closed aggregation for a future real organizer policy."""

from __future__ import annotations

from dataclasses import dataclass

from deployment_readiness import PRODUCTION_READINESS


PHASE5C_REQUIRED_VERIFICATIONS = (
    "real_table_rgb_calibration",
    "real_table_depth_calibration",
    "real_wrist_rgb_calibration",
    "real_wrist_depth_calibration",
    "real_rgb_depth_registration",
    "verified_detector_adapter",
    "verified_source_workspace",
)

UNRESOLVED_CSI_PREREQUISITES = tuple(
    dict.fromkeys(
        tuple(PRODUCTION_READINESS.report()["missing"])
        + PHASE5C_REQUIRED_VERIFICATIONS
        + (
            "organizer_policy_action_strategy",
            "observation_only_physical_checkout",
            "physical_safety_review",
            "production_policy_review",
        )
    )
)


@dataclass(frozen=True)
class RealPolicyReadiness:
    host_contract_verified: bool = False
    mock_interface_verified: bool = False
    phase5c_verifications: dict[str, bool] | None = None
    observation_session_complete: bool = False

    def report(self) -> dict:
        phase5b = PRODUCTION_READINESS.report()
        supplied = self.phase5c_verifications or {}
        unknown = sorted(set(supplied) - set(PHASE5C_REQUIRED_VERIFICATIONS))
        if unknown:
            raise ValueError(f"unknown Phase 5C readiness fields: {unknown}")
        phase5c = {
            name: supplied.get(name) is True for name in PHASE5C_REQUIRED_VERIFICATIONS
        }
        missing = list(phase5b["missing"])
        missing.extend(name for name, complete in phase5c.items() if not complete)
        missing.extend(
            [
                "organizer_policy_action_strategy",
                "observation_only_physical_checkout",
                "physical_safety_review",
                "production_policy_review",
            ]
        )
        return {
            "phase": "5D_host_only",
            "host_contract_verified": self.host_contract_verified is True,
            "mock_interface_verified": self.mock_interface_verified is True,
            "phase5b": phase5b,
            "phase5c_verifications": phase5c,
            "observation_session_complete": self.observation_session_complete is True,
            "observation_only_physical_checkout_pending": True,
            "motion_deployment_unauthorized": True,
            "unresolved_csi_prerequisites": list(dict.fromkeys(missing)),
            "all_requirements_verified": False,
            "production_ready": False,
            "robot_motion_authorized": False,
            "action": None,
        }


PRODUCTION_POLICY_READINESS = RealPolicyReadiness()
