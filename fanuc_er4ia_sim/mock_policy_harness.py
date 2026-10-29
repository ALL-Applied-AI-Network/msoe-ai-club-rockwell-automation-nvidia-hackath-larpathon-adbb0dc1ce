"""Local mock/test compatibility harness; never a production policy entry point."""

from __future__ import annotations

import numpy as np

from organizer_contract import hold_still_action, validate_tcp_actions
from organizer_observation_adapter import AdaptedOrganizerObservation


ORGANIZER_REQUIRES_ACTION_EACH_DECISION_CYCLE = True
EXPLICIT_MOCK_MODES = frozenset({"mock", "test"})


def validate_mock_hold(record: dict, observation: AdaptedOrganizerObservation) -> np.ndarray:
    if record.get("mock_only") is not True or record.get("transmitted") is not False:
        raise ValueError("HOLD record must be local and explicitly mock-only")
    if record.get("robot_motion_authorized") is not False:
        raise ValueError("mock HOLD cannot authorize motion")
    if set(record) != {
        "mock_only", "transmitted", "robot_motion_authorized", "dt_seconds", "tcp"
    }:
        raise ValueError("mock HOLD record has unexpected fields")
    rows = validate_tcp_actions(record["tcp"], dt_seconds=record["dt_seconds"])
    expected = np.concatenate(
        (observation.native.tcp_pose, [observation.native.joints[6]])
    ).reshape(1, 7)
    if not np.array_equal(rows, expected):
        raise ValueError("mock HOLD must exactly reproduce current TCP and gripper")
    return rows


class MockPolicyHarness:
    """Produces a validated local artifact only when mode is explicitly mock/test."""

    def __init__(self, *, mode: str) -> None:
        if mode not in EXPLICIT_MOCK_MODES:
            raise ValueError("mock policy harness is unavailable outside mock/test mode")
        self.mode = mode

    def propose_hold(
        self, observation: AdaptedOrganizerObservation, *, dt_seconds: float = 0.1
    ) -> dict:
        source = {
            "tcp_pose": observation.native.tcp_pose,
            "joints": observation.native.joints,
        }
        rows = hold_still_action(source)
        validate_tcp_actions(rows, dt_seconds=dt_seconds)
        record = {
            "mock_only": True,
            "transmitted": False,
            "robot_motion_authorized": False,
            "dt_seconds": float(dt_seconds),
            "tcp": rows.tolist(),
        }
        validate_mock_hold(record, observation)
        return record

    @staticmethod
    def production_output() -> dict:
        return {
            "action": None,
            "robot_motion_authorized": False,
            "mock_artifacts_included": False,
        }
