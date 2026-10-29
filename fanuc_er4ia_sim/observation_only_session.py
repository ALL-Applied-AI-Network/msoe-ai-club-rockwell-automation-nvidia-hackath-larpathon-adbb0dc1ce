"""Hash-chained, transmission-free state for observation-only review sessions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, auto
import hashlib
import json


class ObservationPhase(Enum):
    RESET = auto()
    TABLE_OBSERVE = auto()
    TABLE_VERIFY = auto()
    WRIST_OBSERVE_REQUESTED = auto()
    WRIST_VERIFY = auto()
    REVIEW_COMPLETE = auto()
    ABORT_LATCHED = auto()


@dataclass(frozen=True)
class ReviewEvent:
    sequence: int
    phase: str
    event: str
    payload_sha256: str
    previous_event_sha256: str | None
    event_sha256: str


def unresolved_wrist_observation_proposal() -> dict:
    return {
        "resolved": False,
        "tcp_pose": None,
        "action_rows": None,
        "requires": [
            "verified_wrist_extrinsics",
            "verified_uframe_utool",
            "verified_tcp_r_mapping",
            "verified_safe_height",
            "human_safety_review",
        ],
        "transmittable": False,
    }


class ObservationOnlySession:
    """Review state only.  It has no action, client, transport, or send method."""

    def __init__(self) -> None:
        self.phase = ObservationPhase.RESET
        self.events: list[ReviewEvent] = []
        self.abort_reason: str | None = None

    @staticmethod
    def _hash(value: object) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()

    def _record(self, event: str, payload: dict) -> ReviewEvent:
        payload_hash = self._hash(payload)
        previous = None if not self.events else self.events[-1].event_sha256
        event_hash = self._hash(
            {
                "sequence": len(self.events) + 1,
                "phase": self.phase.name,
                "event": event,
                "payload_sha256": payload_hash,
                "previous_event_sha256": previous,
            }
        )
        record = ReviewEvent(
            len(self.events) + 1,
            self.phase.name,
            event,
            payload_hash,
            previous,
            event_hash,
        )
        self.events.append(record)
        return record

    def _require(self, expected: ObservationPhase) -> None:
        if self.phase is ObservationPhase.ABORT_LATCHED:
            raise RuntimeError("observation-session abort is latched until reset")
        if self.phase is not expected:
            self.abort(f"invalid_transition:{self.phase.name}:expected:{expected.name}")
            raise RuntimeError("invalid observation-session transition")

    def reset(self) -> None:
        self.__init__()

    def abort(self, reason: str) -> None:
        if self.phase is ObservationPhase.ABORT_LATCHED:
            return
        self.abort_reason = reason
        self.phase = ObservationPhase.ABORT_LATCHED
        self._record("abort", {"reason": reason})

    def begin(self) -> None:
        self._require(ObservationPhase.RESET)
        self.phase = ObservationPhase.TABLE_OBSERVE
        self._record("begin_table_observation", {})

    def record_table_observation(self, observation_sha256: str) -> None:
        self._require(ObservationPhase.TABLE_OBSERVE)
        if not observation_sha256:
            self.abort("missing_table_observation_hash")
            return
        self.phase = ObservationPhase.TABLE_VERIFY
        self._record("table_observation_recorded", {"observation_sha256": observation_sha256})

    def record_table_verification(self, report: dict) -> None:
        self._require(ObservationPhase.TABLE_VERIFY)
        if report.get("passed") is not True or not report.get("record_sha256"):
            self.abort("table_verification_failed")
            return
        self.phase = ObservationPhase.WRIST_OBSERVE_REQUESTED
        self._record(
            "table_verification_passed",
            {
                "record_sha256": report["record_sha256"],
                "wrist_proposal": unresolved_wrist_observation_proposal(),
            },
        )

    def record_wrist_observations(self, observation_sha256: list[str]) -> None:
        self._require(ObservationPhase.WRIST_OBSERVE_REQUESTED)
        if len(observation_sha256) < 2 or any(not value for value in observation_sha256):
            self.abort("insufficient_wrist_observation_hashes")
            return
        self.phase = ObservationPhase.WRIST_VERIFY
        self._record("wrist_observations_recorded", {"observation_sha256": observation_sha256})

    def record_wrist_verification(self, report: dict) -> None:
        self._require(ObservationPhase.WRIST_VERIFY)
        if report.get("passed") is not True or not report.get("report_sha256"):
            self.abort("wrist_verification_failed")
            return
        if report.get("action") is not None or report.get("robot_motion_authorized") is not False:
            self.abort("wrist_report_contains_motion_authority")
            return
        self.phase = ObservationPhase.REVIEW_COMPLETE
        self._record("review_complete", {"report_sha256": report["report_sha256"]})

    def review_record(self) -> dict:
        return {
            "phase": self.phase.name,
            "abort_reason": self.abort_reason,
            "events": [asdict(event) for event in self.events],
            "event_chain_head": None if not self.events else self.events[-1].event_sha256,
            "action": None,
            "robot_motion_authorized": False,
        }
