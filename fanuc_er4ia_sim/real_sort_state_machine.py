"""Host-only, transmission-free state machine for the proposed real sort policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import hashlib

from organizer_contract import hold_still_action, validate_tcp_actions


class Phase(Enum):
    RESET = auto()
    CALIBRATION_CHECK = auto()
    ROBOT_STATE_CHECK = auto()
    TABLE_CAPTURE = auto()
    TABLE_PERCEPTION = auto()
    INVENTORY_GATE = auto()
    SCENE_GATE = auto()
    FREEZE_PLAN = auto()
    PREFLIGHT_ALL = auto()
    SELECT = auto()
    WRIST_OBSERVE_MOVE = auto()
    WAIT_SETTLED = auto()
    WRIST_CAPTURE = auto()
    WRIST_VERIFY = auto()
    PICK_AUTHORIZE = auto()
    APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    VERIFY_LIFT = auto()
    TRANSIT = auto()
    LOWER_TO_RELEASE = auto()
    OPEN = auto()
    RETREAT = auto()
    REOBSERVE = auto()
    PRESERVATION_GATE = auto()
    NEXT = auto()
    COMPLETE = auto()
    HOLD = auto()
    ABORT_LATCHED = auto()


ORDER = tuple(
    phase for phase in Phase if phase not in {Phase.HOLD, Phase.ABORT_LATCHED}
)


@dataclass(frozen=True)
class ProposedAction:
    phase: Phase
    rows: object
    dt_seconds: float
    transmitted: bool = False


class RealSortStateMachine:
    """Pure proposal state.  No client/network import and no send method exist."""

    def __init__(self) -> None:
        self.phase = Phase.RESET
        self.scene_hash: str | None = None
        self.selected_detection_id: str | None = None
        self.grasp_authorization: str | None = None
        self.grasp_authorization_consumed = False
        self.verified_pickups = 0
        self.verified_placements = 0
        self.distractor_interactions = 0
        self.abort_reason: str | None = None

    def reset(self) -> None:
        self.__init__()

    def abort(self, reason: str) -> None:
        self.abort_reason = reason
        self.phase = Phase.ABORT_LATCHED

    def apply_gate(self, passed: bool, *, failure_reason: str) -> Phase:
        """Consume a pure validation result and latch on the first failure."""
        if self.phase is Phase.ABORT_LATCHED:
            raise RuntimeError("abort is latched until reset")
        if passed is not True:
            self.abort(failure_reason)
            return self.phase
        return self.advance()

    def authorize_scene(self, scene_hash: str) -> None:
        if self.phase is Phase.ABORT_LATCHED:
            raise RuntimeError("abort is latched until reset")
        if not scene_hash:
            self.abort("empty_scene_hash")
            raise ValueError("scene hash is required")
        self.scene_hash = scene_hash

    def bind_grasp_authorization(self, observation_hash: str, detection_id: str) -> str:
        if not self.scene_hash:
            raise RuntimeError("scene must be authorized first")
        if not observation_hash or not detection_id:
            raise ValueError("observation hash and detection id are required")
        self.selected_detection_id = detection_id
        payload = f"{self.scene_hash}\0{observation_hash}\0{detection_id}".encode()
        self.grasp_authorization = hashlib.sha256(payload).hexdigest()
        self.grasp_authorization_consumed = False
        return self.grasp_authorization

    def verify_scene_unchanged(self, scene_hash: str) -> None:
        if self.scene_hash != scene_hash:
            self.abort("scene_changed_after_authorization")
            raise RuntimeError("scene changed after authorization")

    def consume_grasp_authorization(self, token: str) -> None:
        if token != self.grasp_authorization or self.grasp_authorization_consumed:
            self.abort("invalid_or_reused_grasp_authorization")
            raise RuntimeError("invalid or reused grasp authorization")
        self.grasp_authorization_consumed = True

    def propose(
        self,
        rows: object,
        *,
        dt_seconds: float,
        authorization_token: str | None = None,
    ) -> ProposedAction:
        if self.phase is Phase.ABORT_LATCHED:
            raise RuntimeError("abort is latched until reset")
        if not self.scene_hash:
            raise RuntimeError("movement proposal requires an authorized scene")
        if self.phase is Phase.DESCEND:
            if authorization_token is None:
                self.abort("descent_without_wrist_authorization")
                raise RuntimeError("descent requires the single-use wrist authorization")
            self.consume_grasp_authorization(authorization_token)
        validated = validate_tcp_actions(rows, dt_seconds=dt_seconds)
        return ProposedAction(self.phase, validated.copy(), float(dt_seconds), False)

    def hold(self, observation: dict, *, dt_seconds: float = 0.1) -> ProposedAction:
        rows = hold_still_action(observation)
        validate_tcp_actions(rows, dt_seconds=dt_seconds)
        return ProposedAction(self.phase, rows, float(dt_seconds), False)

    def advance(self) -> Phase:
        if self.phase is Phase.ABORT_LATCHED:
            raise RuntimeError("abort is latched until reset")
        if self.phase is Phase.COMPLETE:
            return self.phase
        if self.phase is Phase.NEXT:
            self.phase = Phase.COMPLETE if self.preservation_ok() and self.verified_placements == 6 else Phase.SELECT
            return self.phase
        self.phase = ORDER[ORDER.index(self.phase) + 1]
        return self.phase

    def record_pickup(self, *, verified: bool, distractor_touched: bool = False) -> None:
        if distractor_touched:
            self.distractor_interactions += 1
            self.abort("distractor_interaction")
            return
        if not verified:
            self.abort("pickup_not_verified")
            return
        self.verified_pickups += 1

    def record_placement(self, *, verified: bool) -> None:
        if not verified:
            self.abort("placement_not_verified")
            return
        if self.verified_placements >= self.verified_pickups:
            self.abort("placement_without_verified_pickup")
            return
        self.verified_placements += 1

    def preservation_ok(self) -> bool:
        return (
            self.distractor_interactions == 0
            and self.verified_pickups == self.verified_placements
            and self.verified_placements <= 6
            and self.phase is not Phase.ABORT_LATCHED
        )
