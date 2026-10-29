"""Strict, host-only adapter for already-decoded organizer observations.

The adapter has no transport and deliberately accepts only documented field
names.  It binds observations to an explicit reset/trial boundary before
delegating native frame validation and hashing to :mod:`native_observation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from native_observation import NativeObservation, parse_native_observation


REQUIRED_ORGANIZER_FIELDS = {
    "t", "status", "trial", "queued", "joints", "tcp_pose", "images"
}
OPTIONAL_ORGANIZER_FIELDS = {"depth", "format"}
ORGANIZER_OBSERVATION_FIELDS = REQUIRED_ORGANIZER_FIELDS | OPTIONAL_ORGANIZER_FIELDS


@dataclass(frozen=True)
class AdaptedOrganizerObservation:
    native: NativeObservation
    reset_sequence: int
    observation_sequence: int
    source_schema: str = "organizer_decoded_observation_v1"

    def review_metadata(self) -> dict:
        return {
            "source_schema": self.source_schema,
            "reset_sequence": self.reset_sequence,
            "observation_sequence": self.observation_sequence,
            "trial_sha256": self.native.trial_sha256,
            "observation_sha256": self.native.observation_sha256,
            "timestamp": self.native.timestamp,
            "status": self.native.status,
            "queued": self.native.queued,
            "frames": [
                {
                    "camera": item.camera,
                    "stream": item.stream,
                    "content_sha256": item.content_sha256,
                    "dtype": item.dtype,
                    "shape": list(item.shape),
                }
                for item in self.native.provenance
            ],
        }


class OrganizerObservationAdapter:
    """Reset-bound adapter with an abort latch and no action API."""

    def __init__(self) -> None:
        self._reset_sequence = 0
        self._observation_sequence = 0
        self._expected_trial: object | None = None
        self._last_timestamp: float | None = None
        self._abort_reason: str | None = None

    @property
    def abort_reason(self) -> str | None:
        return self._abort_reason

    def reset(self, trial: object) -> None:
        if trial is None:
            raise ValueError("reset requires an explicit non-null trial identity")
        self._reset_sequence += 1
        self._observation_sequence = 0
        self._expected_trial = trial
        self._last_timestamp = None
        self._abort_reason = None

    def _fail(self, reason: str) -> None:
        self._abort_reason = reason
        raise ValueError(reason)

    def adapt(
        self,
        observation: dict,
        *,
        required_rgb: Iterable[str],
        requested_depth: Iterable[str] = (),
        reference_time: float | None = None,
        require_idle_queue: bool = True,
    ) -> AdaptedOrganizerObservation:
        if self._abort_reason is not None:
            raise RuntimeError("organizer observation adapter abort is latched until reset")
        if self._expected_trial is None:
            self._fail("adapter_reset_required")
        if not isinstance(observation, dict):
            self._fail("organizer observation must be a dictionary")
        missing = sorted(REQUIRED_ORGANIZER_FIELDS - set(observation))
        unknown = sorted(set(observation) - ORGANIZER_OBSERVATION_FIELDS)
        if missing:
            self._fail(f"organizer observation missing fields: {missing}")
        if unknown:
            self._fail(f"organizer observation contains unknown fields: {unknown}")
        if "format" in observation and observation["format"] not in ("png", "jpeg"):
            self._fail("organizer observation format must be png or jpeg")
        if observation["status"] != "running":
            self._fail("organizer observation status must be running")
        queued = observation["queued"]
        if require_idle_queue and (isinstance(queued, bool) or queued != 0):
            self._fail("organizer observation queue must be idle for review")
        joints = np.asarray(observation["joints"])
        if joints.shape != (7,) or not np.issubdtype(joints.dtype, np.number):
            self._fail("organizer joints must be a numeric vector with shape (7,)")
        if not np.isfinite(joints).all() or float(joints[6]) not in (0.0, 1.0):
            self._fail("organizer gripper observation must be finite and binary")
        try:
            native = parse_native_observation(
                observation,
                required_rgb=required_rgb,
                requested_depth=requested_depth,
                expected_trial=self._expected_trial,
                reference_time=reference_time,
            )
        except (TypeError, ValueError) as exc:
            self._fail(str(exc))
        if self._last_timestamp is not None and native.timestamp <= self._last_timestamp:
            self._fail("organizer observation timestamps must increase within a reset")
        self._last_timestamp = native.timestamp
        self._observation_sequence += 1
        return AdaptedOrganizerObservation(
            native=native,
            reset_sequence=self._reset_sequence,
            observation_sequence=self._observation_sequence,
        )
