"""Validation boundary for recorded organizer-native camera observations.

This module performs no resizing, inference, client calls, or action creation.
It records content hashes and camera provenance without object or truth labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable

import numpy as np


NATIVE_RGB_SHAPES = {"table": (1080, 1920, 3), "wrist": (720, 1280, 3)}
NATIVE_DEPTH_SHAPES = {"table": (400, 640), "wrist": (480, 848)}
FORBIDDEN_OBSERVATION_KEYS = {
    "truth",
    "ground_truth",
    "scene_state",
    "oracle",
    "oracle_plan",
    "object_names",
    "prim_names",
    "placement",
}


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_vector(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric array with shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array.astype(np.float64, copy=True)


@dataclass(frozen=True)
class FrameProvenance:
    camera: str
    stream: str
    timestamp: float
    trial_sha256: str
    content_sha256: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class NativeObservation:
    timestamp: float
    status: str
    trial_sha256: str
    queued: int
    joints: np.ndarray
    tcp_pose: np.ndarray
    rgb: dict[str, np.ndarray]
    depth_mm: dict[str, np.ndarray]
    provenance: tuple[FrameProvenance, ...]
    observation_sha256: str


def parse_native_observation(
    observation: dict,
    *,
    required_rgb: Iterable[str],
    requested_depth: Iterable[str] = (),
    expected_trial: object | None = None,
    reference_time: float | None = None,
    maximum_age_seconds: float = 0.5,
    maximum_future_skew_seconds: float = 0.05,
) -> NativeObservation:
    if not isinstance(observation, dict):
        raise ValueError("observation must be a dictionary")
    forbidden = sorted(FORBIDDEN_OBSERVATION_KEYS & set(observation))
    if forbidden:
        raise ValueError(f"forbidden truth/state fields: {forbidden}")
    required = {"t", "status", "trial", "queued", "joints", "tcp_pose", "images"}
    missing = sorted(required - set(observation))
    if missing:
        raise ValueError(f"observation missing fields: {missing}")
    timestamp = float(observation["t"])
    if not math.isfinite(timestamp):
        raise ValueError("observation timestamp must be finite")
    if reference_time is not None:
        if not math.isfinite(float(reference_time)):
            raise ValueError("reference time must be finite")
        age = float(reference_time) - timestamp
        if age > maximum_age_seconds:
            raise ValueError("observation is stale")
        if age < -maximum_future_skew_seconds:
            raise ValueError("observation timestamp is unexpectedly in the future")
    status = observation["status"]
    if not isinstance(status, str) or not status.strip():
        raise ValueError("observation status must be a nonempty string")
    trial_hash = _canonical_hash(observation["trial"])
    if expected_trial is not None and trial_hash != _canonical_hash(expected_trial):
        raise ValueError("observation trial does not match the expected trial")
    queued_raw = observation["queued"]
    if isinstance(queued_raw, bool) or not isinstance(queued_raw, (int, np.integer)):
        raise ValueError("queued must be a nonnegative integer")
    queued = int(queued_raw)
    if queued < 0:
        raise ValueError("queued must be a nonnegative integer")
    joints = _finite_vector(observation["joints"], (7,), "joints")
    tcp_pose = _finite_vector(observation["tcp_pose"], (6,), "tcp_pose")

    rgb_names = tuple(required_rgb)
    if not rgb_names or len(rgb_names) != len(set(rgb_names)):
        raise ValueError("required RGB cameras must be a nonempty unique sequence")
    if any(name not in NATIVE_RGB_SHAPES for name in rgb_names):
        raise ValueError("unsupported native RGB camera")
    images = observation["images"]
    if not isinstance(images, dict) or set(images) != set(rgb_names):
        raise ValueError("images must contain exactly the required native RGB streams")
    rgb: dict[str, np.ndarray] = {}
    provenance: list[FrameProvenance] = []
    for camera in rgb_names:
        image = np.asarray(images[camera])
        if image.shape != NATIVE_RGB_SHAPES[camera] or image.dtype != np.uint8:
            raise ValueError(f"{camera} RGB does not match its native uint8 dimensions")
        if int(image.max()) == 0:
            raise ValueError(f"{camera} RGB frame is black")
        if int(image.min()) == int(image.max()):
            raise ValueError(f"{camera} RGB frame is constant")
        preserved = np.array(image, copy=True, order="C")
        preserved.setflags(write=False)
        digest = _array_hash(preserved)
        rgb[camera] = preserved
        provenance.append(
            FrameProvenance(camera, "rgb", timestamp, trial_hash, digest, str(preserved.dtype), preserved.shape)
        )

    depth_names = tuple(requested_depth)
    if len(depth_names) != len(set(depth_names)) or any(
        name not in NATIVE_DEPTH_SHAPES for name in depth_names
    ):
        raise ValueError("requested depth cameras are invalid or duplicated")
    depth_source = observation.get("depth", {})
    if not isinstance(depth_source, dict) or set(depth_source) != set(depth_names):
        raise ValueError("depth must contain exactly the requested native depth streams")
    depth: dict[str, np.ndarray] = {}
    for camera in depth_names:
        frame = np.asarray(depth_source[camera])
        if frame.shape != NATIVE_DEPTH_SHAPES[camera] or frame.dtype != np.uint16:
            raise ValueError(f"{camera} depth does not match its native uint16 dimensions")
        preserved = np.array(frame, copy=True, order="C")
        preserved.setflags(write=False)
        digest = _array_hash(preserved)
        depth[camera] = preserved
        provenance.append(
            FrameProvenance(camera, "depth", timestamp, trial_hash, digest, str(preserved.dtype), preserved.shape)
        )

    observation_hash = _canonical_hash(
        {
            "timestamp": timestamp,
            "status": status,
            "trial_sha256": trial_hash,
            "queued": queued,
            "joints": joints.tolist(),
            "tcp_pose": tcp_pose.tolist(),
            "frames": [record.content_sha256 for record in provenance],
        }
    )
    return NativeObservation(
        timestamp=timestamp,
        status=status,
        trial_sha256=trial_hash,
        queued=queued,
        joints=joints,
        tcp_pose=tcp_pose,
        rgb=rgb,
        depth_mm=depth,
        provenance=tuple(provenance),
        observation_sha256=observation_hash,
    )
