"""Typed real-cell camera calibration contracts with no fabricated defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


REAL_NATIVE_RESOLUTIONS = {
    "table_rgb": (1920, 1080),
    "wrist_rgb": (1280, 720),
    "table_depth": (640, 400),
    "wrist_depth": (848, 480),
}


def _integrity_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CameraCalibration:
    stream: str
    domain: str
    frame_name: str
    intrinsics_row_major: tuple[float, ...]
    distortion: tuple[float, ...]
    extrinsics_row_major: tuple[float, ...]
    resolution_wh: tuple[int, int]
    length_unit: str
    angle_unit: str
    calibration_version: str
    compatibility_id: str
    verified: bool

    def payload(self) -> dict:
        return asdict(self)

    def integrity_sha256(self) -> str:
        return _integrity_hash(self.payload())

    def serialize(self) -> dict:
        return {**self.payload(), "integrity_sha256": self.integrity_sha256()}

    @classmethod
    def deserialize(cls, value: dict) -> "CameraCalibration":
        raw = dict(value)
        supplied = raw.pop("integrity_sha256", None)
        for field in ("intrinsics_row_major", "distortion", "extrinsics_row_major", "resolution_wh"):
            raw[field] = tuple(raw[field])
        result = cls(**raw)
        if supplied != result.integrity_sha256():
            raise ValueError("camera calibration integrity hash mismatch")
        return result


@dataclass(frozen=True)
class RgbDepthRegistration:
    rgb_stream: str
    depth_stream: str
    domain: str
    rgb_resolution_wh: tuple[int, int]
    depth_resolution_wh: tuple[int, int]
    compatibility_id: str
    registration_version: str
    registered: bool
    verified: bool
    method: str

    def payload(self) -> dict:
        return asdict(self)

    def integrity_sha256(self) -> str:
        return _integrity_hash(self.payload())


def validate_camera_calibration(
    calibration: CameraCalibration | None,
    *,
    expected_stream: str,
    expected_compatibility_id: str | None = None,
) -> CameraCalibration:
    if calibration is None:
        raise ValueError(f"{expected_stream} calibration is missing")
    if expected_stream not in REAL_NATIVE_RESOLUTIONS:
        raise ValueError("unsupported camera stream")
    if not calibration.verified:
        raise ValueError(f"{expected_stream} calibration is unverified")
    if calibration.domain != "real_cell":
        raise ValueError("simulation calibration cannot be used for the real cell")
    if calibration.stream != expected_stream:
        raise ValueError("camera calibration stream mismatch")
    if tuple(calibration.resolution_wh) != REAL_NATIVE_RESOLUTIONS[expected_stream]:
        raise ValueError("camera calibration resolution mismatch")
    if expected_compatibility_id is not None and calibration.compatibility_id != expected_compatibility_id:
        raise ValueError("camera calibration compatibility mismatch")
    if not calibration.frame_name or not calibration.calibration_version or not calibration.compatibility_id:
        raise ValueError("camera calibration identity is incomplete")
    if calibration.length_unit != "metre" or calibration.angle_unit != "radian":
        raise ValueError("camera calibration units must be metre and radian")
    if len(calibration.intrinsics_row_major) != 9 or len(calibration.extrinsics_row_major) != 16:
        raise ValueError("camera calibration matrix dimensions are invalid")
    values = (*calibration.intrinsics_row_major, *calibration.distortion, *calibration.extrinsics_row_major)
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("camera calibration contains nonfinite values")
    if calibration.intrinsics_row_major[0] <= 0 or calibration.intrinsics_row_major[4] <= 0:
        raise ValueError("camera focal lengths must be positive")
    return calibration


def validate_registration(
    registration: RgbDepthRegistration | None,
    *,
    rgb_calibration: CameraCalibration,
    depth_calibration: CameraCalibration,
) -> RgbDepthRegistration:
    if registration is None or not registration.verified or not registration.registered:
        raise ValueError("verified RGB/depth registration is required")
    if registration.domain != "real_cell":
        raise ValueError("simulation registration cannot be used for the real cell")
    if registration.rgb_stream != rgb_calibration.stream or registration.depth_stream != depth_calibration.stream:
        raise ValueError("RGB/depth registration stream mismatch")
    if registration.rgb_resolution_wh != rgb_calibration.resolution_wh:
        raise ValueError("RGB registration resolution mismatch")
    if registration.depth_resolution_wh != depth_calibration.resolution_wh:
        raise ValueError("depth registration resolution mismatch")
    if not (
        registration.compatibility_id
        == rgb_calibration.compatibility_id
        == depth_calibration.compatibility_id
    ):
        raise ValueError("RGB/depth calibration compatibility mismatch")
    if not registration.method or not registration.registration_version:
        raise ValueError("registration method/version is missing")
    return registration
