"""Verified cube-yaw to FANUC TCP-R mapping with integrity protection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


def normalize_angle_deg(angle_deg: float) -> float:
    if not math.isfinite(float(angle_deg)):
        raise ValueError("angle must be finite")
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def angular_error_deg(first: float, second: float) -> float:
    return abs(normalize_angle_deg(float(first) - float(second)))


@dataclass(frozen=True)
class OrientationSample:
    cube_face_yaw_deg: float
    aligned_tcp_r_deg: float


@dataclass(frozen=True)
class TcpOrientationCalibration:
    sign: int | None = None
    offset_deg: float | None = None
    uframe: str | None = None
    utool: str | None = None
    verified: bool = False
    maximum_fit_error_deg: float | None = None
    samples: tuple[OrientationSample, ...] = ()
    schema_version: str = "fanuc-tcp-orientation-v1"

    def payload(self) -> dict:
        return asdict(self)

    def integrity_hash(self) -> str:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def serialize(self) -> dict:
        return {**self.payload(), "integrity_sha256": self.integrity_hash()}

    @classmethod
    def deserialize(cls, value: dict) -> "TcpOrientationCalibration":
        raw = dict(value)
        supplied_hash = raw.pop("integrity_sha256", None)
        raw["samples"] = tuple(OrientationSample(**sample) for sample in raw.get("samples", ()))
        result = cls(**raw)
        if supplied_hash != result.integrity_hash():
            raise ValueError("TCP orientation calibration integrity hash mismatch")
        return result


def _circular_mean_deg(values: list[float]) -> float:
    radians = [math.radians(value) for value in values]
    return normalize_angle_deg(
        math.degrees(
            math.atan2(
                sum(math.sin(value) for value in radians),
                sum(math.cos(value) for value in radians),
            )
        )
    )


def fit_orientation_calibration(
    samples: list[OrientationSample],
    *,
    uframe: str,
    utool: str,
    maximum_allowed_error_deg: float = 0.5,
) -> TcpOrientationCalibration:
    if len(samples) < 3:
        raise ValueError("at least three CSI alignment samples are required")
    if not uframe or not utool:
        raise ValueError("UFRAME and UTOOL identities are required")
    if any(
        not math.isfinite(sample.cube_face_yaw_deg)
        or not math.isfinite(sample.aligned_tcp_r_deg)
        for sample in samples
    ):
        raise ValueError("calibration samples must be finite")
    unique = {round(normalize_angle_deg(sample.cube_face_yaw_deg), 6) for sample in samples}
    if len(unique) < 3:
        raise ValueError("calibration samples must cover three distinct cube yaws")
    candidates = []
    for sign in (-1, 1):
        offsets = [
            normalize_angle_deg(sample.aligned_tcp_r_deg - sign * sample.cube_face_yaw_deg)
            for sample in samples
        ]
        offset = _circular_mean_deg(offsets)
        errors = [
            angular_error_deg(
                sample.aligned_tcp_r_deg,
                sign * sample.cube_face_yaw_deg + offset,
            )
            for sample in samples
        ]
        candidates.append((max(errors), sign, offset))
    maximum_error, sign, offset = min(candidates)
    return TcpOrientationCalibration(
        sign=sign,
        offset_deg=offset,
        uframe=uframe,
        utool=utool,
        verified=maximum_error <= maximum_allowed_error_deg,
        maximum_fit_error_deg=maximum_error,
        samples=tuple(samples),
    )


def tcp_r_for_cube_yaw(
    cube_yaw_deg: float,
    calibration: TcpOrientationCalibration,
    *,
    active_uframe: str,
    active_utool: str,
) -> float:
    if not calibration.verified:
        raise ValueError("TCP orientation mapping is not verified")
    if calibration.sign not in (-1, 1) or calibration.offset_deg is None:
        raise ValueError("TCP orientation mapping is incomplete")
    if active_uframe != calibration.uframe:
        raise ValueError("active UFRAME does not match the verified calibration")
    if active_utool != calibration.utool:
        raise ValueError("active UTOOL does not match the verified calibration")
    return normalize_angle_deg(
        calibration.sign * float(cube_yaw_deg) + float(calibration.offset_deg)
    )


UNVERIFIED_TCP_ORIENTATION = TcpOrientationCalibration()
