"""Pure native-depth conversion and local validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from camera_calibration_contract import RgbDepthRegistration


@dataclass(frozen=True)
class DepthFrame:
    metres: np.ndarray
    invalid_mask: np.ndarray
    valid_fraction: float
    source_sha256: str


def depth_mm_to_metres(
    depth_mm: np.ndarray,
    *,
    minimum_valid_m: float = 0.05,
    maximum_valid_m: float = 3.0,
) -> DepthFrame:
    source = np.asarray(depth_mm)
    if source.ndim != 2 or source.dtype != np.uint16:
        raise ValueError("native depth must be a 2D uint16 millimetre array")
    if not (0 < minimum_valid_m < maximum_valid_m):
        raise ValueError("depth range is invalid")
    invalid = source == 0
    metres = source.astype(np.float64) / 1000.0
    out_of_range = (~invalid) & ((metres < minimum_valid_m) | (metres > maximum_valid_m))
    if out_of_range.any():
        raise ValueError("depth contains nonzero values outside the supported range")
    metres[invalid] = np.nan
    metres.setflags(write=False)
    invalid = np.array(invalid, copy=True)
    invalid.setflags(write=False)
    digest = hashlib.sha256(source.tobytes(order="C")).hexdigest()
    return DepthFrame(metres, invalid, float((~invalid).mean()), digest)


def local_depth_statistics(
    frame: DepthFrame,
    *,
    center_xy: tuple[int, int],
    radius_px: int,
    minimum_valid_samples: int,
    maximum_discontinuity_m: float,
) -> dict:
    if radius_px < 0 or minimum_valid_samples <= 0 or maximum_discontinuity_m <= 0:
        raise ValueError("local depth thresholds are invalid")
    x, y = center_xy
    height, width = frame.metres.shape
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("depth coordinate is outside the frame")
    x0, x1 = max(0, x - radius_px), min(width, x + radius_px + 1)
    y0, y1 = max(0, y - radius_px), min(height, y + radius_px + 1)
    values = frame.metres[y0:y1, x0:x1]
    valid = values[np.isfinite(values)]
    if valid.size < minimum_valid_samples:
        raise ValueError("insufficient valid local depth samples")
    span = float(valid.max() - valid.min())
    if span > maximum_discontinuity_m:
        raise ValueError("local depth discontinuity exceeds the allowed span")
    return {
        "sample_count": int(valid.size),
        "median_m": float(np.median(valid)),
        "mean_m": float(valid.mean()),
        "minimum_m": float(valid.min()),
        "maximum_m": float(valid.max()),
        "span_m": span,
    }


def plane_residual_statistics(
    points_m: np.ndarray,
    *,
    normal: tuple[float, float, float],
    offset_m: float,
    minimum_samples: int,
) -> dict:
    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("plane points must be finite Nx3 coordinates")
    if points.shape[0] < minimum_samples or minimum_samples < 3:
        raise ValueError("insufficient plane samples")
    vector = np.asarray(normal, dtype=float)
    magnitude = float(np.linalg.norm(vector))
    if not np.isfinite(magnitude) or magnitude <= 0 or not math.isfinite(float(offset_m)):
        raise ValueError("plane equation is invalid")
    distances = np.abs(points @ vector + float(offset_m)) / magnitude
    return {
        "sample_count": int(points.shape[0]),
        "rms_m": float(np.sqrt(np.mean(distances**2))),
        "maximum_m": float(distances.max()),
        "median_m": float(np.median(distances)),
    }


def lookup_registered_rgb_depth(
    frame: DepthFrame,
    *,
    rgb_xy: tuple[int, int],
    mapped_depth_xy: tuple[int, int] | None,
    registration: RgbDepthRegistration | None,
) -> float:
    """Use only a coordinate supplied by a separately verified registration adapter."""
    if registration is None or not registration.verified or not registration.registered:
        raise ValueError("verified RGB/depth registration is required")
    if mapped_depth_xy is None:
        raise ValueError("an explicit registered RGB-to-depth coordinate is required")
    if not all(isinstance(value, int) for value in (*rgb_xy, *mapped_depth_xy)):
        raise ValueError("RGB and depth coordinates must be integer pixels")
    x, y = mapped_depth_xy
    height, width = frame.metres.shape
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("registered depth coordinate is outside the frame")
    value = float(frame.metres[y, x])
    if not math.isfinite(value):
        raise ValueError("registered depth pixel is invalid")
    return value
