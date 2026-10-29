"""Pure geometry checks for externally surveyed physical bin coordinates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class AxisAlignedBounds:
    minimum_x_mm: float
    maximum_x_mm: float
    minimum_y_mm: float
    maximum_y_mm: float

    def contains_box(self, center: tuple[float, float], half_extent_mm: float) -> bool:
        x, y = center
        return bool(
            x - half_extent_mm >= self.minimum_x_mm
            and x + half_extent_mm <= self.maximum_x_mm
            and y - half_extent_mm >= self.minimum_y_mm
            and y + half_extent_mm <= self.maximum_y_mm
        )


@dataclass(frozen=True)
class BinGeometry:
    outer_x_mm: float = 50.0
    outer_y_mm: float = 50.0
    height_mm: float = 55.0
    pocket_x_mm: float = 35.0
    pocket_y_mm: float = 35.0
    pocket_depth_mm: float = 25.0


@dataclass(frozen=True)
class KeepOut:
    name: str
    bounds: AxisAlignedBounds


def _boxes_overlap(
    center: tuple[float, float], half_x: float, half_y: float, bounds: AxisAlignedBounds
) -> bool:
    x, y = center
    return not (
        x + half_x <= bounds.minimum_x_mm
        or x - half_x >= bounds.maximum_x_mm
        or y + half_y <= bounds.minimum_y_mm
        or y - half_y >= bounds.maximum_y_mm
    )


def validate_bin_layout(
    centres_mm: dict[str, tuple[float, float]],
    *,
    table_bounds: AxisAlignedBounds,
    keep_outs: tuple[KeepOut, ...] = (),
    camera_visible: dict[str, bool] | None,
    maximum_reach_mm: float,
    reach_margin_mm: float,
    minimum_center_spacing_mm: float = 80.0,
    geometry: BinGeometry = BinGeometry(),
) -> dict:
    reasons: list[str] = []
    expected = {"bin_1", "bin_2", "bin_3"}
    if set(centres_mm) != expected:
        reasons.append("expected_exactly_bin_1_bin_2_bin_3")
    if camera_visible is None:
        reasons.append("camera_visibility_unverified")
    for name, center in centres_mm.items():
        if len(center) != 2 or not all(math.isfinite(float(value)) for value in center):
            reasons.append(f"{name}:invalid_center")
            continue
        if not table_bounds.contains_box(center, max(geometry.outer_x_mm, geometry.outer_y_mm) / 2.0):
            reasons.append(f"{name}:footprint_outside_table")
        if math.hypot(*center) > maximum_reach_mm - reach_margin_mm:
            reasons.append(f"{name}:insufficient_reach_margin")
        if camera_visible is not None and camera_visible.get(name) is not True:
            reasons.append(f"{name}:not_camera_visible")
        for keep_out in keep_outs:
            if _boxes_overlap(
                center,
                geometry.outer_x_mm / 2.0,
                geometry.outer_y_mm / 2.0,
                keep_out.bounds,
            ):
                reasons.append(f"{name}:overlaps_keep_out:{keep_out.name}")
    names = sorted(centres_mm)
    spacings = {}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            distance = math.dist(centres_mm[first], centres_mm[second])
            spacings[f"{first}:{second}"] = distance
            if distance < minimum_center_spacing_mm:
                reasons.append(f"insufficient_spacing:{first}:{second}")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "centres_mm": {key: list(value) for key, value in centres_mm.items()},
        "center_spacings_mm": spacings,
        "geometry": asdict(geometry),
    }


def pocket_center_release_target(
    bin_center_mm: tuple[float, float],
    *,
    table_surface_z_uf0_mm: float,
    release_clearance_above_rim_mm: float,
    geometry: BinGeometry = BinGeometry(),
) -> tuple[float, float, float]:
    if release_clearance_above_rim_mm <= 0:
        raise ValueError("release clearance must be positive")
    if not all(math.isfinite(float(value)) for value in (*bin_center_mm, table_surface_z_uf0_mm)):
        raise ValueError("release target inputs must be finite")
    return (
        float(bin_center_mm[0]),
        float(bin_center_mm[1]),
        float(table_surface_z_uf0_mm)
        + geometry.height_mm
        + float(release_clearance_above_rim_mm),
    )
