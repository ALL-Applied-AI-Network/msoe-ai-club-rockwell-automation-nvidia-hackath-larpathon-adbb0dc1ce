"""Strict host geometry for unplaced bins and externally surveyed poses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from physical_geometry_profile import BinDimensions, REAL_CELL_GEOMETRY_PROFILE


REAL_BIN_DIMENSIONS = REAL_CELL_GEOMETRY_PROFILE.bin_dimensions
assert REAL_BIN_DIMENSIONS is not None


def finite_number(value: object, name: str, *, positive: bool = False,
                  nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def finite_tuple(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-value tuple")
    return tuple(finite_number(item, f"{name}[{index}]") for index, item in enumerate(value))


def _require_units(units: object) -> None:
    if units != "millimetres":
        raise ValueError("units must be 'millimetres'")


@dataclass(frozen=True)
class SurveyedBinPose:
    center_xy_mm: tuple[float, float]
    yaw_deg: float
    units: str = "millimetres"

    def validated(self) -> tuple[tuple[float, float], float]:
        _require_units(self.units)
        center = finite_tuple(self.center_xy_mm, 2, "surveyed bin center")
        yaw = finite_number(self.yaw_deg, "surveyed bin yaw")
        return (center[0], center[1]), yaw

    def validate(self) -> None:
        self.validated()


@dataclass(frozen=True)
class Bounds2D:
    minimum_x_mm: float
    maximum_x_mm: float
    minimum_y_mm: float
    maximum_y_mm: float
    units: str = "millimetres"

    def validated(self) -> tuple[float, float, float, float]:
        _require_units(self.units)
        values = tuple(
            finite_number(value, name)
            for name, value in (
                ("minimum_x_mm", self.minimum_x_mm), ("maximum_x_mm", self.maximum_x_mm),
                ("minimum_y_mm", self.minimum_y_mm), ("maximum_y_mm", self.maximum_y_mm),
            )
        )
        if values[0] >= values[1] or values[2] >= values[3]:
            raise ValueError("bounds must have positive area")
        return values

    def validate(self) -> None:
        self.validated()

    def contains(self, point: tuple[float, float]) -> bool:
        minimum_x, maximum_x, minimum_y, maximum_y = self.validated()
        try:
            x, y = finite_tuple(point, 2, "point")
        except ValueError:
            return False
        return minimum_x <= x <= maximum_x and minimum_y <= y <= maximum_y

    def polygon(self) -> tuple[tuple[float, float], ...]:
        minimum_x, maximum_x, minimum_y, maximum_y = self.validated()
        return ((minimum_x, minimum_y), (maximum_x, minimum_y),
                (maximum_x, maximum_y), (minimum_x, maximum_y))


@dataclass(frozen=True)
class ReleaseVerticalReference:
    table_z_mm: float
    rim_top_z_mm: float
    workspace_min_z_mm: float
    workspace_max_z_mm: float
    reach_min_z_mm: float
    reach_max_z_mm: float
    units: str = "millimetres"

    def validated(self) -> tuple[float, float, float, float, float, float]:
        _require_units(self.units)
        values = tuple(
            finite_number(getattr(self, item), item)
            for item in (
                "table_z_mm", "rim_top_z_mm", "workspace_min_z_mm",
                "workspace_max_z_mm", "reach_min_z_mm", "reach_max_z_mm",
            )
        )
        table, rim, workspace_min, workspace_max, reach_min, reach_max = values
        if rim <= table:
            raise ValueError("surveyed rim top must be above the surveyed table")
        if workspace_min >= workspace_max or reach_min >= reach_max:
            raise ValueError("vertical workspace and reach bounds must have positive range")
        return values


def _validated_dimensions(dimensions: object) -> BinDimensions:
    if not isinstance(dimensions, BinDimensions):
        raise ValueError("dimensions must be BinDimensions")
    dimensions.validate()
    return dimensions


def bin_geometry_contract(dimensions: BinDimensions = REAL_BIN_DIMENSIONS) -> dict:
    dimensions = _validated_dimensions(dimensions)
    return {
        "units": "millimetres", "placed": False, "center_xy_mm": None,
        "yaw_deg": None, "release_height_mm": None,
        "exterior": [dimensions.outer_x_mm, dimensions.outer_y_mm, dimensions.outer_height_mm],
        "pocket": [dimensions.pocket_x_mm, dimensions.pocket_y_mm, dimensions.pocket_depth_mm],
        "rim_thickness_xy_mm": [
            (dimensions.outer_x_mm - dimensions.pocket_x_mm) / 2.0,
            (dimensions.outer_y_mm - dimensions.pocket_y_mm) / 2.0,
        ],
        "primitive_geometry_is_cad_perfect": False,
    }


def _rotate(point: tuple[float, float], yaw_deg: float) -> tuple[float, float]:
    angle = math.radians(yaw_deg)
    c, s = math.cos(angle), math.sin(angle)
    return c * point[0] - s * point[1], s * point[0] + c * point[1]


def rectangle_polygon(pose: SurveyedBinPose, size_xy_mm: tuple[float, float], *,
                      expansion_mm: float = 0.0) -> tuple[tuple[float, float], ...]:
    if not isinstance(pose, SurveyedBinPose):
        raise ValueError("pose must be SurveyedBinPose")
    center, yaw = pose.validated()
    size_x, size_y = finite_tuple(size_xy_mm, 2, "rectangle size")
    if size_x <= 0 or size_y <= 0:
        raise ValueError("rectangle dimensions must be positive")
    expansion = finite_number(expansion_mm, "rectangle expansion", nonnegative=True)
    half_x, half_y = size_x / 2.0 + expansion, size_y / 2.0 + expansion
    corners = ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y))
    return tuple((center[0] + _rotate(point, yaw)[0], center[1] + _rotate(point, yaw)[1])
                 for point in corners)


def bin_exterior_polygon(pose: SurveyedBinPose, dimensions: BinDimensions = REAL_BIN_DIMENSIONS,
                         *, expansion_mm: float = 0.0) -> tuple[tuple[float, float], ...]:
    dimensions = _validated_dimensions(dimensions)
    return rectangle_polygon(pose, (dimensions.outer_x_mm, dimensions.outer_y_mm),
                             expansion_mm=expansion_mm)


def bin_pocket_polygon(pose: SurveyedBinPose, dimensions: BinDimensions = REAL_BIN_DIMENSIONS,
                       *, safety_inset_mm: float = 0.0) -> tuple[tuple[float, float], ...]:
    dimensions = _validated_dimensions(dimensions)
    inset = finite_number(safety_inset_mm, "pocket safety inset", nonnegative=True)
    if 2 * inset >= min(dimensions.pocket_x_mm, dimensions.pocket_y_mm):
        raise ValueError("pocket safety inset leaves no usable region")
    return rectangle_polygon(
        pose, (dimensions.pocket_x_mm - 2 * inset, dimensions.pocket_y_mm - 2 * inset)
    )


def _point_inside_pocket_xy(pose: SurveyedBinPose, point_xy_mm: tuple[float, float], *,
                            safety_inset_mm: float, dimensions: BinDimensions) -> bool:
    dimensions = _validated_dimensions(dimensions)
    center, yaw = pose.validated()
    x, y = finite_tuple(point_xy_mm, 2, "pocket point")
    inset = finite_number(safety_inset_mm, "pocket safety inset", nonnegative=True)
    usable_x = dimensions.pocket_x_mm / 2.0 - inset
    usable_y = dimensions.pocket_y_mm / 2.0 - inset
    if usable_x <= 0 or usable_y <= 0:
        raise ValueError("pocket safety inset leaves no usable region")
    local = _rotate((x - center[0], y - center[1]), -yaw)
    return abs(local[0]) <= usable_x and abs(local[1]) <= usable_y


def cube_fully_inside_pocket(pose: SurveyedBinPose | None,
                             cube_center_xy_mm: tuple[float, float], cube_yaw_deg: float, *,
                             cube_edge_mm: float, safety_inset_mm: float = 0.0,
                             dimensions: BinDimensions = REAL_BIN_DIMENSIONS) -> bool:
    try:
        if not isinstance(pose, SurveyedBinPose):
            return False
        dimensions = _validated_dimensions(dimensions)
        center = finite_tuple(cube_center_xy_mm, 2, "cube center")
        yaw = finite_number(cube_yaw_deg, "cube yaw")
        edge = finite_number(cube_edge_mm, "cube edge", positive=True)
        inset = finite_number(safety_inset_mm, "pocket safety inset", nonnegative=True)
        usable = bin_pocket_polygon(pose, dimensions, safety_inset_mm=inset)
        del usable
        cube = rectangle_polygon(SurveyedBinPose((center[0], center[1]), yaw), (edge, edge))
        bin_center, bin_yaw = pose.validated()
        usable_x = dimensions.pocket_x_mm / 2.0 - inset
        usable_y = dimensions.pocket_y_mm / 2.0 - inset
        return all(
            abs(local[0]) <= usable_x and abs(local[1]) <= usable_y
            for local in (
                _rotate((point[0] - bin_center[0], point[1] - bin_center[1]), -bin_yaw)
                for point in cube
            )
        )
    except (TypeError, ValueError):
        return False


def validate_cube_release_target(
    pose: SurveyedBinPose | None,
    target_xyz_mm: tuple[float, float, float] | None,
    *, cube_yaw_deg: float, cube_edge_mm: float, safety_inset_mm: float,
    minimum_above_rim_clearance_mm: float,
    vertical_reference: ReleaseVerticalReference | None,
    dimensions: BinDimensions = REAL_BIN_DIMENSIONS,
) -> dict:
    reasons: list[str] = []
    try:
        if not isinstance(pose, SurveyedBinPose):
            raise ValueError("surveyed bin pose is required")
        if target_xyz_mm is None:
            raise ValueError("3D release target with explicit Z is required")
        x, y, z = finite_tuple(target_xyz_mm, 3, "release target")
        if not isinstance(vertical_reference, ReleaseVerticalReference):
            raise ValueError("surveyed table/rim vertical reference is required")
        table, rim, workspace_min, workspace_max, reach_min, reach_max = vertical_reference.validated()
        del table
        clearance = finite_number(
            minimum_above_rim_clearance_mm, "minimum above-rim clearance", nonnegative=True
        )
        if not _point_inside_pocket_xy(
            pose, (x, y), safety_inset_mm=safety_inset_mm,
            dimensions=_validated_dimensions(dimensions),
        ):
            reasons.append("release_xy_outside_pocket")
        if not cube_fully_inside_pocket(
            pose, (x, y), cube_yaw_deg, cube_edge_mm=cube_edge_mm,
            safety_inset_mm=safety_inset_mm, dimensions=dimensions,
        ):
            reasons.append("cube_footprint_outside_pocket")
        if z < rim + clearance:
            reasons.append("release_z_below_rim_clearance")
        if not workspace_min <= z <= workspace_max:
            reasons.append("release_z_outside_workspace")
        if not reach_min <= z <= reach_max:
            reasons.append("release_z_outside_reach")
    except (TypeError, ValueError) as exc:
        reasons.append(f"invalid_release_target:{exc}")
    return {
        "geometry_valid": not reasons, "reasons": reasons,
        "all_requirements_verified": False, "production_ready": False,
        "action": None, "trajectory": None, "robot_motion_authorized": False,
    }


def _axes(polygon: tuple[tuple[float, float], ...]):
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        edge = second[0] - first[0], second[1] - first[1]
        length = math.hypot(*edge)
        if length > 0:
            yield -edge[1] / length, edge[0] / length


def polygons_overlap(first: tuple[tuple[float, float], ...],
                     second: tuple[tuple[float, float], ...]) -> bool:
    for axis in tuple(_axes(first)) + tuple(_axes(second)):
        a = [point[0] * axis[0] + point[1] * axis[1] for point in first]
        b = [point[0] * axis[0] + point[1] * axis[1] for point in second]
        if max(a) <= min(b) or max(b) <= min(a):
            return False
    return True


def _bin_result(reasons: list[str]) -> dict:
    return {
        "valid": not reasons, "reasons": reasons,
        "primitive_geometry_is_cad_perfect": False,
        "all_requirements_verified": False, "production_ready": False,
        "action": None, "trajectory": None, "robot_motion_authorized": False,
    }


def validate_surveyed_bins(
    poses: dict[str, SurveyedBinPose] | None, *, table_bounds: Bounds2D | None,
    reach_origin_xy_mm: tuple[float, float] | None, maximum_reach_mm: float | None,
    reach_margin_mm: float | None, minimum_edge_spacing_mm: float | None,
    keep_outs: dict[str, Bounds2D] | None,
    dimensions: BinDimensions = REAL_BIN_DIMENSIONS,
) -> dict:
    reasons: list[str] = []
    valid_poses: dict[str, SurveyedBinPose] = {}
    try:
        dimensions = _validated_dimensions(dimensions)
    except (TypeError, ValueError) as exc:
        return _bin_result([f"invalid:bin_dimensions:{exc}"])
    if not isinstance(poses, dict):
        reasons.append("missing_or_invalid:surveyed_bin_poses")
        poses = {}
    if set(poses) != {"bin_1", "bin_2", "bin_3"}:
        reasons.append("expected_exactly_three_surveyed_bins")
    valid_table_bounds = False
    if not isinstance(table_bounds, Bounds2D):
        reasons.append("missing_or_invalid:table_bounds")
    else:
        try:
            table_bounds.validate()
            valid_table_bounds = True
        except ValueError as exc:
            reasons.append(f"invalid:table_bounds:{exc}")
    try:
        reach_origin = finite_tuple(reach_origin_xy_mm, 2, "reach origin")
    except (TypeError, ValueError) as exc:
        reasons.append(f"invalid:reach_origin_xy_mm:{exc}")
        reach_origin = None
    numeric: dict[str, float] = {}
    for name, value, positive in (
        ("maximum_reach_mm", maximum_reach_mm, True),
        ("reach_margin_mm", reach_margin_mm, False),
        ("minimum_edge_spacing_mm", minimum_edge_spacing_mm, False),
    ):
        try:
            numeric[name] = finite_number(value, name, positive=positive, nonnegative=not positive)
        except (TypeError, ValueError) as exc:
            reasons.append(f"invalid:{name}:{exc}")
    if {"maximum_reach_mm", "reach_margin_mm"} <= numeric.keys():
        if numeric["reach_margin_mm"] >= numeric["maximum_reach_mm"]:
            reasons.append("reach_margin_must_be_less_than_maximum_reach")
    valid_keep_outs: dict[str, Bounds2D] = {}
    if not isinstance(keep_outs, dict):
        reasons.append("missing_or_invalid:keep_outs")
    else:
        for name, bounds in keep_outs.items():
            try:
                if not isinstance(name, str) or not name:
                    raise ValueError("keep-out name must be nonempty")
                if not isinstance(bounds, Bounds2D):
                    raise ValueError("keep-out must be Bounds2D")
                bounds.validate()
                valid_keep_outs[name] = bounds
            except (TypeError, ValueError) as exc:
                reasons.append(f"invalid_keep_out:{name}:{exc}")
    polygons: dict[str, tuple[tuple[float, float], ...]] = {}
    for name, pose in poses.items():
        try:
            if not isinstance(pose, SurveyedBinPose):
                raise ValueError("pose must be SurveyedBinPose")
            pose.validate()
            polygon = bin_exterior_polygon(pose, dimensions)
            valid_poses[name] = pose
            polygons[name] = polygon
        except (TypeError, ValueError) as exc:
            reasons.append(f"invalid_bin_pose:{name}:{exc}")
            continue
        if valid_table_bounds and not all(table_bounds.contains(p) for p in polygon):
            reasons.append(f"{name}:outside_table_bounds")
        if reach_origin is not None and {"maximum_reach_mm", "reach_margin_mm"} <= numeric.keys():
            limit = numeric["maximum_reach_mm"] - numeric["reach_margin_mm"]
            if max(math.hypot(p[0] - reach_origin[0], p[1] - reach_origin[1]) for p in polygon) > limit:
                reasons.append(f"{name}:outside_reach")
        for keep_name, bounds in valid_keep_outs.items():
            if polygons_overlap(polygon, bounds.polygon()):
                reasons.append(f"{name}:overlaps_keep_out:{keep_name}")
    if "minimum_edge_spacing_mm" in numeric:
        names = sorted(polygons)
        for index, first in enumerate(names):
            expanded = bin_exterior_polygon(
                valid_poses[first], dimensions,
                expansion_mm=numeric["minimum_edge_spacing_mm"] / 2,
            )
            for second in names[index + 1:]:
                other = bin_exterior_polygon(
                    valid_poses[second], dimensions,
                    expansion_mm=numeric["minimum_edge_spacing_mm"] / 2,
                )
                if polygons_overlap(expanded, other):
                    reasons.append(f"insufficient_spacing:{first}:{second}")
    return _bin_result(reasons)
