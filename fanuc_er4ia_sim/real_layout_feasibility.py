"""Fail-closed, host-only feasibility for externally surveyed geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType

from bin_geometry import (
    Bounds2D,
    SurveyedBinPose,
    bin_exterior_polygon,
    finite_number,
    finite_tuple,
    polygons_overlap,
    rectangle_polygon,
    validate_surveyed_bins,
)
from physical_geometry_profile import (
    PhysicalGeometryProfile,
    REAL_CELL_GEOMETRY_PROFILE,
    resolve_geometry_profile,
)


REAL_CELL_INVENTORY = MappingProxyType({
    "cube_colours": ("red", "red", "blue", "blue", "yellow", "yellow"),
    "vials_used": 2, "caps_used": 2, "bins_used": 3, "bins_available": 5,
})


@dataclass(frozen=True)
class SurveyedObjectPose:
    identifier: str
    kind: str
    center_xy_mm: tuple[float, float]
    yaw_deg: float
    colour: str | None = None
    units: str = "millimetres"

    def validated(self) -> tuple[tuple[float, float], float]:
        if self.units != "millimetres":
            raise ValueError("units must be 'millimetres'")
        if not isinstance(self.identifier, str) or not self.identifier:
            raise ValueError("object identifier is required")
        if self.kind not in {"cube", "vial", "cap"}:
            raise ValueError("supported object kind is required")
        center = finite_tuple(self.center_xy_mm, 2, "object center")
        yaw = finite_number(self.yaw_deg, "object yaw")
        if self.kind == "cube" and self.colour not in {"red", "blue", "yellow"}:
            raise ValueError("cube colour must be red, blue, or yellow")
        if self.kind != "cube" and self.colour is not None:
            raise ValueError("only cubes carry a colour")
        return (center[0], center[1]), yaw

    def validate(self) -> None:
        self.validated()


def _circle_polygon(center: tuple[float, float], radius_mm: float,
                    vertices: int = 24) -> tuple[tuple[float, float], ...]:
    x, y = finite_tuple(center, 2, "circle center")
    radius = finite_number(radius_mm, "circle radius", positive=True)
    if isinstance(vertices, bool) or not isinstance(vertices, int) or vertices < 8:
        raise ValueError("circle vertices must be an integer of at least eight")
    conservative_radius = radius / math.cos(math.pi / vertices)
    return tuple((x + conservative_radius * math.cos(2 * math.pi * i / vertices),
                  y + conservative_radius * math.sin(2 * math.pi * i / vertices))
                 for i in range(vertices))


def _capsule_polygon(center: tuple[float, float], yaw_deg: float,
                     length_mm: float, radius_mm: float,
                     expansion_mm: float = 0.0, arc_steps: int = 12
                     ) -> tuple[tuple[float, float], ...]:
    x, y = finite_tuple(center, 2, "capsule center")
    yaw = finite_number(yaw_deg, "capsule yaw")
    length = finite_number(length_mm, "capsule length", positive=True)
    physical_radius = finite_number(radius_mm, "capsule radius", positive=True)
    expansion = finite_number(expansion_mm, "capsule expansion", nonnegative=True)
    if length < 2 * physical_radius:
        raise ValueError("capsule length must cover its diameter")
    if isinstance(arc_steps, bool) or not isinstance(arc_steps, int) or arc_steps < 4:
        raise ValueError("capsule arc steps must be an integer of at least four")
    conservative_radius = (physical_radius + expansion) / math.cos(math.pi / (2 * arc_steps))
    half_axis = length / 2.0 - physical_radius
    angle = math.radians(yaw)
    axis = math.cos(angle), math.sin(angle)
    points = []
    for sign in (1.0, -1.0):
        end = x + sign * half_axis * axis[0], y + sign * half_axis * axis[1]
        base = angle if sign > 0 else angle + math.pi
        for index in range(arc_steps + 1):
            theta = base - math.pi / 2 + index * math.pi / arc_steps
            points.append((end[0] + conservative_radius * math.cos(theta),
                           end[1] + conservative_radius * math.sin(theta)))
    centroid = (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points))
    return tuple(sorted(points, key=lambda p: math.atan2(p[1] - centroid[1],
                                                         p[0] - centroid[0])))


def object_footprint(pose: SurveyedObjectPose, profile: PhysicalGeometryProfile, *,
                     clearance_mm: float = 0.0) -> tuple[tuple[float, float], ...]:
    if not isinstance(pose, SurveyedObjectPose):
        raise ValueError("pose must be SurveyedObjectPose")
    center, yaw = pose.validated()
    selected = resolve_geometry_profile(profile)
    clearance = finite_number(clearance_mm, "clearance", nonnegative=True)
    if pose.kind == "cube":
        return rectangle_polygon(SurveyedBinPose(center, yaw),
                                 (selected.cube_edge_mm, selected.cube_edge_mm),
                                 expansion_mm=clearance)
    if pose.kind == "cap":
        return _circle_polygon(center, selected.cap_flange_diameter_mm / 2.0 + clearance)
    return _capsule_polygon(center, yaw, selected.vial_length_mm,
                            selected.vial_radius_mm, clearance)


def _inside_bounds(polygon: tuple[tuple[float, float], ...], bounds: Bounds2D) -> bool:
    return all(bounds.contains(point) for point in polygon)


def _inventory_reasons(objects: tuple[SurveyedObjectPose, ...]) -> list[str]:
    reasons: list[str] = []
    if len({pose.identifier for pose in objects}) != len(objects):
        reasons.append("duplicate_object_identifier")
    counts = {kind: sum(p.kind == kind for p in objects) for kind in ("cube", "vial", "cap")}
    if counts != {"cube": 6, "vial": 2, "cap": 2}:
        reasons.append("expected_inventory:6_cubes:2_vials:2_caps")
    colours = {colour: sum(p.kind == "cube" and p.colour == colour for p in objects)
               for colour in ("red", "blue", "yellow")}
    if colours != {"red": 2, "blue": 2, "yellow": 2}:
        reasons.append("expected_cube_inventory:2_red:2_blue:2_yellow")
    return reasons


def _result(reasons: list[str], profile_name: str | None = None) -> dict:
    return {
        "feasible": not reasons, "reasons": reasons, "geometry_profile": profile_name,
        "primitive_bin_geometry_is_cad_perfect": False,
        "all_requirements_verified": False, "production_ready": False,
        "action": None, "trajectory": None, "robot_motion_authorized": False,
    }


def validate_real_layout(
    objects: tuple[SurveyedObjectPose, ...] | None,
    bins: dict[str, SurveyedBinPose] | None,
    *, source_bounds: Bounds2D | None, table_bounds: Bounds2D | None,
    reach_origin_xy_mm: tuple[float, float] | None,
    maximum_reach_mm: float | None, reach_margin_mm: float | None,
    approach_clearance_mm: float | None, carry_clearance_mm: float | None,
    minimum_bin_spacing_mm: float | None, keep_outs: dict[str, Bounds2D] | None,
    geometry_profile: str | PhysicalGeometryProfile | None,
) -> dict:
    """Validate survey data without producing or authorizing an action."""
    reasons: list[str] = []
    profile: PhysicalGeometryProfile | None = None
    try:
        if geometry_profile is None:
            raise ValueError("explicit real-cell geometry profile is required")
        profile = resolve_geometry_profile(geometry_profile)
        if profile.canonical_sha256 != REAL_CELL_GEOMETRY_PROFILE.canonical_sha256:
            reasons.append("real_cell_geometry_profile_required")
    except (TypeError, ValueError) as exc:
        reasons.append(f"invalid:geometry_profile:{exc}")

    valid_objects: list[SurveyedObjectPose] = []
    if not isinstance(objects, tuple):
        reasons.append("missing_or_invalid:source_object_poses")
        objects = ()
    for index, pose in enumerate(objects):
        try:
            if not isinstance(pose, SurveyedObjectPose):
                raise ValueError("pose must be SurveyedObjectPose")
            pose.validate()
            valid_objects.append(pose)
        except (TypeError, ValueError) as exc:
            reasons.append(f"invalid_object_pose:{index}:{exc}")
    reasons.extend(_inventory_reasons(tuple(valid_objects)))

    valid_bounds: dict[str, Bounds2D] = {}
    for name, bounds in (("source_bounds", source_bounds), ("table_bounds", table_bounds)):
        try:
            if not isinstance(bounds, Bounds2D):
                raise ValueError("bounds must be Bounds2D")
            bounds.validate()
            valid_bounds[name] = bounds
        except (TypeError, ValueError) as exc:
            reasons.append(f"invalid:{name}:{exc}")
    try:
        reach_origin = finite_tuple(reach_origin_xy_mm, 2, "reach origin")
    except (TypeError, ValueError) as exc:
        reasons.append(f"invalid:reach_origin_xy_mm:{exc}")
        reach_origin = None
    numeric: dict[str, float] = {}
    for name, value, positive in (
        ("maximum_reach_mm", maximum_reach_mm, True),
        ("reach_margin_mm", reach_margin_mm, False),
        ("approach_clearance_mm", approach_clearance_mm, False),
        ("carry_clearance_mm", carry_clearance_mm, False),
        ("minimum_bin_spacing_mm", minimum_bin_spacing_mm, False),
    ):
        try:
            numeric[name] = finite_number(value, name, positive=positive,
                                          nonnegative=not positive)
        except (TypeError, ValueError) as exc:
            reasons.append(f"invalid:{name}:{exc}")
    if {"maximum_reach_mm", "reach_margin_mm"} <= numeric.keys():
        if numeric["reach_margin_mm"] >= numeric["maximum_reach_mm"]:
            reasons.append("reach_margin_must_be_less_than_maximum_reach")

    valid_keep_outs: dict[str, Bounds2D] = {}
    if not isinstance(keep_outs, dict):
        reasons.append("missing_or_invalid:keep_outs")
        keep_outs = {}
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

    bin_result = validate_surveyed_bins(
        bins, table_bounds=table_bounds, reach_origin_xy_mm=reach_origin_xy_mm,
        maximum_reach_mm=maximum_reach_mm, reach_margin_mm=reach_margin_mm,
        minimum_edge_spacing_mm=minimum_bin_spacing_mm, keep_outs=keep_outs,
    )
    reasons.extend(f"bin:{reason}" for reason in bin_result["reasons"])

    object_polygons: dict[str, tuple[tuple[float, float], ...]] = {}
    geometry_ready = (
        profile is not None and "approach_clearance_mm" in numeric
        and "source_bounds" in valid_bounds and "table_bounds" in valid_bounds
        and reach_origin is not None
        and {"maximum_reach_mm", "reach_margin_mm"} <= numeric.keys()
    )
    if geometry_ready:
        assert profile is not None and reach_origin is not None
        for pose in valid_objects:
            try:
                polygon = object_footprint(
                    pose, profile, clearance_mm=numeric["approach_clearance_mm"]
                )
            except (TypeError, ValueError) as exc:
                reasons.append(f"invalid_object_geometry:{pose.identifier}:{exc}")
                continue
            object_polygons[pose.identifier] = polygon
            if not _inside_bounds(polygon, valid_bounds["source_bounds"]):
                reasons.append(f"{pose.identifier}:outside_source_bounds")
            if not _inside_bounds(polygon, valid_bounds["table_bounds"]):
                reasons.append(f"{pose.identifier}:outside_table_bounds")
            limit = numeric["maximum_reach_mm"] - numeric["reach_margin_mm"]
            if max(math.hypot(p[0] - reach_origin[0], p[1] - reach_origin[1])
                   for p in polygon) > limit:
                reasons.append(f"{pose.identifier}:outside_reach")
            for keep_name, bounds in valid_keep_outs.items():
                if polygons_overlap(polygon, bounds.polygon()):
                    reasons.append(f"{pose.identifier}:overlaps_keep_out:{keep_name}")

    names = sorted(object_polygons)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if polygons_overlap(object_polygons[first], object_polygons[second]):
                reasons.append(f"object_clearance:{first}:{second}")

    valid_bin_polygons: dict[str, tuple[tuple[float, float], ...]] = {}
    if isinstance(bins, dict) and "carry_clearance_mm" in numeric:
        for name, pose in bins.items():
            try:
                if not isinstance(pose, SurveyedBinPose):
                    raise ValueError("pose must be SurveyedBinPose")
                pose.validate()
                valid_bin_polygons[name] = bin_exterior_polygon(
                    pose, expansion_mm=numeric["carry_clearance_mm"]
                )
            except (TypeError, ValueError):
                continue
    for object_name, polygon in object_polygons.items():
        for bin_name, bin_polygon in valid_bin_polygons.items():
            if polygons_overlap(polygon, bin_polygon):
                reasons.append(f"object_bin_clearance:{object_name}:{bin_name}")
    return _result(reasons, None if profile is None else profile.name)
