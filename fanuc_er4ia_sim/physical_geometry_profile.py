"""Immutable opt-in geometry profiles with asset-bound canonical identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from pathlib import PurePosixPath
from types import MappingProxyType


VIAL_ASSET_REPOSITORY_PATH = "fanuc_er4ia_sim/assets/usd_assets/vial.usda"
VIAL_ASSET_SHA256 = "8b0b4470e13f3c401ce77b0cea8d5fb72d8c76a0619c97cd2567712dc294ac32"
LEGACY_CAP_PLUG_TIP_RADIUS_M = float.fromhex("0x1.553ef6b5d462cp-7")


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class BinDimensions:
    outer_x_mm: float
    outer_y_mm: float
    outer_height_mm: float
    pocket_x_mm: float
    pocket_y_mm: float
    pocket_depth_mm: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for item in fields(self):
            _finite_positive(getattr(self, item.name), item.name)
        if self.pocket_x_mm >= self.outer_x_mm or self.pocket_y_mm >= self.outer_y_mm:
            raise ValueError("bin pocket must be strictly inside the exterior footprint")
        if self.pocket_depth_mm > self.outer_height_mm:
            raise ValueError("bin pocket cannot be deeper than the exterior height")


@dataclass(frozen=True)
class VialAssetConvention:
    convention_id: str
    asset_path: str
    asset_sha256: str
    meters_per_authored_unit: float
    authored_outer_diameter_units: float
    authored_length_units: float
    scale_reference_diameter_mm: float
    scale_reference_length_mm: float
    authored_extent: tuple[tuple[float, float, float], tuple[float, float, float]]
    radial_axes: tuple[str, str]
    long_axis: str
    origin_convention: str
    visual_root_rotations_deg: tuple[tuple[str, float | str], ...]
    collision_long_axis: str
    visual_to_collision_axis_mapping: str
    target_diameter_mm: float
    target_length_mm: float
    radial_scale: float
    axial_scale: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in (
            "convention_id", "asset_path", "asset_sha256", "long_axis",
            "origin_convention", "collision_long_axis", "visual_to_collision_axis_mapping",
        ):
            _nonempty(getattr(self, name), name)
        if len(self.asset_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.asset_sha256):
            raise ValueError("asset_sha256 must be a lowercase SHA-256")
        for name in (
            "meters_per_authored_unit", "authored_outer_diameter_units",
            "authored_length_units", "scale_reference_diameter_mm",
            "scale_reference_length_mm", "target_diameter_mm", "target_length_mm",
            "radial_scale", "axial_scale",
        ):
            _finite_positive(getattr(self, name), name)
        if self.radial_axes != ("X", "Y") or self.long_axis != "Z":
            raise ValueError("vial asset must use radial X/Y axes and long Z axis")
        if self.collision_long_axis != "Z":
            raise ValueError("vial collision long axis must be local Z")
        if self.origin_convention != "bottom_center":
            raise ValueError("unsupported vial origin convention")
        if self.visual_root_rotations_deg != (("Z", "surveyed_yaw"), ("Y", 90.0)):
            raise ValueError("unsupported visual root rotation mapping")
        if self.visual_to_collision_axis_mapping != "shared_root_local_Z":
            raise ValueError("unsupported visual-to-collision mapping")
        asset_path = PurePosixPath(self.asset_path)
        if asset_path.is_absolute() or ".." in asset_path.parts:
            raise ValueError("asset_path must be repository-relative")
        authored_diameter_mm = (
            self.authored_outer_diameter_units * self.meters_per_authored_unit * 1000.0
        )
        authored_length_mm = self.authored_length_units * self.meters_per_authored_unit * 1000.0
        if round(authored_diameter_mm, 2) != self.scale_reference_diameter_mm:
            raise ValueError("diameter scale reference must match rounded authored metadata")
        if not math.isclose(authored_length_mm, self.scale_reference_length_mm,
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("length scale reference must match authored metadata")
        if self.radial_scale != self.target_diameter_mm / self.scale_reference_diameter_mm:
            raise ValueError("radial scale must match target/reference diameter")
        if self.axial_scale != self.target_length_mm / self.scale_reference_length_mm:
            raise ValueError("axial scale must match target/reference length")
        if len(self.authored_extent) != 2 or any(len(point) != 3 for point in self.authored_extent):
            raise ValueError("authored extent must contain two XYZ points")
        for point in self.authored_extent:
            for value in point:
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                    raise ValueError("authored extent must be finite")


def _vial_convention(target_diameter_mm: float, target_length_mm: float) -> VialAssetConvention:
    return VialAssetConvention(
        convention_id="csi-vial-usda-inch-bottom-z-v1",
        asset_path=VIAL_ASSET_REPOSITORY_PATH,
        asset_sha256=VIAL_ASSET_SHA256,
        meters_per_authored_unit=0.0254,
        authored_outer_diameter_units=1.0700000552680549,
        authored_length_units=7.9921259842519685,
        scale_reference_diameter_mm=27.18,
        scale_reference_length_mm=203.0,
        authored_extent=((-0.535, -0.535, 0.0), (0.535, 0.535, 7.992126)),
        radial_axes=("X", "Y"),
        long_axis="Z",
        origin_convention="bottom_center",
        visual_root_rotations_deg=(("Z", "surveyed_yaw"), ("Y", 90.0)),
        collision_long_axis="Z",
        visual_to_collision_axis_mapping="shared_root_local_Z",
        target_diameter_mm=target_diameter_mm,
        target_length_mm=target_length_mm,
        radial_scale=target_diameter_mm / 27.18,
        axial_scale=target_length_mm / 203.0,
    )


@dataclass(frozen=True)
class PhysicalGeometryProfile:
    name: str
    semantic_version: str
    compatibility_id: str
    units: str
    source: str
    ownership: str
    cube_edge_mm: float
    vial_diameter_mm: float
    vial_length_mm: float
    cap_flange_diameter_mm: float
    cap_height_mm: float
    cap_flange_thickness_mm: float
    cap_plug_base_diameter_mm: float
    cap_plug_tip_diameter_mm: float
    bin_dimensions: BinDimensions | None
    vial_asset: VialAssetConvention
    canonical_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("name", "semantic_version", "compatibility_id", "source", "ownership"):
            _nonempty(getattr(self, name), name)
        if self.units != "millimetres":
            raise ValueError("geometry profiles use millimetres")
        for name in (
            "cube_edge_mm", "vial_diameter_mm", "vial_length_mm",
            "cap_flange_diameter_mm", "cap_height_mm", "cap_flange_thickness_mm",
            "cap_plug_base_diameter_mm", "cap_plug_tip_diameter_mm",
        ):
            _finite_positive(getattr(self, name), name)
        if self.bin_dimensions is not None:
            if not isinstance(self.bin_dimensions, BinDimensions):
                raise ValueError("bin_dimensions must be BinDimensions or None")
            self.bin_dimensions.validate()
        if not isinstance(self.vial_asset, VialAssetConvention):
            raise ValueError("vial_asset must be VialAssetConvention")
        self.vial_asset.validate()
        if self.vial_asset.target_diameter_mm != self.vial_diameter_mm:
            raise ValueError("vial asset target diameter must match profile")
        if self.vial_asset.target_length_mm != self.vial_length_mm:
            raise ValueError("vial asset target length must match profile")
        object.__setattr__(self, "canonical_sha256", _canonical_sha256(profile_canonical_payload(self)))

    @property
    def vial_radius_mm(self) -> float:
        return self.vial_diameter_mm / 2.0

    def dimensions_metres(self) -> dict[str, float]:
        return {
            "cube_edge": self.cube_edge_mm / 1000.0,
            "vial_radius": self.vial_radius_mm / 1000.0,
            "vial_length": self.vial_length_mm / 1000.0,
            "cap_radius": self.cap_flange_diameter_mm / 2000.0,
            "cap_height": self.cap_height_mm / 1000.0,
            "cap_flange_thickness": self.cap_flange_thickness_mm / 1000.0,
            "cap_plug_base_radius": self.cap_plug_base_diameter_mm / 2000.0,
            "cap_plug_tip_radius": self.cap_plug_tip_diameter_mm / 2000.0,
        }


def profile_canonical_payload(profile: PhysicalGeometryProfile) -> dict:
    payload = {
        item.name: getattr(profile, item.name)
        for item in fields(profile)
        if item.name != "canonical_sha256"
    }
    payload["bin_dimensions"] = None if profile.bin_dimensions is None else asdict(profile.bin_dimensions)
    payload["vial_asset"] = asdict(profile.vial_asset)
    return payload


_CAP_TIP_DIAMETER_MM_EXACT = float.fromhex("0x1.4d3f7ced91687p+4")

HISTORICAL_GEOMETRY_PROFILE = PhysicalGeometryProfile(
    name="historical_scaled_vial", semantic_version="1.1.0",
    compatibility_id="fanuc-starter-historical-geometry-v1", units="millimetres",
    source="team_historical_simulation", ownership="team_reproducibility_profile",
    cube_edge_mm=10.0, vial_diameter_mm=17.0, vial_length_mm=140.0,
    cap_flange_diameter_mm=31.75, cap_height_mm=15.70,
    cap_flange_thickness_mm=0.088 * 25.4, cap_plug_base_diameter_mm=0.930 * 25.4,
    cap_plug_tip_diameter_mm=_CAP_TIP_DIAMETER_MM_EXACT, bin_dimensions=None,
    vial_asset=_vial_convention(17.0, 140.0),
)

REAL_CELL_GEOMETRY_PROFILE = PhysicalGeometryProfile(
    name="real_cell_confirmed", semantic_version="1.1.0",
    compatibility_id="organizer-confirmed-real-cell-geometry-v1", units="millimetres",
    source="organizer_confirmed_physical_geometry", ownership="organizer_dimensions_team_profile",
    cube_edge_mm=10.0, vial_diameter_mm=27.18, vial_length_mm=203.0,
    cap_flange_diameter_mm=31.75, cap_height_mm=15.70,
    cap_flange_thickness_mm=0.088 * 25.4, cap_plug_base_diameter_mm=0.930 * 25.4,
    cap_plug_tip_diameter_mm=_CAP_TIP_DIAMETER_MM_EXACT,
    bin_dimensions=BinDimensions(50.0, 50.0, 55.0, 35.0, 35.0, 25.0),
    vial_asset=_vial_convention(27.18, 203.0),
)

GEOMETRY_PROFILES = MappingProxyType({
    HISTORICAL_GEOMETRY_PROFILE.name: HISTORICAL_GEOMETRY_PROFILE,
    REAL_CELL_GEOMETRY_PROFILE.name: REAL_CELL_GEOMETRY_PROFILE,
})


def recompute_profile_sha256(profile: PhysicalGeometryProfile) -> str:
    return _canonical_sha256(profile_canonical_payload(profile))


def resolve_geometry_profile(selection: str | PhysicalGeometryProfile | None = None) -> PhysicalGeometryProfile:
    """Resolve only an explicit immutable profile; ``None`` preserves history."""
    if selection is None:
        selection = HISTORICAL_GEOMETRY_PROFILE
    if isinstance(selection, str):
        if selection not in GEOMETRY_PROFILES:
            raise ValueError("unknown geometry profile")
        selection = GEOMETRY_PROFILES[selection]
    if not isinstance(selection, PhysicalGeometryProfile):
        raise ValueError("unknown geometry profile")
    if recompute_profile_sha256(selection) != selection.canonical_sha256:
        raise ValueError("forged or stale canonical profile hash")
    known = GEOMETRY_PROFILES.get(selection.name)
    if known is None or known.canonical_sha256 != selection.canonical_sha256:
        raise ValueError("custom or modified geometry profiles are unsupported")
    if recompute_profile_sha256(known) != known.canonical_sha256:
        raise ValueError("registered profile identity is stale")
    return known


def validate_vial_asset(profile: PhysicalGeometryProfile, *, repository_root: Path | None = None) -> str:
    selected = resolve_geometry_profile(profile)
    root = repository_root or Path(__file__).resolve().parent.parent
    asset_path = root / selected.vial_asset.asset_path
    try:
        # USDA is tracked as text. Hash canonical Git LF content so checkout
        # autocrlf policy cannot change the bound asset identity.
        content = asset_path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(content).hexdigest()
    except OSError as exc:
        raise ValueError(f"tracked vial asset is unavailable: {exc}") from exc
    if digest != selected.vial_asset.asset_sha256:
        raise ValueError("tracked vial asset SHA-256 mismatch")
    return digest


def vial_asset_scale(profile: PhysicalGeometryProfile) -> tuple[float, float, float]:
    selected = resolve_geometry_profile(profile)
    return selected.vial_asset.radial_scale, selected.vial_asset.radial_scale, selected.vial_asset.axial_scale


def _finite_coordinate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be an XY tuple")
    return _finite_coordinate(value[0], f"{name}.x"), _finite_coordinate(value[1], f"{name}.y")


def vial_capsule(center_xy_mm: tuple[float, float], yaw_deg: float, profile: PhysicalGeometryProfile) -> dict:
    center = _finite_pair(center_xy_mm, "vial center")
    yaw = _finite_coordinate(yaw_deg, "vial yaw")
    selected = resolve_geometry_profile(profile)
    angle = math.radians(yaw)
    axis = math.cos(angle), math.sin(angle)
    half = selected.vial_length_mm / 2.0
    endpoints = (
        (center[0] - half * axis[0], center[1] - half * axis[1]),
        (center[0] + half * axis[0], center[1] + half * axis[1]),
    )
    return {"center_xy_mm": center, "yaw_deg": yaw, "axis_xy": axis,
            "endpoints_xy_mm": endpoints, "radius_mm": selected.vial_radius_mm,
            "length_mm": selected.vial_length_mm}


def vial_scene_geometry(center_xy_m: tuple[float, float], yaw_deg: float,
                        profile: PhysicalGeometryProfile) -> dict:
    center = _finite_pair(center_xy_m, "vial center")
    yaw = _finite_coordinate(yaw_deg, "vial yaw")
    selected = resolve_geometry_profile(profile)
    dimensions = selected.dimensions_metres()
    angle = math.radians(yaw)
    axis = math.cos(angle), math.sin(angle)
    half = dimensions["vial_length"] / 2.0
    endpoints = (
        (center[0] - half * axis[0], center[1] - half * axis[1]),
        (center[0] + half * axis[0], center[1] + half * axis[1]),
    )
    root_ops: tuple[tuple[str, object], ...] = (("translate_tail", endpoints[0]),)
    if yaw != 0.0:
        root_ops += (("rotate_Z_deg", yaw),)
    root_ops += (("rotate_Y_deg", 90.0),)
    return {
        "origin_xy_m": endpoints[0], "center_xy_m": center, "endpoints_xy_m": endpoints,
        "root_transform_sequence": root_ops,
        "visual_scale_sequence": ("unitsResolve_0.0254", "photoFit_XYZ"),
        "radius_m": dimensions["vial_radius"], "length_m": dimensions["vial_length"],
        "visual_scale": vial_asset_scale(selected),
        "collision_axis": selected.vial_asset.collision_long_axis,
        "collision_radius_m": dimensions["vial_radius"],
        "collision_length_m": dimensions["vial_length"],
        "collision_translate_m": (0.0, 0.0, dimensions["vial_length"] / 2.0),
    }


def historical_cell_compatibility_payload() -> dict:
    profile = resolve_geometry_profile()
    dimensions = profile.dimensions_metres()
    return {
        "profile_sha256": profile.canonical_sha256,
        "dimensions_hex": {name: value.hex() for name, value in dimensions.items()},
        "vial_visual_scale_hex": tuple(value.hex() for value in vial_asset_scale(profile)),
        "vial_root_transform_sequence": profile.vial_asset.visual_root_rotations_deg,
        "vial_root_ops_zero_yaw": vial_scene_geometry(
            (0.0, 0.0), 0.0, profile
        )["root_transform_sequence"],
        "vial_root_ops_nonzero_yaw": vial_scene_geometry(
            (0.0, 0.0), 17.0, profile
        )["root_transform_sequence"],
        "vial_visual_scale_sequence": ("unitsResolve_0.0254", "photoFit_XYZ"),
        "vial_collision_axis": profile.vial_asset.collision_long_axis,
        "vial_collision_translate_rule": "local_Z_half_selected_length",
        "cap_collider_sequence": ("flange_cylinder_Z", "plug_convex_mesh_rings_base_to_tip"),
        "cube_collider_sequence": ("unit_cube", "scale_selected_edge"),
    }


HISTORICAL_CELL_COMPATIBILITY_SHA256 = _canonical_sha256(historical_cell_compatibility_payload())

GEOMETRY_MIGRATION_CONTRACT = MappingProxyType({
    "historical_profile_remains_default": True,
    "historical_evidence_vial_geometry": "scaled_17_mm_diameter_140_mm_length",
    "real_cell_profile_vial_geometry": "physical_27.18_mm_diameter_203_mm_length",
    "historical_evidence_reinterpreted": False,
    "real_profile_isaac_validation_required": True,
    "host_geometry_checks_establish_physical_readiness": False,
    "robot_motion_authorized": False,
})
