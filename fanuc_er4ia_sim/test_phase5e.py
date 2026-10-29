"""Host-only regression tests for corrected Phase 5E physical geometry."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bin_geometry import (
    BinDimensions,
    Bounds2D,
    ReleaseVerticalReference,
    SurveyedBinPose,
    bin_geometry_contract,
    bin_pocket_polygon,
    cube_fully_inside_pocket,
    validate_cube_release_target,
    validate_surveyed_bins,
)
from deployment_readiness import PRODUCTION_READINESS
from physical_geometry_profile import (
    GEOMETRY_MIGRATION_CONTRACT,
    GEOMETRY_PROFILES,
    HISTORICAL_CELL_COMPATIBILITY_SHA256,
    HISTORICAL_GEOMETRY_PROFILE,
    REAL_CELL_GEOMETRY_PROFILE,
    VIAL_ASSET_REPOSITORY_PATH,
    historical_cell_compatibility_payload,
    profile_canonical_payload,
    recompute_profile_sha256,
    resolve_geometry_profile,
    validate_vial_asset,
    vial_asset_scale,
    vial_capsule,
    vial_scene_geometry,
)
from real_layout_feasibility import (
    REAL_CELL_INVENTORY,
    SurveyedObjectPose,
    object_footprint,
    validate_real_layout,
)
from real_policy_readiness import PRODUCTION_POLICY_READINESS
import scene_config as C


ROOT = Path(__file__).resolve().parent


def fixture_objects() -> tuple[SurveyedObjectPose, ...]:
    return (
        SurveyedObjectPose("red_1", "cube", (100, 100), 0, "red"),
        SurveyedObjectPose("red_2", "cube", (150, 100), 10, "red"),
        SurveyedObjectPose("blue_1", "cube", (200, 100), 20, "blue"),
        SurveyedObjectPose("blue_2", "cube", (250, 100), 30, "blue"),
        SurveyedObjectPose("yellow_1", "cube", (300, 100), 40, "yellow"),
        SurveyedObjectPose("yellow_2", "cube", (350, 100), 45, "yellow"),
        SurveyedObjectPose("vial_1", "vial", (200, 225), 0),
        SurveyedObjectPose("vial_2", "vial", (500, 225), 0),
        SurveyedObjectPose("cap_1", "cap", (700, 100), 0),
        SurveyedObjectPose("cap_2", "cap", (750, 100), 0),
    )


def fixture_bins() -> dict[str, SurveyedBinPose]:
    return {
        "bin_1": SurveyedBinPose((150, 500), 0),
        "bin_2": SurveyedBinPose((350, 500), 15),
        "bin_3": SurveyedBinPose((550, 500), 30),
    }


def layout_kwargs() -> dict:
    return {
        "source_bounds": Bounds2D(-20, 850, 25, 350),
        "table_bounds": Bounds2D(-100, 1000, -100, 700),
        "reach_origin_xy_mm": (400, 300),
        "maximum_reach_mm": 900,
        "reach_margin_mm": 20,
        "approach_clearance_mm": 2,
        "carry_clearance_mm": 5,
        "minimum_bin_spacing_mm": 10,
        "keep_outs": {},
        "geometry_profile": REAL_CELL_GEOMETRY_PROFILE,
    }


def release_reference() -> ReleaseVerticalReference:
    return ReleaseVerticalReference(0, 55, 0, 200, 10, 180)


class HistoricalCompatibilityTests(unittest.TestCase):
    def test_every_consumed_default_float_is_bit_identical(self):
        dimensions = HISTORICAL_GEOMETRY_PROFILE.dimensions_metres()
        legacy = {
            "cube_edge": C.CUBE_EDGE,
            "vial_radius": C.VIAL_RADIUS,
            "vial_length": C.VIAL_LENGTH,
            "cap_radius": C.CAP_RADIUS,
            "cap_height": C.CAP_HEIGHT,
            "cap_flange_thickness": C.CAP_FLANGE_THICK,
            "cap_plug_base_radius": C.CAP_PLUG_BASE_RADIUS,
            "cap_plug_tip_radius": C.CAP_PLUG_TIP_RADIUS,
        }
        for name, expected in legacy.items():
            self.assertEqual(dimensions[name], expected, name)
            self.assertEqual(dimensions[name].hex(), expected.hex(), name)
        self.assertEqual(dimensions["cap_plug_tip_radius"].hex(), "0x1.553ef6b5d462cp-7")

    def test_default_vial_transform_visual_and_collision_are_exact(self):
        center, yaw = (0.42, 0.31), 17.0
        scene = vial_scene_geometry(center, yaw, HISTORICAL_GEOMETRY_PROFILE)
        angle = math.radians(yaw)
        expected_origin = (
            center[0] - C.VIAL_LENGTH / 2 * math.cos(angle),
            center[1] - C.VIAL_LENGTH / 2 * math.sin(angle),
        )
        self.assertEqual(scene["origin_xy_m"], expected_origin)
        self.assertEqual(scene["visual_scale"],
                         (C.VIAL_RADIUS / 0.01359, C.VIAL_RADIUS / 0.01359,
                          C.VIAL_LENGTH / 0.203))
        self.assertEqual(scene["collision_axis"], "Z")
        self.assertEqual(scene["collision_radius_m"], C.VIAL_RADIUS)
        self.assertEqual(scene["collision_length_m"], C.VIAL_LENGTH)
        self.assertEqual(scene["collision_translate_m"], (0.0, 0.0, C.VIAL_LENGTH / 2))
        self.assertEqual(
            scene["root_transform_sequence"],
            (("translate_tail", expected_origin), ("rotate_Z_deg", yaw), ("rotate_Y_deg", 90.0)),
        )
        self.assertEqual(scene["visual_scale_sequence"],
                         ("unitsResolve_0.0254", "photoFit_XYZ"))

    def test_default_cap_mesh_and_collider_inputs_are_exact(self):
        dimensions = HISTORICAL_GEOMETRY_PROFILE.dimensions_metres()
        self.assertEqual(dimensions["cap_radius"].hex(), C.CAP_RADIUS.hex())
        self.assertEqual(dimensions["cap_height"].hex(), C.CAP_HEIGHT.hex())
        self.assertEqual(dimensions["cap_flange_thickness"].hex(), C.CAP_FLANGE_THICK.hex())
        self.assertEqual(dimensions["cap_plug_base_radius"].hex(), C.CAP_PLUG_BASE_RADIUS.hex())
        self.assertEqual(dimensions["cap_plug_tip_radius"].hex(), C.CAP_PLUG_TIP_RADIUS.hex())
        self.assertEqual(
            (dimensions["cap_height"] - dimensions["cap_flange_thickness"]).hex(),
            C.CAP_PLUG_LENGTH.hex(),
        )

    def test_default_cube_inputs_are_exact(self):
        edge = HISTORICAL_GEOMETRY_PROFILE.dimensions_metres()["cube_edge"]
        self.assertEqual(edge.hex(), C.CUBE_EDGE.hex())
        self.assertEqual((edge / 0.016).hex(), (C.CUBE_EDGE / 0.016).hex())

    def test_scene_and_consumed_input_signatures_are_independently_recomputed(self):
        scene_payload = {
            "cube_edge": C.CUBE_EDGE, "vial_radius": C.VIAL_RADIUS,
            "vial_length": C.VIAL_LENGTH, "cap_radius": C.CAP_RADIUS,
            "cap_height": C.CAP_HEIGHT, "cap_flange_thick": C.CAP_FLANGE_THICK,
            "cap_plug_base_radius": C.CAP_PLUG_BASE_RADIUS,
            "cap_plug_tip_radius": C.CAP_PLUG_TIP_RADIUS,
            "sort_vials": C.SORT_VIALS, "sort_cubes": C.SORT_CUBES,
            "sort_caps": C.SORT_CAPS, "sort_bins": C.SORT_BINS,
            "show_bins": C.SHOW_BINS,
        }
        digest = hashlib.sha256(
            json.dumps(scene_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(digest, "6e12c582cead281c87e3e35651aa249eda1405ac34be894768da132900bcfafc")
        consumed = hashlib.sha256(json.dumps(
            historical_cell_compatibility_payload(), sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        self.assertEqual(consumed, HISTORICAL_CELL_COMPATIBILITY_SHA256)
        self.assertEqual(consumed, "dd2715bb48cf7a7ae22fc9fd0138739b2e2771f5c32d05b1a500236b542aa65d")


class ProfileIdentityTests(unittest.TestCase):
    def test_default_is_historical_and_no_implicit_selection_exists(self):
        self.assertIs(resolve_geometry_profile(), HISTORICAL_GEOMETRY_PROFILE)
        with patch.dict(os.environ, {"GEOMETRY_PROFILE": "real_cell_confirmed"}):
            self.assertIs(resolve_geometry_profile(), HISTORICAL_GEOMETRY_PROFILE)
        for selection in ("unknown", "real_cell_confirmed.json", True, 1):
            with self.assertRaises(ValueError):
                resolve_geometry_profile(selection)

    def test_registry_and_nested_records_are_immutable(self):
        with self.assertRaises(TypeError):
            GEOMETRY_PROFILES["historical_scaled_vial"] = REAL_CELL_GEOMETRY_PROFILE
        with self.assertRaises(FrozenInstanceError):
            REAL_CELL_GEOMETRY_PROFILE.vial_asset.long_axis = "X"
        with self.assertRaises(TypeError):
            GEOMETRY_MIGRATION_CONTRACT["robot_motion_authorized"] = True

    def test_hash_binds_every_asset_convention(self):
        base = REAL_CELL_GEOMETRY_PROFILE
        changes = {
            "meters_per_authored_unit": 0.001,
            "radial_axes": ("Y", "X"),
            "long_axis": "X",
            "origin_convention": "center",
            "visual_to_collision_axis_mapping": "different",
            "collision_long_axis": "X",
            "visual_root_rotations_deg": (("X", 90.0),),
            "authored_outer_diameter_units": 1.1,
            "authored_length_units": 8.0,
            "asset_sha256": "0" * 64,
        }
        for name, value in changes.items():
            payload = profile_canonical_payload(base)
            payload["vial_asset"][name] = value
            digest = hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()).hexdigest()
            self.assertNotEqual(digest, base.canonical_sha256, name)

    def test_forged_or_stale_hash_is_rejected(self):
        forged = replace(REAL_CELL_GEOMETRY_PROFILE)
        object.__setattr__(forged, "canonical_sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "forged or stale"):
            resolve_geometry_profile(forged)
        self.assertEqual(recompute_profile_sha256(REAL_CELL_GEOMETRY_PROFILE),
                         REAL_CELL_GEOMETRY_PROFILE.canonical_sha256)

    def test_malformed_profile_and_convention_values_fail(self):
        base = REAL_CELL_GEOMETRY_PROFILE
        for value in (0, -1, float("nan"), float("inf"), True, "1"):
            with self.assertRaises(ValueError):
                replace(base, cube_edge_mm=value)
        with self.assertRaises(ValueError):
            replace(base.vial_asset, radial_scale=0)
        with self.assertRaises(ValueError):
            BinDimensions(50, 50, 55, 0, 35, 25).validate()

    def test_asset_hash_is_validated_and_mismatch_fails(self):
        self.assertEqual(validate_vial_asset(REAL_CELL_GEOMETRY_PROFILE),
                         REAL_CELL_GEOMETRY_PROFILE.vial_asset.asset_sha256)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / VIAL_ASSET_REPOSITORY_PATH
            target.parent.mkdir(parents=True)
            target.write_bytes(b"not the tracked vial")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_vial_asset(REAL_CELL_GEOMETRY_PROFILE, repository_root=root)

    def test_real_vial_asset_and_capsule_geometry(self):
        profile = REAL_CELL_GEOMETRY_PROFILE
        self.assertEqual(profile.vial_radius_mm, 13.59)
        self.assertEqual(profile.vial_length_mm, 203.0)
        self.assertEqual(vial_asset_scale(profile), (1.0, 1.0, 1.0))
        capsule = vial_capsule((200, 300), 90, profile)
        self.assertAlmostEqual(capsule["endpoints_xy_mm"][0][1], 198.5)
        self.assertAlmostEqual(capsule["endpoints_xy_mm"][1][1], 401.5)
        footprint = object_footprint(SurveyedObjectPose("v", "vial", (200, 300), 90), profile)
        self.assertGreaterEqual(max(x for x, _ in footprint) - min(x for x, _ in footprint), 27.18)


class BinAndReleaseTests(unittest.TestCase):
    def test_unplaced_contract_and_rotated_geometry(self):
        contract = bin_geometry_contract()
        self.assertEqual(contract["exterior"], [50.0, 50.0, 55.0])
        self.assertEqual(contract["pocket"], [35.0, 35.0, 25.0])
        self.assertEqual(contract["rim_thickness_xy_mm"], [7.5, 7.5])
        self.assertIsNone(contract["center_xy_mm"])
        self.assertIsNone(contract["release_height_mm"])
        self.assertEqual(len(bin_pocket_polygon(SurveyedBinPose((100, 100), 37))), 4)

    def test_cube_containment_rejects_every_malformed_edge(self):
        pose = SurveyedBinPose((100, 100), 30)
        self.assertTrue(cube_fully_inside_pocket(pose, (100, 100), 45,
                                                cube_edge_mm=10, safety_inset_mm=2))
        for edge in (float("nan"), float("inf"), -1, 0, True, "10"):
            self.assertFalse(cube_fully_inside_pocket(
                pose, (100, 100), 0, cube_edge_mm=edge
            ), edge)
        self.assertFalse(cube_fully_inside_pocket(pose, (float("nan"), 100), 0,
                                                 cube_edge_mm=10))
        self.assertFalse(cube_fully_inside_pocket(pose, (100, 100), "0",
                                                 cube_edge_mm=10))
        self.assertFalse(cube_fully_inside_pocket(None, (100, 100), 0,
                                                 cube_edge_mm=10))

    def test_release_requires_3d_target_and_vertical_reference(self):
        pose = SurveyedBinPose((100, 100), 0)
        for target, reference in (
            (None, release_reference()),
            ((100, 100), release_reference()),
            ((100, 100, 70), None),
            ((100, 100, float("nan")), release_reference()),
            ((100, 100, float("inf")), release_reference()),
            ((100, 100, "70"), release_reference()),
        ):
            result = validate_cube_release_target(
                pose, target, cube_yaw_deg=0, cube_edge_mm=10,
                safety_inset_mm=2, minimum_above_rim_clearance_mm=5,
                vertical_reference=reference,
            )
            self.assertFalse(result["geometry_valid"])
            self.assertIsNone(result["action"])

    def test_release_checks_rim_workspace_reach_and_valid_geometry(self):
        pose = SurveyedBinPose((100, 100), 30)
        common = dict(cube_yaw_deg=30, cube_edge_mm=10, safety_inset_mm=2,
                      minimum_above_rim_clearance_mm=5,
                      vertical_reference=release_reference())
        below = validate_cube_release_target(pose, (100, 100, 59), **common)
        self.assertIn("release_z_below_rim_clearance", below["reasons"])
        outside = validate_cube_release_target(pose, (100, 100, 190), **common)
        self.assertIn("release_z_outside_reach", outside["reasons"])
        valid = validate_cube_release_target(pose, (100, 100, 70), **common)
        self.assertTrue(valid["geometry_valid"], valid["reasons"])
        self.assertFalse(valid["all_requirements_verified"])
        self.assertFalse(valid["production_ready"])
        self.assertFalse(valid["robot_motion_authorized"])
        self.assertIsNone(valid["trajectory"])

    def test_overlapping_bins_and_keepouts_fail(self):
        bins = fixture_bins()
        bins["bin_2"] = SurveyedBinPose((160, 500), 0)
        result = validate_surveyed_bins(
            bins, table_bounds=Bounds2D(-100, 1000, -100, 700),
            reach_origin_xy_mm=(400, 300), maximum_reach_mm=900,
            reach_margin_mm=20, minimum_edge_spacing_mm=10,
            keep_outs={"fixture": Bounds2D(125, 175, 475, 525)},
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("insufficient_spacing" in reason for reason in result["reasons"]))
        self.assertTrue(any("overlaps_keep_out" in reason for reason in result["reasons"]))


class MalformedSurveyTests(unittest.TestCase):
    def assert_failed(self, objects=None, bins=None, **overrides):
        values = layout_kwargs()
        values.update(overrides)
        result = validate_real_layout(
            fixture_objects() if objects is None else objects,
            fixture_bins() if bins is None else bins,
            **values,
        )
        self.assertFalse(result["feasible"], result)
        self.assertFalse(result["all_requirements_verified"])
        self.assertFalse(result["production_ready"])
        self.assertFalse(result["robot_motion_authorized"])
        self.assertIsNone(result["action"])
        self.assertIsNone(result["trajectory"])
        return result

    def test_malformed_object_poses_do_not_escape(self):
        for malformed in (
            SurveyedObjectPose("bad", "cube", (float("nan"), 1), 0, "red"),
            SurveyedObjectPose("bad", "cube", (1, 2), float("inf"), "red"),
            SurveyedObjectPose("bad", "cube", (1, 2), True, "red"),
            SurveyedObjectPose("bad", "cube", (1, 2), 0, "red", "inches"),
            None,
        ):
            objects = list(fixture_objects())
            objects[0] = malformed
            result = self.assert_failed(objects=tuple(objects))
            self.assertTrue(any("invalid_object_pose" in r for r in result["reasons"]))

    def test_missing_or_malformed_individual_bins_do_not_escape(self):
        bins = fixture_bins()
        del bins["bin_3"]
        self.assert_failed(bins=bins)
        for bad in (
            SurveyedBinPose((float("nan"), 1), 0),
            SurveyedBinPose((1, 2), "0"),
            SurveyedBinPose((1, 2), 0, "inches"),
            None,
        ):
            bins = fixture_bins()
            bins["bin_2"] = bad
            result = self.assert_failed(bins=bins)
            self.assertTrue(any("invalid_bin_pose" in r for r in result["reasons"]))

    def test_invalid_reach_radius_margin_and_workspace_fail(self):
        for name, value in (
            ("reach_origin_xy_mm", (float("nan"), 0)),
            ("reach_origin_xy_mm", (True, 0)),
            ("maximum_reach_mm", 0), ("maximum_reach_mm", "900"),
            ("reach_margin_mm", -1), ("reach_margin_mm", float("inf")),
        ):
            self.assert_failed(**{name: value})
        self.assert_failed(source_bounds=Bounds2D(0, 0, 0, 1))
        self.assert_failed(source_bounds=Bounds2D(0, 1, 0, 1, "inches"))
        self.assert_failed(table_bounds=Bounds2D(0, 0, 0, 1))

    def test_malformed_keepout_and_clearances_fail(self):
        self.assert_failed(keep_outs={"bad": None})
        self.assert_failed(keep_outs={"bad": Bounds2D(0, 1, 0, 1, "inches")})
        for name in ("approach_clearance_mm", "carry_clearance_mm", "minimum_bin_spacing_mm"):
            self.assert_failed(**{name: True})


class FeasibilityAndSafetyTests(unittest.TestCase):
    def test_valid_geometry_remains_unauthorized(self):
        result = validate_real_layout(fixture_objects(), fixture_bins(), **layout_kwargs())
        self.assertTrue(result["feasible"], result["reasons"])
        for key in ("all_requirements_verified", "production_ready", "robot_motion_authorized"):
            self.assertFalse(result[key])
        self.assertIsNone(result["action"])
        self.assertIsNone(result["trajectory"])

    def test_inventory_and_all_conflict_classes(self):
        self.assertEqual(REAL_CELL_INVENTORY["bins_available"], 5)
        short = validate_real_layout(fixture_objects()[:-1], fixture_bins(), **layout_kwargs())
        self.assertFalse(short["feasible"])
        objects = list(fixture_objects())
        objects[0] = SurveyedObjectPose("red_1", "cube", (200, 225), 0, "red")
        conflict = validate_real_layout(tuple(objects), fixture_bins(), **layout_kwargs())
        self.assertTrue(any("object_clearance" in r for r in conflict["reasons"]))
        bins = fixture_bins()
        bins["bin_1"] = SurveyedBinPose((200, 225), 0)
        conflict = validate_real_layout(fixture_objects(), bins, **layout_kwargs())
        self.assertTrue(any("object_bin_clearance" in r for r in conflict["reasons"]))
        values = layout_kwargs()
        values.update(source_bounds=Bounds2D(90, 360, 90, 120), maximum_reach_mm=100,
                      keep_outs={"fixture": Bounds2D(90, 110, 90, 110)})
        conflict = validate_real_layout(fixture_objects(), fixture_bins(), **values)
        self.assertTrue(any("outside_source_bounds" in r for r in conflict["reasons"]))
        self.assertTrue(any("outside_reach" in r for r in conflict["reasons"]))
        self.assertTrue(any("overlaps_keep_out" in r for r in conflict["reasons"]))

    def test_phase5b_through_5d_readiness_remains_false(self):
        phase5b = PRODUCTION_READINESS.report()
        phase5d = PRODUCTION_POLICY_READINESS.report()
        self.assertFalse(phase5b["all_requirements_verified"])
        self.assertFalse(phase5b["robot_motion_authorized"])
        for key in ("all_requirements_verified", "production_ready", "robot_motion_authorized"):
            self.assertFalse(phase5d[key])

    def test_modules_have_no_execution_imports_or_calls(self):
        forbidden_imports = {"requests", "socket", "http", "urllib", "subprocess", "isaacsim", "omni"}
        forbidden_calls = {"eval", "exec", "open", "send", "post", "request", "connect"}
        for filename in ("physical_geometry_profile.py", "bin_geometry.py",
                         "real_layout_feasibility.py"):
            tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
            imports = {alias.name.split(".")[0] for node in ast.walk(tree)
                       if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
            calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name)}
            self.assertFalse(imports & forbidden_imports, filename)
            self.assertFalse(calls & forbidden_calls, filename)

    def test_scene_selection_is_explicit(self):
        build_source = (ROOT / "build_scene.py").read_text(encoding="utf-8")
        cell_source = (ROOT / "cell.py").read_text(encoding="utf-8")
        self.assertIn("default=HISTORICAL_GEOMETRY_PROFILE.name", build_source)
        self.assertIn("geometry_profile=None", cell_source)
        self.assertIn("validate_vial_asset(profile)", cell_source)


if __name__ == "__main__":
    unittest.main()
