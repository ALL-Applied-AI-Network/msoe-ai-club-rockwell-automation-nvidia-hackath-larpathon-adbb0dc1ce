"""Host-side regression checks for orientation gating and physical clearances.

These use synthetic observations, not Cosmos inference or PhysX execution.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

import scene_config as C
from defect_zone_validation import (
    configured_geometry_validation,
    robot_inactivity_validation,
    slot_records,
    validate_scene_capture,
    zone_marker_paths,
)
from test_support import camera_model as model
from plan_sort import build_plan as task_plan, cap_silhouette, score_observations
from sort_scene import (
    build_defect_report,
    cube_footprint_within_defect_zone,
    evaluate_defect_placement,
    evaluate_distractor_safety,
    validate_plan,
    verify_cube_stack,
)


def build_plan(*args, **kwargs):
    """Exercise the historical mode independently of the hackathon default."""
    return task_plan(*args, task="sort", **kwargs)


class HardwareConfigurationTests(unittest.TestCase):
    def test_audited_defect_sort_constants(self):
        self.assertEqual(C.CUBE_EDGE, 0.010)
        self.assertEqual(
            C.DEFECT_CLASS_BY_COLOUR,
            {
                "red": "Defect Class A",
                "blue": "Defect Class B",
                "yellow": "Defect Class C",
            },
        )
        self.assertEqual(C.DEFECT_ZONE_SIZE, (0.080, 0.060))
        self.assertEqual(C.DEFECT_ZONE_SAFE_INSET, 0.010)
        self.assertEqual(
            C.DEFECT_ZONE_CENTERS,
            {"red": (0.200, 0.320), "blue": (0.300, 0.320), "yellow": (0.400, 0.320)},
        )
        self.assertEqual(
            C.DEFECT_TARGET_SLOTS,
            {
                "red": [(0.185, 0.320), (0.215, 0.320)],
                "blue": [(0.285, 0.320), (0.315, 0.320)],
                "yellow": [(0.385, 0.320), (0.415, 0.320)],
            },
        )
        self.assertEqual(len(slot_records()), 6)
        self.assertEqual(len({tuple(row["xy"]) for row in slot_records()}), 6)
        self.assertEqual(len(zone_marker_paths()), 9)
        geometry = configured_geometry_validation()
        self.assertTrue(all(geometry["checks"].values()))
        self.assertEqual(len(geometry["ik_checks"]), 6)
        self.assertTrue(all(row["reachable"] for row in geometry["ik_checks"]))
        self.assertEqual(C.PASSIVE_JOINT_DRIFT_TOLERANCE_RAD, 1e-3)

        measured_yaad_drift_rad = 0.00010245561134070158
        validation_cases = (
            ("zero commands below tolerance", 0.0009, {}, True, True, True),
            ("zero commands above tolerance", 0.0011, {}, True, False, False),
            (
                "command fails regardless of drift",
                0.0,
                {"articulation_action_commands": 1},
                False,
                True,
                False,
            ),
            (
                "YAAD 3e370799 measured drift",
                measured_yaad_drift_rad,
                {},
                True,
                True,
                True,
            ),
        )
        for (
            name,
            drift,
            commands,
            no_commands,
            passive_drift_passed,
            validation_passed,
        ) in validation_cases:
            with self.subTest(name):
                result = robot_inactivity_validation([0.0], [drift], commands)
                self.assertEqual(result["no_robot_commands_issued"], no_commands)
                self.assertEqual(
                    result["passive_joint_drift_within_tolerance"],
                    passive_drift_passed,
                )
                self.assertEqual(result["robot_commands_issued"], not no_commands)
                self.assertAlmostEqual(result["maximum_passive_joint_drift_rad"], drift)
                self.assertAlmostEqual(
                    result["maximum_passive_joint_drift_degrees"], np.rad2deg(drift)
                )
                self.assertEqual(
                    result["passive_joint_drift_tolerance_rad"],
                    C.PASSIVE_JOINT_DRIFT_TOLERANCE_RAD,
                )
                self.assertEqual(
                    result["no_robot_commands_issued"]
                    and result["passive_joint_drift_within_tolerance"],
                    validation_passed,
                )

        # Structural guard only: the host environment has no pxr runtime, so the
        # Isaac/USD run remains the authority for actual customData type acceptance.
        cell_source = (Path(__file__).parent / "cell.py").read_text(encoding="utf-8")
        marker_source = cell_source.split("def add_defect_zone_markers", 1)[1].split(
            "def add_bin", 1
        )[0]
        for typed_write in (
            'SetCustomDataByKey("colour", str(colour))',
            '"defectClass", str(C.DEFECT_CLASS_BY_COLOUR[colour])',
            'SetCustomDataByKey("centerXY", Gf.Vec2d(x, y))',
            'SetCustomDataByKey("sizeXY", Gf.Vec2d(*C.DEFECT_ZONE_SIZE))',
            'SetCustomDataByKey("slotXY", Gf.Vec2d(slot_x, slot_y))',
        ):
            self.assertIn(typed_write, marker_source)
        self.assertNotIn('SetCustomDataByKey("centerXY", list(', marker_source)
        self.assertNotIn('SetCustomDataByKey("sizeXY", list(', marker_source)
        self.assertNotIn('SetCustomDataByKey("slotXY", [', marker_source)

        validator_source = (
            Path(__file__).parent / "defect_zone_validation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("GetDescendants", validator_source)
        self.assertIn("Usd.PrimRange(root)", validator_source)
        self.assertIn('"all_marker_metadata_matches"', validator_source)

        # Failure-safety is host-testable without pretending to execute USD.  A
        # mocked stage check raises exactly like the failed YAAD run; the real
        # validator must still annotate, serialize drift, and write a FAIL report.
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            (out / "world_camera.png").write_bytes(b"captured")
            (out / "camera.json").write_text(
                json.dumps({"camera": "table", "resolution": [1920, 1080]}),
                encoding="utf-8",
            )

            def fake_annotate(_image, _camera, output):
                Path(output).write_bytes(b"annotated")

            visibility = {
                "all_cubes_inside_frame": True,
                "all_zones_inside_frame": True,
                "all_target_slots_inside_frame": True,
            }
            with (
                patch(
                    "defect_zone_validation.stage_marker_validation",
                    side_effect=AttributeError("synthetic USD traversal failure"),
                ),
                patch(
                    "defect_zone_validation.annotate_capture",
                    side_effect=fake_annotate,
                ),
                patch(
                    "defect_zone_validation.camera_visibility",
                    return_value=visibility,
                ),
                patch(
                    "defect_zone_validation.cv2.imread",
                    return_value=np.zeros((1080, 1920, 3), dtype=np.uint8),
                ),
            ):
                report = validate_scene_capture(
                    object(), out, {"objects": []}, {"objects": {}}, [0, 0], [0, 0]
                )
            persisted = json.loads(
                (out / "defect_zone_validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(persisted["status"], "FAIL")
            self.assertEqual(persisted["exceptions"][0]["phase"], "stage_markers")
            self.assertIn("traceback", persisted["exceptions"][0])
            self.assertEqual(
                persisted["actions"]["maximum_passive_joint_drift_rad"], 0
            )
            self.assertTrue(persisted["checks"]["no_robot_commands_issued"])
            self.assertTrue(
                persisted["checks"]["passive_joint_drift_within_tolerance"]
            )
            self.assertTrue(persisted["checks"]["annotated_capture_written"])

    def test_smc_travel_and_demo_clearance(self):
        self.assertEqual(C.GRIPPER_MODEL, "SMC MHZL2-16D2")
        self.assertAlmostEqual(C.JAW_STROKE, 0.006)
        self.assertAlmostEqual(C.JAW_OPEN - C.JAW_CLOSED, C.JAW_STROKE)
        self.assertAlmostEqual(2 * C.JAW_OPEN, 0.012)
        self.assertGreater(2 * C.JAW_OPEN - C.CUBE_EDGE, 0.0015)
        self.assertGreater(2 * (C.JAW_OPEN + C.FINGER_INNER_TIP), C.CUBE_EDGE * 2**0.5)
        self.assertGreater(C.CUBE_JAW_HALF_TARGET, C.JAW_CLOSED)

    def test_calibrated_tool_stack_and_measured_tip_gap(self):
        self.assertAlmostEqual(C.TOOL_LENGTH, 0.293)
        self.assertAlmostEqual(C.TOOL_LENGTH - C.FLANGE_OFFSET, 0.223)
        self.assertAlmostEqual(C.FINGER_LENGTH, 0.076)
        self.assertAlmostEqual(2 * C.FINGER_INNER_TIP, 0.004, delta=0.00002)
        self.assertAlmostEqual(
            2 * (C.JAW_OPEN + C.FINGER_INNER_TIP), 0.016, delta=0.00002
        )
        self.assertGreater(C.MAT_T, 0)
        self.assertAlmostEqual(C.SURFACE_DZ, -0.0051)

    def test_calibrated_table_camera_covers_and_recovers_targets(self):
        cam = model(C.TABLE_CAM)
        self.assertAlmostEqual(cam.table_z, C.TABLE_TOP_Z - 0.0051)
        for x, y, _ in C.SORT_CUBES:
            point = np.array([x, y, cam.table_z + C.CUBE_EDGE / 2])
            u, v = cam.project(point)
            self.assertTrue(0 < u < cam.W and 0 < v < cam.H)
            np.testing.assert_allclose(
                cam.to_plane(u, v, z=point[2]), point, atol=1e-12
            )
        for cfg, fx in [
            (C.TABLE_CAM, 1027.6),
            (C.SIDE_CAM, 1185.9),
            (C.WRIST_CAM, 619.6),
        ]:
            self.assertAlmostEqual(
                cfg["resolution"][0] * cfg["focal_length"] / cfg["horiz_aperture"],
                fx,
                delta=0.1,
            )


class SortPlanningTests(unittest.TestCase):
    def setUp(self):
        self.cam = model(C.TABLE_CAM)
        jaw = patch.object(C, "JAW_OPEN", 0.012)
        jaw.start()
        self.addCleanup(jaw.stop)

    def detection(self, category, xy, **extra):
        height = {
            "cap": C.CAP_HEIGHT / 2,
            "cube": C.CUBE_EDGE / 2,
            "vial": C.VIAL_RADIUS + 0.001,
        }[category]
        uv = self.cam.project([*xy, self.cam.table_z + height]) / [
            self.cam.W,
            self.cam.H,
        ]
        obj = dict(
            dict(
                category=category, label=category, colour="yellow", center=uv.tolist()
            ),
            **extra,
        )
        if category == "cube":
            obj.update(
                top_edge_endpoints=[
                    (
                        self.cam.project(
                            [
                                xy[0] + dx,
                                xy[1] - C.CUBE_EDGE / 2,
                                self.cam.table_z + C.CUBE_EDGE,
                            ]
                        )
                        / [self.cam.W, self.cam.H]
                    ).tolist()
                    for dx in (-C.CUBE_EDGE / 2, C.CUBE_EDGE / 2)
                ],
                pose_confidence=0.95,
                classification_confidence=0.95,
                pose_fit_iou=0.95,
            )
        return obj

    def defect_inventory(self):
        return [
            self.detection("cube", xy, colour=colour, object_id=f"cube-{i}")
            for i, (xy, colour) in enumerate(
                [
                    ((0.45, 0.15), "red"),
                    ((0.25, 0.15), "blue"),
                    ((0.35, -0.15), "yellow"),
                    ((0.45, -0.15), "blue"),
                    ((0.35, 0.15), "red"),
                    ((0.25, -0.15), "yellow"),
                ]
            )
        ]

    def test_only_confident_narrow_caps_are_selected(self):
        objects = [
            self.detection("cap", xy, orientation=ori, orientation_confidence=conf)
            for xy, ori, conf in [
                ((0.30, 0.10), "narrow_end_up", 0.9),
                ((0.35, 0.10), "wide_end_up", 0.99),
                ((0.40, 0.10), "unknown", 0.99),
                ((0.45, 0.10), "narrow_end_up", 0.4),
            ]
        ]
        plan = build_plan({"objects": objects}, self.cam)
        self.assertEqual([p["detection_id"] for p in plan["picks"]], [0])
        validate_plan(plan)
        self.assertEqual(
            {o["skip_reason"] for o in plan["skipped"]},
            {"wide_end_up", "orientation_unknown", "orientation_low_confidence"},
        )

    def test_tube_axis_is_perpendicular_in_world_not_image(self):
        for yaw in (0, 10.367641, 39.638288, 90, 135):
            theta = np.deg2rad(yaw)
            axis = np.array([np.cos(theta), np.sin(theta)])
            xy = np.array([0.4, -0.1])
            h = self.cam.table_z + C.VIAL_RADIUS + 0.001
            ends = [
                (
                    self.cam.project([*(xy + s * axis * C.VIAL_LENGTH / 2), h])
                    / [self.cam.W, self.cam.H]
                ).tolist()
                for s in (-1, 1)
            ]
            obj = self.detection("vial", xy, axis_endpoints=ends, axis_confidence=0.95)
            plan = build_plan({"objects": [obj]}, self.cam)
            self.assertEqual(len(plan["picks"]), 1)
            p = plan["picks"][0]
            self.assertAlmostEqual(np.dot(axis, p["spread_dir"][:2]), 0, places=8)
            np.testing.assert_allclose(p["xy"], xy, atol=1e-9)
            validate_plan(plan)

    def test_missing_tube_axis_is_skipped(self):
        plan = build_plan({"objects": [self.detection("vial", (0.4, -0.1))]}, self.cam)
        self.assertEqual(plan["picks"], [])

    def test_grid1000_coordinates_include_axis_endpoints(self):
        xy = np.array([0.4, -0.1])
        h = self.cam.table_z + C.VIAL_RADIUS + 0.001
        ends = [
            (
                self.cam.project([*(xy + [s * C.VIAL_LENGTH / 2, 0]), h])
                / [self.cam.W, self.cam.H]
                * 1000
            ).tolist()
            for s in (-1, 1)
        ]
        obj = self.detection("vial", xy, axis_endpoints=ends, axis_confidence=0.9)
        obj["center"] = (np.array(obj["center"]) * 1000).tolist()
        p = build_plan({"objects": [obj]}, self.cam)["picks"][0]
        np.testing.assert_allclose(p["xy"], xy, atol=1e-9)

    def test_extra_caps_balanced_and_clearance_is_physical(self):
        self.assertEqual(len(C.SORT_CAPS), 2)
        self.assertEqual(C.CAP_ORIENTATIONS, ["narrow_end_up"] * 2)
        self.assertAlmostEqual(C.JAW_STROKE, 0.006)
        self.assertLess(C.CUBE_EDGE, 2 * C.JAW_OPEN)
        self.assertLess(2 * C.CAP_PLUG_BASE_RADIUS, 2 * C.JAW_OPEN)
        self.assertGreater(2 * C.CAP_RADIUS, 2 * C.JAW_OPEN)
        self.assertGreater(C.CAP_PICK_HEIGHT, C.CAP_FLANGE_THICK)

    def test_cap_mask_refinement_recovers_offset(self):
        xy = np.array([0.35, 0.15])
        mask = cap_silhouette(self.cam, xy, "narrow_end_up", (self.cam.H, self.cam.W))
        img = np.full((self.cam.H, self.cam.W, 3), 240, np.uint8)
        img[mask > 0] = [25, 185, 225]
        ys, xs = np.where(mask > 0)
        obj = self.detection(
            "cap",
            xy + [0.005, -0.004],
            orientation="narrow_end_up",
            orientation_confidence=0.9,
        )
        obj["bbox"] = [
            xs.min() / self.cam.W,
            ys.min() / self.cam.H,
            xs.max() / self.cam.W,
            ys.max() / self.cam.H,
        ]
        p = build_plan({"objects": [obj]}, self.cam, img)["picks"][0]
        self.assertLess(np.linalg.norm(np.array(p["xy"]) - xy), 0.001)
        self.assertGreater(p["silhouette_iou"], 0.9)

    def test_ground_truth_scores_but_never_changes_plan(self):
        obj = self.detection(
            "cap", (0.35, 0.15), orientation="narrow_end_up", orientation_confidence=0.9
        )
        plan = build_plan({"objects": [obj]}, self.cam)
        original = copy.deepcopy(plan)
        gt = {
            "objects": [
                dict(
                    name="SortCap0",
                    category="cap",
                    xyz=[0.35, 0.15, 0.008],
                    orientation="wide_end_up",
                )
            ]
        }
        score = score_observations(plan, gt)
        self.assertEqual(score["wide_caps_selected"], 1)
        self.assertEqual(plan, original)

    def test_duplicate_detections_do_not_create_two_picks(self):
        obj = self.detection(
            "cap", (0.35, 0.15), orientation="narrow_end_up", orientation_confidence=0.9
        )
        plan = build_plan({"objects": [obj, dict(obj)]}, self.cam)
        self.assertEqual(len(plan["picks"]), 1)

    def test_executor_rejects_wide_cap_even_in_edited_plan(self):
        obj = self.detection(
            "cap", (0.35, 0.15), orientation="narrow_end_up", orientation_confidence=0.9
        )
        plan = build_plan({"objects": [obj]}, self.cam)
        plan["picks"][0]["orientation"] = "wide_end_up"
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_malformed_detection_does_not_discard_good_neighbour(self):
        good = self.detection(
            "cap", (0.35, 0.15), orientation="narrow_end_up", orientation_confidence=0.9
        )
        bad = dict(category="cap", center=None, bbox=None)
        plan = build_plan({"objects": [bad, good]}, self.cam)
        self.assertEqual(len(plan["picks"]), 1)
        self.assertEqual(plan["picks"][0]["detection_id"], 1)

    def test_six_cubes_form_one_tower_in_colour_order(self):
        objects = [
            self.detection("cube", xy, colour=colour)
            for xy, colour in [
                ((0.45, 0.15), "red"),
                ((0.25, 0.15), "blue"),
                ((0.35, -0.15), "yellow"),
                ((0.45, -0.15), "blue"),
                ((0.35, 0.15), "red"),
                ((0.25, -0.15), "yellow"),
            ]
        ]
        objects.append(
            self.detection(
                "cap",
                (0.4, 0),
                orientation="narrow_end_up",
                orientation_confidence=0.99,
            )
        )
        with patch.object(C, "JAW_OPEN", 0.010):
            plan = task_plan({"objects": objects}, self.cam, task="cube_stack")
            validate_plan(plan)
        self.assertEqual(
            [p["colour"] for p in plan["picks"]], list(C.CUBE_STACK_COLOURS)
        )
        self.assertEqual(len({p["detection_id"] for p in plan["picks"]}), 6)
        for level, p in enumerate(plan["picks"]):
            np.testing.assert_allclose(p["place_xy"], C.CUBE_STACK_XY)
            self.assertAlmostEqual(
                p["place_z"], C.SURFACE_DZ + level * C.CUBE_EDGE + C.CUBE_TIP_CLEARANCE
            )
        self.assertEqual(plan["skipped"][0]["skip_reason"], "not_a_cube")

    def test_incomplete_or_wrong_colour_set_cannot_start_tower(self):
        objects = [
            self.detection("cube", (x, y), colour=colour)
            for x, y, colour in C.SORT_CUBES
        ]
        for bad in (
            objects[:-1],
            [dict(o, colour="red") if i == 0 else o for i, o in enumerate(objects)],
        ):
            plan = task_plan({"objects": bad}, self.cam, task="cube_stack")
            self.assertFalse(plan["stack_ready"])
            self.assertEqual(plan["picks"], [])

    def test_executor_rejects_wrong_tower_order_or_height(self):
        objects = [
            self.detection("cube", (x, y), colour=colour)
            for x, y, colour in C.SORT_CUBES
        ]
        plan = task_plan({"objects": objects}, self.cam, task="cube_stack")
        for field, value in [
            ("colour", "red"),
            ("place_z", C.SURFACE_DZ + 0.1),
            ("stack_level", 3),
        ]:
            bad = copy.deepcopy(plan)
            bad["picks"][0][field] = value
            with self.assertRaises(ValueError):
                validate_plan(bad)

    def test_support_uses_adjacent_cubes_not_only_nominal_target(self):
        entries = [dict(tracked_object=f"cube{i}", stack_level=i) for i in range(2)]

        def poses(offsets):
            return {
                f"cube{i}": dict(
                    centre=[
                        C.CUBE_STACK_XY[0] + dx,
                        C.CUBE_STACK_XY[1],
                        C.TABLE_TOP_Z + C.SURFACE_DZ + (i + 0.5) * C.CUBE_EDGE,
                    ]
                )
                for i, dx in enumerate(offsets)
            }

        self.assertTrue(verify_cube_stack(poses([0.00135, 0.00206]), entries))
        # Both within the site tolerance, but without adequate mutual overlap.
        self.assertFalse(verify_cube_stack(poses([-0.002, 0.002]), entries))
        self.assertFalse(verify_cube_stack(poses([0.001, 0.005]), entries))

    def test_defect_sort_valid_two_two_two_inventory_and_distractors(self):
        objects = self.defect_inventory() + [
            self.detection("cap", (0.40, 0.0)),
            self.detection("cap", (0.42, 0.0)),
            self.detection("vial", (0.40, -0.1)),
            self.detection("vial", (0.42, -0.1)),
        ]
        plan = task_plan({"objects": objects}, self.cam, task="defect_sort")
        validate_plan(plan)
        self.assertTrue(plan["defect_sort_ready"])
        self.assertEqual(len(plan["picks"]), 6)
        self.assertEqual(len(set(plan["validated_cube_detection_ids"])), 6)
        self.assertEqual(len({p["object_id"] for p in plan["picks"]}), 6)
        self.assertEqual(
            [o["disposition"] for o in plan["observations"][6:]],
            ["ignored_distractor"] * 4,
        )
        self.assertTrue(all(p["category"] == "cube" for p in plan["picks"]))
        self.assertEqual(
            {p["defect_class"] for p in plan["picks"]},
            set(C.DEFECT_CLASS_BY_COLOUR.values()),
        )

    def test_defect_sort_generates_unique_internal_object_ids_when_absent(self):
        objects = self.defect_inventory()
        for obj in objects:
            obj.pop("object_id")
        plan = task_plan({"objects": objects}, self.cam, task="defect_sort")
        validate_plan(plan)
        self.assertEqual(len({p["object_id"] for p in plan["picks"]}), 6)

    def test_defect_sort_rejects_missing_extra_and_wrong_colour_count(self):
        inventory = self.defect_inventory()
        cases = [
            inventory[:-1],
            inventory
            + [self.detection("cube", (0.29, -0.23), colour="red")],
            [dict(o, colour="red") if i == 1 else o for i, o in enumerate(inventory)],
        ]
        for objects in cases:
            with self.subTest(count=len(objects)):
                plan = task_plan({"objects": objects}, self.cam, task="defect_sort")
                self.assertFalse(plan["defect_sort_ready"])
                self.assertEqual(plan["picks"], [])
                with self.assertRaises(ValueError):
                    validate_plan(plan)

    def test_defect_sort_rejects_duplicate_cube_id_or_detection(self):
        inventory = self.defect_inventory()
        duplicate_id = copy.deepcopy(inventory)
        duplicate_id[1]["detection_id"] = 7
        duplicate_id[2]["detection_id"] = 7
        spatial_duplicate = inventory + [dict(inventory[0], object_id="cube-extra")]
        for objects in (duplicate_id, spatial_duplicate):
            plan = task_plan({"objects": objects}, self.cam, task="defect_sort")
            self.assertFalse(plan["defect_sort_ready"])
            self.assertEqual(plan["picks"], [])

    def test_defect_sort_rejects_ambiguous_or_low_confidence_cube(self):
        for changes in (
            {"colour": "unknown"},
            {"classification_confidence": 0.79},
            {"pose_confidence": 0.79},
            {"pose_fit_iou": 0.74},
        ):
            objects = self.defect_inventory()
            objects[0].update(changes)
            plan = task_plan({"objects": objects}, self.cam, task="defect_sort")
            self.assertFalse(plan["defect_sort_ready"])
            self.assertEqual(plan["picks"], [])

    def test_executor_rejects_distractor_wrong_zone_and_pick_count_tampering(self):
        plan = task_plan(
            {"objects": self.defect_inventory()}, self.cam, task="defect_sort"
        )
        bad_distractor = copy.deepcopy(plan)
        bad_distractor["picks"][0]["category"] = "cap"
        wrong_zone = copy.deepcopy(plan)
        wrong_zone["picks"][0]["assigned_slot"] = list(
            C.DEFECT_TARGET_SLOTS["blue"][0]
        )
        wrong_zone["picks"][0]["place_xy"] = list(C.DEFECT_TARGET_SLOTS["blue"][0])
        short = copy.deepcopy(plan)
        short["picks"].pop()
        for bad in (bad_distractor, wrong_zone, short):
            with self.assertRaises(ValueError):
                validate_plan(bad)

    def test_executor_rejects_duplicate_target_slot(self):
        plan = task_plan(
            {"objects": self.defect_inventory()}, self.cam, task="defect_sort"
        )
        plan["picks"][1]["assigned_slot"] = list(plan["picks"][0]["assigned_slot"])
        plan["picks"][1]["place_xy"] = list(plan["picks"][0]["assigned_slot"])
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_zone_containment_and_five_mm_slot_tolerance(self):
        plan = task_plan(
            {"objects": self.defect_inventory()}, self.cam, task="defect_sort"
        )
        pick = next(p for p in plan["picks"] if p["colour"] == "red")
        slot = np.array(pick["assigned_slot"])
        self.assertTrue(
            evaluate_defect_placement(pick, slot + [0.005, 0])[
                "correct_zone_placement"
            ]
        )
        self.assertFalse(
            evaluate_defect_placement(pick, slot + [0.00501, 0])[
                "correct_zone_placement"
            ]
        )
        self.assertTrue(cube_footprint_within_defect_zone((0.225, 0.320), "red"))
        self.assertFalse(cube_footprint_within_defect_zone((0.2251, 0.320), "red"))
        self.assertFalse(cube_footprint_within_defect_zone((0.2, 0.32), "unknown"))

    def test_flat_cube_height_is_half_edge_and_stack_height_is_separate(self):
        flat = task_plan(
            {"objects": self.defect_inventory()}, self.cam, task="defect_sort"
        )
        self.assertTrue(
            all(
                np.isclose(
                    p["expected_centre_z"], C.SURFACE_DZ + 0.5 * C.CUBE_EDGE
                )
                for p in flat["picks"]
            )
        )
        stack = task_plan(
            {"objects": self.defect_inventory()}, self.cam, task="cube_stack"
        )
        self.assertAlmostEqual(
            stack["picks"][1]["expected_centre_z"],
            C.SURFACE_DZ + 1.5 * C.CUBE_EDGE,
        )

    def test_distractor_displacement_and_lift_thresholds(self):
        before = {
            "cap": {"centre": [0, 0, 0], "yaw_deg": 0},
            "vial": {"centre": [0, 0, 0], "yaw_deg": 0},
        }
        at_limit = {
            "cap": {"centre": [0.002, 0, 0], "yaw_deg": 5},
            "vial": {"centre": [0, 0, 0], "yaw_deg": 0},
        }
        safe = evaluate_distractor_safety(before, at_limit, [], {"cap", "vial"})
        self.assertTrue(safe["distractor_safety_passed"])
        over = copy.deepcopy(at_limit)
        over["cap"]["centre"][0] = 0.00201
        over["vial"]["yaw_deg"] = 5.01
        unsafe = evaluate_distractor_safety(
            before, over, [{"distractor_lifts": ["cap"]}], {"cap", "vial"}
        )
        self.assertFalse(unsafe["distractor_safety_passed"])
        self.assertEqual(unsafe["distractor_grasps"], ["cap"])
        self.assertTrue(
            any(
                row["translation_exceeded"] or row["rotation_exceeded"]
                for row in unsafe["distractor_displacements"]
            )
        )

        half_angle = np.deg2rad(6 / 2)
        rotated = copy.deepcopy(before)
        before["cap"]["orientation"] = [1, 0, 0, 0]
        rotated["cap"]["orientation"] = [np.cos(half_angle), np.sin(half_angle), 0, 0]
        rotated["vial"]["orientation"] = [1, 0, 0, 0]
        before["vial"]["orientation"] = [1, 0, 0, 0]
        quaternion_unsafe = evaluate_distractor_safety(
            before, rotated, [], {"cap", "vial"}
        )
        self.assertFalse(quaternion_unsafe["distractor_safety_passed"])

    def test_defect_report_schema_includes_metrics_retries_and_recoveries(self):
        plan = task_plan(
            {"objects": self.defect_inventory()}, self.cam, task="defect_sort"
        )
        event = {
            "pickup_verified": True,
            "placement_verified": True,
            "correct_zone_placement": True,
            "predicted_colour": "red",
            "expected_colour": "red",
            "predicted_defect_class": "Defect Class A",
            "expected_defect_class": "Defect Class A",
            "classification_correct": True,
            "position_error_mm": 0.2,
            "yaw_error_deg_mod90": 1.0,
            "placement_position_error_mm": 0.3,
            "assigned_slot_error_mm": 0.3,
            "correct_zone_containment": True,
            "retries": [{"reason": "ik", "outcome": "recovered"}],
            "recoveries": [{"reason": "settle", "outcome": "recovered"}],
        }
        report = build_defect_report(
            plan,
            [event],
            1.25,
            2.5,
            {
                "distractor_grasps": [],
                "distractor_displacements": [],
                "distractor_safety_passed": True,
            },
        )
        for field in (
            "defect_parts_detected",
            "successful_pickups",
            "correct_zone_placements",
            "distractor_grasps",
            "distractor_displacements",
            "classification_accuracy",
            "motion_time_s",
            "wall_clock_time_s",
            "retry_count",
            "recovery_count",
        ):
            self.assertIn(field, report)
        for field in (
            "predicted_colour",
            "expected_colour",
            "predicted_defect_class",
            "expected_defect_class",
            "position_error_mm",
            "yaw_error_deg_mod90",
            "placement_position_error_mm",
            "assigned_slot_error_mm",
            "correct_zone_containment",
        ):
            self.assertIn(field, event)

        failed_event = dict(
            event,
            pickup_verified=False,
            placement_verified=False,
            correct_zone_placement=False,
            classification_correct=False,
            retries=[],
            recoveries=[],
        )
        failed_report = build_defect_report(
            plan,
            [failed_event],
            0,
            0,
            {
                "distractor_grasps": ["cap"],
                "distractor_displacements": [{"name": "cap"}],
                "distractor_safety_passed": False,
            },
        )
        self.assertEqual(failed_report["successful_pickups"], 0)
        self.assertEqual(failed_report["correct_zone_placements"], 0)
        self.assertEqual(failed_report["classification_accuracy"], 0)
        self.assertFalse(failed_report["distractor_safety_passed"])

    def test_pipeline_routes_explicit_defect_sort_mode(self):
        script = (Path(__file__).parent / "run_sort_pipeline.sh").read_text()
        self.assertIn("perceive.py\" --run \"$RUN\" --defect-sort", script)
        self.assertIn("plan_sort.py\" --run \"$RUN\" --task defect_sort", script)


if __name__ == "__main__":
    unittest.main()
