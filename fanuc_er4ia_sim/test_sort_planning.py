"""Host-side regression checks for orientation gating and physical clearances.

These use synthetic observations, not Cosmos inference or PhysX execution.
"""

import copy
import unittest
from unittest.mock import patch
import numpy as np

import scene_config as C
from test_support import camera_model as model
from plan_sort import build_plan as task_plan, cap_silhouette, score_observations
from sort_scene import validate_plan, verify_cube_stack


def build_plan(*args, **kwargs):
    """Exercise the historical mode independently of the hackathon default."""
    return task_plan(*args, task="sort", **kwargs)


class HardwareConfigurationTests(unittest.TestCase):
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
            )
        return obj

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


if __name__ == "__main__":
    unittest.main()
