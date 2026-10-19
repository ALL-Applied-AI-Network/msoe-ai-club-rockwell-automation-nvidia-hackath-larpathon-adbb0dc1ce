"""Perspective, square symmetry, crop conversion and uncertain-pose gates."""

import copy, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import scene_config as C
from test_support import camera_model as model
from cube_pose import edge_pose, square_yaw, refine_pose, silhouette
from plan_sort import build_plan
from perceive import inspect_cube_poses
from sort_scene import validate_plan, verify_cube_stack


class CubePoseTests(unittest.TestCase):
    def setUp(self):
        self.cam = model(C.TABLE_CAM)

    def detection(self, xy, colour, yaw):
        t = np.deg2rad(yaw)
        r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        ends = [
            (
                self.cam.project(
                    [
                        *(np.array(xy) + r @ np.array([x, -C.CUBE_EDGE / 2])),
                        self.cam.table_z + C.CUBE_EDGE,
                    ]
                )
                / [self.cam.W, self.cam.H]
            ).tolist()
            for x in (-C.CUBE_EDGE / 2, C.CUBE_EDGE / 2)
        ]
        return dict(
            category="cube",
            colour=colour,
            center=(
                self.cam.project([*xy, self.cam.table_z + C.CUBE_EDGE / 2])
                / [self.cam.W, self.cam.H]
            ).tolist(),
            top_edge_endpoints=ends,
            pose_confidence=0.95,
        )

    def inventory(self):
        return [
            self.detection((x, y), c, a)
            for (x, y, c), a in zip(C.SORT_CUBES, C.SORT_CUBE_YAWS_DEG)
        ]

    def test_perspective_rotations_drive_wrist_and_return_to_aligned_stack(self):
        objects = self.inventory()
        plan = build_plan(
            {"objects": objects, "coordinate_convention": "unit"}, self.cam
        )
        validate_plan(plan)
        self.assertEqual(len(plan["picks"]), 6)
        for p in plan["picks"]:
            yaw = C.SORT_CUBE_YAWS_DEG[p["detection_id"]]
            self.assertAlmostEqual(square_yaw(p["yaw_deg"] - yaw), 0, places=8)
            self.assertAlmostEqual(
                np.dot(
                    p["spread_dir"][:2],
                    [np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))],
                ),
                0,
                places=8,
            )
            np.testing.assert_allclose(p["place_spread_dir"], [0, 1, 0])

    def test_missing_low_confidence_and_diagonal_pose_stop_stack(self):
        for changes in (
            {"top_edge_endpoints": None},
            {"pose_confidence": 0.4},
            {"top_edge_endpoints": [[0, 0], [1, 1]]},
        ):
            objects = self.inventory()
            objects[0].update(changes)
            plan = build_plan(
                {"objects": objects, "coordinate_convention": "unit"}, self.cam
            )
            self.assertFalse(plan["stack_ready"])
            self.assertEqual(plan["picks"], [])

    def test_reversed_edge_and_square_symmetry(self):
        obj = self.detection((0.35, 0.15), "red", 125)
        self.assertAlmostEqual(edge_pose(self.cam, obj["top_edge_endpoints"]), 35)
        self.assertAlmostEqual(
            edge_pose(self.cam, list(reversed(obj["top_edge_endpoints"]))), 35
        )

    def test_stack_rejects_rotated_or_tilted_layer(self):
        entries = [{"tracked_object": "cube", "stack_level": 0}]
        pose = dict(
            centre=[*C.CUBE_STACK_XY, C.TABLE_TOP_Z + C.SURFACE_DZ + C.CUBE_EDGE / 2],
            yaw_deg=0,
            tilt_deg=0,
        )
        self.assertTrue(verify_cube_stack({"cube": pose}, entries))
        self.assertFalse(verify_cube_stack({"cube": dict(pose, yaw_deg=20)}, entries))
        self.assertFalse(verify_cube_stack({"cube": dict(pose, tilt_deg=9)}, entries))

    def test_executor_rejects_tampered_wrist(self):
        plan = build_plan(
            {"objects": self.inventory(), "coordinate_convention": "unit"}, self.cam
        )
        plan["picks"][0]["spread_dir"] = [1, 0, 0]
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_image_fit_recovers_rotated_cube_without_scene_poses(self):
        xy = np.array([0.35, 0.15])
        yaw = 23.0
        mask = silhouette(self.cam, xy, yaw, (self.cam.H, self.cam.W))
        img = np.full((self.cam.H, self.cam.W, 3), 240, np.uint8)
        img[mask > 0] = [45, 35, 165]
        fitted, angle, quality = refine_pose(
            img, self.cam, {"colour": "red"}, xy + [0.001, -0.001], yaw + 8
        )
        self.assertLess(np.linalg.norm(fitted - xy), 0.001)
        self.assertLess(abs(square_yaw(angle - yaw)), 3)
        self.assertGreater(quality, 0.95)

    def test_full_yaw_fit_recentres_approximate_model_detection(self):
        xy = np.array([0.35, 0.15])
        mask = silhouette(self.cam, xy, -31, (self.cam.H, self.cam.W))
        img = np.full((self.cam.H, self.cam.W, 3), 240, np.uint8)
        img[mask > 0] = [165, 65, 35]
        fitted, angle, quality = refine_pose(
            img,
            self.cam,
            {"colour": "blue"},
            xy + [0, -0.006],
            0,
            search_half_width=45,
        )
        self.assertLess(np.linalg.norm(fitted - xy), 0.001)
        self.assertLess(abs(square_yaw(angle + 31)), 3)
        self.assertGreater(quality, 0.95)

    def test_cosmos_reports_calibrated_image_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            im = root / "im.png"
            cv2.imwrite(str(im), np.zeros((1080, 1920, 3), np.uint8))
            (root / "camera.json").write_text(json.dumps(self.cam.j))
            answer = {
                "category": "cube",
                "colour": "blue",
                "yaw_deg": 17.0,
                "pose_confidence": 0.9,
            }
            obj = dict(category="cube", colour="blue", center=[0.5, 0.5])
            with patch("perceive.ask", return_value=json.dumps(answer)), patch(
                "cube_pose.refine_pose",
                return_value=(np.array([0.35, 0.15]), 17.0, 0.95),
            ):
                parsed = inspect_cube_poses(None, None, im, {"objects": [obj]}, root)
            observed = parsed["objects"][0]
            expected = 17.0
            self.assertAlmostEqual(
                edge_pose(self.cam, observed["top_edge_endpoints"]), expected
            )
            self.assertEqual(
                observed["pose_source"], "Cosmos_reports_calibrated_image_pose"
            )
            self.assertTrue((root / "cube_pose_crops/cube_0_measurement.json").exists())
            np.testing.assert_allclose(
                observed["pose_world"]["position_m"],
                [0.35, 0.15, self.cam.table_z + C.CUBE_EDGE / 2],
            )

    def test_cosmos_cannot_replace_measurement_with_invented_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            im = root / "im.png"
            cv2.imwrite(str(im), np.zeros((1080, 1920, 3), np.uint8))
            (root / "camera.json").write_text(json.dumps(self.cam.j))
            for answer in (
                {
                    "category": "cube",
                    "colour": "blue",
                    "yaw_deg": -30.0,
                    "pose_confidence": 0.99,
                },
                {
                    "category": "unknown",
                    "colour": "unknown",
                    "yaw_deg": 17.0,
                    "pose_confidence": 0.2,
                },
            ):
                with patch("perceive.ask", return_value=json.dumps(answer)), patch(
                    "cube_pose.refine_pose",
                    return_value=(np.array([0.35, 0.15]), 17.0, 0.95),
                ):
                    result = inspect_cube_poses(
                        None,
                        None,
                        im,
                        {
                            "objects": [
                                dict(category="cube", colour="blue", center=[0.5, 0.5])
                            ]
                        },
                        root,
                    )
                self.assertIsNone(result["objects"][0]["top_edge_endpoints"])
                self.assertEqual(result["objects"][0]["pose_confidence"], 0)

    def test_uncertain_geometry_does_not_prompt_model_or_fabricate_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            im = root / "im.png"
            cv2.imwrite(str(im), np.zeros((1080, 1920, 3), np.uint8))
            (root / "camera.json").write_text(json.dumps(self.cam.j))
            with patch("perceive.ask") as ask, patch(
                "cube_pose.refine_pose",
                return_value=(np.array([0.35, 0.15]), 17.0, 0.3),
            ):
                result = inspect_cube_poses(
                    None,
                    None,
                    im,
                    {
                        "objects": [
                            dict(category="cube", colour="blue", center=[0.5, 0.5])
                        ]
                    },
                    root,
                )
            ask.assert_not_called()
            self.assertIsNone(result["objects"][0]["top_edge_endpoints"])


if __name__ == "__main__":
    unittest.main()
