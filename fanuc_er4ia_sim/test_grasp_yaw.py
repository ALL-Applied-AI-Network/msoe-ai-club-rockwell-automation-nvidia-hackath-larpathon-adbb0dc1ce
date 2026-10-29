"""Host regressions for the perception-only modulo-90 grasp-yaw estimator."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import scene_config as C
from cube_pose import silhouette
from grasp_yaw import (
    AUTHORIZATION_UNCERTAINTY_LIMIT_DEG,
    angular_distance_mod90,
    estimate_detection_yaw,
    estimate_inventory_yaws,
    normalize_yaw_mod90,
    run_yaw_estimator,
    scaled_camera_workspace,
)
from grasp_yaw_evaluate import evaluate_yaw_predictions
from perception_geometry import simulation_source_workspace
from test_support import camera_model


class GraspYawTests(unittest.TestCase):
    def setUp(self):
        self.cam = camera_model(C.TABLE_CAM)
        self.workspace = simulation_source_workspace(self.cam.j)

    def synthetic(self, yaw=17.0, xy=(0.36, 0.04), color="red", value=35):
        mask = silhouette(
            self.cam, np.asarray(xy, float), yaw, (self.cam.H, self.cam.W)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea).reshape((-1, 2))
        image = np.full((self.cam.H, self.cam.W, 3), 235, np.uint8)
        image[mask > 0] = value
        row = self.detection("cube-0", xy, color, contour)
        return image, row

    @staticmethod
    def detection(identifier, xy, color, contour):
        return {
            "detection_id": identifier,
            "accepted": True,
            "category": "cube",
            "color": color,
            "world_xy_m": list(map(float, xy)),
            "contour_polygon": np.asarray(contour, int).tolist(),
            "workspace_containment": {"passed": True},
            "color_confidence": 0.99,
            "classification_confidence": 0.99,
            "pose_confidence": 0.99,
            "pose_fit_iou": 0.99,
            "expected_size_agreement": 0.99,
            "footprint_agreement": 0.99,
        }

    def inventory(self, rows):
        result = []
        colors = ("red", "red", "blue", "blue", "yellow", "yellow")
        for index, color in enumerate(colors):
            row = copy.deepcopy(rows[index % len(rows)])
            row["detection_id"] = f"cube-{index}"
            row["color"] = color
            result.append(row)
        return {"inventory_valid": True, "accepted_detections": result}

    def test_modulo_90_normalization_and_square_symmetry(self):
        for source, expected in (
            (-225, -45),
            (-90, 0),
            (-46, 44),
            (45, -45),
            (90, 0),
            (134, 44),
        ):
            self.assertAlmostEqual(normalize_yaw_mod90(source), expected)
        self.assertAlmostEqual(angular_distance_mod90(43, -43), 4)
        self.assertAlmostEqual(angular_distance_mod90(17, 107), 0)

    def test_known_synthetic_orientations(self):
        for yaw in (-31.0, 0.0, 23.0):
            with self.subTest(yaw=yaw):
                image, detection = self.synthetic(yaw=yaw)
                result = estimate_detection_yaw(
                    image, self.cam, self.workspace, detection
                )
                self.assertTrue(result["yaw_valid"], result)
                self.assertLessEqual(
                    angular_distance_mod90(
                        result["estimated_yaw_mod_90_deg"], yaw
                    ),
                    1.0,
                )
                self.assertLessEqual(
                    result["estimated_uncertainty_deg"],
                    AUTHORIZATION_UNCERTAINTY_LIMIT_DEG,
                )

    def test_low_contrast_is_invalid(self):
        image, detection = self.synthetic(value=230)
        result = estimate_detection_yaw(image, self.cam, self.workspace, detection)
        self.assertFalse(result["yaw_valid"])
        self.assertIn("insufficient_photometric_contrast", result["invalidity_reasons"])
        self.assertIsNone(result["executable_yaw_mod_90_deg"])

    def test_low_resolution_symmetric_evidence_is_ambiguous(self):
        camera, workspace = scaled_camera_workspace(
            self.cam.j, self.workspace, (320, 180)
        )
        cam = type(self.cam)(camera)
        xy = np.array([0.36, 0.04])
        mask = silhouette(cam, xy, 17, (cam.H, cam.W)).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea).reshape((-1, 2))
        image = np.full((cam.H, cam.W, 3), 235, np.uint8)
        image[mask > 0] = 35
        detection = self.detection("cube-0", xy, "red", contour)
        result = estimate_detection_yaw(image, cam, workspace, detection)
        self.assertFalse(result["yaw_valid"])
        self.assertGreater(result["estimated_uncertainty_deg"], 5.0)
        self.assertIn(
            "competing_orientation_not_separated", result["invalidity_reasons"]
        )

    def test_resolution_scaling_preserves_world_projection(self):
        camera, workspace = scaled_camera_workspace(
            self.cam.j, self.workspace, (3840, 2160)
        )
        cam4k = type(self.cam)(camera)
        point = [0.36, 0.04, self.cam.table_z + 0.005]
        np.testing.assert_allclose(cam4k.project(point), 2 * self.cam.project(point))
        uv = cam4k.project(point)
        np.testing.assert_allclose(cam4k.to_plane(*uv, z=point[2]), point, atol=1e-10)
        self.assertEqual(workspace["compatible_calibration_sha256"], camera["calibration_sha256"])

    def test_four_k_reduces_synthetic_uncertainty(self):
        image, detection = self.synthetic(yaw=23)
        baseline = estimate_detection_yaw(image, self.cam, self.workspace, detection)
        camera, workspace = scaled_camera_workspace(
            self.cam.j, self.workspace, (3840, 2160)
        )
        cam4k = type(self.cam)(camera)
        xy = np.asarray(detection["world_xy_m"])
        mask = silhouette(cam4k, xy, 23, (cam4k.H, cam4k.W)).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea).reshape((-1, 2))
        image4k = np.full((cam4k.H, cam4k.W, 3), 235, np.uint8)
        image4k[mask > 0] = 35
        detection4k = self.detection("cube-0", xy, "red", contour)
        result4k = estimate_detection_yaw(image4k, cam4k, workspace, detection4k)
        self.assertTrue(result4k["yaw_valid"])
        self.assertLess(result4k["estimated_uncertainty_deg"], baseline["estimated_uncertainty_deg"])

    def test_all_six_authorization_fails_if_one_yaw_is_unavailable(self):
        image, row = self.synthetic(yaw=17)
        detections = self.inventory([row])
        complete = estimate_inventory_yaws(image, self.cam, self.workspace, detections)
        self.assertTrue(complete["all_six_yaws_authorized"])
        original = estimate_detection_yaw

        def one_ambiguous(image_arg, cam_arg, workspace_arg, detection_arg):
            result = original(image_arg, cam_arg, workspace_arg, detection_arg)
            if detection_arg["detection_id"] == "cube-0":
                result["yaw_valid"] = False
                result["executable_yaw_mod_90_deg"] = None
                result["estimated_uncertainty_deg"] = 5.25
                result["validity_reasons"] = []
                result["invalidity_reasons"] = ["yaw_uncertainty_exceeds_5_deg"]
            return result

        with patch("grasp_yaw.estimate_detection_yaw", one_ambiguous):
            incomplete = estimate_inventory_yaws(
                image, self.cam, self.workspace, detections
            )
        self.assertEqual(incomplete["valid_yaw_count"], 5)
        self.assertFalse(incomplete["all_six_yaws_authorized"])

    def test_inventory_rechecks_duplicate_and_frozen_gates(self):
        image, row = self.synthetic()
        detections = self.inventory([row])
        detections["accepted_detections"][1]["detection_id"] = "cube-0"
        detections["accepted_detections"][2]["pose_confidence"] = 0.0
        with self.assertRaisesRegex(ValueError, "detection_ids_not_unique"):
            estimate_inventory_yaws(image, self.cam, self.workspace, detections)

    def test_deterministic_reproduction(self):
        image, detection = self.synthetic(yaw=-19)
        first = estimate_detection_yaw(image, self.cam, self.workspace, detection)
        second = estimate_detection_yaw(image, self.cam, self.workspace, detection)
        self.assertEqual(first, second)

    def test_runner_reads_only_declared_inference_inputs(self):
        image, row = self.synthetic(yaw=17)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "image": root / "rgb.png",
                "camera": root / "camera.json",
                "workspace": root / "workspace.json",
                "detections": root / "detections.json",
                "output": root / "yaw.json",
            }
            cv2.imwrite(str(paths["image"]), image)
            paths["camera"].write_text(json.dumps(self.cam.j), encoding="utf-8")
            paths["workspace"].write_text(json.dumps(self.workspace), encoding="utf-8")
            paths["detections"].write_text(
                json.dumps(self.inventory([row])), encoding="utf-8"
            )
            original = Path.read_text
            opened = []

            def audited_read(path, *args, **kwargs):
                opened.append(Path(path).name)
                return original(path, *args, **kwargs)

            with patch.object(Path, "read_text", audited_read):
                result = run_yaw_estimator(**{
                    f"{key}_path": value for key, value in paths.items()
                })
            # CameraModel uses the built-in file API; all Path reads are the two
            # other explicitly declared inputs.  No truth/state/oracle path is
            # accepted by the runner's call signature.
            self.assertEqual(opened, ["workspace.json", "detections.json"])
            self.assertFalse(
                {"ground_truth.json", "scene_state.json", "oracle_plan.json"}
                & set(opened)
            )
            self.assertFalse(result["ground_truth_used"])
            self.assertFalse(result["simulator_state_used"])
            self.assertFalse(result["oracle_plan_used"])
            self.assertFalse(result["object_names_used"])

    def test_evaluator_opens_truth_only_after_predictions_and_matches_modulo_90(self):
        predictions = {
            "ground_truth_used": False,
            "simulator_state_used": False,
            "oracle_plan_used": False,
            "object_names_used": False,
            "yaw_estimates": [{
                "detection_id": "cube-0",
                "color": "red",
                "input_world_xy_m": [0.36, 0.04],
                "estimated_yaw_mod_90_deg": -44.0,
                "yaw_valid": True,
                "estimated_uncertainty_deg": 2.0,
                "confidence": 0.8,
            }],
        }
        truth = {"objects": [{
            "name": "evaluator-only-name",
            "category": "cube",
            "xyz": [0.36, 0.04, 0.9],
            "yaw_deg": 44.0,
        }]}
        result = evaluate_yaw_predictions(predictions, truth)
        self.assertTrue(result["predictions_written_before_evaluation"])
        self.assertAlmostEqual(result["matches"][0]["yaw_error_mod_90_deg"], 2.0)
        self.assertFalse(result["matches"][0]["false_valid_above_5_deg"])


if __name__ == "__main__":
    unittest.main()
