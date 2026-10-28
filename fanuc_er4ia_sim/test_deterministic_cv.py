"""Host regressions for isolated deterministic Dataset 3 perception."""

import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import scene_config as C
from cube_pose import silhouette
from deterministic_cv import (
    _colour_mask,
    _hue_confidence,
    detect_image,
    finalize_inventory,
    run_detector,
    suppress_duplicates,
)
from deterministic_cv_evaluate import evaluate_predictions
from test_support import camera_model


BGR = {"red": (35, 35, 180), "blue": (180, 65, 35), "yellow": (35, 190, 210)}


class DeterministicCVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cam = camera_model(C.TABLE_CAM)

    def blank(self):
        return np.full((self.cam.H, self.cam.W, 3), 240, np.uint8)

    def paint_cube(self, image, xy, yaw, colour):
        image[silhouette(self.cam, xy, yaw, image.shape[:2]) > 0] = BGR[colour]

    def accepted_stub(self, identifier, colour, xy, score=0.9):
        return {
            "candidate_id": identifier,
            "accepted": True,
            "acceptance_reasons": ["valid"],
            "rejection_reasons": [],
            "color": colour,
            "world_xy_m": list(xy),
            "pixel_centroid": [100.0, 100.0],
            "pose_confidence": score,
            "classification_confidence": score,
        }

    def test_each_cube_colour_mask(self):
        for colour, bgr in BGR.items():
            image = np.full((24, 24, 3), bgr, np.uint8)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            self.assertEqual(np.count_nonzero(_colour_mask(hsv, colour)), 24 * 24)
            for other in set(BGR) - {colour}:
                self.assertEqual(np.count_nonzero(_colour_mask(hsv, other)), 0)
            scene = self.blank()
            self.paint_cube(scene, (0.36, 0.07), 18, colour)
            result = detect_image(scene, self.cam)
            self.assertEqual([d["color"] for d in result["accepted_detections"]], [colour])

    def test_red_hue_wraparound(self):
        hsv = np.zeros((8, 16, 3), np.uint8)
        hsv[:, :8] = (1, 220, 180)
        hsv[:, 8:] = (179, 220, 180)
        self.assertEqual(np.count_nonzero(_colour_mask(hsv, "red")), hsv.shape[0] * hsv.shape[1])

    def test_yellow_cube_passes_while_circular_cap_is_rejected_by_geometry(self):
        image = self.blank()
        self.paint_cube(image, (0.36, 0.07), 18, "yellow")
        cap_uv = self.cam.project([0.46, -0.10, self.cam.table_z + C.CUBE_EDGE / 2])
        cv2.circle(image, tuple(np.rint(cap_uv).astype(int)), 18, BGR["yellow"], -1)
        result = detect_image(image, self.cam)
        self.assertEqual(len(result["accepted_detections"]), 1)
        self.assertEqual(result["accepted_detections"][0]["color"], "yellow")
        cap_rejections = [
            c for c in result["rejected_candidates"]
            if c["color"] == "yellow" and c["circularity"] > 0.75
        ]
        self.assertTrue(cap_rejections)
        self.assertIn("projected_size_inconsistent_with_10_mm_cube", cap_rejections[0]["rejection_reasons"])
        self.assertEqual(cap_rejections[0]["diagnostic_label"], "yellow_circular_oversize_candidate")

    def test_clear_vial_is_not_a_colour_candidate(self):
        image = self.blank()
        cv2.rectangle(image, (600, 500), (900, 525), (210, 215, 215), -1)
        result = detect_image(image, self.cam)
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["accepted_cube_count"], 0)

    def test_expected_size_rejection(self):
        image = self.blank()
        cv2.rectangle(image, (700, 500), (780, 580), BGR["blue"], -1)
        result = detect_image(image, self.cam)
        self.assertEqual(result["accepted_cube_count"], 0)
        self.assertTrue(any(
            "projected_size_inconsistent_with_10_mm_cube" in c["rejection_reasons"]
            for c in result["rejected_candidates"]
        ))

    def test_duplicate_suppression_keeps_best_pose(self):
        weak = self.accepted_stub("weak", "red", (0.3, 0.1), 0.8)
        strong = self.accepted_stub("strong", "red", (0.302, 0.101), 0.95)
        kept, rejected = suppress_duplicates([weak, strong])
        self.assertEqual([c["candidate_id"] for c in kept], ["strong"])
        self.assertEqual([c["candidate_id"] for c in rejected], ["weak"])
        self.assertIn("duplicate_of_strong", weak["rejection_reasons"])

    def test_low_confidence_cube_like_component_is_rejected(self):
        image = self.blank()
        mask = silhouette(self.cam, (0.36, 0.07), 18, image.shape[:2]) > 0
        muted_hsv = np.zeros_like(image)
        muted_hsv[:] = (8, 50, 45)
        muted_bgr = cv2.cvtColor(muted_hsv, cv2.COLOR_HSV2BGR)
        image[mask] = muted_bgr[mask]
        result = detect_image(image, self.cam)
        self.assertEqual(result["accepted_cube_count"], 0)
        self.assertTrue(any(
            "colour_confidence_too_low" in c["rejection_reasons"]
            for c in result["rejected_candidates"]
        ))

    def test_incomplete_inventory_fails_closed(self):
        result = finalize_inventory([self.accepted_stub("one", "red", (0.2, 0.1))])
        self.assertFalse(result["inventory_valid"])
        self.assertEqual(result["executable_cube_inventory"], [])

    def test_wrong_two_two_two_inventory_fails_closed(self):
        colours = ["red", "red", "red", "blue", "yellow", "yellow"]
        candidates = [
            self.accepted_stub(str(i), colour, (0.20 + i * 0.02, 0.1))
            for i, colour in enumerate(colours)
        ]
        result = finalize_inventory(candidates)
        self.assertEqual(result["accepted_cube_count"], 6)
        self.assertFalse(result["inventory_valid"])
        self.assertEqual(result["executable_cube_inventory"], [])

    def test_evaluator_uses_one_to_one_matching(self):
        predictions = {
            "inference_source": "deterministic_cv",
            "accepted_detections": [
                dict(self.accepted_stub("a", "red", (0.300, 0.100)), detection_id="a", defect_class="Defect Class A",
                     yaw_valid=False, yaw_mod_90_deg=None, color_confidence=.9, localization_confidence=.9, pose_fit_iou=.9),
                dict(self.accepted_stub("b", "red", (0.301, 0.100)), detection_id="b", defect_class="Defect Class A",
                     yaw_valid=False, yaw_mod_90_deg=None, color_confidence=.9, localization_confidence=.9, pose_fit_iou=.9),
            ],
        }
        for detection in predictions["accepted_detections"]:
            detection.setdefault("pose_confidence", .9)
            detection.setdefault("classification_confidence", .9)
        truth = {"objects": [
            {"name": "cube0", "category": "cube", "colour": "red", "xyz": [0.300, 0.100, .005], "yaw_deg": 0},
            {"name": "cube1", "category": "cube", "colour": "red", "xyz": [0.315, 0.100, .005], "yaw_deg": 0},
        ]}
        result = evaluate_predictions(predictions, truth)
        self.assertEqual(len(result["matched_cubes"]), 2)
        self.assertEqual(len({m["ground_truth_name"] for m in result["matched_cubes"]}), 2)

    def test_detector_io_cannot_read_truth_or_scene_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path, camera_path = root / "image.png", root / "camera.json"
            cv2.imwrite(str(image_path), self.blank())
            camera_path.write_text(json.dumps(self.cam.j))
            (root / "ground_truth.json").write_text("forbidden")
            (root / "scene_state.json").write_text("forbidden")
            original_open = builtins.open

            def guarded_open(file, *args, **kwargs):
                if Path(file).name in {"ground_truth.json", "scene_state.json", "settled_execution_start_state.json"}:
                    raise AssertionError("detector attempted evaluator-only input")
                return original_open(file, *args, **kwargs)

            with patch("builtins.open", guarded_open):
                result = run_detector(image_path, camera_path, root / "detections.json")
            self.assertFalse(result["ground_truth_used"])
            self.assertFalse(result["simulator_state_used"])


if __name__ == "__main__":
    unittest.main()
