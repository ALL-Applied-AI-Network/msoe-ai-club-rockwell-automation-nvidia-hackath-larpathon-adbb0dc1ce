"""Host-only contract tests for Phase 4B randomized validation."""

from __future__ import annotations

import inspect
import json
import math
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import phase4b_capture
import phase4b_design as P
import phase4b_process


class Phase4BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = P.generate_batch_design()

    def test_exact_seed_set_and_deterministic_reproduction(self):
        self.assertEqual(self.batch["seeds"], list(range(4100, 4112)))
        self.assertEqual(self.batch, P.generate_batch_design())
        self.assertEqual(P.generate_scene(4100), P.generate_scene(4100))
        self.assertNotEqual(P.generate_scene(4100)["objects"], P.generate_scene(4101)["objects"])
        for scene in self.batch["scenes"]:
            self.assertEqual(scene["design_sha256"], P.scene_design_sha256(scene))

    def test_object_count_and_colour_invariants(self):
        for scene in self.batch["scenes"]:
            categories = [obj["category"] for obj in scene["objects"]]
            colours = [obj["colour"] for obj in scene["objects"] if obj["category"] == "cube"]
            self.assertEqual({kind: categories.count(kind) for kind in set(categories)}, {"cube": 6, "cap": 2, "vial": 2})
            self.assertEqual({colour: colours.count(colour) for colour in set(colours)}, {"red": 2, "blue": 2, "yellow": 2})

    def test_non_overlap_and_minimum_separation(self):
        for scene in self.batch["scenes"]:
            objects = scene["objects"]
            for index, first in enumerate(objects):
                for second in objects[index + 1:]:
                    self.assertGreaterEqual(
                        P.object_clearance(first, second) + 1e-9,
                        P.MINIMUM_EDGE_CLEARANCE_M,
                        f"seed={scene['seed']} {first['name']} {second['name']}",
                    )

    def test_zone_table_camera_and_reach_containment(self):
        for scene in self.batch["scenes"]:
            self.assertEqual(P.validate_scene_design(scene), [])
            for obj in scene["objects"]:
                self.assertTrue(P._outside_zones(obj))
                self.assertTrue(P._within_workspace(obj))
                self.assertTrue(P._camera_contained(obj))

    def test_every_scene_has_valid_yellow_cap_challenge(self):
        for scene in self.batch["scenes"]:
            challenge = scene["yellow_cap_challenge"]
            self.assertIn(challenge["yellow_cube"], {"SortCube4", "SortCube5"})
            self.assertEqual(challenge["cap"], "SortCap0")
            sampled = challenge["sampled_edge_clearance_m"]
            self.assertGreaterEqual(sampled, P.CHALLENGE_CAP_CLEARANCE_RANGE_M[0])
            self.assertLessEqual(sampled, P.CHALLENGE_CAP_CLEARANCE_RANGE_M[1])
            objects = {obj["name"]: obj for obj in scene["objects"]}
            self.assertGreaterEqual(
                P.object_clearance(objects[challenge["yellow_cube"]], objects[challenge["cap"]]),
                P.MINIMUM_EDGE_CLEARANCE_M + P.GENERATION_RESERVE_M - 1e-9,
            )

    def test_stable_only_capture_gate(self):
        base = {"timed_out": False, "scene_stable_before_commands": True}
        self.assertFalse(P.capture_is_authorized(dict(base, consecutive_stable_window_count=1)))
        self.assertTrue(P.capture_is_authorized(dict(base, consecutive_stable_window_count=2)))
        self.assertFalse(P.capture_is_authorized(dict(base, consecutive_stable_window_count=2, timed_out=True)))
        source = inspect.getsource(phase4b_capture.main)
        self.assertLess(source.index("capture_is_authorized(stability)"), source.index("settled_scene_state.json"))
        self.assertLess(source.index("settled_geometry = geometry_report"), source.index("ground_truth.json"))
        self.assertLess(source.index("settled_scene_state.json"), source.index("world_camera.png"))
        self.assertLess(source.index('if not settled_geometry["passed"]'), source.index("camera = Camera"))

    def test_failed_seed_is_recorded_without_commands(self):
        record = P.record_failed_seed(4107, "timeout")
        self.assertEqual(record["seed"], 4107)
        self.assertEqual(record["capture_status"], "generation_failed")
        self.assertFalse(record["stability_outcome"])
        self.assertFalse(record["robot_commands_issued"])

    def test_capture_module_defers_simulator_imports(self):
        source = Path(phase4b_capture.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from isaacsim import SimulationApp\n\n", source.split("def main", 1)[0])
        self.assertIn("from isaacsim import SimulationApp", inspect.getsource(phase4b_capture.main))

    def test_frozen_detector_hash_and_configuration_are_reproducible(self):
        root = Path(P.__file__).resolve().parent
        first = P.frozen_detector_record(root)
        second = P.frozen_detector_record(root)
        self.assertEqual(first, second)
        self.assertEqual(set(first["source_files"]), set(P.DETECTOR_SOURCE_FILES))
        self.assertEqual(len(first["source_sha256"]), 64)
        self.assertEqual(len(first["configuration_sha256"]), 64)
        self.assertEqual(
            first["appearance_configuration_sha256"],
            "50bf09ac9e2a6b8b38cc658d0d2f87e942aca630c70409ba99923764cdf51356",
        )
        self.assertEqual(len(first["workspace_configuration_sha256"]), 64)
        self.assertEqual(
            first["workspace_configuration"]["coordinate_frame"],
            "fanuc_er4ia_robot_base_world",
        )
        self.assertEqual(len(first["combined_sha256"]), 64)
        self.assertEqual(first["configuration"], P.detector_configuration())

    def test_phase4b_inference_completes_before_truth_evaluation(self):
        source = inspect.getsource(phase4b_process.process_batch)
        pass_one = source.index("# Pass 1")
        detector_call = source.index("run_detector(", pass_one)
        pass_two = source.index("# Pass 2")
        evaluator_call = source.index("run_evaluator(", pass_two)
        self.assertLess(pass_one, detector_call)
        self.assertLess(detector_call, pass_two)
        self.assertLess(pass_two, evaluator_call)
        self.assertIn("ground_truth_used", source)
        self.assertIn("simulator_state_used", source)
        self.assertIn("oracle_plan_used", source)

    def _evaluation(self, error_mm=0.5, benchmark=True):
        return {
            "benchmark_passed": benchmark,
            "detected_cube_count": 6,
            "counts_by_color": {"red": 2, "blue": 2, "yellow": 2},
            "false_positive_count": 0,
            "missed_cube_count": 0,
            "matched_cubes": [
                {"xy_error_mm": error_mm, "color_correct": True, "class_correct": True}
                for _ in range(6)
            ],
            "distractor_rejection": [
                {"category": category, "accepted_as_cube": False}
                for category in ("cap", "cap", "vial", "vial")
            ],
        }

    def _records(self):
        return [
            {
                "seed": seed,
                "capture_status": "captured",
                "stability_outcome": True,
                "evaluation": self._evaluation(),
            }
            for seed in P.SEEDS
        ]

    def test_complete_aggregate_accounting_and_go(self):
        result = P.aggregate_evaluations(self._records())
        self.assertTrue(result["go"])
        self.assertEqual(result["scenes_generated"], 12)
        self.assertEqual(result["scenes_stabilized"], 12)
        self.assertEqual(result["scenes_passing_complete_inventory"], 12)
        self.assertEqual(result["total_accepted_cubes"], 72)
        self.assertEqual(result["total_cube_precision"], 1.0)
        self.assertEqual(result["total_cube_recall"], 1.0)
        self.assertEqual(result["xy_error_mm"]["count"], 72)

    def test_no_seed_omission_or_duplicate_is_allowed(self):
        records = self._records()
        with self.assertRaises(ValueError):
            P.aggregate_evaluations(records[:-1])
        duplicate = records[:-1] + [dict(records[0])]
        with self.assertRaises(ValueError):
            P.aggregate_evaluations(duplicate)

    def test_go_no_go_calculations_fail_closed(self):
        records = self._records()
        records[3]["evaluation"] = self._evaluation(error_mm=5.1, benchmark=False)
        result = P.aggregate_evaluations(records)
        self.assertFalse(result["go"])
        self.assertFalse(result["go_no_go_criteria"]["all_requested_complete_inventories"])
        self.assertFalse(result["go_no_go_criteria"]["maximum_xy_error_at_most_5_mm"])
        self.assertIn("4103", result["failure_categories_by_seed"])

    def test_diagnostic_noise_is_deterministic(self):
        image = np.full((20, 20, 3), 128, np.uint8)
        spec = P.DIAGNOSTIC_TRANSFORMS["gaussian_noise"]
        first = phase4b_process._transform(image, spec, 4100, 5)
        second = phase4b_process._transform(image, spec, 4100, 5)
        np.testing.assert_array_equal(first, second)

    def test_runtime_manifest_has_complete_unique_closure(self):
        root = Path(P.__file__).resolve().parent
        manifest = P.runtime_manifest(root)
        self.assertEqual(manifest["file_count"], len(P.PHASE4B_RUNTIME_FILES))
        paths = [row["path"] for row in manifest["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(len(row["sha256"]) == 64 for row in manifest["files"]))

    def test_vial_root_to_center_uses_local_positive_z(self):
        half = math.sqrt(0.5)
        pose = phase4b_capture._pose_from_world(
            "vial", [1.0, 2.0, 3.0], [half, 0.0, half, 0.0], P.C
        )
        np.testing.assert_allclose(pose["centre"], [1.0 + P.C.VIAL_LENGTH / 2, 2.0, 3.0])
        self.assertAlmostEqual(pose["yaw_deg"], 0.0, places=9)

    def test_vial_long_axis_yaw_ignores_arbitrary_roll(self):
        def rz(angle):
            c, s = math.cos(angle), math.sin(angle)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)

        ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
        for yaw_deg in (-89.0, -45.0, 0.0, 37.0, 89.0, 135.0):
            for roll_deg in (-123.0, 0.0, 77.0):
                rotation = rz(math.radians(yaw_deg)) @ ry90 @ rz(math.radians(roll_deg))
                expected = (yaw_deg + 90.0) % 180.0 - 90.0
                self.assertAlmostEqual(P.capsule_yaw_deg(rotation), expected, places=9)

    def test_historical_wrong_axis_failure_is_removed(self):
        cap = P._make_object("SortCap1", "cap", [0.3534425591727947, -0.12319910386762496], -72.28611251782048)
        vial = P._make_object("SortVial1", "vial", [0.32719677940136155, -0.07106816851795261], -162.89735513789188)
        correct_margin = P.object_clearance(cap, vial) - P.MINIMUM_EDGE_CLEARANCE_M
        wrong = dict(vial, yaw_deg=vial["yaw_deg"] + 90.0)
        wrong_margin = P.object_clearance(cap, wrong) - P.MINIMUM_EDGE_CLEARANCE_M
        self.assertAlmostEqual(correct_margin * 1000, 13.169151816077262, places=6)
        self.assertAlmostEqual(wrong_margin * 1000, -34.62072052837628, places=6)
        self.assertGreater(correct_margin, 0)
        self.assertLess(wrong_margin, 0)

    def test_all_historical_wrong_axis_failures_are_reproduced_then_removed(self):
        expected = {
            4100: ["clearance:SortCap1:SortVial1"],
            4101: ["clearance:SortCube0:SortVial0", "clearance:SortCube2:SortVial1", "clearance:SortCube3:SortVial0"],
            4102: ["clearance:SortCap1:SortVial1"],
            4103: ["workspace:SortVial1"],
            4104: [],
            4105: ["workspace:SortVial1", "clearance:SortCube2:SortVial1"],
            4106: ["workspace:SortVial0", "workspace:SortVial1", "clearance:SortCap1:SortVial0"],
            4107: ["workspace:SortVial0", "clearance:SortCube2:SortVial1", "clearance:SortCube3:SortVial1"],
            4108: ["workspace:SortVial1", "clearance:SortCube5:SortVial0"],
            4109: ["clearance:SortCap1:SortVial1"],
            4110: ["workspace:SortVial1", "clearance:SortCube0:SortVial1", "clearance:SortCube2:SortVial0"],
            4111: ["workspace:SortVial0", "clearance:SortCube0:SortVial1", "clearance:SortCube5:SortVial1"],
        }
        for seed in P.SEEDS:
            historical = P.generate_scene(seed, generation_reserve_m=0.0)
            self.assertEqual(P.validate_scene_design(historical), [])
            wrong = copy.deepcopy(historical)
            for obj in wrong["objects"]:
                if obj["category"] == "vial":
                    obj["yaw_deg"] += 90.0
            self.assertEqual(P.validate_scene_design(wrong), expected[seed])

    def test_geometry_report_contains_signed_margins_and_motion(self):
        scene = P.generate_scene(4100)
        report = P.geometry_report(scene, release_scene=copy.deepcopy(scene))
        self.assertTrue(report["passed"])
        self.assertEqual(report["geometry_convention"]["vial_long_axis"], "rotated_local_positive_z")
        self.assertEqual(len(report["objects"]), 10)
        self.assertEqual(len(report["pairwise_clearances"]), 45)
        self.assertTrue(all("signed_clearance_margin_m" in row for row in report["pairwise_clearances"]))
        self.assertTrue(all(row["release_to_settle"] is not None for row in report["objects"]))

    def test_failure_evidence_is_written_before_rejection(self):
        source = inspect.getsource(phase4b_capture.main)
        written = source.index('_write_json(attempt_dir / "settled_geometry.json"')
        rejected = source.index('if not settled_geometry["passed"]')
        self.assertLess(written, rejected)

    def test_bounded_deterministic_attempt_accounting(self):
        seeds = [P.derive_attempt_seed(4100, index) for index in range(P.MAX_GENERATION_ATTEMPTS)]
        self.assertEqual(seeds[0], 4100)
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(seeds, [P.derive_attempt_seed(4100, index) for index in range(P.MAX_GENERATION_ATTEMPTS)])
        scene = P.generate_scene(4100, 1)
        record = P.generation_attempt_record(4100, 1, scene, {}, {}, ["settled:test"])
        self.assertEqual(record["derived_seed"], seeds[1])
        self.assertFalse(record["accepted"])
        self.assertFalse(record["detector_consulted"])

    def test_canonical_float_hash_precision_and_nonfinite_rejection(self):
        scene = P.generate_scene(4101)
        below = copy.deepcopy(scene)
        below["lighting_intensity_scales"]["DomeLight"] += 1e-14
        above = copy.deepcopy(scene)
        above["lighting_intensity_scales"]["DomeLight"] += 2e-12
        self.assertEqual(P.scene_design_sha256(scene), P.scene_design_sha256(below))
        self.assertNotEqual(P.scene_design_sha256(scene), P.scene_design_sha256(above))
        invalid = copy.deepcopy(scene)
        invalid["lighting_intensity_scales"]["DomeLight"] = float("nan")
        with self.assertRaises(ValueError):
            P.scene_design_sha256(invalid)

    def test_fresh_process_canonical_hash_reproduction(self):
        script = (
            "import json,phase4b_design as P;"
            "print(json.dumps([P.generate_scene(s)['design_sha256'] for s in P.SEEDS]))"
        )
        first = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
        second = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
        self.assertEqual(first, second)
        hashes = json.loads(first)
        self.assertEqual(len(hashes), 12)
        self.assertEqual(hashes[1], P.generate_scene(4101)["design_sha256"])
        self.assertEqual(
            hashes[1],
            "e3bf4038ab4c087e593ae023e2b5b7eef644161bb689bc6f611885fac192a47f",
        )

    def test_frame_content_validation(self):
        cases = [
            (None, "none"),
            (np.zeros((10, 10), np.uint8), "invalid_dimensions_or_channels"),
            (np.zeros((1080, 1920, 3), np.uint8), "all_black"),
            (np.full((1080, 1920, 3), 127, np.uint8), "constant_valued"),
            (np.full((1080, 1920, 4), [10, 20, 30, 255], np.uint8), "constant_valued"),
        ]
        for value, reason in cases:
            row = phase4b_capture.frame_telemetry(value, 1)
            self.assertFalse(row["accepted"])
            self.assertEqual(row["rejection_reason"], reason)
        valid = np.zeros((1080, 1920, 3), np.uint8)
        valid[0, 0] = [1, 2, 3]
        self.assertTrue(phase4b_capture.frame_telemetry(valid, 1)["accepted"])

    def test_bounded_capture_retries_and_telemetry(self):
        frames = iter([None, np.zeros((1080, 1920, 3), np.uint8), np.pad(
            np.ones((1, 1, 3), np.uint8), ((0, 1079), (0, 1919), (0, 0))
        )])
        sync = []
        frame, rows = phase4b_capture.bounded_frame_capture(
            lambda: next(frames), lambda: sync.append(True), max_attempts=3
        )
        self.assertIsNotNone(frame)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(sync), 3)
        self.assertEqual([row["attempt"] for row in rows], [1, 2, 3])

    def test_cleanup_order_and_feature_detection(self):
        events = []
        class Annotator:
            def detach(self, paths): events.append("detach")
        class Camera:
            _rgb_annotator = Annotator()
            def pause(self): events.append("pause")
        class Destroy:
            def render_product(self, path): events.append("destroy")
        class Orchestrator:
            def stop(self): events.append("stop")
            def wait_until_complete(self): events.append("wait")
            def reset(self): events.append("reset")
        class Rep:
            destroy = Destroy()
            orchestrator = Orchestrator()
        class World:
            @staticmethod
            def instance(): return object()
            @staticmethod
            def clear_instance(): events.append("clear_world")
        rows = phase4b_capture.cleanup_capture_resources(Camera(), "/Render/Test", Rep(), World, [])
        self.assertEqual([row["action"] for row in rows], list(phase4b_capture.CLEANUP_ORDER[:-1]))
        self.assertEqual(events, ["detach", "destroy", "stop", "wait", "reset", "pause", "clear_world"])

    def test_camera_is_deferred_and_owned_by_current_stage(self):
        source = inspect.getsource(phase4b_capture.main)
        self.assertLess(source.index('if not settled_geometry["passed"]'), source.index("camera = Camera"))
        self.assertIn("camera_owned_by_stage", source)
        self.assertIn("render_product_owned_by_stage", source)


if __name__ == "__main__":
    unittest.main()
