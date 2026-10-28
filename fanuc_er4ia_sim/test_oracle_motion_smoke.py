"""Host-only tests for the isolated Phase 3A oracle motion smoke contract."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import scene_config as C
from oracle_motion_smoke import (
    NON_SELECTED_CUBE_ROTATION_LIMIT_DEG,
    NON_SELECTED_CUBE_TRANSLATION_LIMIT_M,
    EXECUTION_POSE_ROTATION_TOLERANCE_DEG,
    EXECUTION_POSE_TRANSLATION_TOLERANCE_M,
    ORACLE_SOURCE,
    ORACLE_TASK,
    SELECTED_CUBE,
    TARGET_SLOT,
    TRANSIT_CLEARANCE_M,
    build_oracle_smoke_plan,
    evaluate_smoke_success,
    preflight_oracle_motion,
    validate_execution_start,
    validate_oracle_smoke_plan,
)
from sort_scene import segments
from oracle_smoke_postprocess import (
    VIDEO_STATUS_FILE,
    record_video_status,
    validate_required_motion_outputs,
)


def scene_inputs():
    objects = {}
    truth = []
    for index, (x, y, colour) in enumerate(C.SORT_CUBES):
        name = f"SortCube{index}"
        objects[name] = {
            "position": [x, y, C.TABLE_TOP_Z + C.SURFACE_DZ],
            "orientation": [1, 0, 0, 0],
        }
        truth.append(
            {"name": name, "category": "cube", "colour": colour, "xyz": [x, y, 0.005]}
        )
    for index, (x, y) in enumerate(C.SORT_CAPS):
        name = f"SortCap{index}"
        objects[name] = {
            "position": [x, y, C.TABLE_TOP_Z + C.SURFACE_DZ],
            "orientation": [1, 0, 0, 0],
        }
        truth.append({"name": name, "category": "cap", "colour": "yellow"})
    for index, (x, y, _yaw) in enumerate(C.SORT_VIALS):
        name = f"SortVial{index}"
        objects[name] = {
            "position": [x, y, C.TABLE_TOP_Z + C.SURFACE_DZ],
            "orientation": [1, 0, 0, 0],
        }
        truth.append({"name": name, "category": "vial", "colour": "clear"})
    return {"objects": objects}, {"objects": truth}


def pose_map():
    values = {}
    for index in range(6):
        name = f"SortCube{index}"
        x, y, _colour = C.SORT_CUBES[index]
        values[name] = {
            "centre": [x, y, C.TABLE_TOP_Z + C.SURFACE_DZ + C.CUBE_EDGE / 2],
            "orientation": [1, 0, 0, 0],
        }
    for index, (x, y) in enumerate(C.SORT_CAPS):
        values[f"SortCap{index}"] = {
            "centre": [x, y, C.TABLE_TOP_Z + C.SURFACE_DZ + C.CAP_HEIGHT / 2],
            "orientation": [1, 0, 0, 0],
        }
    for index, (x, y, _yaw) in enumerate(C.SORT_VIALS):
        values[f"SortVial{index}"] = {
            "centre": [x, y, C.TABLE_TOP_Z + C.SURFACE_DZ + C.VIAL_RADIUS],
            "orientation": [1, 0, 0, 0],
        }
    return values


class OracleMotionSmokeTests(unittest.TestCase):
    def setUp(self):
        state, truth = scene_inputs()
        self.scene_state = state
        self.plan = build_oracle_smoke_plan(state, truth)

    def test_plan_is_explicitly_oracle_and_isolated_to_one_red_cube(self):
        self.assertTrue(validate_oracle_smoke_plan(self.plan))
        self.assertEqual(self.plan["task"], ORACLE_TASK)
        self.assertEqual(self.plan["source"], ORACLE_SOURCE)
        self.assertEqual(self.plan["commanded_objects"], [SELECTED_CUBE])
        self.assertEqual(len(self.plan["picks"]), 1)
        pick = self.plan["picks"][0]
        self.assertEqual(pick["pose_source"], "oracle_simulation_pose")
        self.assertEqual(pick["colour"], "red")
        self.assertEqual(pick["defect_class"], "Defect Class A")
        self.assertEqual(tuple(pick["assigned_slot"]), TARGET_SLOT)
        self.assertAlmostEqual(
            pick["expected_centre_z"], C.SURFACE_DZ + 0.5 * C.CUBE_EDGE
        )

    def test_plan_rejects_extra_wrong_or_unsafe_commands(self):
        cases = []
        extra = copy.deepcopy(self.plan)
        extra["picks"].append(copy.deepcopy(extra["picks"][0]))
        cases.append(extra)
        wrong = copy.deepcopy(self.plan)
        wrong["commanded_objects"] = [SELECTED_CUBE, "SortCube3"]
        cases.append(wrong)
        distractor = copy.deepcopy(self.plan)
        distractor["picks"][0]["category"] = "cap"
        cases.append(distractor)
        wrong_slot = copy.deepcopy(self.plan)
        wrong_slot["picks"][0]["assigned_slot"] = C.DEFECT_TARGET_SLOTS["red"][1]
        cases.append(wrong_slot)
        wrong_source = copy.deepcopy(self.plan)
        wrong_source["source"] = "Cosmos3-Nano"
        cases.append(wrong_source)
        wrong_clearance = copy.deepcopy(self.plan)
        wrong_clearance["transit_clearance_m"] = 0.129
        cases.append(wrong_clearance)
        for plan in cases:
            with self.subTest(plan=plan):
                with self.assertRaises(ValueError):
                    validate_oracle_smoke_plan(plan)

    def test_preflight_requires_reachable_130mm_clearance_sequence(self):
        home = np.array([0.30, 0.0, 0.24])
        motion = segments(self.plan["picks"][0], home)
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])

        def reachable(q, *_args, **_kwargs):
            return np.asarray(q, float), True

        with patch("oracle_motion_smoke.ik.solve", side_effect=reachable):
            result = preflight_oracle_motion(
                self.plan,
                home,
                tuple(np.asarray(C.TOOL_AXIS) * C.TOOL_LENGTH),
                limits,
                motion,
            )
        self.assertTrue(result["preflight_passed"])
        self.assertEqual(result["transit_clearance_m"], TRANSIT_CLEARANCE_M)
        self.assertGreater(result["minimum_geometric_clearance_margin_m"], 0.05)
        self.assertEqual(len(result["phase_ik_checks"]), 11)

        with patch(
            "oracle_motion_smoke.ik.solve",
            return_value=(np.zeros(6), False),
        ):
            failed = preflight_oracle_motion(
                self.plan,
                home,
                tuple(np.asarray(C.TOOL_AXIS) * C.TOOL_LENGTH),
                limits,
                motion,
            )
        self.assertFalse(failed["preflight_passed"])

    def test_matching_recreated_scene_passes_before_commands(self):
        result = validate_execution_start(
            self.plan, copy.deepcopy(self.scene_state["objects"])
        )
        self.assertTrue(result["pre_command_validation_passed"])
        self.assertEqual(result["object_counts"], {"cubes": 6, "caps": 2, "vials": 2})
        self.assertEqual(result["translation_error_mm"], 0)
        self.assertEqual(result["rotation_error_deg"], 0)
        self.assertEqual(result["planned_pose"], self.plan["picks"][0]["oracle_simulation_pose"])
        self.assertEqual(
            result["execution_start_pose"],
            self.scene_state["objects"][SELECTED_CUBE],
        )

        source = (Path(__file__).parent / "sort_scene.py").read_text(
            encoding="utf-8"
        )
        recreation_validation_at = source.index("scene_recreation_validation = (")
        home_initialization_at = source.index("arm.set_joint_positions(full)")
        post_settle_validation_at = source.index("execution_start_validation = (")
        stability_at = source.index("evaluate_scene_stability_windows(stability_windows)")
        command_authorization_at = source.index(
            "precommand_authorization = authorize_oracle_robot_commands("
        )
        execution_baseline_at = source.index("before = poses()")
        first_task_command_at = source.index("command(q, j0 + (j1 - j0) * t)")
        self.assertLess(recreation_validation_at, home_initialization_at)
        self.assertLess(home_initialization_at, stability_at)
        self.assertLess(stability_at, post_settle_validation_at)
        self.assertLess(post_settle_validation_at, command_authorization_at)
        self.assertLess(command_authorization_at, execution_baseline_at)
        self.assertLess(execution_baseline_at, first_task_command_at)

    def test_pre_command_translation_above_2mm_fails(self):
        current = copy.deepcopy(self.scene_state["objects"])
        current[SELECTED_CUBE]["position"][0] += (
            EXECUTION_POSE_TRANSLATION_TOLERANCE_M + 1e-5
        )
        result = validate_execution_start(self.plan, current)
        self.assertFalse(result["pre_command_validation_passed"])
        self.assertIn("selected_cube_translation_exceeded", result["failures"])

    def test_pre_command_rotation_above_5deg_fails(self):
        current = copy.deepcopy(self.scene_state["objects"])
        angle = np.deg2rad(EXECUTION_POSE_ROTATION_TOLERANCE_DEG + 0.01)
        current[SELECTED_CUBE]["orientation"] = [
            np.cos(angle / 2),
            0,
            0,
            np.sin(angle / 2),
        ]
        result = validate_execution_start(self.plan, current)
        self.assertFalse(result["pre_command_validation_passed"])
        self.assertIn("selected_cube_rotation_exceeded", result["failures"])

    def test_pre_command_missing_or_extra_object_fails(self):
        missing = copy.deepcopy(self.scene_state["objects"])
        missing.pop("SortCube5")
        extra = copy.deepcopy(self.scene_state["objects"])
        extra["UnexpectedCube"] = copy.deepcopy(extra[SELECTED_CUBE])
        for current in (missing, extra):
            with self.subTest(names=sorted(current)):
                result = validate_execution_start(self.plan, current)
                self.assertFalse(result["pre_command_validation_passed"])
                self.assertTrue(result["missing_objects"] or result["extra_objects"])

    def test_pre_command_wrong_colour_class_or_destination_fails(self):
        plans = []
        wrong_colour = copy.deepcopy(self.plan)
        wrong_colour["picks"][0]["colour"] = "blue"
        plans.append(wrong_colour)
        wrong_class = copy.deepcopy(self.plan)
        wrong_class["picks"][0]["defect_class"] = "Defect Class B"
        plans.append(wrong_class)
        wrong_destination = copy.deepcopy(self.plan)
        wrong_destination["picks"][0]["assigned_slot"] = list(
            C.DEFECT_TARGET_SLOTS["red"][1]
        )
        plans.append(wrong_destination)
        for plan in plans:
            with self.subTest(plan=plan):
                result = validate_execution_start(
                    plan, copy.deepcopy(self.scene_state["objects"])
                )
                self.assertFalse(result["pre_command_validation_passed"])
                self.assertTrue(
                    any(reason.startswith("plan_contract:") for reason in result["failures"])
                )

    def successful_result(self):
        before = pose_map()
        after = copy.deepcopy(before)
        after[SELECTED_CUBE]["centre"] = [
            *TARGET_SLOT,
            C.TABLE_TOP_Z + C.SURFACE_DZ + C.CUBE_EDGE / 2,
        ]
        event = {
            "tracked_object": SELECTED_CUBE,
            "pickup_verified": True,
            "placement_verified": True,
            "correct_zone_containment": True,
            "correct_zone_placement": True,
            "assigned_slot_error_mm": 0.5,
            "objects_lifted": [SELECTED_CUBE],
            "retries": [],
            "recoveries": [],
        }
        preflight = {"preflight_passed": True}
        return before, after, [event], preflight

    def test_success_requires_exact_pick_slot_height_and_untouched_objects(self):
        before, after, events, preflight = self.successful_result()
        result = evaluate_smoke_success(self.plan, events, before, after, preflight)
        self.assertTrue(result["smoke_test_success"])
        self.assertEqual(len(result["distractor_displacements"]), 4)
        self.assertEqual(len(result["non_selected_cube_displacements"]), 5)
        self.assertAlmostEqual(result["final_cube_height_m"], 0.8999)

    def test_all_runtime_abort_conditions_fail_success(self):
        mutators = {
            "lift_not_verified": lambda _b, _a, e, _p: e[0].update(
                pickup_verified=False
            ),
            "wrong_slot_or_zone": lambda _b, _a, e, _p: e[0].update(
                correct_zone_containment=False, assigned_slot_error_mm=5.01
            ),
            "wrong_height": lambda _b, a, _e, _p: a[SELECTED_CUBE][
                "centre"
            ].__setitem__(2, C.TABLE_TOP_Z + C.SURFACE_DZ + 0.008),
            "distractor_lift": lambda _b, _a, e, _p: e[0]["objects_lifted"].append(
                "SortCap0"
            ),
            "distractor_translation": lambda _b, a, _e, _p: a["SortVial0"][
                "centre"
            ].__setitem__(0, a["SortVial0"]["centre"][0] + 0.00201),
            "distractor_rotation": lambda _b, a, _e, _p: a["SortCap1"].update(
                orientation=[np.cos(np.deg2rad(5.01) / 2), 0, 0, np.sin(np.deg2rad(5.01) / 2)]
            ),
            "other_cube_translation": lambda _b, a, _e, _p: a["SortCube0"][
                "centre"
            ].__setitem__(0, a["SortCube0"]["centre"][0] + NON_SELECTED_CUBE_TRANSLATION_LIMIT_M + 1e-5),
            "other_cube_rotation": lambda _b, a, _e, _p: a["SortCube1"].update(
                orientation=[np.cos(np.deg2rad(NON_SELECTED_CUBE_ROTATION_LIMIT_DEG + 0.01) / 2), 0, 0, np.sin(np.deg2rad(NON_SELECTED_CUBE_ROTATION_LIMIT_DEG + 0.01) / 2)]
            ),
            "other_cube_lift": lambda _b, _a, e, _p: e[0]["objects_lifted"].append(
                "SortCube5"
            ),
            "ik_or_clearance_failure": lambda _b, _a, _e, p: p.update(
                preflight_passed=False
            ),
        }
        for name, mutate in mutators.items():
            with self.subTest(name=name):
                before, after, events, preflight = self.successful_result()
                mutate(before, after, events, preflight)
                result = evaluate_smoke_success(
                    self.plan, events, before, after, preflight
                )
                self.assertFalse(result["smoke_test_success"])
                self.assertTrue(result["abort_conditions"])


class OracleSmokePostprocessTests(unittest.TestCase):
    def make_successful_run(self):
        temporary = tempfile.TemporaryDirectory()
        run = Path(temporary.name)
        (run / "execution_report.json").write_text(
            json.dumps({"smoke_test_success": True}), encoding="utf-8"
        )
        for name in (
            "execution_events.json",
            "before_table_camera.png",
            "after_table_camera.png",
            "after_table_camera_annotated.png",
            "execution_progress.log",
        ):
            (run / name).write_bytes(b"evidence")
        (run / "side_frames").mkdir()
        (run / "side_frames" / "f00001.png").write_bytes(b"frame")
        return temporary, run

    def test_successful_motion_plus_successful_video(self):
        temporary, run = self.make_successful_run()
        with temporary:
            validate_required_motion_outputs(run)
            (run / "side_execution.mp4").write_bytes(b"video")
            status = record_video_status(run, status="success")
            self.assertEqual(status["video_status"], "success")
            self.assertEqual(status["video_output"], "side_execution.mp4")

    def test_successful_motion_plus_unavailable_optional_encoder(self):
        temporary, run = self.make_successful_run()
        with temporary:
            validate_required_motion_outputs(run)
            status = record_video_status(
                run,
                status="unavailable",
                error="optional encoder dependency is unavailable",
            )
            self.assertEqual(status["video_status"], "unavailable")
            self.assertIsNone(status["video_output"])
            saved = json.loads((run / VIDEO_STATUS_FILE).read_text(encoding="utf-8"))
            self.assertTrue(saved["video_requested"])
            self.assertIn("unavailable", saved["video_error"])

    def test_actual_motion_failure_still_fails(self):
        temporary, run = self.make_successful_run()
        with temporary:
            (run / "execution_report.json").write_text(
                json.dumps(
                    {"smoke_test_success": False, "abort_reason": "lift failed"}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "motion failed"):
                validate_required_motion_outputs(run)

    def test_missing_required_artifacts_still_fail(self):
        temporary, run = self.make_successful_run()
        with temporary:
            (run / "execution_events.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "execution_events.json"):
                validate_required_motion_outputs(run)


if __name__ == "__main__":
    unittest.main()
