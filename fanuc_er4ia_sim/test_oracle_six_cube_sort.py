"""Host-only tests for the Phase 3B six-cube oracle sort contract."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import scene_config as C
from oracle_motion_smoke import (
    INITIAL_PHYSICS_SETTLING_S,
    MAXIMUM_TOTAL_SETTLING_S,
    ORACLE_SOURCE,
    REQUIRED_CONSECUTIVE_STABLE_WINDOWS,
    STABILITY_ROTATION_THRESHOLD_DEG,
    STABILITY_TRANSLATION_THRESHOLD_M,
    STABILITY_WINDOW_S,
    authorize_oracle_robot_commands,
    build_oracle_smoke_plan,
    build_settled_execution_plan,
    evaluate_scene_stability_windows,
    evaluate_stability_window,
    validate_oracle_smoke_plan,
    validate_settled_distractor_clearance,
)
from oracle_six_cube_sort import (
    EXECUTION_ORDER,
    ORACLE_SIX_TASK,
    advance_abort_state,
    build_oracle_six_cube_plan,
    evaluate_oracle_six_success,
    preflight_oracle_six_motion,
    validate_oracle_six_cube_plan,
    validate_per_cube_safety,
    validate_six_execution_start,
)
from oracle_smoke_postprocess import (
    record_video_status,
    validate_required_motion_outputs,
)
from sort_scene import segments


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


def runtime_poses():
    state, _truth = scene_inputs()
    poses = {}
    for name, pose in state["objects"].items():
        centre = list(pose["position"])
        if name.startswith("SortCube"):
            centre[2] += C.CUBE_EDGE / 2
        elif name.startswith("SortCap"):
            centre[2] += C.CAP_HEIGHT / 2
        else:
            centre[2] += C.VIAL_RADIUS
        poses[name] = {
            **copy.deepcopy(pose),
            "centre": centre,
            "yaw_deg": 0.0,
            "tilt_deg": 0.0,
        }
    return poses


def place_cube(poses, name, slot):
    poses[name]["centre"] = [
        *slot,
        C.TABLE_TOP_Z + C.SURFACE_DZ + 0.5 * C.CUBE_EDGE,
    ]
    poses[name]["position"] = [
        *slot,
        C.TABLE_TOP_Z + C.SURFACE_DZ,
    ]
    poses[name]["orientation"] = [1, 0, 0, 0]
    poses[name]["yaw_deg"] = 0.0


def stability_window(base, name=None, translations=None, rotations_deg=None):
    translations = translations or [0.0, 0.0, 0.0]
    rotations_deg = rotations_deg or [0.0] * len(translations)
    samples = []
    for translation, rotation_deg in zip(translations, rotations_deg):
        sample = copy.deepcopy(base)
        if name is not None:
            sample[name]["centre"][0] += translation
            sample[name]["position"][0] += translation
            angle = np.deg2rad(rotation_deg)
            sample[name]["orientation"] = [
                np.cos(angle / 2),
                0,
                0,
                np.sin(angle / 2),
            ]
        samples.append(sample)
    return samples


class OracleSixCubePlanTests(unittest.TestCase):
    def setUp(self):
        self.state, self.truth = scene_inputs()
        self.plan = build_oracle_six_cube_plan(self.state, self.truth)

    def test_valid_plan_has_exact_mapping_slots_poses_and_order(self):
        self.assertTrue(validate_oracle_six_cube_plan(self.plan))
        self.assertEqual(self.plan["task"], ORACLE_SIX_TASK)
        self.assertEqual(self.plan["source"], ORACLE_SOURCE)
        self.assertEqual(
            self.plan["commanded_objects"],
            [name for name, _colour, _slot in EXECUTION_ORDER],
        )
        self.assertEqual(len(self.plan["picks"]), 6)
        self.assertEqual(len(self.plan["planned_scene_poses"]), 10)
        self.assertEqual(
            {tuple(pick["assigned_slot"]) for pick in self.plan["picks"]},
            {
                tuple(slot)
                for colour in C.DEFECT_TARGET_SLOTS
                for slot in C.DEFECT_TARGET_SLOTS[colour]
            },
        )
        for pick, (name, colour, slot_index) in zip(
            self.plan["picks"], EXECUTION_ORDER
        ):
            self.assertEqual(pick["object_id"], name)
            self.assertEqual(pick["colour"], colour)
            self.assertEqual(
                pick["defect_class"], C.DEFECT_CLASS_BY_COLOUR[colour]
            )
            self.assertEqual(
                pick["assigned_slot"], list(C.DEFECT_TARGET_SLOTS[colour][slot_index])
            )
            self.assertEqual(pick["pose_source"], "oracle_simulation_pose")
            self.assertEqual(len(pick["oracle_simulation_pose"]["orientation"]), 4)

    def test_invalid_inventory_colour_class_and_slot_cases_fail(self):
        missing_state = copy.deepcopy(self.state)
        missing_state["objects"].pop("SortCap0")
        with self.assertRaisesRegex(ValueError, "exactly ten"):
            build_oracle_six_cube_plan(missing_state, self.truth)

        extra_truth = copy.deepcopy(self.truth)
        extra_truth["objects"].append(
            {"name": "Unexpected", "category": "cube", "colour": "red"}
        )
        with self.assertRaisesRegex(ValueError, "exactly ten"):
            build_oracle_six_cube_plan(self.state, extra_truth)

        wrong_colour = copy.deepcopy(self.truth)
        next(
            row for row in wrong_colour["objects"] if row["name"] == "SortCube2"
        )["colour"] = "blue"
        with self.assertRaises(ValueError):
            build_oracle_six_cube_plan(self.state, wrong_colour)

        mutations = []
        duplicate_object = copy.deepcopy(self.plan)
        duplicate_object["picks"][1]["object_id"] = "SortCube2"
        mutations.append(duplicate_object)
        wrong_class = copy.deepcopy(self.plan)
        wrong_class["picks"][0]["defect_class"] = "Defect Class B"
        mutations.append(wrong_class)
        duplicate_slot = copy.deepcopy(self.plan)
        duplicate_slot["picks"][1]["assigned_slot"] = duplicate_slot["picks"][0][
            "assigned_slot"
        ]
        duplicate_slot["picks"][1]["place_xy"] = duplicate_slot["picks"][0][
            "assigned_slot"
        ]
        mutations.append(duplicate_slot)
        wrong_zone = copy.deepcopy(self.plan)
        wrong_zone["picks"][0]["assigned_slot"] = list(
            C.DEFECT_TARGET_SLOTS["blue"][0]
        )
        wrong_zone["picks"][0]["place_xy"] = list(
            C.DEFECT_TARGET_SLOTS["blue"][0]
        )
        mutations.append(wrong_zone)
        for plan in mutations:
            with self.subTest(plan=plan):
                with self.assertRaises(ValueError):
                    validate_oracle_six_cube_plan(plan)

    def test_cross_process_mismatch_for_any_cube_aborts(self):
        for index in range(6):
            current = copy.deepcopy(self.state["objects"])
            current[f"SortCube{index}"]["position"][0] += 0.00201
            result = validate_six_execution_start(self.plan, current)
            with self.subTest(cube=index):
                self.assertFalse(result["pre_command_validation_passed"])
                self.assertIn(f"cube_pose_mismatch:SortCube{index}", result["failures"])

    def test_cross_process_records_all_ten_and_rejects_rotation_or_inventory(self):
        matching = validate_six_execution_start(
            self.plan, copy.deepcopy(self.state["objects"])
        )
        self.assertTrue(matching["pre_command_validation_passed"])
        self.assertEqual(len(matching["objects"]), 10)

        rotated = copy.deepcopy(self.state["objects"])
        angle = np.deg2rad(5.01)
        rotated["SortCube5"]["orientation"] = [
            np.cos(angle / 2),
            0,
            0,
            np.sin(angle / 2),
        ]
        self.assertFalse(
            validate_six_execution_start(self.plan, rotated)[
                "pre_command_validation_passed"
            ]
        )
        missing = copy.deepcopy(self.state["objects"])
        missing.pop("SortVial1")
        self.assertFalse(
            validate_six_execution_start(self.plan, missing)[
                "pre_command_validation_passed"
            ]
        )

    def test_all_six_trajectory_endpoints_and_clearance_preflight(self):
        home = np.array([0.30, 0.0, 0.24])
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])

        def reachable(q, *_args, **_kwargs):
            return np.asarray(q, float), True

        with patch("oracle_motion_smoke.ik.solve", side_effect=reachable):
            result = preflight_oracle_six_motion(
                self.plan,
                home,
                tuple(np.asarray(C.TOOL_AXIS) * C.TOOL_LENGTH),
                limits,
                [segments(pick, home) for pick in self.plan["picks"]],
            )
        self.assertTrue(result["preflight_passed"])
        self.assertEqual(len(result["per_pick"]), 6)
        self.assertTrue(
            all(len(row["phase_ik_checks"]) == 11 for row in result["per_pick"])
        )
        self.assertEqual(result["transit_clearance_m"], 0.130)


class OracleSixCubeRuntimeSafetyTests(unittest.TestCase):
    def setUp(self):
        state, truth = scene_inputs()
        self.plan = build_oracle_six_cube_plan(state, truth)
        self.baseline = runtime_poses()

    def test_displaced_distractor_fails(self):
        current = copy.deepcopy(self.baseline)
        current["SortCap0"]["centre"][0] += 0.00201
        result = validate_per_cube_safety(
            self.plan, self.baseline, current, "SortCube2", [], "pre_pick"
        )
        self.assertFalse(result["passed"])
        self.assertIn("distractor_displaced:SortCap0", result["failures"])

    def test_unintended_cube_or_distractor_lift_fails(self):
        for lifted in (["SortCube2", "SortCube4"], ["SortCube2", "SortVial0"]):
            result = validate_per_cube_safety(
                self.plan,
                self.baseline,
                copy.deepcopy(self.baseline),
                "SortCube2",
                [],
                "post_lift",
                lifted,
            )
            with self.subTest(lifted=lifted):
                self.assertFalse(result["passed"])
                self.assertTrue(any(reason.startswith("unexpected_lift") for reason in result["failures"]))

    def test_previously_placed_cube_disturbed_fails(self):
        current = copy.deepcopy(self.baseline)
        place_cube(current, "SortCube2", C.DEFECT_TARGET_SLOTS["red"][0])
        current["SortCube2"]["centre"][0] += 0.006
        result = validate_per_cube_safety(
            self.plan,
            self.baseline,
            current,
            "SortCube3",
            ["SortCube2"],
            "pre_pick",
        )
        self.assertFalse(result["passed"])
        self.assertIn("placed_cube_invalid:SortCube2", result["failures"])

    def test_occupied_destination_slot_fails(self):
        current = copy.deepcopy(self.baseline)
        place_cube(current, "SortCube2", C.DEFECT_TARGET_SLOTS["red"][1])
        result = validate_per_cube_safety(
            self.plan,
            self.baseline,
            current,
            "SortCube3",
            [],
            "pre_pick",
        )
        self.assertFalse(result["passed"])
        self.assertIn("destination_slot_occupied", result["failures"])

    def test_per_cube_and_final_success(self):
        after = copy.deepcopy(self.baseline)
        events = []
        completed = []
        for pick in self.plan["picks"]:
            name = pick["object_id"]
            place_cube(after, name, pick["assigned_slot"])
            safety = validate_per_cube_safety(
                self.plan,
                self.baseline,
                after,
                name,
                completed,
                "post_place",
            )
            self.assertTrue(safety["passed"])
            completed.append(name)
            events.append(
                {
                    "tracked_object": name,
                    "pickup_verified": True,
                    "placement_verified": True,
                    "objects_lifted": [name],
                    "starting_pose": self.baseline[name],
                    "motion_time_s": 9.2,
                    "wall_clock_time_s": 10.0,
                    "retries": [],
                    "recoveries": [],
                }
            )
        result = evaluate_oracle_six_success(
            self.plan,
            events,
            self.baseline,
            after,
            {"preflight_passed": True},
        )
        self.assertTrue(result["oracle_sort_success"])
        self.assertEqual(result["successful_pickups"], 6)
        self.assertEqual(result["correct_zone_placements"], 6)
        self.assertEqual(result["zone_counts"], {"red": 2, "blue": 2, "yellow": 2})
        self.assertEqual(len(result["per_cube_results"]), 6)

    def test_abort_latches_and_stops_later_authorization(self):
        reason, authorized = advance_abort_state(
            None, {"passed": False, "failures": ["selected lift failed"]}
        )
        self.assertFalse(authorized)
        same_reason, later_authorized = advance_abort_state(
            reason, {"passed": True, "failures": []}
        )
        self.assertFalse(later_authorized)
        self.assertEqual(same_reason, reason)

    def test_optional_video_failure_does_not_invalidate_six_cube_motion(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "execution_report.json").write_text(
                json.dumps(
                    {"task": ORACLE_SIX_TASK, "oracle_sort_success": True}
                ),
                encoding="utf-8",
            )
            for name in (
                "execution_events.json",
                "before_table_camera.png",
                "after_table_camera.png",
                "after_table_camera_annotated.png",
                "execution_progress.log",
            ):
                (run / name).write_bytes(b"evidence")
            (run / "frames").mkdir()
            (run / "frames" / "f00000.png").write_bytes(b"frame")
            validate_required_motion_outputs(run)
            status = record_video_status(
                run,
                status="unavailable",
                error="imageio_ffmpeg unavailable",
                output="table_execution.mp4",
            )
            self.assertEqual(status["video_status"], "unavailable")
            validate_required_motion_outputs(run)


class OracleSceneStabilityTests(unittest.TestCase):
    def setUp(self):
        self.state, self.truth = scene_inputs()
        self.poses = runtime_poses()
        self.six_plan = build_oracle_six_cube_plan(self.state, self.truth)
        self.smoke_plan = build_oracle_smoke_plan(self.state, self.truth)

    def test_shared_gate_policy_constants_and_report_are_exact(self):
        self.assertEqual(INITIAL_PHYSICS_SETTLING_S, 1.0)
        self.assertEqual(STABILITY_WINDOW_S, 1.0)
        self.assertEqual(STABILITY_TRANSLATION_THRESHOLD_M, 0.00025)
        self.assertEqual(STABILITY_ROTATION_THRESHOLD_DEG, 0.25)
        self.assertEqual(REQUIRED_CONSECUTIVE_STABLE_WINDOWS, 2)
        self.assertEqual(MAXIMUM_TOTAL_SETTLING_S, 20.0)

        result = evaluate_scene_stability_windows(
            [stability_window(self.poses), stability_window(self.poses)]
        )
        self.assertEqual(result["minimum_initial_settling_s"], 1.0)
        self.assertEqual(result["stability_window_duration_s"], 1.0)
        self.assertEqual(result["stability_translation_threshold_mm"], 0.25)
        self.assertEqual(result["stability_rotation_threshold_deg"], 0.25)
        self.assertEqual(result["required_consecutive_stable_windows"], 2)
        self.assertEqual(result["maximum_total_settling_s"], 20.0)

    def test_immediately_stable_scene_requires_two_windows(self):
        stable = stability_window(self.poses)
        one = evaluate_scene_stability_windows([stable])
        self.assertFalse(one["scene_stable_before_commands"])
        result = evaluate_scene_stability_windows([stable, stable])
        self.assertTrue(result["scene_stable_before_commands"])
        self.assertEqual(
            result["consecutive_stable_window_count"],
            REQUIRED_CONSECUTIVE_STABLE_WINDOWS,
        )

    def test_vial_can_stabilize_after_several_windows(self):
        moving = stability_window(
            self.poses,
            "SortVial0",
            [0.0, STABILITY_TRANSLATION_THRESHOLD_M * 2],
        )
        stable = stability_window(self.poses)
        result = evaluate_scene_stability_windows(
            [moving, moving, stable, stable]
        )
        self.assertTrue(result["scene_stable_before_commands"])
        self.assertEqual(len(result["windows"]), 4)

    def test_intermediate_excursion_is_not_hidden_by_equal_endpoints(self):
        window = stability_window(
            self.poses,
            "SortVial0",
            [0.0, STABILITY_TRANSLATION_THRESHOLD_M * 2, 0.0],
        )
        result = evaluate_stability_window(window)
        row = next(row for row in result["objects"] if row["name"] == "SortVial0")
        self.assertFalse(row["passed"])
        self.assertGreater(
            row["maximum_translation_excursion_m"],
            STABILITY_TRANSLATION_THRESHOLD_M,
        )

    def test_translation_and_rotation_instability_are_independent(self):
        translated = evaluate_stability_window(
            stability_window(
                self.poses,
                "SortCap0",
                [0.0, STABILITY_TRANSLATION_THRESHOLD_M + 1e-6],
            )
        )
        rotated = evaluate_stability_window(
            stability_window(
                self.poses,
                "SortVial1",
                [0.0, 0.0],
                [0.0, STABILITY_ROTATION_THRESHOLD_DEG + 0.01],
            )
        )
        self.assertIn("SortCap0", translated["unstable_object_names"])
        self.assertIn("SortVial1", rotated["unstable_object_names"])

    def test_timeout_reports_unstable_objects_and_no_stability(self):
        moving = stability_window(
            self.poses,
            "SortVial0",
            [0.0, STABILITY_TRANSLATION_THRESHOLD_M * 2],
        )
        result = evaluate_scene_stability_windows([moving] * 19, timed_out=True)
        self.assertFalse(result["scene_stable_before_commands"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["abort_reason"], "scene_stability_timeout")
        self.assertIn("SortVial0", result["unstable_object_names"])

    def test_different_unstable_objects_reset_consecutive_count(self):
        vial = stability_window(
            self.poses,
            "SortVial0",
            [0.0, STABILITY_TRANSLATION_THRESHOLD_M * 2],
        )
        cap = stability_window(
            self.poses,
            "SortCap1",
            [0.0, STABILITY_TRANSLATION_THRESHOLD_M * 2],
        )
        stable = stability_window(self.poses)
        result = evaluate_scene_stability_windows([vial, stable, cap, stable, stable])
        self.assertTrue(result["scene_stable_before_commands"])
        self.assertEqual(result["windows"][-1]["consecutive_stable_window_count"], 2)

    def test_post_settle_cube_mismatch_is_not_replaced(self):
        settled = copy.deepcopy(self.state["objects"])
        settled["SortCube3"]["position"][0] += 0.00201
        result = validate_six_execution_start(self.six_plan, settled)
        self.assertFalse(result["pre_command_validation_passed"])
        self.assertIn("cube_pose_mismatch:SortCube3", result["failures"])

    def test_settled_distractor_occupying_slot_fails(self):
        settled = copy.deepcopy(self.poses)
        slot = self.six_plan["picks"][0]["assigned_slot"]
        settled["SortCap0"]["centre"][:2] = slot
        settled["SortCap0"]["position"][:2] = slot
        result = validate_settled_distractor_clearance(self.six_plan, settled)
        self.assertFalse(result["settled_distractor_clearance_passed"])
        self.assertTrue(
            any("occupies_slot:SortCap0" in failure for failure in result["failures"])
        )

    def test_settled_distractor_violating_trajectory_clearance_fails(self):
        settled = copy.deepcopy(self.poses)
        pickup = self.six_plan["picks"][0]["xy"]
        settled["SortCap0"]["centre"][:2] = pickup
        settled["SortCap0"]["position"][:2] = pickup
        result = validate_settled_distractor_clearance(self.six_plan, settled)
        row = next(row for row in result["objects"] if row["name"] == "SortCap0")
        self.assertFalse(result["settled_distractor_clearance_passed"])
        self.assertIn("pickup:SortCube2", row["tool_path_violations"])

    def test_no_command_authorization_before_every_gate_passes(self):
        passing = [
            {"scene_stable_before_commands": True},
            {"pre_command_validation_passed": True},
            {"settled_distractor_clearance_passed": True},
            {"preflight_passed": True},
        ]
        self.assertTrue(
            authorize_oracle_robot_commands(*passing)["robot_commands_authorized"]
        )
        for index in range(4):
            failed = copy.deepcopy(passing)
            failed[index][next(iter(failed[index]))] = False
            with self.subTest(gate=index):
                self.assertFalse(
                    authorize_oracle_robot_commands(*failed)[
                        "robot_commands_authorized"
                    ]
                )

    def test_stable_settled_plan_preserves_phase3a_and_phase3b_contracts(self):
        smoke_execution = build_settled_execution_plan(
            self.smoke_plan, self.poses
        )
        six_execution = build_settled_execution_plan(self.six_plan, self.poses)
        self.assertTrue(validate_oracle_smoke_plan(smoke_execution))
        self.assertTrue(validate_oracle_six_cube_plan(six_execution))
        self.assertEqual(len(smoke_execution["settled_pose_updates"]), 1)
        self.assertEqual(len(six_execution["settled_pose_updates"]), 6)


if __name__ == "__main__":
    unittest.main()
