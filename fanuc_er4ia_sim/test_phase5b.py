"""Host-only contract tests for Phase 5B.  No client or robot is imported."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from bin_layout import AxisAlignedBounds, KeepOut, pocket_center_release_target, validate_bin_layout
from deployment_readiness import DeploymentReadiness, REQUIRED_VERIFICATIONS
from fanuc_tcp_orientation import (
    OrientationSample,
    TcpOrientationCalibration,
    fit_orientation_calibration,
    tcp_r_for_cube_yaw,
)
from grasp_clearance import (
    ClearanceMeasurements,
    authorize_grasp_clearance,
    opening_clearance_mm,
    presented_square_width_mm,
)
from organizer_contract import hold_still_action, validate_observation, validate_tcp_actions
from placement_model import PLACEHOLDER_EXAMPLE, build_placement_model, validate_placement_model
from real_sort_state_machine import Phase, RealSortStateMachine
from real_task_config import PRODUCTION_TASK_CONFIG, RealTaskConfig, SIMULATION_MISMATCH, validate_task_inventory
from wrist_cube_verification import WristCubeEvidence, verify_wrist_cube


def observation(*, size=(32, 24), depth=False):
    width, height = size
    value = {
        "t": 0.0,
        "status": "ok",
        "trial": 1,
        "queued": 0,
        "joints": np.array([0, 0, 0, 0, 0, 0, 1.0]),
        "tcp_pose": np.array([300, 0, 100, -180, 0, 0], dtype=float),
        "images": {name: np.zeros((height, width, 3), dtype=np.uint8) for name in ("table", "side", "wrist")},
    }
    if depth:
        value["depth"] = {
            "table": np.zeros((400, 640), dtype=np.uint16),
            "wrist": np.zeros((480, 848), dtype=np.uint16),
        }
    return value


def wrist_evidence(**changes):
    values = dict(
        candidate_count=1,
        table_detection_id="cube-1",
        wrist_detection_id="cube-1",
        expected_colour="red",
        wrist_colour="red",
        table_xy_mm=(300.0, 0.0),
        wrist_xy_mm=(300.5, 0.0),
        expected_yaw_deg=0.0,
        measured_yaw_deg=1.0,
        yaw_uncertainty_deg=0.5,
        closing_axis_uncertainty_mm=0.1,
        depth_valid_fraction=0.95,
        measured_cube_top_height_mm=10.5,
        expected_cube_top_height_mm=10.0,
        table_plane_residual_mm=0.5,
        approach_corridor_clear=True,
    )
    values.update(changes)
    return WristCubeEvidence(**values)


class ClearanceTests(unittest.TestCase):
    def test_axis_aligned_width(self):
        self.assertAlmostEqual(presented_square_width_mm(10, 0), 10)

    def test_diagonal_cube_does_not_fit(self):
        self.assertLess(opening_clearance_mm(45)["total_clearance_mm"], 0)

    def test_unknown_tolerances_fail_closed(self):
        result = authorize_grasp_clearance(0, 0.5, 0.1, ClearanceMeasurements())
        self.assertFalse(result["authorized"])
        self.assertIn("finger_tolerance_unverified", result["reasons"])

    def test_verified_budget_can_pass(self):
        measured = ClearanceMeasurements(
            dimensional_tolerance_per_side_mm=0.1,
            finger_tolerance_per_side_mm=0.1,
        )
        self.assertTrue(authorize_grasp_clearance(0, 0.5, 0.1, measured)["authorized"])

    def test_three_degree_combined_boundary_is_inclusive(self):
        measured = ClearanceMeasurements(10, 12, 0.05, 0.05)
        self.assertTrue(authorize_grasp_clearance(2, 1, 0, measured)["authorized"])
        self.assertFalse(authorize_grasp_clearance(2.01, 1, 0, measured)["authorized"])

    def test_centering_budget_is_enforced(self):
        measured = ClearanceMeasurements(10, 12, 0.05, 0.05)
        result = authorize_grasp_clearance(0, 0, 0.26, measured)
        self.assertIn("centering_error_exceeds_limit", result["reasons"])

    def test_rounded_edges_receive_no_credit(self):
        measured = ClearanceMeasurements(10, 12, 0.1, 0.1)
        result = authorize_grasp_clearance(0, 0, 0, measured)
        self.assertEqual(result["rounded_edge_credit_mm"], 0)
        self.assertEqual(result["hook_geometry_credit_mm"], 0)


class OrientationTests(unittest.TestCase):
    def calibration(self):
        samples = [OrientationSample(0, 10), OrientationSample(30, 40), OrientationSample(70, 80)]
        return fit_orientation_calibration(samples, uframe="UF0", utool="UT1")

    def test_requires_three_samples(self):
        with self.assertRaises(ValueError):
            fit_orientation_calibration([OrientationSample(0, 0)], uframe="UF0", utool="UT1")

    def test_fit_and_map(self):
        self.assertAlmostEqual(tcp_r_for_cube_yaw(20, self.calibration(), active_uframe="UF0", active_utool="UT1"), 30)

    def test_negative_sign_and_wrap(self):
        samples = [OrientationSample(0, 170), OrientationSample(30, 140), OrientationSample(70, 100)]
        calibration = fit_orientation_calibration(samples, uframe="UF0", utool="UT1")
        self.assertEqual(calibration.sign, -1)
        self.assertAlmostEqual(
            tcp_r_for_cube_yaw(-20, calibration, active_uframe="UF0", active_utool="UT1"),
            -170,
        )

    def test_unverified_mapping_rejected(self):
        with self.assertRaises(ValueError):
            tcp_r_for_cube_yaw(0, TcpOrientationCalibration(), active_uframe="UF0", active_utool="UT1")

    def test_frame_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            tcp_r_for_cube_yaw(0, self.calibration(), active_uframe="UF1", active_utool="UT1")

    def test_integrity_round_trip_and_tamper(self):
        payload = self.calibration().serialize()
        self.assertTrue(TcpOrientationCalibration.deserialize(payload).verified)
        payload["offset_deg"] = 99
        with self.assertRaises(ValueError):
            TcpOrientationCalibration.deserialize(payload)


class OrganizerContractTests(unittest.TestCase):
    def test_observation_rgb_and_depth(self):
        self.assertTrue(validate_observation(observation(depth=True), image_size=(32, 24), include_depth=True)["valid"])

    def test_rgb_shape_rejected(self):
        value = observation()
        value["images"]["table"] = np.zeros((1, 1, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            validate_observation(value, image_size=(32, 24))

    def test_depth_dtype_rejected(self):
        value = observation(depth=True)
        value["depth"]["table"] = value["depth"]["table"].astype(np.float32)
        with self.assertRaises(ValueError):
            validate_observation(value, image_size=(32, 24), include_depth=True)

    def test_tcp_action_and_binary_gripper(self):
        row = [[300, 0, 100, -180, 0, 0, 1]]
        self.assertEqual(validate_tcp_actions(row, dt_seconds=0.1).shape, (1, 7))
        row[0][-1] = 0.5
        with self.assertRaises(ValueError):
            validate_tcp_actions(row, dt_seconds=0.1)

    def test_tcp_limits_and_dt(self):
        with self.assertRaises(ValueError):
            validate_tcp_actions([[100, 0, 0, 0, 0, 0, 0]], dt_seconds=0.1)
        with self.assertRaises(ValueError):
            validate_tcp_actions([[300, 0, 0, 0, 0, 0, 0]], dt_seconds=0.01)

    def test_hold_is_numeric_current_pose(self):
        result = hold_still_action(observation())
        self.assertEqual(result.shape, (1, 7))
        self.assertEqual(result[0, 6], 1)


class VerificationAndInventoryTests(unittest.TestCase):
    def test_wrist_verification_passes_complete_evidence(self):
        self.assertTrue(verify_wrist_cube(wrist_evidence())["authorized"])

    def test_each_critical_wrist_gate_fails(self):
        variants = (
            {"candidate_count": 2},
            {"wrist_detection_id": "other"},
            {"wrist_colour": "blue"},
            {"wrist_xy_mm": (303.0, 0.0)},
            {"measured_yaw_deg": 4.0},
            {"yaw_uncertainty_deg": 2.0},
            {"closing_axis_uncertainty_mm": 0.3},
            {"depth_valid_fraction": 0.5},
            {"measured_cube_top_height_mm": 13.0},
            {"table_plane_residual_mm": 2.0},
            {"approach_corridor_clear": False},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertFalse(verify_wrist_cube(wrist_evidence(**variant))["authorized"])

    def test_exact_inventory(self):
        rows = []
        for colour in ("red", "blue", "yellow"):
            for index in range(2):
                rows.append({"category": "cube", "colour": colour, "detection_id": f"{colour}-{index}", "accepted": True})
        self.assertTrue(validate_task_inventory(rows)["authorized"])
        self.assertFalse(validate_task_inventory(rows[:-1])["authorized"])

    def test_caps_vials_and_bins_are_excluded_from_cube_inventory(self):
        rows = []
        for colour in ("red", "blue", "yellow"):
            for index in range(2):
                rows.append({"category": "cube", "colour": colour, "detection_id": f"{colour}-{index}", "accepted": True})
        rows.extend({"category": category} for category in ("cap", "vial", "bin"))
        result = validate_task_inventory(rows)
        self.assertTrue(result["authorized"])
        self.assertEqual(result["ignored_distractor_count"], 3)

    def test_real_vial_mismatch_is_explicit(self):
        self.assertNotEqual(SIMULATION_MISMATCH["current_vial_mm"], SIMULATION_MISMATCH["real_vial_mm"])


class LayoutAndPlacementTests(unittest.TestCase):
    table = AxisAlignedBounds(100, 700, -400, 350)

    def test_bin_layout_and_release_height(self):
        result = validate_bin_layout(
            {"bin_1": (250, 220), "bin_2": (350, 220), "bin_3": (450, 220)},
            table_bounds=self.table,
            camera_visible={"bin_1": True, "bin_2": True, "bin_3": True},
            maximum_reach_mm=700,
            reach_margin_mm=20,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(pocket_center_release_target((250, 220), table_surface_z_uf0_mm=-300, release_clearance_above_rim_mm=15)[2], -230)

    def test_bin_keepout_rejected(self):
        keepout = KeepOut("fixture", AxisAlignedBounds(220, 280, 190, 250))
        result = validate_bin_layout(
            {"bin_1": (250, 220), "bin_2": (350, 220), "bin_3": (450, 220)},
            table_bounds=self.table,
            keep_outs=(keepout,),
            camera_visible={"bin_1": True, "bin_2": True, "bin_3": True},
            maximum_reach_mm=700,
            reach_margin_mm=20,
        )
        self.assertFalse(result["valid"])

    def test_bin_spacing_and_reach_rejected(self):
        result = validate_bin_layout(
            {"bin_1": (250, 220), "bin_2": (260, 220), "bin_3": (690, 220)},
            table_bounds=self.table,
            camera_visible={"bin_1": True, "bin_2": True, "bin_3": True},
            maximum_reach_mm=700,
            reach_margin_mm=20,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("insufficient_spacing" in reason for reason in result["reasons"]))
        self.assertTrue(any("insufficient_reach_margin" in reason for reason in result["reasons"]))

    def rollout(self):
        return {"objects": {
            "red_cube": [[200, 60, 0], [230, 60, 0]],
            "blue_cube": [[260, 60, 0], [290, 60, 0]],
            "yellow_cube": [[320, 60, 0], [350, 60, 0]],
            "cap": [[400, 60], [440, 60]],
            "vial": [[270, -190, 0], [550, -190, 0]],
            "bin_1": [250, 220], "bin_2": [350, 220], "bin_3": [450, 220],
        }}

    def validate(self, document, config=None, objects_mode="same", layout_mode="same"):
        return validate_placement_model(
            document,
            config=config or replace(PRODUCTION_TASK_CONFIG, requested_cube_yaw_deg=0),
            objects_mode=objects_mode,
            layout_mode=layout_mode,
            source_bounds=AxisAlignedBounds(150, 680, -330, 100),
            table_bounds=self.table,
            keep_outs=(),
            bin_visibility={"bin_1": True, "bin_2": True, "bin_3": True},
            maximum_reach_mm=700,
            reach_margin_mm=20,
        )

    def context(self):
        return dict(
            config=replace(PRODUCTION_TASK_CONFIG, requested_cube_yaw_deg=0),
            objects_mode="same",
            layout_mode="same",
            source_bounds=AxisAlignedBounds(150, 680, -330, 100),
            table_bounds=self.table,
            keep_outs=(),
            bin_visibility={"bin_1": True, "bin_2": True, "bin_3": True},
            maximum_reach_mm=700,
            reach_margin_mm=20,
        )

    def test_placeholder_is_not_deployable(self):
        self.assertFalse(PLACEHOLDER_EXAMPLE["deployable"])

    def test_complete_single_rollout(self):
        self.assertTrue(self.validate({"rollouts": [self.rollout()]})["valid"])

    def test_builder_uses_only_supplied_coordinates_and_remains_non_deployable(self):
        result = build_placement_model([self.rollout()], **self.context())
        self.assertTrue(result["validation"]["valid"])
        self.assertFalse(result["deployable"])

    def test_unresolved_yaw_rejected(self):
        with self.assertRaises(ValueError):
            self.validate({"rollouts": [self.rollout()]}, config=RealTaskConfig())

    def test_incomplete_inventory_rejected(self):
        rollout = self.rollout()
        del rollout["objects"]["cap"]
        with self.assertRaises(ValueError):
            self.validate({"rollouts": [rollout]})

    def test_different_mode_requires_ten_rollouts(self):
        with self.assertRaises(ValueError):
            self.validate({"rollouts": [self.rollout()]}, layout_mode="different")


class ReadinessAndStateMachineTests(unittest.TestCase):
    def test_production_defaults_false(self):
        report = DeploymentReadiness().report()
        self.assertEqual(len(report["missing"]), len(REQUIRED_VERIFICATIONS))
        self.assertFalse(report["robot_motion_authorized"])

    def test_even_complete_fixture_cannot_authorize_motion(self):
        report = DeploymentReadiness({name: True for name in REQUIRED_VERIFICATIONS}).report()
        self.assertTrue(report["all_requirements_verified"])
        self.assertFalse(report["robot_motion_authorized"])

    def test_one_missing_measurement_keeps_readiness_false(self):
        values = {name: True for name in REQUIRED_VERIFICATIONS}
        values[REQUIRED_VERIFICATIONS[0]] = False
        report = DeploymentReadiness(values).report()
        self.assertFalse(report["all_requirements_verified"])
        self.assertFalse(report["robot_motion_authorized"])

    def test_no_proposal_before_scene_authorization(self):
        machine = RealSortStateMachine()
        with self.assertRaises(RuntimeError):
            machine.propose([[300, 0, 100, -180, 0, 0, 1]], dt_seconds=0.1)

    def test_proposal_is_never_transmitted(self):
        machine = RealSortStateMachine()
        machine.authorize_scene("scene")
        proposal = machine.propose([[300, 0, 100, -180, 0, 0, 1]], dt_seconds=0.1)
        self.assertFalse(proposal.transmitted)

    def test_hold_phase_returns_validated_current_pose(self):
        machine = RealSortStateMachine()
        machine.phase = Phase.HOLD
        proposal = machine.hold(observation())
        self.assertEqual(proposal.phase, Phase.HOLD)
        self.assertEqual(proposal.rows.shape, (1, 7))
        self.assertFalse(proposal.transmitted)

    def test_authorization_single_use_and_scene_bound(self):
        machine = RealSortStateMachine()
        machine.authorize_scene("scene")
        token = machine.bind_grasp_authorization("obs", "cube")
        machine.consume_grasp_authorization(token)
        with self.assertRaises(RuntimeError):
            machine.consume_grasp_authorization(token)
        self.assertEqual(machine.phase, Phase.ABORT_LATCHED)

    def test_descent_consumes_authorization(self):
        machine = RealSortStateMachine()
        machine.authorize_scene("scene")
        token = machine.bind_grasp_authorization("obs", "cube")
        machine.phase = Phase.DESCEND
        proposal = machine.propose(
            [[300, 0, 100, -180, 0, 0, 1]],
            dt_seconds=0.1,
            authorization_token=token,
        )
        self.assertFalse(proposal.transmitted)
        self.assertTrue(machine.grasp_authorization_consumed)

    def test_descent_without_authorization_aborts(self):
        machine = RealSortStateMachine()
        machine.authorize_scene("scene")
        machine.phase = Phase.DESCEND
        with self.assertRaises(RuntimeError):
            machine.propose([[300, 0, 100, -180, 0, 0, 1]], dt_seconds=0.1)
        self.assertEqual(machine.phase, Phase.ABORT_LATCHED)

    def test_scene_change_latches_abort(self):
        machine = RealSortStateMachine()
        machine.authorize_scene("scene")
        with self.assertRaises(RuntimeError):
            machine.verify_scene_unchanged("changed")
        self.assertEqual(machine.phase, Phase.ABORT_LATCHED)

    def test_abort_requires_reset(self):
        machine = RealSortStateMachine()
        machine.abort("test")
        with self.assertRaises(RuntimeError):
            machine.advance()
        machine.reset()
        self.assertEqual(machine.phase, Phase.RESET)

    def test_failed_gate_latches_first_reason(self):
        machine = RealSortStateMachine()
        self.assertEqual(machine.apply_gate(False, failure_reason="calibration_missing"), Phase.ABORT_LATCHED)
        self.assertEqual(machine.abort_reason, "calibration_missing")
        with self.assertRaises(RuntimeError):
            machine.apply_gate(True, failure_reason="ignored")

    def test_preservation_counts_six(self):
        machine = RealSortStateMachine()
        for _ in range(6):
            machine.record_pickup(verified=True)
            machine.record_placement(verified=True)
        self.assertTrue(machine.preservation_ok())
        machine.phase = Phase.NEXT
        self.assertEqual(machine.advance(), Phase.COMPLETE)

    def test_distractor_touch_aborts(self):
        machine = RealSortStateMachine()
        machine.record_pickup(verified=True, distractor_touched=True)
        self.assertEqual(machine.phase, Phase.ABORT_LATCHED)


if __name__ == "__main__":
    unittest.main()
