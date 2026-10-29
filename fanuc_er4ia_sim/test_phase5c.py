"""Host-only tests for the fail-closed native observation boundary."""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import unittest

import numpy as np

from camera_calibration_contract import (
    CameraCalibration,
    RgbDepthRegistration,
    validate_camera_calibration,
    validate_registration,
)
from deployment_readiness import PRODUCTION_READINESS
from detector_adapter_contract import (
    ORGANIZER_INPUT_CONVENTION,
    SUPPORTED_CONVERSION,
    SUPPORTED_INTERNAL_CONVENTION,
    SUPPORTED_OUTPUT_SCHEMA,
    build_detector_adapter_contract,
    canonical_sha256,
    inspect_detector_callable,
    verify_detector_adapter,
)
from depth_processing import (
    depth_mm_to_metres,
    local_depth_statistics,
    lookup_registered_rgb_depth,
    plane_residual_statistics,
)
from grasp_clearance import ClearanceMeasurements
from native_observation import (
    NATIVE_DEPTH_SHAPES,
    NATIVE_RGB_SHAPES,
    parse_native_observation,
)
from observation_only_session import (
    ObservationOnlySession,
    ObservationPhase,
    unresolved_wrist_observation_proposal,
)
from table_scene_perception import SourceWorkspaceContract, perceive_table_scene
from wrist_observation_gate import WristGateEvidence, verify_wrist_observations


def native_rgb(camera: str, offset: int = 0) -> np.ndarray:
    height, width, channels = NATIVE_RGB_SHAPES[camera]
    image = np.zeros((height, width, channels), dtype=np.uint8)
    row = ((np.arange(width, dtype=np.uint16) + offset) % 251).astype(np.uint8)
    image[:, :, 0] = row
    image[:, :, 1] = (row.astype(np.uint16) * 3 % 251).astype(np.uint8)
    return image


def observation(
    camera: str,
    *,
    timestamp: float = 10.0,
    trial: object = 1,
    offset: int = 0,
    with_depth: bool = False,
    joint_delta: float = 0.0,
) -> dict:
    value = {
        "t": timestamp,
        "status": "running",
        "trial": trial,
        "queued": 0,
        "joints": np.array([joint_delta, 0, 0, 0, 0, 0, 1], dtype=float),
        "tcp_pose": np.array([300, 0, 100, -180, 0, 0], dtype=float),
        "images": {camera: native_rgb(camera, offset)},
    }
    if with_depth:
        value["depth"] = {
            camera: np.full(NATIVE_DEPTH_SHAPES[camera], 500, dtype=np.uint16)
        }
    return value


def calibration(stream: str, *, compatibility: str = "test-real-v1") -> CameraCalibration:
    resolutions = {
        "table_rgb": (1920, 1080),
        "wrist_rgb": (1280, 720),
        "table_depth": (640, 400),
        "wrist_depth": (848, 480),
    }
    return CameraCalibration(
        stream=stream,
        domain="real_cell",
        frame_name="table_uf0" if stream.startswith("table") else "wrist_camera",
        intrinsics_row_major=(1000.0, 0.0, 500.0, 0.0, 1000.0, 400.0, 0.0, 0.0, 1.0),
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        extrinsics_row_major=(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        resolution_wh=resolutions[stream],
        length_unit="metre",
        angle_unit="radian",
        calibration_version="test-fixture-only",
        compatibility_id=compatibility,
        verified=True,
    )


def registration(*, verified: bool = True) -> RgbDepthRegistration:
    return RgbDepthRegistration(
        rgb_stream="wrist_rgb",
        depth_stream="wrist_depth",
        domain="real_cell",
        rgb_resolution_wh=(1280, 720),
        depth_resolution_wh=(848, 480),
        compatibility_id="test-real-v1",
        registration_version="test-fixture-only",
        registered=True,
        verified=verified,
        method="external_verified_registration_adapter",
    )


def workspace() -> SourceWorkspaceContract:
    return SourceWorkspaceContract(
        frame_name="table_uf0",
        polygon_xy_m=((0.2, -0.2), (0.6, -0.2), (0.6, 0.2), (0.2, 0.2)),
        calibration_compatibility_id="test-real-v1",
        survey_version="test-fixture-only",
        public=True,
        verified=True,
    )


def detector_rows() -> list[dict]:
    rows = []
    for colour, y in (("red", -0.1), ("blue", 0.0), ("yellow", 0.1)):
        rows.extend(
            [
                {"category": "cube", "color": colour, "xy_m": [0.3, y], "candidate_confidence": 0.95},
                {"category": "cube", "color": colour, "xy_m": [0.4, y], "candidate_confidence": 0.96},
            ]
        )
    return rows


def detector_result(rows=None) -> dict:
    return {
        "ground_truth_used": False,
        "scene_state_used": False,
        "oracle_plan_used": False,
        "accepted_detections": detector_rows() if rows is None else rows,
    }


ADAPTER_CONFIGURATION = {"profile": "synthetic-test-fixture", "threshold_version": 1}


def verified_fixture_detector(image, camera, source_workspace):
    return detector_result()


def incomplete_fixture_detector(image, camera, source_workspace):
    return detector_result(detector_rows()[:-1])


def duplicate_fixture_detector(image, camera, source_workspace):
    rows = detector_rows()
    rows[-1]["xy_m"] = rows[0]["xy_m"]
    return detector_result(rows)


def forbidden_identity_fixture_detector(image, camera, source_workspace):
    rows = detector_rows()
    rows[0]["object_name"] = "forbidden"
    return detector_result(rows)


def motion_injecting_fixture_detector(image, camera, source_workspace):
    value = detector_result()
    value["robot_motion_authorized"] = True
    value["action"] = [[300, 0, 100, -180, 0, 0, 1]]
    return value


def unreviewed_schema_valid_fixture_detector(image, camera, source_workspace):
    return detector_result()


def adapter_contract(detector=verified_fixture_detector, **overrides):
    runtime = inspect_detector_callable(detector)
    fields = dict(
        adapter_name="synthetic-native-rgb-adapter",
        adapter_semantic_version="1.0.0-test",
        callable_module_name=runtime["module_name"],
        callable_qualified_name=runtime["qualified_name"],
        callable_module_file_sha256=runtime["module_file_sha256"],
        adapter_configuration_sha256=canonical_sha256(ADAPTER_CONFIGURATION),
        input_color_convention=ORGANIZER_INPUT_CONVENTION,
        internal_detector_color_convention=SUPPORTED_INTERNAL_CONVENTION,
        color_conversion_convention=SUPPORTED_CONVERSION,
        output_schema_version=SUPPORTED_OUTPUT_SCHEMA,
        compatible_calibration_ids=("test-real-v1",),
        compatible_workspace_versions=("test-fixture-only",),
        verification_evidence_sha256=canonical_sha256(
            {"review": "synthetic test fixture only", "approved": True}
        ),
        verified=True,
    )
    fields.update(overrides)
    return build_detector_adapter_contract(**fields)


def parsed_wrist(*, timestamp=10.0, trial=1, offset=0, joint_delta=0.0):
    return parse_native_observation(
        observation(
            "wrist",
            timestamp=timestamp,
            trial=trial,
            offset=offset,
            with_depth=True,
            joint_delta=joint_delta,
        ),
        required_rgb=("wrist",),
        requested_depth=("wrist",),
    )


def selected_detection() -> dict:
    return {
        "detection_id": "anon-test-cube",
        "category": "cube",
        "colour": "red",
        "xy_mm": [300.0, 0.0],
        "confidence": 0.95,
        "accepted": True,
    }


def wrist_evidence(**changes) -> WristGateEvidence:
    values = dict(
        candidate_count=1,
        detection_id="anon-test-cube",
        colour="red",
        wrist_xy_mm=(300.5, 0.0),
        measured_yaw_deg=1.0,
        yaw_uncertainty_deg=0.5,
        closing_axis_uncertainty_mm=0.1,
        depth_valid_fraction=0.95,
        measured_cube_top_height_mm=10.5,
        expected_cube_top_height_mm=10.0,
        table_plane_residual_mm=0.5,
        approach_corridor_clear=True,
        centering_error_mm=0.1,
        depth_used=True,
    )
    values.update(changes)
    return WristGateEvidence(**values)


def wrist_report(evidence=None, observations=None, registration_value=None):
    return verify_wrist_observations(
        observations or [parsed_wrist(timestamp=10.0, offset=0), parsed_wrist(timestamp=10.1, offset=1)],
        selected_table_detection=selected_detection(),
        expected_yaw_deg=0.0,
        evidence=evidence or wrist_evidence(),
        wrist_rgb_calibration=calibration("wrist_rgb"),
        wrist_depth_calibration=calibration("wrist_depth"),
        registration=registration() if registration_value is None else registration_value,
        clearance_measurements=ClearanceMeasurements(10, 12, 0.05, 0.05),
    )


class NativeObservationTests(unittest.TestCase):
    def test_native_table_and_wrist_dimensions_preserved(self):
        for camera in ("table", "wrist"):
            with self.subTest(camera=camera):
                parsed = parse_native_observation(observation(camera), required_rgb=(camera,))
                self.assertEqual(parsed.rgb[camera].shape, NATIVE_RGB_SHAPES[camera])
                self.assertEqual(parsed.rgb[camera].dtype, np.uint8)
                self.assertFalse(parsed.rgb[camera].flags.writeable)

    def test_optional_and_required_depth(self):
        without = parse_native_observation(observation("table"), required_rgb=("table",))
        self.assertEqual(without.depth_mm, {})
        with self.assertRaises(ValueError):
            parse_native_observation(
                observation("table"), required_rgb=("table",), requested_depth=("table",)
            )
        present = parse_native_observation(
            observation("table", with_depth=True),
            required_rgb=("table",),
            requested_depth=("table",),
        )
        self.assertEqual(present.depth_mm["table"].shape, (400, 640))

    def test_wrong_shape_and_dtype_rejected(self):
        value = observation("wrist")
        value["images"]["wrist"] = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            parse_native_observation(value, required_rgb=("wrist",))
        value = observation("wrist")
        value["images"]["wrist"] = value["images"]["wrist"].astype(np.float32)
        with self.assertRaises(ValueError):
            parse_native_observation(value, required_rgb=("wrist",))

    def test_black_and_constant_frames_rejected(self):
        for fill in (0, 17):
            value = observation("wrist")
            value["images"]["wrist"].fill(fill)
            with self.subTest(fill=fill), self.assertRaises(ValueError):
                parse_native_observation(value, required_rgb=("wrist",))

    def test_stale_future_and_cross_trial_rejected(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            parse_native_observation(
                observation("wrist", timestamp=1),
                required_rgb=("wrist",),
                reference_time=2,
            )
        with self.assertRaisesRegex(ValueError, "future"):
            parse_native_observation(
                observation("wrist", timestamp=2),
                required_rgb=("wrist",),
                reference_time=1,
            )
        with self.assertRaisesRegex(ValueError, "trial"):
            parse_native_observation(
                observation("wrist", trial=2), required_rgb=("wrist",), expected_trial=1
            )

    def test_truth_and_state_fields_rejected(self):
        for field in ("ground_truth", "scene_state", "oracle_plan", "object_names"):
            value = observation("wrist")
            value[field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_native_observation(value, required_rgb=("wrist",))

    def test_hashes_change_with_frame_content(self):
        first = parse_native_observation(observation("wrist", offset=0), required_rgb=("wrist",))
        second = parse_native_observation(observation("wrist", offset=1), required_rgb=("wrist",))
        self.assertNotEqual(first.observation_sha256, second.observation_sha256)
        self.assertNotEqual(first.provenance[0].content_sha256, second.provenance[0].content_sha256)


class CalibrationContractTests(unittest.TestCase):
    def test_verified_real_calibration_round_trip(self):
        source = calibration("table_rgb")
        restored = CameraCalibration.deserialize(source.serialize())
        self.assertEqual(source, restored)
        self.assertIs(validate_camera_calibration(restored, expected_stream="table_rgb"), restored)

    def test_missing_unverified_simulation_and_wrong_resolution_fail(self):
        with self.assertRaises(ValueError):
            validate_camera_calibration(None, expected_stream="table_rgb")
        with self.assertRaises(ValueError):
            validate_camera_calibration(
                replace(calibration("table_rgb"), verified=False), expected_stream="table_rgb"
            )
        with self.assertRaises(ValueError):
            validate_camera_calibration(
                replace(calibration("table_rgb"), domain="simulation"), expected_stream="table_rgb"
            )
        with self.assertRaises(ValueError):
            validate_camera_calibration(
                replace(calibration("table_rgb"), resolution_wh=(3840, 2160)),
                expected_stream="table_rgb",
            )

    def test_hash_tamper_and_compatibility_mismatch_fail(self):
        payload = calibration("table_rgb").serialize()
        payload["frame_name"] = "tampered"
        with self.assertRaises(ValueError):
            CameraCalibration.deserialize(payload)
        with self.assertRaises(ValueError):
            validate_camera_calibration(
                calibration("table_rgb"),
                expected_stream="table_rgb",
                expected_compatibility_id="other",
            )

    def test_registration_is_explicit_and_compatible(self):
        rgb = calibration("wrist_rgb")
        depth = calibration("wrist_depth")
        self.assertTrue(validate_registration(registration(), rgb_calibration=rgb, depth_calibration=depth).verified)
        with self.assertRaises(ValueError):
            validate_registration(registration(verified=False), rgb_calibration=rgb, depth_calibration=depth)
        with self.assertRaises(ValueError):
            validate_registration(
                registration(),
                rgb_calibration=rgb,
                depth_calibration=calibration("wrist_depth", compatibility="other"),
            )


class DepthProcessingTests(unittest.TestCase):
    def test_zero_depth_mask_and_metre_conversion(self):
        source = np.array([[0, 500], [1000, 0]], dtype=np.uint16)
        frame = depth_mm_to_metres(source)
        self.assertTrue(np.isnan(frame.metres[0, 0]))
        self.assertEqual(frame.metres[0, 1], 0.5)
        self.assertEqual(frame.valid_fraction, 0.5)

    def test_out_of_range_depth_rejected(self):
        with self.assertRaises(ValueError):
            depth_mm_to_metres(np.array([[4000]], dtype=np.uint16))

    def test_local_statistics_samples_and_discontinuity(self):
        source = np.full((5, 5), 500, dtype=np.uint16)
        source[0, 0] = 0
        frame = depth_mm_to_metres(source)
        result = local_depth_statistics(
            frame,
            center_xy=(2, 2),
            radius_px=1,
            minimum_valid_samples=9,
            maximum_discontinuity_m=0.01,
        )
        self.assertEqual(result["sample_count"], 9)
        with self.assertRaises(ValueError):
            local_depth_statistics(
                frame,
                center_xy=(0, 0),
                radius_px=0,
                minimum_valid_samples=1,
                maximum_discontinuity_m=0.01,
            )
        source[2, 2] = 800
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            local_depth_statistics(
                depth_mm_to_metres(source),
                center_xy=(2, 2),
                radius_px=1,
                minimum_valid_samples=9,
                maximum_discontinuity_m=0.05,
            )

    def test_plane_residual(self):
        points = np.array([[0, 0, 1], [1, 0, 1.001], [0, 1, 0.999]], dtype=float)
        result = plane_residual_statistics(
            points, normal=(0, 0, 1), offset_m=-1, minimum_samples=3
        )
        self.assertLess(result["maximum_m"], 0.002)

    def test_rgb_lookup_requires_verified_registration_and_mapping(self):
        frame = depth_mm_to_metres(np.full((2, 2), 500, dtype=np.uint16))
        with self.assertRaises(ValueError):
            lookup_registered_rgb_depth(
                frame, rgb_xy=(1, 1), mapped_depth_xy=(1, 1), registration=None
            )
        with self.assertRaises(ValueError):
            lookup_registered_rgb_depth(
                frame, rgb_xy=(1, 1), mapped_depth_xy=None, registration=registration()
            )
        self.assertEqual(
            lookup_registered_rgb_depth(
                frame, rgb_xy=(1, 1), mapped_depth_xy=(1, 1), registration=registration()
            ),
            0.5,
        )


class TableScenePerceptionTests(unittest.TestCase):
    def parsed_table(self):
        return parse_native_observation(observation("table"), required_rgb=("table",))

    def call(
        self,
        detector=verified_fixture_detector,
        contract=None,
        configuration=None,
    ):
        return perceive_table_scene(
            self.parsed_table(),
            calibration=calibration("table_rgb"),
            workspace=workspace(),
            detector=detector,
            adapter_contract=adapter_contract(detector) if contract is None else contract,
            adapter_configuration=ADAPTER_CONFIGURATION if configuration is None else configuration,
        )

    def test_exact_two_two_two_inventory_and_anonymous_output(self):
        result = self.call()
        self.assertTrue(result["passed"])
        self.assertEqual(result["counts_by_colour"], {"red": 2, "blue": 2, "yellow": 2})
        self.assertTrue(all(row["detection_id"].startswith("anon-") for row in result["detections"]))
        self.assertFalse(result["truth_used"])
        self.assertIsNone(result["action"])
        self.assertFalse(result["robot_motion_authorized"])

    def test_incomplete_and_duplicate_inventory_fail(self):
        self.assertFalse(self.call(incomplete_fixture_detector)["passed"])
        self.assertIn(
            "duplicate_cube_positions", self.call(duplicate_fixture_detector)["reasons"]
        )

    def test_detector_identity_truth_and_oracle_fields_rejected(self):
        with self.assertRaises(ValueError):
            self.call(forbidden_identity_fixture_detector)

    def test_missing_real_calibration_or_workspace_fails(self):
        with self.assertRaises(ValueError):
            perceive_table_scene(
                self.parsed_table(),
                calibration=None,
                workspace=workspace(),
                detector=lambda *_: detector_result(),
                adapter_contract=None,
                adapter_configuration={},
            )
        with self.assertRaises(ValueError):
            perceive_table_scene(
                self.parsed_table(),
                calibration=calibration("table_rgb"),
                workspace=None,
                detector=lambda *_: detector_result(),
                adapter_contract=None,
                adapter_configuration={},
            )

    def test_bare_and_missing_contract_rejected(self):
        with self.assertRaises(TypeError):
            perceive_table_scene(
                self.parsed_table(),
                calibration=calibration("table_rgb"),
                workspace=workspace(),
                detector=verified_fixture_detector,
            )
        with self.assertRaises(ValueError):
            self.call(contract=adapter_contract(verified_fixture_detector, verified=False))
        with self.assertRaises(ValueError):
            perceive_table_scene(
                self.parsed_table(),
                calibration=calibration("table_rgb"),
                workspace=workspace(),
                detector=verified_fixture_detector,
                adapter_contract=None,
                adapter_configuration=ADAPTER_CONFIGURATION,
            )

    def test_missing_and_malformed_hashes_rejected(self):
        cases = (
            {"verification_evidence_sha256": ""},
            {"adapter_configuration_sha256": "not-a-sha256"},
            {"callable_module_file_sha256": "bad"},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.call(contract=adapter_contract(**changes))
        with self.assertRaisesRegex(ValueError, "canonical contract hash mismatch"):
            self.call(
                contract=replace(
                    adapter_contract(), canonical_contract_sha256="1" * 64
                )
            )

    def test_runtime_callable_identity_mismatches_rejected(self):
        cases = (
            {"callable_module_name": "wrong.module"},
            {"callable_qualified_name": "wrong_qualified_name"},
            {"callable_module_file_sha256": "1" * 64},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.call(contract=adapter_contract(**changes))

    def test_lambda_builtin_callable_object_and_dynamic_closure_rejected(self):
        class CallableObject:
            def __call__(self, *_):
                return detector_result()

        def make_closure():
            captured = detector_result()

            def dynamic(*_):
                return captured

            return dynamic

        for candidate in (lambda *_: detector_result(), len, CallableObject(), make_closure()):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                inspect_detector_callable(candidate)

    def test_color_conversion_and_output_schema_rejected(self):
        cases = (
            {"input_color_convention": "opencv_bgr"},
            {"internal_detector_color_convention": "organizer_native_rgb"},
            {"color_conversion_convention": ""},
            {"output_schema_version": "unsupported-v99"},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.call(contract=adapter_contract(**changes))

    def test_calibration_workspace_and_configuration_compatibility_rejected(self):
        with self.assertRaises(ValueError):
            self.call(contract=adapter_contract(compatible_calibration_ids=("other",)))
        with self.assertRaises(ValueError):
            self.call(contract=adapter_contract(compatible_workspace_versions=("other",)))
        with self.assertRaises(ValueError):
            self.call(configuration={"different": True})

    def test_verified_fixture_records_complete_adapter_provenance(self):
        result = self.call()
        expected = {
            "adapter_contract_sha256",
            "adapter_name",
            "adapter_semantic_version",
            "adapter_callable_module",
            "adapter_callable_qualified_name",
            "adapter_module_file_sha256",
            "adapter_configuration_sha256",
            "adapter_color_conversion_convention",
            "adapter_output_schema_version",
            "adapter_verification_evidence_sha256",
            "calibration_sha256",
            "workspace_sha256",
        }
        self.assertTrue(expected <= set(result))
        self.assertEqual(result["adapter_module_file_sha256"], inspect_detector_callable(verified_fixture_detector)["module_file_sha256"])

    def test_adapter_output_cannot_inject_motion_authority(self):
        result = self.call(motion_injecting_fixture_detector)
        self.assertIsNone(result["action"])
        self.assertFalse(result["robot_motion_authorized"])

    def test_schema_valid_unreviewed_callable_cannot_bypass_contract(self):
        reviewed = adapter_contract(verified_fixture_detector)
        with self.assertRaises(ValueError):
            self.call(unreviewed_schema_valid_fixture_detector, contract=reviewed)


class WristGateTests(unittest.TestCase):
    def test_complete_stationary_bundle_passes_review_only(self):
        result = wrist_report()
        self.assertTrue(result["passed"])
        self.assertIsNone(result["action"])
        self.assertFalse(result["robot_motion_authorized"])

    def test_wrong_colour_ambiguous_missing_and_excessive_yaw_fail(self):
        variants = (
            wrist_evidence(colour="blue"),
            wrist_evidence(candidate_count=2),
            wrist_evidence(measured_yaw_deg=None),
            wrist_evidence(measured_yaw_deg=4.0),
        )
        for evidence in variants:
            with self.subTest(evidence=evidence):
                self.assertFalse(wrist_report(evidence=evidence)["passed"])

    def test_unregistered_depth_rejected(self):
        result = wrist_report(registration_value=registration(verified=False))
        self.assertFalse(result["passed"])
        self.assertTrue(any("registration" in reason for reason in result["reasons"]))

    def test_insufficient_depth_and_corridor_failure(self):
        self.assertFalse(wrist_report(evidence=wrist_evidence(depth_valid_fraction=0.5))["passed"])
        self.assertFalse(wrist_report(evidence=wrist_evidence(approach_corridor_clear=False))["passed"])

    def test_cross_trial_stationarity_and_frame_consistency_fail(self):
        cross_trial = [parsed_wrist(timestamp=10, trial=1), parsed_wrist(timestamp=10.1, trial=2, offset=1)]
        self.assertIn("cross_trial_wrist_observations", wrist_report(observations=cross_trial)["reasons"])
        moving = [parsed_wrist(timestamp=10), parsed_wrist(timestamp=10.1, offset=1, joint_delta=1)]
        self.assertIn("robot_not_stationary_joint_delta", wrist_report(observations=moving)["reasons"])
        frozen = [parsed_wrist(timestamp=10), parsed_wrist(timestamp=10.1)]
        self.assertTrue(any("frame" in reason for reason in wrist_report(observations=frozen)["reasons"]))
        inconsistent = [parsed_wrist(timestamp=10), parsed_wrist(timestamp=10.1, offset=100)]
        self.assertIn("repeated_wrist_frames_inconsistent", wrist_report(observations=inconsistent)["reasons"])

    def test_unknown_gripper_tolerances_fail_closed(self):
        result = verify_wrist_observations(
            [parsed_wrist(timestamp=10), parsed_wrist(timestamp=10.1, offset=1)],
            selected_table_detection=selected_detection(),
            expected_yaw_deg=0,
            evidence=wrist_evidence(),
            wrist_rgb_calibration=calibration("wrist_rgb"),
            wrist_depth_calibration=calibration("wrist_depth"),
            registration=registration(),
        )
        self.assertFalse(result["passed"])
        self.assertIn("finger_tolerance_unverified", result["reasons"])


class ObservationSessionTests(unittest.TestCase):
    def test_complete_review_has_hash_chain_and_no_action(self):
        session = ObservationOnlySession()
        session.begin()
        session.record_table_observation("table-observation-hash")
        session.record_table_verification({"passed": True, "record_sha256": "table-report-hash"})
        session.record_wrist_observations(["wrist-1", "wrist-2"])
        session.record_wrist_verification(
            {
                "passed": True,
                "report_sha256": "wrist-report-hash",
                "action": None,
                "robot_motion_authorized": False,
            }
        )
        report = session.review_record()
        self.assertEqual(session.phase, ObservationPhase.REVIEW_COMPLETE)
        self.assertIsNone(report["action"])
        self.assertFalse(report["robot_motion_authorized"])
        self.assertEqual(len({event["event_sha256"] for event in report["events"]}), 5)

    def test_failure_latches_until_reset(self):
        session = ObservationOnlySession()
        session.begin()
        session.record_table_observation("table")
        session.record_table_verification({"passed": False})
        self.assertEqual(session.phase, ObservationPhase.ABORT_LATCHED)
        with self.assertRaises(RuntimeError):
            session.record_wrist_observations(["a", "b"])
        session.reset()
        self.assertEqual(session.phase, ObservationPhase.RESET)

    def test_wrist_proposal_is_unresolved_and_nontransmittable(self):
        proposal = unresolved_wrist_observation_proposal()
        self.assertFalse(proposal["resolved"])
        self.assertIsNone(proposal["tcp_pose"])
        self.assertIsNone(proposal["action_rows"])
        self.assertFalse(proposal["transmittable"])

    def test_no_client_network_simulator_or_transmission_surface(self):
        modules = (
            "native_observation.py",
            "camera_calibration_contract.py",
            "depth_processing.py",
            "table_scene_perception.py",
            "wrist_observation_gate.py",
            "observation_only_session.py",
        )
        forbidden = (
            "cell_client",
            "requests",
            "urllib",
            "socket",
            "subprocess",
            "isaac",
            "cosmos",
            "phase4c_yaw_",
            "yaad_results",
        )
        root = Path(__file__).parent
        for filename in modules:
            text = (root / filename).read_text(encoding="utf-8").lower()
            for value in forbidden:
                with self.subTest(filename=filename, value=value):
                    self.assertNotIn(value, text)
        public = {name for name, _ in inspect.getmembers(ObservationOnlySession) if not name.startswith("_")}
        self.assertTrue({"send", "act", "transmit", "execute"}.isdisjoint(public))

    def test_production_readiness_remains_false_false(self):
        report = PRODUCTION_READINESS.report()
        self.assertFalse(report["all_requirements_verified"])
        self.assertFalse(report["robot_motion_authorized"])


if __name__ == "__main__":
    unittest.main()
