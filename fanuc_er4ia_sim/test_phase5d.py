"""Host-only tests for the Phase 5D organizer observation harness."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from deployment_readiness import PRODUCTION_READINESS
from mock_policy_harness import (
    MockPolicyHarness,
    ORGANIZER_REQUIRES_ACTION_EACH_DECISION_CYCLE,
    validate_mock_hold,
)
from native_observation import NATIVE_DEPTH_SHAPES, NATIVE_RGB_SHAPES
from observation_only_session import ObservationOnlySession, ObservationPhase
import observation_replay
from observation_replay import (
    MAX_JSON_NESTING_DEPTH,
    MAX_METADATA_STRING_LENGTH,
    MAX_REPLAY_JSON_BYTES,
    MAX_REPLAY_RECORDS,
    VerificationEvidenceVerifier,
    _decode_encoded_frame,
    build_verification_envelope,
    encode_frame_fixture,
    load_explicit_json_observations,
    replay_observation_review,
)
from organizer_observation_adapter import OrganizerObservationAdapter
from real_policy_readiness import (
    PHASE5C_REQUIRED_VERIFICATIONS,
    PRODUCTION_POLICY_READINESS,
    RealPolicyReadiness,
    UNRESOLVED_CSI_PREREQUISITES,
)


TRIAL = {"trial_id": "synthetic-trial", "status": "running", "layout": "test"}


def native_rgb(camera: str, offset: int = 0) -> np.ndarray:
    height, width, channels = NATIVE_RGB_SHAPES[camera]
    image = np.zeros((height, width, channels), dtype=np.uint8)
    row = ((np.arange(width, dtype=np.uint16) + offset) % 251).astype(np.uint8)
    image[:, :, 0] = row
    image[:, :, 1] = (row.astype(np.uint16) * 3 % 251).astype(np.uint8)
    return image


def organizer_observation(
    camera: str = "table",
    *,
    timestamp: float = 10.0,
    trial: object = TRIAL,
    queued: int = 0,
    offset: int = 0,
    depth: bool = False,
) -> dict:
    value = {
        "t": timestamp,
        "status": "running",
        "trial": trial,
        "queued": queued,
        "joints": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
        "tcp_pose": np.array([300, 0, 100, -180, 0, 0], dtype=np.float64),
        "format": "png",
        "images": {camera: native_rgb(camera, offset)},
    }
    if depth:
        value["depth"] = {
            camera: np.full(NATIVE_DEPTH_SHAPES[camera], 500, dtype=np.uint16)
        }
    return value


def adapt(value: dict | None = None, camera: str = "table"):
    adapter = OrganizerObservationAdapter()
    adapter.reset(TRIAL)
    return adapter.adapt(
        organizer_observation(camera) if value is None else value,
        required_rgb=(camera,),
        requested_depth=(camera,) if value is not None and "depth" in value else (),
        reference_time=10.01,
    )


def replay_kwargs() -> dict:
    table_observation = organizer_observation("table", timestamp=10.0, depth=True)
    wrist_observations = [
        organizer_observation("wrist", timestamp=10.1, offset=1, depth=True),
        organizer_observation("wrist", timestamp=10.2, offset=2, depth=True),
    ]
    adapter = OrganizerObservationAdapter()
    session = ObservationOnlySession()
    adapter.reset(TRIAL)
    session.begin()
    table = adapter.adapt(
        table_observation,
        required_rgb=("table",),
        requested_depth=("table",),
        reference_time=10.01,
    )
    session.record_table_observation(table.native.observation_sha256)
    detection_hash = hashlib.sha256(b"anonymous-detection").hexdigest()
    table_payload = {
        "passed": True,
        "record_sha256": hashlib.sha256(b"table-review").hexdigest(),
        "anonymous_detection_sha256": [detection_hash],
    }
    table_envelope = build_verification_envelope(
        trial_sha256=table.native.trial_sha256,
        source_observation_sha256=table.native.observation_sha256,
        report_kind="table",
        payload=table_payload,
        parent_session_sha256=session.review_record()["event_chain_head"],
    )
    session.record_table_verification(table_payload)
    wrists = [
        adapter.adapt(
            value,
            required_rgb=("wrist",),
            requested_depth=("wrist",),
            reference_time=reference,
        )
        for value, reference in zip(wrist_observations, (10.11, 10.21))
    ]
    session.record_wrist_observations(
        [value.native.observation_sha256 for value in wrists]
    )
    wrist_payload = {
        "passed": True,
        "report_sha256": hashlib.sha256(b"wrist-review").hexdigest(),
        "selected_anonymous_detection_sha256": detection_hash,
        "action": None,
        "robot_motion_authorized": False,
    }
    wrist_envelope = build_verification_envelope(
        trial_sha256=table.native.trial_sha256,
        source_observation_sha256=wrists[-1].native.observation_sha256,
        report_kind="wrist",
        payload=wrist_payload,
        parent_session_sha256=session.review_record()["event_chain_head"],
        selected_anonymous_detection_sha256=detection_hash,
    )
    return {
        "trial": TRIAL,
        "table_observation": table_observation,
        "wrist_observations": wrist_observations,
        "table_verification_envelope": table_envelope,
        "wrist_verification_envelope": wrist_envelope,
        "table_reference_time": 10.01,
        "wrist_reference_times": [10.11, 10.21],
    }


class OrganizerObservationAdapterTests(unittest.TestCase):
    def test_exact_field_mapping_and_provenance(self):
        result = adapt(organizer_observation(depth=True))
        self.assertEqual(result.native.rgb["table"].shape, (1080, 1920, 3))
        self.assertEqual(result.native.depth_mm["table"].shape, (400, 640))
        self.assertFalse(result.native.rgb["table"].flags.writeable)
        self.assertEqual(len(result.native.observation_sha256), 64)
        self.assertEqual(result.review_metadata()["source_schema"], "organizer_decoded_observation_v1")

    def test_optional_depth_absent(self):
        self.assertEqual(adapt().native.depth_mm, {})

    def test_missing_and_unknown_fields_rejected(self):
        missing = organizer_observation()
        missing.pop("tcp_pose")
        with self.assertRaisesRegex(ValueError, "missing"):
            adapt(missing)
        unknown = organizer_observation()
        unknown["tcp"] = unknown["tcp_pose"]
        with self.assertRaisesRegex(ValueError, "unknown"):
            adapt(unknown)

    def test_truth_state_oracle_placement_and_object_fields_rejected(self):
        for field in ("truth", "scene_state", "oracle_plan", "placement", "object_names"):
            value = organizer_observation()
            value[field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapt(value)

    def test_reset_required_and_abort_latched(self):
        adapter = OrganizerObservationAdapter()
        with self.assertRaisesRegex(ValueError, "reset"):
            adapter.adapt(organizer_observation(), required_rgb=("table",))
        with self.assertRaisesRegex(RuntimeError, "latched"):
            adapter.adapt(organizer_observation(), required_rgb=("table",))
        adapter.reset(TRIAL)
        self.assertEqual(
            adapter.adapt(organizer_observation(), required_rgb=("table",)).observation_sequence,
            1,
        )

    def test_trial_boundary_and_monotonic_timestamp(self):
        adapter = OrganizerObservationAdapter()
        adapter.reset(TRIAL)
        adapter.adapt(organizer_observation(timestamp=10), required_rgb=("table",))
        with self.assertRaisesRegex(ValueError, "trial"):
            adapter.adapt(
                organizer_observation(timestamp=11, trial={"trial_id": "other"}),
                required_rgb=("table",),
            )
        adapter.reset(TRIAL)
        adapter.adapt(organizer_observation(timestamp=10), required_rgb=("table",))
        with self.assertRaisesRegex(ValueError, "increase"):
            adapter.adapt(organizer_observation(timestamp=10), required_rgb=("table",))

    def test_stale_and_future_timestamps(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            adapt_with_reference(organizer_observation(timestamp=1), 10)
        with self.assertRaisesRegex(ValueError, "future"):
            adapt_with_reference(organizer_observation(timestamp=11), 10)

    def test_queued_and_nonrunning_rejected(self):
        with self.assertRaisesRegex(ValueError, "queue"):
            adapt(organizer_observation(queued=1))
        value = organizer_observation()
        value["status"] = "idle"
        with self.assertRaisesRegex(ValueError, "running"):
            adapt(value)

    def test_native_shape_and_dtype_rejected(self):
        bad_shape = organizer_observation()
        bad_shape["images"]["table"] = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            adapt(bad_shape)
        bad_dtype = organizer_observation(depth=True)
        bad_dtype["depth"]["table"] = bad_dtype["depth"]["table"].astype(np.float32)
        with self.assertRaises(ValueError):
            adapt(bad_dtype)

    def test_malformed_vectors_and_gripper_rejected(self):
        for field, replacement in (
            ("tcp_pose", [1, 2]),
            ("joints", [1, 2]),
            ("joints", [0, 0, 0, 0, 0, 0, 0.5]),
        ):
            value = organizer_observation()
            value[field] = replacement
            with self.subTest(field=field, replacement=replacement), self.assertRaises(ValueError):
                adapt(value)


def adapt_with_reference(value: dict, reference: float):
    adapter = OrganizerObservationAdapter()
    adapter.reset(TRIAL)
    return adapter.adapt(value, required_rgb=("table",), reference_time=reference)


def minimal_file_record() -> dict:
    return {
        "t": 10.0,
        "status": "running",
        "trial": TRIAL,
        "queued": 0,
        "joints": [0, 0, 0, 0, 0, 0, 1],
        "tcp_pose": [300, 0, 100, -180, 0, 0],
        "format": "png",
    }


def write_replay_file(path: Path, records: list[dict], **metadata) -> None:
    payload = {"schema": "organizer_observation_replay_v1", "observations": records}
    payload.update(metadata)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ReplayTests(unittest.TestCase):
    def test_deterministic_replay_and_hash_chain(self):
        first = replay_observation_review(**replay_kwargs())
        second = replay_observation_review(**replay_kwargs())
        self.assertEqual(first, second)
        self.assertEqual(first["session"]["phase"], "REVIEW_COMPLETE")
        self.assertEqual(len(first["replay_sha256"]), 64)
        events = first["session"]["events"]
        self.assertIsNone(events[0]["previous_event_sha256"])
        for previous, current in zip(events, events[1:]):
            self.assertEqual(current["previous_event_sha256"], previous["event_sha256"])
        json.dumps(first, sort_keys=True, allow_nan=False)
        self.assertIsNone(first["action"])
        self.assertFalse(first["robot_motion_authorized"])

    def test_cross_trial_observation_rejected(self):
        values = replay_kwargs()
        values["wrist_observations"][1]["trial"] = {"trial_id": "other"}
        with self.assertRaisesRegex(ValueError, "trial"):
            replay_observation_review(**values)

    def test_explicit_file_loader_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            path.write_text(json.dumps({"schema": "wrong", "observations": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_explicit_json_observations(path)
            with self.assertRaises(ValueError):
                load_explicit_json_observations(Path(directory) / "missing.json")

    def test_zero_length_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_explicit_json_observations(path)

    def test_oversized_file_rejected_before_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            with path.open("wb") as stream:
                stream.seek(MAX_REPLAY_JSON_BYTES)
                stream.write(b"x")
            with patch.object(observation_replay.json, "loads", side_effect=AssertionError):
                with self.assertRaisesRegex(ValueError, "byte limit"):
                    load_explicit_json_observations(path)

    def test_maximum_plus_one_bounded_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            with path.open("wb") as stream:
                stream.seek(MAX_REPLAY_JSON_BYTES)
                stream.write(b"x")
            actual = path.stat()
            fields = list(actual)
            fields[6] = MAX_REPLAY_JSON_BYTES
            reported_at_limit = os.stat_result(fields)
            original_stat = Path.stat

            def bounded_race_stat(candidate, *args, **kwargs):
                if candidate == path:
                    return reported_at_limit
                return original_stat(candidate, *args, **kwargs)

            with patch.object(Path, "stat", bounded_race_stat), patch.object(
                observation_replay.json, "loads", side_effect=AssertionError
            ):
                with self.assertRaisesRegex(ValueError, "bounded read"):
                    load_explicit_json_observations(path)

    def test_non_regular_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "regular file"):
                load_explicit_json_observations(directory)

    def test_symlink_is_recorded_but_never_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            link = root / "link.json"
            write_replay_file(target, [minimal_file_record()])
            original_lstat = Path.lstat
            original_resolve = Path.resolve

            def symlink_lstat(path):
                if path == link:
                    return os.stat_result((stat.S_IFLNK, 0, 0, 0, 0, 0, 0, 0, 0, 0))
                return original_lstat(path)

            def symlink_resolve(path, strict=False):
                if path == link:
                    return original_resolve(target, strict=strict)
                return original_resolve(path, strict=strict)

            with patch.object(Path, "lstat", symlink_lstat), patch.object(
                Path, "resolve", symlink_resolve
            ):
                loaded = load_explicit_json_observations(link)
            self.assertTrue(loaded.supplied_path_was_symlink)
            self.assertEqual(Path(loaded.resolved_path), target.resolve())
            self.assertFalse(loaded.trusted_production_evidence)
            self.assertEqual(len(loaded.file_sha256), 64)

    def test_too_many_records_rejected_before_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            write_replay_file(path, [minimal_file_record()] * (MAX_REPLAY_RECORDS + 1))
            with patch.object(observation_replay, "_decode_record", side_effect=AssertionError):
                with self.assertRaisesRegex(ValueError, "too many"):
                    load_explicit_json_observations(path)

    def test_excessive_nesting_rejected_before_json_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            nested = "[" * (MAX_JSON_NESTING_DEPTH + 1) + "0" + "]" * (MAX_JSON_NESTING_DEPTH + 1)
            path.write_text(
                '{"schema":"organizer_observation_replay_v1","observations":['
                + json.dumps(minimal_file_record())
                + '],"nested":'
                + nested
                + "}",
                encoding="utf-8",
            )
            with patch.object(observation_replay.json, "loads", side_effect=AssertionError):
                with self.assertRaisesRegex(ValueError, "nesting"):
                    load_explicit_json_observations(path)

    def test_oversized_metadata_string_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            write_replay_file(
                path,
                [minimal_file_record()],
                note="x" * (MAX_METADATA_STRING_LENGTH + 1),
            )
            with self.assertRaisesRegex(ValueError, "metadata string"):
                load_explicit_json_observations(path)

    def test_session_abort_latch_and_reset_regression(self):
        session = ObservationOnlySession()
        with self.assertRaises(RuntimeError):
            session.record_table_observation("out-of-order")
        self.assertEqual(session.phase, ObservationPhase.ABORT_LATCHED)
        with self.assertRaises(RuntimeError):
            session.begin()
        session.reset()
        session.begin()
        self.assertEqual(session.phase, ObservationPhase.TABLE_OBSERVE)


class EncodedFrameBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = np.zeros(NATIVE_DEPTH_SHAPES["wrist"], dtype=np.uint16)
        cls.valid = encode_frame_fixture(frame, stream="depth", camera="wrist")

    def changed(self, **updates):
        value = dict(self.valid)
        value["shape"] = list(value["shape"])
        value.update(updates)
        return value

    def test_valid_exact_native_frame_passes(self):
        frame, encoded_bytes = _decode_encoded_frame(self.valid, "depth", "wrist")
        self.assertEqual(frame.shape, NATIVE_DEPTH_SHAPES["wrist"])
        self.assertEqual(frame.dtype, np.uint16)
        self.assertEqual(encoded_bytes, len(self.valid["data"]))

    def test_wrong_native_shape_and_excess_dimensions_rejected(self):
        for shape in ([480, 849], [481, 848], [480, 848, 2]):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "shape"):
                _decode_encoded_frame(self.changed(shape=shape), "depth", "wrist")

    def test_declared_element_and_byte_count_mismatch_rejected(self):
        for field, value in (("element_count", 1), ("raw_bytes", 1)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "declared size"):
                _decode_encoded_frame(self.changed(**{field: value}), "depth", "wrist")

    def test_ragged_boolean_negative_and_out_of_range_samples_rejected(self):
        for data in ([[0], [0, 1]], [True], [-1], [65536]):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "base64"):
                _decode_encoded_frame(self.changed(data=data), "depth", "wrist")

    def test_malformed_base64_rejected(self):
        malformed = "!" + self.valid["data"][1:]
        with self.assertRaisesRegex(ValueError, "malformed"):
            _decode_encoded_frame(self.changed(data=malformed), "depth", "wrist")

    def test_wrong_dtype_rejected(self):
        with self.assertRaisesRegex(ValueError, "dtype"):
            _decode_encoded_frame(self.changed(dtype="uint8"), "depth", "wrist")

    def test_invalid_payload_rejected_before_numpy_conversion(self):
        with patch.object(observation_replay.np, "frombuffer", side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError, "shape"):
                _decode_encoded_frame(self.changed(shape=[1, 1]), "depth", "wrist")

    def test_record_encoded_limit_checked_before_frame_conversion(self):
        record = minimal_file_record()
        record["images"] = {"table": {"data": "123456"}}
        with patch.object(
            observation_replay, "MAX_ENCODED_FRAME_BYTES_PER_RECORD", 5
        ), patch.object(
            observation_replay, "_decode_encoded_frame", side_effect=AssertionError
        ):
            with self.assertRaisesRegex(ValueError, "encoded-frame byte limit"):
                observation_replay._decode_record(record)


class EvidenceEnvelopeTests(unittest.TestCase):
    def test_table_evidence_from_another_trial_rejected(self):
        values = replay_kwargs()
        values["table_verification_envelope"]["trial_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "another trial"):
            replay_observation_review(**values)

    def test_wrist_evidence_from_another_trial_rejected(self):
        values = replay_kwargs()
        values["wrist_verification_envelope"]["trial_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "another trial"):
            replay_observation_review(**values)

    def test_wrong_table_source_observation_rejected(self):
        values = replay_kwargs()
        values["table_verification_envelope"]["source_observation_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "wrong source"):
            replay_observation_review(**values)

    def test_wrong_wrist_source_observation_rejected(self):
        values = replay_kwargs()
        values["wrist_verification_envelope"]["source_observation_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "not part"):
            replay_observation_review(**values)

    def test_wrong_selected_detection_rejected(self):
        values = replay_kwargs()
        old = values["wrist_verification_envelope"]
        unknown = hashlib.sha256(b"unknown-detection").hexdigest()
        payload = dict(old["payload"])
        payload["selected_anonymous_detection_sha256"] = unknown
        values["wrist_verification_envelope"] = build_verification_envelope(
            trial_sha256=old["trial_sha256"],
            source_observation_sha256=old["source_observation_sha256"],
            report_kind="wrist",
            payload=payload,
            parent_session_sha256=old["parent_session_sha256"],
            selected_anonymous_detection_sha256=unknown,
        )
        with self.assertRaisesRegex(ValueError, "unknown anonymous"):
            replay_observation_review(**values)

    def test_modified_report_payload_rejected(self):
        values = replay_kwargs()
        values["table_verification_envelope"]["payload"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "payload hash"):
            replay_observation_review(**values)

    def test_stale_parent_hash_rejected(self):
        values = replay_kwargs()
        values["table_verification_envelope"]["parent_session_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "stale"):
            replay_observation_review(**values)

    def test_malformed_canonical_envelope_hash_rejected(self):
        values = replay_kwargs()
        values["table_verification_envelope"]["envelope_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "canonical"):
            replay_observation_review(**values)

    def test_reused_evidence_envelope_rejected(self):
        values = replay_kwargs()
        envelope = values["table_verification_envelope"]
        verifier = VerificationEvidenceVerifier(envelope["trial_sha256"])
        arguments = {
            "report_kind": "table",
            "source_observation_sha256": envelope["source_observation_sha256"],
            "parent_session_sha256": envelope["parent_session_sha256"],
        }
        verifier.verify(envelope, **arguments)
        with self.assertRaisesRegex(ValueError, "twice"):
            verifier.verify(envelope, **arguments)


class MockHarnessTests(unittest.TestCase):
    def test_organizer_requires_action_but_mock_hold_is_local(self):
        self.assertTrue(ORGANIZER_REQUIRES_ACTION_EACH_DECISION_CYCLE)
        current = adapt()
        record = MockPolicyHarness(mode="mock").propose_hold(current)
        self.assertEqual(record["tcp"][0][:6], current.native.tcp_pose.tolist())
        self.assertEqual(record["tcp"][0][6], 1.0)
        self.assertTrue(record["mock_only"])
        self.assertFalse(record["transmitted"])
        self.assertFalse(record["robot_motion_authorized"])

    def test_altered_hold_rejected(self):
        current = adapt()
        record = MockPolicyHarness(mode="test").propose_hold(current)
        record["tcp"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_mock_hold(record, current)

    def test_nonbinary_gripper_rejected(self):
        value = organizer_observation()
        value["joints"][6] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            adapt(value)

    def test_harness_unavailable_outside_explicit_mock_mode(self):
        for mode in ("production", "real", "", "MOCK"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                MockPolicyHarness(mode=mode)

    def test_production_output_has_no_action_or_mock_row(self):
        output = MockPolicyHarness.production_output()
        self.assertIsNone(output["action"])
        self.assertFalse(output["robot_motion_authorized"])
        self.assertNotIn("tcp", output)


class ReadinessAndBoundaryTests(unittest.TestCase):
    def test_defaults_are_fail_closed(self):
        report = PRODUCTION_POLICY_READINESS.report()
        self.assertFalse(report["all_requirements_verified"])
        self.assertFalse(report["production_ready"])
        self.assertFalse(report["robot_motion_authorized"])
        self.assertIsNone(report["action"])
        self.assertFalse(PRODUCTION_READINESS.report()["robot_motion_authorized"])

    def test_all_true_mock_inputs_cannot_authorize_production(self):
        report = RealPolicyReadiness(
            host_contract_verified=True,
            mock_interface_verified=True,
            phase5c_verifications={name: True for name in PHASE5C_REQUIRED_VERIFICATIONS},
            observation_session_complete=True,
        ).report()
        self.assertTrue(report["host_contract_verified"])
        self.assertTrue(report["mock_interface_verified"])
        self.assertTrue(report["observation_only_physical_checkout_pending"])
        self.assertTrue(report["motion_deployment_unauthorized"])
        self.assertFalse(report["all_requirements_verified"])
        self.assertFalse(report["production_ready"])
        self.assertFalse(report["robot_motion_authorized"])

    def test_unresolved_prerequisite_inventory_is_complete_and_stable(self):
        expected = set(PRODUCTION_READINESS.report()["missing"])
        expected.update(PHASE5C_REQUIRED_VERIFICATIONS)
        expected.update(
            {
                "organizer_policy_action_strategy",
                "observation_only_physical_checkout",
                "physical_safety_review",
                "production_policy_review",
            }
        )
        self.assertEqual(set(UNRESOLVED_CSI_PREREQUISITES), expected)

    def test_phase5d_modules_have_no_transport_or_simulator_imports(self):
        root = Path(__file__).resolve().parent
        files = (
            "organizer_observation_adapter.py",
            "observation_replay.py",
            "mock_policy_harness.py",
            "real_policy_readiness.py",
        )
        forbidden_imports = (
            "requests", "socket", "http.client", "cell_client", "mock_cell",
            "isaac", "omni", "controller", "robot_server",
        )
        for filename in files:
            source = (root / filename).read_text(encoding="utf-8").lower()
            imports = [line.strip() for line in source.splitlines() if line.lstrip().startswith(("import ", "from "))]
            with self.subTest(filename=filename):
                self.assertFalse(any(token in line for token in forbidden_imports for line in imports))

    def test_no_production_policy_class_or_transmission_method(self):
        import mock_policy_harness
        import observation_replay
        import organizer_observation_adapter

        combined = "\n".join(
            inspect.getsource(module)
            for module in (mock_policy_harness, observation_replay, organizer_observation_adapter)
        )
        self.assertNotIn("class Policy", combined)
        self.assertNotIn("def send", combined)
        self.assertNotIn("def act(", combined)


if __name__ == "__main__":
    unittest.main()
