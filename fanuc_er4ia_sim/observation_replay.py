"""Bounded, deterministic, action-free replay of explicit observations.

The JSON format is a host-only fixture format, not a bulk image archive. Each
frame is exact native bytes encoded with RFC 4648 base64 and accompanied by its
dtype, shape, element count, and raw byte count. File and structural limits are
applied before JSON parsing or NumPy frame construction.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Iterable

import numpy as np

from native_observation import NATIVE_DEPTH_SHAPES, NATIVE_RGB_SHAPES
from observation_only_session import ObservationOnlySession
from organizer_observation_adapter import OrganizerObservationAdapter


MAX_REPLAY_JSON_BYTES = 24 * 1024 * 1024
MAX_REPLAY_RECORDS = 4
MAX_ENCODED_FRAME_BYTES_PER_RECORD = 10 * 1024 * 1024
MAX_JSON_NESTING_DEPTH = 12
MAX_METADATA_STRING_LENGTH = 4096
REPLAY_SCHEMA = "organizer_observation_replay_v1"
EVIDENCE_SCHEMA = "phase5d_verification_evidence_v1"
FRAME_ENCODING = "base64"

FRAME_CONTRACTS = {
    ("images", name): ("uint8", shape) for name, shape in NATIVE_RGB_SHAPES.items()
} | {
    ("depth", name): ("uint16", shape) for name, shape in NATIVE_DEPTH_SHAPES.items()
}
MAX_NATIVE_ELEMENTS = {
    f"{stream}.{camera}": math.prod(shape)
    for (stream, camera), (_, shape) in FRAME_CONTRACTS.items()
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class LoadedReplay:
    observations: tuple[dict, ...]
    supplied_path: str
    resolved_path: str
    supplied_path_was_symlink: bool
    file_bytes: int
    file_sha256: str
    trusted_production_evidence: bool = False


def _scan_json_before_parse(raw: bytes) -> str:
    """Bound nesting and string tokens before invoking the JSON parser."""
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("replay JSON must be valid UTF-8") from exc
    depth = 0
    in_string = False
    escaped = False
    string_length = 0
    largest_string = 0
    for character in text:
        if in_string:
            if escaped:
                escaped = False
                string_length += 1
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
                largest_string = max(largest_string, string_length)
            else:
                string_length += 1
            continue
        if character == '"':
            in_string = True
            string_length = 0
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_NESTING_DEPTH:
                raise ValueError("replay JSON exceeds the maximum nesting depth")
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("replay JSON structure is unbalanced")
    if in_string or depth != 0:
        raise ValueError("replay JSON structure is incomplete")
    if largest_string > MAX_ENCODED_FRAME_BYTES_PER_RECORD:
        raise ValueError("replay JSON contains an oversized string token")
    return text


def _validate_metadata(value: object, *, depth: int = 0, frame_data: bool = False) -> None:
    if depth > MAX_JSON_NESTING_DEPTH:
        raise ValueError("replay value exceeds the maximum nesting depth")
    if isinstance(value, str):
        limit = MAX_ENCODED_FRAME_BYTES_PER_RECORD if frame_data else MAX_METADATA_STRING_LENGTH
        if len(value) > limit:
            raise ValueError("replay metadata string exceeds its limit")
    elif isinstance(value, dict):
        encoded_frame = set(value) == {
            "encoding", "dtype", "shape", "element_count", "raw_bytes", "data"
        }
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > MAX_METADATA_STRING_LENGTH:
                raise ValueError("replay object key is invalid or oversized")
            _validate_metadata(
                item,
                depth=depth + 1,
                frame_data=frame_data or (encoded_frame and key == "data"),
            )
    elif isinstance(value, list):
        for item in value:
            _validate_metadata(item, depth=depth + 1, frame_data=frame_data)
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise ValueError("replay JSON contains an unsupported value type")


def load_explicit_json_observations(path: str | Path) -> LoadedReplay:
    """Load one explicitly named, bounded fixture without trusting its path."""
    supplied = Path(path)
    try:
        supplied_lstat = supplied.lstat()
    except OSError as exc:
        raise ValueError("explicit replay path is not accessible") from exc
    is_symlink = stat.S_ISLNK(supplied_lstat.st_mode)
    try:
        resolved = supplied.resolve(strict=True)
        resolved_stat = resolved.stat()
    except OSError as exc:
        raise ValueError("explicit replay symlink target is not accessible") from exc
    if not stat.S_ISREG(resolved_stat.st_mode):
        raise ValueError("explicit replay path must resolve to a regular file")
    file_bytes = int(resolved_stat.st_size)
    if file_bytes <= 0:
        raise ValueError("explicit replay file must not be empty")
    if file_bytes > MAX_REPLAY_JSON_BYTES:
        raise ValueError("explicit replay file exceeds the byte limit")
    with resolved.open("rb") as stream:
        raw = stream.read(MAX_REPLAY_JSON_BYTES + 1)
    if len(raw) > MAX_REPLAY_JSON_BYTES or len(raw) != file_bytes:
        raise ValueError("explicit replay file changed or exceeded the bounded read")
    text = _scan_json_before_parse(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("explicit replay JSON is malformed") from exc
    if not isinstance(payload, dict) or payload.get("schema") != REPLAY_SCHEMA:
        raise ValueError("unsupported explicit replay schema")
    records = payload.get("observations")
    if not isinstance(records, list) or not records:
        raise ValueError("explicit replay must contain observations")
    if len(records) > MAX_REPLAY_RECORDS:
        raise ValueError("explicit replay contains too many records")
    _validate_metadata(payload)
    decoded = tuple(_decode_record(source_record) for source_record in records)
    return LoadedReplay(
        observations=decoded,
        supplied_path=str(supplied),
        resolved_path=str(resolved),
        supplied_path_was_symlink=is_symlink,
        file_bytes=file_bytes,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _decode_record(source_record: object) -> dict:
    if not isinstance(source_record, dict):
        raise ValueError("replay observation records must be dictionaries")
    record = dict(source_record)
    record["joints"] = _decode_finite_vector(record.get("joints"), 7, "joints")
    record["tcp_pose"] = _decode_finite_vector(record.get("tcp_pose"), 6, "tcp_pose")
    encoded_total = 0
    encoded_values: list[object] = []
    for stream in ("images", "depth"):
        source = record.get(stream)
        if source is None:
            continue
        if not isinstance(source, dict):
            raise ValueError(f"{stream} must be a camera dictionary")
        encoded_values.extend(source.values())
    for value in encoded_values:
        if not isinstance(value, dict) or not isinstance(value.get("data"), str):
            raise ValueError("replay frame data must be a base64 string")
        encoded_total += len(value["data"])
        if encoded_total > MAX_ENCODED_FRAME_BYTES_PER_RECORD:
            raise ValueError("replay record exceeds the encoded-frame byte limit")
    for stream in ("images", "depth"):
        source = record.get(stream)
        if source is None:
            continue
        decoded: dict[str, np.ndarray] = {}
        for camera, value in source.items():
            array, _ = _decode_encoded_frame(value, stream, camera)
            decoded[camera] = array
        record[stream] = decoded
    return record


def _decode_finite_vector(value: object, length: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a JSON list of length {length}")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} must contain only numeric non-boolean values")
    if not all(math.isfinite(float(item)) for item in value):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(value, dtype=np.float64)


def _decode_encoded_frame(value: object, stream: str, camera: str) -> tuple[np.ndarray, int]:
    contract = FRAME_CONTRACTS.get((stream, camera))
    if contract is None:
        raise ValueError(f"{stream}.{camera} is not a supported native stream")
    dtype_name, expected_shape = contract
    fields = {"encoding", "dtype", "shape", "element_count", "raw_bytes", "data"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{stream}.{camera} must be an exact encoded-frame record")
    if value["encoding"] != FRAME_ENCODING or value["dtype"] != dtype_name:
        raise ValueError(f"{stream}.{camera} encoding or dtype is unsupported")
    if value["shape"] != list(expected_shape):
        raise ValueError(f"{stream}.{camera} shape does not match the native stream")
    expected_elements = math.prod(expected_shape)
    item_size = np.dtype(dtype_name).itemsize
    expected_raw_bytes = expected_elements * item_size
    if (
        isinstance(value["element_count"], bool)
        or value["element_count"] != expected_elements
        or isinstance(value["raw_bytes"], bool)
        or value["raw_bytes"] != expected_raw_bytes
    ):
        raise ValueError(f"{stream}.{camera} declared size does not match its native contract")
    encoded = value["data"]
    if not isinstance(encoded, str) or not encoded.isascii():
        raise ValueError(f"{stream}.{camera} data must be ASCII base64")
    expected_encoded_bytes = 4 * ((expected_raw_bytes + 2) // 3)
    if len(encoded) != expected_encoded_bytes:
        raise ValueError(f"{stream}.{camera} encoded length does not match its native contract")
    if len(encoded) > MAX_ENCODED_FRAME_BYTES_PER_RECORD:
        raise ValueError(f"{stream}.{camera} encoded data exceeds the per-record limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{stream}.{camera} contains malformed base64") from exc
    if len(raw) != expected_raw_bytes:
        raise ValueError(f"{stream}.{camera} decoded length does not match its native contract")
    array = np.frombuffer(raw, dtype=np.dtype(dtype_name)).reshape(expected_shape).copy()
    return array, len(encoded)


def encode_frame_fixture(array: np.ndarray, *, stream: str, camera: str) -> dict:
    """Encode a native test/recording frame into the bounded replay format."""
    contract = FRAME_CONTRACTS.get((stream, camera))
    if contract is None:
        raise ValueError("unsupported native replay stream")
    dtype_name, shape = contract
    if array.dtype != np.dtype(dtype_name) or array.shape != shape:
        raise ValueError("fixture frame does not match its native contract")
    raw = array.tobytes(order="C")
    return {
        "encoding": FRAME_ENCODING,
        "dtype": dtype_name,
        "shape": list(shape),
        "element_count": array.size,
        "raw_bytes": len(raw),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def build_verification_envelope(
    *,
    trial_sha256: str,
    source_observation_sha256: str,
    report_kind: str,
    payload: dict,
    parent_session_sha256: str,
    selected_anonymous_detection_sha256: str | None = None,
) -> dict:
    body = {
        "schema_version": EVIDENCE_SCHEMA,
        "report_kind": report_kind,
        "trial_sha256": trial_sha256,
        "source_observation_sha256": source_observation_sha256,
        "payload": payload,
        "payload_sha256": _canonical_sha256(payload),
        "parent_session_sha256": parent_session_sha256,
        "selected_anonymous_detection_sha256": selected_anonymous_detection_sha256,
    }
    body["envelope_sha256"] = _canonical_sha256(body)
    return body


class VerificationEvidenceVerifier:
    """Verify evidence against one active reset/trial and reject reuse."""

    def __init__(self, active_trial_sha256: str) -> None:
        if not _is_sha256(active_trial_sha256):
            raise ValueError("active trial hash is invalid")
        self.active_trial_sha256 = active_trial_sha256
        self._used_envelopes: set[str] = set()

    def verify(
        self,
        envelope: dict,
        *,
        report_kind: str,
        source_observation_sha256: str,
        parent_session_sha256: str,
        allowed_detection_sha256: set[str] | None = None,
    ) -> dict:
        expected_fields = {
            "schema_version", "report_kind", "trial_sha256",
            "source_observation_sha256", "payload", "payload_sha256",
            "parent_session_sha256", "selected_anonymous_detection_sha256",
            "envelope_sha256",
        }
        if not isinstance(envelope, dict) or set(envelope) != expected_fields:
            raise ValueError("verification evidence envelope fields are invalid")
        if envelope["schema_version"] != EVIDENCE_SCHEMA or envelope["report_kind"] != report_kind:
            raise ValueError("verification evidence schema or report kind is unsupported")
        if envelope["trial_sha256"] != self.active_trial_sha256:
            raise ValueError("verification evidence belongs to another trial")
        if (
            not _is_sha256(source_observation_sha256)
            or envelope["source_observation_sha256"] != source_observation_sha256
        ):
            raise ValueError("verification evidence references the wrong source observation")
        if envelope["parent_session_sha256"] != parent_session_sha256:
            raise ValueError("verification evidence parent session hash is stale")
        payload = envelope["payload"]
        if not isinstance(payload, dict) or envelope["payload_sha256"] != _canonical_sha256(payload):
            raise ValueError("verification evidence payload hash is invalid")
        body = dict(envelope)
        supplied_hash = body.pop("envelope_sha256")
        if not _is_sha256(supplied_hash) or supplied_hash != _canonical_sha256(body):
            raise ValueError("verification evidence canonical envelope hash is invalid")
        if supplied_hash in self._used_envelopes:
            raise ValueError("verification evidence envelope was replayed twice")
        selected = envelope["selected_anonymous_detection_sha256"]
        if report_kind == "table":
            if selected is not None:
                raise ValueError("table evidence cannot select a wrist detection")
        else:
            if not _is_sha256(selected) or allowed_detection_sha256 is None or selected not in allowed_detection_sha256:
                raise ValueError("wrist evidence references an unknown anonymous detection")
            if payload.get("selected_anonymous_detection_sha256") != selected:
                raise ValueError("wrist payload and envelope select different detections")
        self._used_envelopes.add(supplied_hash)
        return dict(payload)


def replay_observation_review(
    *,
    trial: object,
    table_observation: dict,
    wrist_observations: Iterable[dict],
    table_verification_envelope: dict,
    wrist_verification_envelope: dict,
    table_reference_time: float,
    wrist_reference_times: Iterable[float],
) -> dict:
    """Replay one explicit trial through the Phase 5C review state machine."""
    adapter = OrganizerObservationAdapter()
    session = ObservationOnlySession()
    adapter.reset(trial)
    session.begin()
    try:
        table = adapter.adapt(
            table_observation,
            required_rgb=("table",),
            requested_depth=("table",) if "depth" in table_observation else (),
            reference_time=table_reference_time,
        )
        evidence = VerificationEvidenceVerifier(table.native.trial_sha256)
        session.record_table_observation(table.native.observation_sha256)
        table_parent = session.review_record()["event_chain_head"]
        table_payload = evidence.verify(
            table_verification_envelope,
            report_kind="table",
            source_observation_sha256=table.native.observation_sha256,
            parent_session_sha256=table_parent,
        )
        detections = table_payload.get("anonymous_detection_sha256")
        if (
            not isinstance(detections, list)
            or not detections
            or len(detections) != len(set(detections))
            or any(not _is_sha256(item) for item in detections)
        ):
            raise ValueError("table evidence must list unique anonymous detection hashes")
        session.record_table_verification(table_payload)

        wrist_records = list(wrist_observations)
        reference_times = list(wrist_reference_times)
        if len(wrist_records) != len(reference_times):
            raise ValueError("each wrist observation requires an explicit reference time")
        adapted_wrist = [
            adapter.adapt(
                record,
                required_rgb=("wrist",),
                requested_depth=("wrist",) if "depth" in record else (),
                reference_time=reference,
            )
            for record, reference in zip(wrist_records, reference_times)
        ]
        wrist_hashes = [item.native.observation_sha256 for item in adapted_wrist]
        session.record_wrist_observations(wrist_hashes)
        wrist_parent = session.review_record()["event_chain_head"]
        wrist_source = wrist_verification_envelope.get("source_observation_sha256", "")
        if wrist_source not in wrist_hashes:
            raise ValueError("wrist evidence source is not part of the current trial")
        wrist_payload = evidence.verify(
            wrist_verification_envelope,
            report_kind="wrist",
            source_observation_sha256=wrist_source,
            parent_session_sha256=wrist_parent,
            allowed_detection_sha256=set(detections),
        )
        session.record_wrist_verification(wrist_payload)
    except (RuntimeError, TypeError, ValueError) as exc:
        session.abort(f"replay_rejected:{exc}")
        raise

    review = session.review_record()
    record = {
        "schema": "phase5d_observation_replay_v1",
        "table": table.review_metadata(),
        "wrist": [item.review_metadata() for item in adapted_wrist],
        "session": review,
        "action": None,
        "robot_motion_authorized": False,
    }
    record["replay_sha256"] = _canonical_sha256(record)
    return record
