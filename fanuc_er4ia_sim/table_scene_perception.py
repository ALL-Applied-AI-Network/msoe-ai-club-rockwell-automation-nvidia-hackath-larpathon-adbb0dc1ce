"""Fail-closed adapter boundary for native table-RGB cube perception.

The existing frozen detector is not imported here because its CameraModel is a
simulation calibration identity and its image convention is BGR.  A future CSI
adapter may inject that frozen behavior only after explicitly converting native
RGB and supplying a verified real-cell camera/workspace implementation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Callable

from camera_calibration_contract import CameraCalibration, validate_camera_calibration
from detector_adapter_contract import DetectorAdapterContract, verify_detector_adapter
from native_observation import NativeObservation


FORBIDDEN_DETECTION_FIELDS = {
    "name",
    "object_name",
    "object_id",
    "prim",
    "prim_name",
    "prim_path",
    "truth",
    "ground_truth",
    "scene_state",
    "oracle",
    "oracle_plan",
    "placement",
    "target_slot",
}


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SourceWorkspaceContract:
    frame_name: str
    polygon_xy_m: tuple[tuple[float, float], ...]
    calibration_compatibility_id: str
    survey_version: str
    public: bool
    verified: bool

    def payload(self) -> dict:
        return asdict(self)

    def integrity_sha256(self) -> str:
        return _hash(self.payload())


def validate_source_workspace(
    workspace: SourceWorkspaceContract | None,
    *,
    calibration: CameraCalibration,
) -> SourceWorkspaceContract:
    if workspace is None or not workspace.verified or not workspace.public:
        raise ValueError("a public verified source workspace is required")
    if workspace.frame_name != calibration.frame_name:
        raise ValueError("workspace/camera frame mismatch")
    if workspace.calibration_compatibility_id != calibration.compatibility_id:
        raise ValueError("workspace/camera compatibility mismatch")
    if not workspace.survey_version or len(workspace.polygon_xy_m) < 3:
        raise ValueError("workspace survey is incomplete")
    for point in workspace.polygon_xy_m:
        if len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
            raise ValueError("workspace polygon contains an invalid point")
    return workspace


def _anonymous_rows(detector_result: dict, frame_hash: str) -> list[dict]:
    if not isinstance(detector_result, dict):
        raise ValueError("detector result must be a dictionary")
    for flag in ("ground_truth_used", "scene_state_used", "oracle_plan_used"):
        if detector_result.get(flag) is not False:
            raise ValueError(f"detector must explicitly declare {flag}=false")
    source_rows = detector_result.get("accepted_detections")
    if not isinstance(source_rows, list):
        raise ValueError("detector result lacks accepted_detections")
    rows: list[dict] = []
    for index, source in enumerate(source_rows):
        if not isinstance(source, dict):
            raise ValueError("detector row must be a dictionary")
        forbidden = sorted(FORBIDDEN_DETECTION_FIELDS & set(source))
        if forbidden:
            raise ValueError(f"detector row contains forbidden identity/truth fields: {forbidden}")
        if source.get("category", "cube") != "cube":
            raise ValueError("accepted table detections must be cubes")
        colour = source.get("colour", source.get("color"))
        if colour not in {"red", "blue", "yellow"}:
            raise ValueError("detector row has an unsupported colour")
        if "xy_mm" in source:
            xy_mm = tuple(float(value) for value in source["xy_mm"])
        elif "xy_m" in source:
            xy_mm = tuple(float(value) * 1000.0 for value in source["xy_m"])
        else:
            raise ValueError("detector row lacks calibrated XY")
        if len(xy_mm) != 2 or not all(math.isfinite(value) for value in xy_mm):
            raise ValueError("detector row XY is invalid")
        confidence = float(source.get("confidence", source.get("candidate_confidence", math.nan)))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("detector row confidence is invalid")
        anonymous_id = "anon-" + hashlib.sha256(
            f"{frame_hash}:{index}:{colour}:{xy_mm[0]:.9f}:{xy_mm[1]:.9f}".encode("utf-8")
        ).hexdigest()[:16]
        rows.append(
            {
                "detection_id": anonymous_id,
                "category": "cube",
                "colour": colour,
                "xy_mm": [xy_mm[0], xy_mm[1]],
                "confidence": confidence,
                "accepted": True,
            }
        )
    return rows


def perceive_table_scene(
    observation: NativeObservation,
    *,
    calibration: CameraCalibration | None,
    workspace: SourceWorkspaceContract | None,
    detector: Callable[[object, CameraCalibration, SourceWorkspaceContract], dict],
    adapter_contract: DetectorAdapterContract,
    adapter_configuration: dict,
) -> dict:
    if "table" not in observation.rgb:
        raise ValueError("native table RGB is required")
    table_calibration = validate_camera_calibration(calibration, expected_stream="table_rgb")
    source_workspace = validate_source_workspace(workspace, calibration=table_calibration)
    adapter_provenance = verify_detector_adapter(
        detector,
        adapter_contract,
        adapter_configuration=adapter_configuration,
        calibration_compatibility_id=table_calibration.compatibility_id,
        workspace_version=source_workspace.survey_version,
    )
    frame = observation.rgb["table"]
    frame_record = next(
        record for record in observation.provenance if record.camera == "table" and record.stream == "rgb"
    )
    result = detector(frame, table_calibration, source_workspace)
    rows = _anonymous_rows(result, frame_record.content_sha256)
    counts = Counter(row["colour"] for row in rows)
    reasons: list[str] = []
    if len(rows) != 6:
        reasons.append("expected_exactly_six_cubes")
    if counts != Counter({"red": 2, "blue": 2, "yellow": 2}):
        reasons.append("expected_exactly_two_cubes_per_colour")
    positions = [tuple(row["xy_mm"]) for row in rows]
    if len(positions) != len(set(positions)):
        reasons.append("duplicate_cube_positions")
    record = {
        "passed": not reasons,
        "reasons": reasons,
        "observation_sha256": observation.observation_sha256,
        "table_frame_sha256": frame_record.content_sha256,
        "calibration_sha256": table_calibration.integrity_sha256(),
        "workspace_sha256": source_workspace.integrity_sha256(),
        "adapter_contract_sha256": adapter_provenance["contract_sha256"],
        "adapter_name": adapter_provenance["adapter_name"],
        "adapter_semantic_version": adapter_provenance["adapter_semantic_version"],
        "adapter_callable_module": adapter_provenance["module_name"],
        "adapter_callable_qualified_name": adapter_provenance["qualified_name"],
        "adapter_module_file_sha256": adapter_provenance["module_file_sha256"],
        "adapter_configuration_sha256": adapter_provenance["adapter_configuration_sha256"],
        "adapter_color_conversion_convention": adapter_provenance["color_conversion_convention"],
        "adapter_output_schema_version": adapter_provenance["output_schema_version"],
        "adapter_verification_evidence_sha256": adapter_provenance["verification_evidence_sha256"],
        "detections": rows,
        "counts_by_colour": {colour: counts.get(colour, 0) for colour in ("red", "blue", "yellow")},
        "anonymous_detections_only": True,
        "real_world_performance_validated": False,
        "truth_used": False,
        "scene_state_used": False,
        "oracle_used": False,
        "action": None,
        "robot_motion_authorized": False,
    }
    record["record_sha256"] = _hash(record)
    return record
