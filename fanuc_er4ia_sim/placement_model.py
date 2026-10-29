"""Host-only placement model for externally surveyed coordinates.

No deployment coordinates live here.  The validator is intentionally stricter
than the organizer's local parser by requiring a complete task inventory in
every rollout.
"""

from __future__ import annotations

from collections import Counter
import math

from bin_layout import AxisAlignedBounds, KeepOut, validate_bin_layout
from real_task_config import RealTaskConfig


CURRENT_ROLLOUT_COUNT = 10
PLACEHOLDER_EXAMPLE = {
    "deployable": False,
    "reason": "coordinates and fixed yaw must be supplied from verified CSI survey evidence",
    "rollouts": None,
}


def build_placement_model(rollouts: list[dict], **validation_context: object) -> dict:
    """Build only from caller-supplied coordinates, and return only if validated.

    The result is an in-memory model, not a final organizer submission file.
    All surveyed geometry and task configuration must be supplied through the
    same explicit context required by :func:`validate_placement_model`.
    """
    document = {"rollouts": rollouts}
    validation = validate_placement_model(document, **validation_context)
    return {"deployable": False, "document": document, "validation": validation}


def _instances(value: object) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ValueError("placement object value must be a nonempty list")
    if value and isinstance(value[0], (int, float)):
        value = [value]
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) not in (2, 3):
            raise ValueError("each placement must be [x, y] or [x, y, yaw]")
        if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in row):
            raise ValueError("placement coordinates must be finite numbers")
        rows.append([float(v) for v in row])
    return rows


def _radius(name: str) -> float:
    if name.endswith("cube"):
        return 10.0 / math.sqrt(2.0)
    if name == "cap":
        return 31.75 / 2.0
    if name.startswith("bin_"):
        return 25.0
    if name == "vial":
        return 203.0 / 2.0 + 27.18 / 2.0
    raise ValueError(f"unknown placement identifier: {name}")


def validate_placement_model(
    document: dict,
    *,
    config: RealTaskConfig,
    objects_mode: str,
    layout_mode: str,
    source_bounds: AxisAlignedBounds,
    table_bounds: AxisAlignedBounds,
    keep_outs: tuple[KeepOut, ...],
    bin_visibility: dict[str, bool] | None,
    maximum_reach_mm: float,
    reach_margin_mm: float,
) -> dict:
    config.validate()
    if config.requested_cube_yaw_deg is None:
        raise ValueError("fixed cube yaw remains unresolved")
    if objects_mode not in {"same", "different"} or layout_mode not in {"same", "different"}:
        raise ValueError("objects/layout mode must be same or different")
    if not isinstance(document, dict) or set(document) != {"rollouts"}:
        raise ValueError("placement document must contain only top-level rollouts")
    rollouts = document["rollouts"]
    expected_rollouts = 1 if objects_mode == layout_mode == "same" else CURRENT_ROLLOUT_COUNT
    if not isinstance(rollouts, list) or len(rollouts) != expected_rollouts:
        raise ValueError(f"placement requires exactly {expected_rollouts} rollout entries")
    identifiers = config.identifiers
    expected_counts = {
        identifiers["red"]: 2,
        identifiers["blue"]: 2,
        identifiers["yellow"]: 2,
        identifiers["vial"]: 2,
        identifiers["cap"]: 2,
        identifiers["bin_1"]: 1,
        identifiers["bin_2"]: 1,
        identifiers["bin_3"]: 1,
    }
    summaries = []
    for index, rollout in enumerate(rollouts):
        if not isinstance(rollout, dict) or not isinstance(rollout.get("objects"), dict):
            raise ValueError(f"rollout {index} requires an objects mapping")
        if set(rollout) - {"objects", "prompt"}:
            raise ValueError(f"rollout {index} contains unsupported fields")
        parsed = {name: _instances(value) for name, value in rollout["objects"].items()}
        counts = Counter({name: len(rows) for name, rows in parsed.items()})
        if counts != Counter(expected_counts):
            raise ValueError(f"rollout {index} inventory does not match the complete task")
        cube_names = {identifiers[colour] for colour in ("red", "blue", "yellow")}
        positions: list[tuple[str, float, float, float]] = []
        for name, rows in parsed.items():
            for row in rows:
                x, y = row[:2]
                yaw = row[2] if len(row) == 3 else 0.0
                if name in cube_names:
                    if len(row) != 3 or abs((yaw - config.requested_cube_yaw_deg + 180) % 360 - 180) > 1e-9:
                        raise ValueError(f"rollout {index} cube yaw is not the fixed public yaw")
                radius = _radius(name)
                bounds = table_bounds if name.startswith("bin_") else source_bounds
                if not bounds.contains_box((x, y), radius):
                    raise ValueError(f"rollout {index} {name} footprint outside surveyed bounds")
                if math.hypot(x, y) > maximum_reach_mm - reach_margin_mm:
                    raise ValueError(f"rollout {index} {name} lacks conservative reach margin")
                positions.append((name, x, y, radius))
        for left in range(len(positions)):
            for right in range(left + 1, len(positions)):
                a, b = positions[left], positions[right]
                if math.dist(a[1:3], b[1:3]) <= a[3] + b[3]:
                    raise ValueError(f"rollout {index} footprints overlap: {a[0]} and {b[0]}")
        bin_centres = {
            key: tuple(parsed[identifiers[key]][0][:2]) for key in ("bin_1", "bin_2", "bin_3")
        }
        bin_result = validate_bin_layout(
            bin_centres,
            table_bounds=table_bounds,
            keep_outs=keep_outs,
            camera_visible=bin_visibility,
            maximum_reach_mm=maximum_reach_mm,
            reach_margin_mm=reach_margin_mm,
            minimum_center_spacing_mm=config.thresholds.minimum_bin_center_spacing_mm,
        )
        if not bin_result["valid"]:
            raise ValueError(f"rollout {index} bin layout invalid: {bin_result['reasons']}")
        summaries.append({"rollout": index, "object_counts": dict(counts), "bin_layout": bin_result})
    return {
        "valid": True,
        "rollout_count": len(rollouts),
        "complete_inventory_required_by": "team_fail_closed_validator",
        "organizer_requires_each_object_to_move": False,
        "summaries": summaries,
    }
