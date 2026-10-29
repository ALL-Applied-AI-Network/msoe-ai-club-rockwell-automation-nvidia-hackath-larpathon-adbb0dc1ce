"""Public Phase 5B task facts, physical specifications, and ownership boundaries.

This module contains no surveyed deployment coordinates and authorizes no motion.
It deliberately separates organizer facts, team choices, CSI measurements, and
values that remain unresolved.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math


@dataclass(frozen=True)
class PhysicalObjectSpec:
    identifier: str
    dimensions_mm: dict[str, float]
    source: str
    notes: str = ""


@dataclass(frozen=True)
class DeploymentThresholds:
    maximum_combined_yaw_error_deg: float = 3.0
    maximum_yaw_uncertainty_deg: float = 1.0
    maximum_table_to_wrist_xy_disagreement_mm: float = 1.0
    maximum_closing_axis_uncertainty_mm: float = 0.25
    minimum_depth_valid_fraction: float = 0.90
    maximum_cube_top_height_error_mm: float = 2.0
    maximum_table_plane_residual_mm: float = 1.0
    minimum_bin_center_spacing_mm: float = 80.0
    minimum_release_clearance_above_rim_mm: float = 15.0


@dataclass(frozen=True)
class RealTaskConfig:
    requested_cube_yaw_deg: float | None = None
    identifiers: dict[str, str] = field(
        default_factory=lambda: {
            "red": "red_cube",
            "blue": "blue_cube",
            "yellow": "yellow_cube",
            "vial": "vial",
            "cap": "cap",
            "bin_1": "bin_1",
            "bin_2": "bin_2",
            "bin_3": "bin_3",
        }
    )
    cube_counts: dict[str, int] = field(
        default_factory=lambda: {"red": 2, "blue": 2, "yellow": 2}
    )
    destination_by_colour: dict[str, str] = field(
        default_factory=lambda: {"red": "bin_1", "blue": "bin_2", "yellow": "bin_3"}
    )
    distractor_counts: dict[str, int] = field(
        default_factory=lambda: {"vial": 2, "cap": 2}
    )
    available_bin_count: int = 5
    used_bin_count: int = 3
    thresholds: DeploymentThresholds = field(default_factory=DeploymentThresholds)

    def validate(self) -> None:
        if self.requested_cube_yaw_deg is not None and not math.isfinite(
            float(self.requested_cube_yaw_deg)
        ):
            raise ValueError("requested cube yaw must be finite or unresolved")
        if self.cube_counts != {"red": 2, "blue": 2, "yellow": 2}:
            raise ValueError("task requires exactly two cubes of each configured colour")
        if self.distractor_counts != {"vial": 2, "cap": 2}:
            raise ValueError("task requires exactly two vial and two cap distractors")
        if self.used_bin_count != 3 or self.available_bin_count < self.used_bin_count:
            raise ValueError("task uses exactly three of at least five available bins")

    def to_dict(self) -> dict:
        return asdict(self)


ORGANIZER_CONFIRMED = {
    "task_definition": "participant_defined",
    "placement_names": "free_text",
    "multiple_instances": "lists_of_x_y_optional_yaw",
    "cube_inventory": {"red": 2, "blue": 2, "yellow": 2},
    "vial_availability": "more_than_30",
    "cap_availability": "more_than_30",
    "available_bins": 5,
    "bin_outer_mm": [50.0, 50.0, 55.0],
    "bin_pocket_mm": [35.0, 35.0, 25.0],
    "fixed_identical_cube_yaw_permitted": True,
    "wrist_observation_motion_permitted": True,
    "table_rgb_maximum": [1920, 1080],
    "rgb_cameras": ["table", "side", "wrist"],
    "depth_cameras": ["table", "wrist"],
    "gripper_model": "SMC MHZL2-16D2",
    "jaw_travel_per_side_mm": 6.0,
    "straight_face_opening_mm": 12.0,
    "gripper_control": "binary_open_closed",
    "tcp_actions": "absolute",
}

TEAM_SELECTED = {
    "task": "six_cube_colour_sort",
    "destination_by_colour": {"red": "bin_1", "blue": "bin_2", "yellow": "bin_3"},
    "used_bins": ["bin_1", "bin_2", "bin_3"],
    "distractors": {"vial": 2, "cap": 2},
    "cube_yaw_policy": "one_fixed_public_yaw",
    "layout_setup_preference": "onsite_team_member",
}

CSI_MEASURED_DEFAULTS = {
    "table_calibration": None,
    "wrist_rgb_calibration": None,
    "wrist_depth_calibration": None,
    "table_plane": None,
    "source_workspace": None,
    "bin_centres": None,
    "tcp_uframe": None,
    "tcp_utool": None,
    "tcp_r_sign": None,
    "gripper_frame_offset_deg": None,
    "open_gap_mm": None,
    "finger_parallelism_tolerance_mm": None,
    "finger_installation_tolerance_mm": None,
    "wrist_xy_repeatability_mm": None,
    "wrist_yaw_repeatability_deg": None,
    "bin_release_height_mm": None,
}

UNRESOLVED_VALUES = (
    "requested_cube_yaw_deg",
    "surveyed_bin_coordinates",
    "tcp_r_sign_and_gripper_frame_offset",
    "real_finger_tolerance_budget",
    "server_side_placement_semantics_not_checked_by_local_client",
)

PHYSICAL_OBJECTS = {
    "cube": PhysicalObjectSpec(
        "cube", {"edge": 10.0}, "organizer_confirmed", "Rounded edges receive no clearance credit."
    ),
    "vial": PhysicalObjectSpec(
        "vial", {"diameter": 27.18, "length": 203.0}, "organizer_confirmed"
    ),
    "cap": PhysicalObjectSpec(
        "cap",
        {
            "flange_diameter": 31.75,
            "height": 15.70,
            "flange_thickness": 0.088 * 25.4,
            "plug_base_diameter": 0.930 * 25.4,
            "plug_tip_diameter": 0.820 * 25.4,
        },
        "existing_cap_geometry_confirmed_correct",
    ),
    "bin": PhysicalObjectSpec(
        "bin",
        {
            "outer_x": 50.0,
            "outer_y": 50.0,
            "height": 55.0,
            "pocket_x": 35.0,
            "pocket_y": 35.0,
            "pocket_depth": 25.0,
        },
        "organizer_confirmed",
    ),
}

SIMULATION_MISMATCH = {
    "current_vial_mm": {"diameter": 17.0, "length": 140.0},
    "real_vial_mm": {"diameter": 27.18, "length": 203.0},
    "simulation_change_deferred": True,
    "future_migration": [
        "replace visual scale with the physical envelope",
        "replace collision geometry with a surveyed body/end model",
        "update mass and inertia",
        "update segment-distance exclusion and camera-visibility checks",
        "revalidate transit height and distractor monitoring",
    ],
}


def validate_task_inventory(observations: list[dict]) -> dict:
    """Authorize only six unique cube observations with an exact 2/2/2 colour split."""
    cube_rows = [row for row in observations if row.get("category") == "cube"]
    reasons: list[str] = []
    if len(cube_rows) != 6:
        reasons.append("expected_exactly_six_cubes")
    ids = [row.get("detection_id") for row in cube_rows]
    if None in ids or len(ids) != len(set(ids)):
        reasons.append("cube_detection_ids_not_unique")
    counts = Counter(row.get("colour") for row in cube_rows)
    if counts != Counter({"red": 2, "blue": 2, "yellow": 2}):
        reasons.append("expected_exactly_two_cubes_per_colour")
    if any(row.get("accepted") is not True for row in cube_rows):
        reasons.append("one_or_more_cubes_not_accepted")
    return {
        "authorized": not reasons,
        "reasons": reasons,
        "cube_detection_ids": ids,
        "cube_colour_counts": dict(counts),
        "ignored_distractor_count": sum(
            row.get("category") in {"cap", "vial", "bin"} for row in observations
        ),
    }


PRODUCTION_TASK_CONFIG = RealTaskConfig()
PRODUCTION_TASK_CONFIG.validate()
