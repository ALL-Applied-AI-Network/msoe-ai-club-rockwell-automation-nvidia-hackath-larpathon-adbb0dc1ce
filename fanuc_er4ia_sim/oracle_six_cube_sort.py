"""Plan and validate the isolated six-cube oracle defect-sort physics task."""

import argparse
import json
from pathlib import Path

import numpy as np

import scene_config as C
from oracle_motion_smoke import (
    EXECUTION_POSE_ROTATION_TOLERANCE_DEG,
    EXECUTION_POSE_TRANSLATION_TOLERANCE_M,
    NON_SELECTED_CUBE_ROTATION_LIMIT_DEG,
    NON_SELECTED_CUBE_TRANSLATION_LIMIT_M,
    ORACLE_SOURCE,
    TRANSIT_CLEARANCE_M,
    _rotation_delta_deg,
    _yaw_deg,
    expected_scene_object_names,
    preflight_oracle_picks,
)


ORACLE_SIX_TASK = "oracle_six_cube_defect_sort"
EXECUTION_ORDER = (
    ("SortCube2", "red", 0),
    ("SortCube3", "red", 1),
    ("SortCube4", "blue", 0),
    ("SortCube5", "blue", 1),
    ("SortCube0", "yellow", 0),
    ("SortCube1", "yellow", 1),
)
EXPECTED_CUBE_COLOURS = {name: colour for name, colour, _slot in EXECUTION_ORDER}
EXECUTION_ORDER_REASON = (
    "Fill the audited target row monotonically from left to right; each vertical "
    "approach remains separated from earlier slots and every carry stays at "
    "130 mm clearance."
)
FLAT_HEIGHT_TOLERANCE_M = 0.0015


def _pose_position(pose):
    return np.asarray(pose.get("centre", pose.get("position", [])), float)


def _pose_error(planned, current):
    planned_position = _pose_position(planned)
    current_position = _pose_position(current)
    translation = None
    if planned_position.shape == (3,) and current_position.shape == (3,):
        translation = float(np.linalg.norm(current_position - planned_position))
    rotation = None
    planned_q = np.asarray(planned.get("orientation", []), float)
    current_q = np.asarray(current.get("orientation", []), float)
    if planned_q.shape == (4,) and current_q.shape == (4,):
        planned_norm = np.linalg.norm(planned_q)
        current_norm = np.linalg.norm(current_q)
        if planned_norm > 0 and current_norm > 0:
            similarity = abs(np.dot(planned_q, current_q) / (planned_norm * current_norm))
            rotation = float(np.degrees(2 * np.arccos(np.clip(similarity, 0, 1))))
    return translation, rotation


def _cube_contained(xy, colour, yaw_deg=0.0):
    theta = np.deg2rad(float(yaw_deg))
    half_extent = 0.5 * C.CUBE_EDGE * (abs(np.cos(theta)) + abs(np.sin(theta)))
    delta = np.abs(np.asarray(xy, float) - C.DEFECT_ZONE_CENTERS[colour])
    available = 0.5 * np.asarray(C.DEFECT_ZONE_SIZE) - C.DEFECT_ZONE_SAFE_INSET
    return bool(np.all(delta + half_extent <= available + 1e-12))


def _placement_result(pick, pose):
    centre = _pose_position(pose)
    slot = np.asarray(pick["assigned_slot"], float)
    slot_error = (
        float(np.linalg.norm(centre[:2] - slot))
        if centre.shape == (3,)
        else float("inf")
    )
    expected_height = C.TABLE_TOP_Z + C.SURFACE_DZ + 0.5 * C.CUBE_EDGE
    height_error = (
        float(abs(centre[2] - expected_height))
        if centre.shape == (3,)
        else float("inf")
    )
    return {
        "assigned_slot_error_mm": slot_error * 1000,
        "height_error_mm": height_error * 1000,
        "correct_zone_containment": centre.shape == (3,)
        and _cube_contained(centre[:2], pick["colour"], pose.get("yaw_deg", 0.0)),
        "placement_passed": bool(
            slot_error <= C.DEFECT_SLOT_TOLERANCE + 1e-12
            and height_error <= FLAT_HEIGHT_TOLERANCE_M
            and centre.shape == (3,)
            and _cube_contained(
                centre[:2], pick["colour"], pose.get("yaw_deg", 0.0)
            )
        ),
    }


def build_oracle_six_cube_plan(scene_state, ground_truth):
    """Build the all-or-nothing six-cube plan from captured simulator state."""
    objects = scene_state.get("objects", {})
    expected_names = expected_scene_object_names()
    if set(objects) != expected_names:
        raise ValueError("Oracle six-cube scene must contain exactly ten objects")
    truth_rows = ground_truth.get("objects", [])
    truth = {row.get("name"): row for row in truth_rows}
    if set(truth) != expected_names:
        raise ValueError("Oracle six-cube ground truth must contain exactly ten objects")
    counts = {
        category: sum(row.get("category") == category for row in truth_rows)
        for category in ("cube", "cap", "vial")
    }
    if counts != {"cube": 6, "cap": 2, "vial": 2}:
        raise ValueError(f"Wrong oracle inventory: {counts}")
    cube_colours = [truth[name].get("colour") for name in EXPECTED_CUBE_COLOURS]
    if {colour: cube_colours.count(colour) for colour in C.DEFECT_CLASS_BY_COLOUR} != C.DEFECT_EXPECTED_COUNTS:
        raise ValueError("Oracle inventory must contain two cubes of every colour")

    planned_scene_poses = {}
    for name in sorted(expected_names):
        saved = objects[name]
        position = np.asarray(saved.get("position", []), float)
        orientation = np.asarray(saved.get("orientation", []), float)
        if position.shape != (3,) or orientation.shape != (4,):
            raise ValueError(f"Invalid simulator pose for {name}")
        planned_scene_poses[name] = {
            "position": position.tolist(),
            "orientation": orientation.tolist(),
        }

    picks = []
    for execution_index, (name, colour, slot_index) in enumerate(EXECUTION_ORDER):
        row = truth[name]
        if row.get("category") != "cube" or row.get("colour") != colour:
            raise ValueError(f"{name} must be the configured {colour} cube")
        pose = planned_scene_poses[name]
        yaw = _yaw_deg(pose["orientation"])
        theta = np.deg2rad(yaw)
        slot = C.DEFECT_TARGET_SLOTS[colour][slot_index]
        picks.append(
            {
                "execution_index": execution_index,
                "detection_id": f"oracle:{name}",
                "object_id": name,
                "oracle_object_name": name,
                "pose_source": ORACLE_SOURCE,
                "category": "cube",
                "colour": colour,
                "defect_class": C.DEFECT_CLASS_BY_COLOUR[colour],
                "expected_colour": colour,
                "expected_defect_class": C.DEFECT_CLASS_BY_COLOUR[colour],
                "xy": pose["position"][:2],
                "yaw_deg": yaw,
                "spread_dir": [-float(np.sin(theta)), float(np.cos(theta)), 0.0],
                "pick_z": C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE,
                "assigned_slot_index": slot_index,
                "assigned_slot": list(slot),
                "place_xy": list(slot),
                "place_z": C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE,
                "expected_centre_z": C.SURFACE_DZ + 0.5 * C.CUBE_EDGE,
                "place_spread_dir": [0.0, 1.0, 0.0],
                "place_yaw_deg": 0.0,
                "oracle_simulation_pose": pose,
            }
        )
    detection_ids = [pick["detection_id"] for pick in picks]
    return {
        "source": ORACLE_SOURCE,
        "task": ORACLE_SIX_TASK,
        "oracle_only": True,
        "perception_success_claimed": False,
        "defect_sort_ready": True,
        "defect_parts_planned": 6,
        "validated_cube_detection_ids": detection_ids,
        "commanded_objects": [name for name, _colour, _slot in EXECUTION_ORDER],
        "execution_order_reason": EXECUTION_ORDER_REASON,
        "observed_object_names": sorted(objects),
        "planned_scene_poses": planned_scene_poses,
        "open_gap_m": 2 * C.JAW_OPEN,
        "transit_clearance_m": TRANSIT_CLEARANCE_M,
        "distractor_safety": {
            "categories": ["cap", "vial"],
            "unintended_lift_forbidden": True,
            "max_translation_m": C.DISTRACTOR_TRANSLATION_LIMIT,
            "max_rotation_deg": C.DISTRACTOR_ROTATION_LIMIT_DEG,
        },
        "unprocessed_cube_safety": {
            "unintended_lift_forbidden": True,
            "max_translation_m": NON_SELECTED_CUBE_TRANSLATION_LIMIT_M,
            "max_rotation_deg": NON_SELECTED_CUBE_ROTATION_LIMIT_DEG,
        },
        "picks": picks,
    }


def validate_oracle_six_cube_plan(plan):
    """Validate exact inventory, mappings, slots, order, geometry, and poses."""
    if plan.get("task") != ORACLE_SIX_TASK or plan.get("source") != ORACLE_SOURCE:
        raise ValueError("Oracle six-cube task/source label is required")
    if plan.get("oracle_only") is not True or plan.get("defect_sort_ready") is not True:
        raise ValueError("Oracle six-cube plan must be ready and oracle-only")
    if plan.get("perception_success_claimed") is not False:
        raise ValueError("Oracle physics validation must not claim perception success")
    expected_order = [name for name, _colour, _slot in EXECUTION_ORDER]
    if plan.get("commanded_objects") != expected_order:
        raise ValueError("Oracle six-cube execution order changed")
    if plan.get("execution_order_reason") != EXECUTION_ORDER_REASON:
        raise ValueError("Oracle six-cube execution-order justification changed")
    picks = plan.get("picks", [])
    if len(picks) != 6 or plan.get("defect_parts_planned") != 6:
        raise ValueError("Oracle six-cube plan requires exactly six picks")
    if [pick.get("object_id") for pick in picks] != expected_order:
        raise ValueError("Oracle six-cube picks do not match the audited cube order")
    detection_ids = [pick.get("detection_id") for pick in picks]
    if len(set(detection_ids)) != 6 or detection_ids != plan.get("validated_cube_detection_ids"):
        raise ValueError("Oracle six-cube detection IDs must be unique and complete")
    slots = [tuple(pick.get("assigned_slot", [])) for pick in picks]
    expected_slots = [
        tuple(C.DEFECT_TARGET_SLOTS[colour][slot_index])
        for _name, colour, slot_index in EXECUTION_ORDER
    ]
    if slots != expected_slots or len(set(slots)) != 6:
        raise ValueError("Oracle six-cube slots must be exact and unique")
    if not np.isclose(plan.get("open_gap_m", 0), 2 * C.JAW_OPEN):
        raise ValueError("Oracle six-cube plan uses the wrong jaw opening")
    if not np.isclose(plan.get("transit_clearance_m", -1), TRANSIT_CLEARANCE_M):
        raise ValueError("Oracle six-cube plan must retain 130 mm clearance")
    distractor = plan.get("distractor_safety", {})
    if (
        distractor.get("unintended_lift_forbidden") is not True
        or not np.isclose(
            distractor.get("max_translation_m", -1),
            C.DISTRACTOR_TRANSLATION_LIMIT,
        )
        or not np.isclose(
            distractor.get("max_rotation_deg", -1),
            C.DISTRACTOR_ROTATION_LIMIT_DEG,
        )
    ):
        raise ValueError("Oracle six-cube distractor safety contract changed")
    unprocessed = plan.get("unprocessed_cube_safety", {})
    if (
        unprocessed.get("unintended_lift_forbidden") is not True
        or not np.isclose(
            unprocessed.get("max_translation_m", -1),
            NON_SELECTED_CUBE_TRANSLATION_LIMIT_M,
        )
        or not np.isclose(
            unprocessed.get("max_rotation_deg", -1),
            NON_SELECTED_CUBE_ROTATION_LIMIT_DEG,
        )
    ):
        raise ValueError("Oracle six-cube unprocessed-cube safety contract changed")
    for index, (pick, (name, colour, slot_index)) in enumerate(zip(picks, EXECUTION_ORDER)):
        required = {
            "execution_index": index,
            "object_id": name,
            "oracle_object_name": name,
            "pose_source": ORACLE_SOURCE,
            "category": "cube",
            "colour": colour,
            "defect_class": C.DEFECT_CLASS_BY_COLOUR[colour],
            "expected_colour": colour,
            "expected_defect_class": C.DEFECT_CLASS_BY_COLOUR[colour],
            "assigned_slot_index": slot_index,
        }
        if any(pick.get(key) != value for key, value in required.items()):
            raise ValueError(f"Wrong oracle metadata for {name}")
        if not np.allclose(pick.get("place_xy"), pick["assigned_slot"]):
            raise ValueError(f"Wrong destination for {name}")
        if not _cube_contained(pick["assigned_slot"], colour):
            raise ValueError(f"Assigned footprint leaves the {colour} zone")
        if not np.isclose(pick.get("pick_z", -1), C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE):
            raise ValueError(f"Wrong pick height for {name}")
        if not np.isclose(pick.get("place_z", -1), C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE):
            raise ValueError(f"Wrong flat placement height for {name}")
        if not np.isclose(pick.get("expected_centre_z", -1), C.SURFACE_DZ + 0.5 * C.CUBE_EDGE):
            raise ValueError(f"Wrong flat centre height for {name}")
        pose = pick.get("oracle_simulation_pose", {})
        if _pose_position(pose).shape != (3,) or np.asarray(pose.get("orientation", [])).shape != (4,):
            raise ValueError(f"Missing full oracle pose for {name}")
    planned = plan.get("planned_scene_poses", {})
    if set(planned) != expected_scene_object_names():
        raise ValueError("Planned scene poses must contain all ten tracked objects")
    for name, pose in planned.items():
        if _pose_position(pose).shape != (3,) or np.asarray(pose.get("orientation", [])).shape != (4,):
            raise ValueError(f"Invalid planned scene pose for {name}")
    return True


def validate_six_execution_start(plan, current_objects):
    """Compare all ten recreated objects and enforce pose gates on all cubes."""
    failures = []
    try:
        validate_oracle_six_cube_plan(plan)
    except (TypeError, ValueError) as error:
        failures.append(f"plan_contract:{error}")
    expected = expected_scene_object_names()
    actual = set(current_objects)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        failures.append(f"missing_objects:{missing}")
    if extra:
        failures.append(f"extra_objects:{extra}")
    counts = {
        "cubes": sum(name.startswith("SortCube") for name in actual),
        "caps": sum(name.startswith("SortCap") for name in actual),
        "vials": sum(name.startswith("SortVial") for name in actual),
    }
    if counts != {"cubes": 6, "caps": 2, "vials": 2}:
        failures.append(f"wrong_object_counts:{counts}")
    rows = []
    planned_poses = plan.get("planned_scene_poses", {})
    for name in sorted(expected):
        planned = planned_poses.get(name)
        current = current_objects.get(name)
        translation, rotation = (None, None)
        if planned and current:
            translation, rotation = _pose_error(planned, current)
        enforce = name.startswith("SortCube")
        passed = bool(
            planned
            and current
            and (not enforce or (
                translation is not None
                and translation <= EXECUTION_POSE_TRANSLATION_TOLERANCE_M
                and rotation is not None
                and rotation <= EXECUTION_POSE_ROTATION_TOLERANCE_DEG
            ))
        )
        rows.append(
            {
                "name": name,
                "planned_pose": planned,
                "execution_start_pose": current,
                "translation_error_m": translation,
                "translation_error_mm": translation * 1000 if translation is not None else None,
                "rotation_error_deg": rotation,
                "cube_pose_gate_enforced": enforce,
                "passed": passed,
            }
        )
        if enforce and not passed:
            failures.append(f"cube_pose_mismatch:{name}")
    cube_rows = [row for row in rows if row["cube_pose_gate_enforced"]]
    return {
        "planned_pose_source": ORACLE_SOURCE,
        "translation_tolerance_m": EXECUTION_POSE_TRANSLATION_TOLERANCE_M,
        "rotation_tolerance_deg": EXECUTION_POSE_ROTATION_TOLERANCE_DEG,
        "expected_object_names": sorted(expected),
        "actual_object_names": sorted(actual),
        "object_counts": counts,
        "missing_objects": missing,
        "extra_objects": extra,
        "objects": rows,
        "maximum_cube_translation_error_mm": max(
            (row["translation_error_mm"] for row in cube_rows if row["translation_error_mm"] is not None),
            default=None,
        ),
        "maximum_cube_rotation_error_deg": max(
            (row["rotation_error_deg"] for row in cube_rows if row["rotation_error_deg"] is not None),
            default=None,
        ),
        "pre_command_validation_passed": not failures,
        "failures": failures,
    }


def preflight_oracle_six_motion(plan, home, tool, limits, motion_segments_by_pick):
    validate_oracle_six_cube_plan(plan)
    return preflight_oracle_picks(
        plan["picks"], home, tool, limits, motion_segments_by_pick
    )


def validate_per_cube_safety(
    plan,
    baseline,
    current,
    selected_object,
    completed_objects,
    phase,
    objects_lifted=None,
):
    """Apply the Phase 3B gate before pickup, after lift, and after placement."""
    validate_oracle_six_cube_plan(plan)
    picks = {pick["object_id"]: pick for pick in plan["picks"]}
    if selected_object not in picks:
        raise ValueError(f"Unknown selected cube: {selected_object}")
    completed = list(completed_objects)
    failures = []
    rows = {
        "selected_pose": None,
        "destination_occupants": [],
        "previously_placed": [],
        "unprocessed_cubes": [],
        "distractors": [],
    }
    if set(current) != expected_scene_object_names():
        failures.append("tracked_object_inventory_changed")
        return {
            "phase": phase,
            "selected_object": selected_object,
            "completed_objects": completed,
            "objects_lifted": sorted(set(objects_lifted or [])),
            **rows,
            "passed": False,
            "failures": failures,
        }

    if phase == "pre_pick":
        translation, rotation = _pose_error(baseline[selected_object], current[selected_object])
        selected_passed = bool(
            translation is not None
            and translation <= NON_SELECTED_CUBE_TRANSLATION_LIMIT_M
            and rotation is not None
            and rotation <= NON_SELECTED_CUBE_ROTATION_LIMIT_DEG
        )
        rows["selected_pose"] = {
            "translation_mm": translation * 1000 if translation is not None else None,
            "rotation_deg": rotation,
            "passed": selected_passed,
        }
        if not selected_passed:
            failures.append("selected_cube_displaced_before_pick")
        slot = np.asarray(picks[selected_object]["assigned_slot"], float)
        for name in picks:
            if name == selected_object:
                continue
            position = _pose_position(current[name])
            if position.shape == (3,) and np.all(np.abs(position[:2] - slot) < C.CUBE_EDGE):
                rows["destination_occupants"].append(name)
        if rows["destination_occupants"]:
            failures.append("destination_slot_occupied")

    placed_to_check = list(completed)
    if phase == "post_place" and selected_object not in placed_to_check:
        placed_to_check.append(selected_object)
    for name in placed_to_check:
        result = _placement_result(picks[name], current[name])
        rows["previously_placed"].append({"name": name, **result})
        if not result["placement_passed"]:
            failures.append(f"placed_cube_invalid:{name}")

    unprocessed = [
        name for name in picks if name not in completed and name != selected_object
    ]
    for name in unprocessed:
        translation, rotation = _pose_error(baseline[name], current[name])
        passed = bool(
            translation is not None
            and translation <= NON_SELECTED_CUBE_TRANSLATION_LIMIT_M
            and rotation is not None
            and rotation <= NON_SELECTED_CUBE_ROTATION_LIMIT_DEG
        )
        rows["unprocessed_cubes"].append(
            {
                "name": name,
                "translation_mm": translation * 1000 if translation is not None else None,
                "rotation_deg": rotation,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(f"unprocessed_cube_displaced:{name}")

    for name in sorted(n for n in baseline if n.startswith(("SortCap", "SortVial"))):
        translation, rotation = _pose_error(baseline[name], current[name])
        passed = bool(
            translation is not None
            and translation <= C.DISTRACTOR_TRANSLATION_LIMIT
            and rotation is not None
            and rotation <= C.DISTRACTOR_ROTATION_LIMIT_DEG
        )
        rows["distractors"].append(
            {
                "name": name,
                "translation_mm": translation * 1000 if translation is not None else None,
                "rotation_deg": rotation,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(f"distractor_displaced:{name}")

    lifted = sorted(set(objects_lifted or []))
    if phase == "post_lift" and lifted != [selected_object]:
        failures.append(f"unexpected_lift:{lifted}")
    return {
        "phase": phase,
        "selected_object": selected_object,
        "completed_objects": completed,
        "objects_lifted": lifted,
        **rows,
        "passed": not failures,
        "failures": failures,
    }


def advance_abort_state(existing_reason, safety_result):
    """Latch the first failure so no later pick can become authorized."""
    if existing_reason:
        return existing_reason, False
    if not safety_result.get("passed"):
        return "; ".join(safety_result.get("failures", ["safety failure"])), False
    return None, True


def evaluate_oracle_six_success(plan, events, before, after, preflight, abort_reason=None):
    """Build strict aggregate and per-cube Phase 3B acceptance results."""
    validate_oracle_six_cube_plan(plan)
    picks = {pick["object_id"]: pick for pick in plan["picks"]}
    event_by_name = {event.get("tracked_object"): event for event in events}
    per_cube = []
    zone_counts = {colour: 0 for colour in C.DEFECT_CLASS_BY_COLOUR}
    unintended_grasps = []
    for name, colour, _slot_index in EXECUTION_ORDER:
        event = event_by_name.get(name, {})
        final_pose = after.get(name)
        placement = _placement_result(picks[name], final_pose or {})
        if placement["placement_passed"]:
            zone_counts[colour] += 1
        lifted = sorted(set(event.get("objects_lifted", [])))
        if lifted != [name]:
            unintended_grasps.extend(obj for obj in lifted if obj != name)
        per_cube.append(
            {
                "object_id": name,
                "colour": colour,
                "defect_class": C.DEFECT_CLASS_BY_COLOUR[colour],
                "starting_pose": event.get("starting_pose", before.get(name)),
                "assigned_slot": picks[name]["assigned_slot"],
                "pickup_verified": event.get("pickup_verified") is True,
                "objects_lifted": lifted,
                "final_pose": final_pose,
                **placement,
                "motion_time_s": float(event.get("motion_time_s", 0.0)),
                "wall_clock_time_s": float(event.get("wall_clock_time_s", 0.0)),
                "retries": event.get("retries", []),
                "recoveries": event.get("recoveries", []),
            }
        )
    distractors = []
    for name in sorted(n for n in before if n.startswith(("SortCap", "SortVial"))):
        translation, rotation = _pose_error(before[name], after.get(name, {}))
        distractors.append(
            {
                "name": name,
                "translation_mm": translation * 1000 if translation is not None else None,
                "rotation_deg": rotation,
                "translation_exceeded": translation is None or translation > C.DISTRACTOR_TRANSLATION_LIMIT,
                "rotation_exceeded": rotation is None or rotation > C.DISTRACTOR_ROTATION_LIMIT_DEG,
            }
        )
    distractor_grasps = sorted(
        {
            obj
            for event in events
            for obj in event.get("objects_lifted", [])
            if obj.startswith(("SortCap", "SortVial"))
        }
    )
    checks = {
        "defect_parts_detected_planned_6_of_6": plan.get("defect_parts_planned") == 6,
        "successful_pickups_6_of_6": sum(row["pickup_verified"] for row in per_cube) == 6,
        "correct_zone_placements_6_of_6": sum(row["placement_passed"] for row in per_cube) == 6,
        "exactly_two_cubes_per_zone": zone_counts == C.DEFECT_EXPECTED_COUNTS,
        "no_distractor_grasps": not distractor_grasps,
        "no_distractor_displacement_violations": all(
            not row["translation_exceeded"] and not row["rotation_exceeded"]
            for row in distractors
        ),
        "no_unintended_cube_grasps": not unintended_grasps,
        "all_six_events_recorded": len(events) == 6 and len(event_by_name) == 6,
        "ik_and_clearance_preflight_passed": preflight.get("preflight_passed") is True,
        "no_safety_abort": abort_reason is None,
    }
    return {
        "oracle_sort_success": all(checks.values()),
        "success_checks": checks,
        "abort_conditions": [name for name, passed in checks.items() if not passed],
        "defect_parts_detected": 6,
        "defect_parts_detected_source": ORACLE_SOURCE,
        "defect_parts_planned": 6,
        "perception_success_claimed": False,
        "successful_pickups": sum(row["pickup_verified"] for row in per_cube),
        "correct_zone_placements": sum(row["placement_passed"] for row in per_cube),
        "zone_counts": zone_counts,
        "distractor_grasps": distractor_grasps,
        "distractor_displacements": distractors,
        "distractor_displacement_violations": [
            row["name"]
            for row in distractors
            if row["translation_exceeded"] or row["rotation_exceeded"]
        ],
        "unintended_cube_grasps": sorted(set(unintended_grasps)),
        "per_cube_results": per_cube,
    }


def annotate_six_cube_after_image(image_path, camera_json, report, output_path):
    """Annotate all assigned slots and final cube centres on the after image."""
    import cv2
    from backproject import CameraModel

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    camera = CameraModel(str(camera_json))
    bgr = {"red": (20, 20, 230), "blue": (230, 80, 20), "yellow": (0, 210, 240)}
    for row in report.get("per_cube_results", []):
        slot_world = [*row["assigned_slot"], C.TABLE_TOP_Z + C.SURFACE_DZ]
        slot = tuple(camera.project(slot_world).astype(int))
        final_pose = row.get("final_pose") or {}
        centre = _pose_position(final_pose)
        cv2.drawMarker(image, slot, bgr[row["colour"]], cv2.MARKER_CROSS, 22, 2)
        if centre.shape == (3,):
            actual = tuple(camera.project(centre).astype(int))
            cv2.circle(image, actual, 10, (40, 220, 40), 2)
            cv2.putText(image, row["object_id"], (actual[0] + 8, actual[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 220, 40), 1)
    label = "Phase 3B oracle six-cube PASS" if report.get("oracle_sort_success") else "Phase 3B oracle six-cube FAIL"
    cv2.putText(image, label, (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write annotated image: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    scene_state = json.loads((run / "scene_state.json").read_text(encoding="utf-8"))
    ground_truth = json.loads((run / "ground_truth.json").read_text(encoding="utf-8"))
    plan = build_oracle_six_cube_plan(scene_state, ground_truth)
    validate_oracle_six_cube_plan(plan)
    (run / "sort_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "[oracle-six] prepared all six cubes from oracle_simulation_pose in order "
        + ", ".join(plan["commanded_objects"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
