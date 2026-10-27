"""Build and evaluate the isolated Dataset 3 oracle motion smoke test.

This module never performs perception.  Simulator state is accepted only for the
explicit ``oracle_single_cube_smoke`` task and is labelled
``oracle_simulation_pose`` throughout the generated plan and report.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import ik
import scene_config as C


ORACLE_TASK = "oracle_single_cube_smoke"
ORACLE_SOURCE = "oracle_simulation_pose"
SELECTED_CUBE = "SortCube2"
TARGET_SLOT = tuple(C.DEFECT_TARGET_SLOTS["red"][0])
TRANSIT_CLEARANCE_M = 0.130
NON_SELECTED_CUBE_TRANSLATION_LIMIT_M = 0.002
NON_SELECTED_CUBE_ROTATION_LIMIT_DEG = 5.0
EXECUTION_POSE_TRANSLATION_TOLERANCE_M = 0.002
EXECUTION_POSE_ROTATION_TOLERANCE_DEG = 5.0


def expected_scene_object_names():
    return {
        *(f"SortCube{i}" for i in range(6)),
        *(f"SortCap{i}" for i in range(2)),
        *(f"SortVial{i}" for i in range(2)),
    }


def _yaw_deg(quaternion):
    w, x, y, z = np.asarray(quaternion, float)
    return float(
        np.degrees(
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        )
    )


def build_oracle_smoke_plan(scene_state, ground_truth):
    """Build the deterministic one-red-cube plan from captured simulator state."""
    objects = scene_state.get("objects", {})
    expected_names = expected_scene_object_names()
    if set(objects) != expected_names:
        raise ValueError("Oracle smoke scene must contain exactly the ten audited objects")
    truth = {row["name"]: row for row in ground_truth.get("objects", [])}
    selected_truth = truth.get(SELECTED_CUBE)
    if not selected_truth or selected_truth.get("category") != "cube":
        raise ValueError(f"Missing selected oracle cube: {SELECTED_CUBE}")
    if selected_truth.get("colour") != "red":
        raise ValueError("Selected oracle cube must be red / Defect Class A")
    saved = objects[SELECTED_CUBE]
    position = np.asarray(saved.get("position", []), float)
    quaternion = np.asarray(saved.get("orientation", []), float)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("Selected oracle cube has an invalid simulator pose")
    yaw = _yaw_deg(quaternion)
    theta = np.deg2rad(yaw)
    pick = {
        "detection_id": f"oracle:{SELECTED_CUBE}",
        "object_id": SELECTED_CUBE,
        "oracle_object_name": SELECTED_CUBE,
        "pose_source": ORACLE_SOURCE,
        "category": "cube",
        "colour": "red",
        "defect_class": C.DEFECT_CLASS_BY_COLOUR["red"],
        "expected_colour": "red",
        "expected_defect_class": C.DEFECT_CLASS_BY_COLOUR["red"],
        "xy": position[:2].tolist(),
        "yaw_deg": yaw,
        "spread_dir": [-float(np.sin(theta)), float(np.cos(theta)), 0.0],
        "pick_z": C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE,
        "assigned_slot": list(TARGET_SLOT),
        "place_xy": list(TARGET_SLOT),
        "place_z": C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE,
        "expected_centre_z": C.SURFACE_DZ + 0.5 * C.CUBE_EDGE,
        "place_spread_dir": [0.0, 1.0, 0.0],
        "place_yaw_deg": 0.0,
        "oracle_simulation_pose": {
            "position": position.tolist(),
            "orientation": quaternion.tolist(),
        },
    }
    return {
        "source": ORACLE_SOURCE,
        "task": ORACLE_TASK,
        "oracle_only": True,
        "selected_object": SELECTED_CUBE,
        "commanded_objects": [SELECTED_CUBE],
        "observed_object_names": sorted(objects),
        "open_gap_m": 2 * C.JAW_OPEN,
        "transit_clearance_m": TRANSIT_CLEARANCE_M,
        "distractor_safety": {
            "categories": ["cap", "vial"],
            "unintended_lift_forbidden": True,
            "max_translation_m": C.DISTRACTOR_TRANSLATION_LIMIT,
            "max_rotation_deg": C.DISTRACTOR_ROTATION_LIMIT_DEG,
        },
        "non_selected_cube_safety": {
            "unintended_lift_forbidden": True,
            "max_translation_m": NON_SELECTED_CUBE_TRANSLATION_LIMIT_M,
            "max_rotation_deg": NON_SELECTED_CUBE_ROTATION_LIMIT_DEG,
        },
        "picks": [pick],
    }


def validate_oracle_smoke_plan(plan):
    """Reject any oracle plan that could command more than the selected cube."""
    if plan.get("task") != ORACLE_TASK or plan.get("source") != ORACLE_SOURCE:
        raise ValueError("Oracle smoke task/source label is required")
    if plan.get("oracle_only") is not True:
        raise ValueError("Oracle smoke plan must be explicitly oracle-only")
    if plan.get("selected_object") != SELECTED_CUBE:
        raise ValueError("Oracle smoke selected the wrong cube")
    if plan.get("commanded_objects") != [SELECTED_CUBE]:
        raise ValueError("Oracle smoke may command exactly one selected cube")
    picks = plan.get("picks", [])
    if len(picks) != 1:
        raise ValueError("Oracle smoke requires exactly one pick")
    pick = picks[0]
    required = {
        "oracle_object_name": SELECTED_CUBE,
        "pose_source": ORACLE_SOURCE,
        "category": "cube",
        "colour": "red",
        "defect_class": C.DEFECT_CLASS_BY_COLOUR["red"],
        "expected_colour": "red",
        "expected_defect_class": C.DEFECT_CLASS_BY_COLOUR["red"],
    }
    if any(pick.get(key) != value for key, value in required.items()):
        raise ValueError("Oracle smoke pick metadata is not the selected red cube")
    if not np.allclose(pick.get("assigned_slot"), TARGET_SLOT) or not np.allclose(
        pick.get("place_xy"), TARGET_SLOT
    ):
        raise ValueError("Oracle smoke destination must be Red Slot 0")
    if not np.isclose(plan.get("transit_clearance_m", -1), TRANSIT_CLEARANCE_M):
        raise ValueError("Oracle smoke must retain 130 mm transit clearance")
    if not np.isclose(pick.get("pick_z", -1), C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE):
        raise ValueError("Oracle smoke has an invalid pick height")
    if not np.isclose(pick.get("place_z", -1), C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE):
        raise ValueError("Oracle smoke has an invalid flat placement height")
    if not np.isclose(
        pick.get("expected_centre_z", -1), C.SURFACE_DZ + 0.5 * C.CUBE_EDGE
    ):
        raise ValueError("Oracle smoke must use half-cube flat centre height")
    if not np.isclose(plan.get("open_gap_m", 0), 2 * C.JAW_OPEN):
        raise ValueError("Oracle smoke uses the wrong jaw opening")
    x, y = pick["xy"]
    half = C.CUBE_EDGE / 2
    if not (
        C.TABLE_X[0] <= x - half <= x + half <= C.TABLE_X[1]
        and C.TABLE_Y[0] <= y - half <= y + half <= C.TABLE_Y[1]
    ):
        raise ValueError("Oracle smoke source cube footprint is outside the table")
    distractor = plan.get("distractor_safety", {})
    if (
        distractor.get("unintended_lift_forbidden") is not True
        or not np.isclose(
            distractor.get("max_translation_m", -1), C.DISTRACTOR_TRANSLATION_LIMIT
        )
        or not np.isclose(
            distractor.get("max_rotation_deg", -1), C.DISTRACTOR_ROTATION_LIMIT_DEG
        )
    ):
        raise ValueError("Oracle smoke distractor safety contract changed")
    cubes = plan.get("non_selected_cube_safety", {})
    if (
        cubes.get("unintended_lift_forbidden") is not True
        or not np.isclose(
            cubes.get("max_translation_m", -1),
            NON_SELECTED_CUBE_TRANSLATION_LIMIT_M,
        )
        or not np.isclose(
            cubes.get("max_rotation_deg", -1),
            NON_SELECTED_CUBE_ROTATION_LIMIT_DEG,
        )
    ):
        raise ValueError("Oracle smoke non-selected cube safety contract changed")
    for key in ("xy", "place_xy"):
        value = np.asarray(pick.get(key, []), float)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("Oracle smoke contains invalid coordinates")
    planned_pose = pick.get("oracle_simulation_pose", {})
    if (
        np.asarray(planned_pose.get("position", []), float).shape != (3,)
        or np.asarray(planned_pose.get("orientation", []), float).shape != (4,)
    ):
        raise ValueError("Oracle smoke is missing its captured simulator pose")
    return True


def validate_execution_start(plan, current_objects):
    """Compare the recreated Isaac scene with the captured oracle plan."""
    failures = []
    try:
        validate_oracle_smoke_plan(plan)
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

    pick = plan.get("picks", [{}])[0] if plan.get("picks") else {}
    planned = pick.get("oracle_simulation_pose")
    current = current_objects.get(SELECTED_CUBE)
    translation_error_m = None
    rotation_error_deg = None
    if planned and current:
        planned_position = np.asarray(planned.get("position", []), float)
        current_position = np.asarray(current.get("position", []), float)
        planned_orientation = np.asarray(planned.get("orientation", []), float)
        current_orientation = np.asarray(current.get("orientation", []), float)
        if planned_position.shape == (3,) and current_position.shape == (3,):
            translation_error_m = float(
                np.linalg.norm(current_position - planned_position)
            )
        if planned_orientation.shape == (4,) and current_orientation.shape == (4,):
            planned_norm = np.linalg.norm(planned_orientation)
            current_norm = np.linalg.norm(current_orientation)
            if planned_norm > 0 and current_norm > 0:
                similarity = abs(
                    np.dot(planned_orientation, current_orientation)
                    / (planned_norm * current_norm)
                )
                rotation_error_deg = float(
                    np.degrees(2 * np.arccos(np.clip(similarity, 0, 1)))
                )
    if translation_error_m is None:
        failures.append("selected_cube_translation_unavailable")
    elif translation_error_m > EXECUTION_POSE_TRANSLATION_TOLERANCE_M:
        failures.append("selected_cube_translation_exceeded")
    if rotation_error_deg is None:
        failures.append("selected_cube_rotation_unavailable")
    elif rotation_error_deg > EXECUTION_POSE_ROTATION_TOLERANCE_DEG:
        failures.append("selected_cube_rotation_exceeded")

    return {
        "planned_pose_source": ORACLE_SOURCE,
        "planned_pose": planned,
        "execution_start_pose": current,
        "translation_error_m": translation_error_m,
        "translation_error_mm": (
            translation_error_m * 1000 if translation_error_m is not None else None
        ),
        "rotation_error_deg": rotation_error_deg,
        "translation_tolerance_m": EXECUTION_POSE_TRANSLATION_TOLERANCE_M,
        "rotation_tolerance_deg": EXECUTION_POSE_ROTATION_TOLERANCE_DEG,
        "expected_object_names": sorted(expected),
        "actual_object_names": sorted(actual),
        "object_counts": counts,
        "missing_objects": missing,
        "extra_objects": extra,
        "selected_colour": pick.get("colour"),
        "selected_defect_class": pick.get("defect_class"),
        "destination_slot": pick.get("assigned_slot"),
        "pre_command_validation_passed": not failures,
        "failures": failures,
    }


def preflight_oracle_motion(plan, home, tool, limits, motion_segments):
    """Check every phase endpoint and geometric clearance before robot commands."""
    validate_oracle_smoke_plan(plan)
    pick = plan["picks"][0]
    expected_names = [
        "approach",
        "descend",
        "close",
        "dwell",
        "lift",
        "carry",
        "lower",
        "release",
        "settle",
        "retreat",
        "home",
    ]
    if [row[0] for row in motion_segments] != expected_names:
        raise ValueError("Oracle smoke motion phases changed")
    pick_point = np.array([*pick["xy"], pick["pick_z"]], float)
    place_point = np.array([*pick["place_xy"], pick["place_z"]], float)
    safe_pick = pick_point + [0, 0, TRANSIT_CLEARANCE_M]
    safe_place = place_point + [0, 0, TRANSIT_CLEARANCE_M]
    if not np.allclose(motion_segments[0][2], safe_pick) or not np.allclose(
        motion_segments[5][2], safe_place
    ):
        raise ValueError("Oracle smoke clearance waypoints are not 130 mm")
    tallest_object_height = max(C.CUBE_EDGE, C.CAP_HEIGHT, 2 * C.VIAL_RADIUS)
    clearance_margin = min(safe_pick[2], safe_place[2]) - (
        C.SURFACE_DZ + tallest_object_height
    )
    geometric_clearance_passed = bool(
        clearance_margin > 0.05
        and np.allclose(motion_segments[4][2], safe_pick)
        and np.allclose(motion_segments[9][2], safe_place)
    )
    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
    down = [0.0, 0.0, -1.0]
    q, home_ok = ik.solve(q, home, down, limits, tool_offset=tool)
    rows = []
    for name, _p0, p1, _duration, _j0, _j1, spread in motion_segments:
        q, ok = ik.solve(q, p1, down, limits, tool_offset=tool, spread_dir=spread)
        if not ok:
            q, ok = ik.solve(
                q,
                p1,
                down,
                limits,
                tool_offset=tool,
                spread_dir=spread,
                iters=500,
                damping=0.02,
            )
        rows.append({"phase": name, "ik_reachable": bool(ok)})
        if not ok:
            break
    result = {
        "home_ik_reachable": bool(home_ok),
        "phase_ik_checks": rows,
        "all_phase_endpoints_ik_reachable": bool(home_ok)
        and len(rows) == len(motion_segments)
        and all(row["ik_reachable"] for row in rows),
        "transit_clearance_m": TRANSIT_CLEARANCE_M,
        "minimum_geometric_clearance_margin_m": float(clearance_margin),
        "geometric_clearance_validation_passed": geometric_clearance_passed,
    }
    result["preflight_passed"] = bool(
        result["all_phase_endpoints_ik_reachable"]
        and result["geometric_clearance_validation_passed"]
    )
    return result


def _rotation_delta_deg(before_pose, after_pose):
    before_q = np.asarray(before_pose.get("orientation", []), float)
    after_q = np.asarray(after_pose.get("orientation", []), float)
    if before_q.shape != (4,) or after_q.shape != (4,):
        return float("inf")
    similarity = abs(
        np.dot(before_q, after_q)
        / (np.linalg.norm(before_q) * np.linalg.norm(after_q))
    )
    return float(np.degrees(2 * np.arccos(np.clip(similarity, 0, 1))))


def evaluate_object_isolation(before, after, events, selected=SELECTED_CUBE):
    """Measure distractors and all five non-selected cubes independently."""
    lifted = {
        name
        for event in events
        for name in event.get("objects_lifted", []) + event.get("distractor_lifts", [])
    }
    distractor_rows = []
    cube_rows = []
    for name in sorted(before):
        if name == selected or name not in after:
            continue
        translation = float(
            np.linalg.norm(
                np.asarray(after[name]["centre"], float)
                - np.asarray(before[name]["centre"], float)
            )
        )
        rotation = _rotation_delta_deg(before[name], after[name])
        if name.startswith(("SortCap", "SortVial")):
            distractor_rows.append(
                {
                    "name": name,
                    "translation_mm": translation * 1000,
                    "rotation_deg": rotation,
                    "lifted": name in lifted,
                    "translation_exceeded": translation
                    > C.DISTRACTOR_TRANSLATION_LIMIT,
                    "rotation_exceeded": rotation > C.DISTRACTOR_ROTATION_LIMIT_DEG,
                }
            )
        elif name.startswith("SortCube"):
            cube_rows.append(
                {
                    "name": name,
                    "translation_mm": translation * 1000,
                    "rotation_deg": rotation,
                    "lifted": name in lifted,
                    "translation_exceeded": translation
                    > NON_SELECTED_CUBE_TRANSLATION_LIMIT_M,
                    "rotation_exceeded": rotation
                    > NON_SELECTED_CUBE_ROTATION_LIMIT_DEG,
                }
            )
    distractor_passed = len(distractor_rows) == 4 and all(
        not row["lifted"]
        and not row["translation_exceeded"]
        and not row["rotation_exceeded"]
        for row in distractor_rows
    )
    cubes_passed = len(cube_rows) == 5 and all(
        not row["lifted"]
        and not row["translation_exceeded"]
        and not row["rotation_exceeded"]
        for row in cube_rows
    )
    return {
        "distractor_displacements": distractor_rows,
        "distractor_grasps": sorted(
            row["name"] for row in distractor_rows if row["lifted"]
        ),
        "distractor_safety_passed": distractor_passed,
        "non_selected_cube_displacements": cube_rows,
        "non_selected_cube_grasps": sorted(
            row["name"] for row in cube_rows if row["lifted"]
        ),
        "non_selected_cube_safety_passed": cubes_passed,
    }


def evaluate_smoke_success(plan, events, before, after, preflight):
    """Return the strict Phase 3A success fields and abort reasons."""
    validate_oracle_smoke_plan(plan)
    isolation = evaluate_object_isolation(before, after, events)
    event = events[0] if len(events) == 1 else {}
    selected_after = after.get(SELECTED_CUBE, {})
    final_height = selected_after.get("centre", [None, None, None])[2]
    expected_height = C.TABLE_TOP_Z + C.SURFACE_DZ + 0.5 * C.CUBE_EDGE
    height_passed = final_height is not None and abs(final_height - expected_height) <= 0.0015
    checks = {
        "exactly_one_intended_cube_picked": len(events) == 1
        and event.get("tracked_object") == SELECTED_CUBE
        and event.get("pickup_verified") is True,
        "intended_cube_placement_verified": event.get("placement_verified") is True
        and event.get("correct_zone_placement") is True,
        "correct_zone_containment": event.get("correct_zone_containment") is True,
        "assigned_slot_error_within_5mm": float(
            event.get("assigned_slot_error_mm", float("inf"))
        )
        <= C.DEFECT_SLOT_TOLERANCE * 1000 + 1e-9,
        "final_cube_height_within_tolerance": height_passed,
        "no_distractor_grasped": not isolation["distractor_grasps"],
        "distractor_safety_passed": isolation["distractor_safety_passed"],
        "non_selected_cube_safety_passed": isolation[
            "non_selected_cube_safety_passed"
        ],
        "ik_and_clearance_preflight_passed": preflight.get("preflight_passed")
        is True,
    }
    return {
        **isolation,
        "success_checks": checks,
        "smoke_test_success": all(checks.values()),
        "final_cube_height_m": final_height,
        "expected_flat_cube_centre_height_m": expected_height,
        "abort_conditions": [name for name, passed in checks.items() if not passed],
    }


def annotate_after_image(image_path, camera_json, report, output_path):
    """Annotate the selected final centre and Red Slot 0 on the after image."""
    import cv2

    from backproject import CameraModel

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    camera = CameraModel(str(camera_json))
    after = report["after"][SELECTED_CUBE]["centre"]
    actual = tuple(camera.project(after).astype(int))
    slot_world = [*TARGET_SLOT, C.TABLE_TOP_Z + C.SURFACE_DZ]
    slot = tuple(camera.project(slot_world).astype(int))
    cv2.drawMarker(image, slot, (20, 20, 230), cv2.MARKER_CROSS, 24, 3)
    cv2.circle(image, actual, 12, (40, 220, 40), 3)
    cv2.putText(
        image,
        "Red Slot 0",
        (slot[0] + 10, slot[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 230),
        2,
    )
    cv2.putText(
        image,
        f"{SELECTED_CUBE} oracle motion smoke",
        (actual[0] + 12, actual[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (40, 220, 40),
        2,
    )
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write annotated image: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    scene_state = json.loads((run / "scene_state.json").read_text(encoding="utf-8"))
    ground_truth = json.loads(
        (run / "ground_truth.json").read_text(encoding="utf-8")
    )
    plan = build_oracle_smoke_plan(scene_state, ground_truth)
    validate_oracle_smoke_plan(plan)
    (run / "sort_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[oracle-smoke] prepared {SELECTED_CUBE} from {ORACLE_SOURCE} "
        f"to Red Slot 0 {TARGET_SLOT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
