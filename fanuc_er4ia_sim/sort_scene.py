"""Execute the Cosmos sort plan with physical grasps and pose-based reporting.

Run ./run_sort_pipeline.sh RUN_DIR to capture, perceive, plan, render and encode.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scene_config as C
import fk
import ik
from cube_pose import square_yaw


def ease(t):
    return t * t * (3 - 2 * t)


def turn(a, b, t):
    a, b = np.array(a, float), np.array(b, float)
    angle = np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a[:2], b[:2])) * t
    c, s = np.cos(angle), np.sin(angle)
    return np.array([a[0] * c - a[1] * s, a[0] * s + a[1] * c, 0.0])


def verify_cube_stack(positions, entries):
    """Check site, layer heights and actual adjacent-cube support geometry."""
    centres = []
    for entry in entries:
        name = entry.get("tracked_object")
        if name not in positions:
            return False
        pose = positions[name]
        if "yaw_deg" in pose and (
            abs(square_yaw(pose["yaw_deg"])) >= 8 or pose.get("tilt_deg", 0) >= 5
        ):
            return False
        centre = np.array(pose["centre"])
        if np.linalg.norm(centre[:2] - C.CUBE_STACK_XY) > C.CUBE_STACK_SITE_TOLERANCE:
            return False
        height = (
            C.TABLE_TOP_Z + C.SURFACE_DZ + (entry["stack_level"] + 0.5) * C.CUBE_EDGE
        )
        if abs(centre[2] - height) > 0.0015:
            return False
        if (
            centres
            and np.linalg.norm(centre[:2] - centres[-1][:2])
            > C.CUBE_STACK_LAYER_TOLERANCE
        ):
            return False
        centres.append(centre)
    return True


def cube_footprint_within_defect_zone(xy, colour, yaw_deg=0.0):
    """Require the complete rotated cube footprint inside the inset zone."""
    if colour not in C.DEFECT_ZONE_CENTERS:
        return False
    theta = np.deg2rad(float(yaw_deg))
    half_extent = 0.5 * C.CUBE_EDGE * (abs(np.cos(theta)) + abs(np.sin(theta)))
    delta = np.abs(np.asarray(xy, float) - C.DEFECT_ZONE_CENTERS[colour])
    available = 0.5 * np.asarray(C.DEFECT_ZONE_SIZE) - C.DEFECT_ZONE_SAFE_INSET
    return bool(np.all(delta + half_extent <= available + 1e-12))


def evaluate_defect_placement(pick, xy, yaw_deg=0.0):
    """Host-testable flat placement result for one assigned defect slot."""
    assigned = np.asarray(pick.get("assigned_slot", []), float)
    position = np.asarray(xy, float)
    if assigned.shape != (2,) or position.shape != (2,):
        slot_error = float("inf")
    else:
        slot_error = float(np.linalg.norm(position - assigned))
    contained = cube_footprint_within_defect_zone(
        position, pick.get("colour"), yaw_deg
    )
    return {
        "placement_position_error_mm": slot_error * 1000,
        "assigned_slot_error_mm": slot_error * 1000,
        "correct_zone_containment": contained,
        "correct_zone_placement": bool(
            contained and slot_error <= C.DEFECT_SLOT_TOLERANCE + 1e-12
        ),
    }


def _angle_delta_deg(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _pose_rotation_delta_deg(before_pose, after_pose):
    before_q = np.asarray(before_pose.get("orientation", []), float)
    after_q = np.asarray(after_pose.get("orientation", []), float)
    if before_q.shape == (4,) and after_q.shape == (4,):
        before_norm = np.linalg.norm(before_q)
        after_norm = np.linalg.norm(after_q)
        if before_norm > 0 and after_norm > 0:
            similarity = abs(np.dot(before_q, after_q) / (before_norm * after_norm))
            return float(np.degrees(2 * np.arccos(np.clip(similarity, 0, 1))))
    return _angle_delta_deg(
        after_pose.get("yaw_deg", 0), before_pose.get("yaw_deg", 0)
    )


def evaluate_distractor_safety(before, after, events, distractor_names):
    """Evaluate the deferred cap/vial lift, translation and rotation contract."""
    lifted = {
        name
        for event in events
        for name in event.get("distractor_lifts", event.get("objects_lifted", []))
        if name in distractor_names
    }
    displacements = []
    for name in distractor_names:
        if name not in before or name not in after:
            continue
        translation = float(
            np.linalg.norm(
                np.asarray(after[name]["centre"], float)
                - np.asarray(before[name]["centre"], float)
            )
        )
        rotation = _pose_rotation_delta_deg(before[name], after[name])
        displacements.append(
            {
                "name": name,
                "translation_mm": translation * 1000,
                "rotation_deg": rotation,
                "translation_exceeded": translation > C.DISTRACTOR_TRANSLATION_LIMIT,
                "rotation_exceeded": rotation > C.DISTRACTOR_ROTATION_LIMIT_DEG,
            }
        )
    return {
        "distractor_grasps": sorted(lifted),
        "distractor_displacements": displacements,
        "distractor_safety_passed": not lifted
        and not any(
            row["translation_exceeded"] or row["rotation_exceeded"]
            for row in displacements
        ),
    }


def build_defect_report(plan, events, motion_time_s, wall_clock_time_s, safety):
    """Build the Phase 1 report fields without requiring Isaac imports."""
    classified = [e for e in events if e.get("classification_correct") is not None]
    retries = [r for e in events for r in e.get("retries", [])]
    recoveries = [r for e in events for r in e.get("recoveries", [])]
    return {
        "defect_parts_detected": len(plan.get("validated_cube_detection_ids", [])),
        "successful_pickups": sum(bool(e.get("pickup_verified")) for e in events),
        "correct_zone_placements": sum(
            bool(e.get("placement_verified") and e.get("correct_zone_placement"))
            for e in events
        ),
        "distractor_grasps": safety.get("distractor_grasps", []),
        "distractor_displacements": safety.get("distractor_displacements", []),
        "distractor_safety_passed": safety.get("distractor_safety_passed", False),
        "classification_accuracy": (
            sum(bool(e["classification_correct"]) for e in classified) / len(classified)
            if classified
            else None
        ),
        "motion_time_s": float(motion_time_s),
        "wall_clock_time_s": float(wall_clock_time_s),
        "retry_count": len(retries),
        "recovery_count": len(recoveries),
        "retries": retries,
        "recoveries": recoveries,
    }


def segments(pick, home):
    p = np.array([*pick["xy"], pick["pick_z"]])
    d = np.array([*pick["place_xy"], pick["place_z"]])
    up, above = p + [0, 0, 0.13], d + [0, 0, 0.13]
    spread = pick.get("spread_dir")
    dest_spread = pick.get("place_spread_dir", spread)
    # Seat on the expected cube width instead of forcing rigid fingers through it.
    shut = C.CUBE_JAW_HALF_TARGET if pick["category"] == "cube" else C.JAW_CLOSED
    return [
        ("approach", home, up, 1.2, C.JAW_OPEN, C.JAW_OPEN, spread),
        ("descend", up, p, 0.8, C.JAW_OPEN, C.JAW_OPEN, spread),
        ("close", p, p, 0.6, C.JAW_OPEN, shut, spread),
        ("dwell", p, p, 0.5, shut, shut, spread),
        ("lift", p, up, 0.8, shut, shut, spread),
        ("carry", up, above, 1.4, shut, shut, dest_spread),
        ("lower", above, d, 0.8, shut, shut, dest_spread),
        ("release", d, d, 0.6, shut, C.JAW_OPEN, dest_spread),
        ("settle", d, d, 0.6, C.JAW_OPEN, C.JAW_OPEN, dest_spread),
        ("retreat", d, above, 0.8, C.JAW_OPEN, C.JAW_OPEN, dest_spread),
        ("home", above, home, 1.1, C.JAW_OPEN, C.JAW_OPEN, None),
    ]


def validate_plan(plan):
    if plan.get("source") != "Cosmos3-Nano":
        raise ValueError(
            "Expected a Cosmos3-Nano sort plan; run the perception pipeline first"
        )
    if not np.isclose(plan.get("open_gap_m", 0), 2 * C.JAW_OPEN):
        raise ValueError(
            "Plan uses a different jaw opening; regenerate capture and perception"
        )
    if plan.get("task") == "defect_sort":
        picks = plan.get("picks", [])
        if not plan.get("defect_sort_ready") or len(picks) != 6:
            raise ValueError("Defect sort requires exactly six validated cube picks")
        if any(p.get("category") != "cube" for p in picks):
            raise ValueError("Defect plan may contain cubes only; caps/vials are distractors")
        detection_ids = [p.get("detection_id") for p in picks]
        if len(set(detection_ids)) != 6:
            raise ValueError("Defect plan repeats a cube detection ID")
        if set(detection_ids) != set(plan.get("validated_cube_detection_ids", [])):
            raise ValueError("Defect picks do not match validated cube detections")
        observed_ids = {
            o.get("detection_id")
            for o in plan.get("observations", [])
            if o.get("category") == "cube"
            and o.get("eligible")
            and o.get("disposition") == "validated_defect_part"
        }
        if set(detection_ids) != observed_ids:
            raise ValueError("Defect picks contain an unvalidated detection ID")
        object_ids = [p.get("object_id") for p in picks]
        if len(object_ids) != 6 or None in object_ids or len(set(object_ids)) != 6:
            raise ValueError("Defect plan repeats an object ID")
        slots = [tuple(p.get("assigned_slot", [])) for p in picks]
        if any(len(slot) != 2 for slot in slots) or len(set(slots)) != 6:
            raise ValueError("Every defect cube requires a unique target slot")
        expected_slots = {
            tuple(slot) for colour in C.DEFECT_TARGET_SLOTS for slot in C.DEFECT_TARGET_SLOTS[colour]
        }
        if set(slots) != expected_slots:
            raise ValueError("Defect plan target slots differ from the audited slots")
        for p in picks:
            colour = p.get("colour")
            if colour not in C.DEFECT_CLASS_BY_COLOUR:
                raise ValueError("Unknown cube colour in defect plan")
            if p.get("defect_class") != C.DEFECT_CLASS_BY_COLOUR[colour]:
                raise ValueError("Wrong defect class for cube colour")
            if p.get("expected_colour") != colour or p.get(
                "expected_defect_class"
            ) != C.DEFECT_CLASS_BY_COLOUR[colour]:
                raise ValueError("Wrong expected colour/defect class metadata")
            if tuple(p["assigned_slot"]) not in {
                tuple(slot) for slot in C.DEFECT_TARGET_SLOTS[colour]
            } or not np.allclose(p.get("place_xy"), p["assigned_slot"]):
                raise ValueError("Cube is assigned to the wrong colour zone")
            if not cube_footprint_within_defect_zone(p["assigned_slot"], colour):
                raise ValueError("Assigned cube footprint is not contained in its zone")
            if float(p.get("classification_confidence", 0)) < C.CUBE_CLASSIFICATION_MIN_CONFIDENCE:
                raise ValueError("Cube classification confidence is below threshold")
            if float(p.get("pose_confidence", 0)) < C.CUBE_POSE_MIN_CONFIDENCE:
                raise ValueError("Cube pose confidence is below threshold")
            if float(p.get("pose_fit_iou", 0)) < C.CUBE_POSE_FIT_MIN_IOU:
                raise ValueError("Cube localization/pose fit is below threshold")
            if not np.isclose(p.get("pick_z", -1), C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE):
                raise ValueError("Invalid flat cube pick height")
            if not np.isclose(p.get("place_z", -1), C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE):
                raise ValueError("Invalid flat cube placement height")
            if not np.isclose(
                p.get("expected_centre_z", -1), C.SURFACE_DZ + 0.5 * C.CUBE_EDGE
            ):
                raise ValueError("Flat cube centre height must be half a cube edge")
        safety = plan.get("distractor_safety", {})
        if (
            not safety.get("unintended_lift_forbidden")
            or not np.isclose(safety.get("max_translation_m", -1), C.DISTRACTOR_TRANSLATION_LIMIT)
            or not np.isclose(safety.get("max_rotation_deg", -1), C.DISTRACTOR_ROTATION_LIMIT_DEG)
        ):
            raise ValueError("Defect plan has invalid distractor safety thresholds")
    if plan.get("task") == "cube_stack" and plan["picks"]:
        if [p.get("colour") for p in plan["picks"]] != list(C.CUBE_STACK_COLOURS):
            raise ValueError(
                "Tower must contain two yellow, two blue, then two red cubes"
            )
        if len({p["detection_id"] for p in plan["picks"]}) != len(C.CUBE_STACK_COLOURS):
            raise ValueError("Tower repeats a cube detection")
        for level, p in enumerate(plan["picks"]):
            yaw = float(p.get("yaw_deg", float("nan")))
            theta = np.deg2rad(yaw)
            expected = [-np.sin(theta), np.cos(theta), 0.0]
            spread = np.asarray(p.get("spread_dir", []), float)
            destination = np.asarray(p.get("place_spread_dir", []), float)
            if (
                not np.isfinite(yaw)
                or not np.isfinite(float(p.get("pose_confidence", 0)))
                or float(p.get("pose_confidence", 0)) < 0.8
                or spread.shape != (3,)
                or not np.allclose(spread, expected)
                or destination.shape != (3,)
                or not np.allclose(destination, [0.0, 1.0, 0.0])
            ):
                raise ValueError(
                    "Cube grasp requires a confident pose and aligned wrist"
                )
            if p["category"] != "cube" or p.get("stack_level") != level:
                raise ValueError("Invalid cube tower level")
            if not np.allclose(p["place_xy"], C.CUBE_STACK_XY) or not np.isclose(
                p["place_z"], C.SURFACE_DZ + level * C.CUBE_EDGE + C.CUBE_TIP_CLEARANCE
            ):
                raise ValueError("Invalid tower destination or placement height")
            if not np.isclose(p.get("pick_z", -1), C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE):
                raise ValueError(
                    "Cube grasp height differs from the current hooked-tip setup"
                )
            if not np.isclose(
                p.get("expected_centre_z", -1),
                C.SURFACE_DZ + (level + 0.5) * C.CUBE_EDGE,
            ):
                raise ValueError("Invalid expected tower height")
    for p in plan["picks"]:
        if p["category"] == "cap":
            if (
                p.get("orientation") != "narrow_end_up"
                or float(p.get("orientation_confidence", 0))
                < C.CAP_ORIENTATION_MIN_CONFIDENCE
            ):
                raise ValueError(
                    "Plan contains a cap without a confident narrow-end-up observation"
                )
            if not np.isclose(p["pick_z"], C.SURFACE_DZ + C.CAP_PICK_HEIGHT):
                raise ValueError("Cap pick height must clear the wide flange")
        if p["category"] == "vial":
            axis = np.array(p.get("axis_world", []), float)
            spread = np.array(p.get("spread_dir", []), float)
            if (
                axis.shape != (2,)
                or spread.shape != (3,)
                or not np.isclose(np.linalg.norm(spread), 1)
                or abs(np.dot(axis, spread[:2])) > 1e-5
            ):
                raise ValueError("Tube must have a perpendicular jaw closing direction")
        for key in ("xy", "place_xy"):
            if len(p[key]) != 2 or not np.all(np.isfinite(p[key])):
                raise ValueError("Invalid plan coordinates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        required=True,
        help="directory containing sort_plan.json and scene_state.json",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--camera", choices=["table", "side", "wrist"], default="table")
    ap.add_argument(
        "--physics-only",
        action="store_true",
        help="validate physical grasps without rendering frames",
    )
    a = ap.parse_args()
    if a.fps != 30:
        ap.error("Use 30 fps: two 1/60 s physics steps per video frame")
    run = Path(a.run).resolve()
    out = Path(a.out).resolve() if a.out else run / "frames"
    if out.exists() and any(out.glob("*.png")):
        raise SystemExit(
            f"Output already contains PNGs: {out}. Choose a fresh run directory."
        )
    plan = json.loads((run / "sort_plan.json").read_text())
    validate_plan(plan)
    if not plan["picks"]:
        raise SystemExit(
            "No eligible picks; inspect detections.json and sort_plan.json before rendering"
        )
    state = json.loads((run / "scene_state.json").read_text())
    # Resolve paths and validate before Kit changes the working directory.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import cv2
        import omni.usd
        from pxr import UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        import cell

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
        stage = omni.usd.get_context().get_stage()
        cell.build(stage)
        wrappers = {}
        for name, saved in state["objects"].items():
            prim = stage.GetPrimAtPath("/World/" + name)
            if not prim.IsValid():
                raise ValueError(f"Captured object missing: {name}")
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
            wrapper = SingleXFormPrim("/World/" + name)
            wrapper.set_world_pose(
                np.array(saved["position"]), np.array(saved["orientation"])
            )
            wrappers[name] = wrapper
        cam_paths = {
            "table": cell.TABLE_CAM_PATH,
            "side": cell.SIDE_CAM_PATH,
            "wrist": cell.WRIST_CAM_PATH,
        }
        configs = {"table": C.TABLE_CAM, "side": C.SIDE_CAM, "wrist": C.WRIST_CAM}
        cam = (
            None
            if a.physics_only
            else Camera(
                prim_path=cam_paths[a.camera],
                resolution=configs[a.camera]["resolution"],
            )
        )
        cell.enable_ccd(world)
        world.reset()
        if cam is not None:
            cam.initialize()
        arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
        arm.initialize()
        names = list(arm.dof_names)
        arm_idx = [names.index(f"J{i}") for i in range(1, 7)]
        idx = {name: i for i, name in enumerate(names)}
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        tool = tuple(np.array(C.TOOL_AXIS) * cell.tip_offset())
        home = np.array([0.30, 0.0, 0.24])
        down = [0.0, 0.0, -1.0]
        q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        q, ok = ik.solve(q, home, down, limits, tool_offset=tool)
        if not ok:
            raise RuntimeError("Home IK failed")

        def command(q6, jaw):
            full = arm.get_joint_positions().copy()
            full[arm_idx] = q6
            for joint, value in cell.jaw_targets(jaw).items():
                full[idx[joint]] = value
            arm.apply_action(ArticulationAction(joint_positions=full))

        full = arm.get_joint_positions().copy()
        full[arm_idx] = q
        for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
            full[idx[joint]] = value
        arm.set_joint_positions(full)
        for _ in range(120):
            command(q, C.JAW_OPEN)
            world.step(render=not a.physics_only)
        for name in wrappers:
            UsdPhysics.RigidBodyAPI(
                stage.GetPrimAtPath("/World/" + name)
            ).CreateKinematicEnabledAttr().Set(False)
        for _ in range(30):
            command(q, C.JAW_OPEN)
            world.step(render=not a.physics_only)

        def poses():
            values = {}
            for name, wrapper in wrappers.items():
                pos, quat = wrapper.get_world_pose()
                w, x, y, z = quat
                rot = np.array(
                    [
                        [
                            1 - 2 * (y * y + z * z),
                            2 * (x * y - z * w),
                            2 * (x * z + y * w),
                        ],
                        [
                            2 * (x * y + z * w),
                            1 - 2 * (x * x + z * z),
                            2 * (y * z - x * w),
                        ],
                        [
                            2 * (x * z - y * w),
                            2 * (y * z + x * w),
                            1 - 2 * (x * x + y * y),
                        ],
                    ]
                )
                centre = np.array(pos)
                if name.startswith("SortVial"):
                    centre += rot @ np.array([0, 0, C.VIAL_LENGTH / 2])
                elif name.startswith("SortCap"):
                    centre += rot @ np.array([0, 0, C.CAP_HEIGHT / 2])
                elif name.startswith("SortCube"):
                    centre += rot @ np.array([0, 0, C.CUBE_EDGE / 2])
                values[name] = {
                    "position": pos.tolist(),
                    "orientation": quat.tolist(),
                    "centre": centre.tolist(),
                    "yaw_deg": float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))),
                    "tilt_deg": float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1)))),
                }
            return values

        before = poses()
        events = []
        frame = 0
        abort_reason = None
        motion_time = 0.0
        wall_start = time.perf_counter()
        out.mkdir(parents=True, exist_ok=True)
        # Additional fixed side camera provides a clear lift/contact review.
        side = None
        if not a.physics_only and a.camera != "side":
            side = Camera(
                prim_path=cell.SIDE_CAM_PATH, resolution=C.SIDE_CAM["resolution"]
            )
            side.initialize()
        side_out = run / "side_frames"
        side_out.mkdir(exist_ok=True)
        for pick in plan["picks"]:
            # Select equivalent jaw-direction signs once for the whole grasp,
            # so approach/descend and carry/lower cannot differ by 180 degrees.
            pick = dict(pick)
            if pick.get("spread_dir") is not None:
                r, _ = fk.fk(q)
                current = r @ np.array(C.FINGER_SPREAD_AXIS)
                spread = np.array(pick["spread_dir"], float)
                if np.dot(current, spread) < 0:
                    spread = -spread
                pick["spread_dir"] = spread.tolist()
                if pick.get("place_spread_dir") is not None:
                    destination = np.array(pick["place_spread_dir"], float)
                    if np.dot(spread, destination) < 0:
                        destination = -destination
                    pick["place_spread_dir"] = destination.tolist()
            start = poses()
            prefix = {"cap": "SortCap", "cube": "SortCube", "vial": "SortVial"}[
                pick["category"]
            ]
            candidates = [name for name in wrappers if name.startswith(prefix)]
            # This association is for execution reporting only, never motion commands.
            tracked = (
                min(
                    candidates,
                    key=lambda name: np.linalg.norm(
                        np.array(start[name]["centre"][:2]) - pick["xy"]
                    ),
                )
                if candidates
                else None
            )
            if (
                tracked
                and np.linalg.norm(np.array(start[tracked]["centre"][:2]) - pick["xy"])
                > 0.04
            ):
                tracked = None
            event = {
                "detection_id": pick["detection_id"],
                "category": pick["category"],
                "tracked_object": tracked,
                "pickup_verified": False,
                "placement_verified": False,
                "trajectory_ik_failures": 0,
                "predicted_colour": pick.get("colour"),
                "expected_colour": None,
                "predicted_defect_class": pick.get("defect_class"),
                "expected_defect_class": None,
                "classification_correct": None,
                "position_error_mm": None,
                "yaw_error_deg_mod90": None,
                "placement_position_error_mm": None,
                "assigned_slot_error_mm": None,
                "correct_zone_containment": None,
                "correct_zone_placement": False,
                "motion_time_s": 0.0,
                "retries": [],
                "recoveries": [],
            }
            if "stack_level" in pick:
                event.update(
                    stack_level=pick["stack_level"],
                    colour=pick["colour"],
                    perceived_yaw_deg=pick["yaw_deg"],
                    grasp_spread_dir=pick["spread_dir"],
                    place_spread_dir=pick["place_spread_dir"],
                )
            event["initial_positions"] = start
            for name, p0, p1, duration, j0, j1, spread in segments(pick, home):
                motion_time += duration
                event["motion_time_s"] += duration
                spread_start = None
                if spread is not None and name in ("approach", "carry"):
                    r, _ = fk.fk(q)
                    s = r @ np.array(C.FINGER_SPREAD_AXIS)
                    s[2] = 0
                    s /= np.linalg.norm(s)
                    spread = np.array(spread, float)
                    spread_start = s
                n = round(duration * a.fps)
                for i in range(n):
                    t = ease((i + 1) / n)
                    target = p0 + (p1 - p0) * t
                    direction = (
                        turn(spread_start, spread, t)
                        if spread_start is not None
                        else spread
                    )
                    q, ok = ik.solve(
                        q, target, down, limits, tool_offset=tool, spread_dir=direction
                    )
                    if not ok:
                        q, ok = ik.solve(
                            q,
                            target,
                            down,
                            limits,
                            tool_offset=tool,
                            spread_dir=direction,
                            iters=500,
                            damping=0.02,
                        )
                        event["retries"].append(
                            {
                                "reason": f"primary_ik_failed:{name}",
                                "outcome": "recovered" if ok else "failed",
                            }
                        )
                        if ok:
                            event["recoveries"].append(
                                {
                                    "reason": f"fallback_ik:{name}",
                                    "outcome": "recovered",
                                }
                            )
                    r, actual = fk.tool_pose(q, tool)
                    aligned = (
                        direction is None
                        or abs(np.dot(r @ np.array(C.FINGER_SPREAD_AXIS), direction))
                        > 0.999
                    )
                    if not ok or np.linalg.norm(actual - target) > 0.001 or not aligned:
                        event["trajectory_ik_failures"] += 1
                        raise RuntimeError(
                            f'IK failed: detection {pick["detection_id"]}, {name}, frame {i}; refusing off-target descent'
                        )
                    command(q, j0 + (j1 - j0) * t)
                    for _ in range(2):
                        world.step(render=not a.physics_only)
                    if cam is not None:
                        rgba = cam.get_rgba()
                        if rgba is None:
                            raise RuntimeError("Camera returned no image")
                        cv2.imwrite(
                            str(out / f"f{frame:05d}.png"),
                            cv2.cvtColor(
                                rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
                            ),
                        )
                    if side is not None:
                        rgba_side = side.get_rgba()
                        if rgba_side is not None:
                            cv2.imwrite(
                                str(side_out / f"f{frame:05d}.png"),
                                cv2.cvtColor(
                                    rgba_side[:, :, :3].astype(np.uint8),
                                    cv2.COLOR_RGB2BGR,
                                ),
                            )
                    frame += 1
                if name in ("descend", "dwell", "lift", "lower", "release"):
                    measured = arm.get_joint_positions()
                    rot, tip = fk.tool_pose(measured[arm_idx], tool)
                    measured_spread = rot @ np.array(C.FINGER_SPREAD_AXIS)
                    event.setdefault("phase_measurements", {})[name] = {
                        "actual_tip_robot_frame_m": tip.tolist(),
                        "actual_spread_dir": measured_spread.tolist(),
                        "commanded_spread_dir": (
                            np.asarray(direction).tolist()
                            if direction is not None
                            else None
                        ),
                        "wrist_joint_deg": float(np.degrees(measured[arm_idx][-1])),
                        "commanded_tip_robot_frame_m": target.tolist(),
                        "tip_error_mm": float(np.linalg.norm(tip - target) * 1000),
                        "object_centre_world_m": (
                            poses()[tracked]["centre"] if tracked else None
                        ),
                    }
                if name == "dwell":
                    jp = arm.get_joint_positions()
                    event["jaw_gap_mm"] = float(
                        (2 * C.JAW_OPEN - abs(jp[idx["jaw_l"]]) - abs(jp[idx["jaw_r"]]))
                        * 1000
                    )
                if name == "lift":
                    lifted = poses()
                    event["distractor_lifts"] = [
                        key
                        for key in wrappers
                        if (key.startswith("SortCap") or key.startswith("SortVial"))
                        and lifted[key]["centre"][2] - start[key]["centre"][2]
                        > 1e-6
                    ]
                    event["objects_lifted"] = [
                        key
                        for key in wrappers
                        if lifted[key]["centre"][2] - start[key]["centre"][2] > 0.025
                    ]
                    event["pickup_verified"] = tracked in event["objects_lifted"]
                    print(
                        f'[sort] detection {pick["detection_id"]}: lifted {event["objects_lifted"]}; pickup={event["pickup_verified"]}',
                        flush=True,
                    )
            finish = poses()
            if tracked:
                position = np.array(finish[tracked]["centre"])
                event["placement_error_mm"] = float(
                    np.linalg.norm(position[:2] - pick["place_xy"]) * 1000
                )
                expected_z = (
                    C.TABLE_TOP_Z
                    + C.SURFACE_DZ
                    + {
                        "cube": 0.5 * C.CUBE_EDGE,
                        "cap": C.CAP_HEIGHT / 2,
                        "vial": C.VIAL_RADIUS,
                    }[pick["category"]]
                )
                if "stack_level" in pick:
                    expected_z = C.TABLE_TOP_Z + pick["expected_centre_z"]
                event["placement_verified"] = bool(
                    event["pickup_verified"]
                    and event["placement_error_mm"] < 20
                    and abs(position[2] - expected_z) < 0.01
                )
                if "stack_level" in pick:
                    event["placement_verified"] = bool(
                        event["pickup_verified"]
                        and event["placement_error_mm"]
                        < C.CUBE_STACK_SITE_TOLERANCE * 1000
                        and abs(position[2] - expected_z) < 0.0015
                    )
                    event["placement_yaw_error_deg_mod90"] = abs(
                        square_yaw(finish[tracked]["yaw_deg"])
                    )
                    event["placement_tilt_deg"] = finish[tracked]["tilt_deg"]
                    event["placement_pose_verified"] = bool(
                        event["placement_yaw_error_deg_mod90"] < 8
                        and event["placement_tilt_deg"] < 5
                    )
                    event["placement_verified"] = (
                        event["placement_verified"] and event["placement_pose_verified"]
                    )
                    event["stack_support_verified"] = verify_cube_stack(
                        finish, events + [event]
                    )
                elif plan.get("task") == "defect_sort":
                    placement = evaluate_defect_placement(
                        pick, position[:2], finish[tracked].get("yaw_deg", 0)
                    )
                    event.update(placement)
                    event["placement_verified"] = bool(
                        event["pickup_verified"]
                        and placement["correct_zone_placement"]
                        and abs(position[2] - expected_z) < 0.0015
                    )
            event["retry_count"] = len(event["retries"])
            event["recovery_count"] = len(event["recoveries"])
            events.append(event)
            (run / "execution_events.json").write_text(
                json.dumps(events, indent=2) + "\n"
            )
            if plan.get("task") == "cube_stack" and (
                not event["placement_verified"]
                or not event.get("stack_support_verified")
            ):
                abort_reason = f'Cube tower stopped at level {pick["stack_level"]}: pickup, placement or supporting stack failed'
                print(f"[sort] {abort_reason}", flush=True)
                break
        after = poses()
        gt = json.loads((run / "ground_truth.json").read_text())
        gt_by_name = {row["name"]: row for row in gt.get("objects", [])}
        for event, pick in zip(events, plan["picks"]):
            actual = gt_by_name.get(event.get("tracked_object"))
            if not actual or event.get("category") != "cube":
                continue
            expected_colour = actual.get("colour")
            event["expected_colour"] = expected_colour
            event["expected_defect_class"] = C.DEFECT_CLASS_BY_COLOUR.get(
                expected_colour
            )
            event["classification_correct"] = bool(
                event.get("predicted_colour") == expected_colour
                and event.get("predicted_defect_class")
                == event.get("expected_defect_class")
            )
            event["position_error_mm"] = float(
                np.linalg.norm(np.asarray(pick["xy"]) - actual["xyz"][:2]) * 1000
            )
            if "yaw_deg" in actual and "yaw_deg" in pick:
                event["yaw_error_deg_mod90"] = abs(
                    square_yaw(pick["yaw_deg"] - actual["yaw_deg"])
                )
        distractor_names = {
            name
            for name in wrappers
            if name.startswith("SortCap") or name.startswith("SortVial")
        }
        distractor_safety = evaluate_distractor_safety(
            before, after, events, distractor_names
        )
        defect_summary = build_defect_report(
            plan,
            events,
            motion_time,
            time.perf_counter() - wall_start,
            distractor_safety,
        )
        (run / "execution_events.json").write_text(
            json.dumps(events, indent=2) + "\n"
        )
        report = {
            "before": before,
            "after": after,
            "picks": events,
            "frames": frame,
            "rendered": not a.physics_only,
            "task": plan.get("task", "sort"),
            "abort_reason": abort_reason,
            "hardware": {
                "gripper_model": C.GRIPPER_MODEL,
                "jaw_stroke_m": C.JAW_STROKE,
                "open_gap_m": 2 * C.JAW_OPEN,
                "finger_length_m": C.FINGER_LENGTH,
                "cube_edge_m": C.CUBE_EDGE,
                "cube_tip_clearance_m": C.CUBE_TIP_CLEARANCE,
            },
            "task_complete": bool(plan["picks"])
            and abort_reason is None
            and len(events) == len(plan["picks"])
            and all(e["placement_verified"] for e in events),
            "verified_pickups": sum(e["pickup_verified"] for e in events),
            "verified_placements": sum(e["placement_verified"] for e in events),
        }
        if plan.get("task") == "defect_sort":
            report.update(defect_summary)
            report["task_complete"] = bool(
                report["task_complete"]
                and report["successful_pickups"] == 6
                and report["correct_zone_placements"] == 6
                and report["distractor_safety_passed"]
            )
        # Evaluate untouched wide-up objects only after executing the vision plan.
        wide = [
            g["name"] for g in gt["objects"] if g.get("orientation") == "wide_end_up"
        ]
        report["wide_caps"] = [
            {
                "name": name,
                "displacement_mm": float(
                    np.linalg.norm(
                        np.array(after[name]["centre"]) - before[name]["centre"]
                    )
                    * 1000
                ),
                "picked_up": any(name in e.get("objects_lifted", []) for e in events),
            }
            for name in wide
        ]
        (run / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f'[sort] {report["verified_pickups"]}/{len(events)} pickups; {report["verified_placements"]} verified placements; {frame} frames',
            flush=True,
        )
    except Exception:
        # Kit shutdown may terminate Python before the exception is displayed.
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
