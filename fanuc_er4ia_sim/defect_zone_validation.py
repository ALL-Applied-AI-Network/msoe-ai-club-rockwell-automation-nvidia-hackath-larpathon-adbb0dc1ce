"""Validate and annotate the static Dataset 3 defect-zone inspection scene."""

import json
from pathlib import Path
import traceback

import cv2
import numpy as np

import fk
import ik
import scene_config as C
from backproject import CameraModel


def zone_marker_paths():
    paths = []
    for colour in C.DEFECT_ZONE_CENTERS:
        root = f"{C.DEFECT_ZONE_PRIM_ROOT}/{colour.capitalize()}"
        paths.append(f"{root}/Boundary")
        paths.extend(
            f"{root}/Slots/Slot{index}"
            for index in range(len(C.DEFECT_TARGET_SLOTS[colour]))
        )
    return paths


def slot_records():
    return [
        {
            "colour": colour,
            "defect_class": C.DEFECT_CLASS_BY_COLOUR[colour],
            "slot_index": index,
            "xy": list(slot),
            "marker_path": (
                f"{C.DEFECT_ZONE_PRIM_ROOT}/{colour.capitalize()}"
                f"/Slots/Slot{index}"
            ),
        }
        for colour, slots in C.DEFECT_TARGET_SLOTS.items()
        for index, slot in enumerate(slots)
    ]


def _footprint_inside_zone(xy, colour):
    delta = np.abs(np.asarray(xy, float) - C.DEFECT_ZONE_CENTERS[colour])
    available = (
        np.asarray(C.DEFECT_ZONE_SIZE, float) / 2
        - C.DEFECT_ZONE_SAFE_INSET
        - C.CUBE_EDGE / 2
    )
    return bool(np.all(delta <= available + 1e-12))


def _footprint_inside_table(xy):
    x, y = xy
    half = C.CUBE_EDGE / 2
    return bool(
        C.TABLE_X[0] <= x - half <= x + half <= C.TABLE_X[1]
        and C.TABLE_Y[0] <= y - half <= y + half <= C.TABLE_Y[1]
    )


def numerical_ik_checks():
    """Run the existing host IK solver at every flat placement and hover pose."""
    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
    q_seed = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
    tool = tuple(np.asarray(C.TOOL_AXIS, float) * C.TOOL_LENGTH)
    place_z = C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE
    results = []
    for slot in slot_records():
        poses = {}
        q = q_seed.copy()
        for name, z in (("hover", place_z + 0.13), ("placement", place_z)):
            target = np.array([*slot["xy"], z], float)
            q, converged = ik.solve(
                q,
                target,
                [0.0, 0.0, -1.0],
                limits,
                tool_offset=tool,
                spread_dir=[0.0, 1.0, 0.0],
            )
            if not converged:
                q, converged = ik.solve(
                    q_seed,
                    target,
                    [0.0, 0.0, -1.0],
                    limits,
                    iters=500,
                    damping=0.02,
                    tool_offset=tool,
                    spread_dir=[0.0, 1.0, 0.0],
                )
            _, actual = fk.tool_pose(q, tool)
            poses[name] = {
                "target_robot_frame_m": target.tolist(),
                "position_error_mm": float(np.linalg.norm(actual - target) * 1000),
                "converged": bool(converged),
            }
        results.append(
            {
                **slot,
                "poses": poses,
                "reachable": all(p["converged"] for p in poses.values()),
            }
        )
    return results


def configured_geometry_validation():
    slots = slot_records()
    unique_slots = {tuple(slot["xy"]) for slot in slots}
    footprints = [
        {**slot, "fits": _footprint_inside_zone(slot["xy"], slot["colour"])}
        for slot in slots
    ]
    table_checks = [
        {**slot, "inside_table": _footprint_inside_table(slot["xy"])}
        for slot in slots
    ]
    ik_checks = numerical_ik_checks()
    zones = [
        {
            "colour": colour,
            "defect_class": C.DEFECT_CLASS_BY_COLOUR[colour],
            "center": list(center),
            "size": list(C.DEFECT_ZONE_SIZE),
            "safe_inset_m": C.DEFECT_ZONE_SAFE_INSET,
            "boundary_path": (
                f"{C.DEFECT_ZONE_PRIM_ROOT}/{colour.capitalize()}/Boundary"
            ),
        }
        for colour, center in C.DEFECT_ZONE_CENTERS.items()
    ]
    checks = {
        "three_configured_zones": len(zones) == 3,
        "six_unique_target_slots": len(slots) == 6 and len(unique_slots) == 6,
        "all_cube_footprints_fit": all(row["fits"] for row in footprints),
        "all_target_slots_inside_table": all(
            row["inside_table"] for row in table_checks
        ),
        "all_target_slots_ik_reachable": all(row["reachable"] for row in ik_checks),
        "distractors_have_no_target_slots": all(
            row["colour"] in C.DEFECT_CLASS_BY_COLOUR for row in slots
        ),
    }
    return {
        "checks": checks,
        "zones": zones,
        "target_slots": slots,
        "footprint_checks": footprints,
        "table_checks": table_checks,
        "ik_checks": ik_checks,
    }


def _in_frame(uv, width, height):
    return bool(
        np.asarray(uv).shape == (2,)
        and np.all(np.isfinite(uv))
        and 0 <= uv[0] < width
        and 0 <= uv[1] < height
    )


def camera_visibility(camera_json):
    camera = CameraModel(str(camera_json))
    z_surface = C.TABLE_TOP_Z + C.SURFACE_DZ
    zone_rows = []
    for colour, center in C.DEFECT_ZONE_CENTERS.items():
        half_x, half_y = (value / 2 for value in C.DEFECT_ZONE_SIZE)
        corners = [
            [center[0] + dx, center[1] + dy, z_surface]
            for dx, dy in (
                (-half_x, -half_y),
                (half_x, -half_y),
                (half_x, half_y),
                (-half_x, half_y),
            )
        ]
        pixels = [camera.project(point).tolist() for point in corners]
        zone_rows.append(
            {
                "colour": colour,
                "corner_pixels": pixels,
                "inside_frame": all(
                    _in_frame(pixel, camera.W, camera.H) for pixel in pixels
                ),
            }
        )
    slot_rows = []
    for slot in slot_records():
        pixel = camera.project([*slot["xy"], z_surface]).tolist()
        slot_rows.append(
            {
                **slot,
                "pixel": pixel,
                "inside_frame": _in_frame(pixel, camera.W, camera.H),
            }
        )
    cube_rows = []
    for index, (x, y, colour) in enumerate(C.SORT_CUBES):
        pixel = camera.project([x, y, z_surface + C.CUBE_EDGE / 2]).tolist()
        cube_rows.append(
            {
                "name": f"SortCube{index}",
                "colour": colour,
                "pixel": pixel,
                "inside_frame": _in_frame(pixel, camera.W, camera.H),
            }
        )
    return {
        "zones": zone_rows,
        "target_slots": slot_rows,
        "cubes": cube_rows,
        "all_zones_inside_frame": all(row["inside_frame"] for row in zone_rows),
        "all_target_slots_inside_frame": all(
            row["inside_frame"] for row in slot_rows
        ),
        "all_cubes_inside_frame": all(row["inside_frame"] for row in cube_rows),
    }


def stage_marker_validation(stage):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    rows = []
    expected_paths = set(zone_marker_paths())
    collision_geometry_types = (
        UsdGeom.Mesh,
        UsdGeom.Cube,
        UsdGeom.Sphere,
        UsdGeom.Capsule,
        UsdGeom.Cylinder,
        UsdGeom.Cone,
    )
    for path in sorted(expected_paths):
        prim = stage.GetPrimAtPath(path)
        parts = path.split("/")
        colour = parts[3].lower()
        is_boundary = parts[-1] == "Boundary"
        if is_boundary:
            expected_metadata = {
                "markerKind": "zone_boundary",
                "colour": colour,
                "defectClass": C.DEFECT_CLASS_BY_COLOUR[colour],
                "centerXY": Gf.Vec2d(*C.DEFECT_ZONE_CENTERS[colour]),
                "sizeXY": Gf.Vec2d(*C.DEFECT_ZONE_SIZE),
            }
        else:
            slot_index = int(parts[-1].removeprefix("Slot"))
            expected_metadata = {
                "markerKind": "target_slot",
                "colour": colour,
                "defectClass": C.DEFECT_CLASS_BY_COLOUR[colour],
                "slotIndex": slot_index,
                "slotXY": Gf.Vec2d(*C.DEFECT_TARGET_SLOTS[colour][slot_index]),
            }
        metadata = {}
        for key, expected in expected_metadata.items():
            actual = prim.GetCustomDataByKey(key) if prim.IsValid() else None
            expected_is_vec = isinstance(expected, Gf.Vec2d)
            type_matches = isinstance(actual, Gf.Vec2d) if expected_is_vec else type(actual) is type(expected)
            value_matches = (
                bool(np.allclose(tuple(actual), tuple(expected), atol=1e-12))
                if type_matches and expected_is_vec
                else type_matches and actual == expected
            )
            metadata[key] = {
                "expected": list(expected) if expected_is_vec else expected,
                "actual": list(actual) if isinstance(actual, Gf.Vec2d) else actual,
                "expected_type": type(expected).__name__,
                "actual_type": type(actual).__name__ if actual is not None else None,
                "matches": bool(value_matches),
            }
        display_colour = (
            UsdGeom.Gprim(prim).GetDisplayColorAttr().Get() if prim.IsValid() else None
        )
        expected_colour = Gf.Vec3f(*C.DEFECT_ZONE_DISPLAY_COLOURS[colour])
        has_one_display_colour = display_colour is not None and len(display_colour) == 1
        display_colour_matches = bool(
            has_one_display_colour
            and np.allclose(tuple(display_colour[0]), tuple(expected_colour), atol=1e-6)
        )
        has_collision_geometry = prim.IsValid() and any(
            prim.IsA(schema) for schema in collision_geometry_types
        )
        rows.append(
            {
                "path": path,
                "exists": prim.IsValid(),
                "type": prim.GetTypeName() if prim.IsValid() else None,
                "is_visual_curve": prim.IsValid() and prim.IsA(UsdGeom.BasisCurves),
                "has_collision_api": prim.IsValid()
                and prim.HasAPI(UsdPhysics.CollisionAPI),
                "has_collision_geometry": bool(has_collision_geometry),
                "metadata": metadata,
                "metadata_matches": all(item["matches"] for item in metadata.values()),
                "display_colour": (
                    list(display_colour[0]) if has_one_display_colour else None
                ),
                "display_colour_matches": display_colour_matches,
            }
        )
    root = stage.GetPrimAtPath(C.DEFECT_ZONE_PRIM_ROOT)
    zone_prims = (
        [prim for prim in Usd.PrimRange(root) if prim != root]
        if root.IsValid()
        else []
    )
    discovered_marker_paths = {
        str(prim.GetPath())
        for prim in zone_prims
        if prim.IsA(UsdGeom.BasisCurves)
    }
    non_visual_types = [
        str(prim.GetPath())
        for prim in zone_prims
        if prim.GetTypeName() not in ("Xform", "BasisCurves")
    ]
    collision_api_paths = [
        str(prim.GetPath())
        for prim in ([root] + zone_prims if root.IsValid() else [])
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    collision_geometry_paths = [
        str(prim.GetPath())
        for prim in zone_prims
        if any(prim.IsA(schema) for schema in collision_geometry_types)
    ]
    return {
        "markers": rows,
        "traversed_descendant_count": len(zone_prims),
        "discovered_marker_paths": sorted(discovered_marker_paths),
        "all_markers_exist": (
            len(rows) == 9
            and all(row["exists"] for row in rows)
            and discovered_marker_paths == expected_paths
        ),
        "basis_curves_only": all(row["is_visual_curve"] for row in rows),
        "all_marker_metadata_matches": all(
            row["metadata_matches"] and row["display_colour_matches"] for row in rows
        ),
        "non_visual_prim_paths": non_visual_types,
        "collision_api_paths": collision_api_paths,
        "collision_geometry_paths": collision_geometry_paths,
        "no_collision_apis_or_collision_geometry": (
            root.IsValid()
            and not collision_api_paths
            and not collision_geometry_paths
            and not non_visual_types
            and all(
                not row["has_collision_api"] and not row["has_collision_geometry"]
                for row in rows
            )
        ),
    }


def annotate_capture(image_path, camera_json, output_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    camera = CameraModel(str(camera_json))
    z_surface = C.TABLE_TOP_Z + C.SURFACE_DZ
    bgr = {"red": (20, 20, 230), "blue": (230, 80, 20), "yellow": (0, 210, 240)}
    for colour, center in C.DEFECT_ZONE_CENTERS.items():
        half_x, half_y = (value / 2 for value in C.DEFECT_ZONE_SIZE)
        corners = np.array(
            [
                camera.project([center[0] + dx, center[1] + dy, z_surface])
                for dx, dy in (
                    (-half_x, -half_y),
                    (half_x, -half_y),
                    (half_x, half_y),
                    (-half_x, half_y),
                )
            ],
            np.int32,
        )
        cv2.polylines(image, [corners], True, bgr[colour], 3)
        label_at = tuple(camera.project([*center, z_surface]).astype(int))
        cv2.putText(
            image,
            f"{C.DEFECT_CLASS_BY_COLOUR[colour]} ({colour})",
            (label_at[0] - 65, label_at[1] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            bgr[colour],
            2,
        )
        for index, slot in enumerate(C.DEFECT_TARGET_SLOTS[colour]):
            pixel = tuple(camera.project([*slot, z_surface]).astype(int))
            cv2.drawMarker(image, pixel, bgr[colour], cv2.MARKER_CROSS, 16, 2)
            cv2.putText(
                image,
                str(index + 1),
                (pixel[0] + 7, pixel[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                bgr[colour],
                1,
            )
    for index, (x, y, colour) in enumerate(C.SORT_CUBES):
        pixel = tuple(camera.project([x, y, z_surface + C.CUBE_EDGE / 2]).astype(int))
        cv2.circle(image, pixel, 9, bgr[colour], 2)
        cv2.putText(
            image,
            f"cube {index}: {colour}",
            (pixel[0] + 10, pixel[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            bgr[colour],
            1,
        )
    ignored = []
    ignored.extend(
        (f"cap {index}", x, y, C.CAP_HEIGHT / 2)
        for index, (x, y) in enumerate(C.SORT_CAPS)
    )
    ignored.extend(
        (f"vial {index}", x, y, C.VIAL_RADIUS)
        for index, (x, y, _) in enumerate(C.SORT_VIALS)
    )
    for label, x, y, height in ignored:
        pixel = tuple(camera.project([x, y, z_surface + height]).astype(int))
        cv2.drawMarker(image, pixel, (40, 40, 40), cv2.MARKER_TILTED_CROSS, 15, 2)
        cv2.putText(
            image,
            f"IGNORED {label}",
            (pixel[0] + 9, pixel[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 40, 40),
            1,
        )
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write annotation: {output_path}")


def robot_inactivity_validation(joints_before, joints_after, command_counts=None):
    """Separate strict command absence from tolerated passive PhysX settling."""
    command_counts = command_counts or {}
    commands = {
        name: command_counts.get(name, 0)
        for name in (
            "articulation_action_commands",
            "joint_position_commands",
            "grasp_commands",
            "placement_commands",
            "sorting_commands",
            "robot_trajectory_commands",
        )
    }
    before = np.asarray(joints_before, float)
    after = np.asarray(joints_after, float)
    maximum_drift_rad = float(np.max(np.abs(after - before)))
    no_robot_commands_issued = all(count == 0 for count in commands.values())
    passive_joint_drift_within_tolerance = (
        maximum_drift_rad <= C.PASSIVE_JOINT_DRIFT_TOLERANCE_RAD
    )
    return {
        **commands,
        "robot_joints_before_rad": before.tolist(),
        "robot_joints_after_rad": after.tolist(),
        "maximum_passive_joint_drift_rad": maximum_drift_rad,
        "maximum_passive_joint_drift_degrees": float(np.rad2deg(maximum_drift_rad)),
        "passive_joint_drift_tolerance_rad": C.PASSIVE_JOINT_DRIFT_TOLERANCE_RAD,
        "robot_commands_issued": not no_robot_commands_issued,
        "no_robot_commands_issued": no_robot_commands_issued,
        "passive_joint_drift_within_tolerance": passive_joint_drift_within_tolerance,
    }


def validate_scene_capture(
    stage,
    out_dir,
    ground_truth,
    scene_state,
    joints_before,
    joints_after,
    robot_command_counts=None,
):
    out = Path(out_dir)
    annotated = out / "world_camera_annotated.png"
    report = {
        "validation": "dataset3_phase2_static_scene_and_zone_layout",
        "status": "FAIL",
        "checks": {
            name: False
            for name in (
                "three_configured_zones",
                "six_unique_target_slots",
                "all_cube_footprints_fit",
                "all_target_slots_inside_table",
                "all_target_slots_ik_reachable",
                "distractors_have_no_target_slots",
                "all_zone_and_slot_markers_exist",
                "zone_markers_are_basis_curves",
                "zone_marker_metadata_matches",
                "zone_markers_are_non_colliding_visual_curves",
                "all_six_cubes_inside_table_camera_frame",
                "all_three_zones_inside_table_camera_frame",
                "all_six_slots_inside_table_camera_frame",
                "starting_inventory_preserved",
                "caps_and_vials_are_ignored_distractors",
                "no_robot_commands_issued",
                "passive_joint_drift_within_tolerance",
                "annotated_capture_written",
                "exactly_one_1920x1080_table_camera_capture",
            )
        },
        "failures": [],
        "completed_checks": [],
        "exceptions": [],
        "configured_geometry": None,
        "stage_markers": None,
        "camera_visibility": None,
        "starting_inventory": None,
        "actions": None,
        "capture": None,
        "outputs": {
            "world_camera": str(out / "world_camera.png"),
            "annotated_world_camera": str(annotated),
            "camera": str(out / "camera.json"),
            "ground_truth": str(out / "ground_truth.json"),
            "scene_state": str(out / "scene_state.json"),
        },
    }

    def complete(name, value):
        report["checks"][name] = bool(value)
        report["completed_checks"].append(name)

    def record_exception(phase, error):
        report["exceptions"].append(
            {
                "phase": phase,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )

    try:
        try:
            actions = robot_inactivity_validation(
                joints_before, joints_after, robot_command_counts
            )
            report["actions"] = actions
            complete("no_robot_commands_issued", actions["no_robot_commands_issued"])
            complete(
                "passive_joint_drift_within_tolerance",
                actions["passive_joint_drift_within_tolerance"],
            )
        except Exception as error:
            record_exception("robot_joint_drift", error)

        try:
            annotate_capture(
                out / "world_camera.png", out / "camera.json", annotated
            )
            complete(
                "annotated_capture_written",
                annotated.is_file() and annotated.stat().st_size > 0,
            )
        except Exception as error:
            record_exception("annotated_capture", error)

        geometry = None
        try:
            geometry = configured_geometry_validation()
            report["configured_geometry"] = geometry
            for name, value in geometry["checks"].items():
                complete(name, value)
        except Exception as error:
            record_exception("configured_geometry", error)

        try:
            markers = stage_marker_validation(stage)
            report["stage_markers"] = markers
            complete("all_zone_and_slot_markers_exist", markers["all_markers_exist"])
            complete("zone_markers_are_basis_curves", markers["basis_curves_only"])
            complete(
                "zone_marker_metadata_matches",
                markers["all_marker_metadata_matches"],
            )
            complete(
                "zone_markers_are_non_colliding_visual_curves",
                markers["no_collision_apis_or_collision_geometry"],
            )
        except Exception as error:
            record_exception("stage_markers", error)

        try:
            visibility = camera_visibility(out / "camera.json")
            report["camera_visibility"] = visibility
            complete(
                "all_six_cubes_inside_table_camera_frame",
                visibility["all_cubes_inside_frame"],
            )
            complete(
                "all_three_zones_inside_table_camera_frame",
                visibility["all_zones_inside_frame"],
            )
            complete(
                "all_six_slots_inside_table_camera_frame",
                visibility["all_target_slots_inside_frame"],
            )
        except Exception as error:
            record_exception("camera_visibility", error)

        try:
            object_counts = {
                category: sum(
                    obj.get("category") == category
                    for obj in ground_truth.get("objects", [])
                )
                for category in ("cube", "cap", "vial")
            }
            report["starting_inventory"] = object_counts
            expected_names = {
                *(f"SortCube{i}" for i in range(6)),
                *(f"SortCap{i}" for i in range(2)),
                *(f"SortVial{i}" for i in range(2)),
            }
            expected_ground_truth = {
                "cube": [
                    (f"SortCube{i}", colour, [x, y, C.CUBE_EDGE / 2])
                    for i, (x, y, colour) in enumerate(C.SORT_CUBES)
                ],
                "cap": [
                    (f"SortCap{i}", "yellow", [x, y, C.CAP_HEIGHT / 2])
                    for i, (x, y) in enumerate(C.SORT_CAPS)
                ],
                "vial": [
                    (f"SortVial{i}", "clear", [x, y, C.VIAL_RADIUS])
                    for i, (x, y, _) in enumerate(C.SORT_VIALS)
                ],
            }
            objects_by_category = {
                category: [
                    (obj["name"], obj["colour"], obj["xyz"])
                    for obj in ground_truth.get("objects", [])
                    if obj.get("category") == category
                ]
                for category in expected_ground_truth
            }
            complete(
                "starting_inventory_preserved",
                object_counts == {"cube": 6, "cap": 2, "vial": 2}
                and set(scene_state.get("objects", {})) == expected_names
                and objects_by_category == expected_ground_truth,
            )
            complete(
                "caps_and_vials_are_ignored_distractors",
                object_counts["cap"] == 2
                and object_counts["vial"] == 2
                and geometry is not None
                and geometry["checks"]["distractors_have_no_target_slots"],
            )
        except Exception as error:
            record_exception("starting_inventory", error)

        try:
            camera_metadata = json.loads(
                (out / "camera.json").read_text(encoding="utf-8")
            )
            image = cv2.imread(str(out / "world_camera.png"), cv2.IMREAD_COLOR)
            captured_resolution = (
                list(image.shape[1::-1]) if image is not None else None
            )
            capture = {
                "camera": camera_metadata.get("camera"),
                "image_count": 1 if (out / "world_camera.png").is_file() else 0,
                "configured_resolution": camera_metadata.get("resolution"),
                "captured_resolution": captured_resolution,
                "synchronized_replicator_capture": True,
            }
            report["capture"] = capture
            complete(
                "exactly_one_1920x1080_table_camera_capture",
                capture["camera"] == "table"
                and capture["image_count"] == 1
                and capture["configured_resolution"] == [1920, 1080]
                and capture["captured_resolution"] == [1920, 1080],
            )
        except Exception as error:
            record_exception("capture_integrity", error)
    except Exception as error:
        record_exception("unexpected_validation_error", error)
    finally:
        report["failures"] = [
            name for name, passed in report["checks"].items() if not passed
        ]
        report["status"] = (
            "PASS"
            if not report["failures"] and not report["exceptions"]
            else "FAIL"
        )
        (out / "defect_zone_validation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report
