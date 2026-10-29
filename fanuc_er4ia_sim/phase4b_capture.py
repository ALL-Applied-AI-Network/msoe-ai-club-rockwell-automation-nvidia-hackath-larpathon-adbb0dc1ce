"""Capture the fixed Phase 4B randomized batch in one Isaac SimulationApp.

The module is host-importable: every simulator import is deferred until main().
No articulation is initialized or commanded.  Images and evaluator truth are
written only after the shared ten-object stability gate passes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from oracle_motion_smoke import (
    INITIAL_PHYSICS_SETTLING_S,
    MAXIMUM_TOTAL_SETTLING_S,
    STABILITY_WINDOW_S,
    evaluate_scene_stability_windows,
)
from phase4b_design import (
    MAX_GENERATION_ATTEMPTS,
    PHASE_VERSION,
    RESOLUTION,
    SEEDS,
    capsule_yaw_deg,
    capture_is_authorized,
    derive_attempt_seed,
    frozen_detector_record,
    generate_batch_design,
    generate_scene,
    generation_attempt_record,
    geometry_report,
    proposed_yaad_request,
    record_failed_seed,
    resource_estimate,
    runtime_manifest,
    validate_scene_design,
)
from perception_geometry import decorate_camera_calibration, simulation_source_workspace

MAX_SYNCHRONIZED_CAPTURE_ATTEMPTS = 5
SOURCE_DESIGN_RESOLUTION = tuple(RESOLUTION)
CLEANUP_ORDER = (
    "detach_annotators",
    "destroy_render_product",
    "stop_reset_replicator",
    "release_camera",
    "clear_world",
    "replace_stage",
)


def diagnostic_marker(event: str, **fields) -> None:
    """Write an opt-in, flushed and fsynced capture progress marker."""
    path = os.environ.get("PHASE4C_DIAGNOSTIC_PROGRESS")
    if not path:
        return
    record = {
        "event": str(event),
        "utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    line = json.dumps(record, sort_keys=True)
    print(f"[phase4c-diagnostic] {line}", flush=True)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _configure_scene_constants(scene, C) -> None:
    cubes = [obj for obj in scene["objects"] if obj["category"] == "cube"]
    caps = [obj for obj in scene["objects"] if obj["category"] == "cap"]
    vials = [obj for obj in scene["objects"] if obj["category"] == "vial"]
    C.SORT_CUBES = [(*obj["xy_m"], obj["colour"]) for obj in cubes]
    C.SORT_CUBE_YAWS_DEG = [obj["yaw_deg"] for obj in cubes]
    C.SORT_CAPS = [(*obj["xy_m"], obj["yaw_deg"]) for obj in caps]
    C.CAP_ORIENTATIONS = ["narrow_end_up"] * 2
    C.SORT_VIALS = [(*obj["xy_m"], obj["yaw_deg"]) for obj in vials]
    C.TABLE_CAM = dict(C.TABLE_CAM, resolution=RESOLUTION)


def _rotation_matrix(quaternion) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _pose_from_world(category, position, orientation, C):
    position = np.asarray(position, float)
    orientation = np.asarray(orientation, float)
    rotation = _rotation_matrix(orientation)
    centre = position.copy()
    if category == "vial":
        centre += rotation @ np.array([0, 0, C.VIAL_LENGTH / 2])
        yaw = capsule_yaw_deg(rotation)
    elif category == "cap":
        centre += rotation @ np.array([0, 0, C.CAP_HEIGHT / 2])
        yaw = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    else:
        centre += rotation @ np.array([0, 0, C.CUBE_EDGE / 2])
        yaw = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return {
        "position": position.tolist(),
        "orientation": orientation.tolist(),
        "centre": centre.tolist(),
        "yaw_deg": yaw,
        "tilt_deg": float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
    }


def _pose_sampler(wrappers, categories, C):
    def poses():
        return {
            name: _pose_from_world(categories[name], *wrapper.get_world_pose(), C)
            for name, wrapper in wrappers.items()
        }
    return poses


def _scene_from_poses(scene, poses):
    objects = []
    for source in scene["objects"]:
        pose = poses[source["name"]]
        item = dict(source)
        item["xy_m"] = pose["centre"][:2]
        item["yaw_deg"] = pose["yaw_deg"]
        objects.append(item)
    return dict(scene, objects=objects)


def frame_telemetry(value, attempt: int) -> dict:
    telemetry = {
        "attempt": int(attempt),
        "value_type": type(value).__name__,
        "dtype": str(getattr(value, "dtype", None)),
        "shape": list(getattr(value, "shape", ())),
        "minimum": None,
        "maximum": None,
        "peak_to_peak": None,
        "channel_peak_to_peak": None,
        "nonzero_count": None,
        "accepted": False,
        "rejection_reason": None,
    }
    if value is None:
        telemetry["rejection_reason"] = "none"
        return telemetry
    if getattr(value, "ndim", 0) != 3 or value.shape[2] < 3:
        telemetry["rejection_reason"] = "invalid_dimensions_or_channels"
        return telemetry
    rgb = np.asarray(value[:, :, :3])
    if rgb.shape[:2] != (RESOLUTION[1], RESOLUTION[0]):
        telemetry["rejection_reason"] = "unexpected_resolution"
        return telemetry
    telemetry.update(
        minimum=float(np.min(rgb)),
        maximum=float(np.max(rgb)),
        peak_to_peak=float(np.ptp(rgb)),
        channel_peak_to_peak=[float(np.ptp(rgb[:, :, channel])) for channel in range(3)],
        nonzero_count=int(np.count_nonzero(rgb)),
    )
    if telemetry["nonzero_count"] == 0:
        telemetry["rejection_reason"] = "all_black"
    elif max(telemetry["channel_peak_to_peak"]) == 0:
        telemetry["rejection_reason"] = "constant_valued"
    else:
        telemetry["accepted"] = True
    return telemetry


def bounded_frame_capture(frame_supplier, synchronize, max_attempts=MAX_SYNCHRONIZED_CAPTURE_ATTEMPTS):
    telemetry_rows = []
    for attempt in range(1, max_attempts + 1):
        synchronize()
        candidate = frame_supplier()
        telemetry = frame_telemetry(candidate, attempt)
        telemetry_rows.append(telemetry)
        if telemetry["accepted"]:
            return candidate, telemetry_rows
    return None, telemetry_rows


def _call_optional(target, name, *args):
    function = getattr(target, name, None)
    if not callable(function):
        return False, "unsupported"
    try:
        function(*args)
        return True, "ran"
    except Exception as exc:
        return False, f"error:{type(exc).__name__}:{exc}"


def cleanup_capture_resources(camera, render_product_path, rep, World, log_lines) -> list[dict]:
    """Best-effort Isaac 6.0 cleanup in the required deterministic order."""
    actions = []
    annotator = getattr(camera, "_rgb_annotator", None) if camera is not None else None
    if annotator is not None and render_product_path:
        ran, status = _call_optional(annotator, "detach", [render_product_path])
    else:
        ran, status = False, "not_present"
    actions.append({"action": "detach_annotators", "ran": ran, "status": status})

    destroy = getattr(rep, "destroy", None)
    if destroy is not None and render_product_path:
        ran, status = _call_optional(destroy, "render_product", render_product_path)
    else:
        ran, status = False, "unsupported_or_not_present"
    actions.append({"action": "destroy_render_product", "ran": ran, "status": status})

    orchestrator = getattr(rep, "orchestrator", None)
    methods = []
    for name in ("stop", "wait_until_complete", "reset"):
        ran, status = _call_optional(orchestrator, name) if orchestrator is not None else (False, "unsupported")
        methods.append({"method": name, "ran": ran, "status": status})
    actions.append({"action": "stop_reset_replicator", "methods": methods})

    ran, status = _call_optional(camera, "pause") if camera is not None else (False, "not_present")
    actions.append({"action": "release_camera", "ran": ran, "status": status})

    try:
        if World.instance() is not None:
            World.clear_instance()
            ran, status = True, "ran"
        else:
            ran, status = False, "no_instance"
    except Exception as exc:
        ran, status = False, f"error:{type(exc).__name__}:{exc}"
    actions.append({"action": "clear_world", "ran": ran, "status": status})
    gc.collect()
    for row in actions:
        log_lines.append("cleanup=" + json.dumps(row, sort_keys=True))
    return actions


def _settled_ground_truth(scene, settled, C) -> dict:
    rows = []
    authored = {obj["name"]: obj for obj in scene["objects"]}
    table_z = C.TABLE_TOP_Z + C.SURFACE_DZ
    for name in sorted(settled):
        source = authored[name]
        pose = settled[name]
        centre = pose["centre"]
        row = {
            "name": name,
            "category": source["category"],
            "colour": source.get("colour", "yellow" if source["category"] == "cap" else "clear"),
            "xyz": [centre[0], centre[1], centre[2] - table_z],
            "yaw_deg": pose["yaw_deg"],
        }
        if source["category"] == "cube":
            row.update(size=C.CUBE_EDGE, defect_class=source["defect_class"])
        elif source["category"] == "cap":
            row.update(size=C.CAP_RADIUS * 2, orientation="narrow_end_up")
        else:
            row.update(size=C.VIAL_RADIUS * 2, length=C.VIAL_LENGTH)
        rows.append(row)
    return {
        "truth_timing": "after_stability_passed",
        "objects": rows,
        "defect_zones": [
            {
                "colour": colour,
                "defect_class": C.DEFECT_CLASS_BY_COLOUR[colour],
                "center": list(center),
                "size": list(C.DEFECT_ZONE_SIZE),
                "target_slots": [list(slot) for slot in C.DEFECT_TARGET_SLOTS[colour]],
            }
            for colour, center in C.DEFECT_ZONE_CENTERS.items()
        ],
    }


def main() -> None:
    global RESOLUTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=list(RESOLUTION),
        help="raster resolution only; physical camera pose/aperture are unchanged",
    )
    parser.add_argument(
        "--maximum-attempts",
        type=int,
        default=MAX_GENERATION_ATTEMPTS,
        help="bounded deterministic attempts; defaults to the Phase 4B policy",
    )
    args = parser.parse_args()
    requested_resolution = tuple(args.resolution)
    if requested_resolution[0] <= 0 or requested_resolution[1] <= 0:
        raise SystemExit("capture resolution must be positive")
    if requested_resolution[0] * RESOLUTION[1] != requested_resolution[1] * RESOLUTION[0]:
        raise SystemExit("capture resolution must preserve the calibrated aspect ratio")
    if not 1 <= args.maximum_attempts <= MAX_GENERATION_ATTEMPTS:
        raise SystemExit(
            f"maximum attempts must be in [1, {MAX_GENERATION_ATTEMPTS}]"
        )
    RESOLUTION = requested_resolution
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.iterdir()}
    diagnostic_prestart = {
        "diagnostic_console.log",
        "diagnostic_progress.jsonl",
        "faulthandler.log",
        "kit_logs",
        "startup_environment.json",
    }
    unexpected = existing - (
        diagnostic_prestart if os.environ.get("PHASE4C_DIAGNOSTIC_PROGRESS") else set()
    )
    if unexpected:
        raise SystemExit(
            f"Phase 4B output directory contains unexpected files: {sorted(unexpected)}"
        )
    root = Path(__file__).resolve().parent
    design = generate_batch_design(args.seeds)
    frozen = frozen_detector_record(root)
    _write_json(output / "batch_design.json", design)
    _write_json(output / "frozen_detector.json", frozen)
    _write_json(output / "runtime_dependency_manifest.json", runtime_manifest(root))
    _write_json(output / "resource_estimate.json", resource_estimate())
    _write_json(output / "proposed_yaad_request.json", proposed_yaad_request())

    diagnostic_marker("before_isaac_import")
    from isaacsim import SimulationApp
    diagnostic_marker("after_isaac_import")

    diagnostic_marker("before_simulation_app")
    sim_app = SimulationApp({"headless": True})
    diagnostic_marker("after_simulation_app")
    records = []
    try:
        import cv2
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleXFormPrim
        from isaacsim.sensors.camera import Camera

        sys.path.insert(0, str(root))
        import cell
        import scene_config as C

        for preview_scene in design["scenes"]:
            seed = preview_scene["seed"]
            diagnostic_marker(f"before_seed_{seed}_stage", seed=seed)
            scene_dir = output / f"seed_{seed}"
            scene_dir.mkdir()
            attempts_dir = scene_dir / "attempts"
            attempts_dir.mkdir()
            log_lines = [f"phase={PHASE_VERSION}", f"seed={seed}", "robot_commands_issued=false"]
            attempt_records = []
            final_record = None
            terminal_error = None
            for attempt_index in range(args.maximum_attempts):
                attempt_dir = attempts_dir / f"attempt_{attempt_index:02d}"
                attempt_dir.mkdir()
                try:
                    scene = generate_scene(seed, attempt_index)
                except Exception as exc:
                    reason = f"design_generation:{type(exc).__name__}:{exc}"
                    attempt_record = {
                        "top_level_seed": seed,
                        "attempt_index": attempt_index,
                        "derived_seed": derive_attempt_seed(seed, attempt_index),
                        "design_sha256": None,
                        "initial_constraints": None,
                        "settled_constraints": None,
                        "rejection_reasons": [reason],
                        "accepted": False,
                        "detector_consulted": False,
                    }
                    attempt_records.append(attempt_record)
                    _write_json(attempt_dir / "attempt_record.json", attempt_record)
                    log_lines.append(f"attempt={attempt_index} generation_rejected={reason}")
                    continue
                _write_json(attempt_dir / "design.json", scene)
                camera = None
                render_product_path = None
                world = None
                cleanup = []
                initial_geometry = None
                settled_geometry = None
                try:
                    _configure_scene_constants(scene, C)
                    omni.usd.get_context().new_stage()
                    log_lines.append(f"cleanup={json.dumps({'action': 'replace_stage', 'ran': True, 'attempt': attempt_index})}")
                    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
                    stage = omni.usd.get_context().get_stage()
                    stage_identifier = str(stage.GetRootLayer().identifier)
                    cell.build(stage)
                    diagnostic_marker(
                        f"after_seed_{seed}_stage",
                        seed=seed,
                        attempt_index=attempt_index,
                        stage_identifier=stage_identifier,
                    )
                    lights = {}
                    for light_name, scale in scene["lighting_intensity_scales"].items():
                        prim = stage.GetPrimAtPath(f"/World/{light_name}")
                        if not prim.IsValid():
                            raise RuntimeError(f"missing light {light_name}")
                        light = UsdLux.LightAPI(prim)
                        attr = light.GetIntensityAttr()
                        attr.Set(float(attr.Get()) * scale)
                        lights[light_name] = {
                            "path": f"/World/{light_name}",
                            "valid": bool(prim.IsValid()),
                            "intensity": float(attr.Get()),
                        }
                    expected_names = {obj["name"] for obj in scene["objects"]}
                    categories = {obj["name"]: obj["category"] for obj in scene["objects"]}
                    actual_names = {
                        prim.GetName() for prim in stage.Traverse()
                        if prim.GetPath().pathString.count("/") == 2
                        and prim.GetName().startswith(("SortCube", "SortCap", "SortVial"))
                    }
                    if actual_names != expected_names:
                        raise RuntimeError(
                            f"fresh-stage inventory mismatch expected={sorted(expected_names)} actual={sorted(actual_names)}"
                        )
                    wrappers = {}
                    diagnostic_marker(
                        "before_physics_release",
                        seed=seed,
                        attempt_index=attempt_index,
                    )
                    for name in sorted(expected_names):
                        prim = stage.GetPrimAtPath(f"/World/{name}")
                        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
                        wrappers[name] = SingleXFormPrim(f"/World/{name}")
                    cell.enable_ccd(world)
                    world.reset()
                    poses = _pose_sampler(wrappers, categories, C)

                    for _ in range(90):
                        world.step(render=True)
                    release_poses = poses()
                    release_scene = _scene_from_poses(scene, release_poses)
                    initial_geometry = geometry_report(release_scene, root_poses=release_poses)
                    _write_json(attempt_dir / "pre_release_geometry.json", initial_geometry)
                    if not initial_geometry["passed"]:
                        reasons = ["pre_release:" + value for value in initial_geometry["failure_reasons"]]
                        attempt_record = generation_attempt_record(
                            seed, attempt_index, scene, initial_geometry, None, reasons
                        )
                        attempt_records.append(attempt_record)
                        _write_json(attempt_dir / "attempt_record.json", attempt_record)
                        continue

                    for name in sorted(expected_names):
                        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(f"/World/{name}")).CreateKinematicEnabledAttr().Set(False)
                    physics_hz = 60
                    initial_steps = round(INITIAL_PHYSICS_SETTLING_S * physics_hz)
                    window_steps = round(STABILITY_WINDOW_S * physics_hz)
                    maximum_steps = round(MAXIMUM_TOTAL_SETTLING_S * physics_hz)
                    settling_steps = 0
                    for _ in range(initial_steps):
                        world.step(render=False)
                        settling_steps += 1
                    windows = []
                    stability = None
                    while settling_steps + window_steps <= maximum_steps:
                        window_number = len(windows) + 1
                        diagnostic_marker(
                            "stability_window_started",
                            seed=seed,
                            attempt_index=attempt_index,
                            window=window_number,
                        )
                        samples = [poses()]
                        for _ in range(window_steps):
                            world.step(render=False)
                            settling_steps += 1
                            samples.append(poses())
                        windows.append(samples)
                        stability = evaluate_scene_stability_windows(windows)
                        stability["actual_settling_duration_s"] = settling_steps / physics_hz
                        diagnostic_marker(
                            "stability_window_completed",
                            seed=seed,
                            attempt_index=attempt_index,
                            window=window_number,
                            passed=stability["windows"][-1]["passed"],
                            consecutive_stable_window_count=stability[
                                "consecutive_stable_window_count"
                            ],
                            actual_settling_duration_s=stability[
                                "actual_settling_duration_s"
                            ],
                        )
                        log_lines.append(
                            f"attempt={attempt_index} stability_window={len(windows)} "
                            f"passed={stability['windows'][-1]['passed']} "
                            f"consecutive={stability['consecutive_stable_window_count']}"
                        )
                        if capture_is_authorized(stability):
                            break
                    if not stability or not capture_is_authorized(stability):
                        stability = evaluate_scene_stability_windows(windows, timed_out=True)
                        stability["actual_settling_duration_s"] = settling_steps / physics_hz
                        _write_json(attempt_dir / "stability_report.json", stability)
                        reasons = ["scene_stability_timeout"]
                        attempt_record = generation_attempt_record(
                            seed, attempt_index, scene, initial_geometry, None, reasons
                        )
                        attempt_records.append(attempt_record)
                        _write_json(attempt_dir / "attempt_record.json", attempt_record)
                        continue

                    settled = poses()
                    settled_scene = _scene_from_poses(scene, settled)
                    settled_geometry = geometry_report(
                        settled_scene,
                        root_poses=settled,
                        release_scene=release_scene,
                        release_root_poses=release_poses,
                    )
                    _write_json(attempt_dir / "stability_report.json", stability)
                    _write_json(attempt_dir / "release_scene_state.json", release_poses)
                    _write_json(attempt_dir / "settled_scene_state.json", settled)
                    _write_json(attempt_dir / "settled_geometry.json", settled_geometry)
                    if not settled_geometry["passed"]:
                        reasons = ["settled:" + value for value in settled_geometry["failure_reasons"]]
                        attempt_record = generation_attempt_record(
                            seed, attempt_index, scene, initial_geometry, settled_geometry, reasons
                        )
                        attempt_records.append(attempt_record)
                        _write_json(attempt_dir / "attempt_record.json", attempt_record)
                        continue

                    # Rendering resources are deliberately created only after
                    # both stability and corrected settled geometry pass.
                    diagnostic_marker(
                        "before_camera_creation",
                        seed=seed,
                        attempt_index=attempt_index,
                        requested_resolution=list(RESOLUTION),
                    )
                    camera = Camera(prim_path=cell.TABLE_CAM_PATH, resolution=RESOLUTION)
                    camera.initialize()
                    diagnostic_marker(
                        "after_camera_creation",
                        seed=seed,
                        attempt_index=attempt_index,
                        camera_prim_path=cell.TABLE_CAM_PATH,
                    )
                    timeline = omni.timeline.get_timeline_interface()
                    if not timeline.is_playing():
                        timeline.play()
                    if not timeline.is_playing():
                        raise RuntimeError("timeline_failed_to_start")
                    camera_prim_raw = stage.GetPrimAtPath(cell.TABLE_CAM_PATH)
                    camera_valid = bool(camera_prim_raw.IsValid())
                    if not camera_valid or camera_prim_raw.GetTypeName() != "Camera":
                        raise RuntimeError("current_stage_table_camera_invalid")
                    camera_owned_by_stage = camera_prim_raw.GetStage() == stage
                    render_product_path = camera.get_render_product_path()
                    render_product_prim = stage.GetPrimAtPath(render_product_path)
                    render_product_valid = bool(render_product_prim.IsValid())
                    render_product_owned_by_stage = (
                        render_product_valid and render_product_prim.GetStage() == stage
                    )
                    if not camera_owned_by_stage or not render_product_owned_by_stage:
                        raise RuntimeError("camera_or_render_product_not_owned_by_current_stage")
                    rgb_annotator = getattr(camera, "_rgb_annotator", None)
                    attached_paths = getattr(rgb_annotator, "_render_product_paths", None)
                    if attached_paths is not None:
                        attached_paths = [str(value) for value in attached_paths]
                    lifecycle = {
                        "stage_identifier": stage_identifier,
                        "camera": {
                            "path": cell.TABLE_CAM_PATH,
                            "valid": camera_valid,
                            "type": camera_prim_raw.GetTypeName(),
                            "owned_by_current_stage": camera_owned_by_stage,
                        },
                        "lights": lights,
                        "render_product": {
                            "path": str(render_product_path),
                            "valid": render_product_valid,
                            "owned_by_current_stage": render_product_owned_by_stage,
                        },
                        "annotator": {
                            "present": rgb_annotator is not None,
                            "type": type(rgb_annotator).__name__ if rgb_annotator is not None else None,
                            "attached_render_products": attached_paths,
                            "attachment_introspection_supported": attached_paths is not None,
                            "ownership_basis": "current_stage_render_product",
                            "owned_by_current_stage": render_product_owned_by_stage,
                        },
                        "timeline_playing": bool(timeline.is_playing()),
                        "capture_attempts": [],
                    }
                    synchronization_attempt = 0

                    def synchronize_capture():
                        nonlocal synchronization_attempt
                        synchronization_attempt += 1
                        diagnostic_marker(
                            "before_replicator_sync",
                            seed=seed,
                            attempt_index=attempt_index,
                            capture_attempt=synchronization_attempt,
                        )
                        world.step(render=True)
                        rep.orchestrator.step(wait_for_render=True)
                        diagnostic_marker(
                            "after_replicator_sync",
                            seed=seed,
                            attempt_index=attempt_index,
                            capture_attempt=synchronization_attempt,
                        )

                    rgba, capture_rows = bounded_frame_capture(
                        camera.get_rgba,
                        synchronize_capture,
                        MAX_SYNCHRONIZED_CAPTURE_ATTEMPTS,
                    )
                    lifecycle["capture_attempts"] = capture_rows
                    for telemetry in capture_rows:
                        log_lines.append("capture_attempt=" + json.dumps(telemetry, sort_keys=True))
                    _write_json(attempt_dir / "camera_render_lifecycle.json", lifecycle)
                    if rgba is None:
                        terminal_error = "bounded_synchronized_capture_failed"
                        attempt_record = generation_attempt_record(
                            seed,
                            attempt_index,
                            scene,
                            initial_geometry,
                            settled_geometry,
                            [terminal_error],
                        )
                        attempt_records.append(attempt_record)
                        _write_json(attempt_dir / "attempt_record.json", attempt_record)
                        break

                    image = np.asarray(rgba[:, :, :3]).astype(np.uint8)
                    camera_prim = UsdGeom.Camera(camera_prim_raw)
                    xf = UsdGeom.Xformable(camera_prim.GetPrim()).ComputeLocalToWorldTransform(0)
                    camera_json = decorate_camera_calibration({
                        "camera": "table",
                        "resolution": list(RESOLUTION),
                        "focal_length_mm": float(camera_prim.GetFocalLengthAttr().Get()),
                        "horizontal_aperture_mm": float(camera_prim.GetHorizontalApertureAttr().Get()),
                        "vertical_aperture_mm": float(camera_prim.GetVerticalApertureAttr().Get()),
                        "camera_to_world": [[float(xf[r][c]) for c in range(4)] for r in range(4)],
                        "table_top_z": float(C.TABLE_TOP_Z + C.SURFACE_DZ),
                        "robot_base_xy": [0.0, 0.0],
                    })
                    workspace_json = simulation_source_workspace(camera_json)
                    image_path = scene_dir / "world_camera.png"
                    if not cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError("world_camera_write_failed")
                    decoded_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if decoded_image is None:
                        raise RuntimeError("world_camera_decode_after_write_failed")
                    actual_decoded_resolution = [
                        int(decoded_image.shape[1]),
                        int(decoded_image.shape[0]),
                    ]
                    diagnostic_marker(
                        "image_written",
                        seed=seed,
                        attempt_index=attempt_index,
                        path=str(image_path),
                        actual_decoded_resolution=actual_decoded_resolution,
                    )
                    _write_json(scene_dir / "camera.json", camera_json)
                    _write_json(scene_dir / "workspace.json", workspace_json)
                    _write_json(scene_dir / "release_scene_state.json", release_poses)
                    _write_json(scene_dir / "settled_scene_state.json", settled)
                    _write_json(scene_dir / "settled_geometry.json", settled_geometry)
                    _write_json(scene_dir / "stability_report.json", stability)
                    _write_json(scene_dir / "ground_truth.json", _settled_ground_truth(scene, settled, C))
                    _write_json(scene_dir / "camera_render_lifecycle.json", lifecycle)
                    resolution_metadata = {
                        "source_design_resolution": list(SOURCE_DESIGN_RESOLUTION),
                        "requested_capture_resolution": list(RESOLUTION),
                        "actual_decoded_output_resolution": actual_decoded_resolution,
                        "camera_pose_or_aperture_changed": False,
                    }
                    _write_json(
                        scene_dir / "capture_resolution_metadata.json",
                        resolution_metadata,
                    )
                    attempt_record = generation_attempt_record(
                        seed, attempt_index, scene, initial_geometry, settled_geometry, []
                    )
                    attempt_records.append(attempt_record)
                    _write_json(attempt_dir / "attempt_record.json", attempt_record)
                    final_record = {
                        "seed": seed,
                        "capture_status": "captured",
                        "stability_outcome": True,
                        "robot_commands_issued": False,
                        "accepted_attempt_index": attempt_index,
                        "accepted_derived_seed": scene["derived_seed"],
                        "generation_attempts": attempt_records,
                        "scene_generation_parameters": scene,
                        "starting_poses": {
                            obj["name"]: {"xy_m": obj["xy_m"], "yaw_deg": obj["yaw_deg"]}
                            for obj in scene["objects"]
                        },
                        "settled_poses": settled,
                        "actual_lighting_intensities": {
                            name: row["intensity"] for name, row in lights.items()
                        },
                        "stage_identifier": stage_identifier,
                        "render_product_path": str(render_product_path),
                        "detector_frozen_sha256": frozen["combined_sha256"],
                        "workspace_configuration_sha256": workspace_json["configuration_sha256"],
                        "capture_resolution_metadata": resolution_metadata,
                    }
                    log_lines.append(
                        f"capture_status=captured_after_stability attempt={attempt_index}"
                    )
                    break
                except Exception as exc:
                    terminal_error = f"{type(exc).__name__}: {exc}"
                    attempt_record = generation_attempt_record(
                        seed,
                        attempt_index,
                        scene,
                        initial_geometry,
                        settled_geometry,
                        [terminal_error],
                    )
                    attempt_records.append(attempt_record)
                    _write_json(attempt_dir / "attempt_record.json", attempt_record)
                    log_lines.append(f"attempt={attempt_index} terminal_error={terminal_error}")
                    break
                finally:
                    cleanup = cleanup_capture_resources(
                        camera, render_product_path, rep, World, log_lines
                    )
                    _write_json(attempt_dir / "cleanup_report.json", {
                        "required_order": list(CLEANUP_ORDER),
                        "actions": cleanup,
                        "stage_replaced_only_after_prior_cleanup": True,
                    })

            _write_json(scene_dir / "generation_attempts.json", {
                "top_level_seed": seed,
                "maximum_attempts": args.maximum_attempts,
                "attempts": attempt_records,
            })
            if final_record is None:
                error = terminal_error or "bounded_generation_attempts_exhausted"
                final_record = record_failed_seed(seed, error)
                final_record["generation_attempts"] = attempt_records
                print(f"[phase4b] seed {seed} failed safely: {error}", flush=True)
            records.append(final_record)
            _write_json(scene_dir / "scene_capture_record.json", final_record)
            (scene_dir / "scene.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        _write_json(output / "batch_capture_report.json", {"phase_version": PHASE_VERSION, "records": records})
    finally:
        diagnostic_marker("before_simulation_app_close")
        diagnostic_marker("simulation_app_close_invoked")
        sim_app.close()
        diagnostic_marker("after_simulation_app_close")


if __name__ == "__main__":
    main()
