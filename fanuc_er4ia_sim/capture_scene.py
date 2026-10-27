"""
Stage 1 of the perception pipeline (runs under Isaac's interpreter).

Builds the cell, renders the world camera, and writes out everything the later
stages need: the image, the camera's exact intrinsics/extrinsics, and the
ground-truth object poses so the VLM's guesses can actually be scored.

    $ISAAC/python.sh capture_scene.py --out run/
"""

import argparse, json, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="run")
parser.add_argument("--camera", default="table", choices=["table", "side", "wrist"])
parser.add_argument(
    "--resolution",
    help="Capture resolution as WIDTHxHEIGHT; for example 640x360",
)
parser.add_argument(
    "--validate-defect-zones",
    action="store_true",
    help="validate and annotate one static Dataset 3 table-camera scene",
)
args = parser.parse_args()
# resolve before Isaac starts: Kit changes the working directory
args.out = os.path.abspath(args.out)

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})

import numpy as np
import cv2
from pxr import UsdGeom, Gf
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C, cell

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
# Keep the arrangement identical across capture and execution while shaders load.
from pxr import UsdPhysics

for prim in stage.Traverse():
    if prim.GetName().startswith(("SortCube", "SortVial", "SortCap")) and prim.HasAPI(
        UsdPhysics.RigidBodyAPI
    ):
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)

_CAMS = {
    "table": (cell.TABLE_CAM_PATH, C.TABLE_CAM),
    "side": (cell.SIDE_CAM_PATH, C.SIDE_CAM),
    "wrist": (cell.WRIST_CAM_PATH, C.WRIST_CAM),
}
cam_path, cfg = _CAMS[args.camera]
cfg = dict(cfg)

if args.validate_defect_zones and args.camera != "table":
    parser.error("--validate-defect-zones requires --camera table")

if args.resolution:
    width, height = (int(value) for value in args.resolution.lower().split("x", 1))
    if width <= 0 or height <= 0:
        raise ValueError("--resolution dimensions must be positive")
    cfg["resolution"] = (width, height)
elif args.validate_defect_zones:
    cfg["resolution"] = (1920, 1080)
elif args.camera == "table" and C.SORT_TASK in ("cube_stack", "defect_sort"):
    cfg["resolution"] = tuple(2 * v for v in cfg["resolution"])

from isaacsim.sensors.camera import Camera

cam = Camera(prim_path=cam_path, resolution=cfg["resolution"])
cell.enable_ccd(world)
world.reset()
cam.initialize()

import omni.timeline

camera_prim = stage.GetPrimAtPath(cam_path)
render_product_path = cam.get_render_product_path()
render_product_prim = stage.GetPrimAtPath(render_product_path)
timeline = omni.timeline.get_timeline_interface()
if not timeline.is_playing():
    timeline.play()

print(
    "[capture] setup "
    f"camera={cam_path} camera_valid={camera_prim.IsValid()} "
    f"camera_type={camera_prim.GetTypeName()} "
    f"render_product={render_product_path} "
    f"render_product_valid={render_product_prim.IsValid()} "
    f"timeline_playing={timeline.is_playing()} "
    f"resolution={cfg['resolution']}",
    flush=True,
)

# The ordinary capture parks the arm out of view.  Static defect-zone validation
# must not command or reposition the robot at all.
from isaacsim.core.prims import SingleArticulation

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
if not args.validate_defect_zones:
    from isaacsim.core.utils.types import ArticulationAction

    names = list(arm.dof_names)
    q = arm.get_joint_positions()
    for j, deg in C.CAPTURE_POSE_DEG.items():
        q[names.index(j)] = np.deg2rad(deg)
    for jn, value in cell.jaw_targets(C.JAW_OPEN).items():
        q[names.index(jn)] = value
    arm.set_joint_positions(q)
    arm.apply_action(ArticulationAction(joint_positions=q))
inspection_joints_before = arm.get_joint_positions().tolist()

# MDL materials need frames to compile before synchronized capture.
for _ in range(90):
    world.step(render=True)

max_synchronized_attempts = 3
rgba = None
for attempt in range(1, max_synchronized_attempts + 1):
    if attempt == 1:
        print("[capture] before Replicator synchronized step", flush=True)
    rep.orchestrator.step(wait_for_render=True)
    if attempt == 1:
        print("[capture] after Replicator synchronized step", flush=True)
    candidate = cam.get_rgba()
    print(
        f"[capture] synchronized RGB attempt {attempt}/{max_synchronized_attempts} "
        f"type={type(candidate).__name__} shape={getattr(candidate, 'shape', None)}",
        flush=True,
    )
    if (
        candidate is not None
        and getattr(candidate, "ndim", 0) == 3
        and candidate.shape[2] >= 3
    ):
        rgba = candidate
        print(f"[capture] RGB ready after {90 + attempt} rendered steps", flush=True)
        break

if rgba is None:
    raise RuntimeError(
        f"RGB annotator produced no valid frame after {max_synchronized_attempts} "
        "synchronized attempts; "
        f"camera_valid={camera_prim.IsValid()}, "
        f"render_product_valid={render_product_prim.IsValid()}, "
        f"timeline_playing={timeline.is_playing()}, "
        f"resolution={cfg['resolution']}"
    )
inspection_joints_after = arm.get_joint_positions().tolist()

os.makedirs(args.out, exist_ok=True)
rgb = rgba[:, :, :3].astype(np.uint8)
img_path = os.path.join(args.out, "world_camera.png")
cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# ---- camera model, read back from the stage rather than recomputed
prim = stage.GetPrimAtPath(cam_path)
gcam = UsdGeom.Camera(prim)
xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
W, H = cfg["resolution"]
focal = gcam.GetFocalLengthAttr().Get()
h_ap = gcam.GetHorizontalApertureAttr().Get()
v_ap = gcam.GetVerticalApertureAttr().Get()

cam_json = {
    "camera": args.camera,
    "resolution": [W, H],
    "focal_length_mm": float(focal),
    "horizontal_aperture_mm": float(h_ap),
    "vertical_aperture_mm": float(v_ap),
    # row-major 4x4, USD convention (row-vector times matrix)
    "camera_to_world": [[float(xf[r][c]) for c in range(4)] for r in range(4)],
    # Calibrated work-surface height, independent of the robot base plane.
    "table_top_z": float(C.TABLE_TOP_Z + C.SURFACE_DZ),
    "robot_base_xy": [0.0, 0.0],
}
with open(os.path.join(args.out, "camera.json"), "w") as f:
    json.dump(cam_json, f, indent=2)

# ---- ground truth, for scoring the VLM later
from isaacsim.core.prims import SingleXFormPrim

gt = {
    "objects": [],
    "bins": {},
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
scene_state = {"objects": {}}
for prim in stage.Traverse():
    if prim.GetName().startswith(("SortCube", "SortVial", "SortCap")) and prim.HasAPI(
        UsdPhysics.RigidBodyAPI
    ):
        pos, quat = SingleXFormPrim(str(prim.GetPath())).get_world_pose()
        scene_state["objects"][prim.GetName()] = {
            "position": pos.tolist(),
            "orientation": quat.tolist(),
        }

for i, (x, y, colour) in enumerate(C.SORT_CUBES):
    gt["objects"].append(
        {
            "name": f"SortCube{i}",
            "category": "cube",
            "colour": colour,
            "xyz": [x, y, C.CUBE_EDGE / 2],
            "size": C.CUBE_EDGE,
            "yaw_deg": C.SORT_CUBE_YAWS_DEG[i],
        }
    )

for i, (x, y) in enumerate(C.SORT_CAPS):
    gt["objects"].append(
        {
            "name": f"SortCap{i}",
            "category": "cap",
            "colour": "yellow",
            "xyz": [x, y, C.CAP_HEIGHT / 2],
            "orientation": C.CAP_ORIENTATIONS[i],
            "size": C.CAP_RADIUS * 2,
        }
    )

for i, (x, y_ctr, yaw) in enumerate(C.SORT_VIALS):
    gt["objects"].append(
        {
            "name": f"SortVial{i}",
            "category": "vial",
            "colour": "clear",
            "xyz": [x, y_ctr, C.VIAL_RADIUS],
            "size": C.VIAL_RADIUS * 2,
            "length": C.VIAL_LENGTH,
            "yaw_deg": yaw,
        }
    )

with open(os.path.join(args.out, "ground_truth.json"), "w") as f:
    json.dump(gt, f, indent=2)
with open(os.path.join(args.out, "scene_state.json"), "w") as f:
    json.dump(scene_state, f, indent=2)

validation_report = None
try:
    if args.validate_defect_zones:
        from defect_zone_validation import validate_scene_capture

        validation_report = validate_scene_capture(
            stage,
            args.out,
            gt,
            scene_state,
            inspection_joints_before,
            inspection_joints_after,
        )
        print(
            f"[capture] defect-zone validation {validation_report['status']} "
            f"failures={validation_report['failures']}",
            flush=True,
        )

    print(f"[capture] image      {img_path}  ({W}x{H})")
    print(f"[capture] camera     {args.out}/camera.json")
    print(f"[capture] gt objects {len(gt['objects'])}")
    if validation_report is not None and validation_report["status"] != "PASS":
        raise SystemExit(
            "Defect-zone validation failed; inspect defect_zone_validation.json"
        )
finally:
    sim_app.close()
