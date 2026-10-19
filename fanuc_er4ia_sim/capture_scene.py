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
args = parser.parse_args()
# resolve before Isaac starts: Kit changes the working directory
args.out = os.path.abspath(args.out)

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})

import numpy as np
import cv2
from pxr import UsdGeom, Gf
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
# Capture more top-face detail for 10 mm cubes; execution videos stay at 1080p.
if args.camera == "table" and C.SORT_TASK == "cube_stack":
    cfg["resolution"] = tuple(2 * v for v in cfg["resolution"])

from isaacsim.sensors.camera import Camera

cam = Camera(prim_path=cam_path, resolution=cfg["resolution"])
cell.enable_ccd(world)
world.reset()
cam.initialize()

# park the arm out of the overhead camera's view before the snapshot
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
names = list(arm.dof_names)
q = arm.get_joint_positions()
for j, deg in C.CAPTURE_POSE_DEG.items():
    q[names.index(j)] = np.deg2rad(deg)
for jn, v in cell.jaw_targets(C.JAW_OPEN).items():
    q[names.index(jn)] = v
arm.set_joint_positions(q)
arm.apply_action(ArticulationAction(joint_positions=q))

# MDL materials need frames to compile before capture
for _ in range(90):
    world.step(render=True)

os.makedirs(args.out, exist_ok=True)
rgba = cam.get_rgba()
if rgba is None:
    raise RuntimeError("Camera returned no image; check the host GPU")
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

gt = {"objects": [], "bins": {}}
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

print(f"[capture] image      {img_path}  ({W}x{H})")
print(f"[capture] camera     {args.out}/camera.json")
print(f"[capture] gt objects {len(gt['objects'])}")
sim_app.close()
