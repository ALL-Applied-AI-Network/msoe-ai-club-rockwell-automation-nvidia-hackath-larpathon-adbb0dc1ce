"""
Build the FANUC ER-4iA grasping cell in Isaac Sim 6.0.

    /isaacsim/python.sh ~/fanuc_er4ia_sim/build_scene.py            # interactive
    /isaacsim/python.sh ~/fanuc_er4ia_sim/build_scene.py --headless --save out.usd

Scene construction lives in cell.py, dimensions in scene_config.py.
"""

import argparse, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--save", default="", help="write the composed stage to this .usd")
parser.add_argument(
    "--shoot", default="", help="render one frame from each camera into this dir"
)
parser.add_argument(
    "--pathtrace",
    action="store_true",
    help="render with path tracing instead of RTX real-time",
)
parser.add_argument(
    "--realtime",
    action="store_true",
    help="use RTX real-time for captures (path tracing is default)",
)
parser.add_argument(
    "--dynamic-capture",
    action="store_true",
    help="allow task objects to move during capture warm-up",
)
parser.add_argument("--spp", type=int, default=64)
args = parser.parse_args()
# Kit changes the process working directory, so a relative output path silently
# writes somewhere the caller will never find. Resolve before Isaac starts.
if args.shoot:
    args.shoot = os.path.abspath(args.shoot)
if args.save:
    args.save = os.path.abspath(args.save)

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": args.headless})

import numpy as np
import omni.usd
from pxr import UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C, cell, render_settings

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
if args.pathtrace or (args.shoot and not args.realtime):
    render_settings.enable_path_tracing(args.spp)

# Comparison stills must preserve the arrangement that was fitted to the real
# photographs. In the supplied renders a tube moved ~0.17 m during warm-up.
# This affects --shoot only; interactive/demo physics stays dynamic.
if args.shoot and C.CALIBRATION_SNAPSHOT and not args.dynamic_capture:
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith(
            ("/World/SortVial", "/World/SortCap", "/World/SortCube")
        ):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)

cell.enable_ccd(world)
world.reset()

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
names = list(arm.dof_names)
print(f"[scene] articulation DOFs: {names}")
q = arm.get_joint_positions()

# For --shoot: use WRIST_PHOTO_POSE (arm over table, tool pointing down) so
# all three cameras see the work area with the arm in the same pose as the real
# calibration photos. For interactive/save mode, use HOME_POSE.
shoot_pose = C.WRIST_PHOTO_POSE_DEG if args.shoot else C.HOME_POSE_DEG
for j, deg in shoot_pose.items():
    if j in names:
        q[names.index(j)] = np.deg2rad(deg)
arm.set_joint_positions(q)
arm.apply_action(ArticulationAction(joint_positions=q))

# MDL materials need a few frames to compile; physics needs time to
# depenentrate any initially-overlapping objects and let them settle.
# Render warm-up is required even in headless mode for transparent materials.
for _ in range(29):
    world.step(render=True)
for _ in range(270):
    world.step(render=False)  # physics settle — no render needed
world.step(render=True)  # sync arm world transforms for the wrist camera

if args.shoot:
    os.makedirs(args.shoot, exist_ok=True)
    import omni.replicator.core as rep

    # Capture the actual flange-mounted wrist camera alongside the fixed views.
    # One arm pose and one world state for all three outputs.
    writers, products = [], []
    for path, cfg, tag in (
        (cell.TABLE_CAM_PATH, C.TABLE_CAM, "tablecam"),
        (cell.SIDE_CAM_PATH, C.SIDE_CAM, "sidecam"),
        (cell.WRIST_CAM_PATH, C.WRIST_CAM, "wristcam"),
    ):
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"Camera missing: {path}")
        rp = rep.create.render_product(path, cfg["resolution"])
        w = rep.WriterRegistry.get("BasicWriter")
        w.initialize(output_dir=os.path.join(args.shoot, tag), rgb=True)
        w.attach([rp])
        writers.append(w)
        products.append(rp)
    rep.orchestrator.step()
    rep.orchestrator.wait_until_complete()
    import json
    from pxr import UsdGeom

    # Record the actual articulation state and source settings with every run.
    metadata = {
        "snapshot": bool(C.CALIBRATION_SNAPSHOT and not args.dynamic_capture),
        "joint_positions_deg": dict(
            zip(names, np.rad2deg(arm.get_joint_positions()).tolist())
        ),
        "requested_joint_positions_deg": shoot_pose,
        "tool_length_m": C.TOOL_LENGTH,
        "gripper_model": C.GRIPPER_MODEL,
        "jaw_stroke_per_side_m": C.JAW_STROKE,
        "straight_face_gap_m": {"closed": 2 * C.JAW_CLOSED, "open": 2 * C.JAW_OPEN},
        "cap_orientations": dict(
            zip((f"SortCap{i}" for i in range(len(C.SORT_CAPS))), C.CAP_ORIENTATIONS)
        ),
        "cameras": {"table": C.TABLE_CAM, "side": C.SIDE_CAM, "wrist": C.WRIST_CAM},
        "authored_object_transforms": {},
    }
    for prim in stage.Traverse():
        if (
            prim.GetParent().GetPath().pathString == "/World"
            and prim.GetName().startswith(("SortVial", "SortCap", "SortCube"))
        ):
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            metadata["authored_object_transforms"][prim.GetName()] = [
                list(row) for row in matrix
            ]
    with open(os.path.join(args.shoot, "capture_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[scene] renders written to {args.shoot}")

if args.save:
    stage.Export(args.save)
    print(f"[scene] stage exported to {args.save}")

if not args.headless:
    while sim_app.is_running():
        world.step(render=True)
sim_app.close()
