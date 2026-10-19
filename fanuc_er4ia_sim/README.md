# FANUC ER-4iA hackathon starter

Build an Isaac Sim cell that picks six rotated 10 mm cubes and stacks them
**yellow → yellow → blue → blue → red → red**. Two caps and two tubes are distractors.

This minimal starter contains code, instructions, tests and required geometry.
**No images, videos, saved runs, model weights or virtual environments are included.**

- People: follow [SETUP.md](SETUP.md).
- Coding agents: read [AGENTS.md](AGENTS.md), then follow the same setup guide.
- Physical-system calibration: [docs/CALIBRATION_V05.md](docs/CALIBRATION_V05.md).
- Geometry/perception assumptions: [docs/MODEL_NOTES.md](docs/MODEL_NOTES.md).

```bash
./bootstrap.sh
# Configure ISAAC, COSMOS and model paths as described in SETUP.md.
./run_sort_pipeline.sh runs/stack_01
```

Bootstrap installs host dependencies, generates procedural material textures
locally and runs host tests. It does not install Isaac Sim or Cosmos. Detailed
finger geometry is retained; photo-derived finishes and serial markings are
replaced by a silver material when the optional photo maps are absent.

## Architecture and entry points

| File | Responsibility |
| --- | --- |
| `scene_config.py` | Dimensions, cube positions/rotations, stack order and camera configuration |
| `cell.py`, `gripper_geometry.py` | USD scene, gripper geometry and contact physics |
| `perceive.py`, `cube_pose.py` | Cosmos semantics and calibrated image-derived tabletop pose |
| `plan_sort.py`, `backproject.py`, `plan_grasps.py` | Vision-only planning and image/camera geometry |
| `sort_scene.py`, `ik.py`, `fk.py` | Physical execution, wrist alignment and success checks |
| `build_scene.py` | Interactive scene or headless build/render |
| `run_sort_pipeline.sh` | Capture → perceive → plan → execute → encode |

Cosmos identifies cube locations and colours. A calibrated image-geometry
front end measures yaw; Cosmos receives that measurement and confirms/reports
the pose. Standalone VLM angle estimation was unreliable. This hybrid system
uses no simulator ground-truth object poses to command grasps.

The actuator has **6 mm travel per jaw**, giving a **12 mm maximum straight-face
gap**. Square yaw is equivalent every 90°. Keep these constraints when editing.
Changes to layout, camera, cube size or contact geometry need a fresh run.

Rebuild this media-free package with `./share.sh --output ../starter.zip`.
