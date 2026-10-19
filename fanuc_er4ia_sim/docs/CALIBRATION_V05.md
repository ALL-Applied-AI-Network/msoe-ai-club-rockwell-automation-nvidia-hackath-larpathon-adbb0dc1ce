# Physical-cell calibration imported from v0.5

The source was `fanuc_er4ia_twin_v0.5.zip`, provided on 2026-10-09.
SHA-256: `95913e3f5484a441ac0ee4ec53f2b2dfa0d917cd257b07832fb40b3e32537837`.
It includes a screw-picking twin and a separate cube-starter compatibility patch.
We imported the compatible calibration plus the main twin's measured finger
length/opening, retaining the six-cube stacking task and hybrid image/Cosmos
pose pipeline. Screw policies, recorded media and background plates are omitted.

## Dimensions and conventions

| Quantity | Current value | Basis |
| --- | --- | --- |
| J6_link origin to faceplate | 70 mm along local +X | Source wrist-centre interpretation |
| Faceplate to fingertip | 223 mm | Controller UTOOL1 Z and source ruler check |
| J6_link origin to fingertip | 293 mm | Sum of the two offsets |
| Exposed finger blade | 76 × 12 × 6 mm | Source measured length; photo-derived width/profile |
| Upper cylindrical spacer | 72.3 mm | Fits 223 mm stack with 76 mm blades |
| Stroke | 6 mm per jaw | SMC MHZL2-16D2 |
| Straight-face gap | 0–12 mm | Actuator travel |
| Hooked fingertip gap | 3.99–15.99 mm | Profile fit to measured 4 ±1 / 15 ±2 mm gaps |
| Work surface relative to base | −5.1 mm | Source home clearance measurement |
| Mat thickness | 2 mm | Independent positive geometry thickness |

`TABLE_TOP_Z` remains the robot base world Z (0.90 m). The work surface is
`TABLE_TOP_Z + SURFACE_DZ` (0.8949 m). Table and mat geometry move together;
negative surface offset must never become a negative collider thickness.
The detailed visual mesh was regenerated for the new blade length and hook.
Contact retains the full profile proxies and soft-tip friction estimates
(static 1.5, dynamic 1.3). Material compliance and actual friction remain unmeasured.

The source reports zero TCP discrepancy over 52 controller poses, compared
with 63.3 mm previously. Those pose samples were not supplied, so this is a
source claim rather than an independently reproduced physical calibration.
The spacer length closes the measured total stack; its individual dimensions
have not been independently surveyed.

The source's physical-controller convention is `sim_q3 = controller_J3 + J2`;
controller UF0 is 330 mm above the base plane. Simulation already uses relative
joint rotations. Apply such conversion only in a future hardware interface,
with explicit frame checks; do not add J2 again in simulation FK/IK.

## Cameras

`scene_config.py` contains the supplied eye/look-at/up transforms and intrinsics.
Table/side transforms use world coordinates, wrist uses J6_link coordinates.
Horizontal aperture is 20.955 mm for all three pinhole cameras.

| Camera | Resolution | Focal length | Horizontal focal pixels |
| --- | --- | --- | --- |
| Table | 1920×1080 | 11.2158 mm | 1027.6 |
| Side | 1920×1080 | 12.9427 mm | 1185.9 |
| Wrist | 1280×720 | 10.1440 mm | 619.6 |

The source contains ArUco camera solves. The rebuilt side camera has a reported
reprojection RMS of 3.99 pixels. Table/wrist camera mounts were checked with
render overlays, but were not physically revalidated after that rebuild.
The simulator uses the source's pinhole approximations, without the measured
small lens-distortion coefficients. Capture reads actual USD transforms and
intrinsics for perception; always recapture and replan after calibration changes.

The supplied cube patch made the camera bracket's signed length negative after
moving the housing. The imported builder keeps the signed centre offset and
uses a positive plate extent, avoiding an invalid PhysX box collider.

## Validation and agent instructions

Run `bootstrap.sh` (33 host tests), then the full pipeline from `SETUP.md`.
Require all six eligible cube detections and six verified physical pickups and
placements in yellow, yellow, blue, blue, red, red order. Check the execution
report, not just Isaac's exit code. Ground truth is used only for evaluation.
Keep the calibrated geometry, per-jaw stroke, pose confidence and support gates.

Local calibration source JSONs and validation reports are kept in
`calibration/v05/` in the development workspace. They are not required at
runtime and are omitted from the starter ZIP. Simulation success does not
establish hardware transfer accuracy; recheck tool/frame and camera calibration
on the physical cell before interpreting simulation positions as robot targets.

The wider table view exposed an image-fit defect: the initial pose search was
centred on approximate Cosmos coordinates instead of the matching colour blob.
One detected cube fell outside the ±4 mm fit window. Initial fitting now seeds
from the local image component (maximum shift 20 pixels at 1080p), retaining
the ±4 mm window and IoU ≥0.85 gate. Fits at XY search boundaries are rejected.
Silhouettes now test continuous polygons at pixel centres instead of rounding
vertices and adding inclusive boundary pixels. Captured cubes start on the
surface, removing a 1 mm floating offset that biased the assumed top-face plane.
A regression checks recovery from a 6 mm approximate-centre error without
using configured object poses or ground truth.


Local regression results: 33 host tests pass. Authored geometry checks verify
293 mm TCP, positive box extents, 6 mm/jaw limits, detailed finger profile and
high-friction material bindings. Actual Cosmos inference and PhysX execution
complete all six layers. Refined position/yaw errors are at most 0.19 mm / 2.01°;
placement position/yaw errors are at most 0.44 mm / 1.99°. The extracted starter
passes the same tests, renders a fresh capture and physically replays the plan
with all six placements. These errors measure agreement within simulation.
