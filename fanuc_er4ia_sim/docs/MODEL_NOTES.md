# Geometry, perception and validation notes

## Scene and task

The scene has six 10 mm cubes, two caps and two tubes. The original yellow cube
locations retain their photo-fitted XY positions. Four additional cube locations
use seed `20261009`. Five cubes have nonzero tabletop rotations. The tower site
is `(0.28, 0.10)` in robot base XY, with yellow, yellow, blue, blue, red, red layers.
Dimensions are metres in code; rotations are degrees unless a function states
otherwise. The robot base origin is at world Z = 0.90 m. The calibrated work surface
is 5.1 mm below that plane (world Z = 0.8949 m); the mat is 2 mm thick.

The planning image uses a 4K capture of the calibrated table camera. Video
rendering uses 1920×1080. Camera transforms and intrinsics are read back from
USD for pixel-to-world conversion.

## Gripper

The supplied STEP and CAD-derived mesh identify the SMC MHZL2-16D2 actuator.
Each jaw has 6 mm travel: nominal straight-face gap is 0 mm closed and 12 mm
open. The custom fingers are approximately 76 × 12 × 6 mm. Their profile,
rounded shoulders, mounting holes and finish were reconstructed from real
photos and tape-measure views. Hidden radii, some finger dimensions and sensor
mount details remain estimates.

The open-ended hook profile leaves an approximately 4.0 mm gap at its tips when
closed, and 16.0 mm when open. This differs from the straight-face gap.
Cube grasps use 2.5 mm tip clearance and a closure target accounting for the
hook opening. The distal 32 mm retains a higher-friction material: static 1.5,
dynamic 1.3. Tips are rigid; deformation and measured force/friction have not
been calibrated. Detailed meshes are visual; contact uses the validated profile
proxies. Cosmetic holes and bevels do not change the collision shape.

Only runtime geometry is bundled. Procedural metal/environment textures are
generated locally by `make_textures.py`; photo-derived finger maps, markings
and comparison images are omitted. Missing photo maps use a silver material.
Finger contact proxies follow the updated blade length and hook opening;
the high-friction material is retained. The wrist-centre-to-tip length is
293 mm: 70 mm to the faceplate plus controller UTOOL1 length 223 mm.
See [CALIBRATION_V05.md](CALIBRATION_V05.md) for source and limitations.

## Pose estimator and gates

Cosmos locates/classifies cubes. Image-only colour segmentation and calibrated
square silhouette fitting then measure XY and yaw. The fit searches the full
90° symmetry interval, within ±4 mm of the image colour-component centre. Cosmos receives
this measurement with a crop, confirms category/colour, and reports yaw.
Standalone VLM corner regression, numeric in-context examples and outline
selection did not reliably estimate orientation in development trials.

The image fit requires IoU ≥0.85. Cosmos confidence must be ≥0.8, colour/category
must agree, and reported yaw must be within 3° of the image measurement modulo
90°. The planner back-projects the reported top edge at cube-top height and
checks/refines pose within ±20° and ±4 mm, requiring IoU ≥0.75 and rejecting
XY/yaw boundary optima. Exactly two eligible cubes of each colour are required.

The wrist follows each grasp pose, then turns the held cube to the common stack
orientation. Runtime checks pickup, target position/height and adjacent-layer
support. Placed yaw must be within 8° modulo 90°, with tilt below 5°.
This estimator assumes upright, known-size cubes on the calibrated horizontal
surface. It does not estimate arbitrary six-degree-of-freedom object poses.

## Validated example and limits

Validation after the v0.5 calibration recorded six correct cube colours and
six successful physical pickups and placements. Maximum refined XY perception
error was 0.19 mm, yaw error 2.01°, lateral placement error 0.44 mm and placed
yaw error 1.99° modulo 90°. The extracted media-free starter also passed all
33 host tests, generated a fresh camera capture and replayed the vision plan
with six successful pickups/placements. Its image-fit overlaps were ≥0.958.
Recorded runs and media are not bundled. Revalidate on your own machine.
Raw model yaw reflects image measurements supplied to Cosmos, rather than
independent model-only inference. Ground truth is used only for evaluation.

Caps and tubes are excluded from the task, but loose distractors may settle
or roll under physics. Motion, grasp and appearance are simulation results;
these checks do not establish hardware transfer accuracy. New layouts,
cameras, cube sizes or contact settings require another physical validation.
