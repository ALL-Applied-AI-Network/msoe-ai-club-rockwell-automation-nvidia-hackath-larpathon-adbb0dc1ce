"""
Scene parameters for the FANUC ER-4iA grasping cell.

Coordinate frame: origin at the robot base mounting face (on the table top),
+X pointing away from the robot toward the work area / camera side,
+Y to the robot's left, +Z up. Meters and degrees, matching the asset
(metersPerUnit = 1, upAxis = Z).

Provenance tags:
  EXACT     - read out of the NVIDIA/FANUC SimReady USD, or a vendor datasheet
  SCAN      - measured off the Scaniverse site scan (CSI-site-digital-model.zip,
              "204107.glb"), registered to the robot by its base-plate bolts
  VIDEO     - measured in the real-cell video (pick_m6_scrwes_CSI_hackathon26.mov),
              mapped onto the scan by a homography through printed mat features
              (~2 mm RMS), or scaled against a part of known size
  PHOTO     - estimated from the 2026-09 bench photos of the gripper
  GUESS     - not observable; placeholder pending measurement
"""

# ------------------------------------------------------------------ robot
# EXACT. Present only in the Isaac 6.0 asset bundle; 5.0/5.1 ship CRX10IAL only.
ROBOT_USD_SUBPATH = "/Isaac/Robots/Fanuc/er4ia/er4ia.usd"
ROBOT_PRIM = "/World/ER4iA"

# EXACT (er4ia_physics.usd). Degrees.
JOINT_LIMITS_DEG = {
    "J1": (-170.0, 170.0),
    "J2": (-110.0, 120.0),
    "J3": (-69.0, 205.0),
    "J4": (-190.0, 190.0),
    "J5": (-120.0, 120.0),
    "J6": (-360.0, 360.0),
}

# EXACT (joint localPos chain). Useful for sanity-checking the layout scale.
SHOULDER_HEIGHT = 0.33  # base mount face -> J2 axis
UPPER_ARM = 0.26  # J2 -> J3
ELBOW_OFFSET = 0.02  # J3 -> J4
FOREARM = 0.29  # J4 -> J5 (wrist center)
MAX_REACH = 0.55  # ~ UPPER_ARM + FOREARM

# VIDEO: arm hangs forward and down over the work area between moves.
HOME_POSE_DEG = {"J1": 0.0, "J2": 35.0, "J3": -25.0, "J4": 0.0, "J5": -80.0, "J6": 0.0}

# PHOTO-FIT 2026-10-08: same pose for all three camera captures. Solved with
# fk.py from the fitted wrist view and the existing rigid camera mount.
# Visual pose estimate, not recorded encoder data.
WRIST_PHOTO_POSE_DEG = {
    "J1": -3.45906,
    "J2": 21.320524,
    "J3": -25.72834,
    "J4": -8.620343,
    "J5": -34.733829,
    "J6": -79.793869,
}

# Where the arm parks while the overhead camera takes the planning snapshot:
# swung ~80 deg to the robot's right, fingertips 0.30 m above the mat at
# (0.05, -0.32) -- clear of the screws and the bin. (At the zero pose the
# fingers hang straight over the screw target and hide one from the camera.)
CAPTURE_POSE_DEG = {
    "J1": -81.1,
    "J2": 7.4,
    "J3": 5.8,
    "J4": 0.0,
    "J5": -88.4,
    "J6": 55.9,
}

# VIDEO: the real J1 base casting is mid grey, not black as the asset renders it.
ROBOT_BASE_GREY = True

# ------------------------------------------------------------------ work cell
# GUESS. The site scan does not reach the floor; it only shows the top is at
# least 687 mm up. Only affects world height, not the task.
TABLE_TOP_Z = 0.90
# SCAN: tabletop extents in the robot frame. The robot is square to the table
# (base-plate and table edges agree to within 0.1 deg) and sits 371 mm from the
# back edge, not centred.
TABLE_X = (-0.371, 0.759)
TABLE_Y = (-0.772, 0.792)
TABLE_THICK = 0.05
TABLE_SIZE = (TABLE_X[1] - TABLE_X[0], TABLE_Y[1] - TABLE_Y[0], TABLE_THICK)
TABLE_CENTER = ((TABLE_X[0] + TABLE_X[1]) / 2, (TABLE_Y[0] + TABLE_Y[1]) / 2)
# SCAN + VIDEO: yellow "CAUTION" tape along the front and both side edges of
# the tabletop (not the back edge, which abuts the enclosure).
EDGE_BAND_W = 0.048

# SCAN: the printed mat artwork, rectified into the robot frame by
# make_mat_texture.py. It is a stock FANUC LR Mate mat -- its arcs are labelled
# "200iD" / "200iD/4S" and are centred ~0.37 m BEHIND the ER-4iA's J1, which is
# why the earlier assumption (arcs = ER-4iA reach, centred on J1) was wrong.
# The "#" of black tape next to the bin and the black pad are in it too.
MAT_TEXTURE = "assets/mat_print.png"
MAT_MARKINGS = False  # False -> clean white table (matches real calibration photos)

# ------------------------------------------------------------------ props
# VIDEO: dark fixture at the table's front-left corner (+X, -Y): a grey base
# plate carrying a raised tray, a two-hole block and a pill bottle. It is a
# distractor for perception, not a task object. Footprint mapped from the
# video; internal layout approximate.
FIXTURE_XY = (0.555, -0.640)
FIXTURE_PLATE = (0.230, 0.260, 0.012)
FIXTURE_TRAY = (0.130, 0.170, 0.026)  # raised tray, recessed top
FIXTURE_BLOCK = (0.055, 0.115, 0.034)  # block with two round holes
BOTTLE_R = 0.018
BOTTLE_H = 0.058

# ------------------------------------------------------------------ gripper
# PHOTO 2026-10-09: PXL_20261008_154537672.jpg. Bare machined aluminium
# fingers with an AA-sized circular recess; no tape on the working profile.
# Operator update: soft contact tips, modeled separately below.
# AA diameter anchors the recess independently of the earlier camera fit.
# SMC body/jaws use the supplied STEP; custom blades and upper spacer use
# tape/photo estimates, not a precision survey.
# Source for AA envelope: https://data.energizer.com/pdfs/e91_max_na.pdf
AA_DIAMETER = 0.01450
AA_POCKET_CLEARANCE = 0.0  # nominal conformal fit; adjust after measurement

ADAPTER_R = 0.026  # black flange adapter
FLANGE_OFFSET = 0.070  # v0.5: J6 wrist centre to faceplate, along local +X
ADAPTER_T = 0.012
ZCYL_R = 0.023  # approximate transverse half-envelope
ZCYL_LEN = 0.0723  # 223 mm faceplate-to-tip stack with measured 76 mm blades
ZCYL_RECTANGULAR = False  # ee_global: round upper spacer; SMC body below it
ZCYL_BODY_W = 0.038
ZCYL_BODY_D = 0.034
COLLAR_R = 0.028
COLLAR_T = 0.010
COLLAR_AT = 0.004
ZFLANGE_R = 0.013
ZFLANGE_T = 0.010
BODY_LEN = 0.0527
BODY_W = 0.047
BODY_D = 0.0236
PLATE_T = 0.0  # incorporated in supplied SMC CAD body
PLATE_W = 0.042
PLATE_D = 0.028
CARRIAGE_LEN = 0.02115
CARRIAGE_W = 0.0155
CARRIAGE_D = 0.008
CARRIAGE_OVERLAP = 0.02115  # custom finger starts at CAD body front

FINGER_LENGTH = 0.076  # v0.5: measured exposed blade length
FINGER_WIDTH = 0.012  # profile stock width; > AA recess radius
FINGER_THICK = 0.006  # extrusion / plate thickness
SCALLOP_R = (AA_DIAMETER + AA_POCKET_CLEARANCE) / 2
SCALLOP_ABOVE_TIP = (
    0.00697  # v0.5: measured tip gap approximately 4 mm closed / 16 mm open
)
SCALLOP_AT = 1.0 - SCALLOP_ABOVE_TIP / FINGER_LENGTH
FINGER_TIP_BEVEL = 0.014  # outer bevel visible in the close-up
FINGER_TIP_WIDTH = 0.0075
FINGER_EDGE_BREAK = 0.0006
FINGER_MOUNT_HOLES = (0.010, 0.023)  # axial mounting-screw positions
FINGER_TAPE = False
# Soft contact tips reported by the operator. Friction is an initial estimate;
# geometry keeps the measured AA recess and jaw gap. No compliance is modeled.
SOFT_TIP_LENGTH = 0.032
SOFT_TIP_STATIC_FRIC = 1.5
SOFT_TIP_DYNAMIC_FRIC = 1.3
TOOL_LENGTH = (  # J6_link origin (wrist centre) -> fingertip
    FLANGE_OFFSET
    + ADAPTER_T
    + ZCYL_LEN
    + ZFLANGE_T
    + BODY_LEN
    + PLATE_T
    + CARRIAGE_LEN
    - CARRIAGE_OVERLAP
    + FINGER_LENGTH
)

# SMC MHZL2-16D2 has 6 mm travel PER JAW. Straight-face gap is 0/12 mm.
# v0.5 reports physical fingertip gaps 4 +/- 1 mm closed and 15 +/- 2 mm open.
GRIPPER_MODEL = "SMC MHZL2-16D2"
GRIPPER_CAD = "assets/gripper/smc_mhzl2_16d2.json"
JAW_STROKE = 0.006
JAW_CLOSED = 0.0
JAW_OPEN = JAW_CLOSED + JAW_STROKE
assert abs(JAW_OPEN - JAW_CLOSED - JAW_STROKE) < 1e-12
# How far above the work surface the fingertips are commanded when picking.
# At 1 mm the position-controlled arm pressed the taped tips into the mat, and
# tape-on-mat friction held the jaws open: they were still 1.5-5 mm short of
# the part when the lift began, and one screw was left behind. 2.5 mm keeps the
# tips off the mat and still below a lying screw's axis (~4.5 mm).
TIP_CLEARANCE = 0.0025

# Optional tape for older task configurations. Disabled for the bare AA jaws.
TAPE_LEN = 0.032  # how far up from the tip it is wrapped
TAPE_PAD = 0.0006  # how proud of the blade it sits
TAPE_STATIC_FRIC = 1.5
TAPE_DYNAMIC_FRIC = 1.3
JAW_MASS = 0.032  # one blade + its carriage
JAW_DRIVE_STIFF = 4.0e4
JAW_DRIVE_DAMP = 3.0e3
JAW_MAX_FORCE = 180.0

TOOL_AXIS = (1.0, 0.0, 0.0)
FINGER_SPREAD_AXIS = (0.0, 1.0, 0.0)

# ------------------------------------------------------------------ wrist camera
# PHOTO 2026-10-09: compact grey industrial camera (~50 × 50 × 32 mm body) on a
# small bracket to one side of the tool.  NO ring light — the earlier Banner
# LEDRR70X70 ring-light model was wrong; the real install is just the camera.
CAMBODY = (0.050, 0.050, 0.032)  # PHOTO estimate: square grey camera housing
CAMBODY_GREY = (0.50, 0.50, 0.52)  # medium grey housing colour
# Offset and height estimated from 7672; optical direction retains the 90° mount.
WRISTCAM_LATERAL = 0.0979
# Reference: PXL_20261008_154537672.jpg. Rotate the fitted camera assembly
# +90 degrees about flange +X (tool axis), independently of the jaws.
WRISTCAM_ROLL_DEG = 90.0
WRISTCAM_DIR = (0.0, 0.1277, 0.9918)
CAM_ABOVE_TIP = TOOL_LENGTH + 0.006 - 0.1633  # housing follows calibrated lens
BRACKET_T = 0.004
# Visible EOAT hardware estimated from the same close-up.
CAM_CORNER_R = 0.005
CAM_LID_T = 0.002
SHOW_TOOL_DETAIL = True
LASER_RADIUS = 0.0035
LASER_LENGTH = 0.032


# ------------------------------------------------------------------ task objects (sort demo)
# USD assets in assets/usd_assets/; dimensions are CAD millimetres.
# The files use inches (metersPerUnit = 0.0254); Isaac's USD composition converts
# automatically. cell.py applies additional photo-fit scaling to visuals.
#   vial.usda       27.18 x 27.18 x 203.0 mm  origin = bottom centre
#   vial_cap.usda   31.75 x 31.75 x  15.7 mm  origin = flange underside
#   cube.usda       16.0  x 16.0  x  16.0 mm  origin = bottom-face centre
#                   colour variants: red (default), yellow, blue

# -- screws disabled --
SCREWS = []

# -- cube (cube.usda, 16 mm)
# PHOTO 2026-10-08: real scene has 2 cubes, both yellow.
CUBE_EDGE = 0.010  # hackathon: 2 mm clearance inside measured 12 mm open gap
# The bottom of the open-ended AA socket remains separated at full closure.
# Target contact at the hooked ends, allowing 0.3 mm per-jaw seating travel.
FINGER_INNER_TIP = (SCALLOP_R**2 - SCALLOP_ABOVE_TIP**2) ** 0.5
CUBE_TIP_CLEARANCE = TIP_CLEARANCE
CUBE_JAW_HALF_TARGET = max(JAW_CLOSED, CUBE_EDGE / 2 - FINGER_INNER_TIP - 0.0003)
assert 2 * (JAW_OPEN + FINGER_INNER_TIP) > CUBE_EDGE * 2**0.5
CUBE_MASS = 0.004
CUBE_MATS = ["red", "yellow", "blue"]  # variant names in the USD colour set
SORT_CUBES = [
    (0.365769, 0.072782, "yellow"),
    (0.419421, 0.178986, "yellow"),
]  # PHOTO-FIT; see calibration/uwm_20261008/fit.json

# -- vial (vial.usda, 27.18 mm dia x 203 mm long)
# Each entry: (x_ctr, y, yaw_deg) where yaw_deg is the angle of the vial
# long-axis from world +X in the XY plane (0° = along +X, 90° = along +Y).
# cell.py: RotateY(+90°) maps local Z→world X, then RotateZ(yaw_deg) tilts it.
# Translation accounts for the offset: T = (x_ctr - L/2·cos(yaw), y - L/2·sin(yaw)).
VIAL_RADIUS = 0.017 / 2  # RENDER-FIT: reduce tube silhouette width
VIAL_LENGTH = 0.1400  # PHOTO-FIT: 140 mm, conditional on table prior
VIAL_MASS = 0.020
SORT_VIALS = [
    (0.423296, -0.218227, 39.638288),
    (0.399653, -0.007706, 10.367641),
]  # PHOTO-FIT; see calibration/uwm_20261008/fit.json

# -- cap (vial_cap.usda, 31.75 mm dia puck; lies flat)
CAP_RADIUS = 0.03175 / 2  # RENDER-FIT: CAD diameter matches real silhouette
CAP_HEIGHT = 0.01570
CAP_MASS = 0.005
SORT_CAPS = [
    (0.461165, 0.085596),
    (0.376209, -0.120583),
]  # PHOTO-FIT; see calibration/uwm_20261008/fit.json
# CAD customData: flange 1.250 in, plug .930 -> .820 in, flange .088 in.
# Preserve the real two-diameter geometry rather than shrinking the entire cap.
CAP_FLANGE_THICK = 0.088 * 0.0254
CAP_PLUG_LENGTH = CAP_HEIGHT - CAP_FLANGE_THICK
CAP_PLUG_BASE_RADIUS = 0.930 * 0.0254 / 2
CAP_PLUG_TIP_RADIUS = 0.820 * 0.0254 / 2
CAP_ORIENTATIONS = ["narrow_end_up", "narrow_end_up"]
# Hackathon task: all six cubes form one tower, bottom to top.
SORT_TASK = "cube_stack"
CUBE_STACK_XY = (0.28, 0.10)
CUBE_STACK_COLOURS = ("yellow", "yellow", "blue", "blue", "red", "red")
CUBE_STACK_SITE_TOLERANCE = CUBE_EDGE / 3
CUBE_STACK_LAYER_TOLERANCE = (
    CUBE_EDGE / 4
)  # adjacent centres retain at least 75% overlap per axis
CUBE_LAYOUT_SEED = 20261009


def hackathon_cubes(seed):
    """Reproducible random additions, separated from objects and the tower site.

    Sample a conservative workspace; host validation checks grasp/hover IK.
    The original two yellow cubes retain their calibrated positions.
    """
    import random
    import math

    rng = random.Random(seed)
    cubes = list(SORT_CUBES[:2])
    for colour in ("red", "red", "blue", "blue"):
        for _ in range(10000):
            xy = (rng.uniform(0.24, 0.46), rng.uniform(-0.15, 0.28))
            if any(
                math.dist(xy, p[:2]) < 0.065
                for p in cubes + SORT_CAPS + [CUBE_STACK_XY]
            ):
                continue
            blocked = False
            for x, y, yaw in SORT_VIALS:
                axis = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)))
                t = max(
                    -VIAL_LENGTH / 2,
                    min(VIAL_LENGTH / 2, (xy[0] - x) * axis[0] + (xy[1] - y) * axis[1]),
                )
                if math.dist(xy, (x + t * axis[0], y + t * axis[1])) < 0.055:
                    blocked = True
            if not blocked:
                cubes.append((*xy, colour))
                break
        else:
            raise ValueError("Cannot place separated cubes in the workspace")
    return cubes


SORT_CUBES = hackathon_cubes(CUBE_LAYOUT_SEED)
# Tip enters above the flange. The straight retaining lip grips the plug.
CAP_PICK_CLEARANCE = 0.001
CAP_PICK_HEIGHT = CAP_FLANGE_THICK + CAP_PICK_CLEARANCE
CAP_ORIENTATION_MIN_CONFIDENCE = 0.70
# Sorting destinations are flat surface slots, not the old absent sort bins.
SORT_VIAL_TARGETS = [(0.30, 0.08), (0.30, -0.08)]
SORT_CAP_TARGETS = [(x, y) for y in (0.32, 0.38) for x in (0.20, 0.26, 0.32, 0.38)]

# v0.5 measured work surface is 5.1 mm below the robot base mounting plane.
# TABLE_TOP_Z remains the robot base world Z; mat/table geometry follows this offset.
MAT_T = 0.002  # slab thickness; independent of signed base-to-surface offset
SURFACE_DZ = -0.0051

# ------------------------------------------------------------------ sort bins
# PHOTO 2026-10-08: sort bins and fixture are NOT present in the real calibration
# scene. Set these to True once placed on the table.
SHOW_BINS = False
SHOW_FIXTURE = False

# Three labelled bins, coloured by category.
# "vial" gets a larger pocket for the 27 mm body.
SORT_BINS = {
    "cube": {
        "xy": (0.200, 0.230),
        "outer": (0.062, 0.062),
        "pocket": 0.040,
        "pocket_r": 0.008,
        "wall_h": 0.060,
        "floor_t": 0.015,
        "mat": "plastic_red",
        "release_z": 0.092,
    },
    "cap": {
        "xy": (0.330, 0.230),
        "outer": (0.075, 0.075),
        "pocket": 0.050,
        "pocket_r": 0.010,
        "wall_h": 0.060,
        "floor_t": 0.015,
        "mat": "bin_yellow",
        "release_z": 0.092,
    },
    "vial": {
        "xy": (0.460, 0.220),
        "outer": (0.090, 0.090),
        "pocket": 0.065,
        "pocket_r": 0.012,
        "wall_h": 0.065,
        "floor_t": 0.015,
        "mat": "plastic_blue",
        "release_z": 0.095,
    },
}
# Legacy aliases so code that still references the old single-bin names compiles
BIN_XY = SORT_BINS["cube"]["xy"]
BIN_SIZE = (SORT_BINS["cube"]["pocket"], SORT_BINS["cube"]["pocket"])
BIN_RELEASE_Z = SORT_BINS["cube"]["release_z"]

# ------------------------------------------------------------------ cameras
# Cameras are specified as eye + look_at in the parent's frame, not Euler
# angles -- it is the only formulation that stays readable while iterating.

# The wrist camera lens position: camera bottom is CAM_ABOVE_TIP above the
# fingertip; optical centre is at the lens front, outside the opaque housing.
_WC_AT = TOOL_LENGTH - CAM_ABOVE_TIP + 0.006
_WC_EYE = tuple(
    a * _WC_AT + d * WRISTCAM_LATERAL for a, d in zip(TOOL_AXIS, WRISTCAM_DIR)
)
WRIST_CAM = dict(
    parent=ROBOT_PRIM + "/J6_link",
    eye=(0.1633, 0.0125, 0.0971),
    look_at=(0.3632, 0.0180, 0.0930),
    up=(0.0198, 0.0231, 0.9995),  # flange coordinates
    resolution=(1280, 720),
    focal_length=10.1440,  # fx 619.6 px at horiz_aperture 20.955
    horiz_aperture=20.955,
)

# v0.5 ArUco calibration, supplied pinhole approximation. Table/wrist mounts
# were not physically revalidated after the testbed rebuild; see docs/CALIBRATION_V05.md.
TABLE_CAM = dict(
    parent="/World",
    eye=(0.7770, -0.0426, TABLE_TOP_Z + 0.8024),
    look_at=(0.1636, -0.0210, TABLE_TOP_Z),
    up=(-0.7943, -0.0398, 0.6062),
    resolution=(1920, 1080),
    focal_length=11.2158,  # 1027.6 px at horiz_aperture 20.955
    horiz_aperture=20.955,
    fisheye=False,
)

# v0.5 post-rebuild ArUco side-camera solve, reprojection RMS 3.99 px.
SIDE_CAM = dict(
    parent="/World",
    eye=(0.4577, 0.6287, TABLE_TOP_Z + 0.1378),
    look_at=(0.4350, 0.1293, TABLE_TOP_Z + 0.1409),
    up=(0.0293, 0.0050, 0.9996),
    resolution=(1920, 1080),
    focal_length=12.9427,  # 1185.9 px at horiz_aperture 20.955
    horiz_aperture=20.955,
    fisheye=False,
)

# VIDEO: both cameras frame this region.
WORK_AREA_CENTER = (0.40, 0.0, 0.0)

# PHOTO-FIT geometry is conditional on the retained table camera calibration.
# CAD visuals scale with their colliders; source USDs stay intact.
SHOW_BACKDROP = True
BACKDROP_WIDTH = 0.94
# User update 2026-10-09: paper was cut and laid flat. Retain the previous
# tabletop footprint; the exact cut dimensions have not been measured.
BACKDROP_X = (0.18, TABLE_X[1] - EDGE_BAND_W)
BACKDROP_ROUGHNESS = 0.30

# RENDER-FIT snapshots hold the reference arrangement during shader warm-up.
# Normal demo / interactive builds continue to use dynamic task objects.
CALIBRATION_SNAPSHOT = True


# Square yaw is observable modulo 90 degrees; these grasps require wrist turns.
SORT_CUBE_YAWS_DEG = [0.0, 26.0, 32.0, -23.0, 14.0, -36.0]

assert abs(TOOL_LENGTH - 0.293) < 1e-9, "v0.5 calibrated TCP length"
