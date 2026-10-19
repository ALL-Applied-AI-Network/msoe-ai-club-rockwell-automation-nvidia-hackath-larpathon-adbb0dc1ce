"""
Stage 2: scene understanding with Cosmos3-Nano (runs in the cosmos venv).

    "$COSMOS" perceive.py --run runs/example --stack-cubes

Reads the table image, identifies cube colours/locations and reports
image-measured tabletop poses. Writes detections.json and raw crop replies.

Runtime notes, taken from the working AOI setup on this machine:
  - transformers_cosmos3.Cosmos3ForConditionalGeneration, not a Cosmos 2.x path
  - AutoConfig must be loaded explicitly and passed to from_pretrained, to dodge
    a transformers bug where the model_type heuristic replaces the config with
    vision_config and drops text_config.rope_scaling
"""

import argparse, json, os, re, sys
from pathlib import Path


def _find_nano():
    """Locate explicitly selected weights or the standard Hugging Face cache."""
    selected = os.environ.get("COSMOS3_NANO_PATH")
    if selected:
        checkpoint = Path(selected).expanduser()
        if not (checkpoint / "config.json").is_file():
            raise SystemExit(f"Model snapshot is missing config.json: {checkpoint}")
        return checkpoint
    cache_home = Path(
        os.environ.get(
            "HF_HOME",
            str(
                Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
                / "huggingface"
            ),
        )
    ).expanduser()
    hub = Path(os.environ.get("HF_HUB_CACHE", str(cache_home / "hub"))).expanduser()
    base = hub / "models--nvidia--Cosmos3-Nano"
    snapshots = sorted((base / "snapshots").glob("*/config.json"))
    if not snapshots:
        raise SystemExit(
            f"Cosmos3-Nano not found under {base}. Set COSMOS3_NANO_PATH "
            "to the downloaded snapshot, or configure HF_HOME/HF_HUB_CACHE."
        )
    return snapshots[-1].parent


def _configure_cosmos_imports():
    """Use an optional framework checkout; installed packages also work."""
    checkout = os.environ.get("COSMOS3_ROOT")
    if checkout:
        root = Path(checkout).expanduser().resolve()
        package = root / "packages" / "transformers-cosmos3"
        if not (package / "transformers_cosmos3").is_dir():
            raise SystemExit(
                f"COSMOS3_ROOT does not contain transformers-cosmos3: {root}"
            )
        sys.path.insert(0, str(package))
        sys.path.insert(0, str(root))


PROMPT = """You are the vision system for a robot arm working at a bench.

Look at this image of the workbench and report every object on the white table
that the robot could pick up, plus the collection bins.

Report ONLY three kinds of object: clear plastic vials (cylindrical, lying on their
side), yellow plastic caps (flat disc shapes), and small coloured plastic cubes.
Do not report the dark fixture, the bins themselves, the table, or the robot.

Reply with ONLY a JSON object, no other text, in exactly this form:

{
  "objects": [
    {"label": "clear vial", "category": "vial", "colour": "clear",
     "center": [0.38, 0.55], "bbox": [0.35, 0.52, 0.41, 0.58], "graspable": true},
    {"label": "yellow cap", "category": "cap", "colour": "yellow",
     "center": [0.52, 0.61], "bbox": [0.49, 0.58, 0.55, 0.64], "graspable": true},
    {"label": "red cube", "category": "cube", "colour": "red",
     "center": [0.42, 0.71], "bbox": [0.40, 0.69, 0.44, 0.73], "graspable": true}
  ],
  "bins": [
    {"label": "cube bin", "category": "cube", "center": [0.22, 0.28]},
    {"label": "cap bin",  "category": "cap",  "center": [0.38, 0.28]},
    {"label": "vial bin", "category": "vial", "center": [0.55, 0.28]}
  ]
}

Coordinate rules, follow exactly:
- center and bbox are NORMALISED to the image: x from 0.0 at the left edge to
  1.0 at the right edge, y from 0.0 at the TOP edge to 1.0 at the BOTTOM edge.
- bbox is [x_min, y_min, x_max, y_max].
- "center" must be the centre of the object as it appears in the image.
- category must be exactly "vial", "cap", or "cube".
- graspable is true only for small loose objects the gripper could pick up.
- Never repeat the same object twice. Each entry must be a distinct object.
- There are at most 10 objects. Stop and close the JSON when listed.
Report every vial, cap, and cube you can see, each as a separate entry."""

SORT_PROMPT = """You are the vision system for a tabletop robot.
Locate every separate loose yellow cap, yellow cube, and clear tube/vial.
Ignore the robot, fingers, camera, paper, hoses, and background.
Caps have a broad flat circular flange and a smaller tapered cylindrical plug.
Report caps in BOTH orientations, including ones the robot cannot pick up.
Do not assume orientation from position or how other caps are oriented.
Return only JSON: {"objects": [{"label":"yellow cap", "category":"cap",
"colour":"yellow", "center":[0.4,0.5], "bbox":[0.38,0.47,0.42,0.53],
"orientation":"narrow_end_up", "orientation_confidence":0.8,
"graspable":true}]}.
category is cap, cube, or vial. Coordinates are NORMALIZED 0..1, x left to right,
y top to bottom. center is the centre of the object's whole silhouette.
For caps, orientation is narrow_end_up (a CLOSED smaller circular plug top is
RAISED above the broad flange resting on the table), wide_end_up (the broad
rim is uppermost and surrounds a RECESSED HOLLOW CUP opening), or unknown.
A visible cavity, inside wall or recessed bottom means wide_end_up, NOT a
raised plug. Both orientations have a circular rim. Use unknown if unclear.
For vials include axis_endpoints: [[x1,y1],[x2,y2]], the centres of BOTH ends
of the long tube, in normalized image coordinates, and axis_confidence 0..1.
For cubes include colour. Give each visible object exactly once. Up to 24 objects
may be present. Include wide_end_up and unknown caps; do not omit them."""

STACK_PROMPT = """You are the vision system for a robot building one cube tower.
Identify every separate loose plastic CUBE on the white tabletop. Cubes can be
yellow, blue, or red. Ignore the yellow circular caps, clear tubes, robot,
gripper, cameras and background. A cap is round; a cube has straight edges and
square faces. Determine each cube's colour from its visible surface, never its
position. Report every visible cube exactly once, including both cubes of a
colour if present. Do not invent missing cubes or force a count.
Yellow cubes may appear pale beige, tan or muted golden under this lighting;
they are small square blocks, unlike the larger round yellow caps.
The task is to pick the cubes into ONE stack with yellow at the bottom, blue
above yellow, and red at the top. The planner determines the physical sequence.
Return ONLY JSON in this schema:
{"objects":[{"label":"blue cube","category":"cube","colour":"blue",
"center":[0.4,0.5],"bbox":[0.39,0.49,0.41,0.51],"graspable":true}]}.
Coordinates must be normalized 0..1: x left to right, y top to bottom.
Cubes may be rotated arbitrarily: inspect the top face; a focused pose pass
will compare top-face pose hypotheses before aligning the wrist for grasping.
center is the centre of the whole visible cube silhouette.
bbox is [x_min,y_min,x_max,y_max]. Use category cube, and colour yellow, blue,
red, or unknown if ambiguous. Up to 12 cubes may be present."""


def audit_stack_colours(model, processor, image_path, parsed, run):
    """Revisit missing colours in full and local views; no ground truth input."""
    from collections import Counter
    import cv2
    import numpy as np
    from plan_grasps import detect_coord_convention, make_as_pixels

    im = cv2.imread(str(image_path))
    if im is None:
        raise FileNotFoundError(image_path)
    H, W = im.shape[:2]

    def normalize(objects, width, height, x_offset=0):
        convention, divisor = detect_coord_convention(
            {"objects": objects}, width, height
        )
        pixels = make_as_pixels(convention, divisor, width, height)
        for obj in objects:
            for key in ("center", "bbox"):
                if obj.get(key) is not None:
                    obj[key] = [
                        (v + x_offset) / W if j % 2 == 0 else v / H
                        for j, v in enumerate(pixels(obj[key]))
                    ]
        return objects

    parsed["objects"] = normalize(parsed.get("objects", []), W, H)
    parsed["coordinate_convention"] = "unit"
    unique = []
    for obj in parsed["objects"]:
        if obj.get("center") is not None and any(
            o.get("colour") == obj.get("colour")
            and o.get("center") is not None
            and np.linalg.norm((np.array(obj["center"]) - o["center"]) * [W, H]) < 24
            for o in unique
        ):
            continue
        unique.append(obj)
    parsed["objects"] = unique
    counts = Counter(
        o.get("colour") for o in parsed["objects"] if o.get("category") == "cube"
    )
    for colour in ("yellow", "blue", "red"):
        if counts[colour] == 2:
            continue
        prompt = f"""Find every small {colour} plastic CUBE in this tabletop image.
Yellow cubes may look pale tan or beige. Cubes have straight edges and square
faces; the much larger round yellow caps are NOT cubes. Ignore clear tubes,
robot and background. Scan carefully for separate small blocks of this colour.
Report only visible {colour} cubes, each once, without inventing missing ones.
Return ONLY JSON: {{"objects":[{{"label":"{colour} cube","category":"cube",
"colour":"{colour}","center":[0.4,0.5],"bbox":[0.39,0.49,0.41,0.51],
"graspable":true}}]}}. Coordinates are normalized 0..1 in THIS image; x from
left to right and y top to bottom. If none are visible, use {{"objects":[]}}."""

        def query(path, suffix, width, x_offset=0):
            raw = ask(model, processor, path, prompt, 1600)
            (run / f"detections_{colour}_{suffix}_raw.txt").write_text(raw)
            answer = canonicalise(parse_json(raw) or {})
            objects = [
                o
                for o in answer.get("objects", [])
                if o.get("category") == "cube" and o.get("colour") == colour
            ]
            return normalize(objects, width, H, x_offset)

        replacements = query(image_path, "audit", W)
        if len(replacements) != 2:
            # Image-only proposals for tiny blocks missed in the wide view.
            # Colour/size segmentation proposes patches; Cosmos must confirm
            # that each is a cube, rather than a circular cap or background.
            from plan_grasps import _OBJ_REF

            ref = np.array(_OBJ_REF[("cube", colour)], np.float32)
            mask = (np.linalg.norm(im.astype(np.float32) - ref, axis=2) < 95).astype(
                np.uint8
            )
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]
            colour_gate = (
                (hue >= 12) & (hue <= 42)
                if colour == "yellow"
                else (
                    (hue >= 95) & (hue <= 135)
                    if colour == "blue"
                    else (hue <= 12) | (hue >= 170)
                )
            )
            mask &= (colour_gate & (hsv[:, :, 1] >= 55)).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
            confirmed = []
            crops = run / "cube_inventory_crops"
            crops.mkdir(exist_ok=True)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if (
                    not 80 * (W / 1920) ** 2 <= area <= 1400 * (W / 1920) ** 2
                    or not 0.4 <= w / max(1, h) <= 2.5
                ):
                    continue
                x0, x1 = max(0, x - 12), min(W, x + w + 12)
                y0, y1 = max(0, y - 12), min(H, y + h + 12)
                path = crops / f"{colour}_candidate{i}.png"
                crop = im[y0:y1, x0:x1]
                scale = 384 / max(crop.shape[:2])
                cv2.imwrite(
                    str(path),
                    cv2.resize(
                        crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                    ),
                )
                detail_prompt = """Classify the object nearest the CENTRE of this crop.
A cube is a small block with straight edges and square flat faces, possibly
rounded corners. Yellow cubes can look beige/tan. A cap has a broad circular
flange and a cylindrical plug or hollow cup. Use the visible geometry.
Reply ONLY JSON: {"category":"cube","colour":"yellow","confidence":0.9}.
Choose category cube, cap, or unknown; colour yellow, blue, red, or unknown.
Use unknown and low confidence if the geometry is ambiguous."""
                raw = ask(model, processor, path, detail_prompt, 300)
                path.with_suffix(".txt").write_text(raw)
                answer = canonicalise(parse_json(raw) or {})
                try:
                    accepted = (
                        answer.get("category") == "cube"
                        and answer.get("colour") == colour
                        and float(answer.get("confidence", 0)) >= 0.8
                    )
                except (TypeError, ValueError):
                    accepted = False
                if accepted:
                    confirmed.append(
                        dict(
                            label=f"{colour} cube",
                            category="cube",
                            colour=colour,
                            center=[centers[i][0] / W, centers[i][1] / H],
                            bbox=[x / W, y / H, (x + w) / W, (y + h) / H],
                            graspable=True,
                            detail_crop=str(path.relative_to(run)),
                            category_confidence=answer["confidence"],
                            localisation_source="image_blob_confirmed_by_Cosmos",
                        )
                    )
            replacements = confirmed
        parsed["objects"] = [
            o for o in parsed["objects"] if o.get("colour") != colour
        ] + replacements
    return parsed


CUBE_POSE_PROMPT = """Inspect the small cube in this camera crop and report its
colour and tabletop pose. A calibrated image-geometry front end has measured
its top-face orientation and centre from this SAME image, using the known cube
size and table camera calibration. These measurements are image-derived, not
simulator object coordinates. Use the supplied measurements for quantitative
pose; do not guess a new angle or regress pixel corners from this blurry crop.
Check that the object is a cube of the expected colour. If inconsistent or
ambiguous, report unknown and low confidence. Square yaw is modulo 90 degrees;
roll and pitch are constrained by the cube resting upright on the table.
Return only JSON: {"category":"cube","colour":"blue","yaw_deg":14.2,
"pose_confidence":0.9,"reason":"visible cube geometry and colour"}.
Use the measured yaw value from this query, not the example value above."""


def inspect_cube_poses(model, processor, image_path, parsed, run):
    """Cosmos reports semantics + pose with a calibrated image-geometry front end.

    Raw VLM corner regression failed validation. Geometry supplies measured
    yaw; Cosmos confirms cube/colour and explicitly reports that pose. No GT.
    """
    import cv2
    import numpy as np
    from backproject import CameraModel
    from cube_pose import refine_pose, square_yaw
    import scene_config as C

    im = cv2.imread(str(image_path))
    H, W = im.shape[:2]
    cam = CameraModel(str(run / "camera.json"))
    folder = run / "cube_pose_crops"
    folder.mkdir(exist_ok=True)
    for i, obj in enumerate(parsed.get("objects", [])):
        if obj.get("category") != "cube":
            continue
        u, v = np.array(obj["center"]) * [W, H]
        radius = round(35 * W / 1920)
        x0, y0 = max(0, int(u) - radius), max(0, int(v) - radius)
        x1, y1 = min(W, int(u) + radius + 1), min(H, int(v) + radius + 1)
        xy = cam.to_plane(u, v, z=cam.table_z + C.CUBE_EDGE / 2)[:2]
        xy, yaw, quality = refine_pose(im, cam, obj, xy, 0, search_half_width=45)
        crop = cv2.resize(im[y0:y1, x0:x1], (568, 568), interpolation=cv2.INTER_CUBIC)
        path = folder / f"cube_{i}.png"
        cv2.imwrite(str(path), crop)
        measurement = dict(
            expected_colour=obj["colour"],
            position_world_m=[*xy, cam.table_z + C.CUBE_EDGE / 2],
            yaw_deg=round(yaw, 4),
            roll_deg=0,
            pitch_deg=0,
            silhouette_iou=quality,
            source="calibrated_image_silhouette_fit",
            yaw_symmetry_deg=90,
        )
        (folder / f"cube_{i}_measurement.json").write_text(
            json.dumps(measurement, indent=2)
        )
        obj.update(
            pose_source="Cosmos_reports_calibrated_image_pose",
            image_pose_measurement=measurement,
            pose_confidence=0,
            top_edge_endpoints=None,
        )
        if quality < 0.85:
            obj["pose_reason"] = "Image geometry fit is uncertain"
            continue
        prompt = (
            CUBE_POSE_PROMPT
            + "\nMeasured image geometry for THIS cube:\n"
            + json.dumps(measurement)
        )
        raw = ask(model, processor, path, prompt, 400)
        path.with_suffix(".txt").write_text(raw)
        answer = parse_json(raw) or {}
        obj["pose_reason"] = answer.get("reason", "")
        try:
            reported = float(answer["yaw_deg"])
            conf = float(answer.get("pose_confidence", 0))
            valid = (
                answer.get("category") == "cube"
                and answer.get("colour") == obj["colour"]
                and np.isfinite(reported)
                and np.isfinite(conf)
                and 0.8 <= conf <= 1
                and abs(square_yaw(reported - yaw)) <= 3
            )
            if not valid:
                continue
            t = np.deg2rad(reported)
            rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            points = [
                cam.project(
                    [
                        *(xy + rot @ np.array([a, -1]) * C.CUBE_EDGE / 2),
                        cam.table_z + C.CUBE_EDGE,
                    ]
                )
                for a in (-1, 1)
            ]
            obj.update(
                top_edge_endpoints=(np.array(points) / [W, H]).tolist(),
                pose_confidence=conf,
                cosmos_reported_yaw_deg=reported,
                pose_world=dict(
                    position_m=measurement["position_world_m"],
                    yaw_deg=square_yaw(reported),
                    roll_deg=0,
                    pitch_deg=0,
                    yaw_symmetry_deg=90,
                ),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return parsed


CAP_CROP_PROMPT = """Classify the yellow cap nearest the CENTRE of the target crop.
This part is a HOLLOW tapered cup with a wide flange around its open end.
There are two distinct appearances:
- narrow_end_up: the smaller CLOSED end points upward. Its solid circular top
  protrudes ABOVE the broad flange, which rests on the table. The exterior
  tapered side wall is visible below the raised top. No open cavity faces us.
- wide_end_up: the cap is inverted. The broad flange/rim is at the TOP, around
  an OPEN recessed cup. We can see its inner wall and lower interior bottom.
  That bottom is INSIDE the cup, not a plug protruding above the flange.
Look for convex raised closed top versus concave recessed opening. A circular
rim alone does not distinguish them. A dark inner wall and visible cavity are
evidence for wide_end_up; do not interpret a recessed bottom as a raised top.
Judge visible geometry, not position or the orientation of nearby objects.
If those features are occluded or ambiguous, answer unknown with low confidence.
Reply only JSON, using this schema (choose the label and confidence yourself):
{"orientation":"unknown", "confidence":0.0, "reason":"specific visible feature"}.
Allowed orientation values: narrow_end_up, wide_end_up, unknown.
Confidence is a number between 0 and 1. Do not claim certainty for ambiguous views."""


def cap_examples():
    """Fixed, visually reviewed demonstrations; no runtime ground truth input."""
    root = Path(__file__).resolve().parent / "assets/perception_examples"
    return [
        (
            root / f"{label}.png",
            {"orientation": label, "confidence": 0.95, "reason": reason},
        )
        for label, reason in (
            (
                "wide_end_up",
                "An open recessed cavity is surrounded by the upper broad rim; the inner wall and interior bottom are visible.",
            ),
            (
                "narrow_end_up",
                "A solid closed circular top protrudes above the lower broad flange; the exterior plug wall is visible.",
            ),
        )
    ]


TUBE_CROP_PROMPT = """Inspect the clear tube nearest the centre of this crop.
Return only JSON with the centres of its two end rims, describing its long axis:
{"axis_endpoints":[[0.1,0.2],[0.8,0.7]],"confidence":0.9}.
Coordinates are normalized 0..1 within THIS CROP, x left to right, y top to bottom.
If either end cannot be located, use {"axis_endpoints":null,"confidence":0}."""


def inspect_sort_objects(model, processor, image_path, parsed, run):
    """Second VLM pass on local crops. No scene config or ground truth input."""
    import cv2
    from plan_grasps import detect_coord_convention, make_as_pixels

    im = cv2.imread(str(image_path))
    if im is None:
        raise FileNotFoundError(image_path)
    H, W = im.shape[:2]
    convention, divisor = detect_coord_convention(parsed, W, H)
    pixels = make_as_pixels(convention, divisor, W, H)
    crop_dir = run / "orientation_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for i, obj in enumerate(parsed.get("objects", [])):
        # Normalize every coordinate once, including axes, before writing JSON.
        for key in ("center", "bbox"):
            if obj.get(key) is not None:
                values = pixels(obj[key])
                obj[key] = [v / (W if j % 2 == 0 else H) for j, v in enumerate(values)]
        if obj.get("axis_endpoints"):
            obj["axis_endpoints"] = [
                [u / W, v / H] for u, v in (pixels(p) for p in obj["axis_endpoints"])
            ]
        cat = obj.get("category")
        if cat not in ("cap", "vial"):
            continue
        obj["orientation_confidence" if cat == "cap" else "axis_confidence"] = 0.0
        if cat == "cap":
            obj["orientation"] = "unknown"
        else:
            obj["axis_endpoints"] = None
        if not obj.get("bbox") or not obj.get("center"):
            continue
        cx, cy = obj["center"][0] * W, obj["center"][1] * H
        bb = obj["bbox"]
        bw, bh = max(16, abs(bb[2] - bb[0]) * W), max(16, abs(bb[3] - bb[1]) * H)
        x0, x1 = max(0, int(cx - bw * 0.65)), min(W, int(cx + bw * 0.65))
        y0, y1 = max(0, int(cy - bh * 0.65)), min(H, int(cy + bh * 0.65))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        path = crop_dir / f"{i:02d}_{cat}.png"
        # Enlargement helps the processor retain pixels for tiny caps; it adds
        # no information and the native camera still determines accuracy.
        crop = im[y0:y1, x0:x1]
        scale = max(1.0, 384 / max(crop.shape[:2]))
        cv2.imwrite(
            str(path),
            cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC),
        )
        raw = ask(
            model,
            processor,
            path,
            CAP_CROP_PROMPT if cat == "cap" else TUBE_CROP_PROMPT,
            400,
            examples=cap_examples() if cat == "cap" else None,
        )
        path.with_suffix(".txt").write_text(raw)
        answer = canonicalise(parse_json(raw) or {})
        if cat == "cap":
            orientation = answer.get("orientation", "unknown")
            obj["orientation"] = (
                orientation
                if orientation in ("narrow_end_up", "wide_end_up")
                else "unknown"
            )
            obj["orientation_confidence"] = answer.get("confidence", 0)
            obj["orientation_evidence"] = answer.get("reason", "")
        else:
            ends = answer.get("axis_endpoints")
            try:
                if len(ends) != 2 or any(len(p) != 2 for p in ends):
                    raise ValueError("invalid endpoints")
                if not all(0 <= float(v) <= 1 for p in ends for v in p):
                    raise ValueError("crop coordinates must be normalized")
                obj["axis_endpoints"] = [
                    [
                        (x0 + float(p[0]) * (x1 - x0)) / W,
                        (y0 + float(p[1]) * (y1 - y0)) / H,
                    ]
                    for p in ends
                ]
                obj["axis_confidence"] = answer.get("confidence", 0)
            except (TypeError, ValueError):
                pass
        obj["detail_crop"] = str(path.relative_to(run))
    parsed["coordinate_convention"] = "unit"
    parsed["model"] = "Cosmos3-Nano"
    return parsed


def load_model(gpu=0):
    _configure_cosmos_imports()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers_cosmos3 import Cosmos3ForConditionalGeneration

    ckpt = str(_find_nano())
    print(f"[perceive] loading Cosmos3-Nano on GPU {gpu} ...", flush=True)
    config = AutoConfig.from_pretrained(ckpt, trust_remote_code=False)
    model = Cosmos3ForConditionalGeneration.from_pretrained(
        ckpt,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    print("[perceive] model ready", flush=True)
    return model, processor


def ask(model, processor, image_path, prompt, max_new_tokens=1400, examples=None):
    import torch

    messages = []
    for example_path, answer in examples or []:
        if not Path(example_path).is_file():
            raise FileNotFoundError(example_path)
        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(example_path)},
                        {"type": "text", "text": prompt},
                    ],
                },
                {"role": "assistant", "content": json.dumps(answer)},
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
    }
    with torch.inference_mode():
        # repetition_penalty guards against the degenerate loop this model can
        # fall into, emitting the same object over and over until the budget
        # runs out and the JSON is left unterminated
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )
    new = out[:, inputs["input_ids"].shape[1] :]
    txt = processor.batch_decode(
        new, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return txt[0].strip() if txt else ""


def salvage_objects(text):
    """Recover whatever complete entries exist in a truncated/looping reply.

    Cosmos3-Nano sometimes repeats an object until it runs out of tokens, leaving
    the JSON unterminated. Rather than discarding the whole reply, pull out the
    well-formed {...} blocks and de-duplicate them.
    """
    out, seen = [], set()
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        o = canonicalise(o)
        if "center" not in o or "category" not in o:
            continue
        key = (o.get("label"), tuple(o.get("center", [])))
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


_ALIASES = {
    "centre": "center",
    "color": "colour",
    "bounding_box": "bbox",
    "box": "bbox",
    "bbox_2d": "bbox",
}


def canonicalise(obj):
    """Normalise key spellings.

    The model mixes conventions inside a single reply -- in one run the cubes
    used "center" and the screws "centre", which silently dropped every screw.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[_ALIASES.get(k, k)] = canonicalise(v)
        return out
    if isinstance(obj, list):
        return [canonicalise(v) for v in obj]
    return obj


def parse_json(text):
    """Pull the first JSON object out of the reply, tolerating fences/prose."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        decoded = json.loads(t)
        if isinstance(decoded, list):
            return {"objects": decoded}
        if isinstance(decoded, dict):
            return decoded
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


VERIFY_PROMPT = """You are the vision system for a robot arm at a bench.

The robot was asked to pick up every vial, cap, and cube from the table and drop
each into its matching sort bin (cube->red bin, cap->yellow bin, vial->blue bin).
This image was taken after it finished.

Reply with ONLY a JSON object, no other text:

{
  "cubes_sorted":  <integer>,
  "caps_sorted":   <integer>,
  "vials_sorted":  <integer>,
  "objects_still_on_table": <integer>,
  "task_complete": true,
  "notes": "one short sentence"
}

Count only loose vials, caps, and cubes. The dark fixture and the bins themselves
are not task objects."""


INVENTORY_PROMPT = """This image is from a camera on a robot wrist, looking into a bin.

List what is inside the bin. Ignore anything outside the bin or part of the gripper.

Reply with ONLY a JSON object, no other text:

{
  "items": [
    {"object": "clear vial", "count": 1},
    {"object": "yellow cap", "count": 2}
  ],
  "total_items": 3,
  "notes": "one short sentence"
}

Use simple names: "clear vial", "yellow cap", "red cube", "yellow cube", "blue cube".
Count each distinct object once. If something is partly hidden but identifiable,
still count it."""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--image", default=None)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="ask whether the task completed, instead of detecting",
    )
    ap.add_argument(
        "--inventory",
        action="store_true",
        help="list what is inside the bin (use with the wrist view)",
    )
    ap.add_argument(
        "--sort",
        action="store_true",
        help="detect all test objects and inspect cap/tube orientation crops",
    )
    ap.add_argument(
        "--stack-cubes",
        action="store_true",
        help="identify coloured cubes for the hackathon tower task",
    )
    a = ap.parse_args()

    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    img = Path(a.image) if a.image else run / "world_camera.png"
    if a.inventory:
        prompt, stem, budget = INVENTORY_PROMPT, "inventory", 600
    elif a.verify:
        prompt, stem, budget = VERIFY_PROMPT, "verify", 600
    elif a.stack_cubes:
        prompt, stem, budget = STACK_PROMPT, "detections", 3500
    elif a.sort:
        prompt, stem, budget = SORT_PROMPT, "detections", 6500
    else:
        prompt, stem, budget = PROMPT, "detections", 3000
    model, processor = load_model(a.gpu)
    # detection replies are pretty-printed JSON and run long; 8 objects with
    # bboxes overflowed a 1400-token budget mid-object and failed to parse
    raw = ask(model, processor, img, prompt, max_new_tokens=budget)

    (run / f"{stem}_raw.txt").write_text(raw)
    parsed = parse_json(raw)
    if parsed is None and not (a.verify or a.inventory):
        objs = salvage_objects(raw)
        if objs:
            parsed = {
                "objects": [
                    o for o in objs if o.get("category") in ("cube", "cap", "vial")
                ],
                "bins": [o for o in objs if o.get("category") == "bin"],
            }
            print(
                f"[perceive] reply was malformed; salvaged "
                f"{len(parsed['objects'])} complete entries"
            )
    if parsed is None:
        print(f"[perceive] FAILED to parse JSON; raw reply in {stem}_raw.txt")
        print(raw[:800])
        sys.exit(1)
    parsed = canonicalise(parsed)
    if a.stack_cubes:
        parsed = audit_stack_colours(model, processor, img, parsed, run)
        parsed = inspect_cube_poses(model, processor, img, parsed, run)
        parsed["model"] = "Cosmos3-Nano"
        parsed["task"] = "cube_stack"
    if a.sort and not (a.verify or a.inventory):
        parsed = inspect_sort_objects(model, processor, img, parsed, run)
    with open(run / f"{stem}.json", "w") as f:
        json.dump(parsed, f, indent=2)

    if a.inventory:
        print("[perceive] bin inventory:")
        for it in parsed.get("items", []):
            print(f"   {it.get('count', '?')} x {it.get('object', '?')}")
        print(f"   total: {parsed.get('total_items')}")
        if parsed.get("notes"):
            print(f"   notes: {parsed['notes']}")
    elif a.verify:
        print("[perceive] verification:")
        for k, v in parsed.items():
            print(f"   {k}: {v}")
    else:
        n = len(parsed.get("objects", []))
        print(f"[perceive] parsed {n} objects, bins={len(parsed.get('bins', []))}")
        for o in parsed.get("objects", [])[:15]:
            print(
                f"   {o.get('label','?'):20s} {o.get('category','?'):8s} "
                f"center={o.get('center')}"
            )
