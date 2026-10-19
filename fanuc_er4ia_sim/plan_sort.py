"""Plan sorting from Cosmos observations; ground truth is used only for scoring."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import scene_config as C
from backproject import CameraModel
from cube_pose import edge_pose, refine_pose, square_yaw
from plan_grasps import detect_coord_convention, make_as_pixels, refine_centre


def confidence(value):
    try:
        value = float(value)
        return value if np.isfinite(value) and 0 <= value <= 1 else 0.0
    except (ValueError, TypeError):
        return 0.0


def cap_rings(orientation):
    """Outer silhouette rings in metres above the support surface."""
    if orientation == "narrow_end_up":
        return [
            (0, C.CAP_RADIUS),
            (C.CAP_FLANGE_THICK, C.CAP_RADIUS),
            (C.CAP_FLANGE_THICK, C.CAP_PLUG_BASE_RADIUS),
            (C.CAP_HEIGHT, C.CAP_PLUG_TIP_RADIUS),
        ]
    return [
        (0, C.CAP_PLUG_TIP_RADIUS),
        (C.CAP_PLUG_LENGTH, C.CAP_PLUG_BASE_RADIUS),
        (C.CAP_PLUG_LENGTH, C.CAP_RADIUS),
        (C.CAP_HEIGHT, C.CAP_RADIUS),
    ]


def cap_silhouette(cam, xy, orientation, shape, origin=(0, 0)):
    """Project the compound cap silhouette; independent of simulator state."""
    mask = np.zeros(shape, np.uint8)
    rings = []
    for z, r in cap_rings(orientation):
        points = [
            cam.project([xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), cam.table_z + z])
            for a in np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ]
        if any(p is None for p in points):
            return mask
        ring = np.round(np.array(points) - origin).astype(np.int32)
        rings.append(ring)
        cv2.fillPoly(mask, [ring], 1)
    for a, b in zip(rings, rings[1:]):
        cv2.fillConvexPoly(mask, cv2.convexHull(np.concatenate([a, b])), 1)
    return mask


def refine_cap(img, cam, obj, seed):
    """Fit known cap silhouette to yellow pixels in the VLM's own local box.

    Orientation comes exclusively from Cosmos; the fit only refines translation.
    Modeling both heights avoids back-projecting a flange centroid as a top face.
    """
    H, W = img.shape[:2]
    bb = np.array(obj.get("bbox", []), float)
    if bb.shape != (4,) or not np.all(np.isfinite(bb)):
        return seed, 0.0
    bb *= [W, H, W, H]
    x0, y0 = np.maximum(0, np.floor(bb[:2] - 12)).astype(int)
    x1, y1 = np.minimum([W, H], np.ceil(bb[2:] + 12)).astype(int)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return seed, 0.0
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    observed = cv2.inRange(hsv, (14, 65, 45), (42, 255, 255)) > 0
    if observed.sum() < 15:
        return seed, 0.0
    orientation = obj.get("orientation", "unknown")
    best_xy, best_score = np.array(seed), -1.0
    for step, radius in ((0.003, 0.015), (0.0005, 0.002)):
        centre = best_xy.copy()
        for dx in np.arange(-radius, radius + step / 2, step):
            for dy in np.arange(-radius, radius + step / 2, step):
                xy = centre + [dx, dy]
                predicted = (
                    cap_silhouette(cam, xy, orientation, observed.shape, (x0, y0)) > 0
                )
                union = np.logical_or(predicted, observed).sum()
                score = np.logical_and(predicted, observed).sum() / max(1, union)
                if score > best_score:
                    best_xy, best_score = xy, float(score)
    return best_xy, best_score


def build_plan(det, cam, img=None, task=None):
    """No ground-truth argument: object eligibility and poses come from vision."""
    task = task or C.SORT_TASK
    if task not in ("sort", "cube_stack"):
        raise ValueError("Unknown task")
    convention, divisor = detect_coord_convention(det, cam.W, cam.H)
    if det.get("coordinate_convention") == "unit":
        convention, divisor = "unit", 1.0
    pixels = make_as_pixels(convention, divisor, cam.W, cam.H)
    observations, rejected = [], []
    for i, original in enumerate(det.get("objects", [])):
        obj = dict(original)
        obj["detection_id"] = i
        cat = obj.get("category")
        reason = None
        if task == "cube_stack" and cat != "cube":
            obj.update(eligible=False, skip_reason="not_a_cube")
            observations.append(obj)
            rejected.append(obj)
            continue
        try:
            uv = np.array(pixels(obj["center"]), float)
            if (
                uv.shape != (2,)
                or not np.all(np.isfinite(uv))
                or not (0 <= uv[0] < cam.W and 0 <= uv[1] < cam.H)
            ):
                raise ValueError("invalid centre")
            if obj.get("bbox"):
                bb = pixels(obj["bbox"])
                obj["bbox"] = [
                    v / (cam.W if j % 2 == 0 else cam.H) for j, v in enumerate(bb)
                ]
            h = {
                "cap": C.CAP_HEIGHT / 2,
                "cube": C.CUBE_EDGE / 2,
                "vial": C.VIAL_RADIUS + 0.001,
            }.get(cat)
            if h is None:
                raise ValueError("unsupported category")
            xy = cam.to_plane(*uv, z=cam.table_z + h)
            if xy is None:
                raise ValueError("ray does not intersect surface")
            xy = xy[:2]
            obj["xy_raw"] = xy.tolist()
            if cat == "cap":
                orientation = obj.get("orientation", "unknown")
                if orientation != "narrow_end_up":
                    reason = (
                        "wide_end_up"
                        if orientation == "wide_end_up"
                        else "orientation_unknown"
                    )
                elif (
                    confidence(obj.get("orientation_confidence"))
                    < C.CAP_ORIENTATION_MIN_CONFIDENCE
                ):
                    reason = "orientation_low_confidence"
                if img is not None:
                    xy, score = refine_cap(img, cam, obj, xy)
                    obj["silhouette_iou"] = score
                    if reason is None and score < 0.45:
                        reason = "localisation_uncertain"
                obj["pick_z"] = C.SURFACE_DZ + C.CAP_PICK_HEIGHT
                obj["jaw_half"] = C.CAP_PLUG_BASE_RADIUS
                if 2 * C.CAP_PLUG_BASE_RADIUS >= 2 * C.JAW_OPEN:
                    reason = "plug_exceeds_opening"
            elif cat == "vial":
                ends = obj.get("axis_endpoints")
                if confidence(obj.get("axis_confidence")) < 0.70:
                    reason = "axis_low_confidence"
                try:
                    if len(ends) != 2:
                        raise ValueError("missing axis")
                    ep = [np.array(pixels(p), float) for p in ends]
                    if any(
                        p.shape != (2,)
                        or not np.all(np.isfinite(p))
                        or not (0 <= p[0] < cam.W and 0 <= p[1] < cam.H)
                        for p in ep
                    ):
                        raise ValueError("invalid endpoints")
                    points = [cam.to_plane(*p, z=cam.table_z + h) for p in ep]
                    if any(p is None for p in points):
                        raise ValueError("invalid axis rays")
                    axis = points[1][:2] - points[0][:2]
                    length = np.linalg.norm(axis)
                    if not 0.65 * C.VIAL_LENGTH <= length <= 1.35 * C.VIAL_LENGTH:
                        raise ValueError("axis length inconsistent with tube")
                    xy = (points[0][:2] + points[1][:2]) / 2
                    axis /= length
                    obj["axis_world"] = axis.tolist()
                    obj["spread_dir"] = [-float(axis[1]), float(axis[0]), 0.0]
                except (TypeError, ValueError, KeyError):
                    reason = "axis_missing_or_invalid"
                obj["pick_z"] = C.SURFACE_DZ + C.TIP_CLEARANCE
                obj["jaw_half"] = C.VIAL_RADIUS
            else:
                obj["colour"] = str(obj.get("colour", "unknown")).lower()
                if task == "cube_stack" and obj["colour"] not in C.CUBE_STACK_COLOURS:
                    reason = "cube_colour_unknown"
                if img is not None:
                    local_det = dict(obj, center=[uv[0] / cam.W, uv[1] / cam.H])
                    u, v, method, _ = refine_centre(
                        img, cam, local_det, make_as_pixels("unit", 1.0, cam.W, cam.H)
                    )
                    obj["localisation_method"] = method
                    if method != "refined":
                        reason = "localisation_uncertain"
                    else:
                        xy = cam.to_plane(u, v, z=cam.table_z + h)[:2]
                if task == "cube_stack":
                    try:
                        yaw = edge_pose(cam, obj.get("top_edge_endpoints"))
                        obj["cosmos_yaw_deg"] = yaw
                        if confidence(obj.get("pose_confidence")) < 0.8:
                            reason = "cube_pose_low_confidence"
                        if img is not None:
                            xy, yaw, fit = refine_pose(img, cam, obj, xy, yaw)
                            obj["pose_silhouette_iou"] = fit
                            obj["pose_refinement"] = "image_only_bounded_silhouette_fit"
                            if fit < 0.75:
                                reason = "cube_pose_fit_uncertain"
                        obj["yaw_deg"] = yaw
                        theta = np.deg2rad(yaw)
                        obj["spread_dir"] = [
                            -float(np.sin(theta)),
                            float(np.cos(theta)),
                            0.0,
                        ]
                    except (TypeError, ValueError, IndexError):
                        reason = "cube_pose_missing_or_invalid"
                obj["pick_z"] = C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE
                obj["jaw_half"] = C.CUBE_EDGE / 2
                if C.CUBE_EDGE >= 2 * C.JAW_OPEN:
                    reason = "cube_exceeds_opening"
            obj["xy"] = xy.tolist()
            if not (
                C.BACKDROP_X[0] <= xy[0] <= C.BACKDROP_X[1]
                and abs(xy[1]) <= C.BACKDROP_WIDTH / 2
            ):
                reason = "outside_work_surface"
            if any(
                o.get("category") == cat
                and "xy" in o
                and np.linalg.norm(xy - np.array(o["xy"])) < 0.012
                for o in observations
            ):
                reason = "duplicate_detection"
        except (TypeError, ValueError, KeyError, IndexError):
            reason = "invalid_detection"
        obj["eligible"] = reason is None
        obj["skip_reason"] = reason
        observations.append(obj)
        if reason:
            rejected.append(obj)

    eligible = [o for o in observations if o["eligible"]]
    cubes = [o for o in eligible if o["category"] == "cube"]
    vials = [o for o in eligible if o["category"] == "vial"]
    caps = [o for o in eligible if o["category"] == "cap"]
    picks = []
    stack_ready = False
    if task == "cube_stack":
        from collections import Counter

        stack_ready = Counter(o["colour"] for o in cubes) == Counter(
            C.CUBE_STACK_COLOURS
        )
        if stack_ready:
            remaining = sorted(cubes, key=lambda o: (o["xy"][0], o["xy"][1]))
            for level, colour in enumerate(C.CUBE_STACK_COLOURS):
                o = next(o for o in remaining if o["colour"] == colour)
                remaining.remove(o)
                picks.append(
                    dict(
                        o,
                        stack_level=level,
                        place_xy=list(C.CUBE_STACK_XY),
                        place_z=C.SURFACE_DZ
                        + level * C.CUBE_EDGE
                        + C.CUBE_TIP_CLEARANCE,
                        expected_centre_z=C.SURFACE_DZ + (level + 0.5) * C.CUBE_EDGE,
                        place_spread_dir=[0.0, 1.0, 0.0],
                        place_yaw_deg=0.0,
                    )
                )
        else:
            for o in cubes:
                o.update(eligible=False, skip_reason="incomplete_cube_colour_set")
                rejected.append(o)
        # Caps and tubes are distractors for this task, never grasp targets.
        vials, caps = [], []
    elif len(cubes) == 2:
        cubes.sort(key=lambda o: o["xy"][1])
        picks.append(
            dict(
                cubes[0],
                place_xy=cubes[1]["xy"],
                place_z=C.SURFACE_DZ + C.CUBE_EDGE + C.CUBE_TIP_CLEARANCE,
            )
        )
    elif cubes:
        for o in cubes:
            o["eligible"] = False
            o["skip_reason"] = "cube_stack_requires_two_detections"
            rejected.append(o)
    for k, o in enumerate(vials):
        if k >= len(C.SORT_VIAL_TARGETS):
            o["eligible"] = False
            o["skip_reason"] = "no_vial_destination"
            rejected.append(o)
            continue
        # Target tubes along world X. Fingers close along world Y at placement.
        picks.append(
            dict(
                o,
                place_xy=list(C.SORT_VIAL_TARGETS[k]),
                place_z=o["pick_z"],
                place_spread_dir=[0.0, 1.0, 0.0],
            )
        )
    for k, o in enumerate(caps):
        if k >= len(C.SORT_CAP_TARGETS):
            o["eligible"] = False
            o["skip_reason"] = "no_cap_destination"
            rejected.append(o)
            continue
        picks.append(dict(o, place_xy=list(C.SORT_CAP_TARGETS[k]), place_z=o["pick_z"]))
    return {
        "source": "Cosmos3-Nano",
        "gripper_model": C.GRIPPER_MODEL,
        "task": task,
        "stack_ready": stack_ready,
        "stack_colours_bottom_to_top": (
            list(C.CUBE_STACK_COLOURS) if task == "cube_stack" else None
        ),
        "open_gap_m": 2 * C.JAW_OPEN,
        "picks": picks,
        "observations": observations,
        "skipped": rejected,
    }


def score_observations(plan, gt):
    """One-to-one matching for evaluation ONLY, after the plan is finalized."""
    matches, missed, unmatched = [], [], []
    categories = (
        ("cube",) if plan.get("task") == "cube_stack" else ("cap", "cube", "vial")
    )
    for category in categories:
        predicted = [
            o
            for o in plan["observations"]
            if o.get("category") == category
            and "xy" in o
            and o.get("skip_reason") != "duplicate_detection"
        ]
        actual = [g for g in gt["objects"] if g["category"] == category]
        used_p, used_g = set(), set()
        if predicted and actual:
            distances = np.array(
                [
                    [np.linalg.norm(np.array(p["xy"]) - g["xyz"][:2]) for g in actual]
                    for p in predicted
                ]
            )
            pi, gi = linear_sum_assignment(distances)
            for pidx, gidx in zip(pi, gi):
                if distances[pidx, gidx] > 0.025:
                    continue
                p, g = predicted[pidx], actual[gidx]
                row = {
                    "detection_id": p["detection_id"],
                    "name": g["name"],
                    "category": category,
                    "position_error_mm": float(distances[pidx, gidx] * 1000),
                    "selected": any(
                        x["detection_id"] == p["detection_id"] for x in plan["picks"]
                    ),
                }
                if category == "cap":
                    row.update(
                        expected_orientation=g["orientation"],
                        predicted_orientation=p.get("orientation", "unknown"),
                        orientation_correct=p.get("orientation") == g["orientation"],
                        unsafe_selection=row["selected"]
                        and g["orientation"] != "narrow_end_up",
                    )
                if category == "cube":
                    row.update(
                        expected_colour=g["colour"],
                        predicted_colour=p.get("colour", "unknown"),
                        colour_correct=p.get("colour") == g["colour"],
                    )
                    if "yaw_deg" in p and "yaw_deg" in g:
                        row.update(
                            expected_yaw_deg=g["yaw_deg"],
                            predicted_yaw_deg=p["yaw_deg"],
                            yaw_error_deg_mod90=abs(
                                square_yaw(p["yaw_deg"] - g["yaw_deg"])
                            ),
                            cosmos_yaw_error_deg_mod90=abs(
                                square_yaw(p["cosmos_yaw_deg"] - g["yaw_deg"])
                            ),
                        )
                if category == "vial" and p.get("axis_world"):
                    theta = np.deg2rad(g["yaw_deg"])
                    row["axis_error_deg"] = float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    abs(
                                        np.dot(
                                            p["axis_world"],
                                            [np.cos(theta), np.sin(theta)],
                                        )
                                    ),
                                    0,
                                    1,
                                )
                            )
                        )
                    )
                matches.append(row)
                used_p.add(pidx)
                used_g.add(gidx)
        missed.extend(g["name"] for j, g in enumerate(actual) if j not in used_g)
        unmatched.extend(
            p["detection_id"] for j, p in enumerate(predicted) if j not in used_p
        )
    cap_matches = [m for m in matches if m["category"] == "cap"]
    return {
        "matches": matches,
        "missed_objects": missed,
        "unmatched_detections": unmatched,
        "caps_matched": len(cap_matches),
        "cap_orientations_correct": sum(m["orientation_correct"] for m in cap_matches),
        "wide_caps_selected": sum(m["unsafe_selection"] for m in cap_matches),
        "cap_orientation_confusion": [
            [
                a,
                b,
                sum(
                    m["expected_orientation"] == a and m["predicted_orientation"] == b
                    for m in cap_matches
                ),
            ]
            for a in ("narrow_end_up", "wide_end_up")
            for b in ("narrow_end_up", "wide_end_up", "unknown")
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", choices=["sort", "cube_stack"], default=C.SORT_TASK)
    args = ap.parse_args()
    run = Path(args.run)
    cam = CameraModel(str(run / "camera.json"))
    img = cv2.imread(str(run / "world_camera.png"))
    if img is None:
        raise FileNotFoundError(run / "world_camera.png")
    plan = build_plan(
        json.loads((run / "detections.json").read_text()), cam, img, task=args.task
    )
    (run / "sort_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    # Ground truth is loaded only after the executable plan has been saved.
    if (run / "ground_truth.json").exists():
        evaluation = score_observations(
            plan, json.loads((run / "ground_truth.json").read_text())
        )
        (run / "perception_evaluation.json").write_text(
            json.dumps(evaluation, indent=2) + "\n"
        )
    overlay = img.copy()
    for o in plan["observations"]:
        if "xy" not in o:
            continue
        height = {
            "cube": C.CUBE_EDGE / 2,
            "cap": C.CAP_HEIGHT / 2,
            "vial": C.VIAL_RADIUS,
        }.get(o["category"], 0)
        uv = cam.project([*o["xy"], cam.table_z + height])
        point = tuple(np.round(uv).astype(int))
        colour = (0, 190, 0) if o["eligible"] else (0, 0, 230)
        cv2.circle(overlay, point, 10, colour, 2)
        label = (
            f'{o["detection_id"]}: '
            + (
                o.get("colour", "unknown") + " cube"
                if o["category"] == "cube"
                else o.get("orientation", o["category"])
            )
            + ("" if o["eligible"] else " SKIP")
        )
        cv2.putText(
            overlay,
            label,
            (point[0] + 10, point[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
        )
        if "yaw_deg" in o:
            label = f"{o.get('colour')} {o['yaw_deg']:.1f} deg"
            cv2.putText(
                overlay,
                label,
                (point[0] + 10, point[1] + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
            )
            spread = np.array(o["spread_dir"][:2]) * 0.016
            pts = [
                cam.project(
                    [*(np.array(o["xy"]) + s * spread), cam.table_z + C.CUBE_EDGE / 2]
                )
                for s in (-1, 1)
            ]
            cv2.line(
                overlay,
                tuple(pts[0].astype(int)),
                tuple(pts[1].astype(int)),
                (255, 100, 0),
                2,
            )
        if o.get("axis_world"):
            axis = np.array(o["axis_world"]) * 0.045
            pts = [
                cam.project(
                    [*(np.array(o["xy"]) + s * axis), cam.table_z + C.VIAL_RADIUS]
                )
                for s in (-1, 1)
            ]
            cv2.line(
                overlay,
                tuple(pts[0].astype(int)),
                tuple(pts[1].astype(int)),
                (255, 100, 0),
                2,
            )
    cv2.imwrite(str(run / "detections_overlay.png"), overlay)
    print(f'[sort plan] {len(plan["picks"])} picks; {len(plan["skipped"])} skipped')
    for o in plan["skipped"]:
        print(f'  skip detection {o["detection_id"]}: {o["skip_reason"]}')


if __name__ == "__main__":
    main()
