"""Tabletop square pose from Cosmos top-edge pixels, refined using image only."""

import cv2
import numpy as np
from scipy.optimize import differential_evolution
import scene_config as C


def square_yaw(degrees):
    return (float(degrees) + 45) % 90 - 45


def edge_pose(cam, endpoints):
    uv = np.asarray(endpoints, float)
    if (
        uv.shape != (2, 2)
        or not np.all(np.isfinite(uv))
        or np.any(uv < 0)
        or np.any(uv > 1)
    ):
        raise ValueError("invalid top edge")
    points = np.array(
        [cam.to_plane(*(p * [cam.W, cam.H]), z=cam.table_z + C.CUBE_EDGE) for p in uv]
    )
    edge = points[1, :2] - points[0, :2]
    if not 0.60 * C.CUBE_EDGE <= np.linalg.norm(edge) <= 1.40 * C.CUBE_EDGE:
        raise ValueError("top edge length inconsistent with cube")
    return square_yaw(np.degrees(np.arctan2(edge[1], edge[0])))


def silhouette(cam, xy, yaw, shape, origin=(0, 0)):
    t = np.deg2rad(yaw)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    points = [
        cam.project(
            [
                *(np.asarray(xy) + rot @ np.array([x, y]) * C.CUBE_EDGE / 2),
                cam.table_z + z,
            ]
        )
        for z in (0, C.CUBE_EDGE)
        for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    # Projected coordinates are continuous; integer image pixels sample at
    # (column + 0.5, row + 0.5). Rounding vertices and filling inclusive edges
    # inflated small cube outlines and biased the wider calibrated camera fit.
    hull = cv2.convexHull(np.asarray(points, np.float32)).reshape(-1, 2) - origin
    mask = np.zeros(shape, np.uint8)
    lo = np.maximum(np.floor(hull.min(0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(hull.max(0)).astype(int) + 1, shape[::-1])
    xx, yy = np.meshgrid(np.arange(lo[0], hi[0]) + 0.5, np.arange(lo[1], hi[1]) + 0.5)
    if not xx.size:
        return mask
    samples = np.stack([xx, yy], -1)
    edges = np.roll(hull, -1, axis=0) - hull
    delta = samples[:, :, None, :] - hull
    cross = edges[:, 0] * delta[:, :, :, 1] - edges[:, 1] * delta[:, :, :, 0]
    mask[lo[1] : hi[1], lo[0] : hi[0]] = np.all(cross >= 0, axis=-1) | np.all(
        cross <= 0, axis=-1
    )
    return mask


def refine_pose(img, cam, obj, xy, yaw, search_half_width=20):
    """Bounded silhouette fit around VLM pose; neither config poses nor GT used."""
    H, W = img.shape[:2]
    u, v = cam.project([*xy, cam.table_z + C.CUBE_EDGE / 2])
    radius = round(40 * W / 1920)
    x0, y0 = max(0, int(u) - radius), max(0, int(v) - radius)
    crop = img[y0 : min(H, int(v) + radius + 1), x0 : min(W, int(u) + radius + 1)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    colour = obj["colour"]
    gate = (
        (h >= 12) & (h <= 42)
        if colour == "yellow"
        else (h >= 95) & (h <= 135) if colour == "blue" else (h <= 12) | (h >= 170)
    )
    mask = (gate & (s >= 45) & (hsv[:, :, 2] >= 35)).astype(np.uint8)
    n, labels, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
    candidates = [i for i in range(1, n) if stats[i, 4] >= 40]
    if not candidates:
        return xy, yaw, 0.0
    component = min(
        candidates, key=lambda i: np.linalg.norm(centres[i] - [u - x0, v - y0])
    )
    observed = labels == component
    if search_half_width >= 45:
        # Initial Cosmos centres are approximate. Seed the full yaw search at
        # the same-colour image component, as the planner already does. Keep
        # the local +/-4 mm fitting range and reject distant proposals.
        centre = centres[component] + [x0, y0]
        if np.linalg.norm(centre - [u, v]) > 20 * W / 1920:
            return xy, yaw, 0.0
        xy = cam.to_plane(*centre, z=cam.table_z + C.CUBE_EDGE / 2)[:2]

    def score(p):
        predicted = silhouette(cam, p[:2], p[2], mask.shape, (x0, y0)) > 0
        return float(
            np.logical_and(predicted, observed).sum()
            / max(1, np.logical_or(predicted, observed).sum())
        )

    bounds = [
        (xy[0] - 0.004, xy[0] + 0.004),
        (xy[1] - 0.004, xy[1] + 0.004),
        (yaw - search_half_width, yaw + search_half_width),
    ]
    result = differential_evolution(
        lambda p: -score(p),
        bounds,
        seed=8,
        popsize=15,
        maxiter=70,
        polish=False,
        tol=0.0001,
    )
    fit = score(result.x)
    # A boundary optimum means the observed pose may lie outside the allowed
    # correction range; refuse to turn a bad VLM edge into an unchecked grasp.
    if np.any(np.abs(result.x[:2] - xy) > 0.0038) or (
        search_half_width < 45 and abs(result.x[2] - yaw) > search_half_width - 2
    ):
        fit = 0.0
    return result.x[:2], square_yaw(result.x[2]), fit
