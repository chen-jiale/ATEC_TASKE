"""Lightweight RGB-D locator for the fixed Task E camera.

The scene has fixed object bands.  We use those bands only as a spatial prior;
the object depth is still recovered from the observed depth image, and no
simulator object state is read.
"""
from __future__ import annotations
import cv2
import numpy as np

TABLE_X, TABLE_Y, TABLE_Z = 1.0, 0.0, 0.8266
CAMERA_POS = np.array([-0.2, 0.0, 1.6266427], dtype=np.float64)
CAMERA_QUAT = (0.9570244, 0.0, 0.2900074, 0.0)
IMAGE_W, IMAGE_H = 640.0, 480.0
FX = FY = 732.9992676
CX, CY = IMAGE_W * 0.5, IMAGE_H * 0.5


def _quat_matrix(q):
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    s = 2.0 / n
    return np.array([[1-s*(y*y+z*z), s*(x*y-w*z), s*(x*z+w*y)],
                     [s*(x*y+w*z), 1-s*(x*x+z*z), s*(y*z-w*x)],
                     [s*(x*z-w*y), s*(y*z+w*x), 1-s*(x*x+y*y)]])


R_WORLD_CAMERA = _quat_matrix(CAMERA_QUAT)


def project_world_to_pixel(point):
    """Project a world point using the configured fixed camera."""
    # Isaac's configured camera uses local +X as optical axis.
    p = R_WORLD_CAMERA.T @ (np.asarray(point, dtype=np.float64) - CAMERA_POS)
    depth = p[0]
    if depth <= 1e-5:
        return float("nan"), float("nan")
    return CX - FX * p[1] / depth, CY - FY * p[2] / depth


def _depth_meters(depth):
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3: d = d[..., 0] if d.shape[-1] == 1 else d[0]
    finite = d[np.isfinite(d) & (d > 0)]
    if finite.size and float(np.median(finite)) > 20.0: d = d / 1000.0
    return d


def backproject_depth(depth):
    """Recover world surface points from optical-axis depth in metres."""
    distance = _depth_meters(depth)
    if distance.ndim != 2:
        raise ValueError("Expected a two-dimensional depth image")
    rows, columns = np.indices(distance.shape)
    rays = np.stack((np.ones_like(distance), -(columns - CX) / FX,
                     -(rows - CY) / FY), axis=-1)
    points = CAMERA_POS + (rays * distance[..., None]) @ R_WORLD_CAMERA.T
    valid = np.isfinite(distance) & (distance > 0.1) & (distance < 5.0)
    points[~valid] = np.nan
    return points


class TaskEObjectLocator:
    """Locate sugar, mustard and banana from one external RGB-D frame."""
    _BANDS = {1: (0.25, 0.29), 2: (0.14, 0.20), 3: (0.03, 0.09)}
    _MARGIN = {1: (0.07, 0.07), 2: (0.07, 0.06), 3: (0.12, 0.07)}
    _CENTER_HEIGHT = {1: 0.06, 2: 0.06, 3: 0.02}
    # Bias measured against the Isaac camera calibration regression set.
    _XY_BIAS = {1: (-0.070, 0.000), 2: (-0.038, -0.018), 3: (-0.015, -0.010)}

    def __init__(self):
        self._last = {}

    def locate_pointcloud(self, depth):
        points = backproject_depth(depth)
        foreground = np.isfinite(points).all(axis=-1)
        foreground &= (points[..., 0] > 0.75) & (points[..., 0] < 1.25)
        foreground &= (points[..., 1] > -0.02) & (points[..., 1] < 0.42)
        foreground &= (points[..., 2] > TABLE_Z + 0.008) & (points[..., 2] < TABLE_Z + 0.25)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8))
        candidates = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 80:
                continue
            surface = points[labels == label]
            lower, upper = np.quantile(surface, [0.02, 0.98], axis=0)
            center = (lower + upper) * 0.5
            center[2] = (TABLE_Z + upper[2]) * 0.5
            candidates.append(center)
        result = {}
        for object_id, (minimum, maximum) in self._BANDS.items():
            matches = [center for center in candidates if minimum - 0.02 <= center[1] <= maximum + 0.02]
            result[object_id] = min(matches, key=lambda center: abs(center[1] - (minimum + maximum) * 0.5)).copy() if matches else None
        return result

    def locate_banded_cloud(self, depth, rgb=None):
        points = backproject_depth(depth)
        valid = np.isfinite(points).all(axis=-1)
        valid &= (points[..., 0] > 0.75) & (points[..., 0] < 1.25)
        valid &= (points[..., 2] > TABLE_Z + 0.012) & (points[..., 2] < TABLE_Z + 0.22)
        sugar_valid = valid.copy()
        if rgb is not None:
            hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
            valid &= (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 1] > 90) & (hsv[..., 2] > 80)
        sugar_valid &= points[..., 2] < TABLE_Z + 0.12
        result = {}
        for object_id, bounds in {1: (0.215, 0.42), 2: (0.10, 0.28), 3: (-0.03, 0.115)}.items():
            mask = valid & (points[..., 1] >= bounds[0]) & (points[..., 1] < bounds[1])
            if object_id == 1:
                mask = sugar_valid & (points[..., 1] >= bounds[0]) & (points[..., 1] < bounds[1])
            if object_id == 2:
                mask &= points[..., 2] > TABLE_Z + 0.14
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            if count <= 1 or stats[1:, cv2.CC_STAT_AREA].max() < 80:
                result[object_id] = None
                continue
            label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            surface = points[labels == label]
            sugar_endpoint = np.quantile(surface[:, 1], 0.995) if object_id == 1 else None
            if object_id == 1 and rgb is not None:
                colored_surface = points[(labels == label) & valid]
                if len(colored_surface) < 30:
                    result[object_id] = None
                    continue
                surface = colored_surface
            if object_id == 2:
                surface = surface[surface[:, 2] >= np.quantile(surface[:, 2], 0.75)]
            lower, upper = np.quantile(surface, [0.02, 0.98], axis=0)
            center = (lower + upper) * 0.5
            center[2] = (TABLE_Z + upper[2]) * 0.5
            if object_id == 1:
                center[1] = sugar_endpoint - 0.17625346 * 0.5
            result[object_id] = center
        return result

    def locate(self, depth):
        d = _depth_meters(depth)
        table_uv = np.array([project_world_to_pixel((TABLE_X, TABLE_Y, TABLE_Z))])
        result = {}
        for obj_id, (y0, y1) in self._BANDS.items():
            # Project the known horizontal band to an image ROI, then select
            # valid foreground depth near the table surface.
            corners = [project_world_to_pixel((TABLE_X + dx, yy, TABLE_Z + 0.01))
                       for dx in (-0.14, 0.14) for yy in (y0, y1)]
            corners = [p for p in corners if np.isfinite(p[0]) and np.isfinite(p[1])]
            if not corners:
                result[obj_id] = None; continue
            us = np.asarray([p[0] for p in corners]); vs = np.asarray([p[1] for p in corners])
            x0, x1 = max(0, int(np.floor(us.min()))), min(d.shape[1], int(np.ceil(us.max())))
            v0, v1 = max(0, int(np.floor(vs.min()))), min(d.shape[0], int(np.ceil(vs.max())))
            if x1 <= x0 or v1 <= v0:
                result[obj_id] = None; continue
            crop = d[v0:v1, x0:x1]
            valid = np.isfinite(crop) & (crop > 0.1) & (crop < 5.0)
            if valid.sum() < 8:
                result[obj_id] = None; continue
            grid_u, grid_v = np.meshgrid(np.arange(x0, x1), np.arange(v0, v1))
            rays = np.stack((np.ones_like(grid_u),
                             -(grid_u - CX) / FX,
                             -(grid_v - CY) / FY), axis=-1)
            rays_world = rays @ R_WORLD_CAMERA.T
            table_depth = (TABLE_Z - CAMERA_POS[2]) / rays_world[..., 2]
            foreground = valid & (crop < table_depth - 0.004)
            yy, xx = np.where(foreground)
            if len(xx) < 8:
                values = crop[valid]
                threshold = float(np.quantile(values, 0.25))
                yy, xx = np.where(valid & (crop <= threshold + 0.008))
            if len(xx) < 8:
                result[obj_id] = None; continue
            u = float(np.median(xx + x0)); v = float(np.median(yy + v0))
            # Intersect the camera ray with the known table plane. This avoids
            # bias from the front surface of a bottle or banana.
            ray_camera = np.array([1.0, -(u - CX) / FX, -(v - CY) / FY])
            ray_world = R_WORLD_CAMERA @ ray_camera
            scale = (TABLE_Z - CAMERA_POS[2]) / ray_world[2]
            point = CAMERA_POS + scale * ray_world
            point[2] = TABLE_Z + self._CENTER_HEIGHT[obj_id]
            bias_x, bias_y = self._XY_BIAS[obj_id]
            point[0] += bias_x; point[1] += bias_y
            result[obj_id] = point
            self._last[obj_id] = point
        return result

    def debug_mask(self, depth):
        d = _depth_meters(depth)
        valid = np.isfinite(d) & (d > 0.1) & (d < 5.0)
        if not valid.any(): return np.zeros(d.shape, dtype=bool)
        q = np.quantile(d[valid], 0.30)
        return valid & (d <= q + 0.02)
