"""Xense gel depth maps and press depth from MuJoCo contacts."""

from __future__ import annotations

import cv2
import mujoco
import numpy as np

DEPTH_W, DEPTH_H = 100, 175
GEL_W_MM, GEL_H_MM = 19.4, 30.8
REST_DEPTH_MM = 0.2
CONTACT_STAMP_RADIUS_MM = 4.0
MAX_PRESS_MM = 6.0

GEL_NAMES = ("xense_gel_left", "xense_gel_right")

SKIP_CONTACT_SUBSTR = (
    "finger",
    "gripper",
    "hand",
    "hande",
    "xense_gel",
    "screw",
    "slider",
)

# Geoms queried for gel distance (not the full ~500-geom robot mesh each step).
DISTANCE_TARGET_SUBSTR = (
    "held_plug",
    "socket_wall",
)

_INDEX_CACHE: dict[int, "_GelDistanceIndex"] = {}


class _GelDistanceIndex:
    __slots__ = ("gel_ids", "target_ids", "plug_id", "wall_ids")

    def __init__(self, model: mujoco.MjModel) -> None:
        self.gel_ids = tuple(_find_geom_id(model, name) for name in GEL_NAMES)
        targets: list[int] = []
        walls: list[int] = []
        plug_id = -1
        for i in range(model.ngeom):
            name = _geom_name(model, i)
            if any(k in name for k in SKIP_CONTACT_SUBSTR):
                continue
            if "held_plug" in name:
                plug_id = i
            if any(token in name for token in DISTANCE_TARGET_SUBSTR):
                targets.append(i)
                if "socket_wall" in name:
                    walls.append(i)
        self.target_ids = tuple(targets)
        self.plug_id = plug_id
        self.wall_ids = tuple(walls)


def _distance_index(model: mujoco.MjModel) -> _GelDistanceIndex:
    key = id(model)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = _GelDistanceIndex(model)
    return _INDEX_CACHE[key]


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def _find_geom_id(model: mujoco.MjModel, token: str) -> int:
    for i in range(model.ngeom):
        if token in _geom_name(model, i):
            return i
    return -1


def _is_external_contact(model: mujoco.MjModel, gel_id: int, other_id: int) -> bool:
    other = _geom_name(model, other_id)
    if gel_id == other_id:
        return False
    if any(k in other for k in SKIP_CONTACT_SUBSTR):
        return False
    return True


def _local_xy_mm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_geom_id: int,
    world_pos: np.ndarray,
) -> tuple[float, float]:
    del model
    geom_xpos = data.geom_xpos[frame_geom_id]
    geom_xmat = data.geom_xmat[frame_geom_id].reshape(3, 3)
    local = geom_xmat.T @ (world_pos - geom_xpos)
    # Gel box: local Z = pad normal; 19.4 mm / 30.8 mm on local X / Y.
    return float(local[0] * 1000.0), float(local[1] * 1000.0)


def _xy_to_pixel(px_mm: float, py_mm: float) -> tuple[int, int]:
    ix = int((px_mm + GEL_W_MM / 2.0) / GEL_W_MM * DEPTH_W)
    iy = int((GEL_H_MM / 2.0 - py_mm) / GEL_H_MM * DEPTH_H)
    return ix, iy


def _stamp_contact(depth: np.ndarray, ix: int, iy: int, pen_mm: float) -> None:
    radius_px_x = max(2, int(CONTACT_STAMP_RADIUS_MM / GEL_W_MM * DEPTH_W))
    radius_px_y = max(2, int(CONTACT_STAMP_RADIUS_MM / GEL_H_MM * DEPTH_H))
    val = -float(pen_mm)
    y0 = max(0, iy - radius_px_y)
    y1 = min(DEPTH_H, iy + radius_px_y + 1)
    x0 = max(0, ix - radius_px_x)
    x1 = min(DEPTH_W, ix + radius_px_x + 1)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            dx = (xx - ix) * (GEL_W_MM / DEPTH_W)
            dy = (yy - iy) * (GEL_H_MM / DEPTH_H)
            if dx * dx + dy * dy > CONTACT_STAMP_RADIUS_MM * CONTACT_STAMP_RADIUS_MM:
                continue
            depth[yy, xx] = min(depth[yy, xx], val)


def _stamp_socket_via_plug(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gel_id: int,
    depth: np.ndarray,
    index: _GelDistanceIndex,
) -> None:
    """Insertion: gels stay outside the slot; show slot wall press via held_plug vs socket_wall."""
    if index.plug_id < 0 or not index.wall_ids:
        return
    fromto = np.zeros(6, dtype=np.float64)
    half_w = GEL_W_MM / 2.0
    half_h = GEL_H_MM / 2.0
    for wall_id in index.wall_ids:
        dist = mujoco.mj_geomDistance(model, data, index.plug_id, wall_id, 100.0, fromto)
        if dist >= 0.0:
            continue
        pos = 0.5 * (fromto[:3] + fromto[3:6])
        px_mm, py_mm = _local_xy_mm(model, data, gel_id, pos)
        px_mm = float(np.clip(px_mm, -half_w, half_w))
        py_mm = float(np.clip(py_mm, -half_h, half_h))
        ix, iy = _xy_to_pixel(px_mm, py_mm)
        pen_mm = float(-dist * 1000.0)
        ix = int(np.clip(ix, 0, DEPTH_W - 1))
        iy = int(np.clip(iy, 0, DEPTH_H - 1))
        _stamp_contact(depth, ix, iy, pen_mm)


def _stamp_geom_pair_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gel_id: int,
    other_id: int,
    depth: np.ndarray,
) -> None:
    """Use analytic geom distance when the contact array is empty (common in this sim stack)."""
    fromto = np.zeros(6, dtype=np.float64)
    dist = mujoco.mj_geomDistance(model, data, gel_id, other_id, 100.0, fromto)
    if dist >= 0.0:
        return
    pos = 0.5 * (fromto[:3] + fromto[3:6])
    px_mm, py_mm = _local_xy_mm(model, data, gel_id, pos)
    ix, iy = _xy_to_pixel(px_mm, py_mm)
    pen_mm = float(-dist * 1000.0)
    ix = int(np.clip(ix, 0, DEPTH_W - 1))
    iy = int(np.clip(iy, 0, DEPTH_H - 1))
    _stamp_contact(depth, ix, iy, pen_mm)


def _spread_depth(depth: np.ndarray) -> np.ndarray:
    penetration = np.clip(-depth, 0.0, None)
    if not np.any(penetration > 0):
        return depth
    blurred = cv2.GaussianBlur(penetration, (0, 0), sigmaX=2.5, sigmaY=2.5)
    merged = np.maximum(penetration, blurred * 0.85)
    out = depth.copy()
    out[merged > 0] = -merged[merged > 0]
    return out


def compute_gel_depth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gel_token: str,
    *,
    socket_proxy: bool = False,
) -> np.ndarray:
    depth = np.full((DEPTH_H, DEPTH_W), REST_DEPTH_MM, dtype=np.float32)
    gel_id = _find_geom_id(model, gel_token)
    if gel_id < 0:
        return depth

    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist >= 0:
            continue
        if contact.geom1 != gel_id and contact.geom2 != gel_id:
            continue
        other_id = contact.geom2 if contact.geom1 == gel_id else contact.geom1
        if not _is_external_contact(model, gel_id, other_id):
            continue

        px_mm, py_mm = _local_xy_mm(model, data, gel_id, contact.pos)
        ix, iy = _xy_to_pixel(px_mm, py_mm)
        pen_mm = float(-contact.dist * 1000.0)
        ix = int(np.clip(ix, 0, DEPTH_W - 1))
        iy = int(np.clip(iy, 0, DEPTH_H - 1))
        _stamp_contact(depth, ix, iy, pen_mm)

    index = _distance_index(model)
    for other_id in index.target_ids:
        if other_id == gel_id:
            continue
        _stamp_geom_pair_distance(model, data, gel_id, other_id, depth)
    if socket_proxy:
        _stamp_socket_via_plug(model, data, gel_id, depth, index)

    if np.any(depth < 0):
        depth = _spread_depth(depth)
    return depth


def peak_press_mm(depth: np.ndarray) -> float:
    return float(np.clip(-np.min(depth), 0.0, MAX_PRESS_MM))


def max_plug_wall_penetration_mm(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    max_pen = 0.0
    for i in range(int(data.ncon)):
        contact = data.contact[i]
        if contact.dist >= 0.0:
            continue
        n1 = _geom_name(model, contact.geom1)
        n2 = _geom_name(model, contact.geom2)
        pair = f"{n1} {n2}"
        if "held_plug" in pair and "socket_wall" in pair:
            max_pen = max(max_pen, float(-contact.dist * 1000.0))
    index = _distance_index(model)
    if index.plug_id >= 0 and index.wall_ids:
        fromto = np.zeros(6, dtype=np.float64)
        for wall_id in index.wall_ids:
            dist = mujoco.mj_geomDistance(model, data, index.plug_id, wall_id, 100.0, fromto)
            if dist < 0.0:
                max_pen = max(max_pen, float(-dist * 1000.0))
    return max_pen


def gel_press_depth_mm(model: mujoco.MjModel, data: mujoco.MjData, gel_name: str) -> float:
    return peak_press_mm(compute_gel_depth(model, data, gel_name))


def dual_press_mm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gel_names: tuple[str, str] = GEL_NAMES,
) -> tuple[float, float]:
    return gel_press_depth_mm(model, data, gel_names[0]), gel_press_depth_mm(model, data, gel_names[1])


def compute_dual_gel_depths(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gel_names: tuple[str, str] = GEL_NAMES,
    *,
    socket_proxy: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    depth_l = compute_gel_depth(model, data, gel_names[0], socket_proxy=socket_proxy)
    depth_r = compute_gel_depth(model, data, gel_names[1], socket_proxy=socket_proxy)
    return depth_l, depth_r, peak_press_mm(depth_l), peak_press_mm(depth_r)


def depth_to_colormap(depth: np.ndarray, scale_mm: float | None = None) -> np.ndarray:
    penetration = -np.minimum(depth, 0.0)
    max_pen = float(np.max(penetration)) if np.any(penetration > 0) else 0.0
    scale = scale_mm if scale_mm is not None else max(0.35, max_pen)
    vis = np.clip(penetration / scale, 0.0, 1.0)
    colored = cv2.applyColorMap((vis * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.resize(colored, (200, 350), interpolation=cv2.INTER_NEAREST)


def dual_tactile_canvas(
    depth_l: np.ndarray,
    depth_r: np.ndarray,
    press_l: float,
    press_r: float,
    press_des: float,
) -> np.ndarray:
    scale = max(0.35, press_des, press_l, press_r, 0.05)
    left = depth_to_colormap(depth_l, scale_mm=scale)
    right = depth_to_colormap(depth_r, scale_mm=scale)
    canvas = np.hstack([left, right])
    color = (0, 255, 0) if max(press_l, press_r) > 0.05 else (180, 180, 180)
    cv2.putText(canvas, f"L {press_l:.2f} mm", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(canvas, f"R {press_r:.2f} mm", (208, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return canvas
