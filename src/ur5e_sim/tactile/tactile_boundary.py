"""High-fidelity gel depth and poses from gel–plug geometry (no socket proxy)."""

from __future__ import annotations

import mujoco
import numpy as np

from ur5e_sim.tactile.contact_map import (
    DEPTH_H,
    DEPTH_W,
    GEL_NAMES,
    REST_DEPTH_MM,
    _distance_index,
    _find_geom_id,
    _geom_name,
    _is_external_contact,
    _local_xy_mm,
    _spread_depth,
    _stamp_contact,
    _stamp_geom_pair_distance,
    _xy_to_pixel,
    peak_press_mm,
)

DISTANCE_ONLY_SUBSTR = ("held_plug",)


def _geom_pose_mm(data: mujoco.MjData, geom_id: int) -> np.ndarray:
    mat = data.geom_xmat[geom_id].reshape(3, 3)
    pos = data.geom_xpos[geom_id] * 1000.0
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = mat
    transform[:3, 3] = pos
    return transform


def _gel_depth(model: mujoco.MjModel, data: mujoco.MjData, gel_token: str) -> np.ndarray:
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
        other_name = _geom_name(model, other_id)
        if not any(token in other_name for token in DISTANCE_ONLY_SUBSTR):
            continue
        _stamp_geom_pair_distance(model, data, gel_id, other_id, depth)

    if np.any(depth < 0):
        depth = _spread_depth(depth)
    return depth


def build_dual_depth(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    return _gel_depth(model, data, GEL_NAMES[0]), _gel_depth(model, data, GEL_NAMES[1])


def build_relative_poses(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plug and gel sensor poses in world frame (mm), for FemSensor IPC."""
    index = _distance_index(model)
    plug_id = index.plug_id
    if plug_id < 0:
        identity = np.eye(4, dtype=np.float32)
        return identity, identity, identity, identity
    obj = _geom_pose_mm(data, plug_id).astype(np.float32)
    gel_l = _find_geom_id(model, GEL_NAMES[0])
    gel_r = _find_geom_id(model, GEL_NAMES[1])
    sensor_l = _geom_pose_mm(data, gel_l).astype(np.float32) if gel_l >= 0 else np.eye(4, dtype=np.float32)
    sensor_r = _geom_pose_mm(data, gel_r).astype(np.float32) if gel_r >= 0 else np.eye(4, dtype=np.float32)
    return obj, sensor_l, obj, sensor_r


def compute_dual_boundary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth_l, depth_r = build_dual_depth(model, data)
    obj_l, sensor_l, obj_r, sensor_r = build_relative_poses(model, data)
    return (
        depth_l,
        depth_r,
        peak_press_mm(depth_l),
        peak_press_mm(depth_r),
        obj_l,
        sensor_l,
        obj_r,
        sensor_r,
    )
