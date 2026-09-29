"""Lateral corrections from dual gel press asymmetry (L1 boundary depth)."""

from __future__ import annotations

import mujoco
import numpy as np


def press_imbalance_mm(press_l_mm: float, press_r_mm: float) -> float:
    return float(press_l_mm - press_r_mm)


def _unit3(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float).ravel()[:3]
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    return v / n


def gripper_open_axis_world(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World unit vector from left inner jaw toward right (same axis as L/R gel imbalance)."""
    left = data.site("gripper_left_inner").xpos
    right = data.site("gripper_right_inner").xpos
    return _unit3(right - left)


def lateral_correction_from_gel(
    press_l_mm: float,
    press_r_mm: float,
    lateral_axis_world: np.ndarray,
    *,
    gain_mm_per_mm: float,
    max_step_mm: float,
    reference_axis_world: np.ndarray | None = None,
    sign: float = 1.0,
) -> np.ndarray:
    """World-frame TCP translation from L/R gel imbalance.

    Left gel heavier than right → move opposite socket +X (nominal port width axis).
    Uses the live gripper open axis; ``reference_axis_world`` (nominal port +X) fixes
    180° flips from vision vs scene.
    """
    axis = _unit3(lateral_axis_world)
    if not np.any(axis):
        return np.zeros(3)
    if reference_axis_world is not None:
        ref = _unit3(reference_axis_world)
        if np.any(ref) and float(np.dot(axis, ref)) < 0.0:
            axis = -axis
    imbalance = press_imbalance_mm(press_l_mm, press_r_mm)
    step_mm = float(
        np.clip(-float(sign) * gain_mm_per_mm * imbalance, -max_step_mm, max_step_mm)
    )
    return step_mm * 0.001 * axis


def correction_in_port_plane(
    port_rigid: np.ndarray,
    press_l_mm: float,
    press_r_mm: float,
    *,
    gain_mm_per_mm: float,
    max_step_mm: float,
    sign: float = 1.0,
) -> np.ndarray:
    port = np.asarray(port_rigid, dtype=float)
    return lateral_correction_from_gel(
        press_l_mm,
        press_r_mm,
        port[:3, 0],
        gain_mm_per_mm=gain_mm_per_mm,
        max_step_mm=max_step_mm,
        reference_axis_world=port[:3, 0],
        sign=sign,
    )


def converged(
    press_l_mm: float,
    press_r_mm: float,
    *,
    delta_threshold_mm: float,
    min_peak_mm: float = 0.03,
) -> bool:
    if max(press_l_mm, press_r_mm) < min_peak_mm:
        return False
    return abs(press_imbalance_mm(press_l_mm, press_r_mm)) <= delta_threshold_mm
