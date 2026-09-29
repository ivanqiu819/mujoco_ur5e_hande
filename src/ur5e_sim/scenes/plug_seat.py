"""Physical plug release (no mocap twin in high-fidelity insertion)."""

from __future__ import annotations

import mujoco

GRASP_JOINT_NAMES = ()


def sync_plug_render_visibility(model: mujoco.MjModel, *, plug_seated: bool) -> None:
    """Legacy hook for camera subprocess; held_plug remains the sole physical plug."""
    del model, plug_seated


def hide_deposited_plug(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    del model, data


def show_deposited_plug(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    del model, data
