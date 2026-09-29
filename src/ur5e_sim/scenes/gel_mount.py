"""Xense G1-WS gel box placement on Hand-E inner pad sites (finger local frame)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from ur5e_sim.scenes.materials import load_materials

# G1-WS half-sizes (m) in finger body frame: thin along +Z (grasp normal into gap), pad in X/Y.
GEL_HALF_NORMAL = 0.003
GEL_HALF_WIDTH = 0.0097
GEL_HALF_HEIGHT = 0.0154

GEL_NAMES = ("xense_gel_left", "xense_gel_right")
MOUNT_NAMES = ("xense_mount_left", "xense_mount_right")

# Collision groups (bit masks): plug (2), gel (4), socket wall (8); finger mesh stays on bit0.
HELD_PLUG_CONTYPE = "2"
HELD_PLUG_CONAFFINITY = "13"  # finger (1) + gel (4) + socket wall (8)
GEL_CONTYPE = "4"
GEL_CONAFFINITY = "7"
SOCKET_WALL_CONTYPE = "8"
SOCKET_WALL_CONAFFINITY = "2"  # held_plug only


def _numbers(values) -> str:
    return " ".join(f"{float(x):.10g}" for x in np.asarray(values).ravel())


def gel_local_pose(side: str, site_pos: np.ndarray) -> tuple[np.ndarray, str]:
    """Gel center and box half-sizes on the calibrated inner pad site (finger frame)."""
    del side
    pos = site_pos.copy()
    size = _numbers((GEL_HALF_WIDTH, GEL_HALF_HEIGHT, GEL_HALF_NORMAL))
    return pos, size


def add_gel_geoms(root: ET.Element, *, replace: bool = True) -> None:
    """Attach left/right Xense gel boxes aligned with gripper_*_inner sites.

    Gels live on child bodies so ``held_plug`` vs ``*_gripper`` contact excludes
    (for finger mesh) do not disable gel–plug collision needed for tactile depth.
    """
    for side, gel_name, mount_name in (
        ("left", GEL_NAMES[0], MOUNT_NAMES[0]),
        ("right", GEL_NAMES[1], MOUNT_NAMES[1]),
    ):
        body = root.find(f'.//body[@name="{side}_gripper"]')
        if body is None:
            raise ValueError(f"{side}_gripper missing")
        site = body.find(f'./site[@name="gripper_{side}_inner"]')
        if site is None:
            raise ValueError(f"gripper_{side}_inner site missing; run gripper scene builder")
        if replace:
            for old in body.findall(f'./geom[@name="{gel_name}"]'):
                body.remove(old)
            for old in body.findall(f'./body[@name="{mount_name}"]'):
                body.remove(old)

        site_pos = np.asarray([float(x) for x in site.get("pos", "0 0 0").split()], dtype=float)
        gel_pos, size = gel_local_pose(side, site_pos)

        gel_mat = load_materials()["gel"]
        mount = ET.SubElement(body, "body", name=mount_name, pos="0 0 0")
        gel = ET.SubElement(
            mount,
            "geom",
            name=gel_name,
            type="box",
            pos=_numbers(gel_pos),
            size=size,
            rgba="0.2 0.75 0.35 0.85",
            contype=GEL_CONTYPE,
            conaffinity=GEL_CONAFFINITY,
            group="1",
        )
        for key in ("friction", "condim", "solref", "solimp", "priority"):
            gel.set(key, str(gel_mat[key]))


def scene_has_gels(root: ET.Element) -> bool:
    return all(root.find(f'.//geom[@name="{name}"]') is not None for name in GEL_NAMES)
