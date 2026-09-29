"""Load per-component MuJoCo contact parameters for insertion scenes."""

from __future__ import annotations

from pathlib import Path

import yaml

from ur5e_sim.paths import ROOT

DEFAULT_CONFIG = ROOT / "configs/materials.yaml"


def load_materials(path: Path | None = None) -> dict:
    cfg_path = Path(path or DEFAULT_CONFIG).resolve()
    data = yaml.safe_load(cfg_path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported materials schema: {cfg_path}")
    return data


def _apply_geom_attrs(geom: object, section: dict) -> None:
    for key in ("friction", "condim", "solref", "solimp", "priority"):
        if key in section:
            geom.set(key, str(section[key]))


def apply_finger_collision_materials(root, materials: dict | None = None) -> None:
    materials = materials or load_materials()
    section = materials["finger_collision"]
    for name in ("left_finger_collision", "right_finger_collision"):
        geom = root.find(f'.//geom[@name="{name}"]')
        if geom is not None:
            _apply_geom_attrs(geom, section)


def apply_socket_wall_materials(root, materials: dict | None = None) -> None:
    materials = materials or load_materials()
    section = materials["socket_wall"]
    for geom in root.findall('.//geom'):
        if geom.get("name", "").startswith("socket_wall_"):
            _apply_geom_attrs(geom, section)
