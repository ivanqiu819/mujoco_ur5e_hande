#!/usr/bin/env python3
"""Add a configurable table and the supplied inspection STL to the camera scene."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np


from ur5e_sim.paths import ROOT as ROOT
MESH_DIR = ROOT / "assets" / "meshes"


def vector(value, length, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


def numbers(value):
    return " ".join(f"{float(item):.12g}" for item in value)


def quaternion_matrix(quaternion):
    w, x, y, z = vector(quaternion, 4, "quaternion")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("quaternion has zero length")
    w, x, y, z = (item / norm for item in (w, x, y, z))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def binary_stl_bounds(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.read(80)
        count_bytes = stream.read(4)
    if len(count_bytes) != 4:
        raise ValueError(f"Not a binary STL: {path}")
    triangle_count = struct.unpack("<I", count_bytes)[0]
    if 84 + 50 * triangle_count != size:
        raise ValueError(
            f"STL size does not match its triangle count: {path}")
    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ])
    triangles = np.fromfile(path, dtype=dtype, offset=84,
                            count=triangle_count)
    vertices = triangles["vertices"].reshape(-1, 3).astype(np.float64)
    return triangle_count, vertices.min(axis=0), vertices.max(axis=0)


def corners(lower, upper):
    return np.array([
        [x, y, z]
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ], dtype=np.float64)


def box_inertia(mass, full_size):
    x, y, z = full_size
    return mass / 12.0 * np.array([
        y * y + z * z,
        x * x + z * z,
        x * x + y * y,
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/inspection.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Only inspection config schema_version 1 is supported")

    source = (config_path.parent / config["source_scene"]).resolve()
    output = (config_path.parent / config["output_scene"]).resolve()
    if source == output:
        raise ValueError("Source and output scenes must differ")
    if source.parent != output.parent:
        raise ValueError("Output must stay beside the source scene")

    table = config["table"]
    table_xy = vector(table["center_xy_m"], 2, "table center")
    top_size = vector(table["top_size_m"], 3, "table top size")
    leg_size = vector(table["leg_size_m"], 3, "table leg size")
    top_height = float(table["top_surface_height_m"])
    leg_inset = float(table["leg_inset_m"])
    if np.any(top_size <= 0) or np.any(leg_size <= 0) or top_height <= 0:
        raise ValueError("Table dimensions and height must be positive")
    if not math.isclose(leg_size[2], top_height - top_size[2],
                        abs_tol=1e-9):
        raise ValueError(
            "leg_size_m[2] must equal top_surface_height_m - top thickness")
    if np.any(2 * leg_inset + leg_size[:2] >= top_size[:2]):
        raise ValueError("Table leg inset leaves no valid leg placement")

    workpiece = config["workpiece"]
    mesh_path = (MESH_DIR / workpiece["mesh_file"]).resolve()
    if mesh_path.parent != MESH_DIR.resolve() or not mesh_path.is_file():
        raise FileNotFoundError(f"Workpiece mesh not found: {mesh_path}")
    scale = vector(workpiece["scale"], 3, "workpiece scale")
    if np.any(scale <= 0):
        raise ValueError("Workpiece scale must be positive")
    triangle_count, raw_lower, raw_upper = binary_stl_bounds(mesh_path)
    lower = raw_lower * scale
    upper = raw_upper * scale
    extent = upper - lower
    center_local = (lower + upper) / 2.0
    quaternion = vector(workpiece["quaternion_world_wxyz"], 4,
                        "workpiece quaternion")
    quaternion /= np.linalg.norm(quaternion)
    rotation = quaternion_matrix(quaternion)
    rotated_corners = corners(lower, upper) @ rotation.T
    rotated_lower = rotated_corners.min(axis=0)
    rotated_upper = rotated_corners.max(axis=0)
    desired_xy = vector(workpiece["center_xy_m"], 2, "workpiece center")
    clearance = float(workpiece["table_clearance_m"])
    if clearance < 0:
        raise ValueError("table_clearance_m cannot be negative")
    translation = np.array([
        desired_xy[0] - (rotated_lower[0] + rotated_upper[0]) / 2.0,
        desired_xy[1] - (rotated_lower[1] + rotated_upper[1]) / 2.0,
        top_height + clearance - rotated_lower[2],
    ])

    tree = ET.parse(source)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Source scene must contain asset and worldbody")
    if root.find(".//body[@name='inspection_table']") is not None:
        raise ValueError("Source scene already contains an inspection table")
    if root.find(".//body[@name='inspection_workpiece']") is not None:
        raise ValueError("Source scene already contains an inspection workpiece")

    if config.get("remove_example_obstacle", False):
        obstacle = worldbody.find("body[@name='example_obstacle']")
        if obstacle is not None:
            worldbody.remove(obstacle)

    ET.SubElement(asset, "material", name="inspection_table_top_material",
                  rgba=numbers(vector(table["top_rgba"], 4, "table top rgba")))
    ET.SubElement(asset, "material", name="inspection_table_leg_material",
                  rgba=numbers(vector(table["leg_rgba"], 4, "table leg rgba")))
    ET.SubElement(asset, "material", name="inspection_workpiece_material",
                  rgba=numbers(vector(workpiece["visual_rgba"], 4,
                                      "workpiece rgba")))
    ET.SubElement(asset, "mesh", name="inspection_workpiece_mesh",
                  file=mesh_path.name, scale=numbers(scale))

    table_body = ET.SubElement(
        worldbody, "body", name=str(table["name"]),
        pos=numbers([table_xy[0], table_xy[1], top_height]))
    ET.SubElement(
        table_body, "geom", name="inspection_table_top", type="box",
        pos=numbers([0, 0, -top_size[2] / 2.0]),
        size=numbers(top_size / 2.0), material="inspection_table_top_material",
        contype="1", conaffinity="1")
    leg_x = top_size[0] / 2.0 - leg_inset - leg_size[0] / 2.0
    leg_y = top_size[1] / 2.0 - leg_inset - leg_size[1] / 2.0
    leg_z = -top_size[2] - leg_size[2] / 2.0
    for index, (x, y) in enumerate(
            ((leg_x, leg_y), (leg_x, -leg_y),
             (-leg_x, leg_y), (-leg_x, -leg_y))):
        ET.SubElement(
            table_body, "geom", name=f"inspection_table_leg_{index}",
            type="box", pos=numbers([x, y, leg_z]),
            size=numbers(leg_size / 2.0),
            material="inspection_table_leg_material",
            contype="1", conaffinity="1")

    workpiece_body = ET.SubElement(
        worldbody, "body", name=str(workpiece["name"]),
        pos=numbers(translation), quat=numbers(quaternion))
    if bool(workpiece["dynamic"]):
        mass = float(workpiece["mass_kg"])
        if mass <= 0:
            raise ValueError("Dynamic workpiece mass must be positive")
        ET.SubElement(workpiece_body, "freejoint", name="workpiece_freejoint")
        ET.SubElement(workpiece_body, "inertial", pos=numbers(center_local),
                      mass=f"{mass:.12g}",
                      diaginertia=numbers(box_inertia(mass, extent)))
    ET.SubElement(
        workpiece_body, "geom", name="inspection_workpiece_visual",
        type="mesh", mesh="inspection_workpiece_mesh", **{"class": "visual"},
        material="inspection_workpiece_material")
    if workpiece.get("collision_proxy") != "bounding_box":
        raise ValueError("Only bounding_box collision_proxy is currently supported")
    ET.SubElement(
        workpiece_body, "geom", name="inspection_workpiece_collision",
        type="box", **{"class": "collision"}, pos=numbers(center_local),
        size=numbers(extent / 2.0))
    ET.SubElement(
        workpiece_body, "site", name="workpiece_frame", type="sphere",
        pos=numbers(center_local), size="0.004", group="4",
        rgba="0.15 0.95 0.25 0.75")

    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    world_lower = rotated_lower + translation
    world_upper = rotated_upper + translation
    ET.SubElement(custom, "numeric", name="inspection_workpiece_raw_extent_m",
                  data=numbers(extent))
    ET.SubElement(custom, "numeric", name="inspection_workpiece_world_aabb_m",
                  data=numbers(np.r_[world_lower, world_upper]))
    ET.SubElement(custom, "text", name="inspection_scene_config",
                  data="table_plus_supplied_stl_schema_v1")

    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print("SOURCE_UNCHANGED:", source)
    print("CREATED_SCENE:", output)
    print("STL_TRIANGLE_COUNT:", triangle_count)
    print("STL_RAW_EXTENT:", (raw_upper - raw_lower).tolist())
    print("WORKPIECE_EXTENT_M:", extent.tolist())
    print("WORKPIECE_WORLD_AABB_M:",
          [world_lower.tolist(), world_upper.tolist()])
    print("TABLE_TOP_SURFACE_Z_M:", top_height)
    print("WORKPIECE_DYNAMIC:", bool(workpiece["dynamic"]))


if __name__ == "__main__":
    main()
