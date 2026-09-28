#!/usr/bin/env python3
"""Add the configured Blackfly S + 16 mm eye-in-hand camera to the scene."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ur5e_sim.config_io import read_config


from ur5e_sim.paths import ROOT as ROOT


def vector(value, length, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


def unit(value, name):
    value = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise ValueError(f"{name} has zero length")
    return value / norm


def matrix_to_quaternion(matrix):
    """Rotation matrix to normalized WXYZ quaternion."""
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([0.25 * s,
                      (matrix[2, 1] - matrix[1, 2]) / s,
                      (matrix[0, 2] - matrix[2, 0]) / s,
                      (matrix[1, 0] - matrix[0, 1]) / s])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            q = np.array([(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s,
                          (matrix[0, 1] + matrix[1, 0]) / s,
                          (matrix[0, 2] + matrix[2, 0]) / s])
        elif index == 1:
            s = math.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            q = np.array([(matrix[0, 2] - matrix[2, 0]) / s,
                          (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s,
                          (matrix[1, 2] + matrix[2, 1]) / s])
        else:
            s = math.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            q = np.array([(matrix[1, 0] - matrix[0, 1]) / s,
                          (matrix[0, 2] + matrix[2, 0]) / s,
                          (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def look_at_quaternion(position, target, up_hint):
    """Build MuJoCo camera frame: +X right, +Y up, viewing along -Z."""
    forward = unit(target - position, "mount look direction")
    up_hint = unit(up_hint, "mount up hint")
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("mount up hint is parallel to the look direction")
    right = unit(right, "camera right axis")
    camera_z = -forward
    camera_y = unit(np.cross(camera_z, right), "camera up axis")
    rotation = np.column_stack((right, camera_y, camera_z))
    return matrix_to_quaternion(rotation)


def numbers(value):
    return " ".join(f"{float(item):.12g}" for item in value)


def named(root, tag, name):
    return root.find(f".//{tag}[@name='{name}']")


def combined_inertia(parts):
    """Return mass, center of mass and diagonal inertia for aligned rigid parts."""
    mass = sum(part[0] for part in parts)
    center = sum(part[0] * part[1] for part in parts) / mass
    tensor = np.zeros((3, 3), dtype=float)
    for part_mass, part_center, part_diagonal in parts:
        offset = part_center - center
        tensor += np.diag(part_diagonal)
        tensor += part_mass * (np.dot(offset, offset) * np.eye(3)
                               - np.outer(offset, offset))
    if np.max(np.abs(tensor - np.diag(np.diag(tensor)))) > 1e-12:
        raise ValueError("Mechanical model produced non-diagonal inertia")
    return mass, center, np.diag(tensor)


def validate_config(config):
    if config.get("schema_version") != 2:
        raise ValueError("Only camera config schema_version 2 is supported")
    profile = config["output_profile"]
    width = int(profile["width"])
    height = int(profile["height"])
    if width <= 0 or height <= 0:
        raise ValueError("Camera resolution must be positive")
    intrinsics = profile["intrinsics_nominal"]
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    hardware = config["hardware"]
    sensor_size = vector(hardware["sensor_size_m"], 2, "sensor size")
    native_resolution = vector(hardware["native_resolution"], 2, "native resolution")
    pixel_size = vector(hardware["pixel_size_um"], 2, "pixel size")
    source_roi = vector(profile["source_roi_xywh"], 4, "source ROI")
    resize_scale = vector(profile["resize_scale_xy"], 2, "resize scale")
    if not all(math.isfinite(x) for x in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
        raise ValueError("Camera intrinsics must be finite and focal lengths positive")
    if not (0 <= cx < width and 0 <= cy < height):
        raise ValueError("Camera principal point must lie inside the image")
    if np.any(sensor_size <= 0):
        raise ValueError("Camera sensor size must be positive")
    if np.any(native_resolution <= 0) or np.any(pixel_size <= 0):
        raise ValueError("Native resolution and pixel size must be positive")
    if np.any(source_roi < 0) or np.any(source_roi[2:] <= 0):
        raise ValueError("Source ROI must have non-negative origin and positive size")
    if np.any(source_roi[:2] + source_roi[2:] > native_resolution):
        raise ValueError("Source ROI must lie inside the native image")
    expected_scale = np.array([width, height], dtype=float) / source_roi[2:]
    if not np.allclose(resize_scale, expected_scale, atol=1e-12):
        raise ValueError("resize_scale_xy does not match output size / source ROI")
    depth = config["simulation_ground_truth_depth"]
    near, far = float(depth["min_m"]), float(depth["max_m"])
    noise = float(depth["noise_std_m"])
    if not 0 < near < far or not math.isfinite(far) or noise < 0:
        raise ValueError("Expected valid simulated ground-truth depth limits")
    if float(config["preview_hz"]) <= 0:
        raise ValueError("preview_hz must be positive")
    body_size = vector(hardware["camera_body_size_m"], 3, "camera body size")
    lens_size = hardware["lens_size_m"]
    if (np.any(body_size <= 0) or float(lens_size["diameter"]) <= 0
            or float(lens_size["length"]) <= 0):
        raise ValueError("Camera and lens dimensions must be positive")
    if min(float(hardware["camera_mass_kg"]), float(hardware["lens_mass_kg"])) <= 0:
        raise ValueError("Camera and lens masses must be positive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "scenes/scene_control.xml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/camera.json")
    parser.add_argument("--output", type=Path, default=ROOT / "scenes/scene_camera.xml")
    args = parser.parse_args()
    source, config_path, output = (path.resolve() for path in
                                   (args.source, args.config, args.output))
    if output == source:
        parser.error("Output must differ from source; the calibrated scene is preserved")
    if output.parent != source.parent:
        parser.error("Keep output beside source so relative mesh paths remain valid")

    config = read_config(config_path)
    validate_config(config)
    tree = ET.parse(source)
    root = tree.getroot()
    camera_name = str(config["camera_name"])
    parent_name = str(config["parent_body"])
    parent = named(root, "body", parent_name)
    if parent is None:
        raise ValueError(f"Parent body not found: {parent_name}")
    if named(root, "camera", camera_name) is not None or named(root, "body", "camera_mount") is not None:
        raise ValueError("Source already has this camera module; rebuild from scene_control.xml")

    profile = config["output_profile"]
    width = int(profile["width"])
    height = int(profile["height"])
    visual_global = root.find("visual/global")
    if visual_global is None:
        raise ValueError("Scene has no visual/global section")
    visual_global.set("offwidth", str(width))
    visual_global.set("offheight", str(height))

    asset = root.find("asset")
    if asset is None:
        raise ValueError("Scene has no asset section")
    hardware = config["hardware"]
    mechanical = config["mechanical_model"]
    rgba = vector(mechanical["camera_rgba"], 4, "mechanical_model.camera_rgba")
    ET.SubElement(asset, "material", name="ee_camera_body", rgba=numbers(rgba))
    ET.SubElement(asset, "material", name="ee_camera_lens", rgba="0.05 0.12 0.18 1")
    ET.SubElement(asset, "material", name="ee_camera_bracket", rgba="0.18 0.19 0.20 1")

    position = vector(config["mount"]["position_parent_m"], 3, "mount position")
    look_at = vector(config["mount"]["look_at_parent_m"], 3, "mount look_at")
    up_hint = vector(config["mount"]["up_hint_parent"], 3, "mount up_hint")
    mount_quaternion = look_at_quaternion(position, look_at, up_hint)
    body_size = vector(hardware["camera_body_size_m"], 3, "camera body size")
    body_half = body_size / 2
    body_center = np.array([0.0, 0.0, float(mechanical["camera_body_center_z_m"])])
    camera_mass = float(hardware["camera_mass_kg"])
    camera_inertia = camera_mass / 12 * np.array([
        body_size[1] ** 2 + body_size[2] ** 2,
        body_size[0] ** 2 + body_size[2] ** 2,
        body_size[0] ** 2 + body_size[1] ** 2,
    ])
    lens_radius = float(hardware["lens_size_m"]["diameter"]) / 2
    lens_half_length = float(hardware["lens_size_m"]["length"]) / 2
    lens_center = np.array([0.0, 0.0, float(mechanical["lens_center_z_m"])])
    lens_mass = float(hardware["lens_mass_kg"])
    lens_inertia = np.array([
        lens_mass * (3 * lens_radius ** 2 + (2 * lens_half_length) ** 2) / 12,
        lens_mass * (3 * lens_radius ** 2 + (2 * lens_half_length) ** 2) / 12,
        lens_mass * lens_radius ** 2 / 2,
    ])
    mass, center_of_mass, inertia = combined_inertia([
        (camera_mass, body_center, camera_inertia),
        (lens_mass, lens_center, lens_inertia),
    ])

    # The bracket is still a configurable approximation; camera and lens dimensions
    # and masses below are the selected products' published values.
    bracket_start = vector(mechanical["bracket_start_parent_m"], 3, "bracket start")
    bracket_radius = float(mechanical["bracket_radius_m"])
    ET.SubElement(parent, "geom", name="ee_camera_bracket_visual", type="capsule",
                  **{"class": "visual"}, fromto=numbers(np.r_[bracket_start, position]),
                  size=f"{bracket_radius:.12g}", material="ee_camera_bracket")
    ET.SubElement(parent, "geom", name="ee_camera_bracket_collision", type="capsule",
                  **{"class": "collision"}, fromto=numbers(np.r_[bracket_start, position]),
                  size=f"{bracket_radius:.12g}")

    mount = ET.SubElement(parent, "body", name="camera_mount",
                          pos=numbers(position), quat=numbers(mount_quaternion),
                          gravcomp="1")
    ET.SubElement(mount, "inertial", pos=numbers(center_of_mass), mass=f"{mass:.12g}",
                  diaginertia=numbers(inertia))
    ET.SubElement(mount, "geom", name="ee_camera_housing_visual", type="box",
                  **{"class": "visual"}, pos=numbers(body_center),
                  size=numbers(body_half), material="ee_camera_body")
    ET.SubElement(mount, "geom", name="ee_camera_housing_collision", type="box",
                  **{"class": "collision"}, pos=numbers(body_center),
                  size=numbers(body_half))
    ET.SubElement(mount, "geom", name="ee_camera_lens_visual", type="cylinder",
                  **{"class": "visual"}, pos=numbers(lens_center),
                  size=numbers([lens_radius, lens_half_length]),
                  material="ee_camera_lens")
    ET.SubElement(mount, "geom", name="ee_camera_lens_collision", type="cylinder",
                  **{"class": "collision"}, pos=numbers(lens_center),
                  size=numbers([lens_radius, lens_half_length]))
    lens_front_z = lens_center[2] - lens_half_length
    ET.SubElement(mount, "geom", name="ee_camera_glass_visual", type="cylinder",
                  **{"class": "visual"}, pos=numbers([0, 0, lens_front_z - 0.0005]),
                  size=numbers([0.72 * lens_radius, 0.0005]),
                  material="ee_camera_lens")
    ET.SubElement(mount, "site", name="camera_mount_frame", pos="0 0 0",
                  size="0.002", group="4", rgba="0 0 0 0")

    optical_cfg = config["optical_frame"]
    optical_position = vector(optical_cfg["position_mount_m"], 3, "optical position")
    optical_quaternion = vector(optical_cfg["quaternion_mount_wxyz"], 4,
                                "optical quaternion")
    optical_quaternion = unit(optical_quaternion, "optical quaternion")
    optical = ET.SubElement(mount, "body", name="camera_optical",
                            pos=numbers(optical_position), quat=numbers(optical_quaternion))
    ET.SubElement(optical, "site", name="camera_optical_frame", pos="0 0 0",
                  size="0.002", group="4", rgba="0 0 0 0")
    intrinsics = profile["intrinsics_nominal"]
    ET.SubElement(optical, "camera", name=camera_name, mode="fixed",
                  resolution=f"{width} {height}",
                  focalpixel=numbers([intrinsics["fx"], intrinsics["fy"]]),
                  principalpixel=numbers([intrinsics["cx"], intrinsics["cy"]]),
                  sensorsize=numbers(hardware["sensor_size_m"]))

    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    depth = config["simulation_ground_truth_depth"]
    metadata = {
        "ee_camera_resolution": [width, height],
        "ee_camera_intrinsics": [intrinsics["fx"], intrinsics["fy"],
                                 intrinsics["cx"], intrinsics["cy"]],
        "ee_camera_sensor_size_m": hardware["sensor_size_m"],
        "ee_camera_native_resolution": hardware["native_resolution"],
        "ee_camera_pixel_size_um": hardware["pixel_size_um"],
        "ee_camera_lens_focal_m": [hardware["lens_focal_length_m"]],
        "ee_camera_min_working_distance_m": [hardware["lens_min_working_distance_m"]],
        "ee_camera_real_depth_stream": [1 if hardware["real_depth_stream"] else 0],
        "ee_camera_sim_gt_depth_range_m": [depth["min_m"], depth["max_m"]],
        "ee_camera_sim_gt_depth_noise_std_m": [depth["noise_std_m"]],
        "ee_camera_assembly_mass_kg": [mass],
        "ee_camera_preview_hz": [config["preview_hz"]],
        "ee_camera_resize_scale_xy": profile["resize_scale_xy"],
    }
    for name, values in metadata.items():
        ET.SubElement(custom, "numeric", name=name, data=numbers(values))
    ET.SubElement(custom, "text", name="ee_camera_config",
                  data="blackfly_s_bfs_u3_51s5c_c__edmund_59_870__schema_v2")

    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print("SOURCE_UNCHANGED:", source)
    print("CREATED_SCENE:", output)
    print("CAMERA_NAME:", camera_name)
    print("HARDWARE_CAMERA:", hardware["camera_model"])
    print("HARDWARE_LENS:", hardware["lens_model"])
    print("REAL_DEPTH_STREAM:", bool(hardware["real_depth_stream"]))
    print("PARENT_BODY:", parent_name)
    print("RESOLUTION:", [width, height])
    print("MOUNT_POSITION_PARENT_M:", position.tolist())
    print("MOUNT_QUATERNION_PARENT_WXYZ:", mount_quaternion.tolist())


if __name__ == "__main__":
    main()
