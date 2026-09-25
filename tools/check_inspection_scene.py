#!/usr/bin/env python3
"""Validate the table, supplied workpiece, camera visibility and robot home state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from ur5e_sim.camera.calibration import CameraCapture, camera_transforms, load_spec, ROOT


def geom_world_aabb(model, data, geom_name):
    geom = model.geom(geom_name)
    size = model.geom_size[geom.id]
    local = np.array([
        [x, y, z]
        for x in (-size[0], size[0])
        for y in (-size[1], size[1])
        for z in (-size[2], size[2])
    ])
    rotation = data.geom_xmat[geom.id].reshape(3, 3)
    world = local @ rotation.T + data.geom_xpos[geom.id]
    return world.min(axis=0), world.max(axis=0)


def project_world_points(points_world, world_from_cv, spec):
    """Project world points using the OpenCV-style camera frame."""
    cv_from_world = np.linalg.inv(world_from_cv)
    homogeneous = np.c_[points_world, np.ones(len(points_world))]
    points_cv = (cv_from_world @ homogeneous.T).T[:, :3]
    if np.any(points_cv[:, 2] <= 0):
        raise RuntimeError("At least one workpiece corner is behind the camera")
    pixels = np.column_stack([
        spec.fx * points_cv[:, 0] / points_cv[:, 2] + spec.cx,
        spec.fy * points_cv[:, 1] / points_cv[:, 2] + spec.cy,
    ])
    return points_cv, pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path,
                        default=ROOT / "scenes/scene_inspection.xml")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/inspection.json")
    parser.add_argument("--camera-config", type=Path,
                        default=ROOT / "configs/camera.json")
    parser.add_argument("--capture-dir", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    spec = load_spec(args.camera_config)
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    workpiece_lower, workpiece_upper = geom_world_aabb(
        model, data, "inspection_workpiece_collision")
    table_lower, table_upper = geom_world_aabb(
        model, data, "inspection_table_top")
    clearance = float(workpiece_lower[2] - table_upper[2])
    expected_clearance = float(config["workpiece"]["table_clearance_m"])

    _, world_from_cv = camera_transforms(model, data, spec.name)
    site = model.site("workpiece_frame")
    center_world = data.site_xpos[site.id]
    center_cv_points, center_pixels = project_world_points(
        center_world[None, :], world_from_cv, spec)
    center_cv = center_cv_points[0]
    pixel = center_pixels[0]
    center_visible = (0 <= pixel[0] < spec.width
                      and 0 <= pixel[1] < spec.height)

    workpiece_corners = np.array([
        [x, y, z]
        for x in (workpiece_lower[0], workpiece_upper[0])
        for y in (workpiece_lower[1], workpiece_upper[1])
        for z in (workpiece_lower[2], workpiece_upper[2])
    ])
    _, corner_pixels = project_world_points(
        workpiece_corners, world_from_cv, spec)
    pixel_lower = corner_pixels.min(axis=0)
    pixel_upper = corner_pixels.max(axis=0)
    fully_visible = bool(
        pixel_lower[0] >= 0 and pixel_lower[1] >= 0
        and pixel_upper[0] < spec.width
        and pixel_upper[1] < spec.height)

    validation = config.get("camera_validation", {})
    require_center = bool(validation.get("require_center_visible", True))
    require_full = bool(
        validation.get("require_full_workpiece_visible_at_home", False))

    failures = []
    if abs(clearance - expected_clearance) > 1e-5:
        failures.append(
            f"workpiece/table clearance mismatch: {clearance:.8f} m")
    if data.ncon:
        pairs = [
            (model.geom(data.contact[i].geom1).name,
             model.geom(data.contact[i].geom2).name,
             float(data.contact[i].dist))
            for i in range(data.ncon)
        ]
        failures.append(f"unexpected home contacts: {pairs}")
    if require_center and not center_visible:
        failures.append(f"workpiece center is outside camera image: {pixel}")
    if require_full and not fully_visible:
        failures.append(
            "workpiece is not fully visible at home; projected pixel bounds "
            f"are {pixel_lower.tolist()} to {pixel_upper.tolist()}")

    print("SCENE:", args.scene.resolve())
    print("MODEL_COUNTS:", {
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "nbody": model.nbody, "ngeom": model.ngeom,
    })
    print("TABLE_TOP_AABB_M:", [table_lower.tolist(), table_upper.tolist()])
    print("WORKPIECE_AABB_M:",
          [workpiece_lower.tolist(), workpiece_upper.tolist()])
    print("WORKPIECE_CLEARANCE_M:", clearance)
    print("WORKPIECE_CENTER_WORLD_M:", center_world.tolist())
    print("WORKPIECE_CENTER_CAMERA_OPENCV_M:", center_cv.tolist())
    print("WORKPIECE_CENTER_PIXEL_UV:", pixel.tolist())
    print("WORKPIECE_CENTER_VISIBLE:", bool(center_visible))
    print("WORKPIECE_PROJECTED_PIXEL_BOUNDS:",
          [pixel_lower.tolist(), pixel_upper.tolist()])
    print("WORKPIECE_FULLY_VISIBLE_AT_HOME:", fully_visible)
    print("HOME_CONTACT_COUNT:", int(data.ncon))

    if args.capture_dir is not None:
        from PIL import Image
        output = args.capture_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        capture = CameraCapture(model, spec)
        try:
            frames = capture.capture(data, add_sim_depth_noise=False)
        finally:
            capture.close()
        Image.fromarray(frames["rgb"]).save(output / "inspection_rgb.png")
        np.save(output / "inspection_sim_depth_gt.npy",
                frames["sim_depth_gt_m"])
        np.save(output / "inspection_segmentation.npy",
                frames["segmentation"])
        segmentation = frames["segmentation"]
        object_ids = segmentation[..., 0]
        object_types = segmentation[..., 1]
        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        visual_id = model.geom("inspection_workpiece_visual").id
        workpiece_pixels = np.count_nonzero(
            (object_types == geom_type) & (object_ids == visual_id))
        ratio = float(workpiece_pixels / object_ids.size)
        print("CAPTURE_DIR:", output)
        print("RGB_SHAPE:", list(frames["rgb"].shape))
        print("WORKPIECE_VISIBLE_PIXEL_RATIO:", ratio)
        if ratio <= 0:
            failures.append("workpiece is not visible in the rendered image")

    if failures:
        print("INSPECTION_SCENE_CHECK: FAILED")
        for failure in failures:
            print("FAILURE:", failure)
        raise SystemExit(1)
    print("INSPECTION_SCENE_CHECK: PASSED")


if __name__ == "__main__":
    main()
