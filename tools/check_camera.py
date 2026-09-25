#!/usr/bin/env python3
"""Validate camera metadata, transforms, home contacts and optional rendering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from ur5e_sim.camera.calibration import CameraCapture, camera_transforms, load_spec, pose_matrix, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "scenes/scene_camera.xml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/camera.json")
    parser.add_argument(
        "--capture-dir", type=Path,
        help="Optionally save rgb.png, sim_depth_gt.npy and segmentation.npy")
    args = parser.parse_args()
    spec = load_spec(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    camera = model.camera(spec.name)
    compiled_resolution = model.cam_resolution[camera.id].astype(int)
    sensor_size = model.cam_sensorsize[camera.id]
    compiled_focal = (model.cam_intrinsic[camera.id, :2] / sensor_size
                      * compiled_resolution)
    compiled_principal = (model.cam_intrinsic[camera.id, 2:] / sensor_size
                          * compiled_resolution)
    expected = np.array([spec.fx, spec.fy, spec.cx, spec.cy])
    actual = np.r_[compiled_focal, compiled_principal]
    if not np.array_equal(compiled_resolution, [spec.width, spec.height]):
        raise RuntimeError(f"Resolution mismatch: {compiled_resolution}")
    if not np.allclose(actual, expected, atol=1e-9):
        raise RuntimeError(f"Intrinsics mismatch: {actual} != {expected}")
    if data.ncon:
        pairs = [(model.geom(data.contact[i].geom1).name,
                  model.geom(data.contact[i].geom2).name,
                  float(data.contact[i].dist)) for i in range(data.ncon)]
        raise RuntimeError(f"Camera installation creates home contacts: {pairs}")

    world_from_mj, world_from_cv = camera_transforms(model, data, spec.name)
    tcp = model.site("tcp")
    world_from_tcp = pose_matrix(data.site_xpos[tcp.id], data.site_xmat[tcp.id])
    tcp_from_cv = np.linalg.inv(world_from_tcp) @ world_from_cv
    cv_from_tcp = np.linalg.inv(tcp_from_cv)
    optical_to_tcp = float(np.linalg.norm(data.cam_xpos[camera.id]
                                          - data.site_xpos[tcp.id]))
    parent = model.body(str(config["parent_body"]))
    parent_rotation = data.xmat[parent.id].reshape(3, 3)
    look_at_parent = np.asarray(config["mount"]["look_at_parent_m"],
                                dtype=np.float64)
    look_at_world = data.xpos[parent.id] + parent_rotation @ look_at_parent
    optical_to_look_at = float(np.linalg.norm(data.cam_xpos[camera.id]
                                              - look_at_world))
    if optical_to_look_at < spec.lens_min_working_distance_m:
        raise RuntimeError(
            "Configured look-at point lies inside the lens minimum working "
            f"distance: {optical_to_look_at:.6f} < "
            f"{spec.lens_min_working_distance_m:.6f} m")
    hfov = 2 * np.degrees(np.arctan(spec.width / (2 * spec.fx)))
    vfov = 2 * np.degrees(np.arctan(spec.height / (2 * spec.fy)))

    print("CAMERA_CHECK: PASSED")
    print("SCENE:", args.scene.resolve())
    print("CAMERA:", spec.name)
    print("CAMERA_MODEL:", spec.camera_model)
    print("LENS_MODEL:", spec.lens_model)
    print("NATIVE_RESOLUTION_WH:", [spec.native_width, spec.native_height])
    print("OUTPUT_RESOLUTION_WH:", compiled_resolution.tolist())
    print("NOMINAL_INTRINSICS_FX_FY_CX_CY:", actual.tolist())
    print("INTRINSICS_CALIBRATED:", False)
    print("NOMINAL_PINHOLE_FOV_DEG_HV:", [float(hfov), float(vfov)])
    print("LENS_DISTORTION_SIMULATED:", False)
    print("REAL_DEPTH_STREAM:", spec.real_depth_stream)
    print("SIM_GT_DEPTH_RANGE_M:",
          [spec.sim_depth_min_m, spec.sim_depth_max_m])
    print("OPTICAL_TO_TCP_DISTANCE_M:", optical_to_tcp)
    print("OPTICAL_TO_LOOK_AT_DISTANCE_M:", optical_to_look_at)
    print("LENS_MIN_WORKING_DISTANCE_M:", spec.lens_min_working_distance_m)
    print("HOME_CONTACT_COUNT:", int(data.ncon))
    print("T_WORLD_FROM_CAMERA_MUJOCO:\n", world_from_mj)
    print("T_WORLD_FROM_CAMERA_OPENCV:\n", world_from_cv)
    print("T_TCP_FROM_CAMERA_OPENCV:\n", tcp_from_cv)
    print("T_CAMERA_OPENCV_FROM_TCP:\n", cv_from_tcp)

    if args.capture_dir is not None:
        from PIL import Image
        output = args.capture_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        capture = CameraCapture(model, spec)
        try:
            frames = capture.capture(data, add_sim_depth_noise=False)
        finally:
            capture.close()
        Image.fromarray(frames["rgb"]).save(output / "rgb.png")
        np.save(output / "sim_depth_gt.npy", frames["sim_depth_gt_m"])
        np.save(output / "segmentation.npy", frames["segmentation"])
        valid = np.isfinite(frames["sim_depth_gt_m"])
        print("CAPTURE_DIR:", output)
        print("RGB_SHAPE:", list(frames["rgb"].shape))
        print("SIM_GT_DEPTH_VALID_RATIO:", float(np.mean(valid)))
        if np.any(valid):
            print("SIM_GT_DEPTH_VALID_MIN_MAX_M:",
                  [float(np.nanmin(frames["sim_depth_gt_m"])),
                   float(np.nanmax(frames["sim_depth_gt_m"]))])
        print("SEGMENTATION_SHAPE:", list(frames["segmentation"].shape))


if __name__ == "__main__":
    main()
