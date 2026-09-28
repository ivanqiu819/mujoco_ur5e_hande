#!/usr/bin/env python3
"""Reusable simulated output for the Blackfly S eye-in-hand camera.

The selected BFS-U3-51S5C-C is an RGB camera.  MuJoCo depth and segmentation
are exposed separately as simulator ground truth, not as hardware streams.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ur5e_sim.config_io import read_config


from ur5e_sim.paths import ROOT as ROOT


@dataclass(frozen=True)
class CameraSpec:
    name: str
    camera_model: str
    lens_model: str
    native_width: int
    native_height: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    sensor_width_m: float
    sensor_height_m: float
    lens_focal_length_m: float
    lens_min_working_distance_m: float
    lens_max_distortion_percent: float
    real_depth_stream: bool
    sim_depth_enabled: bool
    sim_depth_min_m: float
    sim_depth_max_m: float
    sim_depth_noise_std_m: float
    preview_hz: float

    @property
    def intrinsic_matrix(self):
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


def load_spec(path=ROOT / "configs/camera.json"):
    config = read_config(path)
    if config.get("schema_version") != 2:
        raise ValueError("camera_config.json must use schema_version 2")
    hardware = config["hardware"]
    profile = config["output_profile"]
    intrinsics = profile["intrinsics_nominal"]
    sim_depth = config["simulation_ground_truth_depth"]
    return CameraSpec(
        name=str(config["camera_name"]),
        camera_model=str(hardware["camera_model"]),
        lens_model=str(hardware["lens_model"]),
        native_width=int(hardware["native_resolution"][0]),
        native_height=int(hardware["native_resolution"][1]),
        width=int(profile["width"]),
        height=int(profile["height"]),
        fx=float(intrinsics["fx"]),
        fy=float(intrinsics["fy"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
        sensor_width_m=float(hardware["sensor_size_m"][0]),
        sensor_height_m=float(hardware["sensor_size_m"][1]),
        lens_focal_length_m=float(hardware["lens_focal_length_m"]),
        lens_min_working_distance_m=float(hardware["lens_min_working_distance_m"]),
        lens_max_distortion_percent=float(hardware["lens_max_distortion_percent"]),
        real_depth_stream=bool(hardware["real_depth_stream"]),
        sim_depth_enabled=bool(sim_depth["enabled"]),
        sim_depth_min_m=float(sim_depth["min_m"]),
        sim_depth_max_m=float(sim_depth["max_m"]),
        sim_depth_noise_std_m=float(sim_depth["noise_std_m"]),
        preview_hz=float(config["preview_hz"]),
    )


def pose_matrix(position, rotation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def camera_transforms(model, data, camera_name="ee_camera"):
    """Return world transforms for MuJoCo and OpenCV optical conventions.

    MuJoCo camera: +X right, +Y up, -Z forward.
    OpenCV camera: +X right, +Y down, +Z forward.
    """
    camera = model.camera(camera_name)
    world_from_mujoco = pose_matrix(data.cam_xpos[camera.id], data.cam_xmat[camera.id])
    convention = np.diag([1.0, -1.0, -1.0, 1.0])
    world_from_opencv = world_from_mujoco @ convention
    return world_from_mujoco, world_from_opencv


class CameraCapture:
    def __init__(self, model, spec=None, *, seed=0):
        self.model = model
        self.spec = spec or load_spec()
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.spec.name) < 0:
            raise ValueError(f"Camera not found in model: {self.spec.name}")
        self.renderer = mujoco.Renderer(model, height=self.spec.height,
                                        width=self.spec.width)
        self.random = np.random.default_rng(seed)

    def close(self):
        self.renderer.close()

    def rgb(self, data):
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(data, camera=self.spec.name)
        return self.renderer.render().copy()

    def sim_depth_gt(self, data, *, add_noise=True):
        """Render metric simulator ground-truth depth.

        This does not represent a stream produced by the physical Blackfly S.
        """
        if not self.spec.sim_depth_enabled:
            raise RuntimeError("Simulator ground-truth depth is disabled")
        self.renderer.disable_segmentation_rendering()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.spec.name)
        depth = self.renderer.render().astype(np.float32, copy=True)
        self.renderer.disable_depth_rendering()
        valid = np.isfinite(depth)
        valid &= depth >= self.spec.sim_depth_min_m
        valid &= depth <= self.spec.sim_depth_max_m
        if add_noise and self.spec.sim_depth_noise_std_m > 0:
            depth[valid] += self.random.normal(0, self.spec.sim_depth_noise_std_m,
                                               int(np.count_nonzero(valid)))
            valid &= depth >= self.spec.sim_depth_min_m
            valid &= depth <= self.spec.sim_depth_max_m
        depth[~valid] = np.nan
        return depth

    def segmentation(self, data):
        self.renderer.disable_depth_rendering()
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(data, camera=self.spec.name)
        segmentation = self.renderer.render().astype(np.int32, copy=True)
        self.renderer.disable_segmentation_rendering()
        return segmentation

    def capture(self, data, *, add_sim_depth_noise=True):
        return {
            "rgb": self.rgb(data),
            "sim_depth_gt_m": self.sim_depth_gt(
                data, add_noise=add_sim_depth_noise),
            "segmentation": self.segmentation(data),
        }
