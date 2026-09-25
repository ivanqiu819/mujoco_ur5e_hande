"""相机内参、直观拍摄角度与 OpenCV 坐标变换。

所有 T_camera_port 都把端口坐标转换成相机坐标：
    p_camera = R_camera_port @ p_port + t_camera_port
OpenCV 相机坐标为 X 向画面右、Y 向画面下、Z 向镜头前方。
长度计算使用米，配置中的 distance_mm / offset_mm 使用毫米。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def euler_rotation(angles_deg: list[float] | np.ndarray) -> np.ndarray:
    """[pitch, yaw, roll] 分别绕 X、Y、Z；组合为 Rz @ Ry @ Rx。"""
    pitch, yaw, roll = np.deg2rad(angles_deg)
    sx, sy, sz = np.sin([pitch, yaw, roll])
    cx, cy, cz = np.cos([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def camera_parameters(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """返回 K、畸变系数、(宽, 高)。畸变遵循 OpenCV 系数顺序。"""
    camera = config["camera"]
    intrinsics = camera["intrinsics"]
    K = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.asarray(camera.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
    width, height = camera.get("image_size", [1280, 720])
    return K, dist, (int(width), int(height))


def nominal_pose(config: dict[str, Any]) -> np.ndarray:
    """由拍摄距离和角度生成名义 T_camera_port。

    零角时相机位于端口局部 -Z，正对端口；画面右方为局部 +X。
    正水平角把相机移向 +X，正仰角把相机移向 -Y（端口上方）。
    正 roll 使成像画面顺时针旋转。target_offset_mm 是局部坐标下的
    注视目标偏移，相机围绕这一目标运动，distance_mm 为到目标的距离。
    """
    camera = config["camera"]
    azimuth, elevation = np.deg2rad([
        camera.get("azimuth_deg", 15.0), camera.get("elevation_deg", 15.0),
    ])
    distance = float(camera.get("distance_mm", 180.0)) * 0.001
    target = np.asarray(camera.get("target_offset_mm", [0, 0, 0]), dtype=np.float64) * 0.001
    if distance <= 0 or not np.isfinite(distance):
        raise ValueError("camera.distance_mm 必须大于零且有限")
    # 用解析的右方向避免叉积形式在俯视极点发生除零。
    offset = np.array([
        np.sin(azimuth) * np.cos(elevation),
        -np.sin(elevation),
        -np.cos(azimuth) * np.cos(elevation),
    ])
    camera_center = target + distance * offset
    forward = -offset
    right = np.array([np.cos(azimuth), 0, np.sin(azimuth)])
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    rotation = euler_rotation([0, 0, camera.get("roll_deg", 0)]) @ rotation
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ camera_center
    return transform


def actual_pose(nominal: np.ndarray, simulation: dict[str, Any]) -> np.ndarray:
    """叠加端口局部坐标中的小扰动，仅用于模拟真实拍摄。

    T_true = T_nominal @ T_offset。旋转绕端口原点；平移分量沿名义
    端口的 X/Y/Z 方向。检测器和求解器仍只接收 T_nominal。
    """
    offset = np.eye(4, dtype=np.float64)
    offset[:3, :3] = euler_rotation(simulation.get("rotation_offset_deg", [0, 0, 0]))
    offset[:3, 3] = np.asarray(simulation.get("translation_offset_mm", [0, 0, 0]), dtype=float) * 0.001
    return np.asarray(nominal, dtype=float) @ offset


def pose_vectors(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4×4 T_camera_port 转换成 OpenCV 的 3×1 rvec 和 tvec。"""
    transform = np.asarray(transform, dtype=np.float64)
    return cv2.Rodrigues(transform[:3, :3])[0], transform[:3, 3].reshape(3, 1).copy()


def transform_from_vectors(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """OpenCV 的 rvec/tvec 转换成 4×4 T_camera_port。"""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def project_points(points: np.ndarray, transform: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """把端口局部 N×3 点投影为 N×2 像素坐标；不进行可见性判断。

    投影落在图内不代表未遮挡；渲染器负责遮挡，检测器只能用图像确认边缘。
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return np.empty((0, 2), dtype=np.float64)
    rvec, tvec = pose_vectors(transform)
    pixels, _ = cv2.projectPoints(points, rvec, tvec, K, dist)
    return pixels.reshape(-1, 2)
