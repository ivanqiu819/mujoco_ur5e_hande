"""真实 STL 三角网格的 CPU 渲染器。

每个像素都比较相机深度（Z-buffer），而不是按三角形中心排序覆盖。
网格全程使用端口局部米坐标。材质只是可配置的灰度，不改变几何。
深度图仅供遮挡检查/调试；图像检测器绝不接收它。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .geometry import Geometry


@dataclass
class RenderResult:
    image: np.ndarray  # H×W×3，OpenCV BGR uint8
    depth: np.ndarray  # H×W，相机 Z，米；背景为 inf


@dataclass(frozen=True)
class SurfacePatch:
    """附加的薄片表面；顶点为端口坐标（米），每个三角形一个材质灰度。

    三角形顶点逆时针围绕印刷面的外法向排列。只绘制朝向相机的一面，
    防止从背面看到镜像码。贴纸和 STL 共用深度缓存，不是照片上的覆盖图。
    """

    vertices: np.ndarray
    faces: np.ndarray
    grays: np.ndarray


def triangle_materials(geometry: Geometry, settings: dict[str, Any]) -> np.ndarray:
    """用原 STL 坐标范围选材质；按全部顶点判断，避免跨界的大三角形误着色。"""
    transform = geometry.T_stl_port
    original = (geometry.vertices @ transform[:3, :3].T + transform[:3, 3]) / geometry.unit_scale
    triangles = original[geometry.faces]
    face_min, face_max = triangles.min(axis=1), triangles.max(axis=1)
    materials = np.full(len(geometry.faces), float(settings["default_gray"]))
    # STL 单精度顶点有约 1e-6 模型单位的误差，边界匹配留极小容差。
    tolerance = max(float(np.ptp(original, axis=0).max()) * 1e-7, 1e-7)
    for region in settings["material_regions"]:
        inside = np.all(face_min >= np.asarray(region["min"]) - tolerance, axis=1)
        inside &= np.all(face_max <= np.asarray(region["max"]) + tolerance, axis=1)
        materials[inside] = float(region["gray"])
    return materials


def _clip_near(triangle: np.ndarray, near: float) -> list[np.ndarray]:
    """在相机近裁剪面切三角形，返回零到两个三角形。"""
    polygon: list[np.ndarray] = []
    for start, end in zip(triangle, np.roll(triangle, -1, axis=0)):
        start_inside, end_inside = start[2] >= near, end[2] >= near
        if start_inside:
            polygon.append(start)
        if start_inside != end_inside:
            ratio = (near - start[2]) / (end[2] - start[2])
            polygon.append(start + ratio * (end - start))
    return [np.array([polygon[0], polygon[i], polygon[i + 1]]) for i in range(1, len(polygon) - 1)]


def _rasterize(triangle: np.ndarray, gray: float, K: np.ndarray,
               image: np.ndarray, depth: np.ndarray) -> None:
    """光栅化一个已裁剪三角形；透视深度通过 1/Z 插值恢复。"""
    height, width = depth.shape
    projected = triangle @ K.T
    pixels = projected[:, :2] / projected[:, 2:]
    if not np.isfinite(pixels).all():
        return
    xmin = max(0, int(np.ceil(pixels[:, 0].min())))
    xmax = min(width - 1, int(np.floor(pixels[:, 0].max())))
    ymin = max(0, int(np.ceil(pixels[:, 1].min())))
    ymax = min(height - 1, int(np.floor(pixels[:, 1].max())))
    if xmax < xmin or ymax < ymin:
        return
    a, b, c = pixels
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denominator) < 1e-10:
        return
    # OpenCV 像素坐标的整数点就是像素中心，不额外加 0.5。
    xx, yy = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    wa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
    wb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
    wc = 1 - wa - wb
    inside = (wa >= -1e-8) & (wb >= -1e-8) & (wc >= -1e-8)
    inverse_z = wa / triangle[0, 2] + wb / triangle[1, 2] + wc / triangle[2, 2]
    z = np.full(inverse_z.shape, np.inf)
    np.divide(1.0, inverse_z, out=z, where=inside & (inverse_z > 0))
    previous = depth[ymin:ymax + 1, xmin:xmax + 1]
    update = inside & (z < previous)
    # 交线/共边上的深度可能只差 float32 舍入误差。用确定性的较亮材质
    # 解决这种几何重合，避免三角形文件顺序改变最终像素。
    coincident = inside & np.isclose(z, previous, rtol=1e-7, atol=1e-9)
    region = image[ymin:ymax + 1, xmin:xmax + 1]
    region[update & ~coincident] = gray
    region[coincident] = np.maximum(region[coincident], gray)
    previous[update] = z[update]


def _apply_distortion(image: np.ndarray, depth: np.ndarray, K: np.ndarray,
                      distortion: np.ndarray, background: float) -> tuple[np.ndarray, np.ndarray]:
    """对最终图像施加 OpenCV 畸变，与 projectPoints 使用相同模型。

    remap 需要“目标像素→源像素”，所以把畸变图上的每个像素反求到
    无畸变图。直接在已畸变的三角形顶点之间画直线会遗漏边的弯曲。
    """
    if not np.any(distortion):
        return image, depth
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    grid = np.stack([x, y], axis=-1).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(grid, K, distortion, P=K).reshape(height, width, 2)
    map_x, map_y = undistorted[..., 0], undistorted[..., 1]
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    warped_depth = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=float("inf"))
    return warped, warped_depth


def render_scene(geometry: Geometry, T_camera_port: np.ndarray, K: np.ndarray,
                 dist: np.ndarray, image_size: tuple[int, int],
                 config: dict[str, Any], *, surface_patches: tuple[SurfacePatch, ...] = ()) -> RenderResult:
    """渲染真实网格，返回 RGB 可视内容与独立的调试深度。

    无需显示器或 GPU；近裁剪、透视深度、镜头畸变均在此处理。
    """
    output_width, output_height = image_size
    settings = config["render"]
    samples = int(settings.get("supersample", 3))
    width, height = output_width * samples, output_height * samples
    raster_K = np.asarray(K, dtype=float).copy()
    raster_K[:2] *= samples
    # 奇数采样保证每个原始像素中心恰好落到中心子像素，深度仍可精确返回。
    raster_K[0, 2] += (samples - 1) / 2
    raster_K[1, 2] += (samples - 1) / 2
    background = float(settings["background_gray"])
    image = np.full((height, width), background, dtype=np.float32)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    rotation, translation = T_camera_port[:3, :3], T_camera_port[:3, 3]
    camera_vertices = geometry.vertices @ rotation.T + translation
    triangles = camera_vertices[geometry.faces]
    materials = triangle_materials(geometry, settings)

    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-20)
    light = np.asarray(settings["light_direction_camera"], dtype=float)
    light /= np.linalg.norm(light)
    illumination = settings["ambient"] + (1 - settings["ambient"]) * np.maximum(normals @ light, 0)
    colors = materials * illumination
    near = float(settings["near_clip_mm"]) / 1000
    # 不剔除背面：在开口网格/薄片模型中，背面也应该参与实际遮挡。
    for triangle, color, area in zip(triangles, colors, lengths):
        if area <= 1e-16 or triangle[:, 2].max() < near:
            continue
        for clipped in _clip_near(triangle, near):
            _rasterize(clipped, float(color), raster_K, image, depth)

    # 贴纸的所有黑白格是互不重叠的实体色块；先参与深度和光照计算，
    # 然后与整个场景一起施加畸变、缩小、模糊及噪声。
    for patch in surface_patches:
        patch_triangles = (patch.vertices @ rotation.T + translation)[patch.faces]
        patch_normals = np.cross(patch_triangles[:, 1] - patch_triangles[:, 0],
                                 patch_triangles[:, 2] - patch_triangles[:, 0])
        patch_areas = np.linalg.norm(patch_normals, axis=1)
        patch_normals /= np.maximum(patch_areas[:, None], 1e-20)
        patch_colors = patch.grays * (settings["ambient"] + (1 - settings["ambient"])
                                     * np.maximum(patch_normals @ light, 0))
        for triangle, normal, color, area in zip(patch_triangles, patch_normals, patch_colors, patch_areas):
            if area <= 1e-16 or triangle[:, 2].max() < near or np.dot(normal, triangle.mean(axis=0)) >= 0:
                continue
            for clipped in _clip_near(triangle, near):
                _rasterize(clipped, float(color), raster_K, image, depth)

    image, depth = _apply_distortion(image, depth, raster_K, dist, background)
    if samples > 1:
        image = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)
        depth = depth[samples // 2::samples, samples // 2::samples].copy()
    image = cv2.GaussianBlur(image, (3, 3), 0.55)
    simulation = config["simulation"]
    rng = np.random.default_rng(int(simulation["random_seed"]))
    noise = rng.normal(0, float(simulation["noise_std_gray"]), image.shape)
    gray = np.clip(image + noise, 0, 255).astype(np.uint8)
    return RenderResult(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), depth)
