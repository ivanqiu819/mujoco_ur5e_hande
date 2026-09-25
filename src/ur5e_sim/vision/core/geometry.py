"""读取真实 STL，并把几何和人工标注的特征统一到端口坐标系。

STL 本身不记录单位，也不记录“哪一个孔是端口”。因此这两项必须由
配置提供。这里不生成替代模型，不更改 STL，也不自动猜测端口位置。
内部长度一律是米；端口局部 X 向右、Y 向下、Z 沿插入方向。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class Geometry:
    """端口坐标下的网格及特征；数组长度单位均为米。

    ``T_stl_port`` 把端口局部坐标变换到 STL 的原始轴向和原点。
    该矩阵的平移也以米表示；若要还原 STL 文件中的数值，再除以
    ``unit_scale``。例如毫米文件的 ``unit_scale`` 为 0.001。
    """

    vertices: np.ndarray
    faces: np.ndarray
    landmarks: dict[str, np.ndarray]
    T_stl_port: np.ndarray
    unit_scale: float


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """将 N×3 行向量点集通过 4×4 刚体变换，返回 N×3 数组。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(transform, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} 必须是三个有限数值")
    return vector


def _rectangle(size: Any, scale: float, name: str) -> np.ndarray:
    """前表面四角，顺序固定为左上、右上、右下、左下。"""
    size = np.asarray(size, dtype=np.float64)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError(f"{name} 必须是两个大于零的尺寸 [宽, 高]")
    width, height = size * scale
    return np.array([
        [-width / 2, -height / 2, 0],
        [width / 2, -height / 2, 0],
        [width / 2, height / 2, 0],
        [-width / 2, height / 2, 0],
    ], dtype=np.float64)


def load_geometry(config: dict[str, Any]) -> Geometry:
    """加载 STL，按配置定义端口坐标，并返回供渲染和求解共用的几何。

    ``normal`` 是开口向外方向，``up`` 是模型中的向上方向。
    向上向量会投影到开口平面，因此二者可以不严格正交，但不能平行。
    顶面后侧点按“左后、右后”填写 STL 原坐标，允许用空列表禁用。
    """
    model = config["model"]
    port = config["port"]
    unit = model.get("unit", "mm")
    scales = {"mm": 0.001, "cm": 0.01, "m": 1.0}
    if unit not in scales:
        raise ValueError("model.unit 仅支持 'mm'、'cm' 或 'm'；STL 本身不包含单位")
    scale = scales[unit]
    path = Path(model["path"])
    if not path.is_file():
        raise ValueError(f"找不到 STL 文件：{path}")
    if path.suffix.lower() != ".stl":
        raise ValueError(f"model.path 必须指向 STL 文件：{path}")

    center = _vector(port["center"], "port.center") * scale
    outward = _vector(port["normal"], "port.normal")
    up = _vector(port["up"], "port.up")
    if np.linalg.norm(outward) < 1e-12 or np.linalg.norm(up) < 1e-12:
        raise ValueError("port.normal 和 port.up 均不能为零向量")
    outward = outward / np.linalg.norm(outward)
    up = up / np.linalg.norm(up)
    right = np.cross(up, outward)
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("port.normal 和 port.up 不能平行；无法定义端口的左右方向")
    right /= np.linalg.norm(right)
    inward = -outward
    down = np.cross(inward, right)
    T_stl_port = np.eye(4, dtype=np.float64)
    T_stl_port[:3, :3] = np.column_stack([right, down, inward])
    T_stl_port[:3, 3] = center
    T_port_stl = np.linalg.inv(T_stl_port)

    inner = _rectangle(port["inner_size"], scale, "port.inner_size")
    outer = _rectangle(port["outer_size"], scale, "port.outer_size")
    if np.any(np.asarray(port["inner_size"]) >= np.asarray(port["outer_size"])):
        raise ValueError("内口宽高必须分别小于外口宽高")

    rear = np.asarray(port.get("top_rear_points", []), dtype=np.float64)
    if rear.size == 0:
        rear = np.empty((0, 3), dtype=np.float64)
    elif rear.shape != (2, 3) or not np.isfinite(rear).all():
        raise ValueError("port.top_rear_points 必须是两个 STL 坐标点 [左后, 右后] 或 []")
    rear = transform_points(rear * scale, T_port_stl)
    if len(rear) and (np.any(rear[:, 2] <= 1e-9) or np.linalg.norm(rear[0] - rear[1]) < 1e-9):
        raise ValueError("顶面后侧两点必须不同，且位于端口前表面后方（局部 Z > 0）")

    try:
        # process=False 保留文件给出的三角形，避免隐式合并或修补 CAD。
        mesh = trimesh.load(str(path), force="mesh", process=False)
    except Exception as exc:
        raise ValueError(f"STL 读取失败：{path}；{exc}") from exc
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"STL 没有可用的三角网格：{path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("STL 包含无效顶点或非三角形面")
    if np.max(np.ptp(vertices, axis=0)) <= 0:
        raise ValueError("STL 的所有顶点重合，无法成像")
    triangles = vertices[faces]
    if not np.any(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1) > 0):
        raise ValueError("STL 的所有三角形均退化，无法成像")
    vertices = transform_points(vertices * scale, T_port_stl)
    landmarks = {"inner": inner, "outer": outer, "top_rear": rear}
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    tolerance = max(float(np.linalg.norm(upper - lower)) * 1e-6, 1e-9)
    if np.any(lower > tolerance) or np.any(upper < -tolerance):
        raise ValueError("port.center 位于 STL 网格包围盒之外，请检查坐标和 model.unit")
    for name, points in landmarks.items():
        if len(points) and (np.any(points < lower - tolerance) or np.any(points > upper + tolerance)):
            label = {"inner": "port.inner_size", "outer": "port.outer_size", "top_rear": "port.top_rear_points"}[name]
            raise ValueError(f"{label} 生成的特征点超出 STL 网格包围盒，请检查端口位置、方向和尺寸")
    return Geometry(vertices, faces, landmarks, T_stl_port, scale)
