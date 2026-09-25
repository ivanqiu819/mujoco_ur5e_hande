"""ArUco 贴纸、安装关系与全图识别；不读取拍摄外参或仿真真值。

标记坐标：X 向码右、Y 向码上、Z 朝印刷面外。配置的 T_port_marker
把这个坐标系转换到端口坐标系；贴纸与 STL 始终是同一个刚体。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .camera import euler_rotation
from .geometry import transform_points
from .renderer import SurfacePatch


@dataclass
class MarkerDetection:
    """points 只含通过像素质量检查的目标 ID，键 marker_0/1 对应目标编号。

    diagnostics 保留全部解码 ID、四角及未通过解码的四边形，供报告使用。
    success 仅表示发现可用标记，不代表已确定唯一位姿。
    """

    points: dict[str, np.ndarray]
    diagnostics: dict[str, Any]
    success: bool
    reason: str


def get_dictionary(name: str) -> Any:
    """获得 OpenCV 预定义字典，并对缺失 ArUco 支持给出可读错误。"""
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not all(hasattr(aruco, key) for key in ("ArucoDetector", "generateImageMarker")):
        raise ValueError("当前 OpenCV 缺少 ArUco API；请在 demo 环境安装 requirements.txt 中的 opencv-python>=4.8")
    if not isinstance(name, str) or not name.startswith("DICT_") or not isinstance(getattr(aruco, name, None), int):
        raise ValueError(f"aruco.dictionary 不是有效的 OpenCV 预定义字典：{name!r}")
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def marker_transform(settings: dict[str, Any]) -> np.ndarray:
    """返回 T_port_marker；配置平移为毫米，矩阵平移为米，R=Rz@Ry@Rx。"""
    transform = np.eye(4)
    transform[:3, :3] = euler_rotation(settings["rotation_port_deg"])
    transform[:3, 3] = np.asarray(settings["center_port_mm"], dtype=float) / 1000
    return transform


def marker_corners(size_m: float) -> np.ndarray:
    """IPPE_SQUARE 的固定顺序：左上、右上、右下、左下，均位于标记 Z=0。

    size_m 包含黑色边框，不包含白色留边。Y 朝码上，所以左上角 Y 为正。
    """
    half = size_m / 2
    return np.array([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]], dtype=np.float64)


def marker_image(settings: dict[str, Any]) -> np.ndarray:
    """导出带白边的正面码图；渲染使用实际毫米尺寸，不依赖这张图的分辨率。"""
    dictionary = get_dictionary(settings["dictionary"])
    cell_count = dictionary.markerSize + 2  # 外围黑框固定为一格宽。
    size = cell_count * 70
    code = cv2.aruco.generateImageMarker(dictionary, int(settings["marker_id"]), size)
    margin = max(1, round(size * settings["white_margin_mm"] / settings["marker_size_mm"]))
    return cv2.copyMakeBorder(code, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)


def marker_patch(settings: dict[str, Any]) -> SurfacePatch:
    """把码的每个黑白格及外围白边拆成三角片，安装到端口坐标下。

    不在完整白色底片上重复画黑格：共面片会产生深度冲突。本实现的
    格子内部互不重叠，外围白边也是单独的条带，所有法向均朝印刷面外。
    """
    dictionary = get_dictionary(settings["dictionary"])
    cells = dictionary.markerSize + 2
    bits = cv2.aruco.generateImageMarker(dictionary, int(settings["marker_id"]), cells)
    half, margin = settings["marker_size_mm"] / 2000, settings["white_margin_mm"] / 1000
    borders = np.r_[-half - margin, np.linspace(-half, half, cells + 1), half + margin]
    vertices, faces, grays = [], [], []
    for row in range(cells + 2):
        for column in range(cells + 2):
            x0, x1 = borders[column:column + 2]
            y0, y1 = borders[row:row + 2]
            start = len(vertices)
            # 图片行向下，而标记 Y 向上；因此这里对 y 取负。
            vertices.extend([[x0, -y0, 0], [x1, -y0, 0], [x1, -y1, 0], [x0, -y1, 0]])
            faces.extend([[start, start + 2, start + 1], [start, start + 3, start + 2]])
            gray = bits[row - 1, column - 1] if 1 <= row <= cells and 1 <= column <= cells else 255
            grays.extend([gray, gray])
    return SurfacePatch(transform_points(np.asarray(vertices), marker_transform(settings)),
                        np.asarray(faces, dtype=np.int64), np.asarray(grays, dtype=float))


def detect_markers(image: np.ndarray, settings: dict[str, Any]) -> MarkerDetection:
    """仅由图像检测、解码和精修角点；不需要 K、外参、深度图或 STL。

    OpenCV 返回的四角已经按编码方向排序，画面旋转不会导致 180° 命名混淆。
    不同 ID 保留在诊断中；同一个目标 ID 出现多次时全部交给求解器。
    """
    dictionary = get_dictionary(settings["dictionary"])
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = int(settings.get("corner_refinement_window_px", 3))
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected = detector.detectMarkers(image)
    height, width = image.shape[:2]
    points: dict[str, np.ndarray] = {}
    records = []
    for pixels, marker_id in zip(corners, [] if ids is None else ids.ravel()):
        pixels = np.asarray(pixels, dtype=float).reshape(4, 2)
        record: dict[str, Any] = {"marker_id": int(marker_id), "corners_px": pixels.tolist(),
                                  "target_id": None, "accepted": False}
        sides = np.linalg.norm(np.roll(pixels, -1, axis=0) - pixels, axis=1)
        if marker_id != settings["marker_id"]:
            record["reason"] = "ID 与配置不符，仅用于诊断"
        elif not np.isfinite(pixels).all() or np.any(pixels < 2) or np.any(pixels >= [width - 2, height - 2]):
            record["reason"] = "标记角点出画或过于靠近图像边界"
        elif sides.min() < settings.get("min_marker_side_px", 12):
            record["reason"] = "标记最短边像素不足，无法可靠估计位姿"
        else:
            target = len(points)
            points[f"marker_{target}"] = pixels
            record.update(accepted=True, target_id=target, reason="指定 ID 的四角观测可用")
        records.append(record)
    if points:
        reason = f"检测到 {len(points)} 个可用的目标标记"
    elif records:
        reason = "没有可用的目标标记：" + "；".join(sorted({r["reason"] for r in records}))
    else:
        reason = "未解码出 ArUco 标记；请检查仰角、遮挡、像素大小、字典及白边。"
    diagnostics = {"method": "OpenCV ArUco + subpixel corners", "use_prior": False,
                   "search_scope": "full_image", "target_id_expected": int(settings["marker_id"]),
                   "markers": records, "target_count": len(points),
                   "rejected_quads_px": [q.reshape(4, 2).tolist() for q in rejected]}
    return MarkerDetection(points, diagnostics, bool(points), reason)
