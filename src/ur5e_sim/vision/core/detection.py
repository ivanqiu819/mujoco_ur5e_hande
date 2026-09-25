"""从模拟照片提取端口角点，不读取渲染深度、标签或实际位姿。

名义机位只负责圈定搜索范围和给角点命名。返回坐标全部来自图像中的
梯度边缘：先找内口四边形，再拟合边界直线，最后求相邻直线的交点。
如果图像没有足够证据，宁可报告失败，也不把 CAD 投影当作检测结果。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import project_points


@dataclass
class DetectionResult:
    """points 中每组坐标均为像素；四角按 TL、TR、BR、BL 排列。"""

    points: dict[str, np.ndarray]
    diagnostics: dict
    edges: np.ndarray
    success: bool
    reason: str


def _ordered_quad(quad: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, float]:
    """按名义投影关联语义；枚举顺序也支持相机画面旋转。"""
    choices = [np.roll(q, i, axis=0) for q in (quad, quad[::-1]) for i in range(4)]
    distances = [float(np.mean(np.linalg.norm(q - expected, axis=1))) for q in choices]
    best = int(np.argmin(distances))
    return choices[best].astype(np.float64), distances[best]


def _intersect(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    """齐次直线 ax+by+c=0 的交点；近乎平行时拒绝。"""
    point = np.cross(line_a, line_b)
    if abs(point[2]) < 0.08:
        return None
    return point[:2] / point[2]


def _fit_boundary(
    gray: np.ndarray,
    edge_xy: np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray],
    start: np.ndarray,
    end: np.ndarray,
    band: float,
    min_coverage: float = 0.45,
    polarity: int = 0,
) -> tuple[np.ndarray | None, dict]:
    """在预测线附近找到真实图像直线，并用梯度峰值作亚像素定位。

    搜索角度和法向偏移，避免把环形口相邻的两条边平均成一条虚假边。
    min_coverage 要求边缘在整段长度上分布，而非只经过几个噪声点。
    """
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 5.0:
        return None, {"reason": "边线像素长度不足"}
    tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]])
    midpoint = (start + end) * 0.5
    relative = edge_xy - midpoint
    along, across = relative @ tangent, relative @ normal
    # 舍弃端点附近：相邻边的梯度方向会在拐角处混合。
    selected = (np.abs(along) < length * 0.43) & (np.abs(across) < band + length * 0.09)
    pixels = edge_xy[selected]
    if len(pixels) < 5:
        return None, {"reason": "搜索带没有足够的图像边缘"}
    gx, gy = gradients
    xi, yi = pixels[:, 0].astype(int), pixels[:, 1].astype(int)
    vectors = np.column_stack((gx[yi, xi], gy[yi, xi]))
    magnitude = np.linalg.norm(vectors, axis=1)
    alignment = (vectors @ normal) / np.maximum(magnitude, 1e-6)
    # 内口从亮前唇跨入暗孔，梯度指向外，可排除孔腔深处的假边。
    # 外口可能邻接亮背景或暗底板，不固定其极性。这里只检查变化方向，
    # 不使用固定灰度值。
    aligned = (alignment * polarity > 0.78) if polarity else (np.abs(alignment) > 0.78)
    pixels, magnitude = pixels[aligned], magnitude[aligned]
    if len(pixels) < 5:
        return None, {"reason": "没有方向一致的边缘"}

    # 小角度扫描相当于一个局部 Hough 变换，不依赖物体的固定灰度。
    best_score, best_mask, best_normal = -1.0, None, None
    for angle in np.deg2rad(np.arange(-8.0, 8.01, 1.0)):
        n = normal * np.cos(angle) + tangent * np.sin(angle)
        offsets = (pixels - midpoint) @ n
        for offset in np.arange(-band, band + 0.25, 0.5):
            keep = np.abs(offsets - offset) < 1.15
            count = int(np.sum(keep))
            if count < 5:
                continue
            # 以支持长度为主，预测距离只作为同样强边缘的弱关联先验。
            score = count * (1.0 - 0.10 * abs(offset) / max(band, 1.0))
            if score > best_score:
                best_score, best_mask, best_normal = score, keep, n
    if best_mask is None:
        return None, {"reason": "无法拟合连续边线"}
    points = pixels[best_mask].copy()

    # 以梯度幅值在法线方向的三点抛物线极值，将整数 Canny 像素移至边缘中心。
    grad_magnitude = np.hypot(gx, gy)
    sampled = []
    for shift in (-1.0, 0.0, 1.0):
        coordinates = points + shift * best_normal
        sampled.append(cv2.remap(grad_magnitude, coordinates[:, 0].astype(np.float32)[:, None],
                                 coordinates[:, 1].astype(np.float32)[:, None], cv2.INTER_LINEAR).ravel())
    left, middle, right = sampled
    denominator = left - 2.0 * middle + right
    adjustment = np.divide(0.5 * (left - right), denominator,
                           out=np.zeros_like(middle), where=denominator < -1e-5)
    points += np.clip(adjustment, -0.75, 0.75)[:, None] * best_normal
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    line = np.array([-vy, vx, vy * x0 - vx * y0], dtype=np.float64)
    residual = np.abs(points @ line[:2] + line[2])
    if float(np.mean(residual)) > 0.85:
        return None, {"reason": "边缘直线残差过大"}
    positions = (points - start) @ tangent / length
    bins = np.unique(np.clip((positions * 10).astype(int), 0, 9))
    coverage = len(bins) / 10.0
    if coverage < min_coverage:
        return None, {"reason": "边线被遮挡或支持长度不足", "coverage": coverage}
    return line, {"support_pixels": len(points), "coverage": coverage,
                  "mean_residual_px": float(np.mean(residual))}


def _refine_quad(gray: np.ndarray, edge_xy: np.ndarray, gradients: tuple,
                 guide: np.ndarray, band: float, polarity: int = 0) -> tuple[np.ndarray | None, list[dict]]:
    """四条图像边线求交；任何一条缺失都不补上预测边。"""
    lines, evidence = [], []
    for i in range(4):
        line, detail = _fit_boundary(gray, edge_xy, gradients, guide[i], guide[(i + 1) % 4], band, polarity=polarity)
        evidence.append(detail)
        if line is None:
            return None, evidence
        lines.append(line)
    corners = [_intersect(lines[(i - 1) % 4], lines[i]) for i in range(4)]
    if any(p is None for p in corners):
        return None, evidence
    quad = np.asarray(corners)
    if not cv2.isContourConvex(quad.astype(np.float32)):
        return None, evidence
    return quad, evidence


def _quad_candidates(edges: np.ndarray, expected: np.ndarray, max_shift: float) -> list[tuple]:
    """由闭合 Canny 轮廓产生四边形；同一条边的双轮廓合并去重。"""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    expected_area = abs(cv2.contourArea(expected.astype(np.float32)))
    expected_sides = np.linalg.norm(np.roll(expected, -1, axis=0) - expected, axis=1)
    candidates = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 30:
            continue
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = abs(cv2.contourArea(approx))
        if not 0.45 * expected_area <= area <= 1.85 * expected_area:
            continue
        quad, distance = _ordered_quad(approx.reshape(4, 2), expected)
        if np.max(np.linalg.norm(quad - expected, axis=1)) > max_shift:
            continue
        sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        relative_error = np.abs(sides / expected_sides - 1.0)
        if np.max(relative_error) > 0.45:
            continue
        score = distance + 18.0 * float(np.mean(relative_error))
        candidates.append((score, quad))
    candidates.sort(key=lambda item: item[0])
    unique = []
    for score, quad in candidates:
        if not any(np.mean(np.linalg.norm(quad - old[1], axis=1)) < 2.0 for old in unique):
            unique.append((score, quad))
    return unique


def detect_features(image: np.ndarray, landmarks: dict[str, np.ndarray], nominal: np.ndarray,
                    K: np.ndarray, dist: np.ndarray, config: dict) -> DetectionResult:
    """识别矩形插口，返回图像观测和可审查的失败原因。

    输入 nominal 是名义 T_camera_port，内部坐标单位米。该函数没有接收
    真实位姿、深度图、物体标签的入口；名义角点永远不会直接返回给 PnP。
    top_rear 可选，缺少可辨顶面时仍允许四角平面解算。
    """
    settings = config.get("detection", {})
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0.65)
    edges = cv2.Canny(gray, int(settings.get("canny_low", 20)), int(settings.get("canny_high", 50)))
    diagnostics: dict = {"method": "Canny + image-line intersections", "top_rear": "未检测"}

    def fail(reason: str) -> DetectionResult:
        diagnostics["failure"] = reason
        return DetectionResult({}, diagnostics, edges, False, reason)

    camera_center = -nominal[:3, :3].T @ nominal[:3, 3]
    front_depths = landmarks["outer"] @ nominal[2, :3] + nominal[2, 3]
    if np.any(front_depths <= 0):
        return fail("端口位于相机后方或跨过相机平面，无法形成有效正面图像。")
    if camera_center[2] >= 0:
        return fail("相机位于端口背面，无法观察正面开口。")
    if -camera_center[2] / np.linalg.norm(camera_center) < 0.15:
        return fail("拍摄角度接近侧面，开口过度压缩，无法可靠关联角点。")
    projected = {name: project_points(points, nominal, K, dist) for name, points in landmarks.items()}
    front = projected["outer"]
    expected_inner = projected["inner"]
    width = float(np.mean(np.linalg.norm(front[[1, 2]] - front[[0, 3]], axis=1)))
    height = float(np.mean(np.linalg.norm(front[[3, 2]] - front[[0, 1]], axis=1)))
    diagnostics["nominal_port_size_px"] = [width, height]
    if width < float(settings.get("min_port_width_px", 40)) or height < float(settings.get("min_port_height_px", 12)):
        return fail("端口在图像中像素不足；请缩短拍摄距离或提高焦距/分辨率。")
    h, w = gray.shape
    if np.any(front < 3) or np.any(front[:, 0] >= w - 3) or np.any(front[:, 1] >= h - 3):
        return fail("端口超出画面或贴近边缘；请调整相机方向或距离。")
    margin = float(settings.get("roi_margin_px", 35))
    all_expected = np.vstack(list(projected.values()))
    low = np.maximum(np.floor(all_expected.min(axis=0) - margin).astype(int), 0)
    high = np.minimum(np.ceil(all_expected.max(axis=0) + margin).astype(int), [w, h])
    roi = np.zeros_like(edges)
    roi[low[1]:high[1], low[0]:high[0]] = edges[low[1]:high[1], low[0]:high[0]]
    diagnostics["roi_xyxy"] = [int(low[0]), int(low[1]), int(high[0]), int(high[1])]
    y, x = np.nonzero(roi)
    edge_xy = np.column_stack((x, y)).astype(np.float64)
    gradients = (cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    max_shift = float(settings.get("max_corner_shift_px", 25))
    candidates = _quad_candidates(roi, expected_inner, max_shift)
    diagnostics["inner_quad_candidates"] = len(candidates)
    if not candidates:
        return fail("搜索区域内没有清晰、闭合的矩形内口；可能有遮挡或边缘对比度不足。")

    # 验证内外双矩形的几何关系。单独一条矩形轮廓可能只是孔内的深处边缘。
    valid = []
    for score, initial_inner in candidates[:8]:
        inner, inner_detail = _refine_quad(gray, edge_xy, gradients, initial_inner, 2.6, polarity=-1)
        if inner is None:
            continue
        homography = cv2.getPerspectiveTransform(landmarks["inner"][:, :2].astype(np.float32),
                                                inner.astype(np.float32))
        guide = cv2.perspectiveTransform(landmarks["outer"][:, :2].astype(np.float32)[None], homography)[0]
        # 内外环相距通常仅数像素，搜索带不能跨越整个环宽。
        ring = float(np.min(np.linalg.norm(guide - inner, axis=1)))
        outer, outer_detail = _refine_quad(gray, edge_xy, gradients, guide, min(2.6, ring * 0.38))
        if outer is None:
            continue
        if not all(cv2.pointPolygonTest(outer.astype(np.float32), tuple(p), True) > 1.0 for p in inner):
            continue
        outer_shift = float(np.max(np.linalg.norm(outer - front, axis=1)))
        if outer_shift > max_shift:
            continue
        geometry_error = float(np.mean(np.linalg.norm(outer - guide, axis=1)))
        valid.append((score + geometry_error * 2, inner, outer, inner_detail, outer_detail))
    if not valid:
        return fail("检测到矩形候选，但内外口边缘不完整或尺寸关系不一致；拒绝输出不可靠角点。")
    valid.sort(key=lambda item: item[0])
    if len(valid) > 1 and valid[1][0] - valid[0][0] < 1.0:
        different = np.mean(np.linalg.norm(valid[0][1] - valid[1][1], axis=1))
        if different > 2.0:
            return fail("多个角点关联具有相近得分，无法唯一判断端口边界。")
    _, inner, outer, inner_detail, outer_detail = valid[0]
    if np.any(outer < 2) or np.any(outer[:, 0] >= w - 2) or np.any(outer[:, 1] >= h - 2):
        return fail("实际检测端口超出画面，角点不完整。")
    points = {"inner": inner, "outer": outer}
    diagnostics["inner_lines"] = inner_detail
    diagnostics["outer_lines"] = outer_detail
    diagnostics["mean_nominal_corner_shift_px"] = float(np.mean(np.linalg.norm(outer - front, axis=1)))

    # 顶面后角点不共面，能缓解正面小矩形的位姿歧义。只在三条边均有图像证据时采用。
    if "top_rear" in landmarks and len(landmarks["top_rear"]) == 2:
        top_y = float(landmarks["outer"][0, 1])
        rear = projected["top_rear"] + np.mean(outer - front, axis=0)
        top_guide = np.array([outer[0], outer[1], rear[1], rear[0]])
        top_area = abs(cv2.contourArea(top_guide.astype(np.float32)))
        if camera_center[1] >= top_y or top_area < width * 6:
            diagnostics["top_rear"] = "顶面不可见或投影过薄，仅提供平面角点。"
        elif np.any(rear < 4) or np.any(rear[:, 0] >= w - 4) or np.any(rear[:, 1] >= h - 4):
            diagnostics["top_rear"] = "顶面后角点出画，仅提供平面角点。"
        else:
            band = float(settings.get("search_band_px", 12))
            lines, detail = [], []
            for start, end in ((outer[0], rear[0]), (outer[1], rear[1]), (rear[0], rear[1])):
                line, evidence = _fit_boundary(gray, edge_xy, gradients, start, end, band, 0.55)
                lines.append(line)
                detail.append(evidence)
            diagnostics["top_lines"] = detail
            if all(line is not None for line in lines):
                left, right = _intersect(lines[0], lines[2]), _intersect(lines[1], lines[2])
                if left is not None and right is not None:
                    observed = np.array([left, right])
                    # 两条侧边应通过已观察到的前端外角，防止关联到附近底板或螺钉。
                    start_error = [abs(lines[i] @ np.r_[outer[i], 1.0]) for i in range(2)]
                    if max(start_error) < 2.2 and np.max(np.linalg.norm(observed - rear, axis=1)) < max_shift:
                        points["top_rear"] = observed
                        diagnostics["top_rear"] = "已由三条图像边线交点确认。"
            if "top_rear" not in points:
                diagnostics["top_rear"] = "顶面边缘证据不足，仅提供平面角点。"
    return DetectionResult(points, diagnostics, edges, True, "检测成功")
