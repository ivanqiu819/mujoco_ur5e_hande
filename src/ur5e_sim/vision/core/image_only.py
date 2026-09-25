"""不依赖名义机位的全图矩形端口定位。

数据流：图像四边形 → 内外口关联 → IPPE 假设 → 图像顶边验证 → PnP。
本模块没有相机外参或仿真配置的输入；所有用于引导搜索的位姿都由
当前图片计算。缺乏区分证据时保留多个候选，不猜测哪一解是真值。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .camera import project_points
from .detection import DetectionResult, _fit_boundary, _intersect, _refine_quad
from .pose import _estimate_nonplanar, _estimate_planar, evaluate_pose


def _quad_distance(first: np.ndarray, second: np.ndarray) -> float:
    """比较无语义的四边形，忽略起点及轮廓遍历方向。"""
    return min(float(np.mean(np.linalg.norm(first - np.roll(q, i, axis=0), axis=1)))
               for q in (second, second[::-1]) for i in range(4))


def _image_quads(edges: np.ndarray, options: dict[str, Any]) -> list[np.ndarray]:
    """从整幅边缘图提出候选；过滤仅依赖实际像素，不预测位置或大小。"""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    minimum_area = options.get("min_port_width_px", 40) * options.get("min_port_height_px", 12) * 0.3
    quads = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        if abs(cv2.contourArea(polygon)) < minimum_area:
            continue
        quad = polygon.reshape(4, 2).astype(np.float64)
        # 固定为图像中的顺时针，便于内口梯度方向检查；尚未指定 CAD 角点名。
        if cv2.contourArea(quad.astype(np.float32), oriented=True) < 0:
            quad = quad[::-1]
        if not any(_quad_distance(quad, old) < 2.0 for old in quads):
            quads.append(quad)
    return quads


def _associations(gray: np.ndarray, edge_xy: np.ndarray, gradients: tuple,
                  quad: np.ndarray, landmarks: dict[str, np.ndarray],
                  options: dict[str, Any]) -> list[dict[str, Any]]:
    """精修内口，并尝试八种模型角点对应；外框必须有独立图像证据。"""
    inner, inner_lines = _refine_quad(gray, edge_xy, gradients, quad, 2.6, polarity=-1)
    if inner is None:
        return []
    height, width = gray.shape
    associations = []
    for reverse in (False, True):
        base = inner[::-1] if reverse else inner
        for shift in range(4):
            ordered = np.roll(base, shift, axis=0)
            H = cv2.getPerspectiveTransform(landmarks["inner"][:, :2].astype(np.float32),
                                            ordered.astype(np.float32))
            guide = cv2.perspectiveTransform(landmarks["outer"][:, :2].astype(np.float32)[None], H)[0]
            if not np.isfinite(guide).all() or np.any(guide < 2) or np.any(guide >= [width - 2, height - 2]):
                continue
            gap = float(np.min(np.linalg.norm(guide - ordered, axis=1)))
            if gap < 1.5:
                continue
            outer, outer_lines = _refine_quad(gray, edge_xy, gradients, guide, min(2.6, gap * 0.38))
            if outer is None:
                continue
            if np.any(outer < 2) or np.any(outer >= [width - 2, height - 2]):
                continue
            if not all(cv2.pointPolygonTest(outer.astype(np.float32), tuple(p), True) > 1 for p in ordered):
                continue
            sides = np.linalg.norm(np.roll(outer, -1, axis=0) - outer, axis=1)
            # 宽高按 CAD 对应计算，因此画面 roll 旋转不会改变这两个门限的意义。
            if np.mean(sides[[0, 2]]) < options.get("min_port_width_px", 40):
                continue
            if np.mean(sides[[1, 3]]) < options.get("min_port_height_px", 12):
                continue
            line_details = inner_lines + outer_lines
            associations.append({
                "points": {"inner": ordered, "outer": outer},
                "correspondence": {"reversed": reverse, "cyclic_shift": shift},
                "geometry_error_px": float(np.mean(np.linalg.norm(outer - guide, axis=1))),
                "line_residual_px": float(np.mean([line["mean_residual_px"] for line in line_details])),
                "edge_coverage": float(np.mean([line["coverage"] for line in line_details])),
            })
    return associations


def _top_observations(gray: np.ndarray, edge_xy: np.ndarray, gradients: tuple,
                      points: dict[str, np.ndarray], landmarks: dict[str, np.ndarray],
                      hypothesis: np.ndarray, K: np.ndarray, dist: np.ndarray,
                      options: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    """用图像产生的 IPPE 假设寻找顶边，绝不把预测角点作为观测返回。"""
    rear_model = landmarks.get("top_rear", np.empty((0, 3)))
    if len(rear_model) != 2:
        return None, {"reason": "未配置顶面后侧角点"}
    camera_center = -hypothesis[:3, :3].T @ hypothesis[:3, 3]
    if camera_center[1] >= landmarks["outer"][0, 1]:
        return None, {"reason": "该图像假设下顶面不可见"}
    if np.any(rear_model @ hypothesis[2, :3] + hypothesis[2, 3] <= 0):
        return None, {"reason": "该假设的顶面位于相机后方"}
    rear = project_points(rear_model, hypothesis, K, dist)
    outer = points["outer"]
    height, width = gray.shape
    if not np.isfinite(rear).all() or np.any(rear < 4) or np.any(rear >= [width - 4, height - 4]):
        return None, {"reason": "该假设的顶面后角点出画"}
    top = np.array([outer[0], outer[1], rear[1], rear[0]], dtype=np.float32)
    if abs(cv2.contourArea(top)) < np.linalg.norm(outer[1] - outer[0]) * 6:
        return None, {"reason": "该假设的顶面投影过薄"}
    band = float(options.get("search_band_px", 12))
    lines, evidence = [], []
    for start, end in ((outer[0], rear[0]), (outer[1], rear[1]), (rear[0], rear[1])):
        line, detail = _fit_boundary(gray, edge_xy, gradients, start, end, band, 0.55)
        lines.append(line)
        evidence.append(detail)
    if any(line is None for line in lines):
        return None, {"reason": "顶面三条边的图像证据不足", "lines": evidence}
    left, right = _intersect(lines[0], lines[2]), _intersect(lines[1], lines[2])
    if left is None or right is None:
        return None, {"reason": "顶面边线近乎平行", "lines": evidence}
    observed = np.array([left, right])
    if not np.isfinite(observed).all() or np.any(observed < 2) or np.any(observed >= [width - 2, height - 2]):
        return None, {"reason": "观测到的顶面交点出画", "lines": evidence}
    front_error = max(abs(lines[i] @ np.r_[outer[i], 1.0]) for i in range(2))
    if front_error > 2.2 or np.max(np.linalg.norm(observed - rear, axis=1)) > 2 * band:
        return None, {"reason": "顶面边线不能与已观测的前框连接", "lines": evidence}
    return observed, {"reason": "三条真实边线交点支持该假设", "lines": evidence}


def _append_unique(candidates: list[dict[str, Any]], candidate: dict[str, Any]) -> None:
    """合并不同 IPPE 初值精修后得到的同一解；对称的不同朝向仍然保留。"""
    for index, old in enumerate(candidates):
        if old["target_id"] != candidate["target_id"] or old["method"] != candidate["method"]:
            continue
        difference = evaluate_pose(candidate["T_camera_port"], old["T_camera_port"])
        if difference["rotation_error_deg"] < 0.1 and difference["translation_error_mm"] < 0.1:
            if candidate["score_px"] < old["score_px"]:
                candidates[index] = candidate
            return
    candidates.append(candidate)


def _choose_candidates(candidates: list[dict[str, Any]], ambiguity_margin: float) -> list[int]:
    """每个目标先比较证据类型，再比较像素评分；多个目标不擅自选一个。"""
    active = []
    for target_id in sorted({c["target_id"] for c in candidates}):
        indices = [i for i, c in enumerate(candidates) if c["target_id"] == target_id]
        with_depth = [i for i in indices if candidates[i]["method"] == "nonplanar_pnp"]
        eligible = with_depth or indices
        best_score = min(candidates[i]["score_px"] for i in eligible)
        for i in indices:
            candidate = candidates[i]
            candidate["active"] = i in eligible and candidate["score_px"] <= best_score + ambiguity_margin
            candidate["selection_reason"] = ("有竞争力的图像候选" if candidate["active"] else
                "同一目标有非共面图像证据" if i not in eligible else "图像评分落后于同类候选")
            if candidate["active"]:
                active.append(i)
    return active


def estimate_from_image(image: np.ndarray, landmarks: dict[str, np.ndarray],
                         K: np.ndarray, dist: np.ndarray,
                         detection_options: dict[str, Any], pose_options: dict[str, Any]
                         ) -> tuple[DetectionResult, dict[str, Any]]:
    """全图检测并估计位姿；只允许图像、模型、相机内参和算法参数进入。

    返回现有 DetectionResult 和 JSON 可序列化结果。ambiguous 时最终位姿
    和 selected_index 为空；所有假设及其观测保存在 candidates 中。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0.65)
    edges = cv2.Canny(gray, int(detection_options.get("canny_low", 20)), int(detection_options.get("canny_high", 50)))
    y, x = np.nonzero(edges)
    edge_xy = np.column_stack([x, y]).astype(np.float64)
    gradients = (cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    quads = _image_quads(edges, detection_options)
    targets: list[np.ndarray] = []
    candidates: list[dict[str, Any]] = []
    threshold = float(pose_options.get("reprojection_threshold_px", 3))
    front_model = np.vstack([landmarks["inner"], landmarks["outer"]])
    for quad in quads:
        for association in _associations(gray, edge_xy, gradients, quad, landmarks, detection_options):
            points = association["points"]
            front_pixels = np.vstack([points["inner"], points["outer"]])
            planar = _estimate_planar(front_model, front_pixels, K, dist, None, pose_options)
            if planar["status"] != "ok":
                continue
            target_id = next((i for i, q in enumerate(targets) if _quad_distance(points["inner"], q) < 2.5), len(targets))
            if target_id == len(targets):
                targets.append(points["inner"])
            # 所有项均来自图像，单位为像素；不含位姿接近先验的惩罚。
            edge_penalty = (0.25 * association["geometry_error_px"] + 0.25 * association["line_residual_px"]
                            + (1 - association["edge_coverage"]))
            for seed in planar["candidates"]:
                if seed["reprojection_rmse_px"] > threshold:
                    continue
                transform = np.asarray(seed["T_camera_port"])
                rear, top_evidence = _top_observations(gray, edge_xy, gradients, points, landmarks,
                                                       transform, K, dist, detection_options)
                base = {
                    "target_id": target_id, "method": "planar_ippe", "T_camera_port": seed["T_camera_port"],
                    "reprojection_rmse_px": seed["reprojection_rmse_px"],
                    "score_px": seed["reprojection_rmse_px"] + edge_penalty,
                    "points_px": {name: value.tolist() for name, value in points.items()},
                    "correspondence": association["correspondence"],
                    "geometry_error_px": association["geometry_error_px"],
                    "edge_coverage": association["edge_coverage"], "top_evidence": top_evidence,
                }
                _append_unique(candidates, base)
                if rear is None:
                    continue
                all_model = np.vstack([front_model, landmarks["top_rear"]])
                all_pixels = np.vstack([front_pixels, rear])
                nonplanar = _estimate_nonplanar(all_model, all_pixels, K, dist, pose_options)
                if nonplanar["status"] != "ok" or not {8, 9}.issubset(nonplanar.get("inlier_indices", [])):
                    continue
                refined = {**base, **nonplanar, "method": "nonplanar_pnp",
                           "points_px": {**base["points_px"], "top_rear": rear.tolist()},
                           "score_px": nonplanar["reprojection_rmse_px"] + edge_penalty}
                _append_unique(candidates, refined)

    candidates.sort(key=lambda c: (c["target_id"], c["score_px"], c["method"]))
    for index, candidate in enumerate(candidates):
        candidate["candidate_id"] = index
    active = _choose_candidates(candidates, float(pose_options.get("ambiguity_margin_px", 0.25)))
    result: dict[str, Any] = {
        "status": "failed", "method": None, "T_camera_port": None, "selected_index": None,
        "use_prior": False, "candidates": candidates, "active_candidate_indices": active, "warnings": [],
    }
    diagnostics: dict[str, Any] = {
        "method": "Full-image contours + IPPE hypotheses + image-edge verification", "use_prior": False,
        "search_scope": "full_image", "quad_count": len(quads), "target_count": len(targets),
        "targets": [{"target_id": i, "inner_px": quad.tolist()} for i, quad in enumerate(targets)],
    }
    detected_points: dict[str, np.ndarray] = {}
    if not active:
        reason = "全图未找到具有完整内外边缘和有效位姿的矩形端口；请检查遮挡、像素大小和边缘对比度。"
        result["warnings"].append(reason)
    elif len(active) > 1:
        reason = "存在多个图像支持的目标或位姿；缺少区分证据，保留候选而不指定唯一位姿。"
        result.update(status="ambiguous", warnings=[reason])
    else:
        selected = candidates[active[0]]
        status = "ok" if selected["method"] == "nonplanar_pnp" else "planar_only"
        result.update(status=status, method=selected["method"], T_camera_port=selected["T_camera_port"], selected_index=active[0])
        detected_points = {name: np.asarray(value) for name, value in selected["points_px"].items()}
        reason = "全图视觉定位成功"
        if status == "planar_only":
            result["warnings"].append("只有平面图像约束，请检查保存的其他候选和评分差距。")
    diagnostics["result_status"] = result["status"]
    return DetectionResult(detected_points, diagnostics, edges, bool(active), reason), result
