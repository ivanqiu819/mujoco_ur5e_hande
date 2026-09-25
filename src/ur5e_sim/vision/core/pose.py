"""从已关联的图像角点求位姿；该模块不接触任何仿真真值。

前框角点共面，IPPE 通常给出两组候选；保留候选以展示单目平面歧义，
按重投影误差和名义机位先验排序。顶面后侧角点可见时，再用非共面
PnP/RANSAC 求解，且 LM 精修严格只使用 RANSAC 的内点。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .camera import pose_vectors, project_points, transform_from_vectors
from .geometry import transform_points


def estimate_marker_pose(observations: list[np.ndarray], size_m: float,
                         K: np.ndarray, dist: np.ndarray, T_port_marker: np.ndarray,
                         options: dict[str, Any]) -> dict[str, Any]:
    """用单个方形标记的四角估计位姿，再通过已知安装关系换算到端口。

    observations 可含多个同 ID 目标，四角顺序必须来自 ArUco 解码。
    每个目标运行 IPPE_SQUARE；四点不足以做可靠的外点筛选，不使用 RANSAC。
    本函数不接收相机名义机位或真值。标记 +Z 朝外，与端口 +Z 朝内不同。
    """
    from .aruco import marker_corners
    model = marker_corners(size_m)
    T_marker_port = np.linalg.inv(T_port_marker)
    threshold = float(options.get("reprojection_threshold_px", 3))
    candidates: list[dict[str, Any]] = []
    diagnostics = []

    def valid_marker_pose(transform: np.ndarray) -> bool:
        if not np.isfinite(transform).all() or np.any(transform_points(model, transform)[:, 2] <= 1e-6):
            return False
        camera_in_marker = -transform[:3, :3].T @ transform[:3, 3]
        return bool(camera_in_marker[2] > 1e-6)

    for target, observation in enumerate(observations):
        pixels = np.ascontiguousarray(observation, dtype=np.float64)
        if pixels.shape != (4, 2) or not np.isfinite(pixels).all():
            diagnostics.append({"target_id": target, "reason": "标记角点必须为有限的 4×2 数组"})
            continue
        if not cv2.isContourConvex(pixels.astype(np.float32)) or abs(cv2.contourArea(pixels.astype(np.float32))) < 1:
            diagnostics.append({"target_id": target, "reason": "标记角点退化或不构成凸四边形"})
            continue
        try:
            count, rvecs, tvecs, _ = cv2.solvePnPGeneric(model, pixels, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error as exc:
            diagnostics.append({"target_id": target, "reason": f"IPPE_SQUARE 失败：{exc}"})
            continue
        if not count:
            diagnostics.append({"target_id": target, "reason": "IPPE_SQUARE 未返回位姿"})
        for seed_index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            transform = transform_from_vectors(rvec, tvec)
            detail: dict[str, Any] = {"target_id": target, "seed_index": seed_index}
            if not valid_marker_pose(transform):
                diagnostics.append({**detail, "reason": "候选违反正深度或标记正面朝向约束"})
                continue
            rmse = reprojection_rmse(model, pixels, transform, K, dist)
            detail["initial_reprojection_rmse_px"] = rmse
            # 分别精修两个初值；若收敛到同一解则合并，否则保留真实平面歧义。
            try:
                refined_r, refined_t = cv2.solvePnPRefineLM(model, pixels, K, dist, rvec.copy(), tvec.copy())
                refined = transform_from_vectors(refined_r, refined_t)
                refined_error = reprojection_rmse(model, pixels, refined, K, dist)
                if valid_marker_pose(refined) and np.isfinite(refined_error) and refined_error <= rmse:
                    transform, rmse = refined, refined_error
                    detail["refined"] = True
            except cv2.error:
                detail["refined"] = False  # 精修失败可保留仍满足图像门限的 IPPE 初值。
            if not np.isfinite(rmse) or rmse > threshold:
                diagnostics.append({**detail, "reason": "候选重投影误差超过门限"})
                continue
            port_pose = transform @ T_marker_port
            candidate = {
                "target_id": target, "method": "aruco_ippe_square", "seed_index": seed_index,
                "T_camera_marker": transform.tolist(), "T_camera_port": port_pose.tolist(),
                "T_port_camera": np.linalg.inv(port_pose).tolist(),
                "reprojection_rmse_px": rmse, "score_px": rmse,
                "points_px": {"marker": pixels.tolist()}, "refinement": detail,
            }
            duplicate = None
            for index, old in enumerate(candidates):
                if old["target_id"] != target:
                    continue
                difference = evaluate_pose(transform, old["T_camera_marker"])
                if difference["rotation_error_deg"] < 0.1 and difference["translation_error_mm"] < 0.1:
                    duplicate = index
                    break
            if duplicate is None:
                candidates.append(candidate)
            elif rmse < candidates[duplicate]["score_px"]:
                candidates[duplicate] = candidate

    candidates.sort(key=lambda c: (c["target_id"], c["score_px"]))
    margin = float(options.get("ambiguity_margin_px", 0.25))
    active = []
    for index, candidate in enumerate(candidates):
        best = min(c["score_px"] for c in candidates if c["target_id"] == candidate["target_id"])
        candidate.update(candidate_id=index, active=candidate["score_px"] <= best + margin)
        candidate["selection_reason"] = "保留的图像候选" if candidate["active"] else "重投影误差落后于同目标候选"
        if candidate["active"]:
            active.append(index)
    # 多个同 ID 的可用观测意味着身份无法唯一确定，即使某个目标 PnP 失败
    # 也不能擅自认为另一个就是已标定的工件。
    ambiguous = bool(active) and (len(active) > 1 or len(observations) > 1)
    result: dict[str, Any] = {
        "status": "ambiguous" if ambiguous else "ok" if active else "failed",
        "route": "aruco", "use_prior": False, "method": None, "selected_index": None,
        "T_camera_marker": None, "T_camera_port": None, "T_port_camera": None,
        "T_port_marker": np.asarray(T_port_marker).tolist(),
        "marker_size_m": size_m, "candidates": candidates, "active_candidate_indices": active,
        "solver_diagnostics": diagnostics, "warnings": [],
    }
    if result["status"] == "ok":
        selected = candidates[active[0]]
        result.update(selected_index=active[0], method="aruco_ippe_square")
        for key in ("T_camera_marker", "T_camera_port", "T_port_camera"):
            result[key] = selected[key]
    elif ambiguous:
        result["warnings"].append("存在重复目标 ID 或图像评分接近的平面位姿；保留候选，不指定唯一端口位姿。")
    else:
        result["warnings"].append("没有满足正深度、标记正面朝向和重投影要求的 ArUco 位姿。")
    return result


def evaluate_pose(estimated: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """比较两个 T_camera_port；旋转误差为 SO(3) 夹角，平移误差为毫米。

    可用于名义先验距离，也可由上层报告模块调用以比较仿真真值。
    estimate_pose 本身仅用此函数比较名义机位，不接收真实位姿。
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    relative = estimated[:3, :3] @ reference[:3, :3].T
    angle = np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    distance = np.linalg.norm(estimated[:3, 3] - reference[:3, 3]) * 1000
    return {"rotation_error_deg": float(angle), "translation_error_mm": float(distance)}


def reprojection_rmse(points: np.ndarray, pixels: np.ndarray, transform: np.ndarray,
                      K: np.ndarray, dist: np.ndarray) -> float:
    """每个点的二维欧氏残差的均方根（像素）。"""
    residuals = project_points(points, transform, K, dist) - pixels
    return float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))


def _pose_is_valid(transform: np.ndarray, points: np.ndarray) -> bool:
    if not np.isfinite(transform).all():
        return False
    if np.any(transform_points(points, transform)[:, 2] <= 1e-6):
        return False
    # 开口朝外方向是局部 -Z；相机必须位于开口前侧。
    camera_in_port = -transform[:3, :3].T @ transform[:3, 3]
    return bool(camera_in_port[2] < -1e-6)


def _stack_matches(landmarks: dict[str, np.ndarray], detected: dict[str, np.ndarray],
                   names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    points, pixels = [], []
    for name in names:
        if name not in detected or name not in landmarks:
            continue
        object_points = np.asarray(landmarks[name], dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(detected[name], dtype=np.float64)
        if not np.isfinite(object_points).all():
            raise ValueError(f"{name} 模型特征点包含非有限数值")
        if not len(object_points) and image_points.size == 0:
            continue
        if image_points.shape != (len(object_points), 2) or not np.isfinite(image_points).all():
            raise ValueError(f"{name} 图像角点应为 {len(object_points)}×2 有限数值，且按模型语义排序")
        if len(object_points):
            points.append(object_points)
            pixels.append(image_points)
    return (np.vstack(points), np.vstack(pixels)) if points else (np.empty((0, 3)), np.empty((0, 2)))


def _estimate_planar(points: np.ndarray, pixels: np.ndarray, K: np.ndarray, dist: np.ndarray,
                     nominal: np.ndarray | None, options: dict[str, Any]) -> dict[str, Any]:
    """生成 IPPE 候选；nominal=None 时只按图像误差评分，不指定选中解。"""
    result: dict[str, Any] = {"status": "failed", "candidates": [], "selected_index": None}
    if len(points) < 4:
        result["reason"] = "前表面有效角点不足四个，无法执行平面 IPPE"
        return result
    if (np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-8) != 2
            or np.linalg.matrix_rank(pixels - pixels.mean(axis=0), tol=1e-6) != 2):
        result["reason"] = "平面角点退化或不共面；IPPE 需要非共线的共面模型点和图像点"
        return result
    try:
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(points, pixels, K, dist, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error as exc:
        result["reason"] = f"平面 IPPE 求解失败：{exc}"
        return result
    if not count:
        result["reason"] = "平面 IPPE 没有返回候选解"
        return result
    for rvec, tvec in zip(rvecs, tvecs):
        transform = transform_from_vectors(rvec, tvec)
        if not _pose_is_valid(transform, points):
            continue
        rmse = reprojection_rmse(points, pixels, transform, K, dist)
        candidate = {
            "T_camera_port": transform.tolist(), "reprojection_rmse_px": rmse,
            "score": rmse,
        }
        if nominal is not None:
            prior = evaluate_pose(transform, nominal)
            candidate.update(prior_rotation_deg=prior["rotation_error_deg"],
                             prior_translation_mm=prior["translation_error_mm"])
            candidate["score"] += (options.get("prior_rotation_weight", 0.025) * prior["rotation_error_deg"]
                                   + options.get("prior_translation_weight", 0.004) * prior["translation_error_mm"])
        result["candidates"].append(candidate)
    if not result["candidates"]:
        result["reason"] = "IPPE 候选均违反正深度或开口朝向约束"
        return result
    result["candidates"].sort(key=lambda candidate: candidate["score"])
    result["selected_index"] = 0 if nominal is not None else None
    if result["candidates"][0]["reprojection_rmse_px"] > options.get("reprojection_threshold_px", 3.0):
        result["reason"] = "最佳平面候选重投影误差过大，请检查角点关联或遮挡"
        return result
    result["status"] = "ok"
    return result


def _estimate_nonplanar(points: np.ndarray, pixels: np.ndarray, K: np.ndarray, dist: np.ndarray,
                        options: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "failed", "point_count": len(points)}
    if len(points) < 6 or np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-8) < 3:
        result["reason"] = "非共面求解需要至少六个点，且必须含有效顶面后侧角点"
        return result
    threshold = float(options.get("reprojection_threshold_px", 3.0))
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            points, pixels, K, dist, iterationsCount=int(options.get("ransac_iterations", 500)),
            reprojectionError=threshold, confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 6:
            result["reason"] = "PnP RANSAC 没有找到至少六个一致内点"
            return result
        indices = inliers.reshape(-1)
        result.update(inlier_indices=indices.tolist(), inlier_count=len(indices))
        inlier_points = np.ascontiguousarray(points[indices])
        inlier_pixels = np.ascontiguousarray(pixels[indices])
        if np.linalg.matrix_rank(inlier_points - inlier_points.mean(axis=0), tol=1e-8) < 3:
            result["reason"] = "RANSAC 保留的内点全部共面，不能声称获得非共面约束"
            return result
        # 不要把 points/pixels 全集传给 LM：外点一旦回来就会再次拉偏位姿。
        rvec, tvec = cv2.solvePnPRefineLM(inlier_points, inlier_pixels, K, dist, rvec, tvec)
        transform = transform_from_vectors(rvec, tvec)
        if not _pose_is_valid(transform, points):
            result["reason"] = "非共面位姿违反正深度或开口朝向约束"
            return result
        rmse = reprojection_rmse(inlier_points, inlier_pixels, transform, K, dist)
        if not np.isfinite(rmse) or rmse > threshold:
            result["reason"] = "非共面位姿精修后的内点重投影误差过大"
            return result
        result.update(status="ok", T_camera_port=transform.tolist(), reprojection_rmse_px=rmse)
    except cv2.error as exc:
        result["reason"] = f"非共面 PnP 求解失败：{exc}"
    return result


def estimate_pose(landmarks: dict[str, np.ndarray], detected: dict[str, np.ndarray],
                  K: np.ndarray, dist: np.ndarray, nominal: np.ndarray,
                  config: dict[str, Any]) -> dict[str, Any]:
    """用检测角点估计端口位姿，返回可以直接写入 JSON 的结果。

    detected 的 inner/outer 顺序为左上、右上、右下、左下，top_rear
    为左后、右后。遮挡或不可辨的整组特征应不提供，不能用名义投影补齐。
    status=ok 表示非共面解成功，planar_only 表示只获得平面候选，failed
    表示没有可用位姿。T_camera_port 的平移始终以米表示。
    """
    options = config.get("pose", {})
    try:
        planar_points, planar_pixels = _stack_matches(landmarks, detected, ("inner", "outer"))
    except (TypeError, ValueError) as exc:
        reason = f"角点输入无效：{exc}"
        return {"status": "failed", "method": None, "T_camera_port": None,
                "planar": {"status": "failed", "candidates": [], "selected_index": None, "reason": reason},
                "nonplanar": {"status": "unavailable", "reason": reason}, "warnings": [reason]}
    planar = _estimate_planar(planar_points, planar_pixels, K, dist, nominal, options)
    result: dict[str, Any] = {
        "status": "failed", "method": None, "T_camera_port": None,
        "planar": planar, "nonplanar": {"status": "unavailable", "reason": "顶面后侧角点不可辨或未配置"},
        "warnings": [],
    }
    if "top_rear" in detected and len(landmarks.get("top_rear", [])):
        try:
            points, pixels = _stack_matches(landmarks, detected, ("inner", "outer", "top_rear"))
            result["nonplanar"] = _estimate_nonplanar(points, pixels, K, dist, options)
        except (TypeError, ValueError) as exc:
            result["nonplanar"] = {"status": "failed", "reason": f"非共面角点输入无效：{exc}"}
    if result["nonplanar"]["status"] == "ok":
        result.update(status="ok", method="nonplanar_pnp", T_camera_port=result["nonplanar"]["T_camera_port"])
    elif planar["status"] == "ok":
        selected = planar["candidates"][planar["selected_index"]]
        result.update(status="planar_only", method="planar_ippe", T_camera_port=selected["T_camera_port"])
        result["warnings"].append(result["nonplanar"]["reason"])
        result["warnings"].append("仅有共面特征；单目平面存在姿态歧义，请同时查看 IPPE 候选及名义先验")
    else:
        result["warnings"].append(planar.get("reason", "没有有效平面候选"))
        result["warnings"].append(result["nonplanar"]["reason"])
    return result
