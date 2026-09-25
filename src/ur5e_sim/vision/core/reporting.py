"""结果保存与可视化；真值仅在这里用于对比，不参与检测或求解。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np
import yaml

from .camera import camera_parameters, nominal_pose, pose_vectors, project_points
from .aruco import MarkerDetection, marker_corners
from .detection import DetectionResult
from .geometry import Geometry, transform_points
from .pose import evaluate_pose
from .renderer import SurfacePatch, render_scene


def save_image(path: Path, image: np.ndarray) -> None:
    """检查编码与写入，兼容包含中文的文件路径。"""
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"图片编码失败：{path}")
    path.write_bytes(encoded.tobytes())


def _caption(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 16 + 27 * len(lines)), (245, 245, 245), -1)
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (14, 27 + index * 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def save_model_preview(path: Path, geometry: Geometry, config: dict[str, Any], *,
                       surface_patches: tuple[SurfacePatch, ...] = ()) -> None:
    """生成拍全装配体的预览，不改变用户设置的测量相机。"""
    preview_config = deepcopy(config)
    lo, hi = geometry.vertices.min(axis=0), geometry.vertices.max(axis=0)
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2)
    # 包围球放入较小视场角，保证整个模型在图内。
    distance = radius / np.sin(np.arctan(350 / 900)) * 1.15
    preview_config["camera"].update(
        distance_mm=distance * 1000, azimuth_deg=25, elevation_deg=27, roll_deg=0,
        target_offset_mm=(center * 1000).tolist(), image_size=[1000, 700],
        intrinsics={"fx": 900, "fy": 900, "cx": 500, "cy": 350}, distortion=[0] * 5,
    )
    preview_config["simulation"]["noise_std_gray"] = 0
    K, dist, size = camera_parameters(preview_config)
    transform = nominal_pose(preview_config)
    preview = render_scene(geometry, transform, K, dist, size, preview_config, surface_patches=surface_patches).image
    rvec, tvec = pose_vectors(transform)
    cv2.drawFrameAxes(preview, K, dist, rvec, tvec, 0.012, 2)
    extent = np.ptp(geometry.vertices @ geometry.T_stl_port[:3, :3].T, axis=0) * 1000
    lines = ["STL assembly | port axes: X right, Y down, Z insertion",
             f"Size in STL axes: {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} mm"]
    save_image(path, _caption(preview, lines))


def aruco_detection_image(image: np.ndarray, detection: MarkerDetection, status: str) -> np.ndarray:
    """标出所有解码 ID、编码角点顺序以及解码失败的四边形，全部来自图像。"""
    canvas = image.copy()
    for rejected in detection.diagnostics["rejected_quads_px"]:
        cv2.polylines(canvas, [np.rint(rejected).astype(np.int32)], True, (170, 100, 170), 1)
    for record in detection.diagnostics["markers"]:
        pixels = np.asarray(record["corners_px"])
        color = (30, 190, 20) if record["accepted"] else (20, 140, 230)
        cv2.polylines(canvas, [np.rint(pixels).astype(np.int32)], True, color, 2, cv2.LINE_AA)
        label = f"ID {record['marker_id']}"
        if record["accepted"]:
            label += f" / Target {record['target_id']}"
        cv2.putText(canvas, label, tuple(np.rint(pixels[0] - [0, 12]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        for index, pixel in enumerate(pixels):
            point = tuple(np.rint(pixel).astype(int))
            cv2.circle(canvas, point, 3, (0, 0, 255) if index == 0 else color, -1)
            cv2.putText(canvas, str(index), (point[0] + 4, point[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return _caption(canvas, [f"ArUco | {status} | no camera pose prior",
                             "green: target | orange: ignored | purple: rejected | red corner: 0"])


def save_aruco_comparison(path: Path, image: np.ndarray, geometry: Geometry, truth: np.ndarray,
                          result: dict[str, Any], K: np.ndarray, dist: np.ndarray) -> None:
    """复用候选对比面板，补入标记的端口坐标；不修改原几何或估计结果。"""
    corners = transform_points(marker_corners(result["marker_size_m"]), np.asarray(result["T_port_marker"]))
    report_geometry = replace(geometry, landmarks={**geometry.landmarks, "marker": corners})
    _save_image_only_comparison(path, image, report_geometry, truth, result, K, dist)


def detection_image(image: np.ndarray, detection: DetectionResult) -> np.ndarray:
    """绘制实际检测点及搜索区域；失败时仍提供 ROI 线索。"""
    canvas = image.copy()
    for target in detection.diagnostics.get("targets", []):
        pixels = np.asarray(target["inner_px"])
        cv2.polylines(canvas, [np.rint(pixels).astype(np.int32)], True, (255, 120, 0), 2, cv2.LINE_AA)
        position = tuple(np.rint(pixels.min(axis=0) - [0, 8]).astype(int))
        cv2.putText(canvas, f"Target {target['target_id']}", position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (220, 90, 0), 2, cv2.LINE_AA)
    roi = detection.diagnostics.get("roi_xyxy")
    if roi:
        x0, y0, x1, y1 = roi
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (220, 170, 30), 1)
    styles = {"inner": ((255, 100, 20), "I"), "outer": ((30, 190, 20), "O"),
              "top_rear": ((20, 150, 255), "T")}
    for name, pixels in detection.points.items():
        color, prefix = styles[name]
        if len(pixels) >= 4:
            cv2.polylines(canvas, [np.rint(pixels).astype(np.int32)], True, color, 1, cv2.LINE_AA)
        for index, point in enumerate(pixels):
            x, y = np.rint(point).astype(int)
            cv2.circle(canvas, (x, y), 3, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"{prefix}{index + 1}", (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    lines = ["Image observations | I: inner, O: outer, T: rear top",
             "Detection succeeded" if detection.success else "Detection failed - see pose_result.json"]
    if detection.diagnostics.get("use_prior") is False:
        lines = ["Full-image detection | no pose prior",
                 "Ambiguous: see numbered pose candidates" if detection.diagnostics.get("result_status") == "ambiguous"
                 else "Target detected" if detection.success else "No reliable target - see JSON"]
    return _caption(canvas, lines)


def _pose_panel(image: np.ndarray, points: np.ndarray, truth: np.ndarray,
                estimated: np.ndarray | None, K: np.ndarray, dist: np.ndarray,
                title: str, metrics: str, crop: tuple[int, int, int, int]) -> np.ndarray:
    canvas = image.copy()
    for pixel in project_points(points, truth, K, dist):
        if np.isfinite(pixel).all() and np.max(np.abs(pixel)) < 1e6:
            cv2.circle(canvas, tuple(np.rint(pixel).astype(int)), 3, (0, 180, 0), -1, cv2.LINE_AA)
    if estimated is not None:
        for pixel in project_points(points, estimated, K, dist):
            if np.isfinite(pixel).all() and np.max(np.abs(pixel)) < 1e6:
                cv2.drawMarker(canvas, tuple(np.rint(pixel).astype(int)), (0, 0, 230), cv2.MARKER_CROSS, 8, 1)
        rvec, tvec = pose_vectors(estimated)
        cv2.drawFrameAxes(canvas, K, dist, rvec, tvec, 0.010, 1)
    x0, y0, x1, y1 = crop
    region = canvas[y0:y1, x0:x1]
    scale = min(640 / region.shape[1], 380 / region.shape[0])
    region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    panel = np.full((460, 640, 3), 245, np.uint8)
    left = (640 - region.shape[1]) // 2
    top = 80 + (380 - region.shape[0]) // 2
    panel[top:top + region.shape[0], left:left + region.shape[1]] = region
    return _caption(panel, [title, metrics])


def save_pose_comparison(path: Path, image: np.ndarray, geometry: Geometry,
                         truth: np.ndarray, result: dict[str, Any], K: np.ndarray,
                         dist: np.ndarray, detection: DetectionResult) -> None:
    """放大目标区域，对比真值、平面解和非共面解。"""
    if result.get("use_prior") is False:
        _save_image_only_comparison(path, image, geometry, truth, result, K, dist)
        return
    points = np.vstack([value for value in geometry.landmarks.values() if len(value)])
    height, width = image.shape[:2]
    crop = (0, 0, width, height)
    roi = detection.diagnostics.get("roi_xyxy")
    if roi and roi[2] > roi[0] and roi[3] > roi[1]:
        crop = tuple(roi)
    panels = [_pose_panel(image, points, truth, truth, K, dist, "Ground truth", "green: truth | red: estimate", crop)]
    planar = result.get("planar", {})
    planar_transform = None
    if planar.get("status") == "ok":
        planar_transform = np.asarray(planar["candidates"][planar["selected_index"]]["T_camera_port"])
    nonplanar = result.get("nonplanar", {})
    nonplanar_transform = np.asarray(nonplanar["T_camera_port"]) if nonplanar.get("status") == "ok" else None
    for title, estimate in (("Planar IPPE + nominal prior", planar_transform), ("Nonplanar PnP + inlier LM", nonplanar_transform)):
        metrics = "Unavailable - see JSON diagnosis"
        if estimate is not None:
            error = evaluate_pose(estimate, truth)
            metrics = f"rotation: {error['rotation_error_deg']:.3f} deg | translation: {error['translation_error_mm']:.3f} mm"
        panels.append(_pose_panel(image, points, truth, estimate, K, dist, title, metrics, crop))
    save_image(path, np.hstack(panels))


def _save_image_only_comparison(path: Path, image: np.ndarray, geometry: Geometry,
                                truth: np.ndarray, result: dict[str, Any],
                                K: np.ndarray, dist: np.ndarray) -> None:
    """按图像评分顺序展示全部有效候选；真值误差只用于事后标注。"""
    height, width = image.shape[:2]
    all_points = np.vstack([p for p in geometry.landmarks.values() if len(p)])
    indices = result.get("active_candidate_indices", [])
    selected = result.get("selected_index")
    panels = [_pose_panel(image, all_points, truth, truth, K, dist, "Ground truth | evaluation only",
                          f"{result.get('route', 'Image-only')} status: {result['status']}", (0, 0, width, height))]
    for index in indices:
        candidate = result["candidates"][index]
        observed = np.vstack(list(candidate["points_px"].values()))
        low = np.maximum(np.floor(observed.min(axis=0) - 30).astype(int), 0)
        high = np.minimum(np.ceil(observed.max(axis=0) + 30).astype(int), [width, height])
        crop = (int(low[0]), int(low[1]), int(high[0]), int(high[1]))
        estimate = np.asarray(candidate["T_camera_port"])
        errors = evaluate_pose(estimate, truth)
        title = f"C{index} / Target {candidate['target_id']} / {candidate['method']}"
        if selected == index:
            title += " (selected)"
        text = f"score={candidate['score_px']:.3f}px | err={errors['rotation_error_deg']:.2f}deg, {errors['translation_error_mm']:.2f}mm"
        model_points = np.vstack([geometry.landmarks[name] for name in candidate["points_px"]])
        if result.get("route") == "aruco":
            # 同时显示端口轮廓，便于观察标记到端口的位姿传递误差。
            model_points = np.vstack([model_points, geometry.landmarks["outer"]])
            crop = (0, 0, width, height)
        panels.append(_pose_panel(image, model_points, truth, estimate, K, dist, title, text, crop))
    if len(panels) % 2:
        panels.append(np.full_like(panels[0], 245))
    save_image(path, np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)]))


def save_report(output: Path, config: dict[str, Any], geometry: Geometry,
                nominal: np.ndarray, truth: np.ndarray, result: dict[str, Any],
                detection: DetectionResult | MarkerDetection | None) -> dict[str, Any]:
    """保存 JSON 和生效配置；每种位姿的误差都在这里单独评估。"""
    report = deepcopy(result)
    report.update(
        use_prior=config.get("runtime", {}).get("use_prior", result.get("use_prior", True)),
        source_config=config["source_config"], model_path=config["model"]["path"],
        units={"input_model": config["model"]["unit"], "coordinates_and_translation": "meter", "translation_error": "mm"},
        coordinate_frame={"x": "right", "y": "down", "z": "insertion", "T_camera_port": "p_camera = T_camera_port @ p_port (homogeneous coordinates)"},
        mesh={"triangle_count": len(geometry.faces), "vertex_count": len(geometry.vertices), "T_stl_port": geometry.T_stl_port.tolist()},
        camera=config["camera"], nominal_T_camera_port=nominal.tolist(),
        ground_truth={"T_camera_port": truth.tolist()},
        landmarks_port_m={key: value.tolist() for key, value in geometry.landmarks.items()},
    )
    if detection is not None:
        report["detection"] = {"success": detection.success, "reason": detection.reason,
                               "points_px": {k: v.tolist() for k, v in detection.points.items()},
                               "diagnostics": detection.diagnostics}
    if report.get("T_camera_port") is not None:
        report["errors_vs_truth"] = evaluate_pose(np.asarray(report["T_camera_port"]), truth)
    for candidate in report.get("candidates", []):
        candidate["errors_vs_truth"] = evaluate_pose(np.asarray(candidate["T_camera_port"]), truth)
    for candidate in report.get("planar", {}).get("candidates", []):
        candidate["errors_vs_truth"] = evaluate_pose(np.asarray(candidate["T_camera_port"]), truth)
    if report.get("nonplanar", {}).get("status") == "ok":
        report["nonplanar"]["errors_vs_truth"] = evaluate_pose(np.asarray(report["nonplanar"]["T_camera_port"]), truth)
    if report.get("route") == "aruco":
        marker_truth = truth @ np.asarray(report["T_port_marker"])
        report["ground_truth"]["T_camera_marker"] = marker_truth.tolist()
        report["coordinate_frame"].update(
            marker={"x": "code right", "y": "code up", "z": "out of printed face"},
            T_port_marker="p_port = T_port_marker @ p_marker",
            T_camera_marker="p_camera = T_camera_marker @ p_marker",
            T_port_camera="p_port = T_port_camera @ p_camera",
            composition="T_camera_port = T_camera_marker @ inverse(T_port_marker)")
        report["aruco"] = deepcopy(config["aruco"])
        for candidate in report.get("candidates", []):
            candidate["marker_errors_vs_truth"] = evaluate_pose(candidate["T_camera_marker"], marker_truth)
    (output / "pose_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "actual_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return report
