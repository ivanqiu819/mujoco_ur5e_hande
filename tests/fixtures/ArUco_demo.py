"""顶面贴 ArUco 的独立仿真入口；与 run_demo.py 共用 YAML 和相机。

python ArUco_demo.py
python ArUco_demo.py --config configs/socket.yaml --output-dir outputs/aruco_test
python ArUco_demo.py --preview-only
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from ur5e_sim.vision.core.aruco import detect_markers, marker_image, marker_patch, marker_transform
from ur5e_sim.vision.core.camera import actual_pose, camera_parameters, nominal_pose
from ur5e_sim.vision.core.config import load_config
from ur5e_sim.vision.core.geometry import load_geometry
from ur5e_sim.vision.core.pose import estimate_marker_pose
from ur5e_sim.vision.core.renderer import render_scene
from ur5e_sim.vision.core.reporting import (aruco_detection_image, save_aruco_comparison, save_image,
                                 save_model_preview, save_report)

from ur5e_sim.paths import ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STL 顶面贴 ArUco：由标记安装关系定位端口")
    parser.add_argument("--config", type=Path, default=ROOT / "tests/fixtures/socket.yaml", help="共用的 YAML 配置")
    parser.add_argument("--output-dir", type=Path, help="覆盖 aruco.output_directory；相对路径按启动目录解析")
    parser.add_argument("--preview-only", action="store_true", help="只输出码图、装配体预览和模拟照片")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config, require_aruco=True)
    settings = config["aruco"]
    if args.output_dir is not None:
        settings["output_directory"] = str(args.output_dir.expanduser().resolve())
    output = Path(settings["output_directory"])
    config["runtime"] = {"route": "aruco", "use_prior": False, "preview_only": args.preview_only,
                         "output_directory": str(output)}
    geometry = load_geometry(config)
    K, dist, size = camera_parameters(config)
    nominal = nominal_pose(config)
    truth = actual_pose(nominal, config["simulation"])
    installation = marker_transform(settings)
    patches = (marker_patch(settings),)
    result = {"status": "failed", "route": "aruco", "method": None, "use_prior": False,
              "T_camera_marker": None, "T_camera_port": None, "T_port_camera": None,
              "T_port_marker": installation.tolist(), "marker_size_m": settings["marker_size_mm"] / 1000,
              "selected_index": None, "candidates": [], "active_candidate_indices": [],
              "warnings": ["运行尚未完成；若程序已退出，请检查终端错误信息。"]}
    output.mkdir(parents=True, exist_ok=True)
    for name in ("aruco_marker.png", "cad_preview.png", "simulated_rgb.png", "detected_features.png",
                 "pose_comparison.png", "pose_result.json", "actual_config.yaml"):
        (output / name).unlink(missing_ok=True)
    save_report(output, config, geometry, nominal, truth, result, None)
    print(f"ArUco：{settings['dictionary']} / ID {settings['marker_id']} / 边长 {settings['marker_size_mm']:g} mm")
    print(f"拍摄：{config['camera']['distance_mm']:g} mm，水平角 {config['camera']['azimuth_deg']:g}°，仰角 {config['camera']['elevation_deg']:g}°")
    print("定位模式：仅 ArUco 图像与已知安装关系，不使用名义机位")
    detection = None
    try:
        save_image(output / "aruco_marker.png", marker_image(settings))
        save_model_preview(output / "cad_preview.png", geometry, config, surface_patches=patches)
        rendered = render_scene(geometry, truth, K, dist, size, config, surface_patches=patches)
        save_image(output / "simulated_rgb.png", rendered.image)
        if args.preview_only:
            result.update(status="preview_only", warnings=[])
        else:
            # 检测器不接收安装外参、相机外参、完整配置或深度图。
            detector_settings = {key: settings[key] for key in
                                 ("dictionary", "marker_id", "corner_refinement_window_px", "min_marker_side_px")}
            detection = detect_markers(rendered.image, detector_settings)
            result = estimate_marker_pose(list(detection.points.values()), settings["marker_size_mm"] / 1000,
                                           K, dist, installation, config["pose"])
            if not detection.success:
                result["warnings"] = [detection.reason]
            save_image(output / "detected_features.png", aruco_detection_image(rendered.image, detection, result["status"]))
            save_aruco_comparison(output / "pose_comparison.png", rendered.image, geometry, truth, result, K, dist)
        report = save_report(output, config, geometry, nominal, truth, result, detection)
    except (ValueError, OSError, cv2.error) as exc:
        result.update(status="failed", method=None, selected_index=None, T_camera_marker=None,
                      T_camera_port=None, T_port_camera=None, warnings=[f"运行失败：{exc}"])
        save_report(output, config, geometry, nominal, truth, result, detection)
        raise
    if result["status"] == "ok":
        errors = report["errors_vs_truth"]
        print(f"端口位姿：旋转误差 {errors['rotation_error_deg']:.3f}°，平移误差 {errors['translation_error_mm']:.3f} mm")
    elif result["status"] == "ambiguous":
        print("位姿存在歧义：" + "；".join(result["warnings"]))
    elif result["status"] == "failed":
        print("ArUco 定位失败：" + "；".join(result["warnings"]))
    else:
        print("预览完成，已跳过识别和位姿求解。")
    print(f"结果目录：{output}")
    return 2 if result["status"] in ("failed", "ambiguous") else 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (ValueError, OSError, cv2.error) as exc:
        print(f"运行错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())  # 2 表示失败或歧义；调试器可能在正常的状态退出处暂停。
