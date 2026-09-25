"""STL 端口仿真入口：只负责串起各模块，算法在 port_pose 包中。

常用命令：
    python run_demo.py
    python run_demo.py --config configs/socket.yaml --output-dir outputs_test
    python run_demo.py --preview-only
    python run_demo.py --no-use-prior --output-dir outputs/image_only
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from ur5e_sim.vision.core.camera import actual_pose, camera_parameters, nominal_pose
from ur5e_sim.vision.core.config import load_config
from ur5e_sim.vision.core.detection import detect_features
from ur5e_sim.vision.core.geometry import load_geometry
from ur5e_sim.vision.core.image_only import estimate_from_image
from ur5e_sim.vision.core.pose import estimate_pose
from ur5e_sim.vision.core.renderer import render_scene
from ur5e_sim.vision.core.reporting import (
    detection_image, save_image, save_model_preview, save_pose_comparison, save_report,
)

from ur5e_sim.paths import ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="真实 STL 端口：仿真拍摄、图像检测与位姿估计")
    parser.add_argument("--config", type=Path, default=ROOT / "tests/fixtures/socket.yaml",
                        help="YAML 配置文件；默认使用附带插座模型")
    parser.add_argument("--output-dir", type=Path, help="覆盖输出目录，相对路径按启动目录解析")
    parser.add_argument("--preview-only", action="store_true", default=False, help="只输出模型预览和模拟照片，不运行检测和求解")
    parser.add_argument("--use-prior", action=argparse.BooleanOptionalAction, default=False,
                        help="是否使用名义位姿辅助识别；--no-use-prior 改为全图视觉初始化")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    # 命令行开关也写进快照，便于确认每份结果究竟使用了哪一种模式。
    config["runtime"] = {"use_prior": args.use_prior, "preview_only": args.preview_only}
    if args.output_dir is not None:
        config["output"]["directory"] = str(args.output_dir.expanduser().resolve())
    output = Path(config["output"]["directory"])
    geometry = load_geometry(config)
    K, dist, size = camera_parameters(config)
    nominal = nominal_pose(config)
    truth = actual_pose(nominal, config["simulation"])
    output.mkdir(parents=True, exist_ok=True)
    # 清理上一次运行的已知产物，避免仅预览/失败后误读旧的成功结果。
    for filename in ("cad_preview.png", "simulated_rgb.png", "detected_features.png", "detection_edges.png", "pose_comparison.png", "pose_result.json", "actual_config.yaml"):
        (output / filename).unlink(missing_ok=True)
    # 先保存配置及未完成状态；即使计算被中断，也不会留下此前的成功结果。
    save_report(output, config, geometry, nominal, truth,
                {"status": "failed", "method": None, "T_camera_port": None,
                 "warnings": ["运行尚未完成；若程序已退出，请检查终端中的错误信息。"]}, None)
    print(f"模型：{config['model']['path']}（{len(geometry.faces)} 个三角形，单位 {config['model']['unit']}）")
    print(f"拍摄：距离 {config['camera']['distance_mm']:g} mm，水平角 {config['camera']['azimuth_deg']:g}°，仰角 {config['camera']['elevation_deg']:g}°")
    print("定位模式：" + ("使用名义位姿" if args.use_prior else "无名义位姿，全图视觉初始化"))
    try:
        save_model_preview(output / "cad_preview.png", geometry, config)
        rendered = render_scene(geometry, truth, K, dist, size, config)
        save_image(output / "simulated_rgb.png", rendered.image)
    except (ValueError, OSError, cv2.error) as exc:
        save_report(output, config, geometry, nominal, truth,
                    {"status": "failed", "method": None, "T_camera_port": None,
                     "warnings": [f"成像失败：{exc}"]}, None)
        raise
    if args.preview_only:
        save_report(output, config, geometry, nominal, truth,
                    {"status": "preview_only", "method": None, "T_camera_port": None}, None)
        print(f"预览完成：{output}")
        return 0
    if args.use_prior:
        detection = detect_features(rendered.image, geometry.landmarks, nominal, K, dist, config)
        if detection.success:
            result = estimate_pose(geometry.landmarks, detection.points, K, dist, nominal, config)
        else:
            result = {"status": "failed", "method": None, "T_camera_port": None, "warnings": [detection.reason]}
    else:
        # 严格隔离输入：不传 nominal、truth、深度或包含拍摄位姿的完整配置。
        detection, result = estimate_from_image(rendered.image, geometry.landmarks, K, dist,
                                                config["detection"], config["pose"])
    result["use_prior"] = args.use_prior
    save_image(output / "detected_features.png", detection_image(rendered.image, detection))
    save_image(output / "detection_edges.png", detection.edges)
    save_pose_comparison(output / "pose_comparison.png", rendered.image, geometry, truth, result, K, dist, detection)
    report = save_report(output, config, geometry, nominal, truth, result, detection)
    if result["status"] == "ambiguous":
        print(f"位姿存在歧义：保留 {len(result['active_candidate_indices'])} 个有效候选，未指定唯一位姿。")
        print("请查看 pose_result.json 和 pose_comparison.png 中的候选及图像评分。")
    elif result["status"] == "failed":
        print("未能可靠估计位姿：" + "；".join(result.get("warnings", [])))
    else:
        errors = report["errors_vs_truth"]
        print(f"结果：{result['method']}；旋转误差 {errors['rotation_error_deg']:.3f}°，平移误差 {errors['translation_error_mm']:.3f} mm")
        for warning in result.get("warnings", []):
            print("说明：" + warning)
    print(f"结果目录：{output}")
    return 2 if result["status"] in ("failed", "ambiguous") else 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (ValueError, OSError, cv2.error) as exc:
        print(f"运行错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
