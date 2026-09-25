"""图像安全退化、深度遮挡及可选的完整流程验收。

快速测试：python -m unittest discover -s tests -v
完整测试：PORT_POSE_FULL_TESTS=1 python -m unittest discover -s tests -v
完整测试会在临时目录运行 CLI，不覆盖项目 outputs/。
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from ur5e_sim.vision.core.camera import actual_pose, camera_parameters, nominal_pose, project_points
from tests.vision_fixtures import simulation_config
from ur5e_sim.vision.core.detection import detect_features
from ur5e_sim.vision.core.geometry import Geometry, load_geometry
from ur5e_sim.vision.core.pose import estimate_pose, evaluate_pose
from ur5e_sim.vision.core.renderer import render_scene


from ur5e_sim.paths import ROOT


class RenderDepthTests(unittest.TestCase):
    def test_rendered_radial_distortion_matches_solver_projection(self) -> None:
        """用正对相机的方片边界检查 remap 与 projectPoints 的畸变方向。"""
        config = simulation_config()
        config["simulation"]["noise_std_gray"] = 0
        config["render"]["material_regions"] = []
        vertices = np.array([[-0.04, -0.04, 0.1], [0.04, -0.04, 0.1],
                             [0.04, 0.04, 0.1], [-0.04, 0.04, 0.1]])
        geometry = Geometry(vertices, np.array([[0, 1, 2], [0, 2, 3]]), {}, np.eye(4), 1.0)
        K = np.array([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]])
        distortion = np.array([0.8, 0, 0, 0, 0])
        result = render_scene(geometry, np.eye(4), K, distortion, (128, 128), config)
        edge_midpoints = np.array([[-0.04, 0, 0.1], [0.04, 0, 0.1]])
        projected = project_points(edge_midpoints, np.eye(4), K, distortion)
        occupied = np.flatnonzero(np.isfinite(result.depth[64]))
        self.assertLess(abs(occupied.min() - projected[0, 0]), 1.2)
        self.assertLess(abs(occupied.max() - projected[1, 0]), 1.2)
        # 未施加畸变时边界为 x=24、104；这里必须能分辨出方向一致的外扩。
        self.assertLess(occupied.min(), 21)
        self.assertGreater(occupied.max(), 107)

    def test_overlapping_triangles_use_per_pixel_depth_regardless_of_face_order(self) -> None:
        """两个三角形在同一画面位置交叠，前后关系在画面左右反转。

        按三角形平均深度整体排序不能正确画这个例子，必须逐像素比较。
        """
        config = simulation_config()
        config["simulation"]["noise_std_gray"] = 0
        config["render"]["material_regions"] = []
        projected = np.array([[-0.2, -0.2], [0.2, -0.2], [0.0, 0.2]])
        sloping_depths = np.array([0.1, 0.3, 0.2])
        sloping = np.column_stack([projected * sloping_depths[:, None], sloping_depths])
        flat = np.column_stack([projected * 0.18, np.full(3, 0.18)])
        vertices = np.vstack([sloping, flat])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        K = np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]])
        geometry = Geometry(vertices, faces, {}, np.eye(4), 1.0)
        result = render_scene(geometry, np.eye(4), K, np.zeros(5), (64, 64), config)
        reversed_geometry = Geometry(vertices, faces[::-1], {}, np.eye(4), 1.0)
        reversed_result = render_scene(reversed_geometry, np.eye(4), K, np.zeros(5), (64, 64), config)
        self.assertAlmostEqual(float(result.depth[20, 22]), 0.125, places=6)
        self.assertAlmostEqual(float(result.depth[20, 42]), 0.180, places=6)
        self.assertTrue(np.isinf(result.depth[0, 0]))
        np.testing.assert_array_equal(result.depth, reversed_result.depth)
        np.testing.assert_array_equal(result.image, reversed_result.image)


class DetectionFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = simulation_config()
        cls.geometry = load_geometry(cls.config)
        cls.blank = np.full((720, 1280, 3), 245, dtype=np.uint8)

    def test_no_image_evidence_never_returns_nominal_corner_coordinates(self) -> None:
        K, dist, _ = camera_parameters(self.config)
        detection = detect_features(self.blank, self.geometry.landmarks,
                                    nominal_pose(self.config), K, dist, self.config)
        self.assertFalse(detection.success)
        self.assertEqual(detection.points, {})
        self.assertTrue(detection.reason)
        self.assertEqual(detection.edges.shape, self.blank.shape[:2])

    def test_back_view_out_of_frame_and_insufficient_pixels_have_diagnostics(self) -> None:
        scenarios = [
            ({"azimuth_deg": 180}, "背面"),
            ({"distance_mm": 2000}, "像素不足"),
            ({"target_offset_mm": [160, 0, 0]}, "超出画面"),
        ]
        for settings, message in scenarios:
            with self.subTest(camera=settings):
                config = copy.deepcopy(self.config)
                config["camera"].update(settings)
                K, dist, _ = camera_parameters(config)
                result = detect_features(self.blank, self.geometry.landmarks,
                                         nominal_pose(config), K, dist, config)
                self.assertFalse(result.success)
                self.assertIn(message, result.reason)
                self.assertEqual(result.points, {})


@unittest.skipUnless(os.environ.get("PORT_POSE_FULL_TESTS") == "1",
                     "设置 PORT_POSE_FULL_TESTS=1 运行完整渲染和 CLI 验收")
class FullPipelineTests(unittest.TestCase):
    def test_oblique_views_recover_nonplanar_pose_within_accuracy_target(self) -> None:
        config = simulation_config()
        geometry = load_geometry(config)
        K, dist, size = camera_parameters(config)
        for azimuth, elevation in ((-15, 15), (0, 20), (30, 20)):
            with self.subTest(azimuth=azimuth, elevation=elevation):
                config["camera"].update(azimuth_deg=azimuth, elevation_deg=elevation)
                nominal = nominal_pose(config)
                truth = actual_pose(nominal, config["simulation"])
                rendered = render_scene(geometry, truth, K, dist, size, config)
                detected = detect_features(rendered.image, geometry.landmarks, nominal, K, dist, config)
                self.assertTrue(detected.success, detected.reason)
                result = estimate_pose(geometry.landmarks, detected.points, K, dist, nominal, config)
                self.assertEqual(result["status"], "ok", result)
                errors = evaluate_pose(result["T_camera_port"], truth)
                self.assertLessEqual(errors["rotation_error_deg"], 3.0, errors)
                self.assertLessEqual(errors["translation_error_mm"], 3.0, errors)

    def test_front_view_reports_planar_ambiguity_without_top_features(self) -> None:
        config = simulation_config()
        config["camera"].update(azimuth_deg=0, elevation_deg=0)
        geometry = load_geometry(config)
        K, dist, size = camera_parameters(config)
        nominal = nominal_pose(config)
        truth = actual_pose(nominal, config["simulation"])
        rendered = render_scene(geometry, truth, K, dist, size, config)
        detected = detect_features(rendered.image, geometry.landmarks, nominal, K, dist, config)
        self.assertTrue(detected.success, detected.reason)
        self.assertNotIn("top_rear", detected.points)
        result = estimate_pose(geometry.landmarks, detected.points, K, dist, nominal, config)
        self.assertEqual(result["status"], "planar_only", result)
        self.assertTrue(result["warnings"])
        self.assertGreaterEqual(len(result["planar"]["candidates"]), 1)
        # 正视的小矩形存在平面姿态歧义，不将非共面精度目标套用在此处。

    def test_partial_port_occlusion_does_not_return_invented_corners(self) -> None:
        config = simulation_config()
        geometry = load_geometry(config)
        K, dist, size = camera_parameters(config)
        nominal = nominal_pose(config)
        truth = actual_pose(nominal, config["simulation"])
        rendered = render_scene(geometry, truth, K, dist, size, config)
        occluded = rendered.image.copy()
        expected = project_points(geometry.landmarks["outer"], nominal, K, dist)
        x0, y0 = np.floor(expected.min(axis=0) - 10).astype(int)
        x1, y1 = np.ceil(expected.max(axis=0) + 10).astype(int)
        # 只盖住插口右半侧，装配体、顶面和插口左半侧仍然保留。
        occluded[y0:y1, (x0 + x1) // 2:x1] = config["render"]["background_gray"]
        detected = detect_features(occluded, geometry.landmarks, nominal, K, dist, config)
        self.assertFalse(detected.success, detected.diagnostics)
        self.assertEqual(detected.points, {})
        self.assertTrue(detected.reason)

    def test_default_stl_image_detection_and_pose_meet_accuracy_target(self) -> None:
        config = simulation_config()
        geometry = load_geometry(config)
        K, dist, size = camera_parameters(config)
        nominal = nominal_pose(config)
        truth = actual_pose(nominal, config["simulation"])
        rendered = render_scene(geometry, truth, K, dist, size, config)
        # 不把 truth、rendered.depth 或真实角点传给检测器。
        detected = detect_features(rendered.image, geometry.landmarks, nominal, K, dist, config)
        self.assertTrue(detected.success, detected.reason)
        result = estimate_pose(geometry.landmarks, detected.points, K, dist, nominal, config)
        self.assertEqual(result["status"], "ok", result)
        errors = evaluate_pose(result["T_camera_port"], truth)
        self.assertLessEqual(errors["rotation_error_deg"], 3.0, errors)
        self.assertLessEqual(errors["translation_error_mm"], 3.0, errors)

    def test_preview_cli_runs_from_unrelated_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "tests/fixtures/run_demo.py"), "--preview-only", "--output-dir", "预览输出"],
                cwd=directory, text=True, encoding="utf-8", capture_output=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            output = Path(directory) / "预览输出"
            for name in ("cad_preview.png", "simulated_rgb.png", "actual_config.yaml", "pose_result.json"):
                self.assertTrue((output / name).is_file(), name)
            result = json.loads((output / "pose_result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "preview_only")

    def test_configuration_error_cli_returns_one_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "tests/fixtures/run_demo.py"), "--config", "不存在.yaml"],
                cwd=directory, text=True, encoding="utf-8", capture_output=True, timeout=30,
            )
            self.assertEqual(completed.returncode, 1)
            message = completed.stdout + completed.stderr
            self.assertIn("配置", message)
            self.assertNotIn("Traceback", message)

    def test_backside_cli_keeps_failure_diagnostics_without_estimated_pose(self) -> None:
        config = simulation_config()
        config["camera"]["azimuth_deg"] = 180
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "背面拍摄.yaml"
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "tests/fixtures/run_demo.py"), "--config", str(config_path),
                 "--use-prior", "--output-dir", "失败诊断"],
                cwd=directory, text=True, encoding="utf-8", capture_output=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            output = Path(directory) / "失败诊断"
            result = json.loads((output / "pose_result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["T_camera_port"])
            self.assertIn("背面", result["detection"]["reason"])
            for name in ("simulated_rgb.png", "detected_features.png", "detection_edges.png", "pose_comparison.png"):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
