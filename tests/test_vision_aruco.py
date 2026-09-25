"""ArUco 数值、图像与端到端验收。完整渲染需 PORT_POSE_FULL_TESTS=1。"""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import yaml

from tests.vision_fixtures import ROOT, aruco_config
from ur5e_sim.vision.core.aruco import detect_markers, marker_corners, marker_image, marker_patch, marker_transform
from ur5e_sim.vision.core.camera import actual_pose, camera_parameters, euler_rotation, nominal_pose, project_points
from ur5e_sim.vision.core.config import load_config
from ur5e_sim.vision.core.geometry import Geometry, load_geometry, transform_points
from ur5e_sim.vision.core.pose import estimate_marker_pose, evaluate_pose
from ur5e_sim.vision.core.renderer import render_scene
from ur5e_sim.vision.core.reporting import save_report


def solve(image: np.ndarray, config: dict) -> tuple:
    K, dist, _ = camera_parameters(config)
    settings = config["aruco"]
    detected = detect_markers(image, settings)
    result = estimate_marker_pose(list(detected.points.values()), settings["marker_size_mm"] / 1000,
                                  K, dist, marker_transform(settings), config["pose"])
    return detected, result


def code_scene(settings: dict, ids: tuple[int, ...]) -> np.ndarray:
    """直接绘制编码图案，不使用 STL/投影/外参，验证 ID 与目标数量处理。"""
    image = np.full((480, 800, 3), 230, np.uint8)
    for index, marker_id in enumerate(ids):
        code = marker_image({**settings, "marker_id": marker_id})
        code = cv2.resize(code, (180, 180), interpolation=cv2.INTER_NEAREST)
        x = 80 + 320 * index
        image[140:320, x:x + 180] = cv2.cvtColor(code, cv2.COLOR_GRAY2BGR)
    return image


class ArUcoNumericTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = aruco_config()
        self.settings = self.config["aruco"]
        self.K, self.dist, _ = camera_parameters(self.config)
        self.installation = marker_transform(self.settings)
        self.model = marker_corners(.014)

    def test_installation_axes_corners_units_and_round_trip(self) -> None:
        np.testing.assert_allclose(self.installation[:3, 3], [0, -.00405, .016])
        np.testing.assert_allclose(self.installation[:3, 2], [0, -1, 0], atol=1e-14)
        port_points = transform_points(self.model, self.installation)
        np.testing.assert_allclose(port_points[0], [-.007, -.00405, .023], atol=1e-14)
        np.testing.assert_allclose(transform_points(port_points, np.linalg.inv(self.installation)), self.model, atol=1e-14)
        patch = marker_patch(self.settings)
        np.testing.assert_allclose(patch.vertices.min(axis=0), [-.009, -.00405, .007], atol=1e-14)
        np.testing.assert_allclose(patch.vertices.max(axis=0), [.009, -.00405, .025], atol=1e-14)
        triangles = patch.vertices[patch.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        self.assertTrue(np.all(normals[:, 1] < 0))

    def test_known_pose_and_inverse_composition_without_renderer(self) -> None:
        expected = actual_pose(nominal_pose(self.config), self.config["simulation"])
        marker_pose = expected @ self.installation
        pixels = project_points(self.model, marker_pose, self.K, self.dist)
        result = estimate_marker_pose([pixels], .014, self.K, self.dist, self.installation, self.config["pose"])
        self.assertEqual(result["status"], "ok", result)
        np.testing.assert_allclose(result["T_camera_marker"], marker_pose, atol=1e-7)
        np.testing.assert_allclose(result["T_camera_port"], expected, atol=1e-7)
        np.testing.assert_allclose(np.asarray(result["T_camera_port"]) @ result["T_port_camera"], np.eye(4), atol=1e-12)
        # 更换已知安装关系只应改变端口结果，不影响标记观测的位姿。
        other_installation = self.installation.copy()
        other_installation[:3, 3] += [.010, -.003, .004]
        other = estimate_marker_pose([pixels], .014, self.K, self.dist, other_installation, self.config["pose"])
        np.testing.assert_allclose(other["T_camera_marker"], marker_pose, atol=1e-7)
        np.testing.assert_allclose(other["T_camera_port"], marker_pose @ np.linalg.inv(other_installation), atol=1e-7)

    def test_real_planar_ambiguity_keeps_candidates_and_null_final_pose(self) -> None:
        # 小码近正视、带确定性亚像素噪声：两个倾斜解都可解释同一组角点。
        transform = np.eye(4)
        transform[:3, :3] = euler_rotation([170, 0, 0])
        transform[:3, 3] = [.015, .002, .3]
        pixels = project_points(self.model, transform, self.K, self.dist)
        pixels += np.random.default_rng(42).normal(0, .03, (4, 2))
        result = estimate_marker_pose([pixels], .014, self.K, self.dist, self.installation, self.config["pose"])
        self.assertEqual(result["status"], "ambiguous")
        self.assertGreaterEqual(len(result["active_candidate_indices"]), 2)
        for key in ("selected_index", "T_camera_marker", "T_camera_port", "T_port_camera"):
            self.assertIsNone(result[key])

    def test_no_camera_prior_or_truth_in_detection_and_ranking(self) -> None:
        image = code_scene(self.settings, (0,))
        detection, baseline = solve(image, self.config)
        changed = copy.deepcopy(self.config)
        changed["pose"].update(prior_rotation_weight=1e10, prior_translation_weight=1e10)
        del changed["camera"], changed["simulation"]
        with mock.patch("ur5e_sim.vision.core.camera.nominal_pose", side_effect=AssertionError("名义位姿泄漏")), \
             mock.patch("ur5e_sim.vision.core.camera.actual_pose", side_effect=AssertionError("真值泄漏")), \
             mock.patch("ur5e_sim.vision.core.detection.detect_features", side_effect=AssertionError("不应调用矩形检测")):
            observed = detect_markers(image, changed["aruco"])
            result = estimate_marker_pose(list(observed.points.values()), .014, self.K, self.dist,
                                          self.installation, changed["pose"])
        self.assertEqual(result, baseline)
        original = copy.deepcopy(baseline)
        geometry = load_geometry(self.config)
        with tempfile.TemporaryDirectory() as directory:
            for depth in (.1, .9):
                truth = np.eye(4); truth[2, 3] = depth
                report = save_report(Path(directory), self.config, geometry, np.eye(4), truth, baseline, detection)
                self.assertEqual(report["active_candidate_indices"], baseline["active_candidate_indices"])
                self.assertEqual(report["selected_index"], baseline["selected_index"])
                self.assertEqual(report["T_camera_port"], baseline["T_camera_port"])
        self.assertEqual(original, baseline)

    def test_blank_wrong_id_and_duplicate_ids(self) -> None:
        for ids in ((), (1,)):
            with self.subTest(ids=ids):
                detection, result = solve(code_scene(self.settings, ids), self.config)
                self.assertFalse(detection.success)
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["T_camera_port"])
                self.assertEqual(len(detection.diagnostics["markers"]), len(ids))
        detection, result = solve(code_scene(self.settings, (0, 0)), self.config)
        self.assertEqual(len(detection.points), 2)
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["T_camera_port"])
        active = [result["candidates"][i] for i in result["active_candidate_indices"]]
        self.assertEqual({c["target_id"] for c in active}, {0, 1})

    def test_code_rotation_preserves_decoded_corner_identity(self) -> None:
        image = code_scene(self.settings, (0,))
        original = detect_markers(image, self.settings).points["marker_0"]
        rotated = detect_markers(cv2.rotate(image, cv2.ROTATE_180), self.settings).points["marker_0"]
        np.testing.assert_allclose(rotated, np.array([image.shape[1] - 1, image.shape[0] - 1]) - original, atol=.01)

    def test_invalid_and_missing_aruco_configuration(self) -> None:
        cases = [("marker_size_mm", 0), ("white_margin_mm", -1), ("marker_id", 50),
                 ("marker_id", True), ("dictionary", "BAD_DICTIONARY"),
                 ("center_port_mm", [0, 1]), ("rotation_port_deg", [90, "0", 0]),
                 ("corner_refinement_window_px", 1.5), ("output_directory", "")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "配置.yaml"
            for key, value in cases:
                with self.subTest(field=key, value=value):
                    config = copy.deepcopy(self.config)
                    config["aruco"][key] = value
                    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "aruco"):
                        load_config(path, require_aruco=True)
            config = copy.deepcopy(self.config)
            del config["aruco"]
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            self.assertNotIn("aruco", load_config(path))  # 原路线不要求这个分区。
            with self.assertRaisesRegex(ValueError, "aruco"):
                load_config(path, require_aruco=True)
            config["aruco"] = {**self.settings, "output_directory": "中文结果"}
            path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            self.assertEqual(load_config(path, require_aruco=True)["aruco"]["output_directory"], str(Path(directory) / "中文结果"))

    def test_degenerate_corners_fail_without_pnp_exception(self) -> None:
        result = estimate_marker_pose([np.zeros((4, 2))], .014, self.K, self.dist,
                                      self.installation, self.config["pose"])
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["T_camera_port"])


@unittest.skipUnless(os.environ.get("PORT_POSE_FULL_TESTS") == "1", "完整 ArUco 渲染验收")
class ArUcoIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = aruco_config()
        cls.geometry = load_geometry(cls.config)
        cls.K, cls.dist, cls.size = camera_parameters(cls.config)
        cls.truth = actual_pose(nominal_pose(cls.config), cls.config["simulation"])
        cls.patch = marker_patch(cls.config["aruco"])
        cls.rendered = render_scene(cls.geometry, cls.truth, cls.K, cls.dist, cls.size,
                                    cls.config, surface_patches=(cls.patch,))

    def test_default_oblique_rotated_and_distorted_views_meet_accuracy(self) -> None:
        views = [(150, 0, 25, 0, [0] * 5), (180, -15, 25, 0, [0] * 5),
                 (180, 15, 25, 0, [0] * 5), (180, 15, 25, 45, [0] * 5),
                 (150, 0, 25, 0, [.15, -.03, .001, -.001, 0])]
        for distance, azimuth, elevation, roll, distortion in views:
            with self.subTest(view=(distance, azimuth, elevation, roll, distortion)):
                config = copy.deepcopy(self.config)
                config["camera"].update(distance_mm=distance, azimuth_deg=azimuth, elevation_deg=elevation,
                                        roll_deg=roll, distortion=distortion)
                K, dist, size = camera_parameters(config)
                truth = actual_pose(nominal_pose(config), config["simulation"])
                rendered = render_scene(self.geometry, truth, K, dist, size, config, surface_patches=(self.patch,))
                detected, result = solve(rendered.image, config)
                self.assertEqual(result["status"], "ok", detected.reason)
                errors = evaluate_pose(result["T_camera_port"], truth)
                self.assertLessEqual(errors["rotation_error_deg"], 3, errors)
                self.assertLessEqual(errors["translation_error_mm"], 3, errors)

    def test_port_occluded_but_marker_visible_still_recovers_port(self) -> None:
        image = self.rendered.image.copy()
        port = project_points(self.geometry.landmarks["outer"], self.truth, self.K, self.dist)
        polygon = np.rint(port).astype(np.int32)
        cv2.fillConvexPoly(image, polygon, (130, 130, 130))
        detected, result = solve(image, self.config)
        self.assertTrue(detected.success)
        self.assertEqual(result["status"], "ok")
        self.assertLess(evaluate_pose(result["T_camera_port"], self.truth)["translation_error_mm"], 3)

    def test_partial_marker_occlusion_has_no_pose(self) -> None:
        image = self.rendered.image.copy()
        model = transform_points(marker_corners(.014), marker_transform(self.config["aruco"]))
        pixels = project_points(model, self.truth, self.K, self.dist)
        low = np.floor(pixels.min(axis=0) - 3).astype(int)
        high = np.ceil(pixels.max(axis=0) + 3).astype(int)
        image[low[1]:high[1], (low[0] + high[0]) // 2:high[0]] = 130
        detection, result = solve(image, self.config)
        self.assertFalse(detection.success)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["T_camera_port"])

    def test_low_angle_backside_out_of_frame_and_far_distance(self) -> None:
        views = [{"elevation_deg": 0}, {"elevation_deg": -25},
                 {"target_offset_mm": [300, 0, 0]}, {"distance_mm": 3000}]
        for view in views:
            with self.subTest(view=view):
                config = copy.deepcopy(self.config); config["camera"].update(view)
                truth = actual_pose(nominal_pose(config), config["simulation"])
                image = render_scene(self.geometry, truth, self.K, self.dist, self.size,
                                     config, surface_patches=(self.patch,)).image
                detection, result = solve(image, config)
                self.assertFalse(detection.success, detection.diagnostics)
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["T_camera_port"])

    def test_patch_depth_backface_and_seed(self) -> None:
        # 用完全出画的三角形和明确的单色遮挡板隔离测试 Z-buffer。
        # 真实加载器不接受空 STL，因此这里仍提供一个有效网格。
        geometry = Geometry(np.array([[10, 10, 1], [11, 10, 1], [10, 11, 1]], dtype=float),
                            np.array([[0, 1, 2]]), {}, np.eye(4), 1)
        config = copy.deepcopy(self.config); config["render"]["material_regions"] = []
        config["simulation"]["noise_std_gray"] = 0
        patch = marker_patch({**config["aruco"], "center_port_mm": [0, 0, 200], "rotation_port_deg": [180, 0, 0]})
        front = render_scene(geometry, np.eye(4), self.K, self.dist, self.size, config, surface_patches=(patch,))
        self.assertTrue(detect_markers(front.image, config["aruco"]).success)
        back = replace(patch, faces=patch.faces[:, ::-1])
        backside = render_scene(geometry, np.eye(4), self.K, self.dist, self.size, config, surface_patches=(back,))
        self.assertTrue(np.isinf(backside.depth).all())
        vertices = np.array([[-.02, -.02, .1], [.02, -.02, .1], [.02, .02, .1], [-.02, .02, .1]])
        occluder = Geometry(vertices, np.array([[0, 1, 2], [0, 2, 3]]), {}, np.eye(4), 1)
        covered = render_scene(occluder, np.eye(4), self.K, self.dist, self.size, config, surface_patches=(patch,))
        self.assertFalse(detect_markers(covered.image, config["aruco"]).success)
        self.assertAlmostEqual(float(covered.depth[360, 640]), .1, places=6)
        repeated = render_scene(self.geometry, self.truth, self.K, self.dist, self.size,
                                 self.config, surface_patches=(self.patch,))
        np.testing.assert_array_equal(repeated.image, self.rendered.image)

    def test_aruco_settings_alone_do_not_add_patch_to_old_renderer(self) -> None:
        without_settings = copy.deepcopy(self.config); del without_settings["aruco"]
        first = render_scene(self.geometry, self.truth, self.K, self.dist, self.size, self.config)
        second = render_scene(self.geometry, self.truth, self.K, self.dist, self.size, without_settings)
        np.testing.assert_array_equal(first.image, second.image)
        np.testing.assert_array_equal(first.depth, second.depth)
        self.assertFalse(detect_markers(first.image, self.config["aruco"]).success)

    def test_cli_success_failure_preview_and_configuration_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("success", "failed", "preview", "invalid"):
                with self.subTest(mode=mode):
                    config = copy.deepcopy(self.config)
                    if mode == "failed": config["camera"]["elevation_deg"] = 0
                    if mode == "invalid": config["aruco"]["marker_id"] = 999
                    config["aruco"]["output_directory"] = "配置输出"
                    path = root / "相机.yaml"
                    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
                    # preview 用 YAML 相对输出目录，其余模式用 CLI 覆盖。
                    output = root / ("配置输出" if mode == "preview" else mode)
                    flags = ["--preview-only"] if mode == "preview" else ["--output-dir", str(output)]
                    completed = subprocess.run([sys.executable, str(ROOT / "tests/fixtures/ArUco_demo.py"), "--config", str(path), *flags],
                                               cwd=root, capture_output=True, text=True, timeout=120)
                    code = {"success": 0, "failed": 2, "preview": 0, "invalid": 1}[mode]
                    self.assertEqual(completed.returncode, code, completed.stdout + completed.stderr)
                    self.assertNotIn("Traceback", completed.stderr)
                    if mode == "invalid":
                        self.assertIn("aruco.marker_id", completed.stderr)
                        continue
                    report = json.loads((output / "pose_result.json").read_text())
                    snapshot = yaml.safe_load((output / "actual_config.yaml").read_text())
                    self.assertEqual(snapshot["aruco"]["output_directory"], str(output))
                    self.assertEqual(report["status"], {"success": "ok", "failed": "failed", "preview": "preview_only"}[mode])
                    self.assertFalse(report["use_prior"])
                    for name in ("aruco_marker.png", "cad_preview.png", "simulated_rgb.png"):
                        self.assertTrue((output / name).is_file())
                    if mode != "preview":
                        for name in ("detected_features.png", "pose_comparison.png"):
                            self.assertTrue((output / name).is_file())
                    if mode == "failed": self.assertIsNone(report["T_camera_port"])


if __name__ == "__main__":
    unittest.main()
