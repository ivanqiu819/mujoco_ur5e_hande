"""无名义位姿模式：图像证据、对称歧义、候选保留和命令行集成。"""

from __future__ import annotations

import copy
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

from tests.vision_fixtures import ROOT, simulation_config
from ur5e_sim.vision.core.camera import actual_pose, camera_parameters, nominal_pose, project_points
from ur5e_sim.vision.core.geometry import load_geometry
from ur5e_sim.vision.core.image_only import estimate_from_image
from ur5e_sim.vision.core.pose import evaluate_pose
from ur5e_sim.vision.core.renderer import render_scene
from ur5e_sim.vision.core.reporting import save_report
from tests.fixtures.run_demo import parse_args


def rectangle_image(centers: list[tuple[int, int]], scale: float = 10) -> np.ndarray:
    """独立生成有正确内外尺寸比例的平面测试图，不使用投影或任何外参。"""
    image = np.full((720, 1280, 3), 130, np.uint8)
    for x, y in centers:
        cv2.rectangle(image, (int(x - 10 * scale), int(y - 4 * scale)),
                      (int(x + 10 * scale), int(y + 4 * scale)), (220, 220, 220), -1)
        cv2.rectangle(image, (int(x - 9 * scale), int(y - 3 * scale)),
                      (int(x + 9 * scale), int(y + 3 * scale)), (40, 40, 40), -1)
    return image


class ImageOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = simulation_config()
        cls.geometry = load_geometry(cls.config)
        cls.K, cls.dist, _ = camera_parameters(cls.config)

    def estimate(self, image: np.ndarray, config: dict | None = None) -> tuple:
        options = self.config if config is None else config
        return estimate_from_image(image, self.geometry.landmarks, self.K, self.dist,
                                   options["detection"], options["pose"])

    def test_boolean_cli_default_enable_disable(self) -> None:
        self.assertIs(parse_args([]).use_prior, False)  # 保留用户目前选择的无先验默认值。
        self.assertIs(parse_args(["--use-prior"]).use_prior, True)
        self.assertIs(parse_args(["--no-use-prior"]).use_prior, False)

    def test_same_image_does_not_depend_on_external_camera_or_prior_settings(self) -> None:
        image = rectangle_image([(300, 350)])
        _, baseline = self.estimate(image)
        self.assertEqual(baseline["status"], "ambiguous")
        changed = copy.deepcopy(self.config)
        # 若无先验流程错误地读取这些参数，候选应会明显改变或直接失败。
        changed["camera"].update(distance_mm=9000, azimuth_deg=180, target_offset_mm=[999, 999, 999])
        changed["simulation"]["translation_offset_mm"] = [999, 999, 999]
        changed["pose"].update(prior_rotation_weight=1e9, prior_translation_weight=1e9)
        changed["detection"].update(roi_margin_px=0, max_corner_shift_px=0)
        with mock.patch("ur5e_sim.vision.core.camera.nominal_pose", side_effect=AssertionError("不允许读取名义位姿")), \
             mock.patch("ur5e_sim.vision.core.camera.actual_pose", side_effect=AssertionError("不允许读取实际位姿")), \
             mock.patch("ur5e_sim.vision.core.detection.detect_features", side_effect=AssertionError("不允许调用先验检测器")):
            _, moved = self.estimate(image, changed)
            del changed["camera"], changed["simulation"]
            _, removed = self.estimate(image, changed)
        self.assertEqual(baseline, moved)
        self.assertEqual(baseline, removed)

    def test_multiple_targets_preserved_even_with_one_at_image_center(self) -> None:
        detected, result = self.estimate(rectangle_image([(250, 250), (640, 360)]))
        self.assertEqual(detected.diagnostics["target_count"], 2)
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["T_camera_port"])
        self.assertIsNone(result["selected_index"])
        active = [result["candidates"][i] for i in result["active_candidate_indices"]]
        self.assertEqual({c["target_id"] for c in active}, {0, 1})
        for candidate in active:
            self.assertIn("correspondence", candidate)
            self.assertIn("points_px", candidate)
            self.assertNotIn("errors_vs_truth", candidate)

    def test_similar_solid_rectangle_is_not_a_port_and_does_not_hide_off_center_port(self) -> None:
        image = rectangle_image([(250, 250)])
        cv2.rectangle(image, (540, 320), (740, 400), (40, 40, 40), -1)
        detected, result = self.estimate(image)
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(detected.diagnostics["target_count"], 1)
        center = np.mean(detected.diagnostics["targets"][0]["inner_px"], axis=0)
        np.testing.assert_allclose(center, [250, 250], atol=1)

    def test_blank_occluded_small_and_cropped_images_fail_without_pose(self) -> None:
        occluded = rectangle_image([(300, 350)])
        occluded[290:410, 300:420] = 130
        images = {"blank": rectangle_image([]), "too_small": rectangle_image([(300, 350)], 1),
                  "occluded": occluded, "out_of_frame": rectangle_image([(20, 350)])}
        for name, image in images.items():
            with self.subTest(image=name):
                detected, result = self.estimate(image)
                self.assertFalse(detected.success)
                self.assertTrue(detected.reason)
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["T_camera_port"])
                self.assertIsNone(result["selected_index"])

    def test_truth_evaluation_does_not_change_ranking_or_select_a_candidate(self) -> None:
        detected, result = self.estimate(rectangle_image([(300, 350)]))
        original = copy.deepcopy(result)
        config = copy.deepcopy(self.config)
        config["runtime"] = {"use_prior": False}
        with tempfile.TemporaryDirectory() as directory:
            for index in result["active_candidate_indices"]:
                # 故意将不同候选轮流设为“真值”；报告仍不得改变任何选择。
                truth = np.asarray(result["candidates"][index]["T_camera_port"])
                report = save_report(Path(directory), config, self.geometry, np.eye(4), truth, result, detected)
                self.assertEqual(report["active_candidate_indices"], original["active_candidate_indices"])
                self.assertEqual(report["status"], "ambiguous")
                self.assertIsNone(report["selected_index"])
                self.assertIsNone(report["T_camera_port"])
                self.assertAlmostEqual(report["candidates"][index]["errors_vs_truth"]["translation_error_mm"], 0)
        self.assertEqual(result, original)


@unittest.skipUnless(os.environ.get("PORT_POSE_FULL_TESTS") == "1", "完整 STL 渲染和 CLI 验收")
class ImageOnlyIntegrationTests(unittest.TestCase):
    def test_fixed_180mm_views_and_off_center_target_meet_accuracy(self) -> None:
        config = simulation_config()
        geometry = load_geometry(config)
        K, dist, size = camera_parameters(config)
        views = [(15, 15, 0, [0, 0, 0]), (-15, 15, 0, [0, 0, 0]),
                 (0, 20, 0, [0, 0, 0]), (30, 20, 0, [0, 0, 0]),
                 (15, 15, 45, [0, 0, 0]), (15, 15, 0, [40, 15, 0])]
        for azimuth, elevation, roll, target in views:
            with self.subTest(azimuth=azimuth, elevation=elevation, roll=roll, target=target):
                config["camera"].update(azimuth_deg=azimuth, elevation_deg=elevation,
                                        roll_deg=roll, target_offset_mm=target)
                truth = actual_pose(nominal_pose(config), config["simulation"])
                image = render_scene(geometry, truth, K, dist, size, config).image
                detected, result = estimate_from_image(image, geometry.landmarks, K, dist,
                                                       config["detection"], config["pose"])
                self.assertEqual(result["status"], "ok", detected.reason)
                errors = evaluate_pose(result["T_camera_port"], truth)
                self.assertLessEqual(errors["rotation_error_deg"], 3.0, errors)
                self.assertLessEqual(errors["translation_error_mm"], 3.0, errors)
                selected = result["candidates"][result["selected_index"]]
                self.assertTrue({8, 9}.issubset(selected["inlier_indices"]))

    def test_stl_partial_occlusion_has_no_invented_pose(self) -> None:
        config = simulation_config()
        geometry = load_geometry(config)
        K, dist, size = camera_parameters(config)
        truth = actual_pose(nominal_pose(config), config["simulation"])
        image = render_scene(geometry, truth, K, dist, size, config).image
        projected = project_points(geometry.landmarks["outer"], truth, K, dist)
        x0, y0 = np.floor(projected.min(axis=0) - 10).astype(int)
        x1, y1 = np.ceil(projected.max(axis=0) + 10).astype(int)
        image[y0:y1, (x0 + x1) // 2:x1] = config["render"]["background_gray"]
        detected, result = estimate_from_image(image, geometry.landmarks, K, dist,
                                               config["detection"], config["pose"])
        self.assertFalse(detected.success)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["T_camera_port"])

    def test_cli_modes_same_50mm_image_and_preserved_front_symmetry(self) -> None:
        config = simulation_config()
        config["camera"].update(distance_mm=50, azimuth_deg=0, elevation_deg=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "固定正视.yaml"
            path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            reports, images = {}, {}
            for mode, flag in (("default", []), ("with_prior", ["--use-prior"]),
                               ("image_only", ["--no-use-prior"])):
                uses_prior = mode == "with_prior"
                output = root / mode
                completed = subprocess.run([sys.executable, str(ROOT / "tests/fixtures/run_demo.py"),
                                            "--config", str(path), "--output-dir", str(output), *flag],
                                           cwd=root, capture_output=True, text=True, timeout=120)
                self.assertEqual(completed.returncode, 0 if uses_prior else 2,
                                 completed.stdout + completed.stderr)
                reports[mode] = json.loads((output / "pose_result.json").read_text(encoding="utf-8"))
                snapshot = yaml.safe_load((output / "actual_config.yaml").read_text(encoding="utf-8"))
                self.assertEqual(snapshot["runtime"]["use_prior"], uses_prior)
                self.assertEqual(reports[mode]["use_prior"], uses_prior)
                images[mode] = (output / "simulated_rgb.png").read_bytes()
                for name in ("detected_features.png", "detection_edges.png", "pose_comparison.png"):
                    self.assertTrue((output / name).is_file())
                if mode == "image_only":
                    self.assertIn("位姿存在歧义", completed.stdout)
            self.assertEqual(images["default"], images["with_prior"])
            self.assertEqual(images["default"], images["image_only"])
            self.assertEqual(reports["default"]["candidates"], reports["image_only"]["candidates"])
            result = reports["image_only"]
            self.assertEqual(result["status"], "ambiguous")
            self.assertIsNone(result["T_camera_port"])
            self.assertIsNone(result["selected_index"])
            active = [result["candidates"][i] for i in result["active_candidate_indices"]]
            self.assertGreaterEqual(len(active), 2)
            # 保留一个接近实际朝向和一个相反朝向；不能在检测时依赖真值二选一。
            rotations = [c["errors_vs_truth"]["rotation_error_deg"] for c in active]
            self.assertLess(min(rotations), 3)
            self.assertGreater(max(rotations), 170)


if __name__ == "__main__":
    unittest.main()
