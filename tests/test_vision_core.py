"""快速数值测试：不渲染整幅装配体，不依赖 pytest。

运行：python -m unittest discover -s tests -v
真实照片的识别不在这些测试的保证范围内。
"""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import yaml

from ur5e_sim.vision.core.camera import (
    actual_pose,
    camera_parameters,
    nominal_pose,
    pose_vectors,
    project_points,
    transform_from_vectors,
)
from tests.vision_fixtures import simulation_config
from ur5e_sim.vision.core.config import load_config
from ur5e_sim.vision.core.geometry import load_geometry, transform_points
from ur5e_sim.vision.core.pose import estimate_pose, evaluate_pose


from ur5e_sim.paths import ROOT
DEFAULT_CONFIG = ROOT / "tests/fixtures/socket.yaml"


class GeometryAndCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = simulation_config()
        cls.geometry = load_geometry(cls.config)

    def test_socket_landmarks_are_in_meters_and_right_handed(self) -> None:
        geometry = self.geometry
        np.testing.assert_allclose(geometry.landmarks["inner"][0], [-0.009, -0.003, 0])
        np.testing.assert_allclose(geometry.landmarks["outer"][2], [0.010, 0.004, 0])
        np.testing.assert_allclose(geometry.landmarks["top_rear"],
                                   [[-0.010, -0.004, 0.060], [0.010, -0.004, 0.060]], atol=1e-14)
        np.testing.assert_allclose(geometry.T_stl_port[:3, :3], np.diag([1, -1, -1]))
        np.testing.assert_allclose(geometry.T_stl_port[:3, 3], [0.080, 0.020, 0.115])
        self.assertAlmostEqual(np.linalg.det(geometry.T_stl_port[:3, :3]), 1.0)
        self.assertEqual(geometry.unit_scale, 0.001)

    def test_mm_cm_and_m_files_describe_the_same_physical_geometry(self) -> None:
        """同一个实物用三种 STL 数值单位存储，加载后必须完全等价。"""
        model_vertices_mm = np.array([[70, 16, 55], [90, 16, 55], [70, 24, 115]], float)
        reference_vertices = reference_landmarks = None
        with tempfile.TemporaryDirectory() as directory:
            for unit, divisor in (("mm", 1.0), ("cm", 10.0), ("m", 1000.0)):
                with self.subTest(unit=unit):
                    vertices = model_vertices_mm / divisor
                    path = Path(directory) / f"单位验证_{unit}.STL"
                    vertex_lines = "\n".join("vertex " + " ".join(map(str, row)) for row in vertices)
                    path.write_text("solid test\nfacet normal 0 0 1\nouter loop\n" + vertex_lines
                                    + "\nendloop\nendfacet\nendsolid test\n", encoding="utf-8")
                    config = copy.deepcopy(self.config)
                    config["model"].update(path=str(path), unit=unit)
                    for key in ("center", "inner_size", "outer_size", "top_rear_points"):
                        config["port"][key] = (np.asarray(config["port"][key]) / divisor).tolist()
                    geometry = load_geometry(config)
                    if reference_vertices is None:
                        reference_vertices = geometry.vertices
                        reference_landmarks = geometry.landmarks
                    else:
                        np.testing.assert_allclose(geometry.vertices, reference_vertices, atol=1e-12)
                        for name in geometry.landmarks:
                            np.testing.assert_allclose(geometry.landmarks[name], reference_landmarks[name], atol=1e-12)

    def test_mesh_coordinate_round_trip(self) -> None:
        points = self.geometry.vertices[::max(1, len(self.geometry.vertices) // 100)]
        stl_meters = transform_points(points, self.geometry.T_stl_port)
        restored = transform_points(stl_meters, np.linalg.inv(self.geometry.T_stl_port))
        np.testing.assert_allclose(restored, points, atol=1e-14)

    def test_camera_angles_have_documented_directions(self) -> None:
        for azimuth, elevation in ((0, 0), (25, 0), (-25, 0), (0, 25), (0, -25)):
            with self.subTest(azimuth=azimuth, elevation=elevation):
                config = copy.deepcopy(self.config)
                config["camera"].update(azimuth_deg=azimuth, elevation_deg=elevation)
                transform = nominal_pose(config)
                center = np.linalg.inv(transform)[:3, 3]
                self.assertAlmostEqual(np.linalg.norm(center), 0.180)
                self.assertLess(center[2], 0)
                if azimuth:
                    self.assertGreater(center[0] * azimuth, 0)
                if elevation:
                    self.assertLess(center[1] * elevation, 0)
                np.testing.assert_allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-14)

    def test_front_projection_and_positive_roll(self) -> None:
        config = copy.deepcopy(self.config)
        config["camera"].update(azimuth_deg=0, elevation_deg=0, roll_deg=0)
        K, dist, size = camera_parameters(config)
        self.assertEqual(size, (1280, 720))
        points = np.array([[0, 0, 0], [0.009, 0, 0], [0, 0.003, 0]])
        expected = np.array([[640, 360], [690, 360], [640, 360 + 1000 * 0.003 / 0.180]])
        np.testing.assert_allclose(project_points(points, nominal_pose(config), K, dist), expected)
        config["camera"]["roll_deg"] = 90
        rotated = project_points(points, nominal_pose(config), K, dist)
        np.testing.assert_allclose(rotated[1], [640, 410], atol=1e-10)

    def test_target_offset_is_looked_at_and_distance_is_preserved(self) -> None:
        config = copy.deepcopy(self.config)
        config["camera"]["target_offset_mm"] = [4, -2, 8]
        target = np.array([[0.004, -0.002, 0.008]])
        transform = nominal_pose(config)
        K, dist, _ = camera_parameters(config)
        center = np.linalg.inv(transform)[:3, 3]
        self.assertAlmostEqual(np.linalg.norm(center - target[0]), 0.180)
        np.testing.assert_allclose(project_points(target, transform, K, dist), [[640, 360]], atol=1e-10)

    def test_distortion_projection_agrees_with_analytic_radial_case(self) -> None:
        config = copy.deepcopy(self.config)
        config["camera"].update(azimuth_deg=0, elevation_deg=0, roll_deg=0,
                                 distortion=[0.2, 0, 0, 0, 0])
        K, dist, _ = camera_parameters(config)
        point = np.array([[0.030, 0.010, 0]])
        normalized = point[0, :2] / 0.180
        expected = normalized * (1 + 0.2 * np.dot(normalized, normalized)) * 1000 + [640, 360]
        np.testing.assert_allclose(project_points(point, nominal_pose(config), K, dist)[0], expected, atol=1e-10)

    def test_pose_vector_round_trip_and_local_translation_perturbation(self) -> None:
        nominal = nominal_pose(self.config)
        truth = actual_pose(nominal, self.config["simulation"])
        np.testing.assert_allclose(transform_from_vectors(*pose_vectors(truth)), truth, atol=1e-14)
        delta = np.linalg.inv(nominal) @ truth
        np.testing.assert_allclose(delta[:3, 3], [0.0004, -0.0003, 0.0008], atol=1e-14)


class PoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = simulation_config()
        cls.geometry = load_geometry(cls.config)
        cls.K, cls.dist, _ = camera_parameters(cls.config)
        cls.nominal = nominal_pose(cls.config)
        cls.truth = actual_pose(cls.nominal, cls.config["simulation"])

    def perfect_detections(self) -> dict[str, np.ndarray]:
        return {name: project_points(points, self.truth, self.K, self.dist)
                for name, points in self.geometry.landmarks.items()}

    def assert_precise_pose(self, transform: np.ndarray) -> None:
        errors = evaluate_pose(transform, self.truth)
        self.assertLess(errors["rotation_error_deg"], 1e-3)
        self.assertLess(errors["translation_error_mm"], 1e-3)

    def test_known_pose_recovered_without_renderer_or_detector(self) -> None:
        cv2.setRNGSeed(42)
        result = estimate_pose(self.geometry.landmarks, self.perfect_detections(),
                               self.K, self.dist, self.nominal, self.config)
        self.assertEqual(result["status"], "ok", result)
        self.assert_precise_pose(result["T_camera_port"])

    def test_without_top_features_returns_planar_candidates(self) -> None:
        detected = self.perfect_detections()
        del detected["top_rear"]
        result = estimate_pose(self.geometry.landmarks, detected, self.K, self.dist, self.nominal, self.config)
        self.assertEqual(result["status"], "planar_only")
        self.assertGreaterEqual(len(result["planar"]["candidates"]), 1)
        self.assertTrue(result["warnings"])
        self.assert_precise_pose(result["T_camera_port"])

    def test_outlier_is_removed_and_lm_receives_only_ransac_inliers(self) -> None:
        """外点应被排除，而不是在 LM 阶段又重新混入。"""
        detected = self.perfect_detections()
        detected["outer"][2] += [45, -35]
        cv2.setRNGSeed(42)
        original_refine = cv2.solvePnPRefineLM
        with mock.patch("ur5e_sim.vision.core.pose.cv2.solvePnPRefineLM", wraps=original_refine) as refine:
            result = estimate_pose(self.geometry.landmarks, detected, self.K, self.dist, self.nominal, self.config)
        self.assertEqual(result["status"], "ok", result)
        indices = result["nonplanar"]["inlier_indices"]
        self.assertNotIn(6, indices)  # inner 四点后，outer 的第三点。
        self.assertLess(len(indices), 10)
        refine.assert_called_once()
        object_points, image_points = refine.call_args.args[:2]
        all_objects = np.vstack([self.geometry.landmarks[name] for name in ("inner", "outer", "top_rear")])
        all_pixels = np.vstack([detected[name] for name in ("inner", "outer", "top_rear")])
        np.testing.assert_array_equal(object_points, all_objects[indices])
        np.testing.assert_array_equal(image_points, all_pixels[indices])
        self.assert_precise_pose(result["T_camera_port"])

    def test_no_detections_cannot_produce_success_or_truth_pose(self) -> None:
        result = estimate_pose(self.geometry.landmarks, {}, self.K, self.dist, self.nominal, self.config)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["T_camera_port"])


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(DEFAULT_CONFIG)

    def load_temporary_config(self, config: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "临时配置.yaml"
            path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            return load_config(path)

    def test_paths_are_resolved_against_configuration_directory(self) -> None:
        self.assertEqual(Path(self.config["model"]["path"]), ROOT / "assets/meshes/inspection_workpiece.stl")
        self.assertEqual(Path(self.config["output"]["directory"]), ROOT / "outputs")

    def test_invalid_configs_report_relevant_fields(self) -> None:
        cases = [
            ("camera", "distance_mm", 0, "camera.distance_mm"),
            ("camera", "image_size", [1280.5, 720], "camera.image_size"),
            ("camera", "distortion", [0, 0], "camera.distortion"),
            ("model", "unit", "inch", "model.unit"),
            ("model", "path", str(ROOT / "不存在.STL"), "STL"),
            ("port", "inner_size", [30, 9], "port.outer_size"),
            ("simulation", "noise_std_gray", -1, "simulation.noise_std_gray"),
        ]
        for section, key, value, expected in cases:
            with self.subTest(field=f"{section}.{key}"):
                config = copy.deepcopy(self.config)
                config[section][key] = value
                with self.assertRaisesRegex(ValueError, expected):
                    self.load_temporary_config(config)

    def test_parallel_orientation_vectors_are_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["port"]["up"] = [0, 0, 1]
        with self.assertRaisesRegex(ValueError, "不能平行"):
            load_geometry(config)

    def test_input_type_and_render_size_regressions_report_configuration_fields(self) -> None:
        """这些输入曾通过局部验证，随后以 TypeError 或 OpenCV 断言退出。"""
        cases = [
            ("camera", "elevation_deg", "15", "camera.elevation_deg"),
            ("camera", "distance_mm", True, "camera.distance_mm"),
            ("camera", "target_offset_mm", [0, "0", 0], "camera.target_offset_mm"),
            ("render", "ambient", "0.45", "render.ambient"),
            ("render", "supersample", True, "render.supersample"),
            ("render", "supersample", 2, "render.supersample"),
            ("port", "top_rear_points", {"left": [70, 24, 55], "right": [90, 24, 55]}, "port.top_rear_points"),
            ("port", "top_rear_points", [[70, 24, 55], [90, "24", 55]], "port.top_rear_points"),
            ("camera", "image_size", [16, 32768], "camera.image_size"),
            # 最终图像尚未超过限制，但三倍采样的中间图像已经超过。
            ("camera", "image_size", [12000, 720], "render.supersample"),
        ]
        for section, key, value, expected in cases:
            with self.subTest(field=f"{section}.{key}", value=value):
                config = copy.deepcopy(self.config)
                config[section][key] = value
                with self.assertRaisesRegex(ValueError, expected):
                    self.load_temporary_config(config)

    def test_missing_section_and_invalid_yaml_are_reported(self) -> None:
        del self.config["camera"]
        with self.assertRaisesRegex(ValueError, "camera"):
            self.load_temporary_config(self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("camera: [", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "YAML"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
