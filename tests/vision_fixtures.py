"""固定验收机位：日常修改 YAML 的拍摄角度不会改变测试条件。"""

from pathlib import Path

from ur5e_sim.vision.core.config import load_config


from ur5e_sim.paths import ROOT


def simulation_config() -> dict:
    config = load_config(ROOT / "tests/fixtures/socket.yaml")
    config["camera"] = {
        "distance_mm": 180.0, "azimuth_deg": 15.0, "elevation_deg": 15.0,
        "roll_deg": 0.0, "target_offset_mm": [0, 0, 0], "image_size": [1280, 720],
        "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0},
        "distortion": [0, 0, 0, 0, 0],
    }
    config["simulation"].update(rotation_offset_deg=[1.0, -1.0, 0.5],
                                translation_offset_mm=[0.4, -0.3, 0.8],
                                noise_std_gray=1.4, random_seed=20260917)
    return config


def aruco_config() -> dict:
    """ArUco 验收固定为 150 mm、25°，不随用户日常修改安装参数而变化。"""
    config = simulation_config()
    config["camera"].update(distance_mm=150, azimuth_deg=0, elevation_deg=25)
    config["aruco"] = {
        "dictionary": "DICT_4X4_50", "marker_id": 0, "marker_size_mm": 14.0,
        "white_margin_mm": 2.0, "center_port_mm": [0, -4.05, 16],
        "rotation_port_deg": [90, 0, 0], "corner_refinement_window_px": 3,
        "min_marker_side_px": 12, "output_directory": str(ROOT / "outputs" / "aruco"),
    }
    return config
