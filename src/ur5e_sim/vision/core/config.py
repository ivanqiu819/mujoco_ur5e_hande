"""读取带注释的 YAML，并在运行前报告可操作的配置错误。

配置保持为普通字典，便于在各模块之间阅读和修改；所有路径在这里
一次性解析，数值的单位转换则由负责相应物理量的模块完成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    """验证一维有限数值，错误中保留 YAML 字段名。"""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} 必须是 {length} 个数值组成的列表")
    for item in value:
        _number(item, name)
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 {length} 个数值") from exc
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是 {length} 个有限数值")
    return array


def _number(value: Any, name: str, minimum: float | None = None) -> float:
    # 不把带引号的字符串或 YAML 布尔量当作数值，避免验证通过后别处类型错误。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值，请勿加引号或填写 true/false")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值") from exc
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} 必须是有限数值" + (f"且 >= {minimum}" if minimum is not None else ""))
    return result


def load_config(path: str | Path, *, require_aruco: bool = False) -> dict[str, Any]:
    """读取配置，返回独立字典；相对路径始终以配置文件的目录为基准。"""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"配置文件不存在：{path}")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 格式错误：{exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("配置根节点必须是键值表，请参考 configs/socket.yaml")
    if "cad" in config:
        raise ValueError("这是旧版参数化模型配置；请使用 configs/socket.yaml 配置 STL")

    required = {
        "model": ("path", "unit"),
        "port": ("center", "normal", "up", "inner_size", "outer_size"),
        "camera": ("distance_mm", "azimuth_deg", "elevation_deg", "roll_deg", "image_size", "intrinsics", "distortion"),
        "simulation": ("rotation_offset_deg", "translation_offset_mm", "noise_std_gray", "random_seed"),
        "render": ("background_gray", "default_gray", "ambient", "light_direction_camera", "near_clip_mm", "material_regions"),
        "detection": ("roi_margin_px", "search_band_px", "canny_low", "canny_high", "min_port_width_px", "min_port_height_px", "max_corner_shift_px"),
        "pose": ("reprojection_threshold_px", "prior_rotation_weight", "prior_translation_weight"),
        "output": ("directory",),
    }
    for section, keys in required.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"缺少配置分区：{section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"缺少配置字段：{section}.{key}")

    model = config["model"]
    if model["unit"] not in ("mm", "cm", "m"):
        raise ValueError("model.unit 只支持 mm、cm 或 m；STL 本身不记录单位")
    for section, key in (("model", "path"), ("output", "directory")):
        value = config[section][key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{section}.{key} 必须是非空路径")
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = path.parent / resolved
        config[section][key] = str(resolved.resolve())
    if not Path(model["path"]).is_file():
        raise ValueError(f"STL 文件不存在：{model['path']}；相对路径以配置文件所在目录为准")
    if Path(model["path"]).suffix.lower() != ".stl":
        raise ValueError("本示例的 model.path 需要指向 .stl 文件")

    port = config["port"]
    for key in ("center", "normal", "up"):
        _vector(port[key], 3, f"port.{key}")
    inner = _vector(port["inner_size"], 2, "port.inner_size")
    outer = _vector(port["outer_size"], 2, "port.outer_size")
    if np.any(inner <= 0) or np.any(outer <= inner):
        raise ValueError("port.outer_size 的宽和高必须分别大于 inner_size，且内口尺寸为正")
    rear = port.setdefault("top_rear_points", [])
    if not isinstance(rear, list) or len(rear) not in (0, 2):
        raise ValueError("port.top_rear_points 必须为 [] 或左后、右后两个三维点")
    for point in rear:
        _vector(point, 3, "port.top_rear_points")

    camera = config["camera"]
    size = _vector(camera["image_size"], 2, "camera.image_size")
    if np.any(size < 16) or np.any(size != np.floor(size)):
        raise ValueError("camera.image_size 的宽、高必须是至少 16 的整数")
    if _number(camera["distance_mm"], "camera.distance_mm", 0) == 0:
        raise ValueError("camera.distance_mm 必须大于零")
    for key in ("azimuth_deg", "elevation_deg", "roll_deg"):
        _number(camera[key], f"camera.{key}")
    if abs(camera["elevation_deg"]) >= 89.9:
        raise ValueError("camera.elevation_deg 必须在 (-89.9, 89.9) 度之间，避免轨道极点")
    _vector(camera.setdefault("target_offset_mm", [0, 0, 0]), 3, "camera.target_offset_mm")
    _vector(camera["distortion"], 5, "camera.distortion")
    intrinsics = camera["intrinsics"]
    if not isinstance(intrinsics, dict) or not all(k in intrinsics for k in ("fx", "fy", "cx", "cy")):
        raise ValueError("camera.intrinsics 需要 fx、fy、cx、cy 四项")
    for key in ("fx", "fy", "cx", "cy"):
        value = _number(intrinsics[key], f"camera.intrinsics.{key}")
        if key in ("fx", "fy") and value <= 0:
            raise ValueError(f"camera.intrinsics.{key} 必须大于零")

    simulation = config["simulation"]
    for key in ("rotation_offset_deg", "translation_offset_mm"):
        _vector(simulation[key], 3, f"simulation.{key}")
    _number(simulation["noise_std_gray"], "simulation.noise_std_gray", 0)
    seed = _number(simulation["random_seed"], "simulation.random_seed", 0)
    if seed != int(seed):
        raise ValueError("simulation.random_seed 必须是非负整数")

    render = config["render"]
    sample = render.setdefault("supersample", 3)
    _number(sample, "render.supersample")
    if sample not in (1, 3) or not isinstance(sample, int):
        raise ValueError("render.supersample 只支持整数 1 或 3")
    if np.any(size * sample >= 32767):
        raise ValueError("camera.image_size × render.supersample 的每个维度必须小于 32767（OpenCV 限制）")
    for key in ("background_gray", "default_gray"):
        value = _number(render[key], f"render.{key}", 0)
        if value > 255:
            raise ValueError(f"render.{key} 必须在 0 到 255 之间")
    if not 0 <= _number(render["ambient"], "render.ambient") <= 1:
        raise ValueError("render.ambient 必须在 0 到 1 之间")
    if _number(render["near_clip_mm"], "render.near_clip_mm", 0) == 0:
        raise ValueError("render.near_clip_mm 必须大于零")
    if np.linalg.norm(_vector(render["light_direction_camera"], 3, "render.light_direction_camera")) == 0:
        raise ValueError("render.light_direction_camera 不能是零向量")
    if not isinstance(render["material_regions"], list):
        raise ValueError("render.material_regions 必须是列表（可为空）")
    for index, region in enumerate(render["material_regions"]):
        label = f"render.material_regions[{index}]"
        if not isinstance(region, dict) or not all(k in region for k in ("min", "max", "gray")):
            raise ValueError(f"{label} 需要 min、max 和 gray")
        lo = _vector(region["min"], 3, label + ".min")
        hi = _vector(region["max"], 3, label + ".max")
        if np.any(hi < lo) or not 0 <= _number(region["gray"], label + ".gray") <= 255:
            raise ValueError(f"{label} 的 min/max 或灰度范围无效")
    for key, value in config["detection"].items():
        _number(value, f"detection.{key}", 0)
    if config["detection"]["canny_low"] >= config["detection"]["canny_high"]:
        raise ValueError("detection.canny_low 必须小于 canny_high")
    config["pose"].setdefault("ambiguity_margin_px", 0.25)
    for key, value in config["pose"].items():
        _number(value, f"pose.{key}", 0)
    config["source_config"] = str(path)
    # 原路线不依赖 ArUco 配置或 cv2.aruco；只在新入口中做专项验证。
    if require_aruco:
        _validate_aruco(config, path.parent)
    return config


def _validate_aruco(config: dict[str, Any], directory: Path) -> None:
    """验证贴纸的物理尺寸、安装关系及检测参数；路径相对于 YAML。"""
    settings = config.get("aruco")
    if not isinstance(settings, dict):
        raise ValueError("ArUco_demo.py 需要 aruco 配置分区，请参考 configs/socket.yaml")
    for key in ("dictionary", "marker_id", "marker_size_mm", "white_margin_mm",
                "center_port_mm", "rotation_port_deg", "output_directory"):
        if key not in settings:
            raise ValueError(f"缺少配置字段：aruco.{key}")
    for key in ("marker_size_mm", "white_margin_mm"):
        if _number(settings[key], f"aruco.{key}", 0) == 0:
            raise ValueError(f"aruco.{key} 必须大于零")
    for key in ("center_port_mm", "rotation_port_deg"):
        _vector(settings[key], 3, f"aruco.{key}")
    for key, default in (("corner_refinement_window_px", 3), ("min_marker_side_px", 12)):
        value = _number(settings.setdefault(key, default), f"aruco.{key}", 1)
        if key == "corner_refinement_window_px" and value != int(value):
            raise ValueError("aruco.corner_refinement_window_px 必须是正整数")
    marker_id = _number(settings["marker_id"], "aruco.marker_id", 0)
    if marker_id != int(marker_id):
        raise ValueError("aruco.marker_id 必须是整数")
    # 惰性导入：即使环境不支持 ArUco，也不影响原矩形检测路线。
    from .aruco import get_dictionary
    dictionary = get_dictionary(settings["dictionary"])
    if marker_id >= len(dictionary.bytesList):
        raise ValueError(f"aruco.marker_id 超出字典范围：0 到 {len(dictionary.bytesList) - 1}")
    value = settings["output_directory"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("aruco.output_directory 必须是非空路径")
    path = Path(value).expanduser()
    settings["output_directory"] = str((path if path.is_absolute() else directory / path).resolve())
