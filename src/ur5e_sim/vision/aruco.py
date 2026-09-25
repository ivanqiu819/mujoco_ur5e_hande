"""ArUco decoding and marker-to-port estimation; image input only."""
import numpy as np
from ur5e_sim.config import pose_resources
from .core.aruco import detect_markers, marker_transform
from .core.pose import estimate_marker_pose


def detect(bgr, K, settings):
    config, _ = pose_resources(settings, 'aruco')
    marker = config['aruco']
    observation = detect_markers(bgr, marker)
    result = estimate_marker_pose(list(observation.points.values()), marker['marker_size_mm']/1000,
                                  K, np.zeros(5), marker_transform(marker), config['pose'])
    result.update(diagnostics=observation.diagnostics, reason=observation.reason,
                  corner_refinement_window_px=marker['corner_refinement_window_px'])
    return result
