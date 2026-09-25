"""Image-only adapter. No MuJoCo state or object truth enters the worker."""
from __future__ import annotations

import cv2
import numpy as np

from ur5e_sim.config import pose_resources


def detect_port(rgb, intrinsic, route, settings):
    from ur5e_sim.vision import aruco, pnp
    routes = {'aruco': aruco.detect, 'pnp': pnp.detect}
    if route not in routes:
        raise ValueError('Route must be aruco or pnp')
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    K = np.asarray(intrinsic, dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0,0], K[1,1]) <= 0:
        raise ValueError('Invalid camera intrinsic matrix')
    result = routes[route](bgr, K, settings)
    result.update(route=route, intrinsic_matrix=K.tolist(), distortion_coefficients=[0.0]*5)
    return result
