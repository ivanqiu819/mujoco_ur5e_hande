"""Prior-free image edge/corner PnP; no marker configuration required."""
import numpy as np
from ur5e_sim.config import pose_resources
from .core.image_only import estimate_from_image


def detect(bgr, K, settings):
    config, geometry = pose_resources(settings, 'pnp')
    observation, result = estimate_from_image(bgr, geometry.landmarks, K, np.zeros(5),
                                              config['detection'], config['pose'])
    result.update(diagnostics=observation.diagnostics, reason=observation.reason)
    return result
