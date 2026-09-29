"""MuJoCo contact-based Xense gel depth maps and optional FEM sidecar IPC."""

from ur5e_sim.tactile.contact_map import (
    DEPTH_H,
    DEPTH_W,
    REST_DEPTH_MM,
    compute_dual_gel_depths,
    dual_tactile_canvas,
    gel_press_depth_mm,
    peak_press_mm,
)

__all__ = [
    "DEPTH_H",
    "DEPTH_W",
    "REST_DEPTH_MM",
    "compute_dual_gel_depths",
    "dual_tactile_canvas",
    "gel_press_depth_mm",
    "peak_press_mm",
]
