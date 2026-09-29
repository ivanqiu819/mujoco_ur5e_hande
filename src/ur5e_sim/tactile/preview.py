"""OpenCV depth preview and optional FEM sidecar IPC (no control loop)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from ur5e_sim.tactile.contact_map import (
    DEPTH_H,
    DEPTH_W,
    REST_DEPTH_MM,
    compute_dual_gel_depths,
    dual_tactile_canvas,
)
from ur5e_sim.tactile.fem_sidecar import (
    DEFAULT_IPC,
    resolve_xensim_python,
    start_sidecar,
    stop_sidecar,
    write_ipc,
)
from ur5e_sim.tactile.tactile_boundary import compute_dual_boundary

WINDOW = "Xense depth (gel–plug boundary)"


def model_has_gels(model: mujoco.MjModel) -> bool:
    for token in ("xense_gel_left", "xense_gel_right"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, token) < 0:
            return False
    return True


class TactilePreview:
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        use_fem_sidecar: bool = False,
        xensim_python: str = "",
        ipc_file: Path | None = None,
        show_window: bool = True,
        press_des: float = 1.2,
        tactile_hz: float = 20.0,
        tactile_physics: bool = True,
        debug_proxy_overlay: bool = False,
    ) -> None:
        if not model_has_gels(model):
            raise ValueError("Scene has no xense_gel_left/right geoms; build scene_inspection_tactile.xml")
        self.press_des = press_des
        self.show_window = show_window
        self._tactile_period = 1.0 / max(1.0, float(tactile_hz))
        self._last_tactile = 0.0
        self._cached_press = (0.0, 0.0)
        self.ipc_file = Path(ipc_file or DEFAULT_IPC)
        self._fem_proc = None
        self._use_fem = False
        self.tactile_physics = tactile_physics
        self.debug_proxy_overlay = debug_proxy_overlay

        if show_window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 420, 380)

        if use_fem_sidecar:
            py311 = resolve_xensim_python(xensim_python)
            if py311 is None:
                logging.warning("FEM sidecar requested but xensim_py311 not found.")
            else:
                self._fem_proc = start_sidecar(py311, self.ipc_file)
                self._use_fem = self._fem_proc is not None
                if self._use_fem:
                    rest = np.full((DEPTH_H, DEPTH_W), REST_DEPTH_MM, dtype=np.float32)
                    write_ipc(self.ipc_file, rest, rest, 0.0, 0.0)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
        now = time.monotonic()
        if now - self._last_tactile < self._tactile_period:
            return self._cached_press
        self._last_tactile = now

        obj_l = obj_r = sensor_l = sensor_r = None
        if self.tactile_physics:
            (
                depth_l,
                depth_r,
                press_l,
                press_r,
                obj_l,
                sensor_l,
                obj_r,
                sensor_r,
            ) = compute_dual_boundary(model, data)
        else:
            depth_l, depth_r, press_l, press_r = compute_dual_gel_depths(
                model, data, socket_proxy=True
            )

        self._cached_press = (press_l, press_r)
        if self.show_window:
            canvas = dual_tactile_canvas(depth_l, depth_r, press_l, press_r, self.press_des)
            if self.debug_proxy_overlay and self.tactile_physics:
                proxy_l, proxy_r, _, _ = compute_dual_gel_depths(model, data, socket_proxy=True)
                proxy = dual_tactile_canvas(proxy_l, proxy_r, press_l, press_r, self.press_des)
                canvas = np.vstack([canvas, proxy])
                cv2.putText(canvas, "L1 gel-plug", (8, canvas.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(canvas, "L0 proxy", (8, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(
                canvas,
                f"press max {max(press_l, press_r):.2f} mm",
                (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
            )
            cv2.imshow(WINDOW, canvas)
            cv2.waitKey(1)
        if self._use_fem and self._fem_proc is not None and self._fem_proc.poll() is None:
            write_ipc(
                self.ipc_file,
                depth_l,
                depth_r,
                press_l,
                press_r,
                obj_pose_l=obj_l,
                sensor_pose_l=sensor_l,
                obj_pose_r=obj_r,
                sensor_pose_r=sensor_r,
            )
        elif self._fem_proc is not None and self._fem_proc.poll() is not None:
            logging.warning("FEM sidecar exited; see /tmp/xensim_ur5e_sidecar.log")
            self._use_fem = False
            self._fem_proc = None
        return press_l, press_r

    def close(self) -> None:
        if self.show_window:
            cv2.destroyWindow(WINDOW)
        stop_sidecar(self._fem_proc)
        self._fem_proc = None
