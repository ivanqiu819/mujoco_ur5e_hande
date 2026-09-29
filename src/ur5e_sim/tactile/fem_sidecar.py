"""Launch FemSensor sidecar (Python 3.11 xensim) for admittance demos."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ur5e_sim.paths import ROOT

# Parent xensim workspace (sibling of mujoco_ur5e_hande checkout).
WORKSPACE_ROOT = ROOT.parent
SIDECAR_SCRIPT = WORKSPACE_ROOT / "openpi/examples/libero/tactile_fem_sidecar.py"
DEFAULT_IPC = Path("/tmp/xensim_ur5e_tactile.npz")
DEFAULT_LOG = Path("/tmp/xensim_ur5e_sidecar.log")
DEFAULT_XENSIM_PY = Path.home() / "miniconda3/envs/xensim_py311/bin/python"


def resolve_xensim_python(explicit: str) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    for candidate in (
        DEFAULT_XENSIM_PY,
        Path(shutil.which("python3.11") or ""),
        Path.home() / "miniconda3/envs/xensim_py311/bin/python",
    ):
        if candidate and candidate.exists():
            return candidate
    return None


def write_ipc(
    ipc_file: Path,
    depth_l: np.ndarray,
    depth_r: np.ndarray,
    press_l: float,
    press_r: float,
    *,
    obj_pose_l: np.ndarray | None = None,
    sensor_pose_l: np.ndarray | None = None,
    obj_pose_r: np.ndarray | None = None,
    sensor_pose_r: np.ndarray | None = None,
) -> None:
    tmp = ipc_file.with_suffix(".tmp.npz")
    payload = dict(
        depth_l=depth_l.astype(np.float32),
        depth_r=depth_r.astype(np.float32),
        press_l_mm=np.float32(press_l),
        press_r_mm=np.float32(press_r),
    )
    if obj_pose_l is not None and sensor_pose_l is not None:
        payload["obj_pose_l"] = np.asarray(obj_pose_l, dtype=np.float32)
        payload["sensor_pose_l"] = np.asarray(sensor_pose_l, dtype=np.float32)
    if obj_pose_r is not None and sensor_pose_r is not None:
        payload["obj_pose_r"] = np.asarray(obj_pose_r, dtype=np.float32)
        payload["sensor_pose_r"] = np.asarray(sensor_pose_r, dtype=np.float32)
    np.savez(tmp, **payload)
    os.replace(tmp, ipc_file)


def start_sidecar(xensim_python: Path, ipc_file: Path) -> subprocess.Popen | None:
    if not SIDECAR_SCRIPT.exists():
        logging.warning("FEM sidecar script missing: %s", SIDECAR_SCRIPT)
        return None

    ipc_file.unlink(missing_ok=True)
    ipc_file.with_suffix(".tmp.npz").unlink(missing_ok=True)

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "glx")
    env.setdefault("QT_QPA_PLATFORM", "xcb")
    if "DISPLAY" not in env:
        logging.warning("DISPLAY unset; FEM sidecar may fail to open its window.")

    log_file = DEFAULT_LOG.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(xensim_python), str(SIDECAR_SCRIPT), str(ipc_file)],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(2.0)
    if proc.poll() is not None:
        log_file.close()
        if DEFAULT_LOG.exists():
            logging.warning("FEM sidecar exited early:\n%s", DEFAULT_LOG.read_text(encoding="utf-8"))
        return None

    logging.info("FEM sidecar running. Log: %s", DEFAULT_LOG)
    return proc


def stop_sidecar(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
