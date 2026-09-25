#!/usr/bin/env python3
"""Exercise the coupled Robotiq Hand-E sliders in MuJoCo."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import time

import mujoco


from ur5e_sim.paths import ROOT as PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=PROJECT_ROOT / "scenes/scene.xml")
    parser.add_argument(
        "--position",
        type=float,
        default=0.005,
        help="slider target in metres, range [-0.025, 0.025] (home is 0.020)",
    )
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    if not -0.025 <= args.position <= 0.025:
        raise ValueError("--position must be in [-0.025, 0.025] metres")

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    slider_1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Slider_1")
    slider_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Slider_2")
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    if min(home_id, slider_1, slider_2, gripper) < 0:
        raise RuntimeError("Required home/gripper elements are missing")

    mujoco.mj_resetDataKeyframe(model, data, home_id)
    qadr_1 = int(model.jnt_qposadr[slider_1])
    qadr_2 = int(model.jnt_qposadr[slider_2])
    start = float(data.qpos[qadr_1])

    viewer_context = contextlib.nullcontext(None)
    if args.viewer:
        from mujoco import viewer as mujoco_viewer

        viewer_context = mujoco_viewer.launch_passive(model, data)

    with viewer_context as viewer:
        steps = max(1, round(args.duration / model.opt.timestep))
        for step in range(steps):
            t = (step + 1) / steps
            alpha = t * t * (3.0 - 2.0 * t)
            data.ctrl[gripper] = (1.0 - alpha) * start + alpha * args.position
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                time.sleep(model.opt.timestep)

        for _ in range(max(0, round(args.hold / model.opt.timestep))):
            data.ctrl[gripper] = args.position
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                time.sleep(model.opt.timestep)

    print("SLIDER_TARGET_M:", args.position)
    print("SLIDER_1_FINAL_M:", float(data.qpos[qadr_1]))
    print("SLIDER_2_FINAL_M:", float(data.qpos[qadr_2]))
    print("SYNC_ERROR_M:", abs(float(data.qpos[qadr_1] - data.qpos[qadr_2])))
    print("FINAL_CONTACT_COUNT:", data.ncon)


if __name__ == "__main__":
    main()
