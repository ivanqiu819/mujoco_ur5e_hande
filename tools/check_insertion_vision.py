#!/usr/bin/env python3
"""Render the insertion observation pose and evaluate both existing detectors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from ur5e_sim.camera.calibration import ROOT, camera_transforms, load_spec
from ur5e_sim.camera.capture import InsertionCamera
from ur5e_sim.config import load_settings, matrix_pose, observation_camera_in_port, site_matrix
from ur5e_sim.vision.detect import detect_port
from ur5e_sim.control.kinematics import arm_addresses, solve_ik


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/vision')
    args = parser.parse_args()
    settings = load_settings(args.config)
    model = mujoco.MjModel.from_xml_path(settings['output_scene'])
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    _, world_camera = camera_transforms(model, data)
    handeye = np.linalg.inv(site_matrix(data, 'tcp')) @ world_camera
    nominal = model.numeric('insertion_nominal_port').data.reshape(4, 4)
    target = nominal @ observation_camera_in_port(settings) @ np.linalg.inv(handeye)
    ik = solve_ik(model, data.qpos.copy(), matrix_pose(target))
    if not ik.converged:
        raise RuntimeError('Observation IK failed')
    _, addresses, _ = arm_addresses(model)
    data.qpos[addresses] = ik.q_arm
    mujoco.mj_forward(model, data)
    spec = load_spec(settings['camera_config'])
    capture = InsertionCamera(model, spec, settings)
    try:
        rgb = capture.rgb(data)
    finally:
        capture.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir/'observation.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    _, world_camera = camera_transforms(model, data)
    truth = np.linalg.inv(world_camera) @ site_matrix(data, 'socket_port')
    failures = []
    for route in ('aruco', 'pnp'):
        result = detect_port(rgb, spec.intrinsic_matrix, route, settings)
        print('DETECTION:', route, result['status'], result['reason'])
        if result.get('T_camera_port') is not None:
            from ur5e_sim.vision.core.pose import evaluate_pose
            result['errors_vs_truth'] = evaluate_pose(result['T_camera_port'], truth)
            print('ERROR:', result['errors_vs_truth'])
        error = result.get('errors_vs_truth', {})
        if result['status'] != 'ok' or error.get('translation_error_mm', np.inf) > .3 or error.get('rotation_error_deg', np.inf) > 1:
            failures.append(route)
        (args.output_dir/(route+'.json')).write_text(json.dumps(result, indent=2))
    if failures:
        raise SystemExit('INSERTION_VISION_CHECK: FAILED: '+', '.join(failures))
    print('INSERTION_VISION_CHECK: PASSED')


if __name__ == '__main__':
    main()
