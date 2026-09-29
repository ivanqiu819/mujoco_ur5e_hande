"""Configuration and coordinate helpers shared by the insertion tools."""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from ur5e_sim.camera.calibration import ROOT, pose_matrix
from ur5e_sim.control.kinematics import Pose
from ur5e_sim.control.limits import cartesian_speed


def load_settings(path=None):
    path = Path(path or ROOT / 'configs/insertion.json').resolve()
    settings = json.loads(path.read_text())
    if settings.get('schema_version') != 1:
        raise ValueError('Unsupported insertion configuration')
    for key in ('source_scene', 'output_scene', 'camera_config', 'port_config', 'aruco_config', 'pnp_config'):
        settings[key] = str((path.parent / settings[key]).resolve())
    camera = json.loads(Path(settings['camera_config']).read_text())
    settings['render_supersample'] = camera['render_supersample']
    if settings['render_supersample'] not in (1, 2, 3, 4):
        raise ValueError('render_supersample must be 1, 2, 3 or 4')
    for key in ('fixture_height_m', 'slot_depth_m', 'lift_m', 'preinsert_m',
                'insert_depth_m', 'cartesian_step_m',
                'settle_s', 'detection_timeout_s', 'plug_mass_kg'):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    settings['insert_speed_m_s'] = cartesian_speed(settings['insert_speed_m_s'], field='insert_speed_m_s')
    for key in ('plug_size_m', 'plug_tip_tcp_m', 'table_center_xy_m',
                'workpiece_center_xy_m'):
        expected = 3 if key.startswith('plug_') else 2
        if np.asarray(settings[key]).shape != (expected,) or not np.isfinite(settings[key]).all():
            raise ValueError(f'Invalid {key}')
    if min(settings['plug_size_m']) <= 0 or settings['insert_depth_m'] >= settings['slot_depth_m']:
        raise ValueError('Invalid plug size or insertion depth')
    if not 0 < settings['cartesian_step_m'] <= .001 or settings['collision_samples'] < 2:
        raise ValueError('Cartesian spacing must be <= 1 mm; at least two collision samples')
    if settings.get('route') not in ('aruco','pnp'):
        raise ValueError('route must be aruco or pnp')
    if type(settings['collision_samples']) is not int or not 320 <= settings['collision_samples'] <= 5000:
        raise ValueError('collision_samples must be an integer in 320..5000')
    if settings['settle_s'] > 3:
        raise ValueError('settle_s must be <= 3 s')
    settings.setdefault('tactile_align_enabled', False)
    if settings['tactile_align_enabled']:
        for key in (
            'tactile_probe_m',
            'tactile_align_max_mm',
            'tactile_align_gain_mm_per_mm',
            'tactile_align_delta_threshold_mm',
        ):
            if not np.isfinite(settings[key]) or settings[key] <= 0:
                raise ValueError(f'{key} must be finite and positive when tactile_align_enabled')
        if settings['tactile_probe_m'] > settings['preinsert_m']:
            raise ValueError('tactile_probe_m must not exceed preinsert_m')
        iters = settings.get('tactile_align_max_iters', 10)
        if type(iters) is not int or not 1 <= iters <= 50:
            raise ValueError('tactile_align_max_iters must be an integer in 1..50')
        settings['tactile_align_max_iters'] = iters
    settings.setdefault('align_bias_port_m', [0.0, 0.0, 0.0])
    bias = np.asarray(settings['align_bias_port_m'], dtype=float).ravel()
    if bias.shape != (3,) or not np.isfinite(bias).all():
        raise ValueError('align_bias_port_m must be three finite numbers (port frame, metres)')
    settings['align_bias_port_m'] = bias.tolist()
    return settings


def pose_resources(settings, route=None):
    """Load local geometry and only the requested algorithm's settings."""
    from ur5e_sim.vision.core.geometry import load_geometry
    path = Path(settings['port_config'])
    config = json.loads(path.read_text())
    config['model']['path'] = str((path.parent / config['model']['path']).resolve())
    if config['model']['unit'] not in ('mm', 'cm', 'm'):
        raise ValueError('model.unit must be mm, cm or m')
    port = config['port']
    for key, length in [('center', 3), ('normal', 3), ('up', 3), ('inner_size', 2), ('outer_size', 2)]:
        value = np.asarray(port[key], dtype=float)
        if value.shape != (length,) or not np.isfinite(value).all():
            raise ValueError(f'Invalid port.{key}')
    if np.any(np.asarray(port['inner_size']) <= 0) or np.any(np.asarray(port['outer_size']) <= port['inner_size']):
        raise ValueError('Port outer dimensions must exceed positive inner dimensions')
    if route in (None, 'aruco'):
        options = json.loads(Path(settings['aruco_config']).read_text())
        config['aruco'] = dict(config['marker'], **{k:v for k,v in options.items() if k != 'pose'})
        config['pose'] = options['pose']
    if route == 'pnp':
        config.update(json.loads(Path(settings['pnp_config']).read_text()))
    return config, load_geometry(config)


def matrix_pose(transform):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(transform[:3, :3]).reshape(9))
    return Pose(transform[:3, 3].copy(), quat)


def site_matrix(data, name):
    return pose_matrix(data.site(name).xpos, data.site(name).xmat)


def observation_camera_in_port(settings):
    """OpenCV camera frame relative to the nominal port, in metres."""
    observation = settings['observation']
    az, el = np.deg2rad([observation['azimuth_deg'], observation['elevation_deg']])
    offset = np.array([np.sin(az)*np.cos(el), -np.sin(el), -np.cos(az)*np.cos(el)])
    right = np.array([np.cos(az), 0, np.sin(az)])
    forward = -offset
    return pose_matrix(observation['distance_m']*offset,
                       np.column_stack([right, np.cross(forward, right), forward]))


def scene_signature(settings):
    """Content fingerprint for parameters baked into the generated insertion scene."""
    import hashlib
    # Route and solver controls do not change geometry; physical camera/port/task parameters do.
    task = {k:v for k,v in settings.items() if k not in (
        'source_scene','output_scene','camera_config','port_config','aruco_config','pnp_config',
        'route','render_supersample','detection_timeout_s','collision_samples','settle_s',
        'insert_speed_m_s','cartesian_step_m','preinsert_m','insert_depth_m','lift_m','observation',
        'release_gripper_after_insert',
        'tactile_align_enabled','tactile_probe_m','tactile_align_max_mm',
        'tactile_align_max_iters','tactile_align_gain_mm_per_mm',
        'tactile_align_delta_threshold_mm','align_bias_port_m',
        'admittance_enabled','admittance_step_m','admittance_settle_s',
        'admittance_gain_mm_per_mm','admittance_max_correction_mm',
        'admittance_delta_threshold_mm','admittance_lateral_max_iters',
        'admittance_lateral_sign','admittance_auto_lateral_sign',
        'admittance_wall_press_mm','admittance_depth_hold_max_passes',
        'admittance_gel_feedback_gain','admittance_bias_gate_mm')}
    port = json.loads(Path(settings['port_config']).read_text())
    mesh = (Path(settings['port_config']).parent/port['model']['path']).resolve()
    port['model']['path'] = hashlib.sha256(mesh.read_bytes()).hexdigest()
    camera = json.loads(Path(settings['camera_config']).read_text())
    camera.pop('preview_hz', None)
    payload = {'task':task,'port':port,'camera':camera}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
