"""Client-side insertion orchestration. No model truth or physics mutations."""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import cv2
import mujoco
import numpy as np
from ur5e_sim.client import Client
from ur5e_sim.config import load_settings, observation_camera_in_port, scene_signature
from ur5e_sim.control.trajectory import rigid, line_targets
from ur5e_sim.paths import ROOT


def tcp_matrix(state):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(state['tcp_quat_wxyz']))
    result = np.eye(4); result[:3,:3] = rotation.reshape(3,3); result[:3,3] = state['tcp_position_m']
    return result


def vision_worker(pipe, rgb, K, route, settings):
    try:
        from ur5e_sim.vision.detect import detect_port
        pipe.send((True, detect_port(rgb, K, route, settings)))
    except Exception as exc: pipe.send((False, f'{type(exc).__name__}: {exc}'))
    finally: pipe.close()


def validated_world_pose(result, snapshot, state):
    if result.get('status') != 'ok' or result.get('T_camera_port') is None:
        raise ValueError(f"Detection refused: {result.get('status')}: {result.get('reason') or result.get('warnings')}")
    transform = rigid(result['T_camera_port'])
    if np.max(np.abs(np.asarray(snapshot['qpos'])-state['qpos'])) > .0002:
        raise ValueError('Robot moved during detection; snapshot is stale')
    world = rigid(snapshot['T_world_camera']) @ transform
    if abs(world[2,2]) > np.sin(np.deg2rad(1.0)):
        raise ValueError('Estimated insertion axis is not horizontal within 1 degree')
    return world


def detect_in_worker(client, rgb, snapshot, route, settings):
    ctx = mp.get_context('spawn'); parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=vision_worker, args=(child, rgb, snapshot['intrinsic_matrix'], route, settings))
    process.start(); child.close()
    deadline = time.monotonic()+settings['detection_timeout_s']
    try:
        while not parent.poll(.05):
            client.request(command='ping')
            if not process.is_alive(): raise RuntimeError('Detection worker exited without a result')
            if time.monotonic() >= deadline: raise TimeoutError('Detection timed out')
        ok, result = parent.recv()
        if not ok: raise ValueError(result)
    finally:
        parent.close(); process.join(timeout=.1)
        if process.is_alive(): process.terminate(); process.join(timeout=2)
        process.close()
    return result


class Insertion:
    def __init__(self, client, settings, route, output):
        self.client, self.settings, self.route = client, settings, route
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.tip = np.eye(4); self.tip[:3,3] = settings['plug_tip_tcp_m']
        self.report = {'status':'running','route':route,'phases':{},
                       'insert_speed_m_s':settings['insert_speed_m_s']}
        self.motion_options = {k:settings[k] for k in ('collision_samples','settle_s')}
        self.phase = 'idle'
        self.world_port = self.snapshot = self.result = None
        self.return_path = None

    def record(self, phase):
        self.phase = phase
        self.report['phases'][phase] = self.client.state()
        print('INSERTION_PHASE:', phase, flush=True)
        self.save()

    def save(self):
        (self.output/'cycle.json').write_text(json.dumps(self.report, indent=2)+'\n')

    def inspect(self):
        state = self.client.state()
        if not state['fixed_grip']: raise ValueError('Start server with the preheld insertion scene')
        if state['scene_signature'] != scene_signature(self.settings):
            raise ValueError('Scene/config mismatch: run python tools/build_scenes.py and restart server')
        if state['motion']['protected'] or state['arm_moving'] or state['motion']['phase'] == 'moving':
            raise ValueError('Robot busy or recovery required')
        if np.max(np.abs(np.asarray(state['qpos'])-state['home_qpos'])) > .001:
            raise ValueError('Full cycle requires preheld Home. If a previous version left the plug at the port, '
                             'restart apps/server.py once. Unknown starting poses are not moved automatically.')
        if np.max(np.abs(state['qvel'])) > .001:
            raise ValueError('Home has not settled')
        if state['contact_count'] or abs(state['gripper_gap_mm']-1000*self.settings['plug_size_m'][0]) > .1:
            raise ValueError('Invalid preheld Home or initial contact')
        rgb, meta = self.client.capture()
        from ur5e_sim.camera.calibration import load_spec
        if not np.allclose(meta['intrinsic_matrix'], load_spec(self.settings['camera_config']).intrinsic_matrix):
            raise ValueError('Server camera calibration differs from task configuration')
        state = self.client.state()
        world_tcp = tcp_matrix(state)
        hand_eye = np.linalg.inv(world_tcp) @ np.asarray(meta['T_world_camera'])
        lifted = world_tcp.copy(); lifted[2,3] += self.settings['lift_m']
        observation = np.asarray(state['nominal_port']) @ observation_camera_in_port(self.settings) @ np.linalg.inv(hand_eye)
        # Save this cycle's approach poses. Return through them only after a
        # completed retraction, with the complete reverse path checked again.
        self.return_path = [observation.copy(), lifted.copy(), world_tcp.copy()]
        self.client.move([lifted, observation], **self.motion_options); self.client.wait()
        self.record('observed')
        rgb, self.snapshot = self.client.capture()
        cv2.imwrite(str(self.output/'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        (self.output/'snapshot.json').write_text(json.dumps(self.snapshot,indent=2)+'\n')
        self.result = detect_in_worker(self.client, rgb, self.snapshot, self.route, self.settings)
        (self.output/'detection.json').write_text(json.dumps(self.result,indent=2)+'\n')
        self.world_port = validated_world_pose(self.result, self.snapshot, self.client.state())
        self.report['T_world_port'] = self.world_port.tolist()
        self.record('detected')

    def align(self):
        if self.phase != 'detected' or self.world_port is None:
            raise ValueError('Alignment requires a successful fresh detection')
        target = self.world_port.copy(); target[:3,3] -= target[:3,2]*self.settings['preinsert_m']
        self.client.move([target @ np.linalg.inv(self.tip)], capture_id=self.snapshot['capture_id'], **self.motion_options)
        self.client.wait(); self.record('aligned')

    def insert(self):
        if self.phase != 'aligned': raise ValueError('Insert requires alignment')
        start = tcp_matrix(self.client.state())
        end = self.world_port.copy(); end[:3,3] += end[:3,2]*self.settings['insert_depth_m']
        targets = line_targets(start, end @ np.linalg.inv(self.tip), self.settings['cartesian_step_m'])
        self.client.move(targets, cartesian=True, speed=self.settings['insert_speed_m_s'], recovery=start, **self.motion_options)
        self.client.wait(); self.record('inserted')

    def retract(self):
        self.client.request(op='recover'); self.client.wait(); self.record('retracted')

    def return_home(self):
        if self.phase != 'retracted' or self.return_path is None:
            raise ValueError('Return Home requires this cycle to finish retracting')
        state = self.client.state()
        previous = self.report['phases']['retracted']
        if (state['motion']['protected'] or state['motion']['phase'] != 'recovered'
                or state['generation'] != previous['generation']
                or np.max(np.abs(np.asarray(state['qpos'])-previous['qpos'])) > .0002):
            raise ValueError('Retraction state changed; automatic return Home refused')
        self.client.move(self.return_path, **self.motion_options)
        state = self.client.wait()
        if (np.max(np.abs(np.asarray(state['qpos'])-state['home_qpos'])) > .001
                or np.max(np.abs(state['qvel'])) > .001 or state['contact_count']):
            raise ValueError('Return path did not reach a settled, collision-free Home')
        self.record('home')

    def run(self, step=False):
        try:
            for name in ('inspect','align','insert','retract','return_home'):
                if step:
                    # Keep the control heartbeat alive while the user decides the next step.
                    import select, sys
                    print(f'Enter to {name}; q to stop:', flush=True)
                    while not select.select([sys.stdin],[],[],.2)[0]: self.client.request(command='ping')
                    answer = sys.stdin.readline()
                    if not answer or answer.strip().lower() == 'q': raise KeyboardInterrupt()
                getattr(self, name)()
            self.report['status'] = 'passed'
        except BaseException as exc:
            self.report.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
            try: self.client.stop()
            except (OSError, RuntimeError): pass
            raise
        finally: self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/insertion.json')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--route', choices=['aruco','pnp'])
    parser.add_argument('--step', action='store_true')
    parser.add_argument('--retract-only', action='store_true', help='Explicit recovery after stop/disconnect')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print('INSERTION_STOPPED: invalid configuration:', str(exc), flush=True)
        raise SystemExit(1)
    route = args.route or settings['route']
    for name in ['camera_config','port_config',route+'_config']:
        print('CONFIG:', name, settings[name], flush=True)
    print('CONFIG: task', args.config.resolve(), flush=True)
    print(f"INSERT_SPEED: {settings['insert_speed_m_s']*1000:g} mm/s", flush=True)
    output = args.output_dir or ROOT/'outputs'/time.strftime('%Y%m%d_%H%M%S')/route
    try:
        with Client(args.port) as client:
            task = Insertion(client, settings, route, output)
            if args.retract_only:
                task.retract(); task.report['status']='passed'; task.save()
            else: task.run(args.step)
    except (ValueError, RuntimeError, OSError, TimeoutError, KeyboardInterrupt) as exc:
        print('INSERTION_STOPPED:', str(exc), flush=True)
        raise SystemExit(1)


if __name__ == '__main__': main()
