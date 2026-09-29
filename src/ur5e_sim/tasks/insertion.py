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


def _home_qpos_tolerance(home_qpos: np.ndarray, qpos: np.ndarray) -> tuple[float, float]:
    """Arm+slider vs compliant-grasp drift limits at idle Home (gel–plug preload)."""
    home_qpos = np.asarray(home_qpos, dtype=float).ravel()
    qpos = np.asarray(qpos, dtype=float).ravel()
    if home_qpos.shape != qpos.shape:
        return float('inf'), float('inf')
    n_grasp = 6 if home_qpos.size >= 14 else 0
    core_home = home_qpos[:-n_grasp] if n_grasp else home_qpos
    core_q = qpos[:-n_grasp] if n_grasp else qpos
    core_err = float(np.max(np.abs(core_q - core_home)))
    grasp_err = float(np.max(np.abs(qpos[-n_grasp:] - home_qpos[-n_grasp:]))) if n_grasp else 0.0
    return core_err, grasp_err


def _at_preheld_home(state: dict) -> bool:
    if state.get('home_tcp_position_m') is not None:
        pos_mm = float(
            np.linalg.norm(
                np.asarray(state['tcp_position_m'], dtype=float)
                - np.asarray(state['home_tcp_position_m'], dtype=float)
            )
            * 1000.0
        )
        return pos_mm <= 3.0 and state['contact_count'] == 0
    core_err, _ = _home_qpos_tolerance(state['home_qpos'], state['qpos'])
    return core_err <= 0.008 and state['contact_count'] == 0


def vision_worker(pipe, rgb, K, route, settings):
    try:
        from ur5e_sim.vision.detect import detect_port
        pipe.send((True, detect_port(rgb, K, route, settings)))
    except Exception as exc: pipe.send((False, f'{type(exc).__name__}: {exc}'))
    finally: pipe.close()


def _snapshot_pose_stale(snapshot: dict, state: dict) -> bool:
    """Allow compliant-grasp micro-motion; reject real arm motion during detection."""
    if 'tcp_position_m' in snapshot and 'tcp_position_m' in state:
        pos_mm = float(
            np.linalg.norm(
                np.asarray(snapshot['tcp_position_m'], dtype=float)
                - np.asarray(state['tcp_position_m'], dtype=float)
            )
            * 1000.0
        )
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, np.asarray(snapshot['tcp_quat_wxyz'], dtype=float))
        snap_r = rot.reshape(3, 3)
        mujoco.mju_quat2Mat(rot, np.asarray(state['tcp_quat_wxyz'], dtype=float))
        cur_r = rot.reshape(3, 3)
        angle_deg = float(
            np.degrees(
                np.arccos(np.clip((np.trace(cur_r @ snap_r.T) - 1.0) / 2.0, -1.0, 1.0))
            )
        )
        return pos_mm > 1.5 or angle_deg > 1.0
    core_err, grasp_err = _home_qpos_tolerance(snapshot['qpos'], state['qpos'])
    return core_err > 0.004 or grasp_err > 0.015


def apply_align_bias_port(world_port: np.ndarray, bias_port_m) -> np.ndarray:
    """Shift estimated port pose in port tangent/normal frame (simulated vision error)."""
    port = np.asarray(world_port, dtype=float).copy()
    bias = np.asarray(bias_port_m, dtype=float).ravel()
    if bias.shape != (3,) or not np.any(np.abs(bias) > 0):
        return port
    port[:3, 3] += port[:3, 0] * bias[0] + port[:3, 1] * bias[1] + port[:3, 2] * bias[2]
    return port


def validated_world_pose(result, snapshot, state):
    if result.get('status') != 'ok' or result.get('T_camera_port') is None:
        raise ValueError(f"Detection refused: {result.get('status')}: {result.get('reason') or result.get('warnings')}")
    transform = rigid(result['T_camera_port'])
    if _snapshot_pose_stale(snapshot, state):
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
        self.insert_retract_tcp = None
        self.plug_seated = False

    def _port_plane_axes(self) -> tuple[np.ndarray, np.ndarray]:
        port = np.asarray(self.client.state()['nominal_port'], dtype=float)
        ex = port[:3, 0]
        ey = port[:3, 1]
        ex = ex / (np.linalg.norm(ex) + 1e-12)
        ey = ey / (np.linalg.norm(ey) + 1e-12)
        return ex, ey

    def _align_bias_world_m(self) -> np.ndarray:
        """Injected vision error vector in world frame (port tangent plane)."""
        bias = np.asarray(self.settings.get('align_bias_port_m', [0.0, 0.0, 0.0]), dtype=float).ravel()
        if bias.size < 2 or not np.any(np.abs(bias[:2]) > 1e-9):
            return np.zeros(3)
        ex, ey = self._port_plane_axes()
        return ex * float(bias[0]) + ey * float(bias[1])

    def _bias_remaining_mm(
        self, cumulative_correction: np.ndarray, bias_world: np.ndarray,
    ) -> float:
        if float(np.linalg.norm(bias_world)) < 1e-9:
            return 0.0
        need = -bias_world - np.asarray(cumulative_correction, dtype=float).ravel()[:3]
        return float(np.linalg.norm(need) * 1000.0)

    def record(self, phase):
        self.phase = phase
        state = self.client.state()
        self.report['phases'][phase] = state
        print('INSERTION_PHASE:', phase, flush=True)
        self.save()

    def save(self):
        (self.output/'cycle.json').write_text(json.dumps(self.report, indent=2)+'\n')

    def inspect(self):
        state = self.client.state()
        if not state.get('insertion_scene', state.get('fixed_grip')):
            raise ValueError('Start server with the insertion scene')
        if state['scene_signature'] != scene_signature(self.settings):
            raise ValueError('Scene/config mismatch: run python tools/build_scenes.py and restart server')
        if state['motion']['protected'] or state['arm_moving'] or state['motion']['phase'] == 'moving':
            raise ValueError('Robot busy or recovery required')
        if not _at_preheld_home(state):
            self.client.reset_home()
            state = self.client.state()
        if not _at_preheld_home(state):
            raise ValueError(
                'Full cycle requires preheld Home. Restart apps/server.py or call reset_home failed.'
            )
        if np.max(np.abs(state['qvel'])) > .001:
            raise ValueError('Home has not settled')
        expected_slider_q = float(state['closed_q_m']) + float(self.settings['plug_size_m'][0]) / 2.0
        slider_q = np.asarray(state['slider_q_m'], dtype=float)
        if state['contact_count'] or np.max(np.abs(slider_q - expected_slider_q)) > 0.005:
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
        self.world_port_true = validated_world_pose(self.result, self.snapshot, self.client.state())
        self.world_port = apply_align_bias_port(self.world_port_true, self.settings.get('align_bias_port_m'))
        self.report['T_world_port'] = self.world_port.tolist()
        if np.any(np.abs(self.settings.get('align_bias_port_m', [0, 0, 0]))):
            self.report['align_bias_port_m'] = self.settings['align_bias_port_m']
        self.record('detected')

    def align(self):
        if self.phase != 'detected' or self.world_port is None:
            raise ValueError('Alignment requires a successful fresh detection')
        target = self.world_port.copy(); target[:3,3] -= target[:3,2]*self.settings['preinsert_m']
        self.client.move([target @ np.linalg.inv(self.tip)], capture_id=self.snapshot['capture_id'], **self.motion_options)
        self.client.wait(); self.record('aligned')

    def tactile_align(self):
        if self.phase != 'aligned':
            raise ValueError('Tactile alignment requires visual alignment')
        if not self.settings.get('tactile_align_enabled'):
            return
        print(
            'TACTILE_ALIGN: skip preinsert (gel is grasp preload; correct only after socket contact)',
            flush=True,
        )
        self.record('tactile_aligned')

    def insert(self):
        if self.phase not in ('aligned', 'tactile_aligned'):
            raise ValueError('Insert requires alignment')
        start = tcp_matrix(self.client.state())
        self.insert_retract_tcp = start.copy()
        end = self.world_port.copy()
        end[:3, 3] += end[:3, 2] * float(self.settings['insert_depth_m'])
        tcp_end = end @ np.linalg.inv(self.tip)
        tcp_end[:3, :3] = start[:3, :3]

        if self.settings.get('admittance_enabled', False):
            self._admittance_lat_sign = float(
                self.settings.get('admittance_lateral_sign', 1.0))
            self._admittance_sign_probed = False
            self._admittance_insert(start, tcp_end)
        else:
            targets = line_targets(start, tcp_end, self.settings['cartesian_step_m'])
            self.client.move(targets, cartesian=True,
                             speed=self.settings['insert_speed_m_s'],
                             recovery=start, **self.motion_options)
            self.client.wait()
            self.plug_seated = True
        self.record('inserted')

    # ------------------------------------------------------------------
    #  Admittance-controlled insertion
    # ------------------------------------------------------------------
    def _admittance_insert(self, start: np.ndarray, tcp_end: np.ndarray):
        """Continuous insert with gel monitoring (identical physics to open-loop).

        Sends the full trajectory as one move with recovery, exactly like the
        open-loop path.  During the move, polls gel readings for diagnostics.
        After the move completes naturally, checks whether the plug actually
        seated (stroke remaining < 2mm and no significant grip slip).
        """
        self.plug_seated = False
        self.insert_aborted = False
        cart_step = float(self.settings.get('cartesian_step_m', 0.001))
        speed = self.settings['insert_speed_m_s']

        insert_vec = tcp_end[:3, 3] - start[:3, 3]
        total_dist = float(np.linalg.norm(insert_vec))
        bias_world = self._align_bias_world_m()

        targets = line_targets(start, tcp_end, cart_step)
        max_peak = 0.0

        print(
            f'ADMITTANCE_INSERT: {total_dist*1000:.1f}mm continuous '
            f'(cart_step={cart_step*1000:g}mm  gel=monitor-only)',
            flush=True,
        )

        # Identical move call to the open-loop branch
        self.client.move(targets, cartesian=True,
                         speed=speed,
                         recovery=start, **self.motion_options)

        # Poll during motion for gel diagnostics (no interruption)
        log = []
        poll_i = 0
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            state = self.client.state()
            phase = state['motion']['phase']
            if phase in ('error', 'stopped'):
                self.insert_aborted = True
                break
            if not state['arm_moving'] and phase != 'moving':
                break

            tactile = self.client.read_tactile()
            if tactile.get('enabled'):
                pl = float(tactile['press_l_mm'])
                pr = float(tactile['press_r_mm'])
                peak = max(pl, pr)
                max_peak = max(max_peak, peak)
                if poll_i % 10 == 0:
                    print(
                        f'ADMITTANCE: poll {poll_i}  L={pl:.3f} R={pr:.3f}  ok',
                        flush=True,
                    )
                log.append({
                    'poll': poll_i,
                    'press_l_mm': pl,
                    'press_r_mm': pr,
                })

            poll_i += 1
            time.sleep(0.02)

        # Post-move assessment
        insert_dir = insert_vec / (float(np.linalg.norm(insert_vec)) + 1e-12)
        state = self.client.state()
        tcp_now = np.asarray(state['tcp_position_m'], dtype=float)
        remain_mm = float(
            np.dot(tcp_end[:3, 3] - tcp_now, insert_dir) * 1000.0)
        stroke_done = remain_mm < 3.0
        self.plug_seated = bool(stroke_done and not self.insert_aborted)

        cumulative_correction = np.zeros(3)
        bias_rem = self._bias_remaining_mm(cumulative_correction, bias_world)
        print(
            f'ADMITTANCE_DONE: seated={self.plug_seated}  '
            f'remain={remain_mm:.1f}mm  max_peak={max_peak:.3f}mm  '
            f'bias_rem={bias_rem:.3f}mm',
            flush=True,
        )
        state = self.client.state()
        tip = state.get('plug_tip_world')
        if tip is not None:
            port = np.asarray(state['nominal_port'], dtype=float)
            ex, ey = port[:3, 0], port[:3, 1]
            ex /= np.linalg.norm(ex) + 1e-12
            ey /= np.linalg.norm(ey) + 1e-12
            tip_pos = np.asarray(tip, dtype=float)[:3, 3]
            port_pos = port[:3, 3]
            off = tip_pos - port_pos
            off_x = float(np.dot(off, ex) * 1000.0)
            off_y = float(np.dot(off, ey) * 1000.0)
            print(
                f'PLUG_TIP_OFFSET_MM: port_x={off_x:+.2f} port_y={off_y:+.2f} '
                f'(nominal socket frame)',
                flush=True,
            )
            self.report['plug_tip_offset_port_mm'] = [off_x, off_y]
        self.report['admittance_log'] = log
        self.report['admittance_lateral_sign'] = self._admittance_lat_sign
        self.report['admittance_bias_remaining_mm'] = bias_rem

    def release_plug(self):
        if not getattr(self, 'plug_seated', False):
            print(
                'GRIPPER_HOLD: insert did not seat the plug; skipping release',
                flush=True,
            )
            return
        if self.phase != 'inserted':
            raise ValueError('Release requires completed insertion')
        if not self.settings.get('release_gripper_after_insert', True):
            raise ValueError('Task config disables gripper release')
        self.client.wait()
        self.client.release_plug()
        self.client.wait_gripper()
        self.client.seat_plug()
        settle_s = float(self.settings['settle_s'])
        deadline = time.monotonic() + max(settle_s, 0.5)
        while time.monotonic() < deadline:
            self.client.request(command='ping')
            time.sleep(0.05)
        self.record('released')

    def retract(self):
        state = self.client.state()
        if state['motion'].get('protected'):
            self.client.request(op='recover')
            self.client.wait()
        elif getattr(self, 'insert_retract_tcp', None) is not None:
            current = tcp_matrix(state)
            back = self.insert_retract_tcp.copy()
            back[:3, :3] = current[:3, :3]
            targets = line_targets(current, back, self.settings['cartesian_step_m'])
            self.client.move(
                targets,
                cartesian=True,
                speed=self.settings['insert_speed_m_s'],
                **self.motion_options,
            )
            self.client.wait()
        else:
            raise ValueError('No retraction path (missing insert start pose)')
        self.record('retracted')

    def return_home(self):
        if self.phase != 'retracted' or self.return_path is None:
            raise ValueError('Return Home requires this cycle to finish retracting')
        state = self.client.state()
        previous = self.report['phases']['retracted']
        core_delta, _ = _home_qpos_tolerance(previous['qpos'], state['qpos'])
        phase = state['motion']['phase']
        if phase not in ('recovered', 'done', 'idle'):
            raise ValueError('Retraction state changed; automatic return Home refused')
        if (state['motion']['protected']
                or state['generation'] != previous['generation']
                or core_delta > 0.004):
            raise ValueError('Retraction state changed; automatic return Home refused')
        self.client.move(self.return_path, **self.motion_options)
        state = self.client.wait()
        if not _at_preheld_home(state):
            self.client.reset_home()
            state = self.client.state()
        if not _at_preheld_home(state):
            raise ValueError('Return path did not reach a settled, collision-free Home')
        self.record('home')

    def _cycle_steps(self):
        steps = ['inspect', 'align']
        if self.settings.get('tactile_align_enabled'):
            steps.append('tactile_align')
        steps.append('insert')
        if self.settings.get('release_gripper_after_insert', True):
            steps.append('release_plug')
        steps.extend(['retract', 'return_home'])
        return steps

    def run(self, step=False):
        try:
            for name in self._cycle_steps():
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
    parser.add_argument(
        '--align-bias-mm',
        nargs=2,
        type=float,
        metavar=('PORT_X', 'PORT_Y'),
        help='Inject vision error after detection: shift align target in port X/Y (mm)',
    )
    parser.add_argument(
        '--no-tactile-align',
        action='store_true',
        help='Disable tactile_align stage (compare against biased vision-only align)',
    )
    parser.add_argument(
        '--no-admittance',
        action='store_true',
        help='Disable admittance control during insertion (open-loop insert)',
    )
    parser.add_argument(
        '--invert-admittance-lateral',
        action='store_true',
        help='Flip gel lateral correction sign (if plug moves the wrong way)',
    )
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print('INSERTION_STOPPED: invalid configuration:', str(exc), flush=True)
        raise SystemExit(1)
    if args.align_bias_mm is not None:
        settings['align_bias_port_m'] = [
            float(args.align_bias_mm[0]) * 0.001,
            float(args.align_bias_mm[1]) * 0.001,
            0.0,
        ]
    if args.no_tactile_align:
        settings['tactile_align_enabled'] = False
    if args.no_admittance:
        settings['admittance_enabled'] = False
    if args.invert_admittance_lateral:
        settings['admittance_lateral_sign'] = -float(
            settings.get('admittance_lateral_sign', 1.0)
        )
    route = args.route or settings['route']
    if np.any(np.abs(settings.get('align_bias_port_m', [0, 0, 0]))):
        bx, by, _ = settings['align_bias_port_m']
        print(f"ALIGN_BIAS_PORT_MM: {bx * 1000:g} {by * 1000:g} 0", flush=True)
    if settings.get('tactile_align_enabled'):
        print('TACTILE_ALIGN: enabled', flush=True)
    else:
        print('TACTILE_ALIGN: disabled', flush=True)
    if settings.get('admittance_enabled', False):
        print('ADMITTANCE_INSERT: enabled', flush=True)
    else:
        print('ADMITTANCE_INSERT: disabled (open-loop)', flush=True)
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
