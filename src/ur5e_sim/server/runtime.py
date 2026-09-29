#!/usr/bin/env python3
"""Terminal A: MuJoCo simulation and loopback-only command receiver.

Run apps/debug_pose.py or apps/insert_socket.py in terminal B. State changes run on
the simulation thread. No robot controls are attached to Viewer key presses.
Commands are JSON lines, one local client at a time. Not a real-robot server.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import socket
import time

import mujoco
import numpy as np

from ur5e_sim.control.manual import TeleopController, MotionPlan, handle_key, handle_terminal_command
from ur5e_sim.control.kinematics import set_target_marker, Pose, normalize_quaternion
from ur5e_sim.control.limits import DEFAULT_CARTESIAN_SPEED_M_S
from ur5e_sim.control.trajectory import unexpected_contact_count

from ur5e_sim.paths import ROOT as ROOT
MAX_BYTES = 65536


class Controller(TeleopController):
    """Calibrated opening, slew-limited gripper and conservative arm preflight."""

    def __init__(self, model, data, args):
        for name in ('gripper_closed_q', 'gripper_open_q', 'gripper_zero_gap'):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)<0:
                raise ValueError('Use scene_control.xml; missing calibration: '+name)
        super().__init__(model, data, collision_samples=args.collision_samples,
                         max_joint_speed_rad_s=args.max_joint_speed,
                         linear_step_m=.005, angular_step_deg=2, gripper_step_m=.001)
        self.closed_position = float(model.numeric('gripper_closed_q').data[0])
        self.open_position = float(model.numeric('gripper_open_q').data[0])
        self.gripper_lower = max(self.gripper_lower, self.closed_position)
        self.gripper_command = self.gripper_target
        self.gripper_speed = .020  # m/s per slider
        self.gripper_changed_at = -10.0
        self.finger_bodies = {model.body('left_gripper').id, model.body('right_gripper').id}
        self.slider_ids = [model.joint(n).id for n in ('Slider_1','Slider_2')]
        self.slider_qadr = model.jnt_qposadr[self.slider_ids]
        self.slider_vadr = model.jnt_dofadr[self.slider_ids]
        self.last_gripper_state = 'at_target'
        from ur5e_sim.control.trajectory import TrajectoryExecutor
        self.insertion_scene = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, 'insertion_nominal_port') >= 0
        )
        if self.insertion_scene and model.opt.noslip_iterations < 3:
            model.opt.noslip_iterations = 3
        self.plug_grasp_locked = bool(self.insertion_scene)
        self._socket_wall_original_friction = {}
        if self.insertion_scene:
            for i in range(model.ngeom):
                name = model.geom(i).name or ""
                if name.startswith("socket_wall_"):
                    self._socket_wall_original_friction[i] = model.geom_friction[i].copy()
        self.home_slider_q = None
        self.home_gripper_ctrl = None
        self.home_plug_qpos = None
        self.home_plug_qadr = None
        self.home_tcp_position = None
        self.home_tcp_quat = None
        if self.insertion_scene:
            from ur5e_sim.control.kinematics import site_pose, object_id

            home_id = model.key('home').id
            home_q = model.key_qpos[home_id]
            self.home_slider_q = home_q[self.slider_qadr].copy()
            self.home_gripper_ctrl = float(model.key_ctrl[home_id, self.gripper_id])
            plug_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'plug_free')
            if plug_jid >= 0:
                self.home_plug_qadr = int(model.jnt_qposadr[plug_jid])
                self.home_plug_qpos = home_q[self.home_plug_qadr : self.home_plug_qadr + 7].copy()
            home_data = mujoco.MjData(model)
            home_data.qpos[:] = home_q
            mujoco.mj_forward(model, home_data)
            tcp_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
            home_tcp = site_pose(model, home_data, tcp_id)
            self.home_tcp_position = home_tcp.position.copy()
            self.home_tcp_quat = home_tcp.quaternion_wxyz.copy()
            self.gripper_target = self.home_gripper_ctrl
            self.gripper_command = self.gripper_target
            data.ctrl[self.gripper_id] = self.gripper_command
        self.motion = TrajectoryExecutor(self, {'collision_samples': max(320, args.collision_samples),
                                              'insert_speed_m_s': DEFAULT_CARTESIAN_SPEED_M_S, 'settle_s': .3})
        self.camera = None
        self.last_capture = None
        self.motion_generation = 0
        from ur5e_sim.tactile.preview import model_has_gels

        self.tactile_gels = model_has_gels(model)

    def tactile_read(self) -> dict:
        if not self.tactile_gels:
            return {'enabled': False}
        from ur5e_sim.tactile.tactile_boundary import compute_dual_boundary

        from ur5e_sim.control.tactile_align import gripper_open_axis_world

        boundary = compute_dual_boundary(self.model, self.data)
        press_l, press_r = boundary[2], boundary[3]
        delta = float(press_l - press_r)
        axis = gripper_open_axis_world(self.model, self.data)
        result = {
            'enabled': True,
            'press_l_mm': float(press_l),
            'press_r_mm': float(press_r),
            'delta_press_mm': delta,
            'peak_press_mm': float(max(press_l, press_r)),
        }
        if np.any(axis):
            result['gripper_open_axis_world'] = axis.tolist()
        if self.insertion_scene:
            nominal = self.model.numeric('insertion_nominal_port').data.reshape(4, 4)
            result['nominal_port_x_world'] = nominal[:3, 0].tolist()
        return result

    def gripper_state(self):
        error = float(np.max(np.abs(self.data.qpos[self.slider_qadr]-self.gripper_target)))
        if error < .0002:
            return 'at_target'
        ramping = abs(self.gripper_command-self.gripper_target) > 1e-8
        velocity = float(np.max(np.abs(self.data.qvel[self.slider_vadr])))
        if not ramping and self.data.time-self.gripper_changed_at>.5 and velocity<.0002:
            if self.plug_grasp_locked or self.finger_contacts():
                return 'contact_blocked'
            return 'not_reached'
        return 'moving'

    def finger_contacts(self):
        result = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if any(int(self.model.geom_bodyid[g]) in self.finger_bodies
                   for g in (c.geom1,c.geom2)):
                result.append({'geom1':self.model.geom(c.geom1).name,
                               'geom2':self.model.geom(c.geom2).name,
                               'distance_m':float(c.dist)})
        return result

    def gap_m(self):
        axis = self.data.xmat[self.model.body('left_gripper').id].reshape(3,3)[:,2]
        vector = (self.data.site('gripper_right_inner').xpos -
                  self.data.site('gripper_left_inner').xpos)
        return float(np.dot(vector,axis))

    def state(self):
        pose = self.current_pose()
        result = {'sim_time':float(self.data.time), 'arm_moving':self.plan is not None,
                'tcp_position_m':pose.position.tolist(),
                'tcp_quat_wxyz':pose.quaternion_wxyz.tolist(),
                'tcp_target_position_m':self.commanded_pose.position.tolist(),
                'slider_q_m':self.data.qpos[self.slider_qadr].tolist(),
                'gripper_target_q_m':float(self.gripper_target),
                'gripper_gap_mm':1000*self.gap_m(),
                'gripper_state':self.gripper_state(),
                'contacts':self.finger_contacts(),
                'closed_q_m':self.closed_position,
                'open_q_m':self.open_position}
        result.update(
            motion=self.motion.state(),
            qpos=self.data.qpos.tolist(),
            qvel=self.data.qvel.tolist(),
            insertion_scene=self.insertion_scene,
            plug_grasp_locked=self.plug_grasp_locked,
            fixed_grip=self.insertion_scene,
            fixed_grip_locked=self.plug_grasp_locked,
            contact_count=unexpected_contact_count(self.model, self.data),
            generation=self.motion_generation,
        )
        if self.insertion_scene and self.home_plug_qadr is not None:
            from ur5e_sim.config import site_matrix

            result['plug_tip_world'] = site_matrix(self.data, 'plug_tip').tolist()
        if self.camera is not None:
            result['camera'] = self.camera.status
        if self.insertion_scene:
            result['nominal_port'] = self.model.numeric('insertion_nominal_port').data.reshape(4,4).tolist()
            signature_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_TEXT,'insertion_config_sha256')
            result['scene_signature'] = (self.model.text_data[self.model.text_adr[signature_id]:
                self.model.text_adr[signature_id]+self.model.text_size[signature_id]-1].decode()
                if signature_id >= 0 else None)
            result['home_qpos'] = self.model.key('home').qpos.tolist()
            if self.home_tcp_position is not None:
                result['home_tcp_position_m'] = self.home_tcp_position.tolist()
                result['home_tcp_quat_wxyz'] = self.home_tcp_quat.tolist()
        result['tactile'] = self.tactile_read()
        return result

    def reset_to_home_keyframe(self) -> None:
        if not self.insertion_scene:
            raise ValueError('reset_home is only for the insertion scene')
        self.motion.cancel()
        self.motion.protected = False
        self.motion.recovery_target = None
        self.motion.phase = 'idle'
        self.motion.reason = ''
        self.plan = None
        self.last_capture = None
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('home').id)
        if self.home_slider_q is not None:
            self.data.qpos[self.slider_qadr] = self.home_slider_q
        if self.home_plug_qadr is not None and self.home_plug_qpos is not None:
            self.data.qpos[self.home_plug_qadr : self.home_plug_qadr + 7] = self.home_plug_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.arm_hold_target = self.data.qpos[self.qpos_addresses].copy()
        self.gripper_target = self.home_gripper_ctrl
        self.gripper_command = self.gripper_target
        self.data.ctrl[self.gripper_id] = self.gripper_command
        self.data.ctrl[self.arm_actuator_ids] = self.arm_hold_target
        self.commanded_pose = self.current_pose()
        set_target_marker(self.model, self.data, self.commanded_pose)
        self.plug_grasp_locked = True
        self._restore_socket_wall_friction()
        self.motion_generation += 1
        print('RESET_HOME: keyframe restored', flush=True)

    def seat_plug_in_socket(self) -> None:
        if not self.insertion_scene:
            raise ValueError('seat_plug is only for the insertion scene')
        if self.plug_grasp_locked:
            raise ValueError('Release the gripper before settling the plug')
        if self.gripper_state() != 'at_target':
            raise ValueError('Gripper must finish opening before settling')
        mujoco.mj_forward(self.model, self.data)
        print('PLUG_RELEASED: gripper open; plug settles under contact and gravity', flush=True)

    def release_preheld_plug(self) -> None:
        if not self.insertion_scene:
            raise ValueError('release_plug is only for the insertion scene')
        if not self.plug_grasp_locked:
            raise ValueError('Plug grip is already released')
        if self.plan is not None or self.motion.busy:
            raise ValueError('Arm busy; wait before release')
        gs = self.gripper_state()
        if gs == 'moving' and not self.plug_grasp_locked:
            raise ValueError('Gripper busy; wait before release')
        # Boost socket wall friction before opening the gripper so the plug
        # stays in the socket under gravity (tight-fit hold friction).
        self._boost_socket_wall_friction()
        self.plug_grasp_locked = False
        self.gripper_target = float(self.open_position)
        self.gripper_changed_at = float(self.data.time)
        print('GRIPPER_RELEASE: opening to leave plug in socket', flush=True)

    def _boost_socket_wall_friction(self) -> None:
        """Raise socket wall friction to hold the plug after gripper release.

        During insertion the socket wall has moderate friction so the plug
        slides in without excessive grip-slip.  Once the gripper opens, we
        need high wall friction to keep the plug seated against gravity.
        """
        hold_friction = np.array([2.0, 0.05, 0.005])
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name or ""
            if name.startswith("socket_wall_"):
                self.model.geom_friction[i, :len(hold_friction)] = hold_friction
        print('SOCKET_WALL_FRICTION_BOOSTED: post-release hold friction applied', flush=True)

    def _restore_socket_wall_friction(self) -> None:
        """Restore socket wall friction to its original (scene-build) values."""
        for gid, friction in self._socket_wall_original_friction.items():
            self.model.geom_friction[gid] = friction

    def stop(self):
        self.motion.cancel()
        self.motion_generation += 1
        self.last_capture = None
        super().stop()
        if self.home_gripper_ctrl is not None and self.plug_grasp_locked:
            self.gripper_target = self.home_gripper_ctrl
        else:
            self.gripper_target = float(np.mean(self.data.qpos[self.slider_qadr]))
        self.gripper_command = self.gripper_target
        self.data.ctrl[self.gripper_id] = self.gripper_command
        self.data.ctrl[self.arm_actuator_ids] = self.arm_hold_target
        print('GRIPPER_STOPPED: hold measured slider position (not an instantaneous brake)')

    def set_gripper(self, position_m, *, label):
        if self.insertion_scene and self.plug_grasp_locked:
            print('COMMAND_REJECTED: insertion scene keeps gripper closed until release_plug')
            return False
        if self.plan is not None or self.motion.busy:
            print('COMMAND_REJECTED: arm is moving; wait for completion or send stop')
            return False
        ok = super().set_gripper(position_m, label=label)
        if ok:
            self.gripper_changed_at = float(self.data.time)
        return ok

    def _accept_arm_target(self, target_q_arm, target_pose, *, label):
        if self.motion.busy or self.motion.protected:
            print('COMMAND_REJECTED: motion busy or protected; use stop/recover')
            return False
        # Ignore no environmental contact. The only tolerated pair is the two
        # fingers touching each other shallowly at the calibrated closed stop.
        if self.plan is not None or self.gripper_state() == 'moving':
            print('COMMAND_REJECTED: BUSY; wait for current motion or send stop')
            return False
        start = self.data.qpos.copy()
        probe = mujoco.MjData(self.model)
        delta = float(np.max(np.abs(target_q_arm-start[self.qpos_addresses])))
        samples = max(self.collision_samples, int(np.ceil(delta/.005))+1)
        if samples > 5000:
            print('COMMAND_REJECTED: preflight requires too many samples')
            return False
        for alpha in np.linspace(0,1,samples):
            probe.qpos[:] = start
            probe.qpos[self.qpos_addresses] = (1-alpha)*start[self.qpos_addresses]+alpha*target_q_arm
            mujoco.mj_forward(self.model,probe)
            from ur5e_sim.control.kinematics import allowed_insertion_contact_pair
            for i in range(probe.ncon):
                c = probe.contact[i]
                g1 = self.model.geom(c.geom1).name or ""
                g2 = self.model.geom(c.geom2).name or ""
                if allowed_insertion_contact_pair(g1, g2,
                                                 model=self.model,
                                                 geom1_id=c.geom1,
                                                 geom2_id=c.geom2):
                    continue
                if not self.plug_grasp_locked and ("held_plug" in g1 or "held_plug" in g2):
                    continue
                pair = {int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])}
                closed_touch = (pair == self.finger_bodies and c.dist >= -.00005 and
                                abs(self.gap_m()) <= .0001)
                if not closed_touch:
                    print('COMMAND_REJECTED: collision', g1, g2, 'distance_m=', float(c.dist))
                    return False
        duration = max(.25,1.5*delta/self.max_joint_speed_rad_s)
        self.last_capture = None
        self.motion_generation += 1
        self.plan = MotionPlan(start[self.qpos_addresses].copy(),target_q_arm.copy(),
                               float(self.data.time),duration)
        self.commanded_pose = Pose(target_pose.position.copy(),
                                   normalize_quaternion(target_pose.quaternion_wxyz))
        set_target_marker(self.model,self.data,self.commanded_pose)
        print('COMMAND_ACCEPTED:', label, f'duration={duration:.3f}s samples={samples}')
        return True

    def request_pose(self, target, *, label):
        if self.plan is not None or self.gripper_state() == 'moving':
            print('COMMAND_REJECTED: BUSY; wait for current motion or send stop')
            return False
        return super().request_pose(target,label=label)

    def apply_controls(self):
        self.motion.tick()
        super().apply_controls()
        step = self.gripper_speed*self.model.opt.timestep
        self.gripper_command += float(np.clip(self.gripper_target-self.gripper_command,-step,step))
        self.data.ctrl[self.gripper_id] = self.gripper_command

    def after_step(self):
        self.motion.after_step()

    def close(self):
        if self.camera is not None:
            self.camera.close()

    def print_pose(self):
        super().print_pose()
        print('GRIPPER_GAP_MM:',1000*self.gap_m())
        print('GRIPPER_STATE:',self.gripper_state())


def rpc(c, request):
    """Task-neutral control API. All mutations occur on the simulation thread."""
    op = request['op']
    if op == 'state': return c.state()
    if op == 'tactile': return c.tactile_read()
    if op == 'stop': c.stop(); return None
    if op == 'preview':
        if c.camera is None: raise ValueError('Scene has no camera')
        c.camera.window(bool(request['open'])); return None
    if op == 'capture':
        if c.camera is None: raise ValueError('Scene has no camera')
        if c.plan is not None or c.motion.busy or c.motion.protected:
            raise ValueError('Capture for localization requires an idle unprotected robot')
        meta = c.camera.capture(c.data, plug_seated=not c.plug_grasp_locked)
        c.last_capture = meta | {'generation':c.motion_generation}
        return c.last_capture
    if op == 'recover':
        c.motion.recover(); c.motion_generation += 1; c.last_capture = None; return None
    if op == 'reset_home':
        c.reset_to_home_keyframe()
        return None
    if op == 'release_plug':
        c.release_preheld_plug()
        return None
    if op == 'seat_plug':
        c.seat_plug_in_socket()
        return None
    if op == 'motion':
        from ur5e_sim.control.kinematics import capture_snapshot_valid, motion_start_qpos_valid

        expected = np.asarray(request['start_qpos'], dtype=float)
        if expected.shape != c.data.qpos.shape or not np.isfinite(expected).all():
            raise ValueError('Motion start snapshot is stale')
        if not motion_start_qpos_valid(expected, c.data.qpos):
            raise ValueError('Motion start snapshot is stale')
        token = request.get('capture_id')
        if token is not None:
            snap = c.last_capture
            if snap is None or snap['capture_id'] != token or snap['generation'] != c.motion_generation:
                raise ValueError('Detection capture is stale or invalidated')
            if not capture_snapshot_valid(snap, c.model, c.data):
                raise ValueError('Detection capture is stale or invalidated')
        samples, settle = request.get('collision_samples',320), request.get('settle_s',.3)
        if type(samples) is not int or not 320 <= samples <= 5000:
            raise ValueError('collision_samples must be 320..5000')
        if not np.isfinite(settle) or not 0 < settle <= 3:
            raise ValueError('settle_s must be in (0,3]')
        c.motion.settings.update(collision_samples=samples,settle_s=settle)
        c.motion.execute(request['targets'], cartesian=request.get('cartesian',False),
                         speed=request.get('speed'), recovery=request.get('recovery'))
        c.motion_generation += 1; c.last_capture = None
        return None
    raise ValueError(f'Unknown operation: {op}')


def dispatch(controller, request):
    """No eval, exec, filesystem writes or arbitrary Python from the client."""
    if not isinstance(request,dict) or type(request.get('id')) is not int:
        return {'ok':False,'error':'Expected object with integer id'}
    rid = request['id']
    if request.get('command') == 'ping':
        return {'id':rid,'ok':True,'pong':True}
    if 'op' in request:
        try:
            result = rpc(controller, request)
            return {'id':rid, 'ok':True, 'result':result, 'state':controller.state()}
        except (ValueError, TypeError, KeyError, RuntimeError, OverflowError) as exc:
            return {'id':rid, 'ok':False, 'error':str(exc), 'state':controller.state()}
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            if 'key' in request:
                key = request['key']
                if not isinstance(key,str) or len(key)!=1 or key.lower() not in 'wsadrfikjlou[],.hp ?':
                    raise ValueError('Unsupported control key')
                handle_key(controller,key.lower())
            else:
                command = request.get('command')
                if not isinstance(command,str) or len(command)>2048:
                    raise ValueError('Expected short command string')
                if command.strip() in ('quit','exit','keymode','keys'):
                    raise ValueError('This command belongs to the control client, not the server')
                if command.strip() == 'status':
                    controller.print_pose()
                elif command.split()[:1] == ['speed']:
                    parts = command.split()
                    if len(parts) != 2:
                        raise ValueError('Use speed RAD_PER_SECOND')
                    speed = float(parts[1])
                    if not np.isfinite(speed) or not 0 < speed <= 2:
                        raise ValueError('Speed must be in (0,2] rad/s')
                    controller.max_joint_speed_rad_s = speed
                    print('MAX_JOINT_SPEED_RAD_S:', speed)
                elif command.split()[:2] == ['gripper','gap']:
                    parts = command.split()
                    if len(parts)!=3:
                        raise ValueError('Use gripper gap MILLIMETRES')
                    gap = float(parts[2])/1000
                    if not np.isfinite(gap) or gap<0:
                        raise ValueError('Opening must be finite and non-negative')
                    q = controller.closed_position+gap/2
                    controller.set_gripper(q,label='gripper gap')
                else:
                    handle_terminal_command(controller,command)
    except (ValueError,TypeError,OverflowError) as exc:
        print('COMMAND_ERROR:',str(exc),file=buffer)
    output = buffer.getvalue().strip()
    if output:
        print(output,flush=True)
    ok = not any(token in output for token in ('COMMAND_ERROR','COMMAND_REJECTED'))
    return {'id':rid,'ok':ok,'output':output,'state':controller.state()}


class Link:
    """One nonblocking localhost TCP client; bounded input and output buffers."""
    def __init__(self, port, controller):
        self.controller = controller
        self.listener = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:
            self.listener.bind(('127.0.0.1',port))
            self.listener.listen(1)
            self.listener.setblocking(False)
        except BaseException:
            self.listener.close()
            raise
        self.sock = None
        self.incoming = bytearray()
        self.outgoing = bytearray()
        self.last_seen = time.monotonic()

    def drop(self, reason):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
            self.incoming.clear(); self.outgoing.clear()
            self.controller.stop()
            print('CLIENT_DISCONNECTED:',reason,flush=True)

    def send(self, message):
        self.outgoing.extend(json.dumps(message,allow_nan=False).encode()+b'\n')
        if len(self.outgoing)>MAX_BYTES:
            self.drop('output buffer full')

    def poll(self):
        try:
            new,addr = self.listener.accept()
        except BlockingIOError:
            pass
        else:
            if self.sock is not None:
                new.close()  # a second client must not take over control
            else:
                self.sock = new
                new.setblocking(False)
                self.last_seen = time.monotonic()
                print('CLIENT_CONNECTED:',addr,flush=True)
                self.send({'event':'ready','state':self.controller.state()})
        if self.sock is None:
            return
        try:
            chunk = self.sock.recv(8192)
            if not chunk:
                self.drop('connection closed'); return
            self.incoming.extend(chunk)
            self.last_seen = time.monotonic()
        except BlockingIOError:
            pass
        except OSError:
            self.drop('receive failed'); return
        if len(self.incoming)>MAX_BYTES:
            self.drop('input buffer full'); return
        if time.monotonic()-self.last_seen>5:
            self.drop('heartbeat timeout'); return
        # One bounded request per physics step; clients wait for each reply.
        if b'\n' in self.incoming:
            line,_,rest = self.incoming.partition(b'\n')
            self.incoming = bytearray(rest)
            try:
                request = json.loads(line)
            except (ValueError,UnicodeError):
                self.send({'ok':False,'error':'Invalid JSON'})
            else:
                self.send(dispatch(self.controller,request))
        if self.sock is not None and self.outgoing:
            try:
                count = self.sock.send(self.outgoing)
                del self.outgoing[:count]
            except BlockingIOError:
                pass
            except OSError:
                self.drop('send failed')

    def close(self):
        self.drop('server stopping')
        self.listener.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,default=ROOT/'scenes/scene_insertion.xml')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--camera-config',type=Path,default=ROOT/'configs/camera.json')
    p.add_argument('--no-preview',action='store_true')
    p.add_argument('--evaluation-dir',type=Path,help='Write simulation truth for offline tests only')
    p.add_argument('--fast',action='store_true',help='Unpaced headless testing')
    p.add_argument('--duration',type=float,help='Optional wall-clock run limit')
    p.add_argument('--show-collisions',action='store_true')
    p.add_argument('--collision-samples',type=int,default=120)
    p.add_argument('--max-joint-speed',type=float,default=.75)
    p.add_argument('--status-hz',type=float,default=1)
    p.add_argument('--tactile-preview', action='store_true',
                   help='Show MuJoCo gel depth map (requires xense gels in scene)')
    p.add_argument('--use-fem-sidecar', action='store_true',
                   help='With --tactile-preview, spawn xensim_py311 FEM window via npz IPC')
    p.add_argument('--xensim-python', type=str, default='',
                   help='Python 3.11 with xensim for FEM sidecar')
    p.add_argument('--tactile-hz', type=float, default=20.0,
                   help='Max rate for gel depth/FEM IPC (default 20; sim physics stays at 500 Hz)')
    p.add_argument('--no-tactile-physics', action='store_true',
                   help='Use L0 socket proxy depth instead of gel–plug boundary (debug)')
    p.add_argument('--debug-proxy-overlay', action='store_true',
                   help='Stack L0 proxy under L1 gel–plug depth window')
    a=p.parse_args()
    if not 1<=a.port<=65534 or a.collision_samples<2:
        p.error('Invalid port or collision sample count')
    if not np.isfinite(a.max_joint_speed) or not 0<a.max_joint_speed<=2:
        p.error('--max-joint-speed must be in (0,2] rad/s')
    if not np.isfinite(a.status_hz) or not 0<=a.status_hz<=20:
        p.error('--status-hz must be in [0,20]')
    if not np.isfinite(a.tactile_hz) or not 1 <= a.tactile_hz <= 60:
        p.error('--tactile-hz must be in [1,60]')
    return a


def main():
    args=parse_args()
    model=mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
    mujoco.mj_forward(model,data)
    controller=Controller(model,data,args)
    if args.show_collisions:
        model.geom_rgba[model.geom_group==3] = [.2,.9,.25,.35]
    link=Link(args.port,controller)
    evaluation = None
    tactile = None
    try:
        if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'ee_camera') >= 0:
            from ur5e_sim.camera.live import CameraService
            controller.camera = CameraService(model,args.scene.resolve(),args.camera_config.resolve(),
                                              args.port+1,show=not(args.headless or args.no_preview))
        if args.evaluation_dir:
            args.evaluation_dir.mkdir(parents=True,exist_ok=True)
            evaluation = (args.evaluation_dir/'physics.jsonl').open('w')
        print('CONFIG_CAMERA:', args.camera_config.resolve(), flush=True)
        if args.tactile_preview or args.use_fem_sidecar:
            from ur5e_sim.tactile.preview import TactilePreview
            tactile = TactilePreview(
                model,
                use_fem_sidecar=args.use_fem_sidecar,
                xensim_python=args.xensim_python,
                show_window=not args.headless,
                tactile_hz=args.tactile_hz,
                tactile_physics=not args.no_tactile_physics,
                debug_proxy_overlay=args.debug_proxy_overlay,
            )
            print('TACTILE_PREVIEW: on (FEM sidecar=%s)' % args.use_fem_sidecar, flush=True)
        viewer_context=contextlib.nullcontext(None)
        if not args.headless:
            from mujoco import viewer
            viewer_context=viewer.launch_passive(model,data)  # NO key_callback
        with viewer_context as view:
            print(f'SERVER_READY: 127.0.0.1:{args.port}',flush=True)
            print('Run apps/debug_pose.py or apps/insert_socket.py in terminal B.',flush=True)
            print('CLOSED_Q_M:',controller.closed_position,flush=True)
            previous_status=0.0
            previous_view=0.0
            started_wall=time.monotonic()
            while view is None or view.is_running():
                if args.duration and time.monotonic()-started_wall >= args.duration:
                    break
                began=time.perf_counter()
                link.poll()
                controller.apply_controls()
                mujoco.mj_step(model,data)
                if not np.all(np.isfinite(data.qpos)):
                    raise RuntimeError('Non-finite physics state')
                # mj_step leaves some derived poses at the preceding state.
                mujoco.mj_forward(model,data)
                controller.after_step()
                if controller.camera is not None:
                    controller.camera.poll(
                        data, plug_seated=not controller.plug_grasp_locked
                    )
                if tactile is not None:
                    tactile.update(model, data)
                if evaluation is not None and controller.insertion_scene:
                    from ur5e_sim.config import site_matrix
                    evaluation.write(json.dumps({'sim_time':float(data.time),
                        'tip':site_matrix(data,'plug_tip').tolist(),
                        'port':site_matrix(data,'socket_port').tolist(),
                        'contact_count':unexpected_contact_count(model, data),
                        'warnings':int(np.sum(data.warning.number)),
                        'motion':controller.motion.state(),
                        'camera':None if controller.camera is None else controller.camera.status})+'\n')
                if view is not None and time.monotonic()-previous_view >= 1/30:
                    view.sync()
                    previous_view=time.monotonic()
                now=time.monotonic()
                if args.status_hz and now-previous_status>=1/args.status_hz:
                    s=controller.state()
                    print('STATE:',json.dumps(s,allow_nan=False),flush=True)
                    previous_status=now
                remaining=model.opt.timestep-(time.perf_counter()-began)
                if remaining>0 and not args.fast:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print('SERVER_INTERRUPTED',flush=True)
    finally:
        link.close()
        controller.close()
        if tactile is not None:
            tactile.close()
        if evaluation is not None: evaluation.close()
    print('SERVER_STOPPED',flush=True)


if __name__=='__main__':
    main()
