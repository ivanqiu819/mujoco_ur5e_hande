"""Server-side validated trajectory execution, independent of task and vision."""
from collections import deque
import mujoco
import numpy as np
from ur5e_sim.config import matrix_pose, site_matrix
from ur5e_sim.control.manual import MotionPlan
from ur5e_sim.control.kinematics import (
    allowed_insertion_contact_pair,
    arm_addresses,
    collision_preflight,
    set_target_marker,
    solve_ik,
    tcp_pose_matches,
)
from ur5e_sim.control.limits import cartesian_speed


def rigid(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('Expected finite 4x4 transform')
    r = t[:3, :3]
    if (not np.allclose(t[3], [0,0,0,1], atol=1e-8) or
            not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or np.linalg.det(r) < .999):
        raise ValueError('Expected rigid transform')
    return t


def _geom_name(model, geom_id):
    return model.geom(geom_id).name or ""


def _unexpected_contact_pairs(model, data, *, plug_released: bool = False):
    """Contacts outside the insertion tactile whitelist."""
    pairs = []
    for contact in data.contact:
        g1 = _geom_name(model, contact.geom1)
        g2 = _geom_name(model, contact.geom2)
        if allowed_insertion_contact_pair(g1, g2, model=model,
                                         geom1_id=contact.geom1,
                                         geom2_id=contact.geom2):
            continue
        if plug_released and ("held_plug" in g1 or "held_plug" in g2):
            continue
        pairs.append((g1, g2))
    return pairs


def unexpected_contact_count(model, data) -> int:
    return len(_unexpected_contact_pairs(model, data))


def line_targets(start, end, spacing=.001):
    start, end = rigid(start), rigid(end)
    angle = np.arccos(np.clip((np.trace(end[:3,:3] @ start[:3,:3].T)-1)/2, -1, 1))
    if angle > np.deg2rad(.1):
        raise ValueError('Tool attitude changed; straight recovery refused')
    count = max(1, int(np.ceil(np.linalg.norm(end[:3,3]-start[:3,3])/spacing)))
    if count > 256:
        raise ValueError('Cartesian path exceeds 256 points')
    result = []
    for alpha in np.linspace(0, 1, count+1)[1:]:
        t = end.copy(); t[:3,3] = (1-alpha)*start[:3,3]+alpha*end[:3,3]; result.append(t)
    return result


class TrajectoryExecutor:
    def __init__(self, controller, settings):
        self.c, self.settings = controller, settings
        self.queue = deque()
        self.phase, self.reason = 'idle', ''
        self.completion = self.settle_started = self.motion_ended = None
        self.cartesian = False
        self.protected = False
        self.recovery_target = None

    @property
    def busy(self):
        return self.phase == 'moving'

    def state(self):
        return {'phase': self.phase, 'reason': self.reason, 'protected': self.protected,
                'recovery_target': None if self.recovery_target is None else self.recovery_target.tolist()}

    def cancel(self, reason='stopped'):
        self.queue.clear()
        self.completion = self.settle_started = self.motion_ended = None
        self.phase, self.reason = 'stopped', reason

    def fail(self, message):
        self.c.stop()
        self.phase, self.reason = 'error', message
        print('MOTION_ERROR:', message, flush=True)

    def _gripper_blocks_arm_motion(self) -> bool:
        if self.c.gripper_state() != 'moving':
            return False
        # Preheld plug: sliders comply under gel preload; arm paths must not stall.
        if getattr(self.c, 'insertion_scene', False) and getattr(self.c, 'plug_grasp_locked', False):
            return False
        return True

    def execute(self, targets, *, cartesian=False, speed=None, recovery=None):
        if (self.busy or self.c.plan is not None or self.protected
                or self._gripper_blocks_arm_motion()):
            raise ValueError('BUSY or protected trajectory; stop/recover first')
        if not isinstance(targets, list) or not 1 <= len(targets) <= 256:
            raise ValueError('Expected 1..256 target transforms')
        targets = [rigid(t) for t in targets]
        if cartesian:
            previous = site_matrix(self.c.data, 'tcp')
            for t in targets:
                if np.linalg.norm(t[:3,3]-previous[:3,3]) > .001001:
                    raise ValueError('Cartesian spacing exceeds 1 mm')
                if not np.allclose(t[:3,:3], targets[0][:3,:3], atol=1e-8):
                    raise ValueError('Cartesian path must keep a fixed orientation')
                line_targets(previous, t)
                previous = t
        if speed is not None:
            self.settings['insert_speed_m_s'] = cartesian_speed(speed)
        recovery = None if recovery is None else rigid(recovery)
        if recovery is not None:
            if not cartesian:
                raise ValueError('Protected motion requires a Cartesian path')
            # Recovery must be the measured start of this straight move.
            if not tcp_pose_matches(recovery, site_matrix(self.c.data, 'tcp')):
                raise ValueError('Recovery pose must match the measured path start')
        self.schedule(targets, 'moving', 'done', cartesian)
        if recovery is not None:
            self.recovery_target = recovery.copy()
            self.protected = True

    def recover(self):
        if self.busy or self.c.plan is not None:
            raise ValueError('Stop the current motion before recovery')
        if not self.protected or self.recovery_target is None:
            raise ValueError('No protected motion to recover')
        current = site_matrix(self.c.data, 'tcp')
        goal = self.recovery_target.copy()
        # Compliant grasp may rotate TCP slightly during insert; retract along position only.
        goal[:3, :3] = current[:3, :3]
        self.schedule(line_targets(current, goal), 'moving', 'recovered', True)

    def schedule(self, targets, phase, completion, cartesian=False):
        c = self.c
        self.cartesian = cartesian
        released = hasattr(c, 'plug_grasp_locked') and not c.plug_grasp_locked
        if _unexpected_contact_pairs(c.model, c.data, plug_released=released):
            raise ValueError('Current state has contacts; motion refused')
        seed = c.data.qpos.copy()
        previous = site_matrix(c.data, 'tcp')
        segments = []
        for target in targets:
            pose = matrix_pose(target)
            result = solve_ik(c.model, seed, pose, position_tolerance_m=2e-6,
                              orientation_tolerance_deg=.005)
            if not result.converged:
                raise ValueError(f'{phase}: IK did not converge')
            start = seed[c.qpos_addresses].copy()
            delta = float(np.max(np.abs(result.q_arm-start)))
            samples = max(self.settings['collision_samples'], int(np.ceil(delta/.002))+1)
            if samples > 5000:
                raise ValueError('Path requires too many collision samples')
            ok, reason = collision_preflight(c.model, seed, result.q_arm, samples,
                                              plug_released=released)
            if not ok:
                raise ValueError(f'{phase}: collision: {reason}')
            duration = max(.05 if cartesian else .25, 1.5*delta/c.max_joint_speed_rad_s)
            if cartesian:
                duration = max(duration, 1.5*np.linalg.norm(target[:3, 3]-previous[:3, 3])/self.settings['insert_speed_m_s'])
            segments.append((result.q_arm.copy(), pose, duration))
            seed[c.qpos_addresses] = result.q_arm
            previous = target
        self.queue = deque(segments)
        self.phase, self.completion, self.reason = phase, completion, ''
        self.settle_started = self.motion_ended = None
        print('COMMAND_ACCEPTED:', phase, 'segments=', len(segments))

    def tick(self):
        c = self.c
        try:
            if not self.busy or c.plan is not None:
                return
            # Wait for actual tracking, not just the end of an actuator command.
            _, _, arm_vadr = arm_addresses(c.model)
            error = float(np.max(np.abs(c.data.qpos[c.qpos_addresses] - c.arm_hold_target)))
            speed = float(np.max(np.abs(c.data.qvel[arm_vadr])))
            if self.motion_ended is None:
                self.motion_ended = float(c.data.time)
            if error > 0.002 or speed > 0.001:
                self.settle_started = None
                if c.data.time-self.motion_ended > 3:
                    raise ValueError('Trajectory did not settle')
                return
            if self.settle_started is None:
                self.settle_started = float(c.data.time)
            # Interpolated insertion knots need no extra dwell once tracking is settled.
            dwell = 0 if self.queue and self.cartesian else self.settings['settle_s']
            if c.data.time-self.settle_started < dwell:
                return
            if self.queue:
                q, pose, duration = self.queue.popleft()
                # Revalidate from the measured state before every segment.
                released = hasattr(c, 'plug_grasp_locked') and not c.plug_grasp_locked
                ok, reason = collision_preflight(c.model, c.data.qpos.copy(), q,
                                                 self.settings['collision_samples'],
                                                 plug_released=released)
                if not ok:
                    raise ValueError(f'Execution preflight failed: {reason}')
                c.plan = MotionPlan(c.data.qpos[c.qpos_addresses].copy(), q,
                                    float(c.data.time), duration)
                c.commanded_pose = pose
                set_target_marker(c.model, c.data, pose)
                self.settle_started = self.motion_ended = None
            else:
                self.phase = self.completion
                self.completion = None
                if self.phase == 'recovered':
                    self.protected = False
                    self.recovery_target = None
                print('MOTION_PHASE:', self.phase, flush=True)
        except (ValueError, RuntimeError, OSError, EOFError) as exc:
            self.fail(str(exc))

    def after_step(self):
        c = self.c
        monitor = self.busy or self.protected
        if not monitor and c.insertion_scene and c.plug_grasp_locked:
            monitor = True
        if monitor and self.phase != 'error':
            released = hasattr(c, 'plug_grasp_locked') and not c.plug_grasp_locked
            pairs = _unexpected_contact_pairs(c.model, c.data, plug_released=released)
            if pairs:
                self.fail(f'Unexpected contact during dynamics: {pairs[:3]}')
