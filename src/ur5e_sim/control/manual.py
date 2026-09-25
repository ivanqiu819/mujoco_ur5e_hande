#!/usr/bin/env python3
"""Manual TCP and gripper controls used by the simulation server."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path
import queue
import select
import shlex
import sys
import time

import mujoco
import numpy as np


from ur5e_sim.paths import ROOT as PROJECT_ROOT

from ur5e_sim.control.kinematics import (  # noqa: E402
    ARM_ACTUATORS,
    Pose,
    arm_addresses,
    collision_preflight,
    normalize_quaternion,
    object_id,
    quaternion_multiply,
    set_target_marker,
    site_pose,
    solve_ik,
)


DEFAULT_SCENE = PROJECT_ROOT / "scenes/scene.xml"


@dataclass
class MotionPlan:
    start_q_arm: np.ndarray
    target_q_arm: np.ndarray
    start_time: float
    duration_s: float


def axis_angle_quaternion(axis: int, angle_rad: float) -> np.ndarray:
    result = np.zeros(4, dtype=np.float64)
    result[0] = np.cos(0.5 * angle_rad)
    result[axis + 1] = np.sin(0.5 * angle_rad)
    return result


def world_euler_delta_quaternion(rxyz_deg: np.ndarray) -> np.ndarray:
    """Return an extrinsic world-X/Y/Z rotation as a WXYZ quaternion."""
    rx, ry, rz = np.radians(np.asarray(rxyz_deg, dtype=np.float64))
    qx = axis_angle_quaternion(0, float(rx))
    qy = axis_angle_quaternion(1, float(ry))
    qz = axis_angle_quaternion(2, float(rz))
    return normalize_quaternion(
        quaternion_multiply(qz, quaternion_multiply(qy, qx))
    )


class TeleopController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        collision_samples: int,
        max_joint_speed_rad_s: float,
        linear_step_m: float,
        angular_step_deg: float,
        gripper_step_m: float,
    ) -> None:
        self.model = model
        self.data = data
        self.collision_samples = max(2, int(collision_samples))
        self.max_joint_speed_rad_s = float(max_joint_speed_rad_s)
        self.linear_step_m = float(linear_step_m)
        self.angular_step_deg = float(angular_step_deg)
        self.gripper_step_m = float(gripper_step_m)
        self.quit_requested = False

        self.home_id = object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.tcp_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.slider_id = object_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "Slider_1"
        )
        self.gripper_id = object_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"
        )
        _, self.qpos_addresses, _ = arm_addresses(model)
        self.arm_actuator_ids = np.asarray(
            [
                object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATORS
            ],
            dtype=int,
        )
        self.slider_qpos_address = int(model.jnt_qposadr[self.slider_id])

        self.home_qpos = model.key_qpos[self.home_id].copy()
        home_data = mujoco.MjData(model)
        home_data.qpos[:] = self.home_qpos
        mujoco.mj_forward(model, home_data)
        self.home_pose = site_pose(model, home_data, self.tcp_id)

        self.arm_hold_target = data.qpos[self.qpos_addresses].copy()
        self.gripper_target = float(data.qpos[self.slider_qpos_address])
        self.open_position = float(self.home_qpos[self.slider_qpos_address])
        self.closed_position = 0.0
        self.gripper_lower, self.gripper_upper = self._gripper_limits()
        self.plan: MotionPlan | None = None
        self.commanded_pose = self.current_pose()

        data.ctrl[self.arm_actuator_ids] = self.arm_hold_target
        data.ctrl[self.gripper_id] = self.gripper_target
        set_target_marker(model, data, self.commanded_pose)
        mujoco.mj_forward(model, data)

    def _gripper_limits(self) -> tuple[float, float]:
        if self.model.actuator_ctrllimited[self.gripper_id]:
            lower, upper = self.model.actuator_ctrlrange[self.gripper_id]
            return float(lower), float(upper)
        return -0.025, 0.025

    def current_pose(self) -> Pose:
        return site_pose(self.model, self.data, self.tcp_id)

    def _accept_arm_target(
        self,
        target_q_arm: np.ndarray,
        target_pose: Pose,
        *,
        label: str,
    ) -> bool:
        start_qpos = self.data.qpos.copy()
        collision_free, collision = collision_preflight(
            self.model,
            start_qpos,
            target_q_arm,
            self.collision_samples,
        )
        if not collision_free:
            print(f"COMMAND_REJECTED: {label}: collision: {collision}")
            return False

        start_q_arm = start_qpos[self.qpos_addresses].copy()
        max_delta = float(np.max(np.abs(target_q_arm - start_q_arm)))
        # Smoothstep has a peak derivative of 1.5; account for it when setting
        # a duration from the requested joint-speed limit.
        duration_s = max(
            0.25,
            1.5 * max_delta / max(self.max_joint_speed_rad_s, 1.0e-6),
        )
        self.plan = MotionPlan(
            start_q_arm=start_q_arm,
            target_q_arm=np.asarray(target_q_arm, dtype=np.float64).copy(),
            start_time=float(self.data.time),
            duration_s=duration_s,
        )
        self.commanded_pose = Pose(
            np.asarray(target_pose.position, dtype=np.float64).copy(),
            normalize_quaternion(target_pose.quaternion_wxyz),
        )
        set_target_marker(self.model, self.data, self.commanded_pose)
        print(
            "COMMAND_ACCEPTED:",
            label,
            f"duration={duration_s:.3f}s",
            f"max_joint_delta={max_delta:.6f}rad",
        )
        return True

    def request_pose(self, target: Pose, *, label: str) -> bool:
        position = np.asarray(target.position, dtype=np.float64).reshape(3)
        quaternion = normalize_quaternion(target.quaternion_wxyz)
        if not np.all(np.isfinite(position)):
            print(f"COMMAND_REJECTED: {label}: position is not finite")
            return False

        normalized_target = Pose(position, quaternion)
        result = solve_ik(
            self.model,
            self.data.qpos.copy(),
            normalized_target,
        )
        print(
            "IK_RESULT:",
            f"converged={result.converged}",
            f"iterations={result.iterations}",
            f"position_error={result.position_error_m:.8g}m",
            f"orientation_error={result.orientation_error_deg:.8g}deg",
        )
        if not result.converged:
            print(f"COMMAND_REJECTED: {label}: IK did not converge")
            return False
        return self._accept_arm_target(
            result.q_arm,
            normalized_target,
            label=label,
        )

    def move_relative(self, delta_xyz_m: np.ndarray, *, label: str) -> bool:
        delta = np.asarray(delta_xyz_m, dtype=np.float64).reshape(3)
        return self.request_pose(
            Pose(
                self.commanded_pose.position + delta,
                self.commanded_pose.quaternion_wxyz.copy(),
            ),
            label=label,
        )

    def rotate_relative(self, rxyz_deg: np.ndarray, *, label: str) -> bool:
        delta_quaternion = world_euler_delta_quaternion(rxyz_deg)
        target_quaternion = quaternion_multiply(
            delta_quaternion,
            self.commanded_pose.quaternion_wxyz,
        )
        return self.request_pose(
            Pose(
                self.commanded_pose.position.copy(),
                normalize_quaternion(target_quaternion),
            ),
            label=label,
        )

    def go_home(self) -> bool:
        target_q_arm = self.home_qpos[self.qpos_addresses].copy()
        return self._accept_arm_target(
            target_q_arm,
            self.home_pose,
            label="home",
        )

    def stop(self) -> None:
        self.plan = None
        self.arm_hold_target = self.data.qpos[self.qpos_addresses].copy()
        self.commanded_pose = self.current_pose()
        set_target_marker(self.model, self.data, self.commanded_pose)
        print("MOTION_STOPPED: holding current joint positions")

    def set_gripper(self, position_m: float, *, label: str) -> bool:
        position = float(position_m)
        if not np.isfinite(position):
            print(f"COMMAND_REJECTED: {label}: gripper position is not finite")
            return False
        if not self.gripper_lower <= position <= self.gripper_upper:
            print(
                f"COMMAND_REJECTED: {label}: expected "
                f"[{self.gripper_lower:.6f}, {self.gripper_upper:.6f}] m"
            )
            return False
        self.gripper_target = position
        print(f"GRIPPER_TARGET_M: {position:.6f}")
        return True

    def apply_controls(self) -> None:
        if self.plan is not None:
            elapsed = max(0.0, float(self.data.time) - self.plan.start_time)
            normalized_time = min(1.0, elapsed / self.plan.duration_s)
            alpha = normalized_time**2 * (3.0 - 2.0 * normalized_time)
            arm_command = (
                (1.0 - alpha) * self.plan.start_q_arm
                + alpha * self.plan.target_q_arm
            )
            if normalized_time >= 1.0:
                self.arm_hold_target = self.plan.target_q_arm.copy()
                self.plan = None
                print("TRAJECTORY_COMMAND_COMPLETE")
        else:
            arm_command = self.arm_hold_target

        self.data.ctrl[self.arm_actuator_ids] = arm_command
        self.data.ctrl[self.gripper_id] = self.gripper_target

    def print_pose(self) -> None:
        actual = self.current_pose()
        print("TCP_POSITION_M:", actual.position.tolist())
        print("TCP_QUAT_WXYZ:", actual.quaternion_wxyz.tolist())
        print("TCP_TARGET_POSITION_M:", self.commanded_pose.position.tolist())
        print(
            "TCP_TARGET_QUAT_WXYZ:",
            self.commanded_pose.quaternion_wxyz.tolist(),
        )
        print(
            "GRIPPER_ACTUAL_M:",
            float(self.data.qpos[self.slider_qpos_address]),
        )
        print("GRIPPER_TARGET_M:", self.gripper_target)

    def print_joints(self) -> None:
        print(
            "ARM_QPOS_RAD:",
            self.data.qpos[self.qpos_addresses].tolist(),
        )
        print(
            "ARM_COMMAND_RAD:",
            self.data.ctrl[self.arm_actuator_ids].tolist(),
        )

    def print_contacts(self) -> None:
        print("CONTACT_COUNT:", int(self.data.ncon))
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_a = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            )
            geom_b = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            )
            print(
                f"CONTACT[{index}]:",
                geom_a,
                geom_b,
                f"distance={contact.dist:.8g}",
            )


def print_help() -> None:
    print(
        """
TERMINAL COMMANDS
  pose                              print actual and target TCP/gripper state
  joints                            print actual and commanded arm joints
  contacts                          print current MuJoCo contacts
  tcp X Y Z                         absolute world position; keep target attitude
  tcp X Y Z QW QX QY QZ             absolute world pose (metres, WXYZ)
  tcp-rel DX DY DZ                  relative world translation (metres)
  rotate-rel RX RY RZ               relative world-axis rotation (degrees)
  gripper POSITION                  slider target in metres
  gripper open | close              named gripper targets
  step LINEAR_MM ANGULAR_DEG GRIPPER_MM
                                    change keyboard increments
  speed JOINT_RAD_S                 change maximum smoothstep joint speed
  home                              collision-check and move to home
  stop                              stop arm trajectory and hold current joints
  help                              show this help
  quit                              headless: exit; viewer: close its window

VIEWER KEYS (viewer must have keyboard focus)
  W/S: +X/-X       A/D: +Y/-Y       R/F: +Z/-Z
  I/K: +Rx/-Rx     J/L: +Ry/-Ry     U/O: +Rz/-Rz
  [: close         ]: open           H: home
  P: pose          Space: stop       ?: help
""".strip()
    )


def parse_floats(values: list[str], expected: int, command: str) -> np.ndarray:
    if len(values) != expected:
        raise ValueError(f"{command} expects {expected} numeric arguments")
    result = np.asarray([float(value) for value in values], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{command} arguments must be finite")
    return result


def handle_terminal_command(controller: TeleopController, line: str) -> None:
    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        print("COMMAND_ERROR:", exc)
        return
    if not tokens:
        return

    command = tokens[0].lower()
    arguments = tokens[1:]
    try:
        if command == "pose":
            if arguments:
                raise ValueError("pose takes no arguments")
            controller.print_pose()
        elif command == "joints":
            if arguments:
                raise ValueError("joints takes no arguments")
            controller.print_joints()
        elif command == "contacts":
            if arguments:
                raise ValueError("contacts takes no arguments")
            controller.print_contacts()
        elif command == "tcp":
            if len(arguments) not in (3, 7):
                raise ValueError("tcp expects X Y Z [QW QX QY QZ]")
            position = parse_floats(arguments[:3], 3, "tcp")
            quaternion = (
                controller.commanded_pose.quaternion_wxyz.copy()
                if len(arguments) == 3
                else normalize_quaternion(
                    parse_floats(arguments[3:], 4, "tcp quaternion")
                )
            )
            controller.request_pose(
                Pose(position, quaternion),
                label="tcp absolute",
            )
        elif command in ("tcp-rel", "move-rel"):
            delta = parse_floats(arguments, 3, command)
            controller.move_relative(delta, label=f"{command} {delta.tolist()}")
        elif command in ("rotate-rel", "rot-rel"):
            rotation = parse_floats(arguments, 3, command)
            controller.rotate_relative(
                rotation,
                label=f"{command} {rotation.tolist()} deg",
            )
        elif command == "gripper":
            if len(arguments) != 1:
                raise ValueError("gripper expects POSITION, open, or close")
            value = arguments[0].lower()
            if value == "open":
                target = controller.open_position
            elif value == "close":
                target = controller.closed_position
            else:
                target = float(arguments[0])
            controller.set_gripper(target, label=f"gripper {value}")
        elif command == "step":
            values = parse_floats(arguments, 3, "step")
            if np.any(values <= 0.0):
                raise ValueError("all step values must be positive")
            controller.linear_step_m = float(values[0]) / 1000.0
            controller.angular_step_deg = float(values[1])
            controller.gripper_step_m = float(values[2]) / 1000.0
            print(
                "STEP_UPDATED:",
                f"linear={values[0]:g}mm",
                f"angular={values[1]:g}deg",
                f"gripper={values[2]:g}mm",
            )
        elif command == "speed":
            values = parse_floats(arguments, 1, "speed")
            if values[0] <= 0.0:
                raise ValueError("speed must be positive")
            controller.max_joint_speed_rad_s = float(values[0])
            print("MAX_JOINT_SPEED_RAD_S:", controller.max_joint_speed_rad_s)
        elif command == "home":
            if arguments:
                raise ValueError("home takes no arguments")
            controller.go_home()
        elif command == "stop":
            if arguments:
                raise ValueError("stop takes no arguments")
            controller.stop()
        elif command in ("help", "?"):
            print_help()
        elif command in ("quit", "exit"):
            controller.quit_requested = True
            print("QUIT_REQUESTED")
        else:
            print(f"COMMAND_ERROR: unknown command: {command!r}; type 'help'")
    except (ValueError, TypeError) as exc:
        print("COMMAND_ERROR:", exc)


def handle_key(controller: TeleopController, key: str) -> None:
    linear = controller.linear_step_m
    angular = controller.angular_step_deg
    translations = {
        "w": np.array([linear, 0.0, 0.0]),
        "s": np.array([-linear, 0.0, 0.0]),
        "a": np.array([0.0, linear, 0.0]),
        "d": np.array([0.0, -linear, 0.0]),
        "r": np.array([0.0, 0.0, linear]),
        "f": np.array([0.0, 0.0, -linear]),
    }
    rotations = {
        "i": np.array([angular, 0.0, 0.0]),
        "k": np.array([-angular, 0.0, 0.0]),
        "j": np.array([0.0, angular, 0.0]),
        "l": np.array([0.0, -angular, 0.0]),
        "u": np.array([0.0, 0.0, angular]),
        "o": np.array([0.0, 0.0, -angular]),
    }

    if key in translations:
        controller.move_relative(
            translations[key],
            label=f"key {key.upper()}",
        )
    elif key in rotations:
        controller.rotate_relative(
            rotations[key],
            label=f"key {key.upper()}",
        )
    elif key == "[":
        controller.set_gripper(
            max(
                controller.gripper_lower,
                controller.gripper_target - controller.gripper_step_m,
            ),
            label="key [",
        )
    elif key == "]":
        controller.set_gripper(
            min(
                controller.gripper_upper,
                controller.gripper_target + controller.gripper_step_m,
            ),
            label="key ]",
        )
    elif key == "h":
        controller.go_home()
    elif key == "p":
        controller.print_pose()
    elif key == " ":
        controller.stop()
    elif key == "?":
        print_help()
