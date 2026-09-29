#!/usr/bin/env python3
"""Move the migrated UR5e TCP to a requested world-frame pose.

This first migration test uses MuJoCo's analytic site Jacobian and damped
least-squares IK.  Before execution it checks a straight interpolation in
joint space for contacts.  It is deliberately not advertised as a global
obstacle-avoidance planner; OMPL is the next migration layer for that job.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import numpy as np


from ur5e_sim.paths import ROOT as PROJECT_ROOT
DEFAULT_SCENE = PROJECT_ROOT / "scenes/scene.xml"
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
ARM_ACTUATORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray


@dataclass(frozen=True)
class IkResult:
    q_arm: np.ndarray
    iterations: int
    position_error_m: float
    orientation_error_deg: float
    converged: bool


def normalize_quaternion(q) -> np.ndarray:
    result = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(result))
    if norm < 1.0e-12 or not np.all(np.isfinite(result)):
        raise ValueError("Quaternion must be finite and non-zero")
    return result / norm


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = left
    w1, x1, y1, z1 = right
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=np.float64,
    )


def orientation_error_rotvec_world(
    current_wxyz: np.ndarray,
    target_wxyz: np.ndarray,
) -> np.ndarray:
    current = normalize_quaternion(current_wxyz)
    target = normalize_quaternion(target_wxyz)
    error = quaternion_multiply(
        target,
        np.array([current[0], -current[1], -current[2], -current[3]]),
    )
    if error[0] < 0.0:
        error = -error

    vector_norm = float(np.linalg.norm(error[1:]))
    if vector_norm < 1.0e-12:
        return np.zeros(3, dtype=np.float64)

    angle = 2.0 * np.arctan2(vector_norm, np.clip(error[0], -1.0, 1.0))
    return error[1:] * (angle / vector_norm)


def object_id(model: mujoco.MjModel, obj_type, name: str) -> int:
    result = mujoco.mj_name2id(model, obj_type, name)
    if result < 0:
        raise RuntimeError(f"MuJoCo object does not exist: {name}")
    return result


def site_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
) -> Pose:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
    return Pose(data.site_xpos[site_id].copy(), quaternion)


def arm_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.array(
        [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS],
        dtype=int,
    )
    qpos_addresses = model.jnt_qposadr[joint_ids].copy()
    dof_addresses = model.jnt_dofadr[joint_ids].copy()
    return joint_ids, qpos_addresses, dof_addresses


def solve_ik(
    model: mujoco.MjModel,
    seed_qpos: np.ndarray,
    target: Pose,
    *,
    max_iterations: int = 400,
    position_tolerance_m: float = 5.0e-5,
    orientation_tolerance_deg: float = 0.10,
) -> IkResult:
    data = mujoco.MjData(model)
    data.qpos[:] = seed_qpos
    _, qpos_addresses, dof_addresses = arm_addresses(model)
    site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    jac_position = np.zeros((3, model.nv), dtype=np.float64)
    jac_rotation = np.zeros((3, model.nv), dtype=np.float64)

    position_error = float("inf")
    orientation_error_deg = float("inf")
    converged = False

    for iteration in range(1, max_iterations + 1):
        mujoco.mj_forward(model, data)
        current = site_pose(model, data, site_id)
        position_vector = target.position - current.position
        rotation_vector = orientation_error_rotvec_world(
            current.quaternion_wxyz,
            target.quaternion_wxyz,
        )
        position_error = float(np.linalg.norm(position_vector))
        orientation_error_deg = float(np.degrees(np.linalg.norm(rotation_vector)))

        if (
            position_error <= position_tolerance_m
            and orientation_error_deg <= orientation_tolerance_deg
        ):
            converged = True
            break

        mujoco.mj_jacSite(model, data, jac_position, jac_rotation, site_id)
        # Scaling orientation rows keeps metres and radians numerically balanced.
        orientation_weight = 0.35
        jacobian = np.vstack(
            (
                jac_position[:, dof_addresses],
                orientation_weight * jac_rotation[:, dof_addresses],
            )
        )
        error = np.concatenate(
            (position_vector, orientation_weight * rotation_vector)
        )

        damping = 2.0e-3
        system = jacobian @ jacobian.T + (damping**2) * np.eye(6)
        delta = jacobian.T @ np.linalg.solve(system, error)
        delta = np.clip(delta, -0.08, 0.08)
        data.qpos[qpos_addresses] += 0.65 * delta

        for joint_id, qpos_address in zip(
            [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS],
            qpos_addresses,
        ):
            if model.jnt_limited[joint_id]:
                lower, upper = model.jnt_range[joint_id]
                data.qpos[qpos_address] = np.clip(
                    data.qpos[qpos_address],
                    lower + 1.0e-6,
                    upper - 1.0e-6,
                )

    return IkResult(
        q_arm=data.qpos[qpos_addresses].copy(),
        iterations=iteration,
        position_error_m=position_error,
        orientation_error_deg=orientation_error_deg,
        converged=converged,
    )


def qpos_core_grasp_errors(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    """Max |Δq| for arm+sliders vs compliant grasp DOF."""
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    if ref.shape != cur.shape:
        return float("inf"), float("inf")
    n_grasp = 6 if ref.size >= 14 else 0
    if n_grasp:
        core_err = float(np.max(np.abs(cur[:-n_grasp] - ref[:-n_grasp])))
        grasp_err = float(np.max(np.abs(cur[-n_grasp:] - ref[-n_grasp:])))
        return core_err, grasp_err
    return float(np.max(np.abs(cur - ref))), 0.0


def tcp_pose_drift(snapshot: dict, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    """TCP drift (mm, deg) between snapshot fields and current simulation."""
    if "tcp_position_m" not in snapshot:
        return float("inf"), float("inf")
    pos_mm = float(
        np.linalg.norm(
            np.asarray(snapshot["tcp_position_m"], dtype=float) - data.site("tcp").xpos
        )
        * 1000.0
    )
    snap_q = np.asarray(snapshot["tcp_quat_wxyz"], dtype=float)
    cur_q = np.empty(4)
    mujoco.mju_mat2Quat(cur_q, data.site("tcp").xmat.reshape(9))
    snap_r = np.empty(9)
    cur_r = np.empty(9)
    mujoco.mju_quat2Mat(snap_r, snap_q)
    mujoco.mju_quat2Mat(cur_r, cur_q)
    angle_deg = float(
        np.degrees(
            np.arccos(
                np.clip((np.trace(snap_r.reshape(3, 3) @ cur_r.reshape(3, 3).T) - 1.0) / 2.0, -1.0, 1.0)
            )
        )
    )
    return pos_mm, angle_deg


def tcp_pose_matches(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    pos_tol_m: float = 0.0015,
    rot_tol_deg: float = 1.5,
) -> bool:
    ref = np.asarray(reference, dtype=float).reshape(4, 4)
    cur = np.asarray(current, dtype=float).reshape(4, 4)
    if np.linalg.norm(ref[:3, 3] - cur[:3, 3]) > pos_tol_m:
        return False
    angle_deg = float(
        np.degrees(
            np.arccos(
                np.clip((np.trace(cur[:3, :3] @ ref[:3, :3].T) - 1.0) / 2.0, -1.0, 1.0)
            )
        )
    )
    return angle_deg <= rot_tol_deg


def motion_start_qpos_valid(reference: np.ndarray, current: np.ndarray) -> bool:
    core_err, grasp_err = qpos_core_grasp_errors(reference, current)
    return core_err <= 0.004 and grasp_err <= 0.015


def capture_snapshot_valid(snapshot: dict, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    pos_mm, angle_deg = tcp_pose_drift(snapshot, model, data)
    if np.isfinite(pos_mm) and pos_mm <= 1.5 and angle_deg <= 1.0:
        return True
    core_err, grasp_err = qpos_core_grasp_errors(snapshot["qpos"], data.qpos)
    return core_err <= 0.004 and grasp_err <= 0.015


def _arm_body_ids(model) -> frozenset[int]:
    """Body IDs on the UR5e arm + Hand-E gripper kinematic chain (cached)."""
    if not hasattr(_arm_body_ids, "_cache") or _arm_body_ids._model_ptr != id(model):
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        ids = set()
        def _collect(bid):
            ids.add(bid)
            for i in range(model.nbody):
                if model.body_parentid[i] == bid:
                    _collect(i)
        if base >= 0:
            _collect(base)
        _arm_body_ids._cache = frozenset(ids)
        _arm_body_ids._model_ptr = id(model)
    return _arm_body_ids._cache


def allowed_insertion_contact_pair(geom_a: str, geom_b: str,
                                   model=None, geom1_id: int = -1,
                                   geom2_id: int = -1) -> bool:
    """Contacts ignored during IK preflight and unexpected-contact filtering.

    When *model* is provided, uses the body kinematic chain to whitelist any
    contact between the plug and the arm/gripper assembly.  Falls back to
    name patterns when model is not available.
    """
    if "xense_gel" in geom_a or "xense_gel" in geom_b:
        return True
    plug = "held_plug" in geom_a or "held_plug" in geom_b
    if not plug:
        return False
    other = geom_b if "held_plug" in geom_a else geom_a
    other_id = geom2_id if "held_plug" in geom_a else geom1_id
    # Body-based check: any geom on the arm/gripper chain is allowed.
    if model is not None and other_id >= 0:
        if int(model.geom_bodyid[other_id]) in _arm_body_ids(model):
            return True
    # Name-based fallback (for callers without model access).
    if "socket_wall" in other:
        return True
    if "finger" in other and "part" in other:
        return True
    if "finger_collision" in other:
        return True
    if "screw" in other and ("left" in other or "right" in other):
        return True
    if "hande" in other and "collision" in other:
        return True
    return False


def collision_preflight(
    model: mujoco.MjModel,
    start_qpos: np.ndarray,
    target_q_arm: np.ndarray,
    samples: int,
    *,
    plug_released: bool = False,
) -> tuple[bool, str | None]:
    data = mujoco.MjData(model)
    _, qpos_addresses, _ = arm_addresses(model)

    for index, alpha in enumerate(np.linspace(0.0, 1.0, samples)):
        data.qpos[:] = start_qpos
        data.qpos[qpos_addresses] = (
            (1.0 - alpha) * start_qpos[qpos_addresses]
            + alpha * target_q_arm
        )
        mujoco.mj_forward(model, data)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom_a = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            ) or ""
            geom_b = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            ) or ""
            if allowed_insertion_contact_pair(geom_a, geom_b,
                                             model=model,
                                             geom1_id=contact.geom1,
                                             geom2_id=contact.geom2):
                continue
            if plug_released and ("held_plug" in geom_a or "held_plug" in geom_b):
                continue
            message = (
                f"sample={index}/{samples - 1}, alpha={alpha:.4f}, "
                f"geoms=({geom_a}, {geom_b}), distance={contact.dist:.6g}"
            )
            return False, message
    return True, None


def set_target_marker(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: Pose,
) -> None:
    body_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "tcp_target")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError("tcp_target is not a mocap body")
    data.mocap_pos[mocap_id] = target.position
    data.mocap_quat[mocap_id] = target.quaternion_wxyz


def execute(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_q_arm: np.ndarray,
    duration_s: float,
    hold_s: float,
    show_viewer: bool,
) -> None:
    _, qpos_addresses, _ = arm_addresses(model)
    actuator_ids = np.array(
        [
            object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ARM_ACTUATORS
        ],
        dtype=int,
    )
    gripper_id = object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    slider_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "Slider_1")
    slider_qpos_address = int(model.jnt_qposadr[slider_id])
    start_q_arm = data.qpos[qpos_addresses].copy()
    gripper_command = float(data.qpos[slider_qpos_address])

    viewer_context = contextlib.nullcontext(None)
    if show_viewer:
        from mujoco import viewer as mujoco_viewer

        viewer_context = mujoco_viewer.launch_passive(model, data)

    with viewer_context as viewer:
        steps = max(1, round(duration_s / model.opt.timestep))
        for step in range(steps):
            normalized_time = (step + 1) / steps
            alpha = normalized_time**2 * (3.0 - 2.0 * normalized_time)
            data.ctrl[actuator_ids] = (
                (1.0 - alpha) * start_q_arm + alpha * target_q_arm
            )
            data.ctrl[gripper_id] = gripper_command
            mujoco.mj_step(model, data)
            if viewer is not None:
                if not viewer.is_running():
                    raise RuntimeError("Viewer was closed during motion")
                viewer.sync()
                time.sleep(model.opt.timestep)

        hold_steps = max(0, round(hold_s / model.opt.timestep))
        for _ in range(hold_steps):
            data.ctrl[actuator_ids] = target_q_arm
            data.ctrl[gripper_id] = gripper_command
            mujoco.mj_step(model, data)

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(model.opt.timestep)

        if viewer is not None and viewer.is_running():
            print("MOTION_FINISHED: close the viewer window to exit")
            while viewer.is_running():
                data.ctrl[actuator_ids] = target_q_arm
                data.ctrl[gripper_id] = gripper_command
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="absolute world-frame TCP position in metres",
    )
    target_group.add_argument(
        "--offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.02],
        metavar=("DX", "DY", "DZ"),
        help="world-frame offset from home (default: 0 0 0.02)",
    )
    parser.add_argument(
        "--quaternion",
        nargs=4,
        type=float,
        metavar=("W", "X", "Y", "Z"),
        help="target WXYZ quaternion; omitted means preserve orientation",
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--collision-samples", type=int, default=160)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = args.scene.resolve()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    home_id = object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    tcp_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    start = site_pose(model, data, tcp_id)

    if args.position is not None:
        target_position = np.asarray(args.position, dtype=np.float64)
    else:
        target_position = start.position + np.asarray(args.offset, dtype=np.float64)
    target_quaternion = (
        start.quaternion_wxyz
        if args.quaternion is None
        else normalize_quaternion(args.quaternion)
    )
    target = Pose(target_position, target_quaternion)
    set_target_marker(model, data, target)

    print("SCENE:", scene_path)
    print("TCP_START_POSITION_M:", start.position.tolist())
    print("TCP_START_QUAT_WXYZ:", start.quaternion_wxyz.tolist())
    print("TCP_TARGET_POSITION_M:", target.position.tolist())
    print("TCP_TARGET_QUAT_WXYZ:", target.quaternion_wxyz.tolist())

    result = solve_ik(model, data.qpos.copy(), target)
    print("IK_ITERATIONS:", result.iterations)
    print("IK_POSITION_ERROR_M:", result.position_error_m)
    print("IK_ORIENTATION_ERROR_DEG:", result.orientation_error_deg)
    print("IK_Q_TARGET_RAD:", result.q_arm.tolist())
    print("IK_CONVERGED:", result.converged)
    if not result.converged:
        print("MOVE_NOT_EXECUTED: IK did not converge")
        raise SystemExit(2)

    collision_free, collision = collision_preflight(
        model,
        data.qpos.copy(),
        result.q_arm,
        max(2, args.collision_samples),
    )
    print("STRAIGHT_JOINT_PATH_COLLISION_FREE:", collision_free)
    if not collision_free:
        print("FIRST_COLLISION:", collision)
        print(
            "MOVE_NOT_EXECUTED: this local controller does not route around "
            "obstacles; use the planned OMPL layer for another path"
        )
        raise SystemExit(3)

    if args.check_only:
        print("CHECK_ONLY: target is IK-reachable and sampled path is collision-free")
        return

    execute(
        model,
        data,
        result.q_arm,
        max(args.duration, model.opt.timestep),
        max(args.hold, 0.0),
        args.viewer,
    )
    mujoco.mj_forward(model, data)
    final = site_pose(model, data, tcp_id)
    final_position_error = float(np.linalg.norm(target.position - final.position))
    final_orientation_error = float(
        np.degrees(
            np.linalg.norm(
                orientation_error_rotvec_world(
                    final.quaternion_wxyz,
                    target.quaternion_wxyz,
                )
            )
        )
    )

    print("TCP_FINAL_POSITION_M:", final.position.tolist())
    print("TCP_FINAL_QUAT_WXYZ:", final.quaternion_wxyz.tolist())
    print("FINAL_POSITION_ERROR_M:", final_position_error)
    print("FINAL_ORIENTATION_ERROR_DEG:", final_orientation_error)
    reached = final_position_error <= 2.0e-3 and final_orientation_error <= 1.0
    print("TARGET_REACHED:", reached)
    if not reached:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
