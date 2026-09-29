"""Load grasp compliance and attach held_plug on grasp_anchor (Hand-E frame)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

from ur5e_sim.paths import ROOT
from ur5e_sim.scenes.gel_mount import HELD_PLUG_CONAFFINITY, HELD_PLUG_CONTYPE
from ur5e_sim.scenes.materials import load_materials

DEFAULT_CONFIG = ROOT / "configs/grasp_compliance.yaml"


def load_grasp_compliance(path: Path | None = None) -> dict:
    cfg_path = Path(path or DEFAULT_CONFIG).resolve()
    data = yaml.safe_load(cfg_path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported grasp compliance schema: {cfg_path}")
    return data


def _numbers(values) -> str:
    return " ".join(f"{float(x):.12g}" for x in np.asarray(values).ravel())


def _remove_held_plug(root: ET.Element) -> None:
    for parent in root.iter("body"):
        for child in list(parent.findall("./body[@name='held_plug']")):
            parent.remove(child)
        for child in list(parent.findall("./body[@name='grasp_anchor']")):
            parent.remove(child)


def tcp_world_pose_at_home(root: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    """TCP site pose in world frame at the home keyframe (before held_plug is added)."""
    model = _compile_root(root)
    data = mujoco.MjData(model)
    home_id = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if tcp_id < 0:
        raise ValueError("tcp site missing for held_plug placement")
    pos = data.site_xpos[tcp_id].copy()
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[tcp_id].reshape(9))
    return pos, quat


def _plug_body_children(
    plug: ET.Element,
    *,
    plug_settings: dict,
    materials: dict,
) -> None:
    size = np.asarray(plug_settings["plug_size_m"])
    tip = np.asarray(plug_settings["plug_tip_tcp_m"])
    center = tip - np.array([0.0, 0.0, size[2] / 2.0])
    inertia = plug_settings["plug_mass_kg"] / 12 * np.array(
        [size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2, size[0] ** 2 + size[1] ** 2]
    )
    ET.SubElement(
        plug,
        "inertial",
        pos=_numbers(center),
        mass=str(plug_settings["plug_mass_kg"]),
        diaginertia=_numbers(inertia),
    )
    plug_mat = materials["plug"]
    geom = ET.SubElement(
        plug,
        "geom",
        name="held_plug_collision",
        type="box",
        pos=_numbers(center),
        size=_numbers(size / 2),
        rgba=".15 .3 .75 1",
        contype=HELD_PLUG_CONTYPE,
        conaffinity=HELD_PLUG_CONAFFINITY,
    )
    for key in ("friction", "condim", "solref", "solimp", "priority"):
        geom.set(key, str(plug_mat[key]))
    ET.SubElement(plug, "site", name="plug_tip", pos=_numbers(tip), size=".001", group="4", rgba="0 0 0 0")


def attach_rigid_held_plug(
    root: ET.Element,
    *,
    plug_settings: dict,
    world_pos: np.ndarray | None = None,
    world_quat: np.ndarray | None = None,
    materials: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Place held_plug as a free body at the home TCP pose (friction grasp, gravity on)."""
    materials = materials or load_materials()
    _remove_held_plug(root)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("worldbody missing")
    if world_pos is None or world_quat is None:
        world_pos, world_quat = tcp_world_pose_at_home(root)
    plug = ET.SubElement(
        world,
        "body",
        name="held_plug",
        pos=_numbers(world_pos),
        quat=_numbers(world_quat),
    )
    ET.SubElement(plug, "freejoint", name="plug_free")
    _plug_body_children(plug, plug_settings=plug_settings, materials=materials)
    return np.asarray(world_pos, dtype=float), np.asarray(world_quat, dtype=float)


def append_plug_free_qpos_to_keyframes(root: ET.Element, plug_qpos: np.ndarray) -> None:
    plug_qpos = np.asarray(plug_qpos, dtype=float).ravel()
    if plug_qpos.size != 7:
        raise ValueError("plug freejoint qpos must be 7 (pos + quat)")
    for key in root.findall(".//key"):
        old = np.fromstring(key.get("qpos"), sep=" ")
        key.set("qpos", _numbers(np.r_[old, plug_qpos]))


def settle_plug_home_keyframe(
    root: ET.Element,
    *,
    settle_steps: int,
) -> None:
    """Simulate at home so the free plug settles in the closed gripper; update keyframes."""
    model = _compile_root(root)
    data = mujoco.MjData(model)
    home_id = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    plug_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plug_free")
    if plug_jid < 0:
        raise ValueError("plug_free joint missing")
    plug_adr = int(model.jnt_qposadr[plug_jid])
    plug_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "held_plug")
    plug_dof = int(model.body_dofadr[plug_body])
    qpos_nominal = data.qpos.copy()
    ctrl_nominal = data.ctrl.copy()
    # Phase 1: hold plug position fixed while gripper closes around it.
    for _ in range(min(settle_steps, 400)):
        data.qpos[plug_adr : plug_adr + 3] = qpos_nominal[plug_adr : plug_adr + 3]
        data.qpos[plug_adr + 3 : plug_adr + 7] = qpos_nominal[plug_adr + 3 : plug_adr + 7]
        data.qvel[plug_dof : plug_dof + 6] = 0.0
        data.ctrl[:] = ctrl_nominal
        mujoco.mj_step(model, data)
    # Phase 2: release plug and let it settle under grip + gravity.
    for _ in range(max(1, settle_steps - 400)):
        data.ctrl[:] = ctrl_nominal
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    # Verify plug didn't fall.
    plug_pos_drift = float(np.linalg.norm(
        data.qpos[plug_adr : plug_adr + 3] - qpos_nominal[plug_adr : plug_adr + 3]
    ))
    if plug_pos_drift > 0.005:
        raise RuntimeError(
            f"Plug fell during settle (drift {plug_pos_drift * 1000:.1f} mm). "
            "Check gripper/plug contact parameters."
        )
    # Build settled keyframe qpos: keep arm joints nominal, update plug + sliders.
    qpos = qpos_nominal.copy()
    qpos[plug_adr : plug_adr + 7] = data.qpos[plug_adr : plug_adr + 7]
    # Also save slider positions that reached equilibrium with the squeeze ctrl.
    for jname in ("Slider_1", "Slider_2"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            adr = int(model.jnt_qposadr[jid])
            qpos[adr] = data.qpos[adr]
    for key in root.findall(".//key"):
        old = np.fromstring(key.get("qpos"), sep=" ")
        if old.shape[0] != model.nq:
            raise ValueError(f"Keyframe {key.get('name')} qpos length {old.shape[0]} != nq {model.nq}")
        old[:] = qpos
        key.set("qpos", _numbers(old))
        if key.get("ctrl"):
            ctrl = np.fromstring(key.get("ctrl"), sep=" ")
            if ctrl.shape[0] == model.nu:
                key.set("ctrl", _numbers(ctrl_nominal))


def _tcp_pose_in_hande(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    hande_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hande_base_link")
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if hande_id < 0 or tcp_id < 0:
        raise ValueError("hande_base_link or tcp missing for grasp_anchor placement")
    hande_pos = data.xpos[hande_id].copy()
    hande_mat = data.xmat[hande_id].reshape(3, 3).copy()
    site_pos = data.site_xpos[tcp_id].copy()
    site_mat = data.site_xmat[tcp_id].reshape(3, 3).copy()
    rel_pos = hande_mat.T @ (site_pos - hande_pos)
    rel_mat = hande_mat.T @ site_mat
    rel_quat = np.empty(4)
    mujoco.mju_mat2Quat(rel_quat, rel_mat.reshape(9))
    return rel_pos, rel_quat


def _nested_compliance_body(
    parent: ET.Element,
    *,
    name: str,
    joint_type: str,
    axis: str,
    jrange: tuple[float, float],
    stiffness: float,
    damping: float,
) -> ET.Element:
    body = ET.SubElement(parent, "body", name=name)
    ET.SubElement(body, "inertial", pos="0 0 0", mass="0.01", diaginertia="1e-5 1e-5 1e-5")
    ET.SubElement(
        body,
        "joint",
        name=f"{name}_joint",
        type=joint_type,
        axis=axis,
        range=_numbers(jrange),
        stiffness=f"{stiffness:.10g}",
        damping=f"{damping:.10g}",
        armature="0.001",
    )
    return body


def attach_compliant_held_plug(
    root: ET.Element,
    *,
    plug_settings: dict,
    compliance: dict | None = None,
    meshdir: Path | None = None,
) -> list[str]:
    """Replace rigid wrist-held plug with grasp_anchor + 6-DoF compliance chain."""
    compliance = compliance or load_grasp_compliance()
    _remove_held_plug(root)

    hande = root.find(".//body[@name='hande_base_link']")
    if hande is None:
        raise ValueError("hande_base_link missing")

    compiler = root.find("compiler")
    meshdir_attr = compiler.get("meshdir", ".")
    resolved_meshdir = Path(meshdir) if meshdir is not None else Path(meshdir_attr)
    if not resolved_meshdir.is_absolute():
        resolved_meshdir = (ROOT / "scenes" / resolved_meshdir).resolve()
    compiler.set("meshdir", str(resolved_meshdir))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    compiler.set("meshdir", meshdir_attr)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    anchor_pos, anchor_quat = _tcp_pose_in_hande(model, data)

    tr = compliance["translation"]
    rot = compliance["rotation"]
    t_range = float(tr["range_m"])
    r_range = float(rot["range_rad"])

    anchor = ET.SubElement(
        hande,
        "body",
        name="grasp_anchor",
        pos=_numbers(anchor_pos),
        quat=_numbers(anchor_quat),
    )
    b_tx = _nested_compliance_body(
        anchor, name="grasp_tx", joint_type="slide", axis="1 0 0",
        jrange=(-t_range, t_range), stiffness=float(tr["stiffness"]), damping=float(tr["damping"]),
    )
    b_ty = _nested_compliance_body(
        b_tx, name="grasp_ty", joint_type="slide", axis="0 1 0",
        jrange=(-t_range, t_range), stiffness=float(tr["stiffness"]), damping=float(tr["damping"]),
    )
    b_tz = _nested_compliance_body(
        b_ty, name="grasp_tz", joint_type="slide", axis="0 0 1",
        jrange=(-t_range, t_range), stiffness=float(tr["stiffness"]), damping=float(tr["damping"]),
    )
    b_rx = _nested_compliance_body(
        b_tz, name="grasp_rx", joint_type="hinge", axis="1 0 0",
        jrange=(-r_range, r_range), stiffness=float(rot["stiffness"]), damping=float(rot["damping"]),
    )
    b_ry = _nested_compliance_body(
        b_rx, name="grasp_ry", joint_type="hinge", axis="0 1 0",
        jrange=(-r_range, r_range), stiffness=float(rot["stiffness"]), damping=float(rot["damping"]),
    )
    plug_parent = _nested_compliance_body(
        b_ry, name="grasp_rz", joint_type="hinge", axis="0 0 1",
        jrange=(-r_range, r_range), stiffness=float(rot["stiffness"]), damping=float(rot["damping"]),
    )

    size = np.asarray(plug_settings["plug_size_m"])
    tip = np.asarray(plug_settings["plug_tip_tcp_m"])
    center = tip - np.array([0.0, 0.0, size[2] / 2.0])
    inertia = plug_settings["plug_mass_kg"] / 12 * np.array(
        [size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2, size[0] ** 2 + size[1] ** 2]
    )
    plug = ET.SubElement(plug_parent, "body", name="held_plug", gravcomp="1")
    ET.SubElement(
        plug,
        "inertial",
        pos=_numbers(center),
        mass=str(plug_settings["plug_mass_kg"]),
        diaginertia=_numbers(inertia),
    )
    friction = str(compliance.get("plug_geom_friction", "1 0.005 0.0001"))
    ET.SubElement(
        plug,
        "geom",
        name="held_plug_collision",
        type="box",
        pos=_numbers(center),
        size=_numbers(size / 2),
        rgba=".15 .3 .75 1",
        friction=friction,
        contype=HELD_PLUG_CONTYPE,
        conaffinity=HELD_PLUG_CONAFFINITY,
    )
    ET.SubElement(plug, "site", name="plug_tip", pos=_numbers(tip), size=".001", group="4", rgba="0 0 0 0")
    return [
        "grasp_tx_joint",
        "grasp_ty_joint",
        "grasp_tz_joint",
        "grasp_rx_joint",
        "grasp_ry_joint",
        "grasp_rz_joint",
    ]


def _compile_root(root: ET.Element) -> mujoco.MjModel:
    compiler = root.find("compiler")
    meshdir_attr = compiler.get("meshdir", ".")
    resolved = Path(meshdir_attr)
    if not resolved.is_absolute():
        resolved = (ROOT / "scenes" / meshdir_attr).resolve()
    compiler.set("meshdir", str(resolved))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    compiler.set("meshdir", meshdir_attr)
    return model


def settle_home_keyframe(
    root: ET.Element,
    *,
    settle_steps: int,
    grasp_joint_names: tuple[str, ...] | None = None,
) -> None:
    """Short mj_step at home; update only grasp compliance qpos in keyframes."""
    grasp_joint_names = grasp_joint_names or (
        "grasp_tx_joint",
        "grasp_ty_joint",
        "grasp_tz_joint",
        "grasp_rx_joint",
        "grasp_ry_joint",
        "grasp_rz_joint",
    )
    model = _compile_root(root)
    data = mujoco.MjData(model)
    home_id = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    qpos_nominal = data.qpos.copy()
    grasp_adrs = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in grasp_joint_names
    ]
    for _ in range(max(1, int(settle_steps))):
        mujoco.mj_step(model, data)
    qpos = qpos_nominal.copy()
    for adr in grasp_adrs:
        qpos[adr] = data.qpos[adr]
    # Do not carry large grasp drift into home keyframe after long insertion cycles.
    for adr in grasp_adrs:
        if abs(qpos[adr]) > 0.012:
            qpos[adr] = 0.0
    for key in root.findall(".//key"):
        old = np.fromstring(key.get("qpos"), sep=" ")
        if old.shape[0] != model.nq:
            raise ValueError(f"Keyframe {key.get('name')} qpos length {old.shape[0]} != nq {model.nq}")
        key.set("qpos", _numbers(qpos))
        if key.get("ctrl"):
            ctrl = np.fromstring(key.get("ctrl"), sep=" ")
            if ctrl.shape[0] == model.nu:
                key.set("ctrl", _numbers(data.ctrl))
