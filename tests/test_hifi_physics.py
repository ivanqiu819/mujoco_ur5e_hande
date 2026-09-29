"""High-fidelity physics: grasp stability, plug drop, gel compliance, wall response."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from ur5e_sim.config import load_settings, site_matrix
from ur5e_sim.control.trajectory import unexpected_contact_count
from ur5e_sim.paths import ROOT
from ur5e_sim.tactile.tactile_boundary import compute_dual_boundary


class GraspStabilityTests(unittest.TestCase):
    """Plug must remain in grip during arm motion and settle."""

    @classmethod
    def setUpClass(cls):
        scene = ROOT / "scenes/scene_insertion.xml"
        if not scene.is_file():
            raise unittest.SkipTest("scene_insertion.xml missing")
        cls.model = mujoco.MjModel.from_xml_path(str(scene))
        cls.settings = load_settings()

    def test_home_hold_2s(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.model.key("home").id)
        mujoco.mj_forward(self.model, data)
        ctrl = data.ctrl.copy()
        plug_id = self.model.body("held_plug").id
        z0 = float(data.xpos[plug_id][2])
        for _ in range(1000):
            data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, data)
        mujoco.mj_forward(self.model, data)
        z1 = float(data.xpos[plug_id][2])
        self.assertLess(abs(z1 - z0), 0.010,
                       f"Plug should not drift >10mm at home: dz={abs(z1-z0)*1000:.1f}mm")
        self.assertEqual(unexpected_contact_count(self.model, data), 0)

    def test_plug_has_gravity(self):
        """No gravcomp on plug: verify it falls when unsupported."""
        plug_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "plug_free")
        self.assertGreaterEqual(plug_jid, 0)
        plug_body = self.model.body("held_plug").id
        self.assertAlmostEqual(float(self.model.body_gravcomp[plug_body]), 0.0, places=5,
                              msg="held_plug should NOT have gravcomp")


class PlugDropTests(unittest.TestCase):
    """Plug must fall when gripper opens."""

    @classmethod
    def setUpClass(cls):
        scene = ROOT / "scenes/scene_insertion.xml"
        if not scene.is_file():
            raise unittest.SkipTest("scene_insertion.xml missing")
        cls.model = mujoco.MjModel.from_xml_path(str(scene))

    def test_drop_after_release(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.model.key("home").id)
        for _ in range(200):
            mujoco.mj_step(self.model, data)
        mujoco.mj_forward(self.model, data)
        plug_id = self.model.body("held_plug").id
        z_held = float(data.xpos[plug_id][2])
        # Open gripper fully
        gripper_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        open_q = float(self.model.numeric("gripper_open_q").data[0])
        data.ctrl[gripper_act] = open_q
        for _ in range(500):
            mujoco.mj_step(self.model, data)
        mujoco.mj_forward(self.model, data)
        z_drop = float(data.xpos[plug_id][2])
        drop_mm = (z_held - z_drop) * 1000.0
        self.assertGreater(drop_mm, 20.0,
                          f"Plug should fall >20mm after release: actual={drop_mm:.1f}mm")


class GelComplianceTests(unittest.TestCase):
    """Gel should show measurable penetration when plug is gripped."""

    @classmethod
    def setUpClass(cls):
        scene = ROOT / "scenes/scene_insertion.xml"
        if not scene.is_file():
            raise unittest.SkipTest("scene_insertion.xml missing")
        cls.model = mujoco.MjModel.from_xml_path(str(scene))

    def test_gel_press_at_home(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.model.key("home").id)
        for _ in range(500):
            mujoco.mj_step(self.model, data)
        mujoco.mj_forward(self.model, data)
        boundary = compute_dual_boundary(self.model, data)
        press_l, press_r = float(boundary[2]), float(boundary[3])
        self.assertGreater(press_l, 0.05,
                          f"Left gel should show >0.05mm press: actual={press_l:.4f}")
        self.assertGreater(press_r, 0.05,
                          f"Right gel should show >0.05mm press: actual={press_r:.4f}")

    def test_gel_has_soft_contact_params(self):
        """Gel solref should be softer than the global default (0.005)."""
        for name in ("xense_gel_left", "xense_gel_right"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertGreaterEqual(gid, 0)
            solref = self.model.geom_solref[gid]
            self.assertGreater(float(solref[0]), 0.005,
                              f"{name} solref[0] should be > 0.005 (softer than default)")


if __name__ == "__main__":
    unittest.main()
