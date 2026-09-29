"""Insertion scene includes Xense gels and high-fidelity freejoint plug."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from ur5e_sim.control.trajectory import unexpected_contact_count
from ur5e_sim.paths import ROOT
from ur5e_sim.tactile.contact_map import peak_press_mm
from ur5e_sim.tactile.tactile_boundary import build_dual_depth


class InsertionGelSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene = ROOT / "scenes/scene_insertion.xml"
        if not scene.is_file():
            raise unittest.SkipTest("scene_insertion.xml missing; run scenes.insertion")
        cls.model = mujoco.MjModel.from_xml_path(str(scene))

    def test_gels_present(self):
        for name in ("xense_gel_left", "xense_gel_right"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name),
                0,
            )

    def test_home_gel_grasps_plug_without_unexpected_contact(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.model.key("home").id)
        for _ in range(500):
            mujoco.mj_step(self.model, data)
        mujoco.mj_forward(self.model, data)
        self.assertEqual(unexpected_contact_count(self.model, data), 0)
        self.assertTrue(np.all(np.isfinite(data.qpos)))
        depth_l, depth_r = build_dual_depth(self.model, data)
        press_l, press_r = peak_press_mm(depth_l), peak_press_mm(depth_r)
        self.assertGreater(max(press_l, press_r), 0.05,
                          "Gel should show measurable press depth from plug contact")

    def test_freejoint_plug_on_worldbody(self):
        plug_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "held_plug")
        world = 0  # worldbody ID is always 0
        self.assertGreaterEqual(plug_body, 0)
        self.assertEqual(int(self.model.body_parentid[plug_body]), world,
                        "held_plug should be a direct child of worldbody (freejoint)")
        plug_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "plug_free")
        self.assertGreaterEqual(plug_jid, 0, "plug_free freejoint should exist")
        self.assertEqual(int(self.model.jnt_type[plug_jid]), mujoco.mjtJoint.mjJNT_FREE)
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "held_plug_collision")
        self.assertGreaterEqual(geom_id, 0)
        # No grasp_anchor (old compliant approach not used)
        self.assertLess(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "grasp_anchor"), 0)

    def test_no_deposited_plug_mocap(self):
        dep = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "deposited_plug")
        self.assertLess(dep, 0, "deposited_plug mocap body should not exist")

    def test_no_insertion_fixed_gap(self):
        nid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_NUMERIC, "insertion_fixed_gap")
        self.assertLess(nid, 0, "insertion_fixed_gap numeric should not exist")


if __name__ == "__main__":
    unittest.main()
