#!/usr/bin/env python3
"""Headless regression: closure, TCP, contacts, protocol and disconnect hold."""
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from ur5e_sim.server.runtime import Controller, Link, dispatch

from ur5e_sim.paths import ROOT as ROOT


def make(model=None):
    model = model or mujoco.MjModel.from_xml_path(str(ROOT/'scenes/scene_control.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    return Controller(model, data, SimpleNamespace(collision_samples=120, max_joint_speed=.75))


def step(c, seconds):
    for _ in range(int(seconds/c.model.opt.timestep)):
        c.apply_controls()
        mujoco.mj_step(c.model, c.data)
    mujoco.mj_forward(c.model, c.data)


def command(c, text):
    return dispatch(c, {'id':1, 'command':text})


def main():
    c = make()
    initial = c.current_pose().position.copy()
    step(c, 2)
    assert np.linalg.norm(c.current_pose().position-initial) < 1e-6
    assert command(c, 'gripper close')['ok']
    assert not command(c, 'tcp-rel 0 0 .005')['ok']  # busy
    step(c, 4)
    print('CLOSED_GAP_MM:', 1000*c.gap_m())
    assert abs(c.gap_m()) < .0001
    assert c.gripper_state() == 'at_target'
    assert command(c, 'tcp-rel 0 0 .005')['ok']
    assert not command(c, 'gripper open')['ok']  # avoid changing preflight geometry
    step(c, 2)
    error = np.linalg.norm(c.current_pose().position-(initial+[0, 0, .005]))
    print('CLOSED_TCP_MOVE_ERROR_M:', error)
    assert error < .0002
    assert command(c, 'gripper gap 20')['ok']
    step(c, 2)
    assert abs(c.gap_m()-.020) < .0001
    assert command(c, 'gripper open')['ok']
    step(c, 3)
    assert abs(c.gap_m()-.0899937074) < .0001
    for text in ('gripper gap nan', 'tcp nan 0 0', 'speed inf', 'speed 3', 'gripper gap -1'):
        assert not command(c, text)['ok'], text

    # Place a fixed 10 mm test block between the distal inner pad planes.
    t = make()
    tree = ET.parse(ROOT/'scenes/scene_control.xml')
    tree.find('compiler').set('meshdir', str(ROOT/'assets/meshes'))
    centre = (t.data.site('gripper_left_inner').xpos+t.data.site('gripper_right_inner').xpos)/2
    matrix = t.data.xmat[t.model.body('left_gripper').id]
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, matrix)
    ET.SubElement(tree.find('worldbody'), 'geom', name='test_block', type='box',
                  pos=' '.join(map(str, centre)), quat=' '.join(map(str, quat)),
                  size='.004 .003 .005', contype='1', conaffinity='1')
    blocked = make(mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding='unicode')))
    command(blocked, 'gripper close')
    step(blocked, 5)
    print('BLOCKED_GAP_MM:', 1000*blocked.gap_m(), blocked.gripper_state())
    assert blocked.gripper_state() == 'contact_blocked'
    assert blocked.gap_m() > .008
    assert any('test_block' in (p['geom1'], p['geom2']) for p in blocked.finger_contacts())

    # Real loopback socket: no MuJoCo state is touched by a network worker thread.
    link = Link(0, c)
    sock = socket.create_connection(link.listener.getsockname(), timeout=2)
    reader = sock.makefile('rb')
    try:
        link.poll()
        assert json.loads(reader.readline())['event'] == 'ready'
        sock.sendall(b'{"id":7,"command":"status"}\n')
        link.poll()
        assert json.loads(reader.readline())['id'] == 7
        command(c, 'tcp-rel 0 0 .005')
        reader.close(); sock.close()
        link.poll()
        assert c.plan is None
        hold = c.data.qpos.copy()
        step(c, 1)
        assert np.max(np.abs(c.data.qpos-hold)) < 1e-4
    finally:
        reader.close(); sock.close(); link.close()
    print('CONTROL_REGRESSION: PASSED')


if __name__ == '__main__':
    main()
