#!/usr/bin/env python3
"""Build the pre-held plug task without overwriting any upstream scene."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from ur5e_sim.scenes.gel_mount import (
    SOCKET_WALL_CONAFFINITY,
    SOCKET_WALL_CONTYPE,
    add_gel_geoms,
)
from ur5e_sim.scenes.grasp_compliance import (
    append_plug_free_qpos_to_keyframes,
    attach_rigid_held_plug,
    settle_plug_home_keyframe,
)
from ur5e_sim.scenes.materials import (
    apply_finger_collision_materials,
    apply_socket_wall_materials,
    load_materials,
)
from ur5e_sim.scenes.inspection import quaternion_matrix
from ur5e_sim.config import load_settings, pose_resources
from ur5e_sim.camera.calibration import load_spec


def numbers(values):
    return ' '.join(f'{v:.12g}' for v in np.asarray(values).ravel())


def visual_mesh(path, triangles):
    """Keep original faces; give planar material patches a 1 um backing."""
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    centered = vertices - vertices.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-9) < 3:
        normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
        n = len(vertices)
        vertices = np.vstack([vertices, vertices - normal * 1e-6])
        back = faces[:, ::-1] + n
        sides = []
        for a, b, c in faces:
            for x, y in ((a, b), (b, c), (c, a)):
                sides.extend([[x, y, y+n], [x, y+n, x+n]])
        faces = np.vstack([faces, back, sides])
    with path.open('w') as stream:
        for v in vertices:
            stream.write('v ' + numbers(v) + '\n')
        for f in faces:
            stream.write('f ' + ' '.join(str(i+1) for i in f) + '\n')


def build(settings):
    config, geometry = pose_resources(settings)
    from ur5e_sim.vision.core.aruco import marker_transform
    from ur5e_sim.vision.core.renderer import triangle_materials
    source, output = Path(settings['source_scene']), Path(settings['output_scene'])
    if source == output or source.parent != output.parent:
        raise ValueError('Insertion output must be a different file beside its source')
    root = ET.parse(source).getroot()
    # MJCF principalpixel is a displacement from the image centre, not the
    # OpenCV principal point. The historical upstream XML used cx,cy directly.
    spec = load_spec(settings['camera_config'])
    root.find(f".//camera[@name='{spec.name}']").set(
        'principalpixel', numbers([(spec.width-1)/2-spec.cx, spec.cy-(spec.height-1)/2]))
    samples = settings['render_supersample']
    root.find('visual/global').set('offwidth', str(spec.width*samples))
    root.find('visual/global').set('offheight', str(spec.height*samples))
    assets, world = root.find('asset'), root.find('worldbody')
    meshdir = source.parent / root.find('compiler').get('meshdir')
    local_mesh = meshdir / 'inspection_workpiece.stl'
    if hashlib.sha256(local_mesh.read_bytes()).digest() != hashlib.sha256(Path(config['model']['path']).read_bytes()).digest():
        raise ValueError('Pose geometry must match the inspection STL')
    if config['port']['normal'] != [0, 0, 1] or config['port']['up'] != [0, 1, 0]:
        raise ValueError('This slot proxy requires the supplied socket axes')
    table = root.find(".//body[@name='inspection_table']")
    table.set('pos', numbers([*settings['table_center_xy_m'], settings['table_top_z_m']]))
    body = root.find(".//body[@name='inspection_workpiece']")
    rotation = quaternion_matrix(np.fromstring(body.get('quat'), sep=' '))
    original = geometry.vertices @ geometry.T_stl_port[:3, :3].T + geometry.T_stl_port[:3, 3]
    lower, upper = original.min(axis=0), original.max(axis=0)
    rotated = original @ rotation.T
    translation = np.r_[settings['workpiece_center_xy_m'], 0.] - (rotated.min(axis=0)+rotated.max(axis=0))/2
    translation[2] = settings['table_top_z_m'] + settings['fixture_height_m'] + settings['table_clearance_m'] - rotated[:, 2].min()
    body.set('pos', numbers(translation))
    top = settings['table_top_z_m'] + settings['fixture_height_m']
    footprint = np.ptp(rotated, axis=0)[:2]
    ET.SubElement(world, 'geom', name='insertion_fixture', type='box',
                  pos=numbers([*settings['workpiece_center_xy_m'], top-settings['fixture_height_m']/2]),
                  size=numbers([*(footprint/2), settings['fixture_height_m']/2]), rgba='.32 .34 .36 1')
    body.remove(body.find("geom[@name='inspection_workpiece_collision']"))
    # Conservative solid proxy minus the actual rectangular socket channel.
    centre = geometry.T_stl_port[:3, 3]
    inner = np.asarray(config['port']['inner_size'])*geometry.unit_scale
    slot_low = centre - [inner[0]/2, inner[1]/2, settings['slot_depth_m']]
    slot_high = centre + [inner[0]/2, inner[1]/2, 0]
    triangles = original[geometry.faces]
    for axis, value in ((0, slot_low[0]), (0, slot_high[0]),
                        (1, slot_low[1]), (1, slot_high[1]), (2, slot_low[2])):
        planar = np.all(np.abs(triangles[:, :, axis]-value) < 1e-7, axis=1)
        inside = np.all(triangles.min(axis=1) >= slot_low-1e-7, axis=1)
        inside &= np.all(triangles.max(axis=1) <= slot_high+1e-7, axis=1)
        if not np.any(planar & inside):
            raise ValueError('Slot proxy dimensions do not match an STL cavity wall')
    if np.any(np.asarray(settings['plug_size_m'])[:2] >= inner):
        raise ValueError('Plug cross section must fit inside the socket')
    boxes = [(lower, [slot_low[0], upper[1], upper[2]]),
             ([slot_high[0], lower[1], lower[2]], upper),
             ([slot_low[0], lower[1], lower[2]], [slot_high[0], slot_low[1], upper[2]]),
             ([slot_low[0], slot_high[1], lower[2]], [slot_high[0], upper[1], upper[2]]),
             ([slot_low[0], slot_low[1], lower[2]], [slot_high[0], slot_high[1], slot_low[2]])]
    for i, (lo, hi) in enumerate(boxes):
        lo, hi = np.asarray(lo), np.asarray(hi)
        wall = ET.SubElement(body, 'geom', name=f'socket_wall_{i}', type='box',
                             attrib={'class': 'collision'}, pos=numbers((lo+hi)/2), size=numbers((hi-lo)/2))
        wall.set('contype', SOCKET_WALL_CONTYPE)
        wall.set('conaffinity', SOCKET_WALL_CONAFFINITY)
    apply_socket_wall_materials(root)
    # Split by the same material regions as the original image experiment.
    body.remove(body.find("geom[@name='inspection_workpiece_visual']"))
    grays = triangle_materials(geometry, config['render'])
    for gray in np.unique(grays):
        name = f'insertion_surface_{int(gray)}'
        visual_mesh(meshdir / (name+'.obj'), triangles[grays == gray])
        ET.SubElement(assets, 'mesh', name=name, file=name+'.obj')
        ET.SubElement(body, 'geom', name=name, mesh=name, attrib={'class': 'visual'},
                      rgba=numbers([gray/255]*3+[1]))
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, geometry.T_stl_port[:3, :3].reshape(9))
    ET.SubElement(body, 'site', name='socket_port', pos=numbers(centre), quat=numbers(quat),
                  size='.001', group='4', rgba='0 0 0 0')
    # Physical marker cells; no image-space overlay and no collision geometry.
    settings_marker = config['aruco']
    transform = geometry.T_stl_port @ marker_transform(settings_marker)
    mujoco.mju_mat2Quat(quat, transform[:3, :3].reshape(9))
    marker = ET.SubElement(body, 'body', name='socket_marker', pos=numbers(transform[:3, 3]), quat=numbers(quat))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, settings_marker['dictionary']))
    cells = dictionary.markerSize+2
    bits = cv2.aruco.generateImageMarker(dictionary, settings_marker['marker_id'], cells)
    half, margin = settings_marker['marker_size_mm']/2000, settings_marker['white_margin_mm']/1000
    borders = np.r_[-half-margin, np.linspace(-half, half, cells+1), half+margin]
    for row in range(cells+2):
        for col in range(cells+2):
            x0, x1 = borders[col:col+2]
            y0, y1 = borders[row:row+2]
            gray = bits[row-1, col-1]/255 if 1 <= row <= cells and 1 <= col <= cells else 1
            ET.SubElement(marker, 'geom', name=f'aruco_{row}_{col}', type='box',
                          pos=numbers([(x0+x1)/2, -(y0+y1)/2, -.000002]),
                          size=numbers([(x1-x0)/2, (y1-y0)/2, .000002]),
                          rgba=numbers([gray]*3+[1]), contype='0', conaffinity='0', group='2', mass='0')
    size = np.asarray(settings['plug_size_m'])
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    add_gel_geoms(root, replace=True)
    apply_finger_collision_materials(root)
    materials = load_materials()
    noslip = int(materials.get("noslip_iterations", 3))
    option = root.find("option")
    if option is not None and noslip > 0:
        option.set("noslip_iterations", str(noslip))
    ET.SubElement(contact, 'exclude', body1='left_gripper', body2='right_gripper')
    closed = float(root.find(".//numeric[@name='gripper_closed_q']").get('data'))
    q_gap = closed + size[0] / 2            # slider pos with plug exactly fitting
    q_grip = closed + size[0] / 2 * 0.85    # slightly past plug surface → contact squeeze
    for key in root.findall('.//key'):
        qpos = np.fromstring(key.get('qpos'), sep=' ')
        ctrl = np.fromstring(key.get('ctrl'), sep=' ')
        qpos[-2:] = q_gap                   # start with plug just fitting
        ctrl[-1] = q_grip                    # command tries to close past plug → friction
        key.set('qpos', numbers(qpos))
        key.set('ctrl', numbers(ctrl))
    world_pos, world_quat = attach_rigid_held_plug(root, plug_settings=settings, materials=materials)
    plug_qpos = np.r_[world_pos, world_quat]
    append_plug_free_qpos_to_keyframes(root, plug_qpos)
    settle_plug_home_keyframe(root, settle_steps=int(materials.get("home_plug_settle_steps", 800)))
    # Store the nominal setup separately from runtime workpiece ground truth.
    nominal = np.eye(4)
    nominal[:3, :3] = rotation @ geometry.T_stl_port[:3, :3]
    nominal[:3, 3] = translation + rotation @ centre
    custom = root.find('custom')
    ET.SubElement(custom, 'numeric', name='insertion_nominal_port', data=numbers(nominal))
    from ur5e_sim.config import scene_signature
    ET.SubElement(custom, 'text', name='insertion_config_sha256', data=scene_signature(settings))
    root.set('model', 'UR5e pre-held plug horizontal insertion')
    ET.indent(root, space='  ')
    # Compile before replacing the generated output.
    root.find('compiler').set('meshdir', str(meshdir.resolve()))
    mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    root.find('compiler').set('meshdir', str(meshdir.relative_to(output.parent)))
    ET.ElementTree(root).write(output, encoding='utf-8', xml_declaration=True)
    print('INSERTION_SCENE:', output)
    print('NOMINAL_PORT_WORLD_M:', nominal[:3, 3].tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    build(load_settings(args.config))
