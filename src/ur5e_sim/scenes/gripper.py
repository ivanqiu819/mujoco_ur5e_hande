#!/usr/bin/env python3
"""Calibrate the supplied finger CAD; emit a separate scene without overwriting it.

Collision pieces are convex hulls of triangle surfaces clipped to a small XYZ
grid in each finger's local frame. This retains the stepped finger shape instead
of filling its entire concavity with one box/hull. Only the finger colliders and
measurement metadata change; original meshes, joints, mass, TCP and home stay.
This fit is for the supplied Hand-E mesh, not a factory gripper calibration.
"""
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ur5e_sim.paths import ROOT as ROOT


def clip(poly, axis, bound, keep_above):
    if len(poly) == 0:
        return []
    result = []
    previous = poly[-1]
    p_inside = previous[axis] >= bound if keep_above else previous[axis] <= bound
    for point in poly:
        inside = point[axis] >= bound if keep_above else point[axis] <= bound
        if inside != p_inside:
            t = (bound - previous[axis]) / (point[axis] - previous[axis])
            result.append(previous + t * (point - previous))
        if inside:
            result.append(point)
        previous, p_inside = point, inside
    return result


def local_mesh(model, data, geom):
    mid = int(model.geom_dataid[geom])
    vertices = model.mesh_vert[
        model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]
    ].astype(float)
    world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    body = int(model.geom_bodyid[geom])
    local = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
    faces = model.mesh_face[
        model.mesh_faceadr[mid]:model.mesh_faceadr[mid] + model.mesh_facenum[mid]
    ]
    return local, faces


def fmt(values):
    return " ".join(f"{float(x):.10g}" for x in np.asarray(values).ravel())


def clipped_points(triangles, bounds):
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    candidates = triangles[np.all(high >= bounds[:,0],axis=1) &
                           np.all(low <= bounds[:,1],axis=1)]
    points = []
    for tri in candidates:
        poly = list(tri)
        for axis in range(3):
            poly = clip(poly, axis, bounds[axis,0], True)
            poly = clip(poly, axis, bounds[axis,1], False)
        points.extend(poly)
    if not points:
        return None
    points = np.unique(np.round(points,10),axis=0)
    if len(points)<4 or np.linalg.matrix_rank(points-points.mean(0),tol=1e-8)<3:
        return None
    return points


def build(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Output must differ from source; keep the original scene")
    if source.parent != output.parent:
        raise ValueError("Keep output beside source to preserve mesh/include paths")
    tree = ET.parse(source)
    root = tree.getroot()
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    for name in ('Slider_1', 'Slider_2'):
        data.qpos[model.joint(name).qposadr[0]] = 0
    mujoco.mj_forward(model, data)
    asset = root.find('asset')
    if asset is None:
        raise ValueError("Missing assets")
    pad_sites = []
    counts = {}
    pieces = {}
    for side in ('left', 'right'):
        body = root.find(f'.//body[@name="{side}_gripper"]')
        old = body.find(f'geom[@name="{side}_finger_collision"]')
        if old is None:
            raise ValueError("Expected original finger collision box: " + side)
        body.remove(old)
        mid = model.mesh(f'{side}_finger_2').id
        gid = int(np.flatnonzero(
            (model.geom_dataid == mid) &
            (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
        )[0])
        vertices, faces = local_mesh(model, data, gid)
        # Distal planar gripping faces occupy y >= 44 mm in the supplied CAD.
        pad = vertices[vertices[:, 1] >= 0.044]
        inner_z = pad[:, 2].max() if side == 'left' else pad[:, 2].min()
        local_site = np.array([(pad[:, 0].min() + pad[:, 0].max()) / 2,
                               0.053, inner_z])
        ET.SubElement(body, 'site', name=f'gripper_{side}_inner',
                      pos=fmt(local_site), size='0.001', rgba='0 0 0 0', group='4')
        bid = model.body(f'{side}_gripper').id
        pad_sites.append(data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ local_site)

        # 5-mm width bins and 8-mm height bins. Each hull encloses only a local
        # CAD surface fragment, not the whole bent finger.
        xmin, ymin = vertices[:, :2].min(axis=0)
        xmax, ymax = vertices[:, :2].max(axis=0)
        xs = np.linspace(xmin - 1e-7, xmax + 1e-7, int(np.ceil((xmax-xmin)/.005)) + 1)
        ys = np.linspace(ymin - 1e-7, ymax + 1e-7, int(np.ceil((ymax-ymin)/.008)) + 1)
        triangles = vertices[faces]
        low, high = triangles.min(axis=1), triangles.max(axis=1)
        zmin, zmax = vertices[:,2].min(), vertices[:,2].max()
        zs = np.linspace(zmin-1e-7, zmax+1e-7, int(np.ceil((zmax-zmin)/.004))+1)
        count = 0
        for (x0,x1), (y0,y1), (z0,z1) in itertools.product(
                zip(xs[:-1],xs[1:]), zip(ys[:-1],ys[1:]), zip(zs[:-1],zs[1:])):
                candidates = triangles[(high[:, 0]>=x0) & (low[:, 0]<=x1) &
                                       (high[:, 1]>=y0) & (low[:, 1]<=y1) &
                                       (high[:, 2]>=z0) & (low[:, 2]<=z1)]
                points = []
                for tri in candidates:
                    poly = list(tri)
                    for axis, bound, above in ((0,x0,True),(0,x1,False),
                                                (1,y0,True),(1,y1,False),
                                                (2,z0,True),(2,z1,False)):
                        poly = clip(poly, axis, bound, above)
                    points.extend(poly)
                if not points:
                    continue
                points = np.unique(np.round(points, 10), axis=0)
                if len(points) < 4 or np.linalg.matrix_rank(points-points.mean(0), tol=1e-8)<3:
                    continue
                name = f'{side}_finger_part_{count:02d}'
                ET.SubElement(asset, 'mesh', name=name, vertex=fmt(points))
                ET.SubElement(body, 'geom', {'class':'collision', 'name':name,
                                             'type':'mesh','mesh':name})
                pieces[name] = (body, triangles, np.array([[x0,x1],[y0,y1],[z0,z1]]), points)
                count += 1
        # Screw heads are separate convex visual meshes; retain their geometry.
        for index in (0, 1):
            visual = body.find(f'geom[@mesh="{side}_finger_{index}"]')
            attrs = dict(visual.attrib)
            attrs.pop('material', None)
            attrs.update({'class':'collision','name':f'{side}_screw_{index}'})
            ET.SubElement(body, 'geom', attrs)
        counts[side] = count + 2

    axis = data.xmat[model.body('left_gripper').id].reshape(3,3)[:, 2]
    zero_gap = float(np.dot(pad_sites[1] - pad_sites[0], axis))
    closed_q = -zero_gap / 2
    # Refine only pieces whose coarse convex hull fills a root recess and
    # obstructs closure. Re-clip the ORIGINAL triangles, never the old hull.
    for iteration in range(18):
        probe_root = copy.deepcopy(root)
        probe_root.find('compiler').set('meshdir', str((source.parent/root.find('compiler').get('meshdir','')).resolve()))
        probe = mujoco.MjModel.from_xml_string(ET.tostring(probe_root,encoding='unicode'))
        state = mujoco.MjData(probe)
        mujoco.mj_resetDataKeyframe(probe,state,probe.key('home').id)
        state.qpos[-2:] = closed_q + 0.000005  # 10 um distal gap for refinement
        mujoco.mj_forward(probe,state)
        bad = set()
        for i in range(state.ncon):
            contact = state.contact[i]
            if contact.dist < -0.000002:
                for g in (contact.geom1,contact.geom2):
                    name = probe.geom(g).name
                    if name in pieces:
                        bad.add(name)
        if not bad:
            break
        print('REFINE',iteration,'pieces',len(bad), flush=True)
        for name in sorted(bad):
            body, triangles, bounds, points = pieces.pop(name)
            body.remove(body.find(f'geom[@name="{name}"]'))
            asset.remove(asset.find(f'mesh[@name="{name}"]'))
            axis = int(np.argmax(np.ptp(points,axis=0)))
            split = float((points[:,axis].min()+points[:,axis].max())/2)
            for index in (0,1):
                child_bounds = bounds.copy()
                child_bounds[axis,1-index] = split
                child = clipped_points(triangles,child_bounds)
                if child is None:
                    continue
                child_name = name+str(index)
                ET.SubElement(asset,'mesh',name=child_name,vertex=fmt(child))
                ET.SubElement(body,'geom',{'class':'collision','name':child_name,
                                           'type':'mesh','mesh':child_name})
                pieces[child_name] = (body,triangles,child_bounds,child)
    else:
        raise RuntimeError('Collision fit still blocks closure; original source unchanged')
    counts = {side:sum(name.startswith(side+'_') for name in pieces)+2
              for side in ('left','right')}
    custom = root.find('custom')
    if custom is None:
        custom = ET.SubElement(root, 'custom')
    ET.SubElement(custom, 'numeric', name='gripper_closed_q', data=fmt([closed_q]))
    ET.SubElement(custom, 'numeric', name='gripper_zero_gap', data=fmt([zero_gap]))
    ET.SubElement(custom, 'numeric', name='gripper_open_q', data='0.02')
    ET.SubElement(custom, 'text', name='gripper_calibration', data='supplied_CAD_tip_planes_v1')
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    # Fail early if generated pieces or paths are invalid.
    mujoco.MjModel.from_xml_path(str(output))
    result = {'zero_coordinate_gap_m':zero_gap, 'closed_q_m':closed_q,
              'collision_pieces':counts, 'output':str(output)}
    print(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'scenes/scene.xml')
    parser.add_argument('--output', type=Path, default=ROOT/'scenes/scene_control.xml')
    args = parser.parse_args()
    build(args.source, args.output)
