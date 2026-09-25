"""Exercise the actual server/client/camera processes and evaluate physics offline."""
import argparse
import gzip
import shutil
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import numpy as np
from ur5e_sim.paths import ROOT
from ur5e_sim.client import Client, set_preview
from ur5e_sim.config import load_settings
from ur5e_sim.tasks.insertion import Insertion
from ur5e_sim.control.limits import DEFAULT_CARTESIAN_SPEED_M_S
from ur5e_sim.vision.core.pose import evaluate_pose


def free_port():
    for _ in range(100):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
        if port == 65535: continue
        with socket.socket() as s:
            try: s.bind(('127.0.0.1', port+1)); return port
            except OSError: pass
    raise RuntimeError('No free control/camera port pair')


def start_server(output, viewer=False):
    port = free_port()
    log = (output/'server.log').open('w')
    args = [sys.executable, str(ROOT/'apps/server.py'), '--port',str(port), '--status-hz','0',
            '--evaluation-dir',str(output)]
    if not viewer: args.append('--headless')
    env = dict(os.environ)
    env['MUJOCO_GL'] = 'glfw' if viewer else 'egl'
    process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        if process.poll() is not None: break
        if 'SERVER_READY:' in (output/'server.log').read_text(): return process, log, port
        time.sleep(.1)
    process.terminate(); process.wait(timeout=10); log.close()
    raise RuntimeError((output/'server.log').read_text())


def stop_server(process, log):
    if process.poll() is None: process.send_signal(signal.SIGINT)
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired: process.kill(); process.wait()
    log.close()
    trace=Path(log.name).parent/'physics.jsonl'
    if trace.exists():
        with trace.open('rb') as source, gzip.open(str(trace)+'.gz','wb') as target:
            shutil.copyfileobj(source,target)
        trace.unlink()


def evaluate(output, settings):
    cycle = json.loads((output/'cycle.json').read_text())
    trace = output/'physics.jsonl.gz'
    opener = gzip.open(trace,'rt') if trace.exists() else (output/'physics.jsonl').open()
    with opener as stream:
        rows = [json.loads(line) for line in stream]
    truth = np.asarray(rows[0]['port'])
    error = evaluate_pose(np.asarray(cycle['T_world_port']), truth)
    assert error['translation_error_mm'] <= .3, error
    assert error['rotation_error_deg'] <= 1, error
    assert max(r['contact_count'] for r in rows) == 0
    assert max(r['warnings'] for r in rows) == 0
    protected = [r for r in rows if r['motion']['protected']]
    tip = np.asarray([np.linalg.inv(truth) @ np.asarray(r['tip']) for r in protected])
    times = np.asarray([r['sim_time'] for r in protected])
    lateral = np.linalg.norm(tip[:,:2,3], axis=1)
    speed = np.linalg.norm(np.diff(tip[:,:3,3], axis=0),axis=1)/np.diff(times)
    assert max(lateral) <= .0003, max(lateral)
    speed_limit = cycle.get('insert_speed_m_s', DEFAULT_CARTESIAN_SPEED_M_S)
    assert max(speed) <= speed_limit+1e-6, (max(speed), speed_limit)
    phases = cycle['phases']
    for name, depth in [('aligned',-.03),('inserted',.01),('retracted',-.03)]:
        target = min(rows, key=lambda r:abs(r['sim_time']-phases[name]['sim_time']))
        local = np.linalg.inv(truth) @ np.asarray(target['tip'])
        assert abs(local[2,3]-depth) <= .0003, (name,local[:3,3])
    # Intersect every actual oriented plug box with the socket channel's z slab.
    import itertools
    corners = np.array(list(itertools.product([-.008,.008],[-.002,.002],[-.1,0])))
    edges = [(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(corners[i]!=corners[j]) == 1]
    clearance = float('inf')
    for transform in tip:
        vertices = corners @ transform[:3,:3].T+transform[:3,3]
        points = [v for v in vertices if 0 <= v[2] <= .02]
        for i,j in edges:
            a,b = vertices[i],vertices[j]
            for z in [0,.02]:
                if (a[2]-z)*(b[2]-z)<0:
                    points.append(a+(b-a)*(z-a[2])/(b[2]-a[2]))
        if points:
            pts = np.asarray(points)
            clearance = min(clearance, float(np.min(.009-np.abs(pts[:,0]))), float(np.min(.003-np.abs(pts[:,1]))))
    assert clearance > 0, clearance
    previews = {r['camera']['frame_id']: r['camera'] for r in rows
                if r['camera'] and r['camera'].get('event') == 'preview'}
    assert len(previews) > 30, len(previews)
    report = dict(status='passed', route=cycle['route'], pose_error=error,
                  max_contact_count=0, max_lateral_error_mm=float(max(lateral)*1000),
                  configured_speed_mm_s=speed_limit*1000,
                  max_speed_mm_s=float(max(speed)*1000), min_slot_clearance_mm=clearance*1000,
                  preview_frames=len(previews), preview_final=list(previews.values())[-1])
    np.save(output/'tip_trace.npy', np.c_[times,tip[:,:3,3]])
    (output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=['aruco','pnp','both'], default='both')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/acceptance')
    args = parser.parse_args(); settings = load_settings()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report = {'status':'running','routes':[]}
    try:
        for route in (['aruco','pnp'] if args.route == 'both' else [args.route]):
            output = args.output_dir/route; output.mkdir(parents=True,exist_ok=True)
            process, log, port = start_server(output,args.viewer)
            try:
                with Client(port) as client:
                    if args.viewer:
                        for opened in (True,False,True):
                            set_preview(port,opened)
                            deadline=time.monotonic()+15
                            while time.monotonic()<deadline:
                                status=client.state().get('camera',{})
                                if status.get('event')=='error':raise RuntimeError(status['reason'])
                                if status.get('event')=='preview' and status.get('window_open')==opened:break
                                time.sleep(.1)
                            else:raise RuntimeError('RGB window did not change state')
                    Insertion(client,settings,route,output).run()
            finally: stop_server(process,log)
            result = evaluate(output,settings)
            report['routes'].append(result)
            print('ACCEPTANCE:',json.dumps(result),flush=True)
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
        raise
    finally: (args.output_dir/'cycles.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__': main()
