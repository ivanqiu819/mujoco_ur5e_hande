"""Real socket ownership, live RGB, stale captures and interrupted insertion recovery."""
from pathlib import Path
import time
import numpy as np
from ur5e_sim.paths import ROOT
from ur5e_sim.client import Client
from ur5e_sim.config import load_settings
from ur5e_sim.tasks.insertion import Insertion, tcp_matrix
from ur5e_sim.control.trajectory import line_targets
from check_insertion import start_server, stop_server


def main():
    output=ROOT/'outputs/protocol';output.mkdir(parents=True,exist_ok=True)
    server,log,port=start_server(output)
    client=None
    try:
        client=Client(port)
        # A second control client cannot displace the active owner.
        try:
            other=Client(port);other.close();raise AssertionError('Second controller accepted')
        except ConnectionError: pass
        rgb,snap=client.capture()
        assert rgb.shape==(1072,1280,3) and rgb.dtype==np.uint8
        state=client.state();assert np.max(np.abs(np.asarray(snap['qpos'])-state['qpos']))<.0002
        first_frame=state['camera'].get('frame_id',0)
        time.sleep(.3);state=client.state();assert state['camera']['frame_id']>first_frame
        assert state['camera']['sim_time']<=state['sim_time']
        # A stopped snapshot must not authorize later alignment.
        client.stop()
        try:
            client.move([tcp_matrix(client.state())],capture_id=snap['capture_id'])
            raise AssertionError('Stale capture accepted')
        except RuntimeError as exc: assert 'stale' in str(exc)
        settings=load_settings();task=Insertion(client,settings,'aruco',output)
        task.inspect();task.align()
        start=tcp_matrix(client.state());end=task.world_port.copy()
        end[:3,3]+=end[:3,2]*settings['insert_depth_m']
        client.move(line_targets(start,end@np.linalg.inv(task.tip)),cartesian=True,speed=.005,recovery=start)
        time.sleep(1.2);client.close();client=None;time.sleep(.1)
        client=Client(port);state=client.state()
        assert state['motion']['protected'] and state['motion']['phase']=='stopped',state
        for fields in [dict(command='tcp-rel 0 0 .005'),dict(op='capture')]:
            try: client.request(**fields);raise AssertionError('Protected operation accepted')
            except RuntimeError: pass
        client.request(op='recover');state=client.wait()
        assert not state['motion']['protected'] and state['contact_count']==0
        assert np.linalg.norm(tcp_matrix(state)[:3,3]-start[:3,3])<.0003
        task.client=client
        task.record('recovered')
        task.report.update(status='passed',scenario='disconnect_and_explicit_recovery')
        task.save()
        print('PROTOCOL_CHECK: PASSED',flush=True)
    finally:
        if client: client.close()
        stop_server(server,log)


if __name__=='__main__':main()
