"""Run default ArUco then PnP as separate clients on one unchanged server."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from ur5e_sim.paths import ROOT
from ur5e_sim.config import load_settings
from check_insertion import start_server, stop_server, evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/repeat_insertion')
    args=parser.parse_args();output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    report={'status':'running','routes':[],'same_server':True}
    server,log,port=start_server(output)
    try:
        for route in ('aruco','pnp'):
            folder=output/route;folder.mkdir(exist_ok=True)
            command=[sys.executable,str(ROOT/'apps/insert_socket.py'),'--port',str(port),'--output-dir',str(folder)]
            if route=='pnp':command+=['--route','pnp']
            with (folder/'client.log').open('w') as client_log:
                result=subprocess.run(command,stdout=client_log,stderr=subprocess.STDOUT,timeout=180)
            if result.returncode:
                raise RuntimeError((folder/'client.log').read_text())
            cycle=json.loads((folder/'cycle.json').read_text())
            assert cycle['status']=='passed',cycle
            home=cycle['phases']['home']
            assert np.max(np.abs(np.asarray(home['qpos'])-home['home_qpos']))<=.001,home
            assert home['contact_count']==0 and not home['motion']['protected'],home
            print('REPEAT_CYCLE_PASSED:',route,'home_time=',home['sim_time'],flush=True)
        stop_server(server,log)
        # Split only recorded evaluation data; both clients ran on one server.
        with gzip.open(output/'physics.jsonl.gz','rt') as stream:
            rows=[json.loads(line) for line in stream]
        lower=0.
        for route in ('aruco','pnp'):
            folder=output/route
            cycle=json.loads((folder/'cycle.json').read_text())
            upper=cycle['phases']['home']['sim_time']
            with gzip.open(folder/'physics.jsonl.gz','wt') as stream:
                for row in rows:
                    if lower<=row['sim_time']<=upper:stream.write(json.dumps(row)+'\n')
            lower=upper
            result=evaluate(folder,load_settings())
            home=cycle['phases']['home']
            result['home_max_joint_error']=float(np.max(np.abs(np.asarray(home['qpos'])-home['home_qpos'])))
            report['routes'].append(result)
        report['status']='passed'
        print('REPEAT_INSERTION_CHECK: PASSED',flush=True)
    except BaseException as exc:
        report.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if server.poll() is None:stop_server(server,log)
        (output/'repeat.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
