"""Rebuild the complete reproducible scene chain."""
import subprocess
import sys

if __name__ == '__main__':
    for module in ('gripper', 'camera', 'inspection', 'tactile_gels', 'insertion'):
        subprocess.run([sys.executable, '-m', 'ur5e_sim.scenes.'+module], check=True)
