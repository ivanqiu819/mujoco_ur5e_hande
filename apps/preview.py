"""Open or close the live RGB window without acquiring motion control."""
import argparse
from ur5e_sim.client import set_preview

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--close',action='store_true')
    args=parser.parse_args()
    set_preview(args.port,not args.close)
