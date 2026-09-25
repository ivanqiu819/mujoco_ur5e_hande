#!/usr/bin/env python3
"""Terminal B: interactive MuJoCo commands/keys. Python standard library only."""
from __future__ import annotations

import argparse
import json
import os
import select
import socket
import sys
import termios
import time
import tty

HELP = '''Commands (press Enter):
  pose / status / joints / contacts
  tcp-rel DX DY DZ             metres, world axes
  tcp X Y Z [QW QX QY QZ]      world pose, WXYZ
  rotate-rel RX RY RZ          degrees, world axes
  gripper close / open        full close / home opening
  gripper gap 20              actual opening target, millimetres
  gripper -0.02               RAW per-slider coordinate in metres
  home / stop                 stop holds arm AND gripper at measured positions
  step 5 2 1                  jog steps: translation mm, rotation deg, slider mm
  speed 0.75                  arm command peak speed, rad/s
  keymode                     single keys, terminal B MUST have focus
  quit                        disconnect; server holds current positions
Keys in keymode (no Enter needed):
  W/S: +X/-X    A/D: +Y/-Y    R/F: +Z/-Z
  I/K: +Rx/-Rx  J/L: +Ry/-Ry  U/O: +Rz/-Rz
  [: full close   ]: full open   ,/.: slider close/open one step
  H: home   P: pose   Space: stop   Enter/Esc: command mode
While BUSY, additional arm goals are rejected instead of building a backlog.
Keep Viewer visible; all control input goes to this terminal.
'''


from ur5e_sim.client import Client

class Connection(Client):
    def request(self, command=None, key=None, quiet=False):
        response = super().request(**({'key':key} if key is not None else {'command':command}))
        if not quiet:
            print(response.get('output') or 'OK', flush=True)
            state = response.get('state', {})
            print('TCP:', state.get('tcp_position_m'), 'WXYZ:', state.get('tcp_quat_wxyz'),
                  'gap_mm:', state.get('gripper_gap_mm'), flush=True)
        return response


class Keyboard:
    def __init__(self):
        self.fd=sys.stdin.fileno()
        self.saved=None
        self.keys=False
        self.line=bytearray()

    def set_mode(self, keys):
        if keys:
            if not os.isatty(self.fd):
                print('keymode requires an interactive terminal'); return
            self.saved=termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        elif self.saved is not None:
            termios.tcsetattr(self.fd,termios.TCSADRAIN,self.saved)
            self.saved=None
        self.keys=keys
        self.line.clear()
        print('KEYMODE: terminal B focus; Enter/Esc to return' if keys else 'COMMAND MODE',flush=True)

    def close(self):
        if self.saved is not None:
            termios.tcsetattr(self.fd,termios.TCSADRAIN,self.saved)
            self.saved=None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8765)
    args=p.parse_args()
    if not 1<=args.port<=65535:
        p.error('port must be in 1..65535')
    connection=None
    keyboard=Keyboard()
    try:
        connection=Connection(args.port)
        print(HELP,flush=True)
        last_ping=time.monotonic()
        last_jog=0.0
        running=True
        while running:
            ready,_,_=select.select([keyboard.fd],[],[],.1)
            if ready:
                chunk=os.read(keyboard.fd,4096)
                if not chunk:
                    break
                if keyboard.keys:
                    # Collapse OS repeat bursts. Never replay a queue of old jogs.
                    for character in chunk.decode('ascii',errors='ignore'):
                        key=character.lower()
                        if key in '\r\n\x1b':
                            keyboard.set_mode(False); break
                        if key not in 'wsadrfikjlou[],.hp ?':
                            continue
                        if key=='?':
                            print(HELP); continue
                        if key not in ' p' and time.monotonic()-last_jog<.15:
                            continue
                        try: connection.request(key=key)
                        except RuntimeError as exc: print('COMMAND_REJECTED:', exc, flush=True)
                        last_jog=time.monotonic()
                        break
                else:
                    keyboard.line.extend(chunk)
                    if len(keyboard.line)>4096:
                        keyboard.line.clear(); print('Command too long'); continue
                    while b'\n' in keyboard.line:
                        line,_,rest=keyboard.line.partition(b'\n')
                        keyboard.line=bytearray(rest)
                        command=line.decode('utf-8',errors='replace').strip()
                        if command in ('quit','exit'):
                            running=False; break
                        if command in ('keymode','keys'):
                            keyboard.set_mode(True); break
                        if command in ('help','?'):
                            print(HELP)
                        elif command:
                            try: connection.request(command=command)
                            except RuntimeError as exc: print('COMMAND_REJECTED:', exc, flush=True)
            if time.monotonic()-last_ping>1:
                connection.request(command='ping',quiet=True)
                last_ping=time.monotonic()
    except (OSError,ConnectionError,ValueError,RuntimeError) as exc:
        print('CLIENT_ERROR:',exc,flush=True)
        print('Start apps/server.py in terminal A and check the port.',flush=True)
    except KeyboardInterrupt:
        print('\nCLIENT_INTERRUPTED',flush=True)
    finally:
        keyboard.close()
        if connection is not None:
            connection.close()
    print('CLIENT_STOPPED: server holds current positions after disconnect',flush=True)


if __name__=='__main__':
    main()
