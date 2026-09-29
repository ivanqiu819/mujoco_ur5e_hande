"""Reusable local RPC client; no task or detector logic."""
from __future__ import annotations
import json
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def recv_exact(sock, count):
    chunks = bytearray()
    while len(chunks) < count:
        chunk = sock.recv(count-len(chunks))
        if not chunk: raise ConnectionError('Truncated camera frame')
        chunks.extend(chunk)
    return bytes(chunks)


def read_frame(meta):
    with socket.create_connection(('127.0.0.1', meta['port']), timeout=35) as sock:
        sock.sendall(meta['capture_id'].encode()+b'\n')
        size, = struct.unpack('!I', recv_exact(sock, 4))
        if size > 65536: raise ValueError('Invalid frame header length')
        actual = json.loads(recv_exact(sock, size))
        if actual['capture_id'] != meta['capture_id'] or actual['frame_id'] != meta['frame_id']:
            raise ValueError('Camera frame identity mismatch')
        shape = tuple(actual['shape'])
        if len(shape) != 3 or shape[2] != 3 or not 0 < np.prod(shape) <= 64000000:
            raise ValueError('Invalid RGB shape')
        rgb = np.frombuffer(recv_exact(sock, int(np.prod(shape))), np.uint8).reshape(shape).copy()
        return rgb, actual


class Client:
    def __init__(self, port=8765):
        self.sock = socket.create_connection(('127.0.0.1', port), timeout=60)
        self.buffer = bytearray(); self.serial = 0
        try:
            self.ready = self._read()
        except BaseException:
            self.sock.close()
            raise

    def _read(self):
        while b'\n' not in self.buffer:
            chunk = self.sock.recv(8192)
            if not chunk: raise ConnectionError('Server disconnected or another controller owns it')
            self.buffer.extend(chunk)
            if len(self.buffer) > 65536: raise ConnectionError('Reply too large')
        line, _, rest = self.buffer.partition(b'\n'); self.buffer = bytearray(rest)
        return json.loads(line)

    def request(self, **fields):
        self.serial += 1
        payload = json.dumps(dict(id=self.serial, **fields), allow_nan=False).encode()+b'\n'
        if len(payload) > 65536: raise ValueError('Control request too large')
        self.sock.sendall(payload)
        result = self._read()
        if result.get('id') != self.serial: raise ConnectionError('Unexpected reply sequence')
        if not result.get('ok'): raise RuntimeError(result.get('error') or result.get('output'))
        return result

    def state(self): return self.request(op='state')['result']

    def read_tactile(self): return self.request(op='tactile')['result']
    def stop(self): return self.request(op='stop')

    def reset_home(self): return self.request(op='reset_home')

    def release_plug(self): return self.request(op='release_plug')

    def seat_plug(self): return self.request(op='seat_plug')

    def capture(self):
        meta = self.request(op='capture')['result']
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(read_frame, meta)
            while not future.done():
                self.request(command='ping'); time.sleep(.1)
            rgb, actual = future.result()
        return rgb, actual

    def move(self, targets, *, cartesian=False, speed=None, recovery=None, capture_id=None,
             collision_samples=320, settle_s=.3):
        state = self.state()
        return self.request(op='motion', start_qpos=state['qpos'],
                            targets=[np.asarray(t).tolist() for t in targets], cartesian=cartesian,
                            speed=speed, recovery=None if recovery is None else np.asarray(recovery).tolist(),
                            capture_id=capture_id, collision_samples=collision_samples, settle_s=settle_s)

    def wait(self, timeout=180):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            state = self.state(); phase = state['motion']['phase']
            if phase in ('error','stopped'): raise RuntimeError(state['motion']['reason'] or phase)
            if not state['arm_moving'] and phase != 'moving': return state
            time.sleep(.02)
        self.stop(); raise TimeoutError('Motion completion timed out')

    def wait_gripper(self, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.state()
            gs = state.get('gripper_state')
            if gs == 'at_target':
                return state
            if gs == 'contact_blocked':
                raise RuntimeError('Gripper opening blocked by contact')
            time.sleep(.02)
        raise TimeoutError('Gripper motion timed out')

    def close(self): self.sock.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def set_preview(port=8765, opened=True):
    """Change only the RGB window; never acquires the robot control connection."""
    with socket.create_connection(('127.0.0.1',port+1),timeout=5) as sock:
        sock.sendall(b'preview:open\n' if opened else b'preview:close\n')
        reply=sock.recv(1024).decode().strip()
        if reply != 'OK': raise RuntimeError(reply)
