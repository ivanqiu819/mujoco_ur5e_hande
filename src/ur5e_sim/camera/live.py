"""One render process consumes immutable server snapshots; it never steps physics."""
from __future__ import annotations
import json
import multiprocessing as mp
import queue
import socket
import struct
import threading
import time
import uuid

import mujoco
import numpy as np

from ur5e_sim.config_io import read_config
from .calibration import camera_transforms, load_spec


def render_worker(scene, config, states, captures, results, commands, shutdown, show):
    # OpenGL contexts belong exclusively to this process.
    from .capture import InsertionCamera
    import cv2
    spec = load_spec(config)
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    scale = read_config(config)['render_supersample']
    preview = mujoco.Renderer(model, height=spec.height, width=spec.width)
    capture = InsertionCamera(model, spec, {'render_supersample': scale})
    window = None
    label = None
    photo = None
    frames = 0
    start = time.monotonic()
    def open_window():
        import tkinter as tk
        nonlocal window, label
        if window is not None:
            return
        window = tk.Tk()
        window.title('UR5e • Live RGB')
        label = tk.Label(window)
        label.pack()
        window.protocol('WM_DELETE_WINDOW', close_window)
    def close_window():
        nonlocal window, label
        if window is not None:
            window.destroy()
        window = label = None
    try:
        if show:
            open_window()
        results.put({'event': 'ready'})
        while not shutdown.is_set():
            try:
                command = commands.get_nowait()
                if command == 'open': open_window()
                if command == 'close': close_window()
            except queue.Empty:
                pass
            if window is not None:
                window.update_idletasks(); window.update()
            is_capture = True
            try:
                snapshot = captures.get_nowait()
            except queue.Empty:
                is_capture = False
                try:
                    snapshot = states.get(timeout=.01)
                except queue.Empty:
                    continue
            mujoco.mj_setState(model, data, snapshot.pop('physics'), mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(model, data)
            if is_capture:
                rgb = capture.rgb(data)
                results.put({'event': 'capture', 'meta': snapshot, 'rgb': rgb}, timeout=2)
            else:
                preview.update_scene(data, camera=spec.name)
                rgb = preview.render()
                frames += 1
                fps = frames/max(.001, time.monotonic()-start)
                if window is not None:
                    from PIL import Image, ImageTk
                    # Display may resize; detection always receives the calibrated dimensions.
                    view = Image.fromarray(rgb)
                    view.thumbnail((960, 804))
                    photo = ImageTk.PhotoImage(view)
                    label.configure(image=photo)
                    window.title(f"Live RGB | frame {snapshot['frame_id']} | sim {snapshot['sim_time']:.3f}s | {fps:.1f} fps")
                results.put({'event': 'preview', 'frame_id': snapshot['frame_id'],
                             'sim_time': snapshot['sim_time'], 'fps': fps,
                             'age_s': time.monotonic()-snapshot['wall_time'],
                             'window_open': window is not None}, timeout=2)
    except BaseException as exc:
        try: results.put({'event':'error', 'reason':f'{type(exc).__name__}: {exc}'}, timeout=1)
        except queue.Full: pass
    finally:
        close_window()
        capture.close(); preview.close()


class CameraService:
    def __init__(self, model, scene, config, port, show=False):
        self.model = model
        self.spec = load_spec(config)
        self.serial = 0
        self.last_preview = 0.
        self.status = {'event': 'starting'}
        self.pending = set()
        self.frames = {}
        self.condition = threading.Condition()
        self.closed = False
        self.token = uuid.uuid4().hex
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(('127.0.0.1', port))
        self.listener.listen(4); self.listener.settimeout(.2)
        self.port = self.listener.getsockname()[1]
        ctx = mp.get_context('spawn')
        self.states = ctx.Queue(1); self.captures = ctx.Queue(2)
        self.results = ctx.Queue(8); self.commands = ctx.Queue(4); self.shutdown = ctx.Event()
        self.process = ctx.Process(target=render_worker, args=(str(scene), str(config), self.states,
                                  self.captures, self.results, self.commands, self.shutdown, show), daemon=True)
        self.process.start()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def snapshot(self, data):
        self.serial += 1
        physics = np.empty(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.model, data, physics, mujoco.mjtState.mjSTATE_INTEGRATION)
        _, world_camera = camera_transforms(self.model, data, self.spec.name)
        return {'frame_id': self.serial, 'sim_time': float(data.time), 'wall_time': time.monotonic(),
                'qpos': data.qpos.tolist(), 'T_world_camera': world_camera.tolist(),
                'intrinsic_matrix': self.spec.intrinsic_matrix.tolist(),
                'shape': [self.spec.height, self.spec.width, 3], 'physics': physics}

    def capture(self, data):
        if not self.process.is_alive() or self.status.get('event') == 'error':
            raise ValueError('Camera renderer unavailable')
        if len(self.pending) >= 2:
            raise ValueError('Camera capture queue full')
        snapshot = self.snapshot(data)
        snapshot['capture_id'] = f'{self.token}-{snapshot["frame_id"]}'
        try: self.captures.put_nowait(snapshot)
        except queue.Full: raise ValueError('Camera capture queue full')
        self.pending.add(snapshot['capture_id'])
        return {k:v for k,v in snapshot.items() if k != 'physics'} | {'port':self.port}

    def poll(self, data):
        while True:
            try: result = self.results.get_nowait()
            except queue.Empty: break
            if result['event'] == 'capture':
                token = result['meta']['capture_id']
                with self.condition:
                    self.frames[token] = result
                    self.pending.discard(token)
                    while len(self.frames) > 8: del self.frames[next(iter(self.frames))]
                    self.condition.notify_all()
            else:
                self.status = result
        now = time.monotonic()
        if now-self.last_preview >= 1/self.spec.preview_hz:
            snapshot = self.snapshot(data)
            try: self.states.put_nowait(snapshot)
            except queue.Full:
                try: self.states.get_nowait()
                except queue.Empty: pass
                try: self.states.put_nowait(snapshot)
                except queue.Full: pass
            self.last_preview = now

    def window(self, opened):
        if not self.process.is_alive(): raise ValueError('Camera renderer unavailable')
        try: self.commands.put_nowait('open' if opened else 'close')
        except queue.Full: raise ValueError('Preview command queue full')

    def _serve(self):
        while not self.closed:
            try: client, _ = self.listener.accept()
            except socket.timeout: continue
            except OSError: break
            # The read-only binary channel cannot acquire motion control.
            with client:
                client.settimeout(2)
                try:
                    request = bytearray()
                    while b'\n' not in request and len(request) <= 256:
                        chunk = client.recv(256)
                        if not chunk: break
                        request.extend(chunk)
                    token = bytes(request).split(b'\n')[0].decode('ascii')
                    if token in ('preview:open','preview:close'):
                        try:
                            self.window(token == 'preview:open')
                            client.sendall(b'OK\n')
                        except ValueError as exc:
                            client.sendall(('ERROR: '+str(exc)+'\n').encode())
                        continue
                    deadline = time.monotonic()+30
                    with self.condition:
                        while token not in self.frames and not self.closed:
                            if token not in self.pending or time.monotonic() >= deadline: break
                            self.condition.wait(.1)
                        result = self.frames.get(token)
                    if result is None: continue
                    header = json.dumps(result['meta'], allow_nan=False).encode()
                    client.sendall(struct.pack('!I', len(header))+header+result['rgb'].tobytes())
                except (OSError, UnicodeError): pass

    def close(self):
        self.closed = True
        with self.condition: self.condition.notify_all()
        self.listener.close(); self.thread.join(timeout=3)
        self.shutdown.set(); self.process.join(timeout=3)
        if self.process.is_alive(): self.process.terminate(); self.process.join(timeout=2)
        self.process.close()
        for q in (self.states, self.captures, self.results, self.commands):
            q.cancel_join_thread(); q.close()
