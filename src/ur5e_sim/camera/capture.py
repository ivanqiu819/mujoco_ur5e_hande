"""Antialiased RGB at the calibrated output resolution, not enlarged detections."""
import cv2
import mujoco


class InsertionCamera:
    def __init__(self, model, spec, settings):
        self.spec = spec
        scale = settings['render_supersample']
        self.renderer = mujoco.Renderer(model, height=spec.height*scale,
                                        width=spec.width*scale)

    def rgb(self, data):
        self.renderer.update_scene(data, camera=self.spec.name)
        rgb = self.renderer.render()
        return cv2.resize(rgb, (self.spec.width, self.spec.height), interpolation=cv2.INTER_AREA)

    def close(self):
        self.renderer.close()
