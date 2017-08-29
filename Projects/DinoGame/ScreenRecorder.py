from mss import mss
import numpy as np

class ScreenRecorder:

    def __init__(self, screen_bbox):
        self.screen_bbox = screen_bbox

    def snap(self):
        with mss() as sct:
            monitor = sct.monitors[0]
            monitor['left'] = self.screen_bbox[0]
            monitor['top'] = self.screen_bbox[1]
            monitor['width'] = self.screen_bbox[2]
            monitor['height'] = self.screen_bbox[3]
            return np.array(sct.grab(monitor))
        return None
