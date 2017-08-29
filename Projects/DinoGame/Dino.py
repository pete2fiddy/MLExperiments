import numpy as np
import Projects.DinoGame.GameStateDrawer as GameStateDrawer

class Dino:

    def __init__(self, bbox):
        '''not sure if is np array'''
        self.xy = np.asarray(bbox[:2])
        self.bbox = bbox
        self.velocity = np.zeros(2)
        self.state = "still"

    def update(self, new_bbox):
        new_xy = np.asarray(new_bbox[:2])
        self.velocity = new_xy - self.xy
        self.xy = new_xy
        self.bbox = new_bbox
        self.update_state()

    def update_state(self):
        MIN_JUMPING_VEL_MAG = 2
        if (self.state == "still" and abs(self.velocity[1]) > MIN_JUMPING_VEL_MAG) or (self.state == "jumping" and self.velocity[1] < 0):
            self.state = "jumping"
        elif (self.state == "jumping" or self.state == "falling") and self.velocity[1] > 0:
            self.state = "falling"

        elif self.state != "jumping":
            self.state = "still"



    def draw(self, image):
        return GameStateDrawer.draw_bbox(image, self.bbox, (0,255,0))
