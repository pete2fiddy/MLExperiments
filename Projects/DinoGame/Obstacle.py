import numpy as np
import Projects.DinoGame.GameStateDrawer as GameStateDrawer

class Obstacle:

    def __init__(self, bbox):
        self.bbox = np.asarray(bbox)
    
    def draw(self, image):
        return GameStateDrawer.draw_bbox(image, self.bbox, (255,0,0))

    def match_score(self, obstacle):
        return np.linalg.norm(self.bbox[2:] - obstacle.bbox[2:])

    def closest_match(self, obstacles):
        best_match = None
        best_match_score = None
        for i in range(0, len(obstacles)):
            iter_match_score = self.match_score(obstacles[i])
            if (best_match_score is None or iter_match_score < best_match_score) and obstacles[i].bbox[0] < self.bbox[0]:
                best_match_score = iter_match_score
                best_match = obstacles[i]
        return best_match


    def displacement(self, new_obstacle):
        return new_obstacle.bbox[0] - self.bbox[0]
