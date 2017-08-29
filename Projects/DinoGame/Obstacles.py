import numpy as np
from Projects.DinoGame.Obstacle import Obstacle
class Obstacles:

    def __init__(self):
        self.obstacles = []
        self.closest_obstacle = None
        self.velocity = 0

    '''assumes obstacle_bboxes are already sorted from left to right'''
    def update(self, obstacle_bboxes):
        new_obstacles = self.create_obstacle_list(obstacle_bboxes)
        self.set_velocity(new_obstacles)
        self.obstacles = new_obstacles
        if len(self.obstacles) > 0:
            self.closest_obstacle = self.obstacles[0]
        else:
            self.closest_obstacle = None

    def create_obstacle_list(self, obstacle_bboxes):
        return [Obstacle(obstacle_bboxes[i]) for i in range(0, len(obstacle_bboxes))]

    def draw(self, image):
        if self.closest_obstacle is not None:
            image = self.closest_obstacle.draw(image)
        return image

    def set_velocity(self, new_obstacles):
        matches = self.match_obstacles(new_obstacles)
        self.velocity = 0
        if len(matches) == 0:
            return
        for i in range(0, len(matches)):
            self.velocity += matches[i][0].displacement(matches[i][1])
        '''dividing by -len(matches) so that the velocity is positive'''
        self.velocity /= -len(matches)

    def match_obstacles(self, new_obstacles):
        new_obstacles_clone = list(new_obstacles)
        obstacle_matches = []
        for i in range(0, len(self.obstacles)):
            iter_obstacle_best_match = self.obstacles[i].closest_match(new_obstacles_clone)
            if iter_obstacle_best_match is not None:
                obstacle_matches.append((self.obstacles[i], iter_obstacle_best_match))
                new_obstacles_clone.remove(iter_obstacle_best_match)
                break
        return obstacle_matches

    def get_ml_feature(self, dino):
        if self.closest_obstacle is None:
            return None
        x_dist = self.closest_obstacle.bbox[0] - dino.xy[0]
        vel = self.velocity
        obs_width = self.closest_obstacle.bbox[2]
        obs_height = self.closest_obstacle.bbox[3]
        return np.array([x_dist, vel, obs_width, obs_height])
