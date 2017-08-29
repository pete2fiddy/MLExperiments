import numpy as np
import cv2
import Projects.DinoGame.GameStateDrawer as GameStateDrawer
from Projects.DinoGame.Dino import Dino
from Projects.DinoGame.Obstacles import Obstacles

class DinoWorld:
    '''the score is in the top right portion of the screen, anything in this area
    (specified by the fractions below, fraction X, fraction Y) of the screen
    will be deleted from the bounding boxes'''
    OBSTACLE_REJECT_FRACTION = np.array([.6, .2])


    def __init__(self, screen_bbox):
        self.screen_bbox = screen_bbox
        self.obstacle_reject_area = DinoWorld.OBSTACLE_REJECT_FRACTION * screen_bbox[2:]
        self.game_state = "alive"
        self.dino = Dino([0,0,0,0])
        self.obstacles = Obstacles()

    def update(self, thresh_image):
        bboxes = self.get_game_bboxes(thresh_image)
        self.set_game_over(bboxes)
        dino_bbox, obstacle_bboxes = self.classify_bboxes(bboxes)

        self.obstacles.update(obstacle_bboxes)
        self.dino.update(dino_bbox)
        color_thresh_image = cv2.cvtColor(thresh_image*255, cv2.COLOR_GRAY2RGB)
        color_thresh_image = self.dino.draw(color_thresh_image)
        '''for i in range(0, len(obstacle_bboxes)):
            color_thresh_image = GameStateDrawer.draw_bbox(color_thresh_image, obstacle_bboxes[i], (0,0,255))'''
        color_thresh_image = self.obstacles.draw(color_thresh_image)
        cv2.imshow("Dino World: ", color_thresh_image)
        cv2.waitKey(1)
        return self.game_state

    def get_game_bboxes(self, thresh_image):
        contour_points = self.get_contours(thresh_image)
        bboxes = []
        for i in range(0, len(contour_points)):
            bboxes.append(cv2.boundingRect(contour_points[i]))
        self.remove_non_obstacle_bboxes(bboxes)
        return bboxes

    def set_game_over(self, bboxes):
        NUM_BBOXES_GAME_OVER = 8
        if len(bboxes) >= NUM_BBOXES_GAME_OVER:
            self.game_state = "game_over"
        else:
            self.game_state = "alive"

    def get_contours(self, thresh_image):
        contours = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
        contours = self.convert_contours_to_points(contours)
        return contours

    def convert_contours_to_points(self, contours):
        new_contours = []
        for i in range(0, len(contours)):
            new_contours.append(contours[i][:,0,:])
        return new_contours

    '''returns the dinosaur's bounding box and the bounding boxes of all
     obstacles, sorted from left to right'''
    def classify_bboxes(self, bboxes):
        dino_bbox = self.get_and_remove_dino_bbox(bboxes)
        return dino_bbox, bboxes

    def get_and_remove_dino_bbox(self, bboxes):
        if len(bboxes) == 1:
            return bboxes.pop(0)
        bboxes.sort(key = lambda bbox : bbox[0])
        FARTHEST_LEFT_TIE_MARGIN = 4
        if bboxes[0][0] + FARTHEST_LEFT_TIE_MARGIN > bboxes[1][0] and bboxes[1][1] < bboxes[0][1]:
            return bboxes.pop(1)
        return bboxes.pop(0)

    def remove_non_obstacle_bboxes(self, bboxes):
        i = 0
        while i < len(bboxes):
            if bboxes[i][0] > self.obstacle_reject_area[0] and bboxes[i][1] < self.obstacle_reject_area[1]:
                del bboxes[i]
            else:
                i+=1

    def get_ml_feature(self):
        return self.obstacles.get_ml_feature(self.dino)
