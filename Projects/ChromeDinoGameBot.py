import numpy as np
import cv2
from mss import mss
from PIL import Image
import timeit

class ChromeDinoGameBot:
    MEDIAN_BLUR_RUN_TIMES = 1
    MEDIAN_BLUR_KERNEL_SIZE = 7
    REMOVE_RIGHT_FRACTION = .60
    REMOVE_TOP_FRACTION = 0.2
    T_REX_DIMS = np.array([58,56])
    LEN_BBOXES_TO_GAME_OVER = 8
    def __init__(self, bbox):
        self.bbox = bbox
        self.max_obstacle_x = ChromeDinoGameBot.REMOVE_RIGHT_FRACTION * self.bbox[2]
        self.min_obstacle_y = ChromeDinoGameBot.REMOVE_TOP_FRACTION * self.bbox[3]
        print("max obstacle x: ", self.max_obstacle_x)
        self.running = False

    def toggle_running(self):
        self.running = not self.running
        if self.running:
            self.start()

    def start(self):
        with mss() as sct:

            old_close_obs_bbox = None
            old_dino_bbox = None
            old_time = timeit.default_timer()
            while(self.running):
                monitor = sct.monitors[0]
                monitor['left'] = self.bbox[0]
                monitor['top'] = self.bbox[1]
                monitor['width'] = self.bbox[2]
                monitor['height'] = self.bbox[3]
                sct_img = sct.grab(monitor)
                img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_RGB2GRAY)
                thresh_image = self.get_thresh_image(img)


                contour_points = self.get_contours(thresh_image)
                bboxes = self.get_contour_points_bboxes(contour_points)


                dino_bbox, new_close_obs_bbox = self.classify_game_bboxes(bboxes)
                if dino_bbox is None:
                    '''game over here'''
                    
                new_time = timeit.default_timer()
                new_close_obs_vel = self.get_bbox_velocity(old_close_obs_bbox, new_close_obs_bbox, old_time, new_time)
                print("obstacle velocity: ", new_close_obs_vel)
                dino_bbox = np.array(dino_bbox).astype(np.int)
                color_thresh_image = cv2.cvtColor(255*thresh_image, cv2.COLOR_GRAY2RGB)
                color_thresh_image = self.draw_bbox(color_thresh_image, dino_bbox, (255,0,0))
                if new_close_obs_bbox is not None:
                    color_thresh_image = self.draw_bbox(color_thresh_image, new_close_obs_bbox, (0,255,0))
                cv2.imshow('thresh image: ', color_thresh_image)
                cv2.waitKey(1)
                '''add in the neural network here'''

                old_close_obs_bbox = new_close_obs_bbox
                old_time = new_time



    def get_bbox_velocity(self, bbox_old, bbox_new, old_time, new_time):
        '''if the new bbox is closer to the dino than before, then is assumed that
        it is the same obstacle as before. Otherwise, sets a velocity of zero
        if the previous obstacle is either None or is not the same obstacle as is
        shown now'''
        if bbox_new is None:
            return None
        if bbox_old is None or bbox_old[0] < bbox_new[0]:
            return 0
        return (bbox_old[0]-bbox_new[0])/(1000*(new_time - old_time))

    def get_thresh_image(self, image):
        thresh_image = cv2.threshold(image, 200, 1, cv2.THRESH_BINARY_INV)[1]
        for i in range(0, ChromeDinoGameBot.MEDIAN_BLUR_RUN_TIMES):
            thresh_image = cv2.medianBlur(thresh_image, ChromeDinoGameBot.MEDIAN_BLUR_KERNEL_SIZE)

        return thresh_image

    '''returns the dinosaur's bounding box and the closest obstacle's bounding
    box'''
    def classify_game_bboxes(self, bboxes):
        '''assumes the dinosaur's bounding box is the farthest left one. If two bboxes
        are fairly close to tying along the x axis, the higher of the two is chosen
        to be the dino's bbox as the dino would be jumping over that obstacle'''
        left_right_sorted_bboxes = sorted(bboxes.tolist(), key = lambda bbox : bbox[0])
        dino_bbox = left_right_sorted_bboxes[0]
        if len(left_right_sorted_bboxes) == 1:
            return dino_bbox, None

        X_MARGIN_OF_ERR = 4
        if dino_bbox[0] + X_MARGIN_OF_ERR >  left_right_sorted_bboxes[1][0] and dino_bbox[1] > left_right_sorted_bboxes[1][1]:
            dino_bbox = left_right_sorted_bboxes[1]

        left_right_sorted_bboxes.remove(dino_bbox)
        self.remove_far_right_bboxes(left_right_sorted_bboxes)
        if len(left_right_sorted_bboxes) >= ChromeDinoGameBot.LEN_BBOXES_TO_GAME_OVER:
            return None, None
        if len(left_right_sorted_bboxes) != 0:
            return dino_bbox, left_right_sorted_bboxes[0]
        return dino_bbox, None


    def remove_far_right_bboxes(self, bboxes):
        i = 0
        while i < len(bboxes):
            if bboxes[i][0] > self.max_obstacle_x and bboxes[i][1] < self.min_obstacle_y:
                del bboxes[i]
            else:
                i += 1


    def get_contours(self, thresh_image):
        contours = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
        contours = self.convert_contours_to_points(contours)
        return contours

    def convert_contours_to_points(self, contours):
        new_contours = []
        for i in range(0, len(contours)):
            new_contours.append(contours[i][:,0,:])
        return new_contours

    def get_contour_points_bboxes(self, contour_points):
        bboxes = np.zeros((len(contour_points), 4))
        for i in range(0, len(contour_points)):
            bboxes[i] = cv2.boundingRect(contour_points[i])
        return bboxes

    def draw_bbox(self, image, bbox, color):
        return cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), color)
