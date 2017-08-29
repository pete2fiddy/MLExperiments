import numpy as np
import cv2
from mss import mss
from PIL import Image
from Projects.DinoGame.ScreenRecorder import ScreenRecorder
from Projects.DinoGame.DinoWorld import DinoWorld
import Projects.DinoGame.KeyboardSim as KeyboardSim
import timeit
from Classification.NonLinear.NeuralNetwork.FeedForwardNN import FeedForwardNN
from Function.Cost.SquareError import SquareError
from Function.Output.Softmax import Softmax
from Function.Activation.TanH import TanH
from Function.Activation.RELU import RELU

class ChromeDinoGameBot2:


    def __init__(self, screen_bbox, neighbors_per_set):
        self.screen_capper = ScreenRecorder(screen_bbox)
        self.world = DinoWorld(screen_bbox)
        self.neighbors_per_set = neighbors_per_set
        self.min_ml_features_to_train = 1+self.neighbors_per_set*2
        self.ml_features = []
        self.dino_actions = []
        self.model = FeedForwardNN(np.zeros((1,4)), np.zeros((1,2)), TanH(), SquareError, Softmax, (3,2))
        print('model: ', self.model)

    def start(self):
        '''last_game_state is used so that, on death, the model does not
        train off of data multiple times (which includes nonsensical,
        "game over" data). Checks to see if the last game state is not game
        over, then sets the last game state to game over so that it does not
        train more than once'''
        last_game_state = "alive"
        while(True):
            thresh_image = self.grab_and_preprocess_snap()
            game_state = self.world.update(thresh_image)


            '''check here if game state is "game_over", train
            ML model if it is'''
            if game_state == "alive":
                last_game_state = "alive"
                self.update_ml_features()
                if self.ml_features[len(self.ml_features)-1] is not None:
                    net_prediction = self.model.predict(self.ml_features[len(self.ml_features)-1])
                    KeyboardSim.press_key(net_prediction)
                    #print("net prediction: ", net_prediction)
            else:
                if last_game_state != "game_over" and len(self.ml_features) > self.min_ml_features_to_train:
                    self.train_net()
                    print("----------------------------------")

                KeyboardSim.press_key(1)
                self.ml_features = []
                self.dino_actions = []

                last_game_state = "game_over"





    def update_ml_features(self):
        NUM_ML_FEATURES_TO_TRACK = 20
        if len(self.ml_features) >= NUM_ML_FEATURES_TO_TRACK:
            del self.ml_features[0]
            del self.dino_actions[0]
        self.ml_features.append(self.world.get_ml_feature())
        self.dino_actions.append(self.world.dino.state)

    def train_net(self):
        X,y = None, None
        if self.dino_actions[len(self.dino_actions)-1] == "still":
            '''train net assuming that the prior frame should have been a jump.
            Additionally, assume NUM_NEIGHBORING_FRAMES prior frames to this
            one should be classified as staying still'''
            X,y = self.generate_still_death_data()
            print("cause of death: still")

        elif self.dino_actions[len(self.dino_actions)-1] == "falling":
            '''train net assuming last "jump" command should have been later, +
            FRAME_DELTA frames from the actual jump.  Additionally, assume data
            within +- NUM_NEIGHBORING_FRAMES within the new jump frame should
            be classified as staying'''
            X,y = self.generate_falling_death_data()
            print("cause of death: falling")

        elif self.dino_actions[len(self.dino_actions)-1] == "jumping":
            '''train net assuming last "jump" command should have been earlier, -
            FRAME_DELTA frames from actual jump. Additionally, assume data
            within +- NUM_NEIGHBORING_FRAMES within the new jump frame should
            be classified as staying'''
            X,y = self.generate_jumping_death_data()
            print("cause of death: jumping")
        print("last few dino actions: ", self.dino_actions[len(self.dino_actions)-5:])
        print("X: ", X)
        print("y: ", y)
        self.step_net(X,y)

    def generate_still_death_data(self):
        X = self.ml_features[len(self.ml_features)- self.neighbors_per_set :]
        y = [np.array([0,1]) for i in range(0, len(X))]
        #y[len(y)-1] = np.array([0,1])
        return X,y

    def generate_jumping_death_data(self):
        '''assume the frame prior to jump should have been a jump,
        and frame before should be still'''
        jump_index = self.get_last_jump_index()
        #prior_jump_index = jump_index - 1
        X = self.ml_features[jump_index - self.neighbors_per_set : jump_index]#[self.ml_features[prior_jump_index], self.ml_features[prior_jump_index]]
        y = [np.array([0,1]) for i in range(0, len(X))]#[np.array([1,0]), np.array([0,1])]
        y[0] = np.array([1,0])
        return X,y

    def generate_falling_death_data(self):
        '''assume the frames within neighbors_per_set after jump should have
        been a jump, and jump frame should be still'''
        jump_index = self.get_last_jump_index()
        #after_jump_index = jump_index + 1
        X = self.ml_features[jump_index : jump_index + self.neighbors_per_set]#[self.ml_features[jump_index], self.ml_features[after_jump_index]]
        y = [np.array([1,0]) for i in range(0, len(X))]
        y[0] = np.array([0,1])
        return X,y

    def get_last_jump_index(self):
        for i in range(len(self.dino_actions)-1, 1, -1):
            if self.dino_actions[i] == "jumping" and self.dino_actions[i-1] != "jumping":
                return i
        return None




    def step_net(self, X, y):
        for i in range(0, len(X)):
            self.model.train_step(X[i], y[i], learn_rate = 0.1, bias_learn_rate = 0.01)

    def grab_and_preprocess_snap(self):
        thresh_image = self.binaritize_bw_image(cv2.cvtColor(self.screen_capper.snap(), cv2.COLOR_RGB2GRAY))
        return thresh_image

    def binaritize_bw_image(self, bw_image):
        MEDIAN_BLUR_RUN_TIMES = 1
        MEDIAN_BLUR_KERNEL_SIZE = 7
        thresh_image = cv2.threshold(bw_image, 200, 1, cv2.THRESH_BINARY_INV)[1]
        for i in range(0, MEDIAN_BLUR_RUN_TIMES):
            thresh_image = cv2.medianBlur(thresh_image, MEDIAN_BLUR_KERNEL_SIZE)
        return thresh_image
