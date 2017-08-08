import Classification.BinaryHelper as BinaryHelper
from Classification.Classifiable import Classifiable
import numpy as np

class Perceptron(Classifiable):

    def __init__(self, X, y, positive_y_val):
        self.X = X
        self.y = y
        self.pos_y_val = positive_y_val
        self.y = BinaryHelper.modify_y_to_positive_y_val(self.y, self.pos_y_val)
        self.y[y == 0] = -1
        self.weights = np.zeros((self.X.shape[1] + 1))

    def train(self, learn_rate = 1.0):
        num_correct = 0
        for i in range(0, self.X.shape[0]):
            x = self.X[i]
            y = self.y[i]
            x_predict = self.predict(x, zero_to_one_range = False)
            if x_predict == y:
                num_correct += 1
            self.weights += learn_rate * (y - x_predict) * np.append(x, 1)

    def predict(self, x, zero_to_one_range = True):
        x_with_bias = np.append(x, 1)
        weight_dot = np.dot(x_with_bias, self.weights)
        if not zero_to_one_range:
            return 1 if weight_dot >= 0 else -1
        return 1 if weight_dot >= 0 else 0
