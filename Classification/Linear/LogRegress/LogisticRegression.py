import numpy as np
import Classification.BinaryHelper as BinaryHelper
from math import exp, log
from Classification.Classifiable import Classifiable

'''may want to think about extending an abstract class for classifiers,
but don't know enough about the nature of my future classifiers to do so
in a non-restrictive way'''
class LogisticRegression(Classifiable):

    def __init__(self, X, y, positive_y_val):
        self.X = X
        self.y = y
        self.pos_y_val = positive_y_val
        self.y = BinaryHelper.modify_y_to_positive_y_val(self.y, self.pos_y_val)
        self.weights = ((np.random.rand(X.shape[1])*5.0)-2.5).astype(np.float32)

    def train(self, num_iter = 1000, learn_rate  = 0.001):
        for i in range(0, num_iter):
            self.weights += learn_rate * self.gradient()
            if i%100 == 0:
                print("Accuracy at ", i, "th iteration: ", self.get_train_accuracy())

    def predict(self, x):
        return self.sigmoid(np.dot(x, self.weights))

    def sigmoid(self, x):
        gamma = -x
        if gamma < 0:
            return 1.0 - 1.0 / (1.0 + exp(gamma))
        return 1.0 / (1.0 + exp(-gamma))

    def gradient(self):
        predictions = self.predict_set(self.X)
        gradient = np.zeros(self.X.shape[1])
        for i in range(0, self.X.shape[0]):
            gradient += (predictions[i] - self.y[i]) * self.X[i]
        return gradient

    def get_train_accuracy(self):
        num_correct = 0
        for i in range(0, self.X.shape[0]):
            if self.grade_prediction_of_x(self.X[i], self.y[i]):
                num_correct += 1
        return num_correct/self.X.shape[0]

    def grade_prediction_of_x(self, x, y):
        x_predict = self.predict(x)
        return (int(round(x_predict)) == y)
