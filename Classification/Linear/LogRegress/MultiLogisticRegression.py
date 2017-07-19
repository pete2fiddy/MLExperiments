import numpy as np
from math import exp
from Classification.Linear.LogRegress.LogisticRegression import LogisticRegression
import Classification.SetHelper as SetHelper
from Classification.Classifiable import Classifiable

'''Should convert class labels (assuming they are non-integers from 0 to n feature type).
Keep references to strings as class integers'''
class MultiLogisticRegression(Classifiable):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.num_classes = SetHelper.get_num_classes(self.y)
        '''may want to have one fewer model than required
        and subtract the sum of the other probabilities from
        one to garner the output of the missing model.'''
        self.models = [LogisticRegression(self.X,self.y,i) for i in range(0, self.num_classes)]

    def train(self, num_iter = 1000, learn_rate = 0.001):
        for i in range(0, len(self.models)):
            self.models[i].train(num_iter = num_iter, learn_rate = learn_rate)

    def predict(self, x):
        predictions = np.array([self.models[i].predict(x) for i in range(0, len(self.models))])
        predictions /= np.sum(predictions)
        predict_index = np.argmax(predictions)
        predict_probability = predictions[predict_index]
        return predict_index, predict_probability
    '''
    def get_train_accuracy(self):
        num_correct = 0
        for i in range(0, self.y.shape[0]):
            if self.grade_prediction_of_x(self.X[i], self.y[i]):
                num_correct += 1
        return num_correct/self.y.shape[0]

    def grade_prediction_of_x(self, x, y):
        prediction, _ = self.predict(x)
        return (prediction == y)
    '''
    def get_train_accuracy(self):
        num_correct = 0
        predictions = self.predict_set(self.X)
        for i in range(0, predictions.shape[0]):
            if predictions[i][0] == self.y[i]:
                num_correct += 1
        return num_correct/self.X.shape[0]
