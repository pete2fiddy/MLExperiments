from abc import ABC, abstractmethod
import numpy as np

class MultiClassifiable(ABC):

    @abstractmethod
    def predict(self, x):
        pass

    '''some models do not need to be trained,
    others do. If models need to be trained, override
    this function'''
    def train(self):
        return None

    def predict_set(self, X):
        x_predicts = []
        for i in range(0, X.shape[0]):

            x_predicts.append(self.predict(X[i]))
        x_predicts = np.asarray(x_predicts)
        return x_predicts
