from abc import ABC, abstractmethod
import numpy as np

class Classifiable:

    def predict_set(self, X):
        predictions = []
        for i in range(0, X.shape[0]):
            predictions.append(self.predict(X[i]))
        predictions = np.array(predictions)
        return predictions

    @abstractmethod
    def predict(self, x):
        pass
