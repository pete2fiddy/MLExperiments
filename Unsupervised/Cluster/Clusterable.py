from abc import ABC, abstractmethod
import numpy as np

class Clusterable(ABC):
    @abstractmethod
    def cluster(self, x):
        pass

    def cluster_set(self, X):
        responses = np.zeros((X.shape[0]))
        for i in range(0, responses.shape[0]):
            responses[i] = self.cluster(X[i])
        return responses
