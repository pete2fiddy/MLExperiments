from abc import ABC, abstractmethod
import numpy as np

class KernelType(ABC):


    @abstractmethod
    def kernel_dot(self, x1, x2):
        pass

    '''many kernel dot products can be sped up, as they often
    must be computed many times and matrix operations can be used
    rather than a for loop. For those transformations that cannot
    be simplified this way, a default method is provided and can be
    called that just uses a for loop'''
    @abstractmethod
    def multi_kernel_dot(self, X, x2):
        pass

    def multi_kernel_dot_default(self, X, x2):
        dots_out = np.zeros((X.shape[0]))
        for i in range(0, dots_out.shape[0]):
            dots_out[i] = self.kernel_dot(X[i], x2)
        return dots_out
