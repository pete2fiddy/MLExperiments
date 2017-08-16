from Function.Cost.CostFunc import CostFunc
import numpy as np

class SquareError(CostFunc):

    @staticmethod
    def cost_func(x, y):
        if len(x.shape) > 1:
            return 0.5 * np.sum((x-y)**2, axis = 1)
        return 0.5 * np.sum((x-y)**2)

    @staticmethod
    def d_func(x, y):
        return x-y
