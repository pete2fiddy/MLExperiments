from Function.Activation.ActFunc import ActFunc
import numpy as np

class RELU(ActFunc):
    DEFAULT_POS_SLOPE = 1.0
    DEFAULT_NEG_SLOPE = 0.0


    def __init__(self, pos_slope = None, neg_slope = None):
        self.pos_slope = pos_slope if pos_slope is not None else RELU.DEFAULT_POS_SLOPE
        self.neg_slope = neg_slope if neg_slope is not None else RELU.DEFAULT_NEG_SLOPE

    def act_func(self, X):
        slopes = self.get_slopes(X)
        return X * slopes


    def d_func(self, val):
        return self.get_slopes(val)

    '''returns a 1d vector of slopes for each input, assigning POS_SLOPE to each
    index whose value in X is positive, and NEG_SLOPE to each index whose value
    in X is negative'''
    def get_slopes(self, X):
        slopes = np.zeros(X.shape[0])
        slopes[X >= 0] = self.pos_slope
        slopes[X < 0] = self.neg_slope
        return slopes
