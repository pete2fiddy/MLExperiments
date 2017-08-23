from Function.Activation.ActFunc import ActFunc
import numpy as np

class TanH(ActFunc):

    def act_func(self, X):
        return np.tanh(X)

    def d_func(self, val):
        return 1.0-val**2
