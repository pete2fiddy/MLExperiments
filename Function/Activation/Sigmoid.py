from Function.Activation.ActFunc import ActFunc
from Function.Output.OutputFunc import OutputFunc
import numpy as np

class Sigmoid(ActFunc, OutputFunc):
    '''possible for "overflow encountered in exp" to occur'''
    def act_func(self, X):
        return 1.0/(1.0 + np.exp(-X))

    def out_func(self, X):
        return Sigmoid.act_func(X)

    def d_func(self, val):
        return val * (1.0-val)

    '''
    @staticmethod
    def multi_act_func(X):
        return 1.0/(1.0 + np.exp(-X))
    '''
