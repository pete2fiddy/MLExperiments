from Function.Activation.ActFunc import ActFunc
from Function.Output.OutputFunc import OutputFunc
import numpy as np

class Sigmoid(ActFunc, OutputFunc):
    '''possible for "overflow encountered in exp" to occur'''
    @staticmethod
    def act_func(X):
        return 1.0/(1.0 + np.exp(-X))

    @staticmethod
    def out_func(X):
        return Sigmoid.act_func(X)

    @staticmethod
    def d_func(val):
        return val * (1.0-val)

    '''
    @staticmethod
    def multi_act_func(X):
        return 1.0/(1.0 + np.exp(-X))
    '''
