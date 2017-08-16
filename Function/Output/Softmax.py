from Function.Output.OutputFunc import OutputFunc
import numpy as np

class Softmax(OutputFunc):

    @staticmethod
    def out_func(x):
        exps = np.exp(x)
        return exps/np.sum(exps)

    @staticmethod
    def d_func(val):
        return val*(1.0-val)
