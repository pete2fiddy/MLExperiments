from Classification.NonLinear.SVM.Kernel.KernelType import KernelType
import numpy as np

'''is a kernel function that just computes the linear dot product'''
class LinearKernel(KernelType):


    def kernel_dot(self, x1, x2):
        return np.dot(x1, x2)

    def multi_kernel_dot(self, X, x2):
        '''uses default for now'''
        return np.matmul(X, x2)
        #return self.default_multi_kernel_dot(X, x2)
