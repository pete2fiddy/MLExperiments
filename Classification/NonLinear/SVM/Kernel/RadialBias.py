from Classification.NonLinear.SVM.Kernel.KernelType import KernelType
import numpy as np

class RadialBias(KernelType):

    def __init__(self, gamma):
        self.gamma = gamma

    def kernel_dot(self, x1, x2):
        return np.exp(-self.gamma*np.linalg.norm(x1-x2)**2)

    def multi_kernel_dot(self, X, x2):
        X_sub_x2_mags_sqrd = np.linalg.norm(X-x2, axis = 1)**2
        return np.exp(-self.gamma*X_sub_x2_mags_sqrd)
