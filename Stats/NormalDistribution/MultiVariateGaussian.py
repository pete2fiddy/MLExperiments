import numpy as np
from math import pi, sqrt
import Stats.NormalDistribution.GaussianParams as GaussianParams

class MultiVariateGaussian:

    def __init__(self, mean, covar_mat):
        self.mean = mean
        self.covar_mat = covar_mat
        self.sqrt_det_covar_mat = np.linalg.norm(self.covar_mat)**0.5
        self.inv_covar_mat = np.linalg.inv(self.covar_mat)

    def probability_of(self, x):
        output = 1.0/(self.sqrt_det_covar_mat * (2.0*pi)**(0.5*self.covar_mat.shape[0]))
        x_minus_mean = x-self.mean
        output *= np.exp(-0.5*(x_minus_mean).T.dot(self.inv_covar_mat).dot(x_minus_mean))
        return output

    @classmethod
    def init_with_set(cls, X):
        mean = GaussianParams.get_mean_x(X)
        covar_mat = GaussianParams.get_covariance_matrix(X, mean = mean)
        return MultiVariateGaussian(mean, covar_mat)
