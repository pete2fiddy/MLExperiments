import numpy as np
from math import pi, sqrt
import Stats.NormalDistribution.GaussianParams as GaussianParams

class MultiVariateGaussian:

    def __init__(self, mean, covar_mat):
        self.mean = mean
        self.covar_mat = covar_mat
        self.sqrt_det_covar_mat = np.linalg.norm(self.covar_mat)**0.5
        self.inv_covar_mat = np.linalg.inv(self.covar_mat)
        self.gauss_constant = 1.0/(self.sqrt_det_covar_mat * (2.0*pi)**(0.5*self.covar_mat.shape[0]))

    def probability_of(self, x):
        output = self.gauss_constant
        x_minus_mean = x-self.mean
        output *= np.exp(-0.5*(x_minus_mean).T.dot(self.inv_covar_mat).dot(x_minus_mean))
        return output

    def probability_of_set(self, X):
        '''would be faster if i could calculate the probabilities using numpy
        rather than looping through X and applying single probabilities.
        Would also be easier to merge this with the probability_of function'''
        X_probs = np.apply_along_axis(self.probability_of, 1, X)
        return X_probs



    @classmethod
    def init_with_set(cls, X):
        mean = GaussianParams.get_mean_x(X)
        covar_mat = GaussianParams.get_covariance_matrix(X, mean = mean)
        return MultiVariateGaussian(mean, covar_mat)

    def __repr__(self):
        return "MultiGaussian, mean: " + str(self.mean) + ", Covar mat: " + str(self.covar_mat)
