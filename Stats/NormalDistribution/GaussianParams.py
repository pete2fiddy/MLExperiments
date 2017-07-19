import numpy as np

'''gives the option to pass the mean if already calculated'''
def get_covariance_matrix(X, mean = None):
    X_means = get_mean_x(X)
    covar_mat = np.zeros((X.shape[1], X.shape[1]))
    for i in range(0, covar_mat.shape[0]):
        for j in range(0, covar_mat.shape[1]):
            covar_mat[i,j] = np.dot(X[:, i] - X_means[i], (X[:, j] - X_means[j]).T)/X.shape[0]
    return covar_mat

def get_mean_x(X):
    return np.average(X, axis = 0)
