import numpy as np
from Stats.NormalDistribution.MultiVariateGaussian import MultiVariateGaussian
from Unsupervised.Cluster.Clusterable import Clusterable
import Matrix.MatrixMath as MatrixMath

class GaussianMixtureModel(Clusterable):
    INITIAL_COVARIANCE_FACTOR = 1.0
    '''uses an Expectation Minimization algorithm to maximize the log likelihood
    of the points given the data. Struggles with singularities in the covariance
    matrices if the data can be generalized fairly well with fewer components
    than asked of it. This can be remedied by adding a marginal probability of
    the parameters (covariance matrices, means, etc.) that give a "soft" probabilistic
    bound to the reasonable values of covariance matrices, but that has not yet
    been implemented. See page 461 (by pdf page) Bishop's Pattern Recognition
    and Machine Learning to learn about the implementation of the probabilistic
    bounds.'''
    def __init__(self, X, **params):
        self.X = X
        self.upper_X_bounds = np.amax(self.X, axis = 0)
        self.lower_X_bounds = np.amin(self.X, axis = 0)
        self.initial_covariance_matrix = np.identity(self.X.shape[1], dtype = np.float64)*GaussianMixtureModel.INITIAL_COVARIANCE_FACTOR
        self.init_params(params)
        self.init_gaussians()

    def init_params(self, params):
        self.num_gauss = params["num_clusters"]
        self.convergence_thresh = params["convergence_thresh"]
        self.max_iter = params["max_iter"]
        self.covar_mag_constraints = params["covar_mag_constraints"]
        self.min_cluster_weight = params["min_cluster_weight"]
        '''could initialize means and covariance matrices using KMeans,
        as GMM runs slower and this will allow it to converge faster'''

    def init_gaussians(self):
        self.gaussians = [self.get_random_gaussian() for i in range(0, self.num_gauss)]
        self.cluster_weights = np.full((self.num_gauss), 1.0/self.num_gauss, dtype = np.float64)

    def get_random_gaussian(self):
        mean = (np.random.rand(self.X.shape[1]) * (self.upper_X_bounds-self.lower_X_bounds) + self.lower_X_bounds).astype(np.float64)
        covar_mat = self.initial_covariance_matrix.copy()
        return MultiVariateGaussian(mean, covar_mat)

    def cluster(self, x):
        responses = np.zeros((self.num_gauss), dtype = np.float64)
        for i in range(0, len(self.gaussians)):
            responses[i] = self.cluster_weights[i]*self.gaussians[i].probability_of(x)
        return np.argmax(responses)

    def train(self):
        old_log_likelihood = 0
        for iter in range(0, self.max_iter):
            update_successful = self.update_gaussians()
            if update_successful:
                current_log_likelihood = self.calc_log_likelihood()
                convergence_score = abs(old_log_likelihood - current_log_likelihood)
                print("likelihood convergence:", convergence_score)
                if convergence_score < self.convergence_thresh:
                    break
                old_log_likelihood = current_log_likelihood


    def update_gaussians(self):
        gammas = self.calc_gammas()
        new_means = self.update_means(gammas)
        new_covariance_matrices = self.update_covariance_matrices(gammas, new_means)
        self.cluster_weights = self.update_cluster_weights(gammas)
        if self.do_constraints_break(new_covariance_matrices, self.cluster_weights):
            self.init_gaussians()
            return False
        for i in range(0, len(self.gaussians)):
            try:
                self.gaussians[i] = MultiVariateGaussian(new_means[i], new_covariance_matrices[i])
            except:
                self.init_gaussians()
                return False
        return True

    def calc_gammas(self):
        weighted_cluster_probs = self.calc_cluster_probabilities(self.X, weighted = True)
        return weighted_cluster_probs/np.sum(weighted_cluster_probs, axis = 1)[:,np.newaxis]

    def do_constraints_break(self, new_covariance_matrices, new_cluster_weights):
        diag_mags = np.zeros((new_covariance_matrices.shape[0]))
        for i in range(0, new_covariance_matrices.shape[0]):
            diag_mags[i] = np.linalg.norm(np.diag(new_covariance_matrices[i]))
        return not (np.amin(new_cluster_weights) > self.min_cluster_weight and np.amin(diag_mags) > self.covar_mag_constraints[0] and np.amax(diag_mags) < self.covar_mag_constraints[1] and not np.isnan(new_covariance_matrices).any())

    def update_means(self, gammas):
        new_means = np.array([np.average(self.X, axis = 0, weights = gammas[:,i]) for i in range(0, self.num_gauss)])
        return new_means

    def update_covariance_matrices(self, gammas, new_means):
        covar_mats = np.zeros((self.num_gauss, self.X.shape[1], self.X.shape[1]))
        N_k = np.sum(gammas, axis = 0)
        for k in range(0, covar_mats.shape[0]):
            X_minus_mean_at_k = self.X - new_means[k]
            '''don't understand the notation for einsum, but creates an array of matrices
            where the [ith] index represents the outer product between X_minus_mean_at_k[i]
            with itself'''
            outer_prods = np.einsum('ij...,i...->ij...',X_minus_mean_at_k, X_minus_mean_at_k)
            covar_mats[k] = np.sum(gammas[:,k,np.newaxis,np.newaxis] * outer_prods, axis = 0)
            covar_mats[k] /= N_k[k]
        return covar_mats

    def update_cluster_weights(self, gammas):
        N_k = np.sum(gammas, axis = 0).astype(np.float64)
        return N_k/float(self.X.shape[0])

    def calc_cluster_probabilities(self, X, weighted = True):
        cluster_responses = np.zeros((X.shape[0], len(self.gaussians)))
        for i in range(0, cluster_responses.shape[1]):
            cluster_responses[:,i] = self.gaussians[i].probability_of_set(X)
        if weighted:
            cluster_responses = cluster_responses * self.cluster_weights
        return cluster_responses

    def calc_log_likelihood(self):
        weighted_probs = self.calc_cluster_probabilities(self.X, weighted = True)
        return np.sum(np.log(np.sum(weighted_probs, axis = 1)))
