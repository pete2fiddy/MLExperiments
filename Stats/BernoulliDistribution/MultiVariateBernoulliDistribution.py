import numpy as np

class MultiVariateBernoulliDistribution:

    def __init__(self, means):
        self.means = means

    def probability_of(self, x):
        return np.prod((self.means**x) * (1-self.means)**(1-x))
    
    def probability_of_set(self, X):
        individual_probs = np.zeros(X.shape)
        individual_probs[:] = self.means
        individual_probs[X == 0] = (1-individual_probs)[X == 0]
        return np.prod(individual_probs, axis = 1)

    '''
    def probability_of_set(self, X):
        set_probs = np.zeros(X.shape[0])
        for i in range(0, X.shape[0]):
            set_probs[i] = self.probability_of(X[i])
        return set_probs
    '''
    def __repr__(self):
        return "Bernoulli with mean: " + str(self.means)
