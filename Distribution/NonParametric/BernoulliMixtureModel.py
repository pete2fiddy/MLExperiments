import numpy as np
from Stats.BernoulliDistribution.MultiVariateBernoulliDistribution import MultiVariateBernoulliDistribution
#from Distribution.Distribution import Distribution
import cv2

class BernoulliMixtureModel:#(Distribution):

    def __init__(self, X, num_bernoullis, smooth_amount):
        self.X = X
        self.num_bernoullis = num_bernoullis
        self.smooth_amount = smooth_amount
        self.init_params()

    def init_params(self):
        self.weights = np.random.random(self.num_bernoullis)
        self.weights /= np.sum(self.weights)
        self.bernoullis = [MultiVariateBernoulliDistribution(np.random.random(self.X.shape[1])) for i in range(0, self.num_bernoullis)]

    def train(self, max_iter):
        for iter in range(0, max_iter):
            print("---------------------")
            #print("weights: ", self.weights)
            #print("bernoullis: ", self.bernoullis)
            print("Probabililty of all X: ", np.prod(self.probability_of_set(self.X[:1])))
            self.train_step()
            '''
            map = self.calc_distribution_map()
            map = map.reshape((int(np.sqrt(map.shape)), -1))
            cv2.imshow("Distribution map: ", cv2.resize(np.uint8(255*map), (map.shape[0]*18, map.shape[1]*18)))
            rand_x = self.generate()
            rand_x = rand_x.reshape((int(np.sqrt(rand_x.shape)), -1))
            cv2.imshow("Rand_x: ", cv2.resize(np.uint8(255*rand_x), (rand_x.shape[0]*18, rand_x.shape[1]*18)))
            cv2.waitKey(1)
            '''


    def train_step(self):
        gammas = self.calc_gammas()
        N_k = np.sum(gammas, axis = 0)
        self.weights = N_k/float(self.X.shape[0])
        for k in range(0, len(self.bernoullis)):
            new_mean = self.X * gammas[:,k][:, np.newaxis]
            new_mean = np.sum(new_mean, axis = 0)
            new_mean *= (1.0/N_k[k])
            self.bernoullis[k] = MultiVariateBernoulliDistribution(new_mean)


    def calc_gammas(self):
        weighted_bern_probs = (self.calc_bernoulli_probabilities(self.X, weighted = True).T) + self.smooth_amount
        return ((weighted_bern_probs)/(np.sum(weighted_bern_probs, axis = 1)[:,np.newaxis]))

    def calc_bernoulli_probabilities(self, X, weighted = True):
        probs = []
        for k in range(0, len(self.bernoullis)):
            probs.append(self.bernoullis[k].probability_of_set(X))
            if weighted:
                probs[k] *= self.weights[k]
        return np.asarray(probs)

    def probability_of_set(self, X):
        weighted_bern_probs = self.calc_bernoulli_probabilities(X, weighted = True)
        return np.sum(weighted_bern_probs, axis = 0)

    def calc_distribution_map(self):
        map = np.zeros(self.X.shape[1])
        for k in range(0, len(self.bernoullis)):
            map += self.weights[k] * self.bernoullis[k].means
        return map

    def generate(self):
        dist_map = self.calc_distribution_map()
        rands = np.random.random(dist_map.shape[0])
        x = np.zeros(dist_map.shape[0])
        x[rands < dist_map] = 1
        return x
