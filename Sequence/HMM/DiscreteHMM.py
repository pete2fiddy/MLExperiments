import numpy as np

class DiscreteHMM:
    '''laplace smoothing very finicky and requires obscenely low parameter values
    or else all probabilities for any set are equal. Need to fix'''
    '''may need to change how laplace smoothing is added in when working entirely
    in log space'''
    DEFAULT_LAPLACE_SMOOTH_PARAM = float(10**-300)
    def __init__(self, x, num_states, laplace_smooth_param = None):
        self.x = x
        self.laplace_smooth_param = DiscreteHMM.DEFAULT_LAPLACE_SMOOTH_PARAM if laplace_smooth_param is None else laplace_smooth_param
        self.num_observables = self.x.max()+1
        self.num_states = num_states
        self.init_mats()

    def init_mats(self):
        self.initial_probs = np.full((self.num_states),(1.0/self.num_states))
        self.A = np.random.rand(self.num_states, self.num_states)
        self.A /= np.sum(self.A, axis = 1)[:, np.newaxis]
        self.B = np.random.rand(self.num_states, self.num_observables)
        self.B /= np.sum(self.B, axis = 1)[:, np.newaxis]

    @classmethod
    def init_with_toy_data(cls):
        '''initializes an HMM using the toy data found at:
        http://www.cs.rochester.edu/u/james/CSC248/Lec11.pdf which can be used to
        verify whether forwards and backwards algorithm is working'''
        hmm = DiscreteHMM(np.array([0,1,2,2]), 2)
        hmm.A = np.array([[.6, .4],
                          [.3, .7]])
        hmm.B = np.array([[.3, .4, .3],
                          [.4, .3, .3]])
        hmm.initial_probs =  np.array([.8, .2])
        return hmm


    '''work with log of A and B as well'''
    '''
    (Likely because of laplace smoothing): Transition matrix approaches just the
    average of the number of indices in each row as number of iterations passed
    increases
    '''
    def train(self, max_iter):
        for iter in range(0, max_iter):
            self.train_step()
            if iter % 1 == 0:
                test_observation_length = 100
                x_prob = self.probability_of_sequence(self.x[:test_observation_length])
                rand_prob = self.probability_of_sequence(self.generate_random_observed_sequence(test_observation_length))

                #print("A: ", self.A[0,:])
                #print("B: ", self.B[0,:])

    def generate_random_observed_sequence(self, length):
        return(np.random.rand(length) * self.num_states).astype(np.int)

    def train_step(self):
        alphas = self.calc_forwards(self.x, as_log = True)
        betas = self.calc_backwards(self.x, as_log = True)
        gammas = self.calc_gammas(self.x, alphas, betas, as_log = False)
        self.A = self.step_A(gammas)
        self.B = self.step_B(gammas, self.x)
        prob_of_x = self.probability_of_sequence(self.x)
        print("prob of x: ", prob_of_x)

    def step_A(self, gammas):
        A_new = np.zeros(self.A.shape, dtype = np.float64)
        '''speed up using numpy'''
        for i in range(0, A_new.shape[0]):
            for j in range(0, A_new.shape[1]):
                A_new_numerator = np.sum(gammas[:,i,j])
                A_new_denominator = np.sum(np.sum(gammas[:,i,:], axis = 0))
                A_new[i,j] = (A_new_numerator + self.laplace_smooth_param)/(A_new_denominator + self.laplace_smooth_param * self.num_states)
        return A_new

    '''something wrong with this function. Returns NaNs without laplace smoothing.
    Only updating A causes the probability of the training sequence to increase
    every iteration. Stepping B makes it stutter around and is random.'''
    def step_B(self, gammas, x):
        B_new = np.zeros(self.B.shape, dtype = np.float64)
        '''speed up using numpy'''
        for j in range(0, B_new.shape[0]):
            for k in range(0, B_new.shape[1]):
                numerator_sum_coefficients = (x == k).astype(np.int)
                B_new_numerator = np.sum(np.sum(gammas[:,:,j] * numerator_sum_coefficients[:,np.newaxis], axis = 0))
                B_new_denominator = np.sum(np.sum(gammas[:,:,j], axis = 0))
                '''not sure if should use / or - when working entirely in log space'''
                B_new[j,k] = (B_new_numerator + self.laplace_smooth_param)/(B_new_denominator + self.laplace_smooth_param * self.num_observables)
        return B_new

    def probability_of_sequence(self, x):
        alphas = self.calc_forwards(x, as_log = False)
        return np.sum(alphas[alphas.shape[0]-1])

    def calc_forwards(self, x, as_log = False):
        time_alphas = np.zeros((x.shape[0], self.num_states), dtype = np.float64)
        time_alphas[0] = np.log(self.initial_probs) + np.log(self.B[:,x[0]])
        for t in range(1, x.shape[0]):
            for j in range(0, time_alphas.shape[1]):
                log_sum_terms = np.log(self.B[j,x[t]]) + np.log(self.A[:,j]) + time_alphas[t-1]
                max_log_sum_term = log_sum_terms.max()
                time_alphas[t,j] = max_log_sum_term + np.log(np.sum(np.exp(log_sum_terms - max_log_sum_term)))
        if as_log:
            return time_alphas
        return np.exp(time_alphas)

    def calc_backwards(self, x, as_log = False):
        time_betas = np.zeros((x.shape[0]+1, self.num_states), dtype = np.float64)
        time_betas[x.shape[0]] = np.log(np.ones(self.num_states))
        for t in range(time_betas.shape[0]-2, 0, -1):
            for i in range(0, time_betas.shape[1]):
                log_sum_terms = np.log(self.A[i,:]) + np.log(self.B[:,x[t]]) + time_betas[t+1]
                max_log_sum_term = log_sum_terms.max()
                time_betas[t, i] = max_log_sum_term + np.log(np.sum(np.exp(log_sum_terms - max_log_sum_term)))
        time_betas[0] = np.log(self.initial_probs) + np.log(self.B[:,x[0]]) + time_betas[1]
        if as_log:
            return time_betas
        return np.exp(time_betas)

    def generate_hidden_states(self, length):
        z = np.zeros((length), dtype = np.int)
        z[0] = self.weighted_random(self.initial_probs)
        for t in range(1, z.shape[0]):
            z[t] = self.weighted_random(self.A[z[t-1], :])
        return z

    def generate_observed_states(self, length):
        rand_z = self.generate_hidden_states(length)
        x = np.zeros((length), dtype = np.int)
        for t in range(0, x.shape[0]):
            x[t] = self.weighted_random(self.B[rand_z[t], :])
        return x

    '''would be best if this were moved into a different class for handling
    functions involving randomness'''
    def weighted_random(self, weights):
        bins = np.cumsum(weights)
        rand_f = np.random.rand(1)
        bin_out = np.digitize(rand_f, bins)
        return bin_out


    '''calculates a variable created from emission, transition, forward, and
    backwards probabilities. Used for training'''
    '''confirmed works in complete log space'''
    def calc_gammas(self, x, alphas, betas, as_log = False):
        '''expects alphas and betas to be in log form'''
        gammas = np.zeros((alphas.shape[0],) + self.A.shape, dtype = np.float64)
        '''try to speed up with numpy operations'''
        for t in range(0, gammas.shape[0]):
            for i in range(0, gammas.shape[1]):
                for j in range(0, gammas.shape[2]):
                    gammas[t,i,j] = alphas[t,i] + np.log(self.A[i,j]) + np.log(self.B[j,x[t]]) + betas[t+1,j]
        if as_log:
            return gammas
        return np.exp(gammas)
