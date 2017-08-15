import numpy as np

'''an HMM used for models with discrete observation and
state spaces'''
class CompleteDiscreteHMM:
    '''setting SMOOTH_PARAM too low seems to make the probability of the
    set hover between two values while training. Setting too high
    makes model never train parameters. Appropriate value seems to be dependent
    upon the length of training sequence'''
    SMOOTH_PARAM = float(10.0**-320.0)
    '''may want to laplace smooth somehow based on the number of instances
    of a certain hidden/observed state occurring rather than just adding
    the smooth param to the numerator and denominators of each matrix?'''
    def __init__(self, x, z, inital_probs = None):
        self.x = x
        self.z = z
        self.num_states = self.z.max()+1
        self.num_observables = self.x.max()+1
        self.init_mats()
    def init_mats(self):

        self.initial_probs = np.full((self.num_states), 1.0/self.num_states)#np.random.rand(self.num_states)##np.ones(self.num_states)
        self.initial_probs /= np.sum(self.initial_probs)
        print("sum initial probs: ", np.sum(self.initial_probs))
        self.A = np.random.rand(self.num_states, self.num_states)#*.2 + .4#makes the values closer to 1/2 and not skewed towards 0 or 1
        self.A /= np.sum(self.A, axis = 1)[:,np.newaxis]

        self.B = np.random.rand(self.num_states, self.num_observables)#*.2 + .4
        #self.B = np.ones((self.num_states, self.num_observables), dtype = np.float64)
        self.B /= np.sum(self.B, axis = 1)[:,np.newaxis]

    def train(self, max_iter):
        for iter in range(0, max_iter):
            self.train_step()
            if iter % 10 == 0:
                set_prob = self.probability_of_sequence_at_time(self.x, 8)
                rand_set_prob = self.probability_of_sequence_at_time(self.generate_random_observations(10), 8)
                best_prob_path, best_prob = self.viterbi(self.x[:9])
                print("best prob path: {}, best path prob: {}".format(best_prob_path, best_prob))#, end = '\r')
                print("self.Z: ", self.z[:9])
                #print("self.x performance: {}, rand set performance: {}".format(set_prob, rand_set_prob), end = '\r')

                #print("probability of self.x: {} ".format(self.probability_of_sequence_at_time(self.x, self.x.shape[0]-1)), end = '\r')
                #print("probability of random x: {}".format(self.probability_of_sequence_at_time(self.generate_random_observations(self.x.shape[0]), self.x.shape[0]-1)), end = '\r\r')
                #print("-------------------------------------")

    def generate_random_observations(self, length):
        return (np.random.rand(length) * self.num_states).astype(np.int)

    def train_step(self):
        alphas = self.calc_alphas(self.x)
        betas = self.calc_betas(self.x)
        gammas = self.calc_gammas(self.x, alphas, betas)
        self.A = self.step_A(gammas)
        self.B = self.step_B(gammas, self.x)
        #print("A: ", self.A)
        #print("B: ", self.B)


    def step_A(self, gammas):
        A_new = np.zeros(self.A.shape)
        for i in range(0, A_new.shape[0]):
            for j in range(0, A_new.shape[1]):
                A_new_numerator = np.sum(gammas[:,i,j])

                A_new_denominator = np.sum(np.sum(gammas[:,i,:], axis = 0))
                A_new[i,j] = (A_new_numerator + CompleteDiscreteHMM.SMOOTH_PARAM)/(A_new_denominator + CompleteDiscreteHMM.SMOOTH_PARAM*self.num_states)
                #print("A_new[i,j]: ", A_new[i,j])
        #print("A_new sums: ", np.sum(A_new, axis = 1))
        #A_new_sums = np.sum(A_new, axis = 1)
        #A_new /= A_new_sums[:,np.newaxis]
        #print("A_new sums: ", np.sum(A_new, axis = 1))
        return A_new

    def step_B(self, gammas, x):
        B_new = np.zeros(self.B.shape)
        '''should speed up using numpy operations'''
        for j in range(0, B_new.shape[0]):
            for k in range(0, B_new.shape[1]):
                numerator_sum_multipliers = x==k
                B_new_numerator = np.sum(np.sum(gammas[:,:,j] * numerator_sum_multipliers[:,np.newaxis], axis = 0))
                B_new_denominator = np.sum(np.sum(gammas[:,:,j], axis = 0))

                B_new[j,k] = (B_new_numerator + CompleteDiscreteHMM.SMOOTH_PARAM)/(B_new_denominator + CompleteDiscreteHMM.SMOOTH_PARAM*self.num_observables)
                #print("B_new[j,k]: ", B_new[j,k])
                #B_new[j,k] = B_new_numerator
            #B_new[j,:] /= (np.sum(B_new[j,:])
            #print("B_new at index: ", B_new[j,0])

        #B_new_sums = np.sum(B_new, axis = 1)
        #B_new /= B_new_sums[:,np.newaxis]
        #print("B_new sums: ", np.sum(B_new, axis = 1))
        return B_new

    def calc_gammas(self, x, alphas, betas):
        '''returns a t, A.shape[0], A.shape[1] matrix where
        the time is denoted by the first index, and the indices
        of the matrix is denoted by the last two indices'''
        '''check if A and B misaligned when calculating gammas (doubt it)'''
        gammas = np.zeros((x.shape[0],) + self.A.shape, dtype = np.float64)
        for t in range(0, gammas.shape[0]-1):
            for i in range(0, gammas.shape[1]):
                for j in range(0, gammas.shape[2]):
                    #gammas[t,i,j] = alphas[t, i] * self.A[i,j] * self.B[j,x[t]] * betas[t+1, j]
                    gammas[t,i,j] = np.exp(alphas[t,i]) * (self.A[i,j])*(self.B[j,x[t]]) * np.exp(betas[t+1,j])
        #print("gammas: ", gammas)
        return gammas

    '''both calc_alphas and step_alphas are verified to be working.
    (may not work with mismatched A and B sizes, but that can be
    fixed by just changing the initialization size of alphas)'''
    def calc_alphas(self, x):
        time_alphas = np.zeros((x.shape[0], self.num_states), dtype = np.float64)#self.step_alphas(x, 0)
        time_alphas[0] = self.step_alphas(x, 0)
        for t in range(1,x.shape[0]):
            time_alphas[t] = self.step_alphas(x, t, prev_alphas = time_alphas[t-1])
        return time_alphas

    '''calculating in alphas in log space now works'''
    def step_alphas(self, x, t, prev_alphas = None):
        if t == 0:
            initial_alphas = np.log(self.initial_probs.copy()) + np.log(self.B[:,x[t]])
            return initial_alphas
        alphas = np.zeros(self.A.shape[0])
        for j in range(0, alphas.shape[0]):
            '''prev_alphas does not need to have the log taken of it because it already
            is in log form'''
            log_sum_terms = np.log(self.B[j,x[t]]) + np.log(self.A[:,j]) + prev_alphas
            max_log_sum_term = log_sum_terms.max()
            alphas[j] = max_log_sum_term + np.log(np.sum(np.exp(log_sum_terms - max_log_sum_term)))

        #print("alphas nan?: ", np.isnan(alphas).any())
        return alphas

    '''both calc_betas and step_betas are verified to be working.
    (May index into bad dimensions, if that is the case, then just change
    the initialization size of betas)'''
    def calc_betas(self, x):
        time_betas = np.zeros((x.shape[0]+1, self.num_states))
        time_betas[time_betas.shape[0]-1] = self.step_betas(x, time_betas.shape[0]-1)
        for t in range(x.shape[0]-1, -1, -1):
            time_betas[t] = self.step_betas(x, t, prev_betas = time_betas[t+1])
        return time_betas

    '''calculating betas in log space now works'''
    def step_betas(self, x, t, prev_betas = None):
        if t == x.shape[0]:
            #return np.ones(self.num_states)
            return np.log(np.ones(self.num_states))
        elif t == 0:
            #return self.initial_probs * prev_betas * self.B[:,x[0]]

            return np.log(self.initial_probs)  + np.log(self.B[:,x[0]]) + prev_betas
        betas = np.zeros(self.num_states)
        for i in range(0, betas.shape[0]):
            log_sum_terms = np.log(self.A[i,:]) + np.log(self.B[:,x[t]]) + prev_betas
            max_log_sum_term = log_sum_terms.max()
            #betas[i] = np.sum(self.A[i,:] * self.B[:,x[t]] * prev_betas)
            betas[i] = max_log_sum_term + np.log(np.sum(np.exp(log_sum_terms - max_log_sum_term)))

        #print("betas is NaN?: ", np.isnan(betas).any())
        return betas

    def viterbi(self, x):
        viterbi_probs = np.zeros((x.shape[0]+1, self.num_states))
        #viterbi_probs[0,:] = self.initial_probs * self.B[:,x[0]]
        #print("initial viterbi probs: ", viterbi_probs[0,:])
        viterbi_paths = np.zeros((x.shape[0]+1, self.num_states), dtype = np.int)
        for t in range(0, viterbi_probs.shape[0]):
            self.step_viterbi_probs_and_paths(x, t, viterbi_probs, viterbi_paths)

        #print("paths: ", viterbi_paths)
        #print("probs: ", viterbi_probs)
        max_prob_index = np.argmax(viterbi_probs[viterbi_probs.shape[0]-1])
        max_prob_path_prob = viterbi_probs[viterbi_probs.shape[0]-1, max_prob_index]
        max_prob_path = np.zeros(x.shape[0], dtype = np.int)
        max_prob_path[max_prob_path.shape[0]-1] = viterbi_paths[viterbi_paths.shape[0]-1, max_prob_index]

        for t in range(max_prob_path.shape[0]-2, 0, -1):
            max_prob_path[t] = viterbi_paths[t+1, max_prob_path[t+1]]

        '''max_prob_path_index = np.argmax(viterbi_probs[viterbi_probs.shape[0]-1])
        max_prob_path = viterbi_paths[:, max_prob_path_index]
        max_prob_path_prob = viterbi_probs[viterbi_probs.shape[0]-1, max_prob_path_index]
        '''
        '''max_prob_path = np.zeros(x.shape[0])
        for t in range(0, max_prob_path.shape[0]):
            max_prob_path[t] = viterbi_paths[t,np.argmax(viterbi_probs[t])]
        max_prob_path_prob = 1.0'''
        return max_prob_path, max_prob_path_prob

    def step_viterbi_probs_and_paths(self, x, t, probs, paths):
        if t == 0:
            probs[0,:] = self.initial_probs * self.B[:,x[t]]
            paths[0,:] = np.arange(0, probs.shape[1])
        else:
            for j in range(0, probs.shape[1]):
                t_probs = probs[t-1] * self.B[j, x[t-1]] * self.A[paths[t-1], j]
                #print("t_probs: ", t_probs)
                '''t_probs = np.zeros(probs.shape[1])
                for i in range(0, t_probs.shape[0]):
                    t_probs[i] = probs[t-1,i] * self.B[j,x[t-1]] * self.A[paths[t-1,i],j]'''
                probs[t, j] = np.max(t_probs)
                paths[t, j] = np.argmax(t_probs)




    def probability_of_sequence_at_time(self, x, t):
        alphas = self.calc_alphas(x)
        alpha_sums = np.sum(alphas, axis = 1)
        #print("exp alphas: ", (np.exp(alphas[t])))
        return np.sum(np.exp(alphas[t]))
