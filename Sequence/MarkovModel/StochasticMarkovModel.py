import numpy as np
from mido import Message, MidiFile, MidiTrack

class StochasticMarkovModel:
    LAPLACE_SMOOTH_PARAM = 1.0
    def __init__(self, x):
        self.x = x
        self.num_states = self.x.max()+1

    def train(self):
        self.A = np.zeros((self.num_states, self.num_states))
        for t in range(0, self.x.shape[0]-1):
            transition_states = (self.x[t], self.x[t+1])
            self.A[transition_states[0], transition_states[1]] += 1
        self.A += StochasticMarkovModel.LAPLACE_SMOOTH_PARAM
        self.A /= np.sum(self.A, axis = 1)[:,np.newaxis] + self.A.shape[0]*StochasticMarkovModel.LAPLACE_SMOOTH_PARAM

    def probability_of_sequence(self, x):
        transition_probs = np.zeros(x.shape[0]-1)
        for t in range(0,x.shape[0]-1):
            transition_probs[t] = self.A[x[t], x[t+1]]
        return np.prod(transition_probs)

    def predict(self, x, window):
        predicts = np.zeros((window), dtype = np.int)
        for t in range(0, predicts.shape[0]):
            if t == 0:
                probs = self.A[x[x.shape[0]-1], :]
                predicts[t] = np.argmax(probs)
            else:
                probs = self.A[predicts[t-1], :]
                predicts[t] = np.argmax(probs)
        return predicts

    def generate_random_x(self, length):
        return (np.random.rand(length) * self.num_states).astype(np.int)
