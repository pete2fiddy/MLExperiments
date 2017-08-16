import numpy as np
from Classification.Classifiable import Classifiable

class FeedForwardNN(Classifiable):
    RAND_WEIGHT_RANGE = (-.01, .01)
    def __init__(self, X, y, act_func, cost_func, out_func, nn_shape):
        self.shape = nn_shape
        self.X = X
        self.y = y

        self.act_func = act_func
        self.cost_func = cost_func
        self.out_func = out_func
        self.init_weights()

    def init_weights(self):
        self.weights = []
        self.weights.append(np.random.rand(self.X.shape[1], self.shape[0])*(FeedForwardNN.RAND_WEIGHT_RANGE[1] - FeedForwardNN.RAND_WEIGHT_RANGE[0]) + FeedForwardNN.RAND_WEIGHT_RANGE[0])
        for l in range(0, len(self)-1):
            append_weights_shape = (self.shape[l], self.shape[l+1])
            append_weights = np.random.rand(append_weights_shape[0], append_weights_shape[1])*(FeedForwardNN.RAND_WEIGHT_RANGE[1] - FeedForwardNN.RAND_WEIGHT_RANGE[0]) + FeedForwardNN.RAND_WEIGHT_RANGE[0]
            self.weights.append(append_weights)

    def __len__(self):
        return len(self.shape)

    def forward(self, x):
        '''no bias added in yet'''
        layer_responses = []
        layer_responses.append(x.copy())
        for l in range(0, len(self.weights)-1):
            layer_responses.append(self.act_func.act_func(layer_responses[len(layer_responses)-1].dot(self.weights[l])))
        layer_responses.append(self.out_func.out_func(layer_responses[len(layer_responses)-1].dot(self.weights[len(self.weights)-1])))
        return layer_responses

    def train(self, max_epochs, learn_rate = 0.00005, batch_size = 1):
        '''try randomly selecting batch sets instead of using the same ones
        over and over'''
        split_X = np.array_split(self.X, batch_size, axis = 0)
        split_y = np.array_split(self.y, batch_size, axis = 0)
        for epoch in range(0, max_epochs):

            for i in range(0, len(split_X)):
                X = split_X[i]
                y = split_y[i]
                weight_grads = self.create_empty_weight_matrices()
                for j in range(0, X.shape[0]):
                    iter_grads = self.calc_weight_gradients(X[j], y[j])
                    for k in range(0, len(weight_grads)):
                        weight_grads[k] += iter_grads[k]

                for j in range(0, len(weight_grads)):
                    self.weights[j] -= weight_grads[j] * learn_rate
            set_predicts = self.predict_set(self.X)
            correct_answers = np.argmax(self.y, axis = 1)
            correct_vec = set_predicts == correct_answers
            num_correct = np.sum(correct_vec)
            outputs = np.zeros((self.X.shape[0], self.y.shape[1]))
            for i in range(0, outputs.shape[0]):
                forwards = self.forward(self.X[i])
                outputs[i] = forwards[len(forwards)-1]

            cost = np.sum(self.cost_func.cost_func(outputs, self.y))
            print("percent correct: {}, cost: {}".format(num_correct/self.X.shape[0], cost), end = '\r')

    def create_empty_weight_matrices(self):
        weights = []
        weights.append(np.zeros((self.X.shape[1], self.shape[0])))
        for l in range(0, len(self)-1):
            append_weights_shape = (self.shape[l], self.shape[l+1])
            append_weights = np.zeros((append_weights_shape[0], append_weights_shape[1]))
            weights.append(append_weights)
        return weights

    def calc_weight_gradients(self, x, y):
        layer_responses = self.forward(x)
        weight_grads = [0 for i in range(0, len(self.weights))]
        weight_grads[len(weight_grads)-1] = self.calc_last_layer_weight_gradients(x, y, layer_responses)
        for l in range(len(weight_grads)-2, -1, -1):
            weight_grads[l] = self.backprop_step(layer_responses, weight_grads[l+1], x, y, l)
        return weight_grads

    def backprop_step(self, layer_responses, next_partials, x, y, l):
        grads = np.zeros(self.weights[l].shape)
        for m in range(0, grads.shape[0]):
            for n in range(0, grads.shape[1]):
                grads[m,n] = np.sum(self.weights[l+1][n,:] * self.act_func.d_func(layer_responses[l+1][n]) * layer_responses[l][m] * next_partials[n,:])
        return grads


    def calc_last_layer_weight_gradients(self, x, y, layer_responses):
        grads = np.zeros(self.weights[len(self.weights)-1].shape)
        output = layer_responses[len(layer_responses)-1]
        d_costs = self.cost_func.d_func(output, y)
        for i in range(0, grads.shape[0]):
            for j in range(0, grads.shape[1]):
                d_cost = d_costs[j]

                d_sum = layer_responses[len(layer_responses)-2][i]
                d_out = self.out_func.d_func(output[j])
                grads[i,j] = d_cost * d_sum * d_out
        return grads

    def predict(self, x):
        layer_responses = self.forward(x)
        output = layer_responses[len(layer_responses) - 1]
        #print("output: ", output)
        #print('output: ', output)
        return np.argmax(output)
