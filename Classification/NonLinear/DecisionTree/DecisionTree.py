import numpy as np
from Classification.Classifiable import Classifiable
from Function.Homogeneity.GINIImpurity import GINIImpurity
import Classification.SetHelper as SetHelper
class DecisionTree(Classifiable):

    def __init__(self, X, y, fork_func = GINIImpurity(), max_depth = None, min_split_impurity = 0.1, reuse_features = True):
        self.max_depth = max_depth if max_depth is not None else X.shape[1]
        self.reuse_features = reuse_features
        self.min_split_impurity = min_split_impurity
        self.X = X
        self.y = y
        self.num_classes = SetHelper.get_num_classes(self.y)

        self.fork_func = fork_func

    '''curious about what nonlinear transforms (transforming function by some function
    dependent on more than one index) would do to decision boundary of decision tree'''
    def train(self):
        first_pool = Pool(self.X, self.y, self.fork_func, np.arange(self.X.shape[1]).tolist(), self.num_classes, 0)
        self.first_fork = Fork(first_pool, self.fork_func, self.max_depth, self.min_split_impurity, self.reuse_features)


    '''
    def predict_set(self, X):
        print("X: ", X)
        (_,indices) = self.first_fork.split_set(X, return_indices = True)
        print("indices: ", indices)
        y_predict = np.zeros(X.shape[0])
        y_predict[indices[1]] = 1
        return y_predict
    '''
    def predict(self, x):
        return self.first_fork.predict(x)


class Fork:

    def __init__(self, parent_pool, fork_func, max_depth, min_split_impurity, reuse_features):
        self.parent_pool = parent_pool
        self.parent_pool.connected_fork = self
        self.fork_func = fork_func
        self.max_depth = max_depth
        self.min_split_impurity = min_split_impurity
        self.reuse_features = reuse_features
        self.train()
        self.init_pools()
        self.extend()

    def train(self):
        X = self.parent_pool.X
        y = self.parent_pool.y
        remaining_fork_indices = self.parent_pool.remaining_fork_indices
        '''a given index (i,j) of fork_val_responses represents the impurity of the
        ith remaining_fork_indices and the split value X[remaining_fork_indices[i],j]'''
        fork_val_responses = np.zeros((len(remaining_fork_indices), X.shape[1]))
        for i in range(0, fork_val_responses.shape[0]):
            fork_index = remaining_fork_indices[i]
            '''would be faster to work with sorted sublist, but this is simplest'''
            for j in range(0, fork_val_responses.shape[1]):
                fork_value = X[fork_index, j]
                (_, forked_labels) = self.split_set(X, fork_index, fork_value, y = y)
                fork_val_responses[i,j] = self.fork_func.calc_impurity(forked_labels)
        min_impurity_index = np.where(fork_val_responses == fork_val_responses.min())
        min_impurity_index = (min_impurity_index[0][0], min_impurity_index[1][0])
        self.fork_index = remaining_fork_indices[min_impurity_index[0]]
        self.fork_val = X[remaining_fork_indices[min_impurity_index[0]], min_impurity_index[1]]

    def init_pools(self):
        X_splits, y_splits = self.split_set(self.parent_pool.X, self.fork_index, self.fork_val, y = self.parent_pool.y)
        pool_remaining_fork_indices = self.parent_pool.remaining_fork_indices.copy()
        if not self.reuse_features:
            pool_remaining_fork_indices.remove(self.fork_index)
        self.pools = [Pool(X_splits[0], y_splits[0], self.fork_func, pool_remaining_fork_indices, self.parent_pool.num_classes, self.parent_pool.depth + 1), Pool(X_splits[1], y_splits[1], self.fork_func, pool_remaining_fork_indices, self.parent_pool.num_classes, self.parent_pool.depth + 1)]

    def split_set(self, X, fork_index = None, fork_value = None, y = None, return_indices = False):
        fork_index = self.fork_index if fork_index is None else fork_index
        fork_value = self.fork_val if fork_value is None else fork_value
        where_lower = np.where(X[:,fork_index] < fork_value)
        where_higher = np.where(X[:,fork_index] >= fork_value)
        X_lower = X[where_lower]
        X_higher = X[where_higher]
        if y is not None:
            y_lower = y[where_lower]
            y_higher = y[where_higher]
            if not return_indices:
                return [X_lower, X_higher], [y_lower, y_higher]
            return [X_lower, X_higher], [y_lower, y_higher], [where_lower, where_higher]
        if not return_indices:
            return [X_lower, X_higher]
        return [X_lower, X_higher], [where_lower, where_higher]

    def extend(self):
        if self.parent_pool.depth + 1 < self.max_depth:
            extension_forks = []
            for i in range(0, len(self.pools)):
                if self.pools[i].impurity > self.min_split_impurity:
                    extension_forks.append(Fork(self.pools[i], self.fork_func, self.max_depth, self.min_split_impurity, self.reuse_features))

    def predict(self, x):
        return self.feed_to_pool(x)

    def feed_to_pool(self, x):
        feed_pool = self.pools[0]
        if x[self.fork_index] > self.fork_val:
            feed_pool = self.pools[1]
        return feed_pool.feed_to_fork(x)

class Pool:

    def __init__(self, X, y, fork_func, remaining_fork_indices, num_classes, depth):
        self.X = X
        self.y = y
        self.depth = depth
        self.num_classes = num_classes
        self.init_class_props()
        self.init_impurity(fork_func)
        self.remaining_fork_indices = remaining_fork_indices
        self.connected_fork = None
        print("depth: ", self.depth)

    def init_class_props(self):
        self.class_props = np.zeros(self.num_classes)
        for i in range(0, self.class_props.shape[0]):
            self.class_props[i] = np.count_nonzero(self.y == i)
        self.class_props /= np.sum(self.class_props)
        print("class prop: ", self.class_props)

    def init_impurity(self, fork_func):
        self.impurity = fork_func.calc_impurity([self.y])

    def feed_to_fork(self, x):
        if self.connected_fork is not None:
            return self.connected_fork.feed_to_pool(x)
        predict_class = np.argmax(self.class_props)
        return predict_class
