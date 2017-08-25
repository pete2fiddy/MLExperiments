import numpy as np
from Classification.Classifiable import Classifiable
from Function.Homogeneity.GINIImpurity import GINIImpurity
import Classification.SetHelper as SetHelper
import Parameter.ParamHelper as ParamHelper
class DecisionTree(Classifiable):
    DEFAULT_PARAMS = dict(fork_func = GINIImpurity(), max_depth = 5, min_split_impurity = 0.1, min_samples_split = 2, reuse_features = True)
    def __init__(self, X, y, **params):
        self.params = ParamHelper.filter_non_default_params(params, DecisionTree.DEFAULT_PARAMS)
        self.X = X
        self.y = y
        self.num_classes = SetHelper.get_num_classes(self.y)

    def train(self):
        first_pool = Pool(self.X, self.y, 0, np.arange(self.X.shape[1]).tolist(), self)
        self.first_node = TreeNode(first_pool, self)

    def predict(self, x):
        return self.first_node.feed_to_pool(x)

class TreeNode:

    def __init__(self, parent_pool, dec_tree):
        self.parent_pool = parent_pool
        self.parent_pool.connected_node = self
        self.dec_tree = dec_tree
        self.params = self.dec_tree.params
        self.train()
        self.init_pools()
        self.extend()

    def train(self):
        X = self.parent_pool.X
        y = self.parent_pool.y
        remaining_fork_indices = self.parent_pool.remaining_fork_indices
        '''a given index (i,j) of fork_val_responses represents the impurity of the
        ith remaining_fork_indices and the split value:
        (X[remaining_fork_indices[i+1],j] + X[remaining_fork_indices[i],j])/2'''
        fork_val_responses = np.zeros((X.shape[0], len(remaining_fork_indices)))
        print("fork val responses shape:", fork_val_responses.shape)
        for i in range(0, fork_val_responses.shape[0]):
            for j in range(0, fork_val_responses.shape[1]):
                fork_index = remaining_fork_indices[j]
                fork_val = X[i, fork_index]
                set_splits = self.split_set(X, fork_index, fork_val)
                split_labels = [y[set_splits == False], y[set_splits == True]]
                fork_val_responses[i,j] = self.params["fork_func"].calc_impurity(split_labels)
        min_impurity_index = np.where(fork_val_responses == fork_val_responses.min())
        min_impurity_index = (min_impurity_index[0][0], min_impurity_index[1][0])
        self.fork_index = remaining_fork_indices[min_impurity_index[1]]
        self.fork_val = X[min_impurity_index[0], self.fork_index]

    def init_pools(self):
        set_splits = self.split_set(self.parent_pool.X, self.fork_index, self.fork_val)
        X_splits = [self.parent_pool.X[set_splits == False, :], self.parent_pool.X[set_splits == True, :]]
        y_splits = [self.parent_pool.y[set_splits == False], self.parent_pool.y[set_splits == True]]
        pool_remaining_fork_indices = self.parent_pool.remaining_fork_indices.copy()
        if not self.params["reuse_features"]:
            pool_remaining_fork_indices.remove(self.fork_index)
        self.pools = [Pool(X_splits[0], y_splits[0], self.parent_pool.depth + 1, pool_remaining_fork_indices, self.dec_tree), Pool(X_splits[1], y_splits[1], self.parent_pool.depth + 1, pool_remaining_fork_indices, self.dec_tree)]

    '''returns an array of booleans where, if a given element
    of the array is true, it means that X at that element
    was greater than fork_val at fork_index'''
    def split_set(self, X, fork_index, fork_val):
        greater_thans = np.zeros((X.shape[0]), dtype = np.bool)
        greater_thans[X[:,fork_index] > fork_val] = True
        return greater_thans

    def extend(self):
        if self.parent_pool.depth + 1 < self.params["max_depth"]:
            extension_forks = []
            for i in range(0, len(self.pools)):
                if self.pools[i].impurity > self.params["min_split_impurity"] and self.pools[i].y.shape[0] > self.params["min_samples_split"]:
                    extension_forks.append(TreeNode(self.pools[i], self.dec_tree))

    def feed_to_pool(self, x):
        feed_pool = self.pools[0]
        if x[self.fork_index] > self.fork_val:
            feed_pool = self.pools[1]
        return feed_pool.feed_to_node(x)

class Pool:
    def __init__(self, X, y, depth, remaining_fork_indices, dec_tree):
        self.X = X
        self.y = y
        self.depth = depth
        self.remaining_fork_indices = remaining_fork_indices
        self.dec_tree = dec_tree
        self.connected_node = None

        self.init_class_proportions()
        self.init_impurity()

    def init_class_proportions(self):
        self.class_props = np.zeros(self.dec_tree.num_classes)
        for i in range(0, self.class_props.shape[0]):
            self.class_props[i] = np.count_nonzero(self.y == i)
        self.class_props /= np.sum(self.class_props)

    def init_impurity(self):
        self.impurity = self.dec_tree.params["fork_func"].calc_impurity([self.y])

    def feed_to_node(self, x):
        if self.connected_node is not None:
            return self.connected_node.feed_to_pool(x)
        predict_class = np.argmax(self.class_props)
        return predict_class
