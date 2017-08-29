import numpy as np
from Classification.Classifiable import Classifiable
import Data.Sampling.SampleHelper as SampleHelper
import Classification.SetHelper as SetHelper
import Parameter.ParamHelper as ParamHelper
from Classification.NonLinear.DecisionTree.DecisionTree import DecisionTree

class RandomForest(Classifiable):

    DEFAULT_FOREST_PARAMS = dict(num_trees = 30, tree_min_features = 2, tree_max_features = 2, bootstrap_sample = True, min_subset = 8, max_subset = 20)

    def __init__(self, X, y, tree_params = None, **forest_params):
        self.X = X
        self. y = y
        self.tree_params = tree_params if tree_params is not None else dict()
        self.num_classes = SetHelper.get_num_classes(self.y)
        self.tree_params["num_classes"] = self.num_classes
        self.forest_params = ParamHelper.filter_non_default_params(forest_params, RandomForest.DEFAULT_FOREST_PARAMS)

    def train(self):
        self.trees = []
        self.tree_features = []
        for i in range(0, self.forest_params["num_trees"]):
            subset_size = np.random.randint(self.forest_params["min_subset"], self.forest_params["max_subset"])
            sample_X, sample_y = SampleHelper.boostrap_sample(subset_size, self.X, self.y)
            sample_X, tree_features = SampleHelper.sample_random_features(np.random.randint(self.forest_params["tree_min_features"], self.forest_params["tree_max_features"]+1), sample_X)
            self.trees.append(DecisionTree(sample_X, sample_y, **self.tree_params))
            self.tree_features.append(tree_features)
            self.trees[i].train()

    def rand_sample_X(self, X, y = None):
        subset_size = np.random.randint(self.forest_params["min_subset"], self.forest_params["max_subset"])
        return SampleHelper.boostrap_sample(subset_size, X, y = y)



    def predict(self, x):
        '''to make run faster, at each point, figure out if it is even possible
        for any other category to be chosen other than the one that is currently
        most voted for. Do so by assessing if second to max class vote can possibly
        exceed the first vote in the remaining number of trees. '''
        predictions = np.zeros(self.num_classes)
        for i in range(0, len(self.trees)):
            x_subset = x[self.tree_features[i]]
            tree_predict = self.trees[i].predict(x_subset)
            predictions[tree_predict] += 1
        forest_prediction = np.argmax(predictions)
        return forest_prediction
