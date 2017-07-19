import numpy as np
import Classification.BinaryHelper as BinaryHelper
import Classification.SetHelper as SetHelper
from Stats.NormalDistribution.MultiVariateGaussian import MultiVariateGaussian
from Classification.Classifiable import Classifiable
'''may want to think about extending an abstract class for classifiers,
but don't know enough about the nature of my future classifiers to do so
in a non-restrictive way...'''
class GaussianDiscriminantAnalysis(Classifiable):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.init_models()

    def init_models(self):
        self.models = []
        self.num_classes = SetHelper.get_num_classes(self.y)
        for i in range(0, self.num_classes):
            i_class_subset_X, i_class_subset_y = SetHelper.get_class_subset(self.X, self.y, i)
            self.models.append(MultiVariateGaussian.init_with_set(i_class_subset_X))

    def predict(self, x):
        model_responses = np.array([self.models[i].probability_of(x) for i in range(0, len(self.models))])
        model_responses /= np.sum(model_responses)
        predict_index = np.argmax(model_responses)
        predict_probability = model_responses[predict_index]
        return predict_index, predict_probability

    def get_train_accuracy(self):
        num_correct = 0
        predictions = self.predict_set(self.X)
        for i in range(0, predictions.shape[0]):
            if predictions[i][0] == self.y[i]:
                num_correct += 1
        return num_correct/self.X.shape[0]
