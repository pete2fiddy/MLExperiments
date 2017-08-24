from Function.Homogeneity.HomogeneityFunc import HomogeneityFunc
import numpy as np

class GINIImpurity(HomogeneityFunc):

    def calc_impurity(self, sets):
        set_proportions = np.zeros((len(sets)))
        set_impurities = np.zeros((len(sets)))
        for set_index in range(0, len(sets)):
            props = self.calc_label_proportions(sets[set_index])
            set_impurities[set_index] = np.sum(props * (1-props))
            set_proportions[set_index] = sets[set_index].shape[0]
        set_proportions /= np.sum(set_proportions)
        return np.dot(set_proportions, set_impurities)
