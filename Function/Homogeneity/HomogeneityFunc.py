from abc import ABC, abstractmethod
import numpy as np

class HomogeneityFunc(ABC):

    @abstractmethod
    def calc_impurity(self, labels):
        pass

    def calc_label_proportions(self, labels):
        unique_labels, label_counts = np.unique(labels, return_counts = True)
        label_proportions = (label_counts.astype(np.float))/labels.shape[0]
        return label_proportions
