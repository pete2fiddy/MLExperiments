import numpy as np
'''returns all X and y where y == class label'''
def get_class_subset(X, y, class_label):
    class_indices = np.where(y == class_label)
    return X[class_indices, :][0], y[class_indices]

def get_num_classes(y):
    return np.amax(y) + 1
