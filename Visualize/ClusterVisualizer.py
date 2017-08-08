import numpy as np
from matplotlib import pyplot as plt

def plot_clusters(X, model):
    assert X.shape[1] == 2
    model_predicts = model.cluster_set(X)
    print("model predicts: ", model_predicts)
    plt.scatter(X[:,0], X[:,1], c = model_predicts, cmap = plt.cm.coolwarm)
