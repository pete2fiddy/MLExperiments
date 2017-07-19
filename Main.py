from sklearn import datasets
from Classification.Linear.LogRegress.LogisticRegression import LogisticRegression
from Classification.Linear.LogRegress.MultiLogisticRegression import MultiLogisticRegression
import numpy as np
from Classification.Linear.GDA.GaussianDiscriminantAnalysis import GaussianDiscriminantAnalysis
from matplotlib import pyplot as plt

#print("Load breast cancer: ", datasets.load_breast_cancer(return_X_y = True))
X, y = datasets.load_iris(return_X_y = True)# datasets.load_breast_cancer(return_X_y = True)
print("y: ", y)
#X = X[:, :2 ]
print("X Shape: ", X.shape)
'''not sure how to handle sets whose class labels are not integers from 0 to N'''
model = MultiLogisticRegression(X,y)
model.train()
print("Train accuracy: ", model.get_train_accuracy())
