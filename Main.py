from sklearn import datasets
from Classification.Linear.LogRegress.LogisticRegression import LogisticRegression
from Classification.Linear.LogRegress.MultiLogisticRegression import MultiLogisticRegression
import numpy as np
from Classification.Linear.GDA.GaussianDiscriminantAnalysis import GaussianDiscriminantAnalysis
from matplotlib import pyplot as plt
from Regression.LinearRegression import LinearRegression
from Regression.LinearLOESS import LinearLOESS
import cv2
from PIL import Image
from Classification.Linear.Perceptron.Perceptron import Perceptron
from Classification.NonLinear.SVM.SupportVectorMachine import SupportVectorMachine
from Classification.NonLinear.SVM.MultiClassSVM import MultiClassSVM
from Classification.NonLinear.SVM.Kernel.LinearKernel import LinearKernel
from Classification.NonLinear.SVM.Kernel.RadialBias import RadialBias
import timeit
import Visualize.ClassifyVisualize as ClassifyVisualize
from Classification.MultiClass.OneVRestClassifier import OneVRestClassifier
from Stats.NormalDistribution.MultiVariateGaussian import MultiVariateGaussian
from Unsupervised.Cluster.GaussianMixtureModel import GaussianMixtureModel
import Visualize.ClusterVisualizer as ClusterVisualizer
'''should look up the types of multi classification methods
and implement them, rather than duplicate code a lot'''

'''try using gaussian mixture to find the joint probability of an
intensity at pixel(x,y) across all pixels, could try generating images'''
'''might be doing gaussian discriminant analysis wrong?'''

X_all, y_all = datasets.load_iris(return_X_y = True)
X = X_all[:, [1,2]]

mix_model = GaussianMixtureModel(X, num_clusters = 3, convergence_thresh = 0.001, max_iter = 5000, covar_mag_constraints = (.001, 10000.0), min_cluster_weight = .1)
mix_model.train()
ClusterVisualizer.plot_clusters(X, mix_model)
plt.show()
y = y_all[:]
y_test = y_all[:]
X_test = X_all[:, [1,2]]
'''
X = np.array([[0,1],
              [.5,.5],
              [1,0],
              [4,2],
              [3, 3],
              [2, 4]], dtype = np.float32)
y = np.array([1, 1, 1, 0, 0, 0], dtype = np.float32)
'''
kernel = RadialBias(100.0)## LinearKernel()
svm = OneVRestClassifier(X,y,SupportVectorMachine,SupportVectorMachine.functional_margin,soft_margin_weight = 100.0, kernel = kernel)#MultiClassSVM(X, y, soft_margin_weight = 10.0, kernel = kernel)
svm.train()
#svm = SupportVectorMachine(X, y, kernel, 1, soft_margin_weight = 10.0)
#svm.train()

model_predicts = svm.predict_set(X_test)
X_pos = X[y == 1]
X_neg = X[y == 0]

'''
New thing to try: Gaussian mixture model(like KMeans but uses probability of fitting
to a gaussian around the cluster)

construct a multi class coding model using baye's theorem for each set of
model responses for each correct class of point in training set

COuld run a decision tree on model outputs to predict associated class
'''
indices_where_model_predicts_y = np.where(model_predicts == y)[0]
num_correct = indices_where_model_predicts_y.shape[0]
percent_correct = num_correct/y.shape[0]
print("percent correct: ", percent_correct)

ClassifyVisualize.plot_decision_bounds(X, y, svm, fill = True)
ClassifyVisualize.plot_data(X, y)

plt.show()


'''
try doing fisher's discriminant analysis,
maximum margin classifier.
For perceptron, see if you can change the learn rate
so that it is proportional to 1/(num training set seen) --
weights an already possibly good plane more so that a new
example can't totally skew it

Mutli-Layered Perceptron
See if weighted perceptron possible, takes all misclassifications
and associates a weight with their vector from plane to point. Then
add weighted sum of vectors

'''


'''
NEXT UP: NAIVE BAYES

Rough concept (from the perspective of spam detection in email):

For both spam and non-spam, determine the proportion with which a word appears in spam mail and non-spam mail
Determine the proportion of total spam to total mail

For each email, only count a word appearing one time.

For each new email, update probabilities of words with baye's rule

To examine a new email and determine if spam, do the following:
Start at word 1, calculate probability of spam. Treat as the prior for the next calculation.
Using the determined proportions from before, revise your certainty of being spam by using the
proportiion of occurrences for word 2 in spam. Keep revising certainty and prior certainty
until finished with email. Output is probability of spam. Can either threshold or deem spam
if spam more likely than not.

If too many words, threshold words to only contain ones that appear at least some
threshold number of times.

Problems: When seeing email with new terms, probability of a never seen word
will be zero. Thus, when baye's thereom is calculated, the probability of either spam
or non-spam = 0. Use laplace smoothing to fix this
'''
