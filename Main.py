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
import scipy
import mido
from Sequence.HMM.DiscreteHMM import DiscreteHMM
from Sequence.MarkovModel.StochasticMarkovModel import StochasticMarkovModel
from Classification.NonLinear.NeuralNetwork.FeedForwardNN import FeedForwardNN
from Function.Activation.Sigmoid import Sigmoid
from Function.Cost.SquareError import SquareError
from Function.Output.Softmax import Softmax
from Function.Activation.TanH import TanH
from Function.Activation.RELU import RELU
from Classification.NonLinear.DecisionTree.DecisionTree import DecisionTree
from Classification.Ensemble.RandomForest import RandomForest
from Distribution.NonParametric.BernoulliMixtureModel import BernoulliMixtureModel
from sklearn.datasets import fetch_mldata
from PIL import Image
from Projects.DinoGame.ChromeDinoGameBot2 import ChromeDinoGameBot2
import time

from mss import mss
'''should look up the types of multi classification methods
and implement them, rather than duplicate code a lot'''

'''try using gaussian mixture to find the joint probability of an
intensity at pixel(x,y) across all pixels, could try generating images'''
'''might be doing gaussian discriminant analysis wrong?'''



'''
X_all, y_all = datasets.load_breast_cancer(return_X_y = True)
X = X_all[:, [0,1]].astype(np.float64)
#X = X/X.max()
y = y_all[:]
y_uniques = np.unique(y)

y_modified = np.zeros((y.shape[0], y_uniques.shape[0]))
for unique_index in range(0, y_uniques.shape[0]):
    set_vec = np.zeros((y_uniques.shape[0]))
    set_vec[unique_index] = 1
    y_modified[y == y_uniques[unique_index], :] = set_vec
y_modified = y_modified.astype(np.float64)

print("y_modified shape: ", y_modified.shape)
nn = FeedForwardNN(X, y_modified, RELU(pos_slope = 1.0, neg_slope = 0.1), SquareError, Softmax,(5,5,2))
print("responses: ", nn.forward(X[0]))
try:
    nn.train(1000000, batch_size = 30, learn_rate = 0.0001, bias_learn_rate = 0.0001)
except:
    print("Manually stopped")
ClassifyVisualize.plot_data(X, y)
ClassifyVisualize.plot_decision_bounds(X, y, nn)
plt.show()'''


dino_bot = ChromeDinoGameBot2((550, 315, 800, 200), 1)
time.sleep(3)
dino_bot.start()

'''print("X shape: ", X.shape)
decision_tree = DecisionTree(X, y, max_depth = 6, min_split_impurity = .01, reuse_features = True, min_samples_split = 2)
decision_tree.train()
decision_tree.predict_set(X)
'''

'''need to implement a faster predict_set method for both decision tree and random forest'''
'''forest = RandomForest(X, y, max_depth = 5, num_trees = 50)
forest.train()

ClassifyVisualize.plot_decision_bounds(X, y, forest)
ClassifyVisualize.plot_data(X, y)
plt.show()'''

mnist = fetch_mldata('MNIST original', data_home = 'C:/Users/Peter/sklearn_datasets')#datasets.load_digits()#
X = mnist.data
print("X shape: ", X.shape)
print("X: ", X.max())
y = mnist.target
#X = X[y == 0]
#X[X < 128]=0
#X[X != 0] = 1
#X = X.astype(np.float64)


mixtures = []
num_berns = 30
smooth_amount = 10**-300
n_numbers = 10
round_number = 128
for i in range(0, n_numbers):
    X_i = X[y==i]
    X_i[X_i < round_number] = 0
    X_i[X_i != 0] = 1
    X_i = X_i.astype(np.float64)
    mixtures.append(BernoulliMixtureModel(X_i, num_berns, smooth_amount))
    mixtures[i].train(50)


sub_X = X[y < n_numbers]
sub_y = y[y < n_numbers]

bern_probs = np.zeros((len(mixtures), sub_X.shape[0]))
for i in range(0, bern_probs.shape[0]):
    bern_probs[i] = mixtures[i].probability_of_set(sub_X)

y_predict = np.argmax(bern_probs, axis = 0)
y_correct = np.count_nonzero(y_predict == sub_y)
print("% correct: ", y_correct/sub_X.shape[0])



#X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
#print("X[0]: ", X[0])

#Image.fromarray(np.uint8(255*X[0].reshape(28,28))).show()
#bmm = BernoulliMixtureModel(X, 5, .01)
#bmm.train(100)






'''
midi_path = "C:/Users/Peter/Desktop/Free Time CS Projects/ML Experimenting/Data/Music/BachMIDIs/802-805/BWV804.MID"
midi = mido.MidiFile(midi_path)



Z = []
for msg in midi:
    if msg.type == 'note_on':
        Z.append(msg.note)
song_length = 100

Z = np.array(Z)[:song_length]


def make_unique(mat):
    unique_vals = np.unique(mat)
    #print("Max unique vals: ", unique_vals.max())
    #print("Min unique vals: ", unique_vals.min())
    new_mat = np.zeros(mat.shape, dtype = np.int)
    for i in range(0, unique_vals.shape[0]):
        new_mat[mat == unique_vals[i]] = i
    return new_mat

Z = make_unique(Z)
#print("Z: ", Z)
hmm = DiscreteHMM(Z, 10)
hmm.train(100)
rand_x = hmm.generate_observed_states(20)
print("rand_x: ", rand_x)


'''


'''
data_path = "C:/Users/Peter/Desktop/Free Time CS Projects/ML Experimenting/Data/mixoutALL_shifted.mat"
handwriting_X = scipy.io.loadmat(data_path)
X_old = handwriting_X["mixout"][0]
X = []
Z = []
num_angle_bins = 16
bins = np.arange(num_angle_bins)*360.0/num_angle_bins
for i in range(0, X_old.shape[0]):
    append_X_vels = X_old[i].T[:, :2]
    append_X_angles = np.rad2deg(np.arctan2(append_X_vels[:,1], append_X_vels[:,0]))%360
    discrete_X_angles = np.digitize(append_X_angles, bins)-1
    #X.append(discrete_X_angles)
    X.append(append_X_vels)
    Z.append(discrete_X_angles)

y = handwriting_X["consts"][0,0]["charlabels"][0]

learn_character_index = 1
set_indices = np.where(y == learn_character_index)[0]

X_subset = [X[set_indices[i]] for i in range(0, len(set_indices))]
Z_subset = [Z[set_indices[i]] for i in range(0, len(set_indices))]
y_subset = y[set_indices]
#print("X subset[0]: ", X_subset[0])

print("X")
hmm = HandwritingHMM(X_subset, Z_subset, 30.0, num_angles = 16)
hmm.train()

'''








'''
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
'''
X = np.array([[0,1],
              [.5,.5],
              [1,0],
              [4,2],
              [3, 3],
              [2, 4]], dtype = np.float32)
y = np.array([1, 1, 1, 0, 0, 0], dtype = np.float32)
'''
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

'''
New thing to try: Gaussian mixture model(like KMeans but uses probability of fitting
to a gaussian around the cluster)

construct a multi class coding model using baye's theorem for each set of
model responses for each correct class of point in training set

COuld run a decision tree on model outputs to predict associated class
'''
'''
indices_where_model_predicts_y = np.where(model_predicts == y)[0]
num_correct = indices_where_model_predicts_y.shape[0]
percent_correct = num_correct/y.shape[0]
print("percent correct: ", percent_correct)

ClassifyVisualize.plot_decision_bounds(X, y, svm, fill = True)
ClassifyVisualize.plot_data(X, y)

plt.show()
'''

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
