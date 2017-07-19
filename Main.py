from sklearn import datasets
from Classification.Linear.LogRegress.LogisticRegression import LogisticRegression
from Classification.Linear.LogRegress.MultiLogisticRegression import MultiLogisticRegression
import numpy as np
from Classification.Linear.GDA.GaussianDiscriminantAnalysis import GaussianDiscriminantAnalysis
from matplotlib import pyplot as plt
from Regression.LinearRegression import LinearRegression
from Regression.LinearLOESS import LinearLOESS


'''
#print("Load breast cancer: ", datasets.load_breast_cancer(return_X_y = True))
X, y = datasets.load_iris(return_X_y = True)# datasets.load_breast_cancer(return_X_y = True)
print("y: ", y)
#X = X[:, :2 ]
print("X Shape: ", X.shape)

model = GaussianDiscriminantAnalysis(X,y)
#model.train()
print("Train accuracy: ", model.get_train_accuracy())
'''

poly_weights = -np.array([-1,.5, -.77, -.6, -2,-0.25, 0.6, 0.2, 0.005])
num_points = 150
X_regress = np.zeros((num_points, 1))
y_regress = np.zeros((num_points))
rand_range = (-5, 5)
noise_std_dev = 8.0
for i in range(0, X_regress.shape[0]):
    X_regress[i] = (np.random.rand(X_regress.shape[1])-0.5) * (rand_range[1] - rand_range[0])

    for poly_exp in range(0, poly_weights.shape[0]):
        y_regress[i] += poly_weights[poly_exp] * X_regress[i][0] ** poly_exp

    y_regress[i] += np.random.normal(scale = noise_std_dev)

print("X regress shape: ", X_regress.shape)
lin_loess = LinearLOESS(X_regress, y_regress, 2)

plt.scatter(X_regress[:, 0], y_regress)

regressions = lin_loess.predict_set(np.sort(X_regress[:, 0]))
print("sorted X regress: ", np.sort(X_regress[:,0]))
#plt.plot(np.sort(X_regress[:,0]), regressions)


graph_dx = 0.05
x = rand_range[0]
predict_X = np.zeros((int((rand_range[1] - rand_range[0])/graph_dx), 1))
#predict_y = np.zeros((predict_X.shape[0]))
i = 0
while i < predict_X.shape[0]:
    predict_X[i][0] = x
    i += 1
    x += graph_dx

predict_y = lin_loess.predict_set(predict_X)

plt.plot(predict_X[:, 0], predict_y)
plt.show()
