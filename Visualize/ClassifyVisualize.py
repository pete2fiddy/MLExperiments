import numpy as np
from matplotlib import pyplot as plt



def plot_data(X, y):
    assert X.shape[1] == 2
    plt.scatter(X[:,0], X[:,1], c = y, cmap = plt.cm.coolwarm)

'''mesh_density is the number of points in the decision boundary calculation
over the entire plot'''
def plot_decision_bounds(X, y, model, mesh_density = 500, fill = True, alpha = .5):
    assert X.shape[1] == 2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    '''
    (w)+(h)/mesh_density = step
    '''
    mesh_step = ((x_max-x_min) + (y_max-y_min))/mesh_density
    print("mesh step: ", mesh_step)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                         np.arange(y_min, y_max, mesh_step))

    # here "model" is your model's prediction (classification) function
    Z = model.predict_set((np.c_[xx.ravel(), yy.ravel()]))

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    if fill:
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha = alpha)
    else:
        plt.contour(xx, yy, Z, cmap = plt.cm.coolwarm, alpha = alpha)
