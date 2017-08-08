import numpy as np
from Classification.Classifiable import Classifiable
import Classification.BinaryHelper as BinaryHelper
import cvxopt


class SupportVectorMachine(Classifiable):

    def __init__(self, X, y, pos_y_val = 1, **kwargs):

        self.X = X
        self.pos_y_val = pos_y_val
        self.init_params(**kwargs)
        self.y = BinaryHelper.modify_y_to_range(y, pos_y_val, 1.0, -1.0)



    '''possible params:
    soft_margin_weight, kernel
    '''
    def init_params(self, **kwargs):
        self.soft_margin = True if kwargs["soft_margin_weight"] is not None else False
        '''sets to -1 because this is an impossible alpha value to obtain'''
        self.soft_margin_weight = kwargs["soft_margin_weight"] if self.soft_margin else -1
        self.kernel = kwargs["kernel"]


    def train(self):
        alphas = self.get_alphas()
        support_indices = self.get_support_indices(alphas)
        self.support_alphas = alphas[support_indices]
        '''are not the actual support vectors, but are the versions before
        they are transformed into the space the kernel uses'''
        self.support_vecs = self.X[support_indices, :]
        self.support_labels = self.y[support_indices]
        self.init_bias()

    def get_alphas(self):
        '''solves for Alpha using: https://youtu.be/eHsErlPJWUU?t=2581 (at that time stamp)
        with the same notation as CVXOPT on their documentation of quadratic programming:
        http://cvxopt.org/userguide/coneprog.html'''
        P = cvxopt.matrix(self.get_coefficient_mat())
        q = cvxopt.matrix(-np.ones((self.X.shape[0], 1)))
        G_mat = 0
        h_mat = 0
        '''must initialize quadratic programming differently if the SVM
        is hard or soft margined'''
        if self.soft_margin:
            '''First self.y.shape[0] of this matrix across i ensures that
            alpha is greater than 0 (dotting alpha with G over these indices
            ensure that each alpha has a value greater than one). (since
            those indices of h are 0)

            Second self.y.shape[0] of this matrix across i inesures that alpha
            is less than soft margin weight for the same reasoning (since those
            indices of h are self.soft_margin_weight)'''
            G_mat = np.zeros((self.y.shape[0]*2, self.y.shape[0]))
            G_mat[:self.y.shape[0], :] = -np.eye(self.y.shape[0])
            G_mat[self.y.shape[0]:, :] = np.eye(self.X.shape[0])
            h_mat = np.zeros((self.X.shape[0]*2))
            h_mat[:self.y.shape[0]] = np.zeros((self.y.shape[0]))
            h_mat[self.y.shape[0]:] = np.full((self.y.shape[0]), self.soft_margin_weight)
        else:
            G_mat = -np.eye(self.y.shape[0])
            h_mat = np.zeros((self.y.shape[0]))
        G = cvxopt.matrix(G_mat)
        h = cvxopt.matrix(h_mat)
        A = cvxopt.matrix(self.y.reshape(1, -1))
        b = cvxopt.matrix(np.zeros(1))
        alphas = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])[:,0]
        return alphas

    def init_bias(self):
        '''bias is solved using only one of the support
        vectors. However, you can check to see if something
        is wrong with the SVM by solving it multiple times
        with different support vectors and see if the answer
        is the same'''
        self.bias = (1.0/self.support_labels[0]) - self.dot_with_w(self.support_vecs[0])

    def dot_with_w(self, x):
        '''returns the kernel dot product with w,
        implicitly found through a summation of support
        alphas and support vectors'''

        kernel_dot_vec = self.kernel.multi_kernel_dot(self.support_vecs, x)
        '''could precalculate support alphas * support labels, but would be more
        confusing'''
        sum_vec = self.support_alphas*self.support_labels*kernel_dot_vec
        return np.sum(sum_vec)

    def get_support_indices(self, alphas):
        '''may need to edit, because rarely were alphas exactly 0, but were often
        very very small...'''
        if self.soft_margin:
            return np.where((alphas > 0) & (alphas < self.soft_margin_weight))[0]
        return np.where((alphas > 0))[0]

    def get_coefficient_mat(self):
        X_outer_prod =  np.outer(self.y, self.y)
        for i in range(0, X_outer_prod.shape[0]):
            for j in range(0, X_outer_prod.shape[1]):
                X_outer_prod[i,j] *= self.kernel.kernel_dot(self.X[i], self.X[j])
        return X_outer_prod

    def predict(self, x, zero_to_one_range = True):
        functional_margin = self.functional_margin(x)
        if functional_margin > 0:
            return 1
        if zero_to_one_range:
            return 0
        return -1

    def functional_margin(self, x):
        return self.dot_with_w(x) + self.bias
