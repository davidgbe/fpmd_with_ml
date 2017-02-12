import numpy as np
from numpy.linalg import inv, norm as mag
from math import exp

class GaussianProcess:
    def __init__(self, covariance_func=None):
        if covariance_func is None:
            self.covariance_func = self.default_covariance_func
        else:
            self.covariance_func = covariance_func

    def compute_covariance(self, X_1, X_2=None):
        if X_2 is None:
            X_2 = X_1
        if X_1.shape[1] != X_2.shape[1]:
            raise ValueError('X_1 and X_2 must have the same data dimension')
        (rows_1, cols) = X_1.shape
        (rows_2, cols_2) = X_2.shape
        flattened_1 = np.array(X_1).reshape(rows_1*cols)
        flattened_2 = np.array(X_2).reshape(rows_2*cols)

        covariance_mat = np.zeros((rows_1, rows_2))
        for i in range(0, rows_1*cols, cols):
            for j in range(0, rows_2*cols, cols):
                covariance_mat[i/cols, j/cols] = self.covariance_func(flattened_1[i:i+cols], flattened_2[j:j+cols])
        return covariance_mat

    def predict(self, X, Y, target_x):
        training_cov = self.compute_covariance(X)
        training_cov_inv = inv(training_cov)
        training_target_cov = self.compute_covariance(X, target_x)
        target_cov = self.compute_covariance(target_x)
        Y = Y.reshape(Y.size, 1)

        means = training_target_cov.T.dot(training_cov_inv).dot(Y)
        stdevs = target_cov - training_target_cov.T.dot(training_cov_inv).dot(training_target_cov)

        print 'Results:'
        print means.shape
        print stdevs.shape

        num_predictions = target_x.shape[0]
        print num_predictions
        return (means.reshape(num_predictions), stdevs.reshape(num_predictions))

    def default_covariance_func(self, x_1, x_2):
        return exp(-0.5 * mag(x_1 - x_2)**2.0)
