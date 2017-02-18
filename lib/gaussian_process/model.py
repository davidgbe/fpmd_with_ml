import numpy as np
from numpy.linalg import inv, norm as mag
from math import exp
import time
from gradient_descent import gradient_descent
from kernel_methods import default_covariance_func, cartesian_operation
from functools import partial
# from multiprocessing import Pool

class GaussianProcess:
    def __init__(self, covariance_func=None):
        self.covariance_func = default_covariance_func if covariance_func is None else covariance_func
        self.hyperparams = {'theta_amp': 6009545.557726562, 'theta_length': 220.2023856989457}
        self.covariance_func = partial(self.covariance_func, hyperparams=self.hyperparams)

    def single_predict(self, target_x, training_cov_inv, Y_t, X):
        training_target_cov = cartesian_operation(X, target_x, function=self.covariance_func)
        #target_cov = self.compute_covariance(target_x)

        mean = training_target_cov.T.dot(training_cov_inv).dot(Y_t)
        #stdevs = target_cov - training_target_cov.T.dot(training_cov_inv).dot(training_target_cov)
        return mean.reshape(1)

    def batch_predict(self, X, Y, target_X):
        training_cov_inv = inv(cartesian_operation(X, function=self.covariance_func))
        Y_t = Y.reshape(Y.size, 1)

        predictions =  np.apply_along_axis(self.single_predict, 1, target_X, training_cov_inv, Y_t, X)
        (rows, cols) = predictions.shape
        return predictions.reshape(rows*cols)

    def predict(self, X, Y, target_X):
        return self.batch_predict(X, Y, target_X)

    def fit(self, X, Y):
        self.hyperparams = gradient_descent(self.hyperparams, X, Y)
