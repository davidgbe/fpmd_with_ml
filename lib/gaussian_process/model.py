import numpy as np
from numpy.linalg import pinv, norm as mag
from math import exp
import time
from .gradient_descent import optimize_hyperparams, initial_length_scales
from .kernel_methods import default_covariance_func, cartesian_operation
from functools import partial
from .utilities import create_pool, save_params, load_params
from .grid_search import grid_search
import os

class GaussianProcess:
    def __init__(self, covariance_func=None, use_saved_params=False):
        self.covariance_func = default_covariance_func if covariance_func is None else covariance_func
        self.hyperparams = {'theta_amp': 1.0, 'theta_length': 1.0}
        self.learning_rates = {'theta_amp': 0.001, 'theta_length': 0.0005}
        self.cache_path = os.path.join(os.getcwd(), 'params')
        self.covariance_func = partial(self.covariance_func, hyperparams=self.hyperparams)
        self.use_saved_params = use_saved_params
        self.save_params = partial(save_params, rel_path=self.cache_path)
        self.load_params = partial(load_params, rel_path=self.cache_path)

    def single_predict(self, target_x, training_cov_inv, Y_t, X, cached_pool=None):
        training_target_cov = cartesian_operation(X, target_x, function=self.covariance_func, cached_pool=cached_pool)
        #target_cov = self.compute_covariance(target_x)
        mean = training_target_cov.T.dot(training_cov_inv).dot(Y_t)
        #stdevs = target_cov - training_target_cov.T.dot(training_cov_inv).dot(training_target_cov)
        return mean.reshape(1)

    def batch_predict(self, X, Y, target_X, batch_size=20):
        training_cov = cartesian_operation(X, function=self.covariance_func)
        training_cov_inv = pinv(training_cov)
        Y_t = Y.reshape(Y.size, 1)
        predictions = []

        for i in range(0, target_X.shape[0], batch_size):
            pool = create_pool()
            batch = []
            end = i + batch_size if (i + batch_size) < target_X.shape[0] else target_X.shape[0]
            print(end)
            for j in range(i, end):
                batch.append(self.single_predict(target_X[j], training_cov_inv, Y_t, X, pool))
            pool.close()
            pool.join()
            predictions = predictions + batch

        return np.array(predictions)

    def generate_length_scales(self, X):
        self.hyperparams['length_scales'] = initial_length_scales(X)

    def predict(self, X, Y, target_X):
        if 'length_scales' not in self.hyperparams:
            self.generate_length_scales(X)
        #self.hyperparams['length_scales'] = initial_length_scales(X[:20])
        return self.batch_predict(X, Y, target_X)

    def fit(self, X, Y):
        print('Generating length scales...')
        self.generate_length_scales(X)
        if not self.use_saved_params:
            fixed_params = { 'length_scales': self.hyperparams['length_scales'], 'theta_length': 1.0 }
            self.hyperparams = grid_search(X, Y, { 'theta_amp': [0, 1] }, fixed_params)
            self.save_params('hyperparams', self.hyperparams)
        else:
            self.hyperparams = self.load_params('hyperparams')
            self.generate_length_scales(X)
        print('Finished generating length scales')
        self.covariance_func = partial(self.covariance_func, hyperparams=self.hyperparams)
        self.hyperparams = optimize_hyperparams(self.hyperparams, X, Y, self.learning_rates)
