import numpy as np
from numpy.linalg import inv, norm as mag
from math import exp
import time
from .gradient_descent import optimize_hyperparams, initial_length_scales
from .kernel_methods import default_covariance_func, cartesian_operation
from functools import partial
from .utilities import create_pool, save_params, load_params, zero_mean, normalize
from .grid_search import grid_search
import os
from lib.internal_vector.utilities import compute_feature_mat_scale_factors, compute_iv_distance
from copy import deepcopy

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

    def single_predict(self, target_x, training_cov_inv, Y, X, cached_pool=None):
        training_target_cov = cartesian_operation(X, target_x, function=self.covariance_func, cached_pool=cached_pool)
        print('Predicting...')
        #target_cov = self.compute_covariance(target_x)
        means = training_target_cov.T.dot(training_cov_inv).dot(Y)
        #stdevs = target_cov - training_target_cov.T.dot(training_cov_inv).dot(training_target_cov)
        return means.reshape(means.size)

    def batch_predict(self, X, Y, target_X, batch_size=20):
        print('Creating training covariance')
        training_cov = cartesian_operation(X, function=self.covariance_func)
        print('Inverting covariance mat')
        training_cov_inv = inv(training_cov)
        print('Finished matrix inversion')
        predictions = []

        for i in range(0, target_X.shape[0], batch_size):
            pool = create_pool()
            batch = []
            end = i + batch_size if (i + batch_size) < target_X.shape[0] else target_X.shape[0]
            print(end)
            for j in range(i, end):
                batch.append(self.single_predict(target_X[j, :], training_cov_inv, Y, X, pool))
            pool.close()
            pool.join()
            predictions = predictions + batch

        return np.array(predictions)

    def generate_length_scales(self, X):
        self.hyperparams['length_scales'] = initial_length_scales(X)

    def predict(self, X, Y, target_X):
        # find features that don't vary in data
        zero_cols = np.array((X.std(0) == 0.0))
        zero_cols = zero_cols.reshape(zero_cols.shape[0])

        if zero_cols[zero_cols != False].shape[0] != 0:
            # strip out features that only have one value
            X = X[:, ~zero_cols]
            target_X = target_X[:, ~zero_cols]

        # preprocess training X and Y
        (X, mean_X, std_X) = normalize(X)
        (Y, mean_Y, std_Y) = normalize(Y)

        # preprocess target X with respect to X
        target_X -= mean_X
        target_X = np.divide(target_X, std_X)

        return (self.batch_predict(X, Y, target_X) * std_Y + mean_Y)

    def screened_predict(self, X, Y, target_X, threshold=2.5):
        predictions = []
        for i in range(len(target_X)):
            print("Prediction #%d" % i)
            distances = cartesian_operation(X, target_X[i, :], function=compute_iv_distance)
            print('Prediction:')
            print(distances)
            good_examples_indices = np.where(distances < threshold)[0]
            print(len(good_examples_indices))

            # select only relevant examples
            screened_X = np.take(X, good_examples_indices, axis=0)
            screened_Y = np.take(Y, good_examples_indices, axis=0)

            # normalize screen pool
            (screened_X, mean_X, std_X) = normalize(screened_X)
            (screened_Y, mean_Y, std_Y) = normalize(screened_Y)

            # compute covariances
            screened_training_target_cov = cartesian_operation(screened_X, target_X[i, :], function=self.covariance_func)
            screened_training_cov = cartesian_operation(screened_X, function=self.covariance_func)

            mean = screened_training_target_cov.T.dot(inv(screened_training_cov)).dot(screened_Y)
            mean = mean.reshape(mean.size)

            predictions.append(mean * std_Y + mean_Y)
        return np.array(predictions)

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
