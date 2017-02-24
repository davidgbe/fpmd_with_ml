import numpy as np
from numpy.linalg import inv, norm as mag
from math import exp, log
import time
from kernel_methods import cartesian_operation, default_covariance_func, get_gradient_funcs, squared_distance
from functools import partial
from copy import deepcopy
from random import random
import utilities

def gradient_descent(hyperparams, X, Y, learning_rate=None, epochs=200):
    learning_rate = default_learning_rate if learning_rate is None else learning_rate

    gradients = deepcopy(hyperparams)
    params = deepcopy(hyperparams)
    log_probs = deepcopy(hyperparams)
    covariance_func = partial(default_covariance_func, hyperparams=params)
    gradient_funcs = get_gradient_funcs(params)
    log_prob = 1.0

    # for number of epochs
    for i in range(epochs):
        # generate inverse covariance matrix based on current hyperparameters
        training_cov = cartesian_operation(X, function=covariance_func)
        training_cov_inv = inv(training_cov)
        # for each hyperparameter
        for param_name in hyperparams:
            if not param_name.startswith('theta'):
                continue
            # compute gradient of log probability with respect to the parameter
            gradients[param_name] = gradient_log_prob(gradient_funcs[param_name], X, Y, training_cov_inv)
            scale = 10.0 if param_name is not 'theta_amp' else 1000.0
            print (learning_rate(i, scale) * gradients[param_name]), param_name
            # update each parameter according to learning rate and gradient
            params[param_name] += (learning_rate(i, scale) * gradients[param_name])
        print 'params:'
        print { 'theta_amp': params['theta_amp'], 'theta_length': params['theta_length'] }
        print 'gradients:'
        print { 'theta_amp': gradients['theta_amp'], 'theta_length': gradients['theta_length'] }
        print 'log_prob:'
        new_log_prob = calc_log_prob(X, Y, training_cov_inv, covariance_func)
        print new_log_prob
        if abs(log_prob - new_log_prob) < 0.0001:
            return (params, new_log_prob)
        else:
            log_prob = new_log_prob
    return (params, log_prob)

def gradient_log_prob(gradient_func, X, Y, training_cov_inv):
    print 'Computing gradient of covariance matrix'
    start = time.time()
    gradient_cov_mat = cartesian_operation(X, function=gradient_func)
    end = time.time()
    print '%d seconds' % (end - start)
    term_1 = np.trace(training_cov_inv.dot(gradient_cov_mat))
    term_2 = Y.T.dot(training_cov_inv).dot(gradient_cov_mat).dot(training_cov_inv).dot(Y)
    return 0.5 * (term_1 + term_2)

def calc_log_prob(X, Y, training_cov_inv, covariance_func):
    term_1 = Y.T.dot(training_cov_inv).dot(Y)
    term_2 = log(mag(cartesian_operation(X, function=covariance_func)))
    return -0.5 * (term_1 + term_2)

def default_learning_rate(i, scale=10.0):
    if i < 100:
        return 1.0 * scale
    elif i < 140:
        return 0.5 * scale
    elif i < 170:
        return 0.01 * scale
    else:
        return 0.001 * scale

def generate_random_hyperparams(params, randomize=[]):
    rand_params = deepcopy(params)
    for name in randomize:
        if name not in params:
            raise ValueError('Parameter to randomize should be in params')
        rand_params[name] = 10000.0 * random()
    return rand_params

def initial_length_scales(X):
    X_t = X.T
    length_scales = np.ones(X_t.shape[0])
    for i in range(0, X_t.shape[0]):
        length_scales[i] = cartesian_operation(X_t[i].T, function=squared_distance).std()
    length_scales = np.sqrt(np.reciprocal(length_scales))
    length_scales[length_scales == np.inf] = 1.0
    return length_scales

def optimize_hyperparams(params, X, Y, rand_restarts=30):
    best_candidate = None
    for i in range(0, rand_restarts):
        new_params = generate_random_hyperparams(params)
        try: 
            candidate = gradient_descent(new_params, X, Y)
            # if new candidates log prob is higher than best candidate's
            if best_candidate is None or candidate[1] > best_candidate[1]:
                best_candidate = candidate
        except np.linalg.linalg.LinAlgError as e:
            print 'An error occurred'
            continue
    # return the best set of params found
    return best_candidate[0]


