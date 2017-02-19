import numpy as np
from numpy.linalg import inv, norm as mag
from math import exp, log
import time
from kernel_methods import cartesian_operation, default_covariance_func, get_gradient_funcs
from functools import partial
from copy import deepcopy
from random import random

def gradient_descent(hyperparams, X, Y, learning_rate=None, epochs=4000):
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
        print training_cov
        training_cov_inv = inv(training_cov)
        # for each hyperparameter
        for param_name in hyperparams:
            # compute gradient of log probability with respect to the parameter
            gradients[param_name] = gradient_log_prob(gradient_funcs[param_name], X, Y, training_cov_inv)
        for param_name in hyperparams:
            # update each parameter after all the gradients have been computed
            params[param_name] += (learning_rate(i) * gradients[param_name])
        print 'params:'
        print params
        print 'gradients:'
        print gradients
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

def default_learning_rate(i):
    if i < 2000:
        return 1000.0
    elif i < 3000:
        return 100.0
    elif i < 3500:
        return 50.0
    else:
        return 1.0

def generate_random_hyperparams(params):
    rand_params = deepcopy(params)
    for name in params:
        rand_params[name] = random() * exp(10.0*random())
    return rand_params

def optimize_hyperparams(params, X, Y, rand_restarts=10):
    best_candidate = None
    for i in range(0, rand_restarts):
        new_params = generate_random_hyperparams(params)
        print new_params
        candidate = gradient_descent(new_params, X, Y)
        # if new candidates log prob is higher than best candidate's
        print candidate
        if best_candidate is None or candidate[1] > best_candidate[1]:
            best_candidate = candidate
    # return the best set of params found
    return best_candidate[1]


