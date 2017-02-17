import numpy as np
from numpy.linalg import inv, norm as mag
from math import exp
import time
from kernel_methods import cartesian_operation, default_covariance_func
from functools import partial
from copy import deepcopy

def gradient_descent(param_to_tune, hyperparams, gradient_func, X, Y, learning_rate=None, epochs=4000):
    learning_rate = default_learning_rate if learning_rate is None else learning_rate

    covariance_func = partial(default_covariance_func, hyperparams=hyperparams)
    params = deepcopy(hyperparams)
    gradient_func = partial(gradient_func, hyperparams=params)


    training_cov_inv = inv(cartesian_operation(X, function=covariance_func))
    for i in range(epochs):
        print params
        gradient = gradient_log_prob(gradient_func, X, Y, training_cov_inv)
        print 'gradient'
        print gradient
        params[param_to_tune] += (learning_rate(i) * gradient)
        print params[param_to_tune]
    return params[param_to_tune]

def gradient_log_prob(gradient_func, X, Y, training_cov_inv):
    # print 'Computing gradient of covariance matrix'
    # start = time.time()
    gradient_cov_mat = cartesian_operation(X, function=gradient_func)
    # end = time.time()
    # print end - start
    term_1 = np.trace(training_cov_inv.dot(gradient_cov_mat))
    term_2 = Y.T.dot(training_cov_inv).dot(gradient_cov_mat).dot(training_cov_inv).dot(Y)
    return 0.5 * (term_1 + term_2)

def default_learning_rate(i):
    if i < 2000:
        return 10000.0
    elif i < 3000:
        return 1000.0
    elif i < 3500:
        return 500.0
    else:
        return 10.0
