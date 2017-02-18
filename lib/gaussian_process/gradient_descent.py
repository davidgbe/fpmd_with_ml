import numpy as np
from numpy.linalg import inv, norm as mag
from math import exp
import time
from kernel_methods import cartesian_operation, default_covariance_func, get_gradient_funcs
from functools import partial
from copy import deepcopy

def gradient_descent(hyperparams, X, Y, learning_rate=None, epochs=4000):
    learning_rate = default_learning_rate if learning_rate is None else learning_rate

    gradients = deepcopy(hyperparams)
    params = deepcopy(hyperparams)
    covariance_func = partial(default_covariance_func, hyperparams=params)
    gradient_funcs = get_gradient_funcs(params)

    # for number of epochs
    for i in range(epochs):
        # generate inverse covariance matrix based on current hyperparameters
        training_cov_inv = inv(cartesian_operation(X, function=covariance_func))
        print params
        # for each hyperparameter
        for param_name in hyperparams:
            # compute gradient of log probability with respect to the parameter
            gradients[param_name] = gradient_log_prob(gradient_funcs[param_name], X, Y, training_cov_inv)
            print gradients[param_name]
        for param_name in hyperparams:
            # update each parameter after all the gradients have been computed
            params[param_name] += (learning_rate(i) * gradients[param_name])
            print param_name, params[param_name]
    return params

def gradient_log_prob(gradient_func, X, Y, training_cov_inv):
    print 'Computing gradient of covariance matrix'
    start = time.time()
    gradient_cov_mat = cartesian_operation(X, function=gradient_func)
    end = time.time()
    print '%d seconds' % (end - start)
    term_1 = np.trace(training_cov_inv.dot(gradient_cov_mat))
    term_2 = Y.T.dot(training_cov_inv).dot(gradient_cov_mat).dot(training_cov_inv).dot(Y)
    return 0.5 * (term_1 + term_2)

def default_learning_rate(i):
    if i < 2000:
        return 1000.0
    elif i < 3000:
        return 100.0
    elif i < 3500:
        return 50.0
    else:
        return 1.0
