import numpy as np
from numpy.linalg import inv, norm as mag
from math import exp, log
import time
from .kernel_methods import cartesian_operation, default_covariance_func, get_gradient_funcs, squared_distance
from functools import partial
from copy import deepcopy
from random import random
from .utilities import create_pool, print_memory

def gradient_descent(hyperparams, X, Y, learning_rate=None, epochs=100, cached_pool=None):
    learning_rate = default_learning_rate if learning_rate is None else learning_rate

    gradients = deepcopy(hyperparams)
    params = deepcopy(hyperparams)
    log_probs = deepcopy(hyperparams)
    covariance_func = partial(default_covariance_func, hyperparams=params)
    gradient_funcs = get_gradient_funcs(params)
    log_prob = 0.0
    print_memory()

    training_cov = cartesian_operation(X, function=covariance_func, cached_pool=cached_pool)
    training_cov_inv = inv(training_cov)
    new_log_prob = calc_log_prob(X, Y, training_cov, training_cov_inv)
    print('INITIAL LOG PROB', new_log_prob)

    # for number of epochs
    for i in range(epochs):
        # generate inverse covariance matrix based on current hyperparameters
        training_cov = cartesian_operation(X, function=covariance_func, cached_pool=cached_pool)
        training_cov_inv = inv(training_cov)
        # for each hyperparameter
        for param_name in hyperparams:
            if not param_name.startswith('theta'):
                continue
            # compute gradient of log probability with respect to the parameter
            gradients[param_name] = gradient_log_prob(gradient_funcs[param_name], X, Y, training_cov_inv, cached_pool=cached_pool)
            # update each parameter according to learning rate and gradient
            scale = 1.0
            step = learning_rate(i, epochs, scale) * gradients[param_name]
            print(step, param_name)
            params[param_name] += step
        print('params:')
        print({ 'theta_amp': params['theta_amp'], 'theta_length': params['theta_length'] })
        print('gradients:')
        print({ 'theta_amp': gradients['theta_amp'], 'theta_length': gradients['theta_length'] })
        print('log_prob:')
        new_log_prob = calc_log_prob(X, Y, training_cov, training_cov_inv)
        print(new_log_prob)
        log_prob = new_log_prob
        print("Completed %d" % i)
        print_memory()
    return (params, log_prob)

def gradient_log_prob(gradient_func, X, Y, training_cov_inv, cached_pool=None):
    print('Computing gradient of covariance matrix')
    start = time.time()
    gradient_cov_mat = cartesian_operation(X, function=gradient_func, cached_pool=cached_pool)
    end = time.time()
    print('%d seconds' % (end - start))
    term_1 = np.trace(training_cov_inv.dot(gradient_cov_mat))
    term_2 = Y.T.dot(training_cov_inv).dot(gradient_cov_mat).dot(training_cov_inv).dot(Y)
    return 0.5 * (term_1 + term_2)

def calc_log_prob(X, Y, training_cov, training_cov_inv):
    term_1 = Y.T.dot(training_cov_inv).dot(Y)
    term_2 = log(mag(training_cov))
    return -0.5 * (term_1 + term_2)

def default_learning_rate(i, total, scale=0.1):
    internal_scale = 0.1
    frac = float(i) / total
    if frac < 0.2 :
        return 1.0 * scale * internal_scale
    elif frac < 0.5:
        return 0.5 * scale * internal_scale
    elif frac < 0.95:
        return 0.1 * scale * internal_scale
    else:
        return 0.05 * scale * internal_scale

def generate_random_hyperparams(params, randomize=[]):
    rand_params = deepcopy(params)
    for name in randomize:
        if name not in params:
            raise ValueError('Parameter to randomize should be in params')
        rand_params[name] = 100.0 * random()
    return rand_params

def initial_length_scales(X):
    print("Generating %d scales" % X.shape[1])
    length_scales = X.std(0)
    print_memory()
    length_scales[length_scales == 0.0] = 1.0
    length_scales = np.square(np.reciprocal(length_scales))
    return length_scales.T

def optimize_hyperparams(params, X, Y, rand_restarts=1):
    print('Optimizing hyperparams...')
    pool = create_pool()
    best_candidate = None
    for i in range(0, rand_restarts):
        new_params = generate_random_hyperparams(params)
        try: 
            candidate = gradient_descent(new_params, X, Y, cached_pool=pool)
            # if new candidates log prob is higher than best candidate's
            if best_candidate is None or candidate[1] > best_candidate[1]:
                best_candidate = candidate
        except np.linalg.linalg.LinAlgError as e:
            print('An error occurred')
            print(e)
            continue
    pool.close()
    pool.join()
    # return the best set of params found
    return best_candidate[0]
