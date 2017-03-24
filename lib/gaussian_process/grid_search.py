import numpy as np
from numpy.linalg import inv, norm as mag
from copy import deepcopy
from .gradient_descent import calc_log_prob
from .kernel_methods import cartesian_operation, default_covariance_func
from functools import partial
from .utilities import create_pool

def grid_search(X, Y, params, fixed_params):
    best_param_set = None
    largest_prob = -1 * np.inf
    print(largest_prob)

    pool = create_pool()

    param_names = list(params.keys())
    orders_for_params = list(params.values())
    for param_set in gen_params(param_names, orders_for_params):
        param_set.update(fixed_params)
        #print_params(param_set)
        covariance_func = partial(default_covariance_func, hyperparams=param_set)
        training_cov = cartesian_operation(X, function=covariance_func, cached_pool=pool)
        training_cov_inv = inv(training_cov)
        log_prob = calc_log_prob(X, Y, training_cov, training_cov_inv)
        #print("log prob: %d" % log_prob)

        if log_prob > largest_prob:
            largest_prob = log_prob
            best_param_set = param_set
    pool.close()

    print('best param set:')
    print_params(best_param_set)
    print('largest prob:')
    print(largest_prob)
    return best_param_set

def gen_params(param_names, orders):
    for param_set in iterate_for_params(orders):
        labeled_params = {}
        #print(param_names)
        #print(param_set)

        for i in range(len(param_names)):
            labeled_params[param_names[i]] = param_set[i]
        yield labeled_params

def iterate_for_params(orders, vals_for_iter=[]):
    print(vals_for_iter)
    if len(orders) == 0:
        yield vals_for_iter
    else:
        orders = deepcopy(orders)
        orders_for_param = orders.pop()
        #print(int(mag(orders_for_param[0] - orders_for_param[1])))
        for i in range(int(1 + mag(orders_for_param[0] - orders_for_param[1]))):
            curr_order = min(orders_for_param[0], orders_for_param[1]) + i
            new_vals_for_iter = deepcopy(vals_for_iter)
            new_vals_for_iter.append(10**curr_order)
            for vals in iterate_for_params(orders, new_vals_for_iter):
                yield vals

def print_params(params):
    for param_name in params:
        if param_name != 'length_scales':
            print(param_name + ": %d" % params[param_name])

        

