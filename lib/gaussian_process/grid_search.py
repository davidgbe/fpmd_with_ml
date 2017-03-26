import numpy as np
from numpy.linalg import inv, norm as mag
from copy import deepcopy
from .gradient_descent import calc_log_prob
from .kernel_methods import cartesian_operation, default_covariance_func
from functools import partial
from .utilities import create_pool, print_memory

def grid_search(X, Y, params, fixed_params, segs_per_order_mag=40):
    total_iterations = 1
    for p in params:
        print(p)
        print(params[p])
        total_iterations *= (3 * int(mag(params[p][0] - params[p][1])))
    print(total_iterations)
    print('Beginning grid search...')
    best_param_set = None
    largest_prob = -np.inf
    print(largest_prob)

    pool = create_pool()

    param_names = list(params.keys())
    orders_for_params = [params[param_name] for param_name in param_names]
    count = 0
    for param_set in gen_params(param_names, orders_for_params, segs_per_order_mag):
        print(param_set)
        print_memory()
        #print("%d percent complete" % (float(count) / total_iterations * 100))
        print()
        param_set.update(fixed_params)
        #print_params(param_set)
        covariance_func = partial(default_covariance_func, hyperparams=param_set)
        training_cov = cartesian_operation(X, function=covariance_func, cached_pool=pool)
        training_cov_inv = inv(training_cov)
        log_prob = calc_log_prob(X, Y, training_cov, training_cov_inv)
        print("log prob:")
        print(log_prob)

        count += 1

        if log_prob > largest_prob:
            largest_prob = log_prob
            best_param_set = param_set
    pool.close()
    pool.join()

    for p_name in best_param_set:
        best_param_set[p_name] = abs(best_param_set[p_name])

    print('best param set:')
    print_params(best_param_set)
    print('largest prob:')
    print(largest_prob)
    return best_param_set

def gen_params(param_names, orders, segs_per_order_mag):
    for param_set in iterate_for_params(orders, segs_per_order_mag):
        labeled_params = {}
        #print(param_names)
        #print(param_set)

        for i in range(len(param_names)):
            labeled_params[param_names[i]] = param_set[i]
        yield labeled_params

def iterate_for_params(orders, segs_per_order_mag, vals_for_iter=[]):
    if len(orders) == 0:
        yield vals_for_iter
    else:
        orders = deepcopy(orders)
        orders_for_param = orders.pop()
        #print(int(mag(orders_for_param[0] - orders_for_param[1])))
        for i in range(int(1 + mag(orders_for_param[0] - orders_for_param[1]))):
            curr_order = min(orders_for_param[0], orders_for_param[1]) + i
            for j in range(segs_per_order_mag):
                new_vals_for_iter = deepcopy(vals_for_iter)
                new_vals_for_iter.append(10**curr_order * (1 + 9 * j / segs_per_order_mag))
                for vals in iterate_for_params(orders, segs_per_order_mag, new_vals_for_iter):
                    yield vals

def print_params(params):
    for param_name in params:
        if param_name != 'length_scales':
            print(param_name + ':')
            print(params[param_name])

        

