import numpy as np
from numpy.linalg import norm as mag
from math import exp, ceil
from functools import partial
from multiprocessing import Pool

def default_covariance_func(x_1, x_2, hyperparams):
    return hyperparams['theta_amp']**2.0 * exp(-0.5 * (mag(x_1 - x_2) / hyperparams['theta_length'])**2.0)

# for varying the hyperparameters
def covariance_mat_derivative_theta_length(x_1, x_2, hyperparams):
    return default_covariance_func(x_1, x_2, hyperparams) * mag(x_1 - x_2)**2.0 / hyperparams['theta_length']**3.0

def covariance_mat_derivative_theta_amp(x_1, x_2, hyperparams):
    return 2.0 * hyperparams['theta_amp'] * exp(-0.5 * (mag(x_1 - x_2) / hyperparams['theta_length'])**2.0)

# runs an operation iteratively for every pair selected from X_1 and X_2
def cartesian_operation(X_1, X_2=None, function=None):
    function = default_covariance_func if (function is None) else function
    if X_2 is None:
        X_2 = X_1
    if len(X_1.shape) == 1:
        X_1 = X_1.reshape(1, X_1.size)
    if len(X_2.shape) == 1:
        X_2 = X_2.reshape(1, X_2.size)
    if X_1.shape[1] != X_2.shape[1]:
        raise ValueError('X_1 and X_2 must have the same data dimension')
    (rows_1, cols) = X_1.shape
    (rows_2, cols_2) = X_2.shape
    flattened_1 = np.array(X_1).reshape(rows_1*cols)
    flattened_2 = np.array(X_2).reshape(rows_2*cols)

    transformed_mat = np.zeros((rows_1, rows_2))
    for i in range(0, rows_1*cols, cols):
        for j in range(0, rows_2*cols, cols):
            transformed_mat[i/cols, j/cols] = function(flattened_1[i:i+cols], flattened_2[j:j+cols])
    return transformed_mat

def operation_on_chunk(chunk_1, chunk_2, function, func_input_size):
    transformed_mat = np.zeros((chunk_1.size/func_input_size, chunk_2.size/func_input_size))
    for i in range(0, chunk_1.size, func_input_size):
        for j in range(0, chunk_2.size, func_input_size):
            transformed_mat[i/func_input_size, j/func_input_size] = function(chunk_1[i:i+func_input_size], chunk_2[j:j+func_input_size])
    return transformed_mat

def parallel_cartesian_operation(X_1, X_2=None, function=None):
    chunk_size_1 = 2
    chunk_size_2 = 3
    # must change this to be a parametrized func
    function = default_covariance_func if (function is None) else function
    if X_2 is None:
        X_2 = X_1
    if len(X_1.shape) == 1:
        X_1 = X_1.reshape(1, X_1.size)
    if len(X_2.shape) == 1:
        X_2 = X_2.reshape(1, X_2.size)
    if X_1.shape[1] != X_2.shape[1]:
        raise ValueError('X_1 and X_2 must have the same data dimension')
    (rows_1, cols) = X_1.shape
    (rows_2, cols_2) = X_2.shape
    flattened_1 = np.array(X_1).reshape(rows_1*cols)
    flattened_2 = np.array(X_2).reshape(rows_2*cols)

    print flattened_1
    print flattened_2

    iter_size_1 = chunk_size_1 * cols
    iter_size_2 = chunk_size_2 * cols

    print 'itersizes'
    print iter_size_1
    print iter_size_2

    async_results = []

    pool = Pool(4)

    for i in range(0, rows_1, chunk_size_1):
        for j in range(0, rows_2, chunk_size_2):
            print 'i: %d' % i
            print 'j: %d' % j
            chunk_i = flattened_1[ (cols * i) : ((cols * i) + iter_size_1) ]
            chunk_j = flattened_2[ (cols * j) : ((cols * j) + iter_size_2) ]

            print chunk_i
            print chunk_j
            async_results.append(pool.apply_async(operation_on_chunk, (chunk_i, chunk_j, function, cols)))

    async_results = [ res.get(timeout=100) for res in async_results]

    chunks_num_1 = int(ceil(float(rows_1) / chunk_size_1))
    chunks_num_2 = int(ceil(float(rows_2) / chunk_size_2))

    results = [ np.concatenate(async_results[j:j+chunks_num_2], axis=1) for j in range(0, chunks_num_1*chunks_num_2, chunks_num_2) ]
    print results
    return np.concatenate(results)

def get_gradient_funcs(hyperparams):
    return {
        'theta_amp': partial(covariance_mat_derivative_theta_amp, hyperparams=hyperparams),
        'theta_length': partial(covariance_mat_derivative_theta_length, hyperparams=hyperparams)
    }
