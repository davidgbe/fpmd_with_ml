import numpy as np
from numpy.linalg import norm as mag
from math import exp, ceil, sqrt
from functools import partial
from .utilities import create_pool
import multiprocessing as mp
from lib.internal_vector.utilities import compute_iv_distance
from timeit import timeit
from lib.parallel.utilities import parallel

def covariance_exp_arg(x_1, x_2, hyperparams):
    return compute_iv_distance(x_1, x_2)

def default_covariance_func(x_1, x_2, hyperparams):
    a = hyperparams['theta_amp']**2.0
    l = hyperparams['theta_length']
    return a * exp(-0.5 * covariance_exp_arg(x_1, x_2, hyperparams) / l**2.0)

# UNUSED
# for varying the hyperparameters
def covariance_mat_derivative_theta_length(x_1, x_2, hyperparams):
    l = hyperparams['theta_length']
    return default_covariance_func(x_1, x_2, hyperparams) * covariance_exp_arg(x_1, x_2, hyperparams) / l**3.0

# UNUSED
def covariance_mat_derivative_theta_amp(x_1, x_2, hyperparams):
    l = hyperparams['theta_length']
    return 2.0 * hyperparams['theta_amp'] * exp(-0.5 * covariance_exp_arg(x_1, x_2, hyperparams) / l**2.0)

# applies function pairwise for two arrays
def operation_on_chunk(chunks, function, func_input_size):
    (chunk_1, chunk_2) = chunks
    transformed_mat = np.zeros((int(chunk_1.size/func_input_size), int(chunk_2.size/func_input_size)))
    total = chunk_1.size * chunk_2.size
    for i in range(0, chunk_1.size, func_input_size):
        for j in range(0, chunk_2.size, func_input_size):
            transformed_mat[int(i/func_input_size), int(j/func_input_size)] = function(chunk_1[i:i+func_input_size], chunk_2[j:j+func_input_size])
    return transformed_mat

# runs an operation iteratively for every pair selected from X_1 and X_2
# distributes work across cores provided
def cartesian_operation(X_1, X_2=None, function=None, cores=None, cached_pool=None, max_chunk_size=300, max_concurrent=None):
    cores = mp.cpu_count() - 1 if cores is None else cores
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

    chunk_size_1 = int(sqrt(rows_1 * rows_2 / cores))
    chunk_size_1 = chunk_size_1 if chunk_size_1 <= max_chunk_size else max_chunk_size
    chunk_size_1 = 1 if chunk_size_1 == 0 else chunk_size_1
    # these chunk sizes are always the same
    chunk_size_2 = chunk_size_1

    iter_size_1 = chunk_size_1 * cols
    iter_size_2 = chunk_size_2 * cols

    all_chunks = []
    for i in range(0, rows_1, chunk_size_1):
        for j in range(0, rows_2, chunk_size_2):
            chunk_i = flattened_1[ (cols * i) : ((cols * i) + iter_size_1) ]
            chunk_j = flattened_2[ (cols * j) : ((cols * j) + iter_size_2) ]
            all_chunks.append((chunk_i, chunk_j))

    operation = partial(operation_on_chunk, function=function, func_input_size=cols)

    async_results = []
    if cached_pool is not None:
        async_results = cached_pool.map_async(operation, all_chunks).get()
    else:
        max_concurrent = cores if max_concurrent is None else max_concurrent
        async_results = parallel(operation, all_chunks, max_concurrent, batch_size=1)
        async_results = [result for result_block in async_results for result in result_block]

    chunks_num_1 = int(ceil(float(rows_1) / chunk_size_1))
    chunks_num_2 = int(ceil(float(rows_2) / chunk_size_2))

    async_results = [ np.concatenate(async_results[j:j+chunks_num_2], axis=1) for j in range(0, chunks_num_1*chunks_num_2, chunks_num_2) ]
    return np.concatenate(async_results)

# UNUSED
def get_gradient_funcs(hyperparams):
    return {
        'theta_amp': partial(covariance_mat_derivative_theta_amp, hyperparams=hyperparams),
        'theta_length': partial(covariance_mat_derivative_theta_length, hyperparams=hyperparams)
    }
