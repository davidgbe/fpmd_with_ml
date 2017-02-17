import numpy as np
from numpy.linalg import norm as mag
from math import exp

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
