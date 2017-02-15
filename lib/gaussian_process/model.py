import numpy as np
from numpy.linalg import inv, norm as mag
from math import exp

class GaussianProcess:
    def __init__(self, covariance_func=None):
        if covariance_func is None:
            self.covariance_func = self.default_covariance_func
        else:
            self.covariance_func = covariance_func

    # runs an operation iteratively for every pair selected from X_1 and X_2
    def cartesian_operation(self, X_1, X_2=None, operation=None):
        if operation is None:
            operation = self.covariance_func
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
                transformed_mat[i/cols, j/cols] = operation(flattened_1[i:i+cols], flattened_2[j:j+cols])
        return transformed_mat

    # computes covariance matrix according to the provided covariance function
    def compute_covariance(self, X_1, X_2=None):
        return self.cartesian_operation(X_1, X_2)

    def single_predict(self, target_x, training_cov_inv, Y_t, X):
        training_target_cov = self.compute_covariance(X, target_x)
        #target_cov = self.compute_covariance(target_x)

        mean = training_target_cov.T.dot(training_cov_inv).dot(Y_t)
        #stdevs = target_cov - training_target_cov.T.dot(training_cov_inv).dot(training_target_cov)
        return mean.reshape(1)

    def batch_predict(self, X, Y, target_X):
        training_cov_inv = inv(self.compute_covariance(X))
        Y_t = Y.reshape(Y.size, 1)

        predictions =  np.apply_along_axis(self.single_predict, 1, target_X, training_cov_inv, Y_t, X)
        (rows, cols) = predictions.shape
        return predictions.reshape(rows*cols)

    def predict(self, X, Y, target_X):
        return self.batch_predict(X, Y, target_X)

    def default_covariance_func(self, x_1, x_2):
        #print mag(x_1 - x_2)
        return self.theta_amp * exp(-0.5 * (mag(x_1 - x_2) / self.theta_length)**2.0)

    # for varying the hyperparameters
    def covariance_mat_derivative_theta_length(self, x_1, x_2):
        return self.default_covariance_func(x_1, x_2) * mag(x_1 - x_2) / self.theta_length**3.0

    def covariance_mat_derivative_theta_amp(self, x_1, x_2):
        return exp(-0.5 * (mag(x_1 - x_2) / self.theta_length)**2.0)

    def gradient_log_prob(self, X, Y, training_cov_inv, gradient_func):
        gradient_cov_mat = self.cartesian_operation(X, X_2=None, operation=gradient_func)

        term_1 = np.trace(training_cov_inv.dot(gradient_cov_mat))
        term_2 = Y.T.dot(training_cov_inv).dot(gradient_cov_mat).dot(training_cov_inv).dot(Y)
        return 0.5 * (term_1 + term_2)

    def default_learning_rate(self, i):
        if i < 1000:
            return 0.1
        elif i < 2000:
            return 0.05
        else:
            return 0.01

    def gradient_descent(self, X, Y, initial_val, gradient_func):
        training_cov_inv = inv(self.compute_covariance(X))
        final_val = initial_val
        for i in range(120000):
            final_val += (self.default_learning_rate(i) * self.gradient_log_prob(X, Y, training_cov_inv, gradient_func))
            print final_val
        return final_val

    def fit(self, X, Y):
        #self.theta_amp = self.gradient_descent(X, Y, 1.0, self.covariance_mat_derivative_theta_amp)
        self.theta_amp = 1.0
        self.theta_length = 3000.0
        self.theta_length = self.gradient_descent(X, Y, self.theta_length, self.covariance_mat_derivative_theta_length)

