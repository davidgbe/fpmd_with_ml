import numpy as np
from lib.gaussian_process.kernel_methods import cartesian_operation
import multiprocessing
from lib.internal_vector import utilities

a = np.mat([[1, 2, 3], [2, 4, 5], [6, 7, 3], [2, 4, 8]])
v = utilities.produce_internal_basis(a)
print(utilities.produce_feature_matrix(v))

new_vec = np.array([[1, 2, 4], [1, 2, 6]])
print(utilities.transform_to_basis(new_vec, v))

