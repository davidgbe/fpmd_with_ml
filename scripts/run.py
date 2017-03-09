import numpy as np
from lib.gaussian_process.kernel_methods import cartesian_operation
import multiprocessing

def func(x, y):
  return np.dot(x, y)

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

print(X.shape)
print(multiprocessing.cpu_count())

mat = cartesian_operation(X, function=func)
print(mat)
print(mat.shape)
Y = np.array([1, 3, 5])

target = np.array([[1, 2], [1, 3], [3, 5]])