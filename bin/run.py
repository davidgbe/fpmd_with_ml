import numpy as np
from lib.gaussian_process.kernel_methods import parallel_cartesian_operation

def func(x, y):
  return np.dot(x, y)

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

print X.shape

mat = parallel_cartesian_operation(X, function=func)
print mat
print mat.shape
Y = np.array([1, 3, 5])

target = np.array([[1, 2], [1, 3], [3, 5]])
