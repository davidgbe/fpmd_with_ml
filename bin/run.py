import numpy as np
from lib.gaussian_process.model import GaussianProcess as GP

def func(x, y):
  print (x, y)
  print np.dot(x, y)
  return np.dot(x, y)

model = GP(func)

X = np.array([[1, 2], [3, 4]])
Y = np.array([1, 3])

target = np.array([[1, 2]])

(means, stdevs) = model.predict(X, Y, target)
print means
print stdevs