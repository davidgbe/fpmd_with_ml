import numpy as np
from lib.gaussian_process.model import GaussianProcess as GP

def func(x, y):
  print (x, y)
  print np.dot(x, y)
  return np.dot(x, y)

model = GP()

X = np.array([[1, 2], [3, 4], [5, 6]])
Y = np.array([1, 3, 5])

target = np.array([[1, 2], [1, 3], [3, 5]])

means = model.predict(X, Y, target)
print means
