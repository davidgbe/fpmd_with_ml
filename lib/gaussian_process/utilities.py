import numpy as np
import os
import matplotlib.pyplot as plt

def normalize(mat):
  normed = mat - mat.mean(0)
  # Taken from http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
  with np.errstate(divide='ignore', invalid='ignore'):
    normed = np.true_divide(normed, mat.std(0))
    normed[normed == np.inf] = 0
    normed = np.nan_to_num(normed)
  return normed

def to_grayscale(recon, original):
  return np.multiply(recon, original.std(0)) + original.mean(0)

def file_path(curr_file, *path_elements):
  dir = os.path.dirname(curr_file)
  return os.path.join(dir, *path_elements)

def save_plot(name):
  plt.savefig(file_path(__file__, '../images/%s.png' % name))

def save_image(image_data, name):
  plt.imshow(image_data, interpolation='nearest', cmap='gray')
  save_plot(name)

def save_scatter(X, Y, name):
  plt.plot(X, Y, 'ro')
  save_plot(name)
  plt.clf()

def bucket(data, bucket_size):
  return [ np.mean(data[i:i+bucket_size]) for i in range(0, len(data), bucket_size) ]
