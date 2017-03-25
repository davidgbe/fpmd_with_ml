import os
import struct
import numpy as np
import time
import sys
from lib.gaussian_process.model import GaussianProcess as GP
from lib.gaussian_process import utilities
from numpy.linalg import inv, norm as mag
from math import exp
import multiprocessing as mp

class MNISTTrainer:
  @staticmethod
  def gaussian_process_predict(num_training_examples=None, num_targets=None):
    train_X = MNISTTrainer.load_images_dataset('../datasets/mnist/train-images-idx3-ubyte', num_training_examples)
    zero_cols = np.array((train_X.std(0) == 0.0))
    zero_cols = zero_cols.reshape(zero_cols.shape[1])
    train_X = train_X[:, ~zero_cols]

    train_Y = MNISTTrainer.load_labels('../datasets/mnist/train-labels-idx1-ubyte', num_training_examples)
    (train_X, mean_train_X) = utilities.zero_mean(train_X)
    (train_Y, mean_train_Y) = utilities.zero_mean(train_Y)

    X = MNISTTrainer.load_images_dataset('../datasets/mnist/t10k-images-idx3-ubyte', num_targets)
    X = X[:, ~zero_cols]
    Y = MNISTTrainer.load_labels('../datasets/mnist/t10k-labels-idx1-ubyte', num_targets)
    X = X - mean_train_X

    gp = GP()

    print('Training...')
    gp.fit(train_X, train_Y)

    print('Predicting...')
    predictions = gp.predict(train_X, train_Y, X)
    print(predictions)
    predictions += mean_train_Y
    predictions = [ int(pred) for pred in predictions ]
    # print predictions
    print(predictions)
    # print(std_Y * predictions + mean_Y)
    # print(std_Y * Y + mean_Y)
    print(Y)
    print(utilities.calc_precision(predictions, Y))

  @staticmethod
  def load_labels(rel_path, limit=None):
    print('Loading labels...')
    start = time.time()

    labels_file = open(utilities.file_path(__file__, rel_path), 'rb')

    (mag, num_examples) = MNISTTrainer.read(labels_file, 8, 'i', 4)
    num_examples = limit if (limit is not None and limit < num_examples) else num_examples

    labels = MNISTTrainer.read_bytes(labels_file, num_examples)
    vec_func = np.vectorize(MNISTTrainer.convert_to_unsigned_int)

    labels = vec_func(np.array(labels))

    labels_file.close()

    end = time.time()
    print('Finished loading labels in %d s' % (end - start))
    return labels

  @staticmethod
  def load_images_dataset(rel_path, limit=None):
    print('Loading image dataset...')
    start = time.time()

    images_file = open(utilities.file_path(__file__, rel_path), 'rb')
    (mag, num_examples, rows, cols) = MNISTTrainer.read(images_file, 16, 'i', 4)
    num_examples = limit if (limit is not None and limit < num_examples) else num_examples

    print('Number of examples: %d' % num_examples)
    print('Rows of pixels per image: %d' % rows)
    print('Columns of pixels per image: %d' % cols)

    raw_images = MNISTTrainer.read_bytes(images_file, num_examples * rows * cols)
    vec_func = np.vectorize(MNISTTrainer.convert_to_unsigned_int)
    raw_images = np.mat([ vec_func(np.array(raw_images[i:i + rows * cols])) for i in range(0, len(raw_images), rows * cols) ])
    images_file.close()

    end = time.time()
    print('Images loaded in %d s' % (end - start))
    return raw_images

  @staticmethod
  def read_ints(file, size):
    return MNISTTrainer.read(file, size, 'i', 4)

  @staticmethod
  def read_bytes(file, size):
    return MNISTTrainer.read(file, size, 'c', 1)

  @staticmethod
  def read(file, size, format, format_byte_size):
    bytes_read = bytes(file.read(size))
    output_size = int(size / format_byte_size)
    return struct.unpack('>'  + format * output_size, bytes_read)

  # @staticmethod
  # def save_images(images, rows, cols, prefix='img_'):
  #   for i in range(images.shape[0]):
  #     utilities.save_image(images[i].reshape(rows, cols), prefix + str(i))

  @staticmethod
  def convert_to_unsigned_int(char):
    return 0 if char == b'' else ord(char)

if __name__ == '__main__':
  if len(sys.argv) > 1:
    num_exs = int(sys.argv[1])
    #mp.log_to_stderr().setLevel(mp.util.DEBUG)
    MNISTTrainer.gaussian_process_predict(num_training_examples=num_exs, num_targets=10)
