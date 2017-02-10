import os
import struct
import numpy as np
import time
import sys
from sklearn.metrics import classification_report
from lib.gaussian_process.model import GaussianProcess as GP
from lib.gaussian_process import utilities

class MNISTTrainer:
  @staticmethod
  def gaussian_process_predict(num_training_examples=None, num_targets=None):
    train_X = MNISTTrainer.load_images_dataset('../datasets/mnist/train-images-idx3-ubyte')
    train_Y = MNISTTrainer.load_labels('../datasets/mnist/train-labels-idx1-ubyte')
    if num_training_examples is not None:
      train_X = train_X[:num_training_examples]
      train_Y = train_Y[:num_training_examples]

    X = MNISTTrainer.load_images_dataset('../datasets/mnist/t10k-images-idx3-ubyte')
    Y = MNISTTrainer.load_labels('../datasets/mnist/t10k-labels-idx1-ubyte')
    if num_targets is not None:
      X = X[num_targets]
      Y = Y[num_targets]

    gp = GP()

    print 'Predicting'
    (predictions, errors) = gp.predict(train_X, train_Y, X)

    print classification_report(Y, predictions)

  @staticmethod
  def load_labels(rel_path):
    print 'Loading labels...'
    start = time.time()

    labels_file = open(utilities.file_path(__file__, rel_path), 'r+')

    (mag, num_examples) = MNISTTrainer.read(labels_file, 8, 'i', 4)
    labels = MNISTTrainer.read_bytes(labels_file, num_examples)
    vec_func = np.vectorize(MNISTTrainer.convert_to_unsigned_int)

    labels = vec_func(np.array(labels))

    end = time.time()
    print 'Finished loading labels in %d s' % (end - start)
    return labels

  @staticmethod
  def load_images_dataset(rel_path):
    print 'Loading image dataset...'
    start = time.time()

    images_file = open(utilities.file_path(__file__, rel_path), 'r+')
    (mag, num_examples, rows, cols) = MNISTTrainer.read(images_file, 16, 'i', 4)

    print 'Number of examples: %d' % num_examples
    print 'Rows of pixels per image: %d' % rows
    print 'Columns of pixels per image: %d' % cols

    raw_images = MNISTTrainer.read_bytes(images_file, num_examples * rows * cols)
    vec_func = np.vectorize(MNISTTrainer.convert_to_unsigned_int)
    raw_images = np.mat([ vec_func(np.array(raw_images[i:i + rows * cols])) for i in xrange(0, len(raw_images), rows * cols) ])

    end = time.time()
    print 'Images loaded in %d s' % (end - start)
    return raw_images

  @staticmethod
  def read_ints(file, size):
    return MNISTTrainer.read(file, size, 'i', 4)

  @staticmethod
  def read_bytes(file, size):
    return MNISTTrainer.read(file, size, 'c', 1)

  @staticmethod
  def read(file, size, format, format_byte_size):
    bytes_read = file.read(size)
    output_size = size / format_byte_size
    return struct.unpack('>'  + format * output_size, bytes_read)

  @staticmethod
  def save_images(images, rows, cols, prefix='img_'):
    for i in range(images.shape[0]):
      utilities.save_image(images[i].reshape(rows, cols), prefix + str(i))

  @staticmethod
  def convert_to_unsigned_int(char):
    return 0 if char == '' else ord(char)

if __name__ == '__main__':
  if len(sys.argv) > 1:
    command = sys.argv[1]
    if command == 'predict':
      MNISTTrainer.gaussian_process_predict(num_training_examples=1200, num_targets=10)