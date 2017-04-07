import numpy as np
import os
import multiprocessing as mp
import resource
import psutil
import pickle
# import matplotlib.pyplot as plt

def create_pool(cores=None):
    cores = mp.cpu_count() - 1 if cores is None else cores
    # cores = cores - 10 if cores >= 16 else cores
    print('creating %i threads' % cores)
    return mp.get_context('spawn').Pool(cores)

def normalize(mat):
    mean = mat.mean(0)
    normed = mat - mean
    std = normed.std(0)
    # Taken from http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        normed = np.true_divide(normed, std)
        normed[normed == np.inf] = 0
        normed = np.nan_to_num(normed)
    return (normed, mean, std)

def zero_mean(mat):
    mean = mat.mean(0)
    normed = mat - mean
    return (normed, mean)

def to_grayscale(recon, original):
    return np.multiply(recon, original.std(0)) + original.mean(0)

def file_path(curr_file, *path_elements):
    dir = os.path.dirname(curr_file)
    return os.path.join(dir, *path_elements)

# def save_plot(name):
#     plt.savefig(file_path(__file__, '../../images/%s.png' % name))

# def save_image(image_data, name):
#     plt.imshow(image_data, interpolation='nearest', cmap='gray')
#     save_plot(name)

# def save_scatter(name, Y, X=None):
#     if X is None:
#         X = [i for i in range(len(Y))]
#     plt.plot(X, Y, 'ro')
#     save_plot(name)
#     plt.clf()

def bucket(data, bucket_size):
    return [ np.mean(data[i:i+bucket_size]) for i in range(0, len(data), bucket_size) ]

# http://pythonforbiologists.com/index.php/measuring-memory-usage-in-python/
def get_memory():
    vals = psutil.virtual_memory()
    percent_use = vals.available / vals.total
    return percent_use

def print_memory():
    print('Percent memory usage:')
    print(1.0 - get_memory())


def calc_precision(predictions, actual):
    if len(predictions) != len(actual):
        raise ValueError('Predictions and results are different lengths')
    else:
        num_correct = 0
        for i in range(len(predictions)):
            if predictions[i] == actual[i]:
                num_correct += 1
        return num_correct / len(predictions)

def save_params(name, obj, rel_path):
    whole_path = os.path.join(rel_path, '%s.p' % name)
    pickle.dump(obj, open(whole_path, 'wb'))

def load_params(name, rel_path):
    whole_path = os.path.join(rel_path, '%s.p' % name)
    return pickle.load(open(whole_path, 'rb'))
