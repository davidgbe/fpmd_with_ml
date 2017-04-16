import numpy as np
from math import exp
from numpy.linalg import norm, pinv

def produce_internal_basis(atomic_config_mat, r_cut=1.0, p=1.0):
    num_neighbors = atomic_config_mat.shape[0]
    new_basis = []
    for origin_row in range(num_neighbors):
        origin_vec = atomic_config_mat[origin_row]
        internal_basis_vec = np.zeros(3).reshape(1, 3)
        for row in range(num_neighbors):
            if row != origin_row:
                displacement_vec = (atomic_config_mat[row] - origin_vec).reshape(3)
                mag_displacement = norm(displacement_vec)
                unit = displacement_vec / mag_displacement
                internal_basis_vec += (unit * exp(-1*(mag_displacement / r_cut)**p))
        new_basis.append(internal_basis_vec)
    return np.concatenate(new_basis)

def produce_feature_matrix(basis_mat):
    mags = np.apply_along_axis(norm, 1, basis_mat)
    mags = mags.reshape(mags.shape[0], 1)
    v_norm_trans = np.divide(basis_mat, mags).T
    return basis_mat.dot(v_norm_trans).reshape(1, (basis_mat.shape[0])**2)

def transform_to_basis(real_vecs, basis_trans):
    trans = lambda v: basis_trans.dot(v)
    return np.apply_along_axis(trans, 1, real_vecs)

def compute_feature_mat_scale_factors(feature_mats):
    stdevs = []
    k = int(np.sqrt(feature_mats[0].shape[0]))
    feature_variances = np.square(feature_mats.std(0))
    for i in range(k):
        row_variance = 0
        for j in range(k):
            row_variance += feature_variances[i + k * j]
        stdevs.append(np.sqrt(row_variance))
    return np.array(stdevs)

def compute_iv_distance(x_1, x_2, stdevs):
    dist = 0
    if x_1.shape != x_2.shape:
        raise ValueError('Features matrices must have the same dimensions!')
    k = int(np.sqrt(x_2.shape[0]))
    for col in range(k):
        scale_factor = stdevs[col]
        for row in range(k):
            dist += ((x_1[k*row + col] - x_2[k*row + col]) / scale_factor)**2
    dist /= k
    return np.sqrt(dist)
