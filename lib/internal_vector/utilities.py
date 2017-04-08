import numpy as np
from math import exp
from numpy.linalg import norm, pinv

def produce_internal_basis(atomic_config_mat, r_cut=1.0, p=1.0):
    print(atomic_config_mat)
    num_neighbors = atomic_config_mat.shape[0]
    print(num_neighbors)
    new_basis = []
    for origin_row in range(num_neighbors):
        origin_vec = atomic_config_mat[origin_row]
        internal_basis_vec = np.zeros(3).reshape(1, 3)
        for row in range(num_neighbors):
            if row != origin_row:
                print(atomic_config_mat[row])
                print(origin_vec)
                displacement_vec = np.subtract(atomic_config_mat[row], origin_vec).reshape(origin_vec.shape[1])
                mag_displacement = norm(displacement_vec)
                unit = displacement_vec / mag_displacement
                internal_basis_vec += (unit * exp(-1*(mag_displacement / r_cut)**p))
        new_basis.append(internal_basis_vec)
    return np.concatenate(new_basis)

def produce_feature_matrix(basis_mat):
    mags = np.apply_along_axis(norm, 1, basis_mat)
    mags = mags.reshape(mags.shape[0], 1)
    v_norm_trans = np.divide(basis_mat, mags).T
    return basis_mat.dot(v_norm_trans)

def transform_to_basis(real_vecs, basis_trans):
    trans = lambda v: basis_trans.dot(v)
    return np.apply_along_axis(trans, 1, real_vecs)
