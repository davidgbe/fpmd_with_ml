import numpy as np
from math import exp
from numpy.linalg import norm

def produce_internal_basis(atomic_config_mat, r_cut=1.0, p=1.0):
    num_neighbors = atomic_config_mat.shape[0]
    print(num_neighbors)
    new_basis = []
    for origin_row in range(num_neighbors):
        origin_vec = atomic_config_mat[origin_row]
        internal_basis_vec = np.zeros(3).reshape(1, 3)
        for row in range(num_neighbors):
            if row != origin_row:
                displacement_vec = (atomic_config_mat[row] - origin_vec).reshape(origin_vec.shape[1])
                mag_displacement = norm(displacement_vec)
                print(displacement_vec.shape)
                print(mag_displacement)
                unit = displacement_vec / mag_displacement
                internal_basis_vec += (unit * exp(-1*(mag_displacement / r_cut)**p))
        new_basis.append(internal_basis_vec)
    return np.concatenate(new_basis)

def produce_feature_matrix(basis_mat):
    mags = np.apply_along_axis(norm, 1, basis_mat)
    mags = mags.reshape(mags.shape[0], 1)
    print(mags)
    print(basis_mat)
    v_norm_trans = np.divide(basis_mat, mags).T
    print(v_norm_trans)
    return basis_mat.dot(v_norm_trans)

