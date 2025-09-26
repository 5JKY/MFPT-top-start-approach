import numpy as np

def mfpt_matrix(transMat_instance):
    ria_trans = transMat_instance
    ria_P = ria_trans.trans_mat.T
    idx_fixed_vect = np.where(np.round(ria_trans.eig6_w.real, decimals=10) == 1)[0][0]
    vec_pi = ria_trans.eig6_v[:,idx_fixed_vect].real/np.sum(ria_trans.eig6_v[:,idx_fixed_vect].real)
    N = vec_pi.size
    epsilon = np.ones(N)
    mat_I = np.eye(N)
    mat_E = np.ones((N, N))
    mat_Z = np.linalg.inv(mat_I-ria_P+np.outer(epsilon, vec_pi))
    vec_Zdg = np.diag(mat_Z)
    mat_Zdg = np.diag(vec_Zdg)
    mat_Mdg = np.diag(1/vec_pi)
    mat_M = (mat_I - mat_Z + mat_E @ mat_Zdg) @ mat_Mdg
    mat_Mbar = mat_M - mat_Mdg
    return mat_Mbar

def mfpt_matrix_stable_ira(transMat_instance):
    ria_trans = transMat_instance
    ria_P = ria_trans.trans_mat.T
    idx_fixed_vect = np.where(np.round(ria_trans.eig6_w.real, decimals=10) == 1)[0][0]
    vec_pi = ria_trans.eig6_v[:,idx_fixed_vect].real/np.sum(ria_trans.eig6_v[:,idx_fixed_vect].real)
    N = vec_pi.size
    epsilon = np.ones(N)
    mat_I = np.eye(N)
    mat_A = np.outer(epsilon, vec_pi)
    mat_Q = ria_P[:N-1, :N-1]
    mat_N = np.linalg.inv(mat_I[:N-1, :N-1]-mat_Q)
    l = N-1
    # Insert the l-th row of zeros
    mat_N = np.insert(mat_N, l, 0, axis=0)
    # Insert the l-th column of zeros
    mat_N = np.insert(mat_N, l, 0, axis=1)
    mat_Z = mat_A + (mat_I - mat_A) @ mat_N @ (mat_I - mat_A)
    vec_Zdg = np.diag(mat_Z)
    mat_Zdg = np.diag(vec_Zdg)
    mat_E = np.ones((N, N))
    mat_Mdg = np.diag(1/vec_pi)
    mat_M = (mat_I - mat_Z + mat_E @ mat_Zdg) @ mat_Mdg
    mat_Mbar = mat_M - mat_Mdg
    return mat_Mbar

def mfpt_matrix_stable_ari(transMat_instance):
    ria_trans = transMat_instance
    ria_P = ria_trans.trans_mat.T
    idx_fixed_vect = np.where(np.round(ria_trans.eig6_w.real, decimals=10) == 1)[0][0]
    vec_pi = ria_trans.eig6_v[:,idx_fixed_vect].real/np.sum(ria_trans.eig6_v[:,idx_fixed_vect].real)
    N = vec_pi.size
    epsilon = np.ones(N)
    mat_I = np.eye(N)
    mat_A = np.outer(epsilon, vec_pi)
    mat_Q = ria_P[1:,1:]
    mat_N = np.linalg.inv(mat_I[1:, 1:]-mat_Q)
    l = 0
    # Insert the l-th row of zeros
    mat_N = np.insert(mat_N, l, 0, axis=0)
    # Insert the l-th column of zeros
    mat_N = np.insert(mat_N, l, 0, axis=1)
    mat_Z = mat_A + (mat_I - mat_A) @ mat_N @ (mat_I - mat_A)
    vec_Zdg = np.diag(mat_Z)
    mat_Zdg = np.diag(vec_Zdg)
    mat_E = np.ones((N, N))
    mat_Mdg = np.diag(1/vec_pi)
    mat_M = (mat_I - mat_Z + mat_E @ mat_Zdg) @ mat_Mdg
    mat_Mbar = mat_M - mat_Mdg
    return mat_Mbar