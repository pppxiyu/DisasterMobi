"""
Inter-city and inter-period component matching.

Functions
---------
calculate_dtw_matrix_standard  – pairwise DTW distance matrix
calculate_correlation_matrix   – pairwise Pearson correlation matrix
reconstruct_v_no_with_dtw      – reconstruct NO components from BR using DTW weights
process_weight_matrix          – softmax-style weight matrix from a correlation matrix
reconstruct_v_no_with_coor     – reconstruct via matrix multiplication
get_global_temporal_trends     – core-energy-weighted U3 signatures
broadcast_matrix_by_weekday    – tile a 7-day pattern to any length with weekday alignment
"""
import numpy as np
from dtw import dtw


def calculate_dtw_matrix_standard(V, V_no, n_components=10):
    """
    Pairwise DTW distance matrix between the first n components of V and V_no.
    Returns an (n × n) distance matrix.
    """
    n = min(n_components, V.shape[1], V_no.shape[1])
    dtw_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a = np.asarray(V[:, i]).reshape(-1)
            b = np.asarray(V_no[:, j]).reshape(-1)
            dtw_matrix[i, j] = dtw(a, b, distance_only=True).distance
    return dtw_matrix


def calculate_correlation_matrix(V, V_no):
    """
    Pearson correlation matrix between all columns of V (R1) and V_no (R2).
    Returns an (R1 × R2) matrix.
    """
    V       = np.asarray(V)
    V_no    = np.asarray(V_no)
    r1, r2  = V.shape[1], V_no.shape[1]
    corr    = np.zeros((r1, r2))
    for i in range(r1):
        for j in range(r2):
            c = np.corrcoef(V[:, i], V_no[:, j])[0, 1]
            corr[i, j] = np.nan_to_num(c)
    return corr


def reconstruct_v_no_with_dtw(V_basis, dtw_results, n_components=5, temperature=0.5):
    """
    Reconstructs target components from a basis using DTW-based kernel weights.
    Returns (weights_normalized, V_reconstructed).
    """
    V_b    = V_basis[:, :n_components]
    w      = np.exp(-dtw_results / temperature)
    w_norm = w / w.sum(axis=0, keepdims=True)
    return w_norm, V_b @ w_norm


def process_weight_matrix(corr_results, temperature=0.2):
    """
    Converts a correlation matrix to softmax-style mapping weights.
    Returns weights_normalized (R_basis × R_target).
    """
    sim    = np.maximum(corr_results, 0)
    w      = np.exp(sim / temperature)
    w_norm = w / (w.sum(axis=0, keepdims=True) + 1e-9)
    return w_norm


def reconstruct_v_no_with_coor(V_basis, weight_matrix):
    """
    Reconstructs target temporal components via: V_basis @ weight_matrix.
    Shape: (Time × R_basis) @ (R_basis × R_target) = (Time × R_target).
    """
    return V_basis @ weight_matrix


def get_global_temporal_trends(core_dis, factors_dis, de_scale=False):
    """
    Computes global temporal trends for each temporal component k:
      - de_scale=False: scale U3[:,k] by the Frobenius norm of core[:,:,k]
      - de_scale=True:  return U3[:,k] unscaled

    Returns a matrix of shape (Time, R3).
    """
    U3     = factors_dis[2]
    R3     = U3.shape[1]
    trends = np.zeros((U3.shape[0], R3))
    for k in range(R3):
        intensity  = np.linalg.norm(core_dis[:, :, k], ord='fro')
        trends[:, k] = U3[:, k] if de_scale else intensity * U3[:, k]
    return trends


def broadcast_matrix_by_weekday(source_matrix, first_day_source,
                                 first_day_target, target_len):
    """
    Tiles a 7-row pattern matrix to target_len rows, aligning by day of week.

    source_matrix : (7, n) – one row per weekday
    first_day_source/target : e.g. 'Thursday'
    """
    days      = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    si        = days.index(first_day_source.capitalize())
    ti        = days.index(first_day_target.capitalize())
    src_map   = {(si + i) % 7: source_matrix[i] for i in range(7)}

    out = np.zeros((target_len, source_matrix.shape[1]))
    for t in range(target_len):
        out[t] = src_map[(ti + t) % 7]
    return out
