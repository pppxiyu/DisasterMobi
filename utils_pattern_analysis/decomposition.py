"""
Matrix factorization (NMF) and tensor decomposition (Tucker) functions.

NMF section
-----------
  decompose_mobility_patterns            – standard NMF
  decompose_mobility_patterns_with_init  – NMF with warm-start W/H
  generate_W_init_by_weekday             – broadcasts a 7-day W to any length
  h_slice_to_od_matrix                   – converts a spatial component to OD DataFrame

Tucker section
--------------
  non_negative_tucker_hals               – custom HALS Tucker with temporal regularisation
  non_negative_tucker_decomposition      – convenience wrapper around tensorly's NNTucker
  check_reconstruction_error_detailed    – per-zero / per-nonzero error diagnostics
  project_onto_spatial_basis             – fix U1/U2 from normal data and solve for disaster U3/G
  get_scaled_lambda                      – data-adaptive Laplacian regularisation weight
  calculate_laplacian_from_graph         – build L aligned to tensor node ordering
  regularized_update                     – Laplacian-regularised eigenvector update
  nmf_graph_update                       – graph-regularised multiplicative NMF update
  process_core_tensor                    – top-n values/locations in the core tensor
  check_core_ratio                       – core size appropriateness test (section 2.2.2)
"""
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.decomposition import NMF

import tensorly as tl
from tensorly.tenalg import multi_mode_dot
from tensorly.base import unfold
from tensorly.tucker_tensor import (
    tucker_to_tensor, validate_tucker_rank,
    tucker_normalize, TuckerTensor,
)
from tensorly.decomposition import non_negative_tucker
from tensorly.decomposition._tucker import initialize_tucker
from tensorly.solvers.nnls import hals_nnls, fista, active_set_nnls
from scipy.linalg import eigh
from scipy import sparse


# ── NMF ──────────────────────────────────────────────────────────────────────

def decompose_mobility_patterns(X, n_behaviors=5, l1_reg=0.0):
    """
    NMF: X ≈ W @ H  (X shape: n_time × n_flows)
    Returns W (time × behaviors) and H (behaviors × flows).

    l1_reg : float  (paper's sparsity parameter q, default 0 = no regularisation)
        Applies the symmetric L1 penalty  q·sum(W) + q·sum(H)  from the reference
        paper (Eq. 3).  Converted to sklearn's alpha_W / alpha_H internally so the
        effective penalty equals q regardless of matrix size:
            alpha_W = q / n_features,   alpha_H = q / n_samples,   l1_ratio = 1.
    """
    n_samples, n_features = X.shape
    alpha_W = l1_reg / n_features if l1_reg > 0 else 0.0
    alpha_H = l1_reg / n_samples  if l1_reg > 0 else 0.0

    print(f"NMF (normal): ")
    print(f"    input shape {X.shape}, n_components={n_behaviors}, l1_reg={l1_reg}, ")
    print(f"    min={X.min():.4f}, max={X.max():.4f}, zeros={np.mean(X==0)*100:.1f}%")
    model = NMF(n_components=n_behaviors, init='nndsvd', solver='cd',
                random_state=42, max_iter=5000,
                alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=1.0)
    W = model.fit_transform(X)
    print(f"NMF (normal):")
    print(f"    n_iter={model.n_iter_}, reconstruction_err={model.reconstruction_err_:.4f}")
    return W, model.components_


def normalize_nmf_components(W, H):
    """
    Resolve NMF's scale ambiguity so temporal factors (W columns) are
    comparable across components.

    For each component i, the product W[:,i] ⊗ H[i,:] is invariant to the
    choice of scale, so raw W values cannot be compared across components.
    This function normalises each W column to unit L2 norm and absorbs the
    scale into the corresponding H row:

        W_n[:,i] = W[:,i] / ‖W[:,i]‖₂          (unit temporal signature)
        H_n[i,:] = H[i,:] × ‖W[:,i]‖₂          (spatial factor × magnitude)

    The component weight (importance) is then ‖H_n[i,:]‖₂.

    Returns
    -------
    W_n : ndarray  [time × k]  — normalised temporal factors, columns ∈ unit sphere
    H_n : ndarray  [k × OD]   — spatial factors rescaled to carry all magnitude
    weights : ndarray [k]      — ‖W[:,i]‖₂ × ‖H[i,:]‖₂, component importance
    """
    col_norms = np.linalg.norm(W, axis=0)          # shape (k,)
    col_norms = np.where(col_norms == 0, 1.0, col_norms)  # guard divide-by-zero
    W_n = W / col_norms                             # broadcast over rows
    H_n = H * col_norms[:, np.newaxis]              # broadcast over columns
    weights = col_norms * np.linalg.norm(H, axis=1)
    return W_n, H_n, weights


def decompose_mobility_patterns_with_init(X, W_init, H_init, n_behaviors, l1_reg=0.0):
    """
    NMF with a warm-start W / H (uses 'custom' init).

    l1_reg : float  same semantics as in decompose_mobility_patterns.
    """
    n_samples, n_features = X.shape
    alpha_W = l1_reg / n_features if l1_reg > 0 else 0.0
    alpha_H = l1_reg / n_samples  if l1_reg > 0 else 0.0

    print(f"  NMF (disaster, warm-start): input shape {X.shape}, l1_reg={l1_reg}")
    model = NMF(n_components=n_behaviors, init='custom', solver='cd',
                random_state=42, max_iter=5000,
                alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=1.0)
    W = model.fit_transform(X, W=W_init, H=H_init)
    print(f"  NMF (disaster): done, n_iter={model.n_iter_}, "
          f"reconstruction_err={model.reconstruction_err_:.4f}")
    return W, model.components_


def generate_W_init_by_weekday(W_source, first_day_source, first_day_target,
                                target_len, slots_per_day):
    """
    Broadcasts a 7-day temporal factor matrix (W_source, shape 7*slots × behaviors)
    to a target length, preserving weekday alignment.

    slots_per_day is required (no default) — pass SLOTS_ACTIVE from config so
    the value stays correct when switching between 2h and 3h resolution.
    """
    expected = 7 * slots_per_day
    assert W_source.shape[0] == expected, (
        f"W_source must be exactly 7 days ({expected} rows), got {W_source.shape[0]}."
    )
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    si   = days.index(first_day_source.capitalize())
    day_patterns = {
        (si + i) % 7: W_source[i * slots_per_day: (i + 1) * slots_per_day, :]
        for i in range(7)
    }
    ti = days.index(first_day_target.capitalize())
    n_days = int(np.ceil(target_len / slots_per_day))
    W_init = np.vstack([day_patterns[(ti + d) % 7] for d in range(n_days)])
    return W_init[:target_len, :]


def h_slice_to_od_matrix(h_slice, spatial_mapping):
    """
    Converts a row of H (shape (1, N) or (N,)) and its edge mapping to a
    wide-format OD matrix DataFrame.
    """
    flows       = h_slice.flatten()
    origins     = [e[0] for e in spatial_mapping]
    destinations= [e[1] for e in spatial_mapping]
    df_long     = pd.DataFrame({'origin': origins, 'destination': destinations, 'flow': flows})
    return df_long.pivot(index='origin', columns='destination', values='flow').fillna(0)


# ── Tucker / HALS ─────────────────────────────────────────────────────────────

def non_negative_tucker_hals(
    tensor, rank, n_iter_max=100, init="svd", svd="truncated_svd",
    tol=1e-8, sparsity_coefficients=None, core_sparsity_coefficient=None,
    fixed_modes=None, random_state=None, verbose=False,
    normalize_factors=False, return_errors=False, exact=False,
    algorithm="fista",
    temporal_lambda=0,
):
    """
    Non-negative Tucker decomposition with HALS and optional cosine-similarity
    regularisation on the temporal factor (mode 2).

    temporal_lambda > 0 pulls the temporal factor towards the reference supplied
    via init=(core_ref, [U1_ref, U2_ref, U3_ref]).
    """
    rank   = validate_tucker_rank(tl.shape(tensor), rank=rank)
    n_modes = tl.ndim(tensor)

    reference_factors = None
    if isinstance(init, (tuple, list)) and len(init) == 2:
        reference_factors = init[1]

    if sparsity_coefficients is None or not isinstance(sparsity_coefficients, Iterable):
        sparsity_coefficients = [sparsity_coefficients] * n_modes
    if fixed_modes is None:
        fixed_modes = []
    if tl.ndim(tensor) - 1 in fixed_modes:
        warnings.warn("Fixing the last mode is not supported.")
        fixed_modes.remove(tl.ndim(tensor) - 1)
    for fm in fixed_modes:
        sparsity_coefficients[fm] = None

    modes = [m for m in range(n_modes) if m not in fixed_modes]
    nn_core, nn_factors = initialize_tucker(
        tensor, rank, modes, init=init, svd=svd,
        random_state=random_state, non_negative=True,
    )

    norm_tensor = tl.norm(tensor, 2)
    rec_errors  = []

    for iteration in range(n_iter_max):
        for mode in modes:
            pseudo_inverse = nn_factors.copy()
            for i, f in enumerate(nn_factors):
                if i != mode:
                    pseudo_inverse[i] = tl.dot(tl.conj(tl.transpose(f)), f)

            core_cross = multi_mode_dot(nn_core, pseudo_inverse, skip=mode)
            UtU = tl.dot(unfold(core_cross, mode), tl.transpose(unfold(nn_core, mode)))

            tensor_cross = multi_mode_dot(tensor, nn_factors, skip=mode, transpose=True)
            MtU  = tl.dot(unfold(tensor_cross, mode), tl.transpose(unfold(nn_core, mode)))
            UtM  = tl.transpose(MtU)

            # Cosine-similarity temporal regularisation
            if mode == 2 and temporal_lambda > 0 and reference_factors is not None:
                ref_f      = reference_factors[2]
                ref_norms  = tl.norm(ref_f, order=2, axis=0)
                ref_f_unit = ref_f / (ref_norms + 1e-12)
                UtM       += temporal_lambda * tl.transpose(ref_f_unit)
                rank_dim   = UtU.shape[0]
                UtU       += temporal_lambda * tl.eye(rank_dim, **tl.context(tensor))

            nn_factor = hals_nnls(
                UtM, UtU, tl.transpose(nn_factors[mode]),
                n_iter_max=100,
                sparsity_coefficient=sparsity_coefficients[mode],
                exact=exact,
            )
            nn_factors[mode] = tl.transpose(nn_factor)

        # Core update
        if algorithm == "fista":
            pseudo_inverse[-1] = tl.dot(tl.transpose(nn_factors[-1]), nn_factors[-1])
            core_est  = multi_mode_dot(tensor, nn_factors, transpose=True)
            lr        = 1
            for MtM in pseudo_inverse:
                s   = tl.truncated_svd(MtM, n_eigenvecs=1)[1]
                lr *= 1 / s[0]
            nn_core = fista(core_est, pseudo_inverse, x=nn_core,
                            n_iter_max=n_iter_max,
                            sparsity_coef=core_sparsity_coefficient, lr=lr)
        elif algorithm == "active_set":
            pseudo_inverse[-1] = tl.dot(tl.transpose(nn_factors[-1]), nn_factors[-1])
            core_est_vec       = tl.base.tensor_to_vec(
                tl.tenalg.mode_dot(tensor_cross,
                                   tl.transpose(nn_factors[modes[-1]]), modes[-1])
            )
            pseudo_inverse_kr  = tl.tenalg.kronecker(pseudo_inverse)
            vectorcore         = active_set_nnls(core_est_vec, pseudo_inverse_kr,
                                                  x=nn_core, n_iter_max=n_iter_max)
            nn_core = tl.reshape(vectorcore, tl.shape(nn_core))

        rec_error = tl.norm(tensor - tucker_to_tensor((nn_core, nn_factors)), 2) / norm_tensor
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print(f"  iter {iteration}: error={rec_errors[-1]:.6f}, "
                      f"Δ={rec_errors[-2]-rec_errors[-1]:.2e}")
            if tol and tl.abs(rec_errors[-2] - rec_errors[-1]) < tol:
                break

    if normalize_factors:
        nn_core, nn_factors = tucker_normalize((nn_core, nn_factors))

    result = TuckerTensor((nn_core, nn_factors))
    return (result, rec_errors) if return_errors else result


def non_negative_tucker_decomposition(X, core_shape, n_iter_max=1000,
                                       tol=1e-6, random_state=None, verbose=False):
    """
    Convenience wrapper around tensorly's non_negative_tucker.
    Returns (G, [U1, U2, U3]) as plain NumPy arrays.
    """
    tl.set_backend("numpy")
    core, factors = non_negative_tucker(
        tl.tensor(X, dtype=tl.float64),
        rank=list(core_shape),
        n_iter_max=n_iter_max, init='svd', tol=tol,
        random_state=random_state, verbose=verbose,
        normalize_factors=True,
    )
    G       = np.array(core)
    factors = [np.array(f) for f in factors]

    if verbose:
        X_approx = tl.tucker_to_tensor((tl.tensor(G), [tl.tensor(f) for f in factors]))
        err = np.linalg.norm(X - np.array(X_approx)) / np.linalg.norm(X)
        print(f"Final relative reconstruction error: {err:.6f}")

    return G, factors


# ── Diagnostics ───────────────────────────────────────────────────────────────

def check_reconstruction_error_detailed(original, core, factors):
    """
    Splits error into zero vs non-zero entries of the original tensor.
    Prints a breakdown and returns (rel_err_nonzero, rmse_zero, diff_tensor).
    """
    reconstructed = tl.tucker_to_tensor((core, factors))
    diff          = original - reconstructed

    mask_zero    = (original == 0)
    mask_nonzero = ~mask_zero

    err_nz   = diff[mask_nonzero]
    orig_nz  = original[mask_nonzero]
    rel_err  = np.linalg.norm(err_nz) / np.linalg.norm(orig_nz)

    err_zero         = reconstructed[mask_zero]
    rmse_zero        = np.sqrt(np.mean(err_zero ** 2))
    max_hallucination= np.max(np.abs(err_zero))

    print("=== Reconstruction Error Analysis ===")
    print(f"Sparsity: {np.mean(mask_zero)*100:.2f}%")
    print(f"Non-zero relative error:   {rel_err:.6f}")
    print(f"Zero-location RMSE:        {rmse_zero:.6f}")
    print(f"Zero-location max abs err: {max_hallucination:.6f}")
    return rel_err, rmse_zero, diff


def project_onto_spatial_basis(tensor_dis, factors_norm, ranks, max_iter=10):
    """
    Fixes U1, U2 from normal-period factors and solves for U3 and G using
    disaster-period data. Returns (core_dis, factors_dis).
    """
    U1_fix, U2_fix = factors_norm[0], factors_norm[1]
    Y_init     = tl.tenalg.multi_mode_dot(tensor_dis, [U1_fix.T, U2_fix.T], modes=[0, 1])
    Y_unfolded = tl.unfold(Y_init, 2)
    U3_dis, _, _= np.linalg.svd(Y_unfolded, full_matrices=False)
    U3_dis      = U3_dis[:, :ranks[2]]
    factors_dis = [U1_fix, U2_fix, U3_dis]

    for _ in range(max_iter):
        Y          = tl.tenalg.multi_mode_dot(tensor_dis,
                                               [factors_dis[0].T, factors_dis[1].T], modes=[0, 1])
        Y_unfolded = tl.unfold(Y, 2)
        U, _, _    = np.linalg.svd(Y_unfolded, full_matrices=False)
        factors_dis[2] = U[:, :ranks[2]]

    core_dis = tl.tenalg.multi_mode_dot(tensor_dis, [f.T for f in factors_dis])
    return core_dis, factors_dis


def get_scaled_lambda(X, L, alpha=0.1):
    """Data-adaptive Laplacian regularisation weight."""
    norm_X_sq = (sparse.linalg.norm(X) ** 2 if sparse.issparse(X)
                 else np.linalg.norm(X, axis=None) ** 2)
    norm_L    = (sparse.linalg.norm(L) if sparse.issparse(L)
                 else np.linalg.norm(L, ord='fro'))
    return alpha * (norm_X_sq / norm_L)


def calculate_laplacian_from_graph(G, spatial_mapping):
    """Returns the combinatorial Laplacian L = D - A aligned to the tensor ordering."""
    ordered = [spatial_mapping[i] for i in range(len(spatial_mapping))]
    return nx.laplacian_matrix(G, nodelist=ordered, weight=None).toarray()


def regularized_update(Y_unfolded, L, rank, lmbda):
    """Solves (C - λL)u = γu and returns the top `rank` eigenvectors."""
    C = np.dot(Y_unfolded, Y_unfolded.T)
    eigvals, eigvecs = eigh(C - lmbda * L)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx[:rank]]


def nmf_graph_update(Y_unfolded, U, L, lmbda, epsilon=1e-9):
    """
    Multiplicative update for a spatial factor with graph regularisation.
    Objective: ||Y - UH||² + λ Tr(UᵀLU)
    """
    D = np.diag(np.diag(L))
    A = D - L
    numerator   = np.maximum(np.dot(Y_unfolded, Y_unfolded.T), 0) @ U
    denominator = (U @ U.T @ numerator) + epsilon
    numerator   = numerator   + lmbda * (A @ U)
    denominator = denominator + lmbda * (D @ U) + epsilon
    return U * (numerator / denominator)


def process_core_tensor(core, n=3, plot_distribution=False, time_comp=None):
    """
    Returns the top-n (index, abs_value) pairs in the core tensor.
    If time_comp is set, only the corresponding 2-D slice is examined.
    """
    working = core[:, :, time_comp] if time_comp is not None else core
    if plot_distribution:
        import os as _os
        import matplotlib.pyplot as _plt
        import seaborn as _sns
        _plt.figure(figsize=(10, 4))
        _sns.histplot(working.flatten(), kde=False, color='skyblue')
        _plt.tight_layout()
        _os.makedirs('outputs', exist_ok=True)
        _plt.savefig('outputs/core_distribution.png', bbox_inches='tight', dpi=150)
        _plt.close()

    core_abs    = np.abs(working)
    flat_idx    = np.argsort(core_abs.ravel())[::-1][:n]
    top_idx     = np.unravel_index(flat_idx, working.shape)
    top_vals    = core_abs.ravel()[flat_idx]

    results = []
    for i in range(len(top_vals)):
        loc = (top_idx[0][i], top_idx[1][i], time_comp) if time_comp is not None \
              else (top_idx[0][i], top_idx[1][i], top_idx[2][i])
        results.append({"abs_value": top_vals[i], "location": loc})
    return results


def check_core_ratio(G, alpha=0.25):
    """
    Tests whether the core G is appropriately sized (section 2.2.2).
    Returns (is_appropriate: bool, ratios: dict).
    """
    is_ok, ratios = True, {}
    for n in range(G.ndim):
        axes  = tuple(j for j in range(G.ndim) if j != n)
        v_n   = np.sum(G, axis=axes)
        ratio = np.min(v_n) / np.max(v_n)
        ratios[f"Mode {n}"] = ratio
        if ratio <= alpha:
            is_ok = False
    return is_ok, ratios
