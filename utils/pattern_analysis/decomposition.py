"""
Matrix factorization (NMF) and tensor decomposition (Tucker) functions.

NMF section (production: run_pattern_nmf.py via nmf_pipeline.py)
-----------
  select_segment_columns                 – time-column indices for named segments
  decompose_mobility_patterns            – standard NMF
  fit_nmf_basis_and_project              – fit H on a column subset, project the full window
  normalize_nmf_components               – unit-L2 W columns, scale absorbed into H
  nmf_context_multiplicative             – context-aware shared-factor NMF (Chen et al. 2018)
  project_W_onto_H                       – re-solve W with H frozen

Tucker section (only used by archive/run_pattern_tucker.py)
--------------
  non_negative_tucker_hals               – custom HALS Tucker with temporal regularisation
  non_negative_tucker_decomposition      – convenience wrapper around tensorly's NNTucker
  check_reconstruction_error_detailed    – per-zero / per-nonzero error diagnostics
"""
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

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


# ── NMF ──────────────────────────────────────────────────────────────────────

def select_segment_columns(segments, n_nor, n_dis, n_time):
    """
    Return the sorted unique time-column indices for the named segments.

    Spans match the NMF pipeline's window layout:
        normal=[0, n_nor)   buffer=[n_nor, n_dis)   disaster=[n_dis, n_time)
    Selecting all three yields np.arange(n_time).  Used to pick which time rows
    fit the NMF basis when the decomposition window is sliced.
    """
    spans = {'normal': (0, n_nor), 'buffer': (n_nor, n_dis),
             'disaster': (n_dis, n_time)}
    bad = set(segments) - set(spans)
    if bad:
        raise ValueError(f"unknown segment(s) {sorted(bad)}; valid: {sorted(spans)}")
    cols = (np.concatenate([np.arange(*spans[s]) for s in segments]) if segments
            else np.empty(0, dtype=int))
    return np.unique(cols.astype(int))


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

    print("NMF:")
    print(f"    input shape {X.shape}, n_components={n_behaviors}, l1_reg={l1_reg}, ")
    print(f"    min={X.min():.4f}, max={X.max():.4f}, zeros={np.mean(X==0)*100:.1f}%")
    model = NMF(n_components=n_behaviors, init='nndsvd', solver='cd',
                random_state=42, max_iter=5000,
                alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=1.0)
    W = model.fit_transform(X)
    print("NMF:")
    print(f"    n_iter={model.n_iter_}, reconstruction_err={model.reconstruction_err_:.4f}")
    return W, model.components_


def fit_nmf_basis_and_project(X_fit, X_full, n_behaviors=5, l1_reg=0.0):
    """
    Fit the NMF basis H on a SUBSET of time rows, then project the full window
    onto that frozen basis.

        X_fit  : [n_fit_time  × n_flows]   rows that DEFINE the components
        X_full : [n_full_time × n_flows]   every row, projected onto the basis

    transform() freezes H = components_ and re-solves only W with the same CD
    solver / tol / max_iter (W starts from zeros, so it is deterministic).
    alpha_W = l1_reg / n_features with n_features = n_flows is identical for the
    fit subset and the full window, so the L1 pressure on W is unchanged across
    the projection; the fit row count only affects alpha_H, which is unused once
    H is frozen.  Same l1_reg → (alpha_W, alpha_H) convention as
    decompose_mobility_patterns.

    Returns (W_full, H) exactly like decompose_mobility_patterns.
    """
    n_samples, n_features = X_fit.shape
    if n_behaviors > n_samples:          # nndsvd needs n_components <= n_samples
        raise ValueError(
            f"fit segment has {n_samples} time rows < n_components {n_behaviors}; "
            f"widen NMF_FIT_SEGMENTS or lower n_behaviors."
        )
    alpha_W = l1_reg / n_features if l1_reg > 0 else 0.0
    alpha_H = l1_reg / n_samples  if l1_reg > 0 else 0.0

    print("NMF (fit then project):")
    print(f"    fit shape {X_fit.shape}, project shape {X_full.shape}, "
          f"n_components={n_behaviors}, l1_reg={l1_reg}")
    model = NMF(n_components=n_behaviors, init='nndsvd', solver='cd',
                random_state=42, max_iter=5000,
                alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=1.0)
    model.fit(X_fit)
    W_full = model.transform(X_full)
    print("NMF (fit then project):")
    print(f"    fit n_iter={model.n_iter_}, "
          f"fit reconstruction_err={model.reconstruction_err_:.4f}")
    return W_full, model.components_


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


# ── Context-aware NMF (shared-factor, Chen et al. 2018) ──────────────────────

def _init_wh_nndsvd(X, n_components, random_state=42):
    """
    nndsvd init for W, H, matching the sklearn NMF path used elsewhere; falls
    back to a non-negative random init if sklearn's private initializer is
    unavailable or rejects the shape.
    """
    try:
        from sklearn.decomposition._nmf import _initialize_nmf
        W, H = _initialize_nmf(X, n_components, init='nndsvd',
                               random_state=random_state)
        return np.asarray(W), np.asarray(H)
    except Exception:
        rng   = np.random.default_rng(random_state)
        scale = np.sqrt(np.abs(X).mean() / max(n_components, 1)) if X.size else 1.0
        W = np.abs(rng.standard_normal((X.shape[0], n_components))) * scale
        H = np.abs(rng.standard_normal((n_components, X.shape[1]))) * scale
        return W, H


def nmf_context_multiplicative(X, Y, mask, n_components, lambda_ctx=0.1,
                               l1_reg=0.0, max_iter=5000, tol=1e-6,
                               random_state=42, eps=1e-9, verbose=False):
    """
    Context-aware NMF by non-negative multiplicative updates (shared-factor /
    Chen et al. 2018, *A Context-Aware NMF Framework …*, IEEE MIPR 2018).

        min_{W,H,G ≥ 0}  ½‖X − W H‖²_F
                       + (λ/2) Σ_j m_j ‖Y[j,:] − H[:,j]ᵀ G‖²
                       + q (Σ W + Σ H)

    X    : [n_time × n_OD]   mobility matrix (this pipeline passes X_all.T)
    Y    : [n_OD × C']       per-flow POI feature, built so ‖Y[j,:]‖ tracks the
                             flow's volume (see space_function.build_flow_poi_feature)
    mask : [n_OD] bool       flows carrying a POI constraint; m_j = 0 excludes a
                             flow from the context term entirely (numerator AND
                             denominator), so a missing-POI flow reduces to plain
                             NMF rather than being shrunk toward zero.

    The context term ties each flow's H-column (its k-dim embedding), through a
    single GLOBAL map G [k × C'], to that flow's POI feature.  Because Y is built
    to co-scale with H, G only has to fit the direction (which land-use type),
    not the per-flow magnitude.

    lambda_ctx : RELATIVE context weight.  The effective λ is auto-scaled by
                 ‖X‖²_F / ‖Y‖²_F so the two reconstruction terms are comparable
                 and lambda_ctx means the same thing across cities.
    l1_reg     : symmetric L1 penalty q on W and H (same convention as
                 decompose_mobility_patterns — q is added to the MU denominators).

    Returns (W [n_time × k], H [k × n_OD], G [k × C'], errors).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    m = np.asarray(mask, dtype=float)                  # [n_OD]
    n_od = X.shape[1]
    Cp   = Y.shape[1]

    # Auto-scale λ so ½‖X−WH‖² and (λ/2)‖Y−HᵀG‖² are comparable in magnitude.
    xnorm2 = float(np.sum(X * X))
    ynorm2 = float(np.sum((Y * m[:, None]) ** 2))
    lam    = lambda_ctx * (xnorm2 / (ynorm2 + eps))

    W, H = _init_wh_nndsvd(X, n_components, random_state=random_state)
    rng  = np.random.default_rng(random_state)
    G    = np.abs(rng.standard_normal((n_components, Cp))) * \
           (np.sqrt(ynorm2 / max(n_od, 1)) / np.sqrt(max(n_components, 1)) + eps)

    q       = l1_reg
    errors  = []
    prev    = None
    for it in range(max_iter):
        Hm = H * m[None, :]                             # mask flow-columns of H

        # G update (context only):  G ← G ⊙ (Hm Y) / (Hm Hᵀ G)
        G *= (Hm @ Y) / (Hm @ H.T @ G + eps)

        # H update: reconstruction + masked context + L1
        ctx_num = G @ Y.T                               # [k × n_OD]
        ctx_den = (G @ G.T) @ Hm                        # [k × n_OD], masked
        H *= (W.T @ X + lam * ctx_num) / \
             ((W.T @ W) @ H + lam * ctx_den + q + eps)

        # W update: reconstruction + L1 (context does not involve W)
        W *= (X @ H.T) / (W @ (H @ H.T) + q + eps)

        if it % 10 == 0 or it == max_iter - 1:
            rec = 0.5 * np.sum((X - W @ H) ** 2)
            ctx = 0.5 * lam * np.sum(m[:, None] * (Y - H.T @ G) ** 2)
            obj = rec + ctx + q * (W.sum() + H.sum())
            errors.append(obj)
            if verbose:
                print(f"    iter {it}: obj={obj:.4e} rec={rec:.4e} ctx={ctx:.4e}")
            if prev is not None and abs(prev - obj) < tol * (prev + eps):
                break
            prev = obj
    return W, H, G, errors


def project_W_onto_H(X_full, H, l1_reg=0.0, max_iter=5000, tol=1e-6,
                     random_state=42, eps=1e-9):
    """
    Re-solve W ≥ 0 minimising ½‖X_full − W H‖² + q·ΣW with H FROZEN — the
    projection half of the fit-then-project path for the context-aware solver.

    The context term lives only in the fit stage (it constrains H); projecting
    the full window onto the frozen basis is a plain non-negative least squares
    for W, mirroring sklearn's transform() role in fit_nmf_basis_and_project.
    """
    X_full = np.asarray(X_full, dtype=float)
    H      = np.asarray(H, dtype=float)
    k      = H.shape[0]
    rng    = np.random.default_rng(random_state)
    W      = np.abs(rng.standard_normal((X_full.shape[0], k))) * \
             (np.sqrt(np.abs(X_full).mean() / max(k, 1)) + eps)

    HHt = H @ H.T
    XHt = X_full @ H.T
    q   = l1_reg
    prev = None
    for it in range(max_iter):
        W *= XHt / (W @ HHt + q + eps)
        if it % 10 == 0 or it == max_iter - 1:
            obj = 0.5 * np.sum((X_full - W @ H) ** 2) + q * W.sum()
            if prev is not None and abs(prev - obj) < tol * (prev + eps):
                break
            prev = obj
    return W


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

