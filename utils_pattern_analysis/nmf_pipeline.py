"""
Shared building blocks for the single-NMF (paper-style) decomposition pipeline.

Both run_pattern_nmf.py and run_pattern_distance_decay_paired.py rely on
exactly the same matrix construction and NMF call, so the two helpers below
live here instead of being duplicated in each run script.

Functions
---------
build_city_matrices(graphs, days_window, days_disaster_in_window, filter_factor)
    Take the last `days_window` days of a city's graph sequence, trim overnight
    slots, flatten to a 2-D [OD × time] matrix, and filter out low-activity
    OD pairs.  Returns (X_all, n_nor, mapping) where n_nor is the column index
    that separates the normal portion from the disaster portion (visualisation
    only — the NMF runs on the combined matrix).

decompose_city(X_all, n_behaviors, l1_reg=0.0)
    Run a single NMF on the combined matrix (using the shared
    decompose_mobility_patterns / normalize_nmf_components helpers) and print
    the importance ranking of the resulting components.  Returns
    (W, H, weights).

Why both at once?
-----------------
The paper's framework decomposes the *combined* normal + disaster matrix with
a single NMF.  This keeps the OD-pair row ordering and component identities
consistent across periods, so no alignment / matching step is needed between
W blocks.  W is later split at column n_nor purely for plotting.
"""
import textwrap

import numpy as np

from config import (
    SLOT_PER_DAY, SLOT_TRIM_START, SLOT_TRIM_END, SLOTS_ACTIVE,
)
from utils_pattern_analysis.graph_io import (
    periodic_trim, graphs_to_2d_matrix, filter_inactive_locations_2d,
    print_matrix_diagnostics,
)
from utils_pattern_analysis.decomposition import (
    decompose_mobility_patterns, fit_nmf_basis_and_project,
    normalize_nmf_components,
)


def build_city_matrices(graphs, days_window, days_disaster_in_window,
                         filter_factor):
    """
    Build a [OD-pairs × days_window*SLOTS_ACTIVE] matrix from the last
    days_window days of the graph sequence.

    A single graphs_to_2d_matrix call covers the entire window, so all rows
    have a consistent OD-pair ordering with no alignment step required.
    Filtering is applied to the full window (normal + disaster together).

    Parameters
    ----------
    graphs : list of DiGraph
        Full graph sequence for a city (one graph per time slot).
    days_window : int
        Total number of days kept (counted back from the end).
    days_disaster_in_window : int
        How many trailing days are considered the disaster portion.
    filter_factor : int or float
        Multiplier in the per-OD activity threshold
        `SLOTS_ACTIVE * days_window * filter_factor`.

    Returns
    -------
    X_all   : ndarray  [n_OD × (days_window * SLOTS_ACTIVE)]
    n_nor   : int       column index where the disaster portion begins
                        (used for visualisation, not for the NMF itself).
    mapping : list of (origin_id, dest_id) aligned to X_all rows.
    """
    slot_active           = SLOTS_ACTIVE
    days_normal_in_window = days_window - days_disaster_in_window

    window_graphs = periodic_trim(
        graphs[-SLOT_PER_DAY * days_window:],
        cycle_len=SLOT_PER_DAY,
        trim_start=SLOT_TRIM_START, trim_end=SLOT_TRIM_END,
    )
    X_all, mapping = graphs_to_2d_matrix(window_graphs)

    threshold = slot_active * days_window * filter_factor
    X_all, _, mapping = filter_inactive_locations_2d(
        X_all, X_all, mapping, threshold,
    )

    n_nor = days_normal_in_window * slot_active
    print(f"  Combined: {X_all.shape}  (disaster starts at column {n_nor})")
    print_matrix_diagnostics(X_all, mapping, label='full window')
    return X_all, n_nor, mapping


def print_projection_diagnostics(X_full, W_raw, H_raw, fit_time_cols):
    """
    Reconstruction quality when the basis is fit on a subset and the full
    window is projected onto it.  Reports relative error (‖residual‖/‖X‖) on the
    full / fit / held-out rows, plus the count of near-zero-weight components (a
    too-large-k symptom).  Same indented layout as print_matrix_diagnostics.
    """
    resid = X_full - W_raw @ H_raw
    rel = lambda rows: (np.linalg.norm(resid[rows]) / np.linalg.norm(X_full[rows])
                        if X_full[rows].any() else float('nan'))
    n_time = X_full.shape[0]
    held   = np.setdiff1d(np.arange(n_time), fit_time_cols)
    comp_w = np.linalg.norm(W_raw, axis=0) * np.linalg.norm(H_raw, axis=1)
    n_tiny = int((comp_w < 0.01 * comp_w.max()).sum()) if comp_w.max() > 0 else 0
    held_rel = f"{rel(held):.4f}" if held.size else "n/a"
    print("  Projection diagnostics :")
    print(f"    Fit rows         : {fit_time_cols.size} / {n_time}")
    print(f"    Rel. error full  : {rel(np.arange(n_time)):.4f}")
    print(f"    Rel. error fit   : {rel(fit_time_cols):.4f}")
    print(f"    Rel. error held  : {held_rel}  ({held.size} rows)")
    print(f"    Near-zero comps  : {n_tiny} / {W_raw.shape[1]}")


def decompose_city(X_all, n_behaviors, l1_reg=0.0, fit_time_cols=None):
    """
    Single NMF on the combined normal+disaster matrix.

    After decomposition, W columns are normalised to unit L2 norm so temporal
    signatures are comparable across components.  The scale is absorbed into
    H rows, and component weights ‖W[:,i]‖₂ · ‖H[i,:]‖₂ are printed in
    descending order of importance.

    Parameters
    ----------
    X_all         : ndarray [n_OD × n_time]  (the matrix from build_city_matrices)
    n_behaviors   : int   number of NMF components
    l1_reg        : float symmetric L1 penalty (paper's sparsity parameter q)
    fit_time_cols : None | 1-D int array
        None  -> fit AND project on the full window (the original behaviour,
                 byte-identical to before).
        array -> fit the basis H on those time columns of X_all only, then
                 project the full window onto H, so W still spans every row.
                 W/H shapes and downstream indices are unchanged either way.

    Returns
    -------
    W       : ndarray [n_time × k]   normalised temporal factors
    H       : ndarray [k × n_OD]     spatial factors carrying the magnitude
    weights : ndarray [k]            per-component importance
    """
    if fit_time_cols is None:
        W_raw, H_raw = decompose_mobility_patterns(
            X_all.T, n_behaviors=n_behaviors, l1_reg=l1_reg,
        )
    else:
        X_full = X_all.T                       # [n_time × n_OD]
        W_raw, H_raw = fit_nmf_basis_and_project(
            X_full[fit_time_cols, :], X_full,
            n_behaviors=n_behaviors, l1_reg=l1_reg,
        )
        print_projection_diagnostics(X_full, W_raw, H_raw, fit_time_cols)
    W, H, weights = normalize_nmf_components(W_raw, H_raw)
    order = np.argsort(weights)[::-1]
    weight_str = ", ".join(f"[{i}]={weights[i]:.1f}" for i in order)
    print("Component weights (importance): ")
    print(textwrap.fill(weight_str, width=100,
                        initial_indent="    ", subsequent_indent="    "))
    return W, H, weights


# ── Paired (two-segment) NMF helpers ─────────────────────────────────────────
# Used by run_pattern_distance_decay_paired.py — splits the same trailing
# window into normal vs. disaster halves, runs NMF on each separately, and
# uses warm-start so component identities are preserved across periods.

def build_paired_matrices(graphs, days_window, days_disaster, filter_factor):
    """
    Build two OD-aligned matrices from the LAST `days_window` days of `graphs`:

        X_normal   : first  (days_window - days_disaster) days
        X_disaster : last   days_disaster days

    Both matrices share the SAME OD-pair row ordering because the low-activity
    filter is applied jointly on the full window before splitting.  This
    routine follows the last-N-days trailing window convention (not a fixed
    28-day-normal + 14-day-disaster split); it also skips the weekly
    segment-averaging since normal here is exactly 7 days.

    Returns
    -------
    X_normal   : ndarray [n_OD × (days_normal   * SLOTS_ACTIVE)]
    X_disaster : ndarray [n_OD × (days_disaster * SLOTS_ACTIVE)]
    mapping    : list of (origin_id, dest_id) aligned to the rows of both.
    """
    days_normal = days_window - days_disaster

    window_graphs = periodic_trim(
        graphs[-SLOT_PER_DAY * days_window:],
        cycle_len=SLOT_PER_DAY,
        trim_start=SLOT_TRIM_START, trim_end=SLOT_TRIM_END,
    )
    X_full, mapping = graphs_to_2d_matrix(window_graphs)

    # Joint filter on the full window (same threshold rule as build_city_matrices).
    threshold = SLOTS_ACTIVE * days_window * filter_factor
    X_full, _, mapping = filter_inactive_locations_2d(
        X_full, X_full, mapping, threshold,
    )

    n_nor_col  = days_normal * SLOTS_ACTIVE
    X_normal   = X_full[:, :n_nor_col]
    X_disaster = X_full[:, n_nor_col:]

    print(f"  Paired matrices: normal={X_normal.shape}, "
          f"disaster={X_disaster.shape}, n_OD={X_full.shape[0]}")
    print_matrix_diagnostics(X_normal,   mapping, label='normal half')
    print_matrix_diagnostics(X_disaster, mapping, label='disaster half')
    return X_normal, X_disaster, mapping


def decompose_city_paired(X_normal, X_disaster, n_behaviors, l1_reg=0.0):
    """
    Two INDEPENDENT NMFs (one per period), then match disaster components to
    normal components by spatial (H-row) cosine similarity via optimal
    assignment.

    Why independent (not warm-start): a warm-start NMF seeded with H_normal
    keeps H's *direction* frozen and only rescales it — so H_disaster[k] ends
    up proportional to H_normal[k], the distance distribution is identical, and
    α_disaster ≡ α_normal (Δα ≡ 0, no signal).  Decomposing each period on its
    own lets the disaster genuinely re-learn its spatial patterns; we then
    realign component indices by spatial cosine so component k means the same
    behaviour in both periods (the match cosine is a real 0–1 diagnostic).

    Both NMFs are normalised with normalize_nmf_components so W columns have
    unit L2 norm and H rows carry the magnitude.

    Returns
    -------
    (W_normal,   H_normal,   weights_normal,
     W_disaster, H_disaster, weights_disaster,   # disaster reordered to match normal
     match_cosine)                                # [k] cosine of each matched pair
    """
    from scipy.optimize import linear_sum_assignment

    # 1. Two independent cold-start NMFs.
    print("  NMF on normal half …")
    Wn_raw, Hn_raw = decompose_mobility_patterns(
        X_normal.T, n_behaviors=n_behaviors, l1_reg=l1_reg,
    )
    W_normal, H_normal, weights_normal = normalize_nmf_components(Wn_raw, Hn_raw)

    print("  NMF on disaster half (independent cold start) …")
    Wd_raw, Hd_raw = decompose_mobility_patterns(
        X_disaster.T, n_behaviors=n_behaviors, l1_reg=l1_reg,
    )
    W_disaster, H_disaster, weights_disaster = normalize_nmf_components(Wd_raw, Hd_raw)

    # 2. Spatial cosine between every (normal, disaster) H-row pair.
    Hn_u = H_normal   / (np.linalg.norm(H_normal,   axis=1, keepdims=True) + 1e-12)
    Hd_u = H_disaster / (np.linalg.norm(H_disaster, axis=1, keepdims=True) + 1e-12)
    cos = Hn_u @ Hd_u.T                       # [k_normal × k_disaster] ∈ [0, 1]

    # 3. Optimal one-to-one assignment maximising total spatial similarity,
    #    then reorder the disaster factors so column/row k aligns with normal k.
    row, col = linear_sum_assignment(-cos)    # row = arange(k) for square cos
    W_disaster       = W_disaster[:, col]
    H_disaster       = H_disaster[col, :]
    weights_disaster = weights_disaster[col]
    match_cosine     = cos[row, col]          # match_cosine[k] for normal comp k

    order_n = np.argsort(weights_normal)[::-1]
    print("  Normal-half importance: "
          + ", ".join(f"[{i}]={weights_normal[i]:.1f}" for i in order_n))
    print("  Matched disaster importance (normal-index order): "
          + ", ".join(f"[{i}]={weights_disaster[i]:.1f}" for i in range(n_behaviors)))
    print("  Match cosine (normal↔disaster spatial): "
          + ", ".join(f"[{i}]={match_cosine[i]:.2f}" for i in range(n_behaviors)))
    return (W_normal,   H_normal,   weights_normal,
            W_disaster, H_disaster, weights_disaster,
            match_cosine)
