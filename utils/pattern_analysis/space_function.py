"""
Origin × Destination functional cross-tabulation of NMF components (paper Fig. 9).

Given a single-NMF spatial factor H [k × OD] and a block-group → functional-
category lookup (from the EPA SLD space-function data), this aggregates each
component's OD flows into a [category × category] matrix whose cell (a, b) is
the share of that component's flow going FROM an origin of function a TO a
destination of function b.  One such matrix per component reproduces the
origin/destination land-use heatmaps of the reference paper's Figure 9.

Labelling
---------
SOFT labels (production since 2026-08-04): each block group keeps its full
TF-IWF category-share vector, and every flow is distributed by the outer
product of its endpoints' shares, M_k = Sᵀ F_k S (S = [n_BG × C] share
matrix).  There is no Mix bucket and no dominant-label threshold.  (The
retired HARD-label pair — one-hot S via `dominant_category`, with sub-threshold
block groups pooled into Mix — silently dropped most of the flow through the
Mix cut; recover it from the full-code-2026-08-05 tag if ever needed.)

Conventions
-----------
- Matrices are DIRECTED (origin rows, destination cols; asymmetric).
- Each component is normalised independently to proportions (its kept cells
  sum to 1), so only within-component structure is comparable across cells.
- Endpoints absent from the land-use table contribute zero weight.

Functions
---------
build_flow_poi_feature(mapping, landuse_df, flow_scale, ...) -> [OD × C] flow-POI features
share_lookup_from_landuse(landuse_df, categories)            -> dict{aggr_id: share vector}
build_od_function_matrix_soft(H, mapping, share_lookup, categories) -> (M[k×C×C], retained[k])
"""
import numpy as np
import pandas as pd


def build_flow_poi_feature(mapping, landuse_df, flow_scale, mode='outer',
                           id_col='aggr_id', categories=None):
    """
    Per-flow POI feature matrix Y for context-aware NMF (shared-factor /
    Chen 2018).  The spatial unit is the OD flow itself, so each flow gets ONE
    feature built jointly from its two endpoints.

    Each OD flow j = (origin o, destination d) is described by its endpoints'
    TF-IWF land-use SHARE vectors s_o, s_d (the `share_<cat>` columns of
    landuse_df, each a C-vector summing to 1).  The two are combined into a
    single direction, L2-normalised, then scaled by flow_scale[j] so that
    ‖Y[j,:]‖ = flow_scale[j].

    Pass flow_scale = ‖X_fit[:,j]‖ (the flow's volume over the NMF fit segment)
    so the feature CO-SCALES with H's flow-carrying columns: a volume-free
    share direction would otherwise fight the magnitude H carries (a single
    global G cannot rescale per flow), see docs/technical_notes §3.2/§3.3.

    mode
        'outer' — vec(s_o ⊗ s_d), the C² joint origin×destination type
                  (not separable into an origin term + a destination term;
                  this is the "don't split the endpoints" feature)
        'sum'   — s_o + s_d, the C-dim combined type (lower-dim, loses the
                  joint structure and which-end information)

    A flow is CONSTRAINED only if BOTH endpoints have a non-zero share vector
    (present in landuse_df and not 'Unknown' — Unknown block groups have
    all-zero shares).  Otherwise its Y row is all-zero and it is excluded from
    the context term (mask = False), matching the O×D cross-tab convention of
    dropping flows with an Unknown/missing endpoint.

    Returns
    -------
    Y      : ndarray [n_OD × C']   C' = C² ('outer') or C ('sum')
    labels : list[str]             feature column labels
    mask   : ndarray [n_OD] bool   True where the flow carries a constraint
    """
    if categories is None:
        share_cols = [c for c in landuse_df.columns if c.startswith('share_')]
        categories = [c[len('share_'):] for c in share_cols]
    else:
        share_cols = [f'share_{c}' for c in categories]
    C = len(categories)

    shares = {str(i): np.asarray(v, dtype=float)
              for i, v in zip(landuse_df[id_col].astype(str),
                              landuse_df[share_cols].to_numpy(dtype=float))}

    flow_scale = np.asarray(flow_scale, dtype=float)
    n  = len(mapping)
    Cp = C * C if mode == 'outer' else C
    Y    = np.zeros((n, Cp))
    mask = np.zeros(n, dtype=bool)

    for j, (o, d) in enumerate(mapping):
        s_o = shares.get(str(o))
        s_d = shares.get(str(d))
        if s_o is None or s_d is None or s_o.sum() <= 0 or s_d.sum() <= 0:
            continue                                   # Unknown / missing endpoint
        if mode == 'outer':
            d_vec = np.outer(s_o, s_d).ravel()
        elif mode == 'sum':
            d_vec = s_o + s_d
        else:
            raise ValueError(f"mode must be 'outer' or 'sum', got '{mode}'")
        nrm = np.linalg.norm(d_vec)
        if nrm <= 0:
            continue
        Y[j]    = (d_vec / nrm) * flow_scale[j]         # ‖Y[j,:]‖ = flow_scale[j]
        mask[j] = True

    labels = ([f'{a}*{b}' for a in categories for b in categories]
              if mode == 'outer' else list(categories))
    return Y, labels, mask


def share_lookup_from_landuse(landuse_df, categories, id_col='aggr_id'):
    """Map each block group's id to its CONTINUOUS TF-IWF share vector (the
    share_<cat> columns of classify_dominant_function's output), for the soft
    functional aggregation.  Block groups with no land-use mass (all-zero
    shares) are omitted, so their flows drop out by zero weight instead of
    contributing a fake uniform profile."""
    cols = [f'share_{c}' for c in categories]
    out = {}
    for _, row in landuse_df.iterrows():
        v = row[cols].to_numpy(dtype=float)
        s = v.sum()
        if s > 0:
            out[str(row[id_col])] = v / s
    return out


def build_od_function_matrix_soft(H, mapping, share_lookup, categories):
    """Soft O x D functional cross-tab: every endpoint contributes
    its full TF-IWF share VECTOR instead of one hard dominant label.

    Component i's cross-tab is the flow-weighted expected origin×destination
    profile

        M[i] = Σ_j H[i,j] · outer(P_origin(j), P_dest(j)) / Σ_j H[i,j]

    over the OD pairs whose BOTH endpoints are in `share_lookup` (an endpoint
    with no land-use data drops the pair, tracked by `retained` exactly as the
    hard version tracks its Unknowns).  Because every share vector already sums
    to 1 over `categories`, there is no Mix bucket and nothing to cut later:
    the marginals of M[i] ARE the component's expected functional exposure.

    The outer product treats a trip's origin and destination profiles as
    independent within the pair — the per-pair joint composition is not
    observed, and the marginals (all the downstream features read) are exact
    regardless.

    Returns (M [k × C × C], retained [k]) like the hard version.
    """
    H = np.asarray(H, dtype=float)
    k, C = H.shape[0], len(categories)
    P_o = np.full((len(mapping), C), np.nan)
    P_d = np.full((len(mapping), C), np.nan)
    for j, (o, d) in enumerate(mapping):
        po = share_lookup.get(str(o))
        pd_ = share_lookup.get(str(d))
        if po is not None and pd_ is not None:
            P_o[j], P_d[j] = po, pd_
    keep = np.isfinite(P_o).all(axis=1)
    Po_k, Pd_k = np.nan_to_num(P_o[keep]), np.nan_to_num(P_d[keep])

    M = np.zeros((k, C, C))
    retained = np.zeros(k)
    for comp in range(k):
        row = H[comp]
        total = row.sum()
        w = row[keep]
        M[comp] = np.einsum('j,jc,jd->cd', w, Po_k, Pd_k)
        kept = M[comp].sum()
        retained[comp] = (kept / total) if total > 0 else 0.0
        if kept > 0:
            M[comp] /= kept
    return M, retained
