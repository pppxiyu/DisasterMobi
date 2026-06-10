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
HARD labels: each block group is assigned its single `dominant_category`, so an
OD pair contributes its whole flow to one (origin_cat, dest_cat) cell.

A SOFT variant would instead distribute each flow by the outer product of the
endpoints' category shares,  M_k = Sᵀ F_k S  (S = [n_BG × C] share matrix);
the hard version is the special case where S is one-hot.  Not implemented here.

Conventions
-----------
- Axis categories = the functional categories + 'Mix'; 'Unknown' (and any OD
  endpoint missing a category) is DROPPED.
- Matrices are DIRECTED (origin rows, destination cols; asymmetric).
- Each component is normalised independently to proportions (its kept cells
  sum to 1), so only within-component structure is comparable across cells.

Functions
---------
category_lookup_from_landuse(landuse_df, ...) -> dict{aggr_id: category}
build_od_function_matrix(H, mapping, cat_lookup, categories) -> (M[k×C×C], retained[k])
od_function_to_dataframe(M, categories, weights) -> tidy long-format DataFrame
"""
import numpy as np
import pandas as pd


def category_lookup_from_landuse(landuse_df, id_col='aggr_id',
                                 cat_col='dominant_category'):
    """Map each block group's id to its dominant functional category."""
    return dict(zip(landuse_df[id_col].astype(str), landuse_df[cat_col].astype(str)))


def build_od_function_matrix(H, mapping, cat_lookup, categories):
    """
    Aggregate each NMF component's OD flows into a [C × C] origin×destination
    functional matrix, normalised per component to proportions.

    Parameters
    ----------
    H          : ndarray [k × OD]   spatial factors (normalised; carries magnitude)
    mapping    : list of (origin_id, dest_id) aligned to the columns of H
    cat_lookup : dict {aggr_id: category}
    categories : list[str]   axis categories (order defines rows/cols).  OD pairs
                 whose origin OR destination category is not in this list are
                 dropped (e.g. 'Unknown' / endpoints absent from the lookup).

    Returns
    -------
    M        : ndarray [k × C × C]  per-component proportions (each slice sums to
               1, unless a component had zero in-category flow).
    retained : ndarray [k]          fraction of each component's TOTAL flow that
               fell inside `categories` (1 - share dropped to Unknown/missing).
    """
    H = np.asarray(H, dtype=float)
    k = H.shape[0]
    C = len(categories)
    idx = {c: i for i, c in enumerate(categories)}

    # Per-OD-column category indices (−1 marks a dropped endpoint).
    oi = np.full(len(mapping), -1, dtype=int)
    di = np.full(len(mapping), -1, dtype=int)
    for j, (o, d) in enumerate(mapping):
        co = cat_lookup.get(str(o))
        cd = cat_lookup.get(str(d))
        if co in idx and cd in idx:
            oi[j], di[j] = idx[co], idx[cd]
    keep = (oi >= 0) & (di >= 0)
    oi_k, di_k = oi[keep], di[keep]

    M = np.zeros((k, C, C))
    retained = np.zeros(k)
    for comp in range(k):
        row = H[comp]
        total = row.sum()
        np.add.at(M[comp], (oi_k, di_k), row[keep])   # accumulate kept flows
        kept = M[comp].sum()
        retained[comp] = (kept / total) if total > 0 else 0.0
        if kept > 0:
            M[comp] /= kept                            # per-component proportion
    return M, retained


def od_function_to_dataframe(M, categories, weights=None):
    """
    Flatten [k × C × C] proportions into a tidy long-format DataFrame with
    columns: component, weight, origin_function, dest_function, proportion.
    """
    rows = []
    for comp in range(M.shape[0]):
        w = float(weights[comp]) if weights is not None else np.nan
        for a, o in enumerate(categories):
            for b, d in enumerate(categories):
                rows.append((comp, w, o, d, M[comp, a, b]))
    return pd.DataFrame(rows, columns=['component', 'weight',
                                       'origin_function', 'dest_function',
                                       'proportion'])
