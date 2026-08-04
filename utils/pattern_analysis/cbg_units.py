"""
Census block groups as the unit of analysis, in place of NMF components.

The rest of the pipeline never asks what a "unit" is.  It asks for

    W        [n_time x n_units]   the unit's activity over time
    H        [n_units x n_OD]     the unit's loading on each OD pair
    weights  [n_units]            the unit's overall importance

and every feature, correlation, resilience and cross-city step is written
against those three objects.  So dropping the decomposition does not mean
rewriting the analysis: it means supplying the same three objects with the
block group, rather than a latent component, playing the part of a unit.

    W[:, i]  = all flow TOUCHING block group i per time slot (its trips as an
               origin plus its trips as a destination — the same merge the
               functional shares use, so "activity of unit i" means one thing
               throughout).  A trip that starts and ends inside i counts twice,
               which is what "touching" means and keeps W a linear map of X.
    H[i, p]  = the total flow on OD pair p if i is an endpoint of p, else 0.
               The component analogue weights each pair by how strongly the
               component loads on it; a block group either participates in a
               pair or it does not, so the loading is the pair's own volume.
    weights  = the unit's total activity over the window.

What CANNOT carry over is anything that only exists because a component is a
learned object: there is no basis to reconstruct (so no reconstruction error,
no rank cross-validation) and no per-unit temporal signature to plot against
the others.  The features that describe a unit's context DO carry over, but
two of them change meaning and are supplied here rather than derived from H:
a block group HAS a land use and HAS a median income, so those are read off
the unit itself instead of being averaged over the endpoints of its trips.
"""
import numpy as np
import pandas as pd


def build_cbg_units(X_all, mapping):
    """Build the (W, H, weights) triple with block groups as units.

    Parameters
    ----------
    X_all   : ndarray [n_OD x n_time]   the city's flow matrix
    mapping : list of (origin_id, dest_id) aligned to X_all's rows

    Returns
    -------
    W       : ndarray [n_time x n_units]
    H       : ndarray [n_units x n_OD]
    nodes   : list  the unit ids (aggr_id), in W/H column/row order
    weights : ndarray [n_units]
    """
    X_all = np.asarray(X_all, dtype=float)
    nodes = sorted({n for pair in mapping for n in pair})
    idx = {n: i for i, n in enumerate(nodes)}

    # Incidence: 1 per endpoint, so an internal pair (o == d) contributes 2 —
    # the flow does touch that unit twice.
    inc = np.zeros((len(nodes), X_all.shape[0]))
    for p, (o, d) in enumerate(mapping):
        inc[idx[o], p] += 1.0
        inc[idx[d], p] += 1.0

    W = (inc @ X_all).T                       # [n_time x n_units]
    H = inc * X_all.sum(axis=1)[None, :]      # [n_units x n_OD]
    return W, H, nodes, W.sum(axis=0)


def cbg_functional_features(nodes, landuse, categories, key='aggr_id'):
    """Each unit's OWN land-use composition, in the column names the component
    path produces.

    A block group's land use has no direction, but the downstream code merges
    the two directional shares into func_<cat> = share_from_<cat> +
    share_to_<cat> in eight places.  Splitting the own share evenly across the
    two makes that merge reproduce the share exactly, so not one of those eight
    call sites has to know which unit definition is in play.

    A unit missing from the land-use table gets zeros (it contributes no
    functional signal rather than dropping the whole row).
    """
    lu = landuse.set_index(key) if key in landuse.columns else landuse
    out = {}
    for c in categories:
        col = f'share_{c}'
        s = (lu[col].reindex(nodes).to_numpy(dtype=float) / 2.0
             if col in lu.columns else np.zeros(len(nodes)))
        s = np.nan_to_num(s)
        out[f'share_from_{c}'] = s
        out[f'share_to_{c}'] = s
    return pd.DataFrame(out, index=pd.RangeIndex(len(nodes), name='component'))


def cbg_socioeconomic_features(nodes, income_by_aggr, modes,
                               name='median_income'):
    """Each unit's OWN ACS median household income.

    The component path reports one column per endpoint mode (origin side,
    destination side, both), because a component's income is an average over
    the endpoints of its flows and the sides can differ.  A block group IS an
    endpoint, so every mode collapses to the same number; all the columns are
    still emitted so the feature names — and `median_income_combined`, which
    the cross-city step selects by name — match the component path exactly.
    """
    v = np.array([income_by_aggr.get(n, np.nan) for n in nodes], dtype=float)
    return pd.DataFrame({f'{name}_{m}': v for m in modes},
                        index=pd.RangeIndex(len(nodes), name='component'))
