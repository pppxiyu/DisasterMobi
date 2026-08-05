"""
Utilities for loading raw graph data and converting it into matrix/tensor formats
suitable for NMF decomposition.
"""
import pickle
import numpy as np
import pandas as pd


# ── Loading ───────────────────────────────────────────────────────────────────

def load_graphs(path):
    with open(path, 'rb') as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs from {path}")
    return graphs


def load_graphs_trimmed(path, analysis_days, slot_per_day, label=''):
    """
    Load a graph pkl, validate it strictly, and trim it to the analysis range.

    All checks raise.
      - analysis_days must be set (None is rejected).
      - The pkl must carry interval_duration metadata that matches slot_per_day.
      - The graph count must be divisible by slot_per_day (catches a 3h file
        loaded under a 2h config even without metadata).
      - The pkl must contain at least analysis_days days.

    Parameters
    ----------
    path          : pkl path.
    analysis_days : number of days kept, counted from the start.  Required.
    slot_per_day  : slots per day (= SLOT_PER_DAY from config).
    label         : city name for error messages.

    Returns the first analysis_days * slot_per_day graphs.
    """
    label = label or path
    if analysis_days is None:
        raise ValueError(f"{label}: analysis_days must be set, got None.")

    graphs = load_graphs(path)

    expected_interval = f'{24 // slot_per_day}h'
    actual_interval = graphs[0].graph.get('interval_duration')
    if actual_interval is None:
        raise ValueError(
            f"{label}: the pkl lacks interval_duration metadata. Rebuild it "
            f"with the current pipeline or verify it matches {expected_interval}."
        )
    if actual_interval != expected_interval:
        raise ValueError(
            f"{label}: pkl contains {actual_interval} graphs but config expects "
            f"{expected_interval} (slot_per_day={slot_per_day})."
        )
    if len(graphs) % slot_per_day != 0:
        raise ValueError(
            f"{label}: {len(graphs)} graphs is not divisible by "
            f"slot_per_day={slot_per_day}. The pkl was likely built with a "
            f"different time resolution."
        )

    need = analysis_days * slot_per_day
    if len(graphs) < need:
        raise ValueError(
            f"{label}: analysis_days={analysis_days} needs {need} graphs but "
            f"the pkl has only {len(graphs)} "
            f"({len(graphs) // slot_per_day} days)."
        )
    return graphs[:need]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def periodic_trim(data, cycle_len, trim_start, trim_end):
    """
    Removes overnight slots from every daily cycle.
    Keeps indices where (i % cycle_len) in [trim_start, cycle_len - trim_end).

    Parameters
    ----------
    cycle_len  : total slots per day  (= SLOT_PER_DAY from config)
    trim_start : slots to drop at the start of each day  (= SLOT_TRIM_START)
    trim_end   : slots to drop at the end   of each day  (= SLOT_TRIM_END)

    All three parameters are required — no defaults — so that switching between
    2h and 3h resolution cannot silently use the wrong cycle length.
    """
    return [
        val for i, val in enumerate(data)
        if (i % cycle_len) >= trim_start and (i % cycle_len) < (cycle_len - trim_end)
    ]


def calculate_total_flows(graphs):
    """Returns total edge flow for each graph in the list.
    Only used by utils/neural_network/temporal_decay.py."""
    return [sum(d.get('flow', 0) for _, _, d in G.edges(data=True)) for G in graphs]


# ── Diagnostics ──────────────────────────────────────────────────────────────

def print_matrix_diagnostics(X, mapping, label='', thresholds=(1, 5, 10)):
    """
    Print a summary of a filtered flow matrix or tensor for sanity-checking.

    Parameters
    ----------
    X        : ndarray
        2-D [OD_pairs × time] or 3-D [N × N × time].  Values are flow counts.
    mapping  : list of tuples  |  dict {int: node}
        For 2-D: list of (origin, dest) edge tuples (length = X.shape[0]).
        For 3-D: dict mapping tensor index → node name.
    label    : str
        City / period label printed in the header.
    thresholds : tuple of float
        Flow values whose share will be reported as potential concerns.
    """
    print(f"\n  Data diagnostics : {label}")

    # ── spatial coverage ──────────────────────────────────────────────────────
    if X.ndim == 2:
        n_od   = X.shape[0]
        n_time = X.shape[1]
        if isinstance(mapping, dict):
            n_tracts = len(mapping)
        else:
            n_tracts = len({n for edge in mapping for n in edge})
        max_od   = n_tracts * (n_tracts - 1)   # directed, no self-loops
        od_label = "OD pairs"
    else:  # 3-D tensor
        n_tracts = X.shape[0]
        n_time   = X.shape[2]
        n_od     = int(np.sum(np.any(X > 0, axis=2)))
        max_od   = n_tracts * (n_tracts - 1)
        od_label = "Active OD pairs"
    vals = X.flatten()
    print(f"    Tracts (nodes)   : {n_tracts}")
    print(f"    {od_label:<16} : {n_od:,}  "
          f"(of {max_od:,} possible = {100*n_od/max_od:.1f}%)")
    print(f"    Time steps       : {n_time}")

    # ── zero fraction ────────────────────────────────────────────────────────
    n_total    = vals.size
    n_zero     = int(np.sum(vals == 0))
    pct_zero   = 100 * n_zero / n_total
    print(f"    Zeros            : {n_zero:,} / {n_total:,}  ({pct_zero:.1f}%)")

    # ── nonzero statistics ────────────────────────────────────────────────────
    nz = vals[vals > 0]
    if nz.size == 0:
        print("    (no nonzero values)")
        return

    print(f"    Flow stats (nonzero only, n={nz.size:,}):")
    print(f"        min={nz.min():.3f}, median={np.median(nz):.3f}, "
          f"mean={nz.mean():.3f}, max={nz.max():.3f}")

    # ── concern thresholds ────────────────────────────────────────────────────
    print(f"    Entries below threshold (all cells incl. zeros):")
    for thr in thresholds:
        n_below = int(np.sum(vals < thr))
        print(f"        < {thr:g}  {n_below:,} / {n_total:,}  "
              f"({100*n_below/n_total:.1f}%)")


# ── 2-D (NMF) conversion ─────────────────────────────────────────────────────

def graphs_to_2d_matrix(graph_list):
    """
    Converts a list of DiGraphs into a 2-D NumPy array (Flows × Time).

    Returns
    -------
    matrix       : ndarray of shape (num_edges, T)
    edge_names   : list of (origin_name, dest_name) tuples aligned to matrix rows
    """
    all_nodes = sorted(set().union(*(G.nodes() for G in graph_list)))
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    flow_data = {}
    for t, G in enumerate(graph_list):
        for u, v, d in G.edges(data=True):
            eid = (node_to_idx[u], node_to_idx[v])
            flow_val = d.get('flow', 0)
            if eid not in flow_data:
                flow_data[eid] = [0] * t
            flow_data[eid].append(flow_val)
        for eid in flow_data:
            if len(flow_data[eid]) < t + 1:
                flow_data[eid].append(0)

    eids = list(flow_data.keys())
    matrix = np.array([flow_data[e] for e in eids])
    edge_names = [(idx_to_node[u], idx_to_node[v]) for u, v in eids]
    return matrix, edge_names


def filter_inactive_locations_2d(X, X_impact, edge_names, threshold=10):
    """
    Reconstructs 3-D tensors to apply node-level activity filtering,
    then flattens back to 2-D. Both tensors are filtered with the same mask.
    """
    unique_nodes = sorted({n for edge in edge_names for n in edge})
    node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    N = len(unique_nodes)

    X_3d        = np.zeros((N, N, X.shape[1]))
    X_impact_3d = np.zeros((N, N, X_impact.shape[1]))
    for i, (u, v) in enumerate(edge_names):
        ui, vi = node_to_idx[u], node_to_idx[v]
        X_3d[ui, vi, :] = X[i, :]
        if i < len(X_impact):
            X_impact_3d[ui, vi, :] = X_impact[i, :]

    X_3d_f, X_impact_3d_f, new_map = filter_inactive_locations(
        X_3d, X_impact_3d, idx_to_node, threshold
    )

    rows_X, rows_Xi, filt_edges = [], [], []
    nf = X_3d_f.shape[0]
    for u in range(nf):
        for v in range(nf):
            if np.any(X_3d_f[u, v, :] > 0) or np.any(X_impact_3d_f[u, v, :] > 0):
                filt_edges.append((new_map[u], new_map[v]))
                rows_X.append(X_3d_f[u, v, :])
                rows_Xi.append(X_impact_3d_f[u, v, :])

    return np.array(rows_X), np.array(rows_Xi), filt_edges


def filter_inactive_locations(tensor1, tensor2, spatial_mapping, threshold=10):
    """
    Removes zones with low total activity (inflow + outflow in tensor1).
    The same zone mask is applied to tensor2 to keep spatial alignment.
    Operates on 3-D (O × D × T) tensors; the production 2-D path reaches it
    through filter_inactive_locations_2d, which rebuilds the tensor inline.

    Returns filtered tensor1, tensor2, and the updated index→node mapping.
    """
    outflow = np.sum(tensor1, axis=(1, 2))
    inflow  = np.sum(tensor1, axis=(0, 2))
    keep    = np.where(outflow + inflow >= threshold)[0]

    X1 = tensor1[keep, :, :][:, keep, :]
    X2 = tensor2[keep, :, :][:, keep, :]
    new_map = {i: spatial_mapping[old] for i, old in enumerate(keep)}

    print(f"Locations: {tensor1.shape[0]} → {X1.shape[0]} "
          f"(removed {tensor1.shape[0] - X1.shape[0]} below threshold)")
    return X1, X2, new_map


# ── Geometry ──────────────────────────────────────────────────────────────────

def build_distance_array(mapping, gdf, id_col='aggr_id'):
    """
    OD-pair centroid distances (km) aligned to the NMF mapping / H-column order.

    Centroids are taken in EPSG:3857 (Web Mercator); the distance is the
    straight-line centroid separation / 1000.  Web Mercator inflates absolute
    distance with latitude, but the factor is ~constant within one city, so
    cross-flow / cross-component comparisons are unaffected.  Shared by the
    distance-decay and the NMF spatial-feature analyses so both use one definition.
    """
    proj   = gdf.to_crs(epsg=3857)
    cents  = proj.geometry.centroid
    coords = pd.DataFrame({
        'cx': cents.x.values,
        'cy': cents.y.values,
    }, index=gdf[id_col].astype(str).values)

    origins = np.array([o for o, _ in mapping])
    dests   = np.array([d for _, d in mapping])
    missing = (~pd.Index(origins).isin(coords.index)) | \
              (~pd.Index(dests).isin(coords.index))
    if missing.any():
        raise KeyError(
            f"{int(missing.sum())} OD pairs reference aggr_id values not "
            f"found in the geo-dataframe."
        )

    co = coords.loc[origins][['cx', 'cy']].to_numpy()
    cd = coords.loc[dests  ][['cx', 'cy']].to_numpy()
    return np.linalg.norm(co - cd, axis=1) / 1000.0


def build_income_array(mapping, income_by_aggr, mode='combined'):
    """
    Per-OD-flow median household income (USD) aligned to the NMF mapping / H-column
    order — the socioeconomic analogue of build_distance_array.

    mapping        : list of (origin_aggr_id, dest_aggr_id), one per flow (H column).
    income_by_aggr : dict / Series  aggr_id -> block-group median household income
                     (NaN where the Census suppressed it).
    mode           : 'combined' -> nan-aware mean of the flow's origin & destination
                     income; 'origin' -> the origin block group's income only.

    Returns a float array (len = n_OD); a flow is NaN where its required endpoint
    income(s) are missing (both endpoints missing in 'combined', the origin
    missing in 'origin').
    """
    if not isinstance(income_by_aggr, dict):
        income_by_aggr = dict(income_by_aggr)       # Series -> dict for fast lookups
    o = np.array([income_by_aggr.get(orig, np.nan) for orig, _ in mapping],
                 dtype=float)
    if mode == 'origin':
        return o
    if mode != 'combined':
        raise ValueError(f"mode must be 'combined' or 'origin', got {mode!r}")
    d = np.array([income_by_aggr.get(dest, np.nan) for _, dest in mapping],
                 dtype=float)
    # nan-aware mean over the available endpoints (1 or 2); all-NaN flow -> NaN.
    stacked = np.vstack([o, d])
    valid = ~np.isnan(stacked)
    cnt = valid.sum(axis=0)
    tot = np.where(valid, stacked, 0.0).sum(axis=0)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
