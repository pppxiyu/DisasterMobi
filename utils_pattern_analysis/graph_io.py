"""
Utilities for loading raw graph data and converting it into matrix/tensor formats
suitable for NMF and Tucker decomposition.
"""
import pickle
import numpy as np


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
    """Returns total edge flow for each graph in the list."""
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


# ── 3-D (Tucker) conversion ───────────────────────────────────────────────────

def graphs_to_3d_tensor(graph_list):
    """
    Converts a list of DiGraphs into a 3-D NumPy tensor (Origin × Dest × Time).

    Returns
    -------
    tensor      : ndarray of shape (N, N, T)
    idx_to_node : dict mapping tensor index → node name
    """
    all_nodes = sorted(set().union(*(G.nodes() for G in graph_list)))
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    N, T = len(all_nodes), len(graph_list)
    tensor = np.zeros((N, N, T))
    for t, G in enumerate(graph_list):
        for u, v, d in G.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                tensor[node_to_idx[u], node_to_idx[v], t] = d.get('flow', 1)

    return tensor, idx_to_node


def filter_inactive_locations(tensor1, tensor2, spatial_mapping, threshold=10):
    """
    Removes zones with low total activity (inflow + outflow in tensor1).
    The same zone mask is applied to tensor2 to keep spatial alignment.

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


def calculate_segment_average(tensor, segment_len):
    """
    Splits the time axis (axis 2) into segments and averages them.
    Input:  (I, J, T)
    Output: (I, J, segment_len)

    segment_len is required (no default) — callers must pass SLOTS_ACTIVE * 7
    so the value stays correct when switching between 2h and 3h resolution.
    """
    I, J, T = tensor.shape
    if T % segment_len != 0:
        n = T // segment_len
        tensor = tensor[:, :, :n * segment_len]
        T = tensor.shape[2]
    n_seg = T // segment_len
    return np.mean(tensor.reshape(I, J, n_seg, segment_len), axis=2)
