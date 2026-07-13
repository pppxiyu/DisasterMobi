"""
Graph preprocessing for the neural network pipeline.

Converts raw NetworkX DiGraphs into PyTorch Geometric Data objects representing
the Line Graph L(G) — where original edges become nodes connected if they share
an endpoint in G.

Functions
---------
transform_to_line_graph_data     – DiGraph → PyG Data (Line Graph)
set_graph_targets                – attaches flow targets to each PyG Data object
get_original_index               – maps post-split index to full-series index
add_disaster_time_feature        – stamps each graph with a monotonic disaster-time counter
"""
import networkx as nx
import torch
from torch_geometric.utils import from_networkx


def transform_to_line_graph_data(G, feature_keys=None):
    """
    Converts a NetworkX DiGraph G into a PyG Data object for its Line Graph L(G).

    In L(G) each node represents an edge of G; two nodes in L(G) are connected
    if the corresponding edges share an endpoint in G.
    The original edge attributes of G become node features in L(G).

    Parameters
    ----------
    G            : nx.DiGraph
    feature_keys : list of attribute names to collect into data.x

    Returns
    -------
    torch_geometric.data.Data
    """
    L = nx.line_graph(G)
    for node in L.nodes():
        L.nodes[node].update(G.edges[node])
    return from_networkx(L, group_node_attrs=feature_keys)


def set_graph_targets(graphs, target_attr='y'):
    """
    Sets the prediction target on each PyG Data object.
    Target = the flow feature (first column of x) cloned as a (N, 1) tensor.
    """
    for data in graphs:
        flow_target = data.x[:, 0].clone().view(-1, 1)
        setattr(data, target_attr, flow_target)
    return graphs


def get_original_index(post_index, total_len, n_days_back, slots_per_day):
    """
    Maps an index within the post-disaster sub-series back to the original
    full time-series index.

    Parameters
    ----------
    post_index   : index within the disaster sub-series (0-based)
    total_len    : total number of graphs in the full series
    n_days_back  : number of days used as the disaster period
    slots_per_day: time slots per day — pass SLOT_PER_DAY from config
                   (8 for 3h data, 12 for 2h data)
    """
    split_offset = slots_per_day * n_days_back
    start_of_post = total_len - split_offset
    return start_of_post + post_index


def add_disaster_time_feature(graphs, start_idx, end_idx):
    """
    Adds a 'disaster_time' edge attribute to every graph.

    Within [start_idx, end_idx], the value counts up from 1.
    Outside that range the value is 0.

    Parameters
    ----------
    graphs    : list of NetworkX graphs (full series)
    start_idx : original-series index where disaster starts
    end_idx   : original-series index where disaster ends
    """
    for i, G in enumerate(graphs):
        val = (i - start_idx + 1) if start_idx <= i <= end_idx else 0
        for _, _, d in G.edges(data=True):
            d['disaster_time'] = val
    return graphs
