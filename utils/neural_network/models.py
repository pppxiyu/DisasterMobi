"""
Neural network models and training utilities.

Classes
-------
EarlyStopper              – tracks validation loss and signals when to stop
GraphGRUCell              – single GRU step with TAGConv graph convolutions
GraphGRUPredictor         – wraps GraphGRUCell into a sequence-to-one predictor
TempoPhysicsGraphGRUPredictor – adds a learnable physics (Gamma) gate to GraphGRUPredictor
"""
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import TAGConv


# ── Utilities ─────────────────────────────────────────────────────────────────

class EarlyStopper:
    """Triggers when validation loss fails to improve for `patience` steps."""

    def __init__(self, patience=5, min_delta=1e-5):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.min_val_loss = np.inf
        self.stop        = False

    def stopper(self, val_loss):
        if val_loss < (self.min_val_loss - self.min_delta):
            self.min_val_loss = val_loss
            self.counter      = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ── Building block ────────────────────────────────────────────────────────────

class GraphGRUCell(nn.Module):
    """
    One GRU step using TAGConv (Topology Adaptive Graph Convolutional) layers
    for the update/reset/candidate gates.

    Parameters
    ----------
    in_channels  : number of node input features
    out_channels : hidden state dimension
    k_hops       : number of hops in TAGConv (controls receptive field)
    """

    def __init__(self, in_channels, out_channels, k_hops=3):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        combined          = in_channels + out_channels
        self.conv_z = TAGConv(combined, out_channels, K=k_hops)
        self.conv_r = TAGConv(combined, out_channels, K=k_hops)
        self.conv_h = TAGConv(combined, out_channels, K=k_hops)

    def forward(self, x, edge_index, edge_weight=None, h=None):
        if h is None:
            h = torch.zeros(x.shape[0], self.out_channels, device=x.device)
        x_h    = torch.cat([x, h], dim=-1)
        z      = torch.sigmoid(self.conv_z(x_h, edge_index, edge_weight))
        r      = torch.sigmoid(self.conv_r(x_h, edge_index, edge_weight))
        x_h2   = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.conv_h(x_h2, edge_index, edge_weight))
        return z * h + (1 - z) * h_tilde


# ── Predictors ────────────────────────────────────────────────────────────────

class GraphGRUPredictor(nn.Module):
    """
    Processes a sequence of graph snapshots with GraphGRUCell and maps the
    final hidden state to per-node predictions via an MLP readout.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, k_hops):
        super().__init__()
        self.gru = GraphGRUCell(in_channels, hidden_channels, k_hops=k_hops)
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x_seq_batch):
        h = None
        for snapshot in x_seq_batch:
            h = self.gru(x=snapshot.x, edge_index=snapshot.edge_index,
                          edge_weight=snapshot.edge_attr, h=h)
        return self.readout(h)


class TempoPhysicsGraphGRUPredictor(nn.Module):
    """
    Extends GraphGRUPredictor with a physics-informed Gamma gate.

    The Gamma model (fitted on the observed flow drop) predicts a multiplicative
    reduction factor. A small MLP (physics_gate) learns to re-scale that factor
    based on elapsed disaster time, allowing the model to trust or discount the
    physics signal at different stages of the disaster.

    The last feature in x is expected to be 'disaster_time'; it is split off
    before the GRU so the GRU only sees the remaining features.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, k_hops,
                 gamma_params, target_mean=0.0, target_std=1.0):
        super().__init__()
        # GRU uses all features except disaster_time (last column)
        self.gru = GraphGRUCell(in_channels - 1, hidden_channels, k_hops=k_hops)
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        # Physics buffers (move to GPU automatically with the model)
        for name, val in gamma_params.items():
            self.register_buffer(name, torch.tensor(val))
        self.register_buffer('target_mean', torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer('target_std',  torch.tensor(target_std,  dtype=torch.float32))

        # Learnable gate: maps normalised disaster-time → scaling multiplier
        self.physics_gate = nn.Sequential(
            nn.Linear(1, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )
        self._init_physics_gate()

    def _init_physics_gate(self):
        """Bias initialisation: creates a decreasing signal through neuron 0."""
        for layer in self.physics_gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.physics_gate[0].weight[0, 0] = -2.0
            self.physics_gate[0].bias[0]       = 1.0
            self.physics_gate[2].weight[0, :]  = 0.0
            self.physics_gate[2].weight[0, 0]  = 1.0
            self.physics_gate[2].bias[0]        = 0.0
            self.physics_gate[4].weight[0, 0]  = 1.0
            self.physics_gate[4].bias[0]        = 0.0

    def torch_gamma_predict(self, t):
        """GPU-native Gamma PDF evaluation using log-gamma for numerical stability."""
        shifted = t - self.t0
        mask    = (shifted > 0).float()
        safe_t  = shifted * mask
        log_t1  = (self.alpha - 1) * torch.log(safe_t + 1e-9)
        log_t2  = -safe_t / self.beta
        log_den = torch.lgamma(self.alpha) + self.alpha * torch.log(self.beta)
        return self.A * torch.exp(log_t1 + log_t2 - log_den) * mask

    def forward(self, x_seq_batch):
        h = None
        for snapshot in x_seq_batch:
            gru_feat = snapshot.x[:, :-1]   # drop disaster_time
            h = self.gru(x=gru_feat, edge_index=snapshot.edge_index,
                          edge_weight=snapshot.edge_attr, h=h)

        # Physics gate uses the TARGET step's disaster_time (last snap + 1)
        last_phys    = x_seq_batch[-1].x[:, -1:]
        target_phys  = last_phys + 1
        gamma_out    = self.torch_gamma_predict(target_phys)

        norm_time    = target_phys / 20.0
        raw_gate     = self.physics_gate(norm_time)
        gate         = 1.0 + torch.tanh(raw_gate)  # gate ∈ (0, 2)
        gated_gamma  = gamma_out * gate

        # Denorm → apply physics → renorm
        gru_out      = self.readout(h)
        raw_pred     = gru_out * self.target_std + self.target_mean
        raw_reduced  = raw_pred * (1 + gated_gamma)
        return (raw_reduced - self.target_mean) / self.target_std

