"""
High-level training and hyperparameter tuning orchestration.

GraphTemporalPred
    Manages dataset splits, model construction, training loop, and test evaluation.
    Handles pre-normalisation of node features and targets.

GraphTemporalTuner
    Wraps GraphTemporalPred in an Optuna study for hyperparameter search.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import optuna

from utils_neural_network.dataset import TemporalGraphDataset, temporal_collate_fn
from utils_neural_network.models import (
    EarlyStopper,
    GraphGRUPredictor,
    TempoPhysicsGraphGRUPredictor,
)


class GraphTemporalPred:
    """
    Orchestrates the full train/val/test pipeline for temporal graph prediction.

    Usage
    -----
    predictor = GraphTemporalPred()
    predictor.build_dataset(graphs, window_size=7)
    predictor.build_model(in_channels=3, hidden_channels=128, out_channels=1,
                          k_hops=2, model_name='basic')
    predictor.train_model()
    preds, targets = predictor.evaluate_test_set()
    """

    def __init__(self, device=None):
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None
        self.model         = None

    # ── Dataset ──────────────────────────────────────────────────────────────

    def _get_global_stats(self, graphs):
        all_x   = torch.cat([g.x for g in graphs], dim=0).to(torch.float32)
        mean    = all_x.mean(dim=0)
        std     = all_x.std(dim=0)
        std[std < 1e-5] = 1.0
        return mean, std

    def build_dataset(self, graphs, window_size, train_ratio=0.7, val_ratio=0.15,
                      target_attr='y', slots_per_day=8, start_slot=0):
        """
        Pre-normalises features and targets, then splits into train/val/test
        subsets aligned to whole-day boundaries to prevent data leakage.
        """
        assert slots_per_day == 8 and start_slot == 0, \
            "Currently only slots_per_day=8 and start_slot=0 are supported."

        # Normalise using train-set statistics only
        train_end   = int(len(graphs) * train_ratio)
        G_MEAN, G_STD = self._get_global_stats(graphs[:train_end])

        for g in graphs:
            g.x = ((g.x.to(torch.float32) - G_MEAN) / G_STD).to(torch.float32)
            y   = getattr(g, target_attr)
            setattr(g, target_attr,
                    ((y.to(torch.float32) - G_MEAN[0]) / G_STD[0]).to(torch.float32))

        full_dataset       = TemporalGraphDataset(graphs, window_size, target_attr)
        full_dataset.g_mean = G_MEAN
        full_dataset.g_std  = G_STD

        # Day-boundary splits
        total_days = len(graphs) // slots_per_day
        train_days = round(total_days * train_ratio)
        val_days   = round(total_days * val_ratio)

        train_end_idx = train_days * slots_per_day
        val_end_idx   = (train_days + val_days) * slots_per_day

        train_idx, val_idx, test_idx = [], [], []
        for i, seq_start in enumerate(full_dataset.valid_indices):
            tgt = seq_start + window_size
            if   tgt < train_end_idx: train_idx.append(i)
            elif tgt < val_end_idx:   val_idx.append(i)
            else:                     test_idx.append(i)

        self.train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        self.val_dataset   = torch.utils.data.Subset(full_dataset, val_idx)
        self.test_dataset  = torch.utils.data.Subset(full_dataset, test_idx)

        print(f"Days — Train: {train_days}, Val: {val_days}, "
              f"Test: {total_days - train_days - val_days}")
        print(f"Samples — Train: {len(self.train_dataset)}, "
              f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────

    def build_model(self, in_channels, hidden_channels, out_channels, k_hops,
                    model_name='basic', **kwargs):
        """
        model_name : 'basic' | 'temporal_physics'

        For 'temporal_physics', pass:
          gamma_params : dict with keys A, alpha, beta, t0
          target_mean  : float
          target_std   : float
        """
        if model_name == 'basic':
            self.model = GraphGRUPredictor(
                in_channels, hidden_channels, out_channels, k_hops,
            ).to(self.device)

        elif model_name == 'temporal_physics':
            gamma_params = kwargs.get('gamma_params')
            if gamma_params is None:
                raise ValueError("temporal_physics requires gamma_params kwarg.")
            self.model = TempoPhysicsGraphGRUPredictor(
                in_channels, hidden_channels, out_channels, k_hops,
                gamma_params=gamma_params,
                target_mean=kwargs.get('target_mean', 0.0),
                target_std =kwargs.get('target_std',  1.0),
            ).to(self.device)

        else:
            raise ValueError(f"Unknown model_name: {model_name!r}")

    # ── Training ──────────────────────────────────────────────────────────────

    def train_model(self, batch_size=32, lr=0.001, epochs=1000,
                    dir_cache='./', verbose=True,
                    early_stopper_patience=7, early_stopper_min_delta=1e-4):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size,
            collate_fn=temporal_collate_fn, shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size,
            collate_fn=temporal_collate_fn, shuffle=False,
        )

        loss_fn  = nn.MSELoss()
        optim    = torch.optim.Adam(self.model.parameters(), lr=lr)
        stopper  = EarlyStopper(patience=early_stopper_patience,
                                 min_delta=early_stopper_min_delta)
        best_val = np.inf

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for x_seq, y in train_loader:
                x_seq = [s.to(self.device) for s in x_seq]
                y     = y.to(self.device)
                optim.zero_grad()
                out   = self.model(x_seq)
                if out.shape != y.shape:
                    out = out.view(y.shape)
                loss  = loss_fn(out, y)
                loss.backward()
                optim.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = val_mae = 0.0
            with torch.no_grad():
                for x_seq, y in val_loader:
                    x_seq = [s.to(self.device) for s in x_seq]
                    y     = y.to(self.device)
                    out   = self.model(x_seq)
                    if out.shape != y.shape:
                        out = out.view(y.shape)
                    val_loss += loss_fn(out, y).item()
                    val_mae  += nn.functional.l1_loss(out, y).item()

            val_loss /= len(val_loader)
            val_mae  /= len(val_loader)

            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch:4d} | "
                      f"Train MSE: {train_loss/len(train_loader):.6f} | "
                      f"Val MSE: {val_loss:.6f} | Val MAE: {val_mae:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                os.makedirs(dir_cache, exist_ok=True)
                torch.save(self.model.state_dict(),
                           os.path.join(dir_cache, 'best_model.pth'))

            stopper.stopper(val_loss)
            if stopper.stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

        self.model.load_state_dict(
            torch.load(os.path.join(dir_cache, 'best_model.pth'))
        )
        return best_val

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_test_set(self, batch_size=32):
        """
        Runs inference on the test set and returns (predictions, targets)
        as denormalised NumPy arrays.
        """
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size,
            collate_fn=temporal_collate_fn, shuffle=False,
        )
        loss_fn = nn.MSELoss()
        self.model.eval()

        test_loss = test_mae = 0.0
        preds_list, targets_list = [], []

        with torch.no_grad():
            for x_seq, y in test_loader:
                x_seq = [s.to(self.device) for s in x_seq]
                y     = y.to(self.device)
                out   = self.model(x_seq)
                if out.shape != y.shape:
                    out = out.view(y.shape)
                test_loss += loss_fn(out, y).item()
                test_mae  += nn.functional.l1_loss(out, y).item()
                preds_list.append(out.cpu().numpy())
                targets_list.append(y.cpu().numpy())

        avg_loss = test_loss / len(test_loader)
        avg_mae  = test_mae  / len(test_loader)
        print(f"Test | MSE: {avg_loss:.6f} | MAE: {avg_mae:.6f}")

        preds   = np.concatenate(preds_list)
        targets = np.concatenate(targets_list)

        # Denormalise
        ds = self.train_dataset.dataset
        if hasattr(ds, 'g_mean') and ds.g_mean is not None:
            m, s = ds.g_mean[0].item(), ds.g_std[0].item()
            print(f"Denormalising — Mean: {m:.2f}, Std: {s:.2f}")
            preds   = preds   * s + m
            targets = targets * s + m

        return preds, targets


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

class GraphTemporalTuner:
    """
    Wraps GraphTemporalPred in an Optuna study.

    Parameters
    ----------
    pred_manager : a built GraphTemporalPred instance (with dataset already split)
    in_channels  : number of node features
    out_channels : number of output features (usually 1)
    dir_cache    : directory for saving intermediate best models
    """

    def __init__(self, pred_manager, in_channels, out_channels, dir_cache):
        self.mgr         = pred_manager
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.dir_cache    = dir_cache

    def objective(self, trial):
        hidden  = trial.suggest_categorical("hidden_channels", [64, 96, 128, 160, 192])
        lr      = trial.suggest_float("lr",         1e-4, 1e-2, log=True)
        batch   = trial.suggest_categorical("batch_size", [16, 32, 64])
        k_hops  = trial.suggest_categorical("k_hops",     [2, 3])

        self.mgr.build_model(self.in_channels, hidden, self.out_channels,
                              k_hops=k_hops, model_name='basic')
        return self.mgr.train_model(
            batch_size=batch, lr=lr, epochs=50,
            dir_cache=self.dir_cache, verbose=False,
        )

    def run_study(self, storage, study_name, n_trials=20):
        study = optuna.create_study(
            direction="minimize",
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )
        study.optimize(self.objective, n_trials=n_trials)
        print("Best trial:", study.best_trial.params)
        return study.best_trial.params
