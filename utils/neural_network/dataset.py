"""
PyTorch Dataset and DataLoader collation for temporal graph sequences.

TemporalGraphDataset
    Builds sliding-window sequences that include:
      - recent history (window_size consecutive graphs)
      - daily seasonal anchors (same slot from past num_days days)
      - weekly seasonal anchors (same slot from past num_weeks weeks)

    Only time slots in selected_slots (default 6-9h to 18-21h) are used
    as prediction targets to avoid noisy overnight data.

temporal_collate_fn
    Collates a batch of (sequence, target) pairs by stacking graphs at the
    same temporal position into a single PyG Batch object.
"""
import torch
import torch_geometric as pyG


class TemporalGraphDataset(torch.utils.data.Dataset):
    """
    Parameters
    ----------
    graphs        : list of PyG Data objects (chronological order)
    window_size   : number of recent timesteps fed to the model
    target_attr   : name of the target attribute on each Data object
    num_days      : number of daily lags to include
    num_weeks     : number of weekly lags to include
    selected_slots: which daily slots (0-7) are valid prediction targets
                    (default: daytime slots 2-6)
    """

    def __init__(self, graphs, window_size, target_attr='y',
                 num_days=6, num_weeks=3, selected_slots=(2, 3, 4, 5, 6)):
        super().__init__()
        self.graphs        = graphs
        self.window_size   = window_size
        self.target_attr   = target_attr
        self.num_days      = num_days
        self.num_weeks     = num_weeks
        self.daily_stride  = 8     # slots per day
        self.weekly_stride = 56    # slots per week  (8 × 7)
        self.min_idx       = max(num_days * self.daily_stride,
                                  num_weeks * self.weekly_stride)

        possible = len(graphs) - window_size - self.min_idx
        self.valid_indices = [
            i for i in range(possible)
            if (i + self.min_idx + window_size) % 8 in selected_slots
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual = self.valid_indices[idx] + self.min_idx

        x_recent = self.graphs[actual: actual + self.window_size]
        x_daily  = [self.graphs[(actual + self.window_size) - d * self.daily_stride]
                    for d in range(1, self.num_days + 1)]
        x_weekly = [self.graphs[(actual + self.window_size) - w * self.weekly_stride]
                    for w in range(1, self.num_weeks + 1)]

        # oldest → newest: weekly lags, then daily lags, then recent window
        x_seq = x_weekly[::-1] + x_daily[::-1] + x_recent
        y     = getattr(self.graphs[actual + self.window_size], self.target_attr)
        return x_seq, y


def temporal_collate_fn(batch):
    """
    Collates a list of (x_seq, y) pairs into (x_seq_batch, y_batch).

    x_seq_batch : list of PyG Batch objects, one per timestep in the sequence
    y_batch     : float32 tensor of shape (batch_size, nodes, 1)
    """
    window_size   = len(batch[0][0])
    x_seq_batch   = [
        pyG.data.Batch.from_data_list([item[0][t] for item in batch])
        for t in range(window_size)
    ]
    y_targets = [item[1] for item in batch]
    y_batch   = (torch.stack(y_targets)
                 if isinstance(y_targets[0], torch.Tensor)
                 else torch.tensor(y_targets))
    return x_seq_batch, y_batch.to(torch.float32)
