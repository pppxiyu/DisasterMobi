"""
Visualisation helpers for the neural network pipeline.

Functions
---------
plot_test_results         – aggregated flow and R² over the test period
plot_prediction_scatter   – scatter plot of predicted vs. actual with regression line
plot_gated_physical_function – original vs. learned Gamma curve with gate activation
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import torch


def plot_test_results(targets, predictions=None, selected_slots=(2, 3, 4, 5, 6)):
    """
    Plots aggregated network flow (actual vs. predicted) and per-timestep R²
    over the full test period.

    Sparse predictions (only daytime slots) are placed on a full 8-slot-per-day
    timeline so that day boundaries are clearly visible.
    """
    agg_targets = targets.sum(axis=1).flatten()

    r2_values = []
    if predictions is not None:
        agg_preds = predictions.sum(axis=1).flatten()
        for i in range(len(targets)):
            try:
                r2_values.append(r2_score(targets[i].flatten(), predictions[i].flatten()))
            except Exception:
                r2_values.append(np.nan)

    n_slots = len(selected_slots)
    n_samp  = len(agg_targets)
    n_days  = int(np.ceil(n_samp / n_slots))
    full_len = n_days * 8

    plot_targets = np.full(full_len, np.nan)
    plot_preds   = np.full(full_len, np.nan)
    plot_r2      = np.full(full_len, np.nan)

    s_idx = 0
    for day in range(n_days):
        for slot in range(8):
            if slot in selected_slots and s_idx < n_samp:
                pos = day * 8 + slot
                plot_targets[pos] = agg_targets[s_idx]
                if predictions is not None:
                    plot_preds[pos] = agg_preds[s_idx]
                    plot_r2[pos]    = r2_values[s_idx]
                s_idx += 1

    fig, ax1 = plt.subplots(figsize=(16, 8))
    t = np.arange(full_len)

    ax1.plot(t, plot_targets, label="Actual Flow", color='#D81B60',
             marker='o', markersize=6, linestyle='--', linewidth=2, alpha=0.8)
    if predictions is not None:
        ax1.plot(t, plot_preds, label="Predicted Flow", color='#FF8F00',
                 marker='o', markersize=6, linestyle='--', linewidth=2, alpha=0.8)

    ax1.set_xlabel("Time slot index (8 slots/day)", fontsize=20, fontweight='bold', labelpad=15)
    ax1.set_ylabel("Aggregated Flow Value",          fontsize=20, fontweight='bold', labelpad=15)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.grid(False)

    if predictions is not None:
        ax2 = ax1.twinx()
        ax2.plot(t, plot_r2, label="R² Score", color='#1976D2',
                 marker='o', markersize=6, linestyle='-', linewidth=3.5, alpha=0.9)
        ax2.set_ylabel("R² Value", fontsize=20, fontweight='bold', labelpad=15)
        ax2.tick_params(axis='y', labelsize=18)
        ax2.set_ylim([0, 1.05])
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax2.grid(True, axis='y', linestyle=':', alpha=0.6, color='gray')

    for day in range(n_days + 1):
        ax1.axvline(x=day * 8, color='black', linestyle='-', linewidth=1.2, alpha=0.15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    if predictions is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='lower center', bbox_to_anchor=(0.5, 1.02),
                   ncol=3, fontsize=16, frameon=False)
    else:
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                   ncol=1, fontsize=16, frameon=False)

    plt.tight_layout(); plt.show()


def plot_prediction_scatter(targets, predictions):
    """
    Scatter plot: predicted vs. actual values, with the identity line and
    a linear regression line of best fit.
    """
    y_true = np.array(targets).flatten().reshape(-1, 1)
    y_pred = np.array(predictions).flatten().reshape(-1, 1)

    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    reg   = LinearRegression().fit(y_true, y_pred)
    y_fit = reg.predict(y_true)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='skyblue', edgecolor='black',
                alpha=0.5, s=30, label='Data Points')

    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], color='gray', linestyle='--', alpha=0.6, label='Ideal (y=x)')

    sort_idx = y_true.flatten().argsort()
    plt.plot(y_true[sort_idx], y_fit[sort_idx], color='red', linewidth=2,
             label=f'Best Fit (R²={r2:.3f})')

    plt.title(f"Regression Analysis (RMSE: {rmse:.4f})", fontsize=14, fontweight='bold')
    plt.xlabel("Ground Truth Value", fontsize=11, fontweight='bold')
    plt.ylabel("Predicted Value",    fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(frameon=True, loc='upper left')
    plt.tight_layout(); plt.show()


def plot_gated_physical_function(model, t_max=50, t_step=0.5, time_aware=False):
    """
    Plots the original Gamma curve, the learned gate, and the resulting
    gated Gamma curve for a TempoPhysicsGraphGRUPredictor.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        t_vals       = torch.arange(0, t_max, t_step).view(-1, 1).to(device)
        orig_gamma   = model.torch_gamma_predict(t_vals)

        gate_input   = t_vals / 20.0 if time_aware else orig_gamma
        raw_gate     = model.physics_gate(gate_input)
        gate_vals    = 1.0 + torch.tanh(raw_gate)   # same formula as forward()
        new_gamma    = orig_gamma * gate_vals

        t_np         = t_vals.cpu().numpy().flatten()
        orig_np      = orig_gamma.cpu().numpy().flatten()
        new_np       = new_gamma.cpu().numpy().flatten()
        gate_np      = gate_vals.cpu().numpy().flatten()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(t_np, orig_np, label='Original Gamma', color='#1976D2',
             linestyle='--', linewidth=2)
    ax1.plot(t_np, new_np,  label='Gated Gamma',    color='#D81B60', linewidth=3)
    ax1.set_xlabel("Disaster Time Feature (t)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Reduction Ratio",           fontsize=14, fontweight='bold')
    ax1.set_title("Physical Function: Original vs. Gated", fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(t_np, gate_np, label='Gate Activation', color='#388E3C',
             linestyle=':', linewidth=2.5)
    ax2.set_ylabel("Gate Multiplier", fontsize=14, color='#388E3C', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#388E3C')

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc='lower right', fontsize=12)

    plt.tight_layout(); plt.show()
