"""
Standalone evaluation script.

Loads saved test predictions and model weights, then produces:
  - Aggregated flow vs. predicted flow plot with R² overlay
  - Per-timestep scatter plot
  - Gated physics function plot (only for temporal_physics model)

Run
---
    python run_prediction_eval.py
"""
import os
import pickle

import numpy as np

from config import OUTPUT_DIR, MODEL_CACHE_DIR
from utils_neural_network.visualization import (
    plot_test_results, plot_prediction_scatter, plot_gated_physical_function,
)


# ── Configuration ─────────────────────────────────────────────────────────────

PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, 'test_predictions.pkl')
SELECTED_SLOTS   = (2, 3, 4, 5, 6)

# Set to True to also plot the physics gate (requires the model to be in memory)
PLOT_PHYSICS_GATE = False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"Predictions file not found: {PREDICTIONS_PATH}")
        print("Run run_prediction_training.py first.")
        return

    with open(PREDICTIONS_PATH, 'rb') as f:
        data = pickle.load(f)

    test_preds   = data['predictions']
    test_targets = data['targets']

    print(f"Loaded {len(test_targets)} test samples.")
    print(f"Prediction shape: {test_preds.shape}")

    # Flow + R² overview
    plot_test_results(test_targets, test_preds, selected_slots=SELECTED_SLOTS)

    # Per-timestep scatter (uses first timestep as example)
    if len(test_targets) > 0:
        plot_prediction_scatter(test_targets[0], test_preds[0])

    if PLOT_PHYSICS_GATE:
        import torch
        from utils_neural_network.models import TempoPhysicsGraphGRUPredictor

        # Rebuild the model skeleton just enough to load weights and plot
        # You will need to supply the same gamma_params used during training.
        print("PLOT_PHYSICS_GATE=True but model not rebuilt here. "
              "Call plot_gated_physical_function(predictor.model) from run_prediction_training.py instead.")


if __name__ == '__main__':
    main()
