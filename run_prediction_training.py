"""
End-to-end Graph GRU training pipeline for disaster mobility prediction.

Workflow
--------
1. Load Baton Rouge graphs.
2. (Optional) Load temporal decay results to inject a disaster_time feature.
3. Convert each graph to a PyG Line Graph Data object.
4. Build train/val/test splits.
5. Optionally tune hyperparameters with Optuna.
6. Train the chosen model variant and evaluate on the test set.

Run
---
    python run_prediction_training.py
"""
import os
import pickle
import random

from config import (BR_GRAPH_PATH, OUTPUT_DIR, MODEL_CACHE_DIR, OPTUNA_CACHE_DIR,
                    SLOT_PER_DAY)
from utils.pattern_analysis.graph_io import load_graphs
from utils.neural_network.preprocessing import (
    transform_to_line_graph_data, set_graph_targets,
    get_original_index, add_disaster_time_feature,
)
from utils.neural_network.temporal_decay import DECAY_RESULTS_PATH
from utils.neural_network.trainer import GraphTemporalPred, GraphTemporalTuner
from utils.neural_network.visualization import plot_test_results


# ── Configuration ─────────────────────────────────────────────────────────────

FEATURES_TO_USE = ['flow', 'distance', 'disaster_time']

WINDOW_SIZE  = 7
IN_CHANNELS  = len(FEATURES_TO_USE)
OUT_CHANNELS = 1
TARGET_ATTR  = 'y'
TRAIN_RATIO  = 0.65
VAL_RATIO    = 0.15

DO_TUNING   = False
TUNE_TRIALS = 10
MAX_EPOCHS  = 200

# Best / default hyperparameters
HIDDEN_CHANNELS = 128
LEARNING_RATE   = 0.0035
BATCH_SIZE      = 16
K_HOPS          = 2

MODEL_NAME = 'temporal_physics'   # 'basic' | 'temporal_physics'

# Where the temporal-decay physics prior lives.  The path is imported from its
# producer (utils.neural_network.temporal_decay) so the two cannot drift; build
# it with:  python -m utils.neural_network.temporal_decay


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load graphs
    graphs = load_graphs(BR_GRAPH_PATH)

    # Add disaster_time feature from decay analysis (if available)
    decay_model = None
    if os.path.exists(DECAY_RESULTS_PATH):
        with open(DECAY_RESULTS_PATH, 'rb') as f:
            decay_results = pickle.load(f)
        meta        = decay_results['impact_metadata']
        # 'recovery_model' is the current key (a StepWiseModel); 'gamma_model' is the
        # legacy key of the same object in pickles written before 2026-07-12.
        decay_model = decay_results.get('recovery_model',
                                        decay_results.get('gamma_model'))

        # MISMATCH, unresolved 2026-08-05: the producer
        # (utils.neural_network.temporal_decay) fits with N_DAYS_BACK = 23 since
        # the BR file was extended to 2021-09-16; this 10 is the pre-extension
        # value and was never updated, so the impact window is mapped back onto
        # the full series with the wrong split offset.  NOT changed here because
        # either fix moves GRU results — owner decision pending.
        n_days_back = 10
        start_orig  = get_original_index(
            meta['impact_period']['start_index'], len(graphs), n_days_back, SLOT_PER_DAY,
        )
        end_orig    = get_original_index(
            meta['impact_period']['end_index'],   len(graphs), n_days_back, SLOT_PER_DAY,
        )
        graphs = add_disaster_time_feature(graphs, start_orig, end_orig)
        print(f"disaster_time feature added: impact [{start_orig}, {end_orig}]")
    else:
        print(f"Warning: decay results not found at {DECAY_RESULTS_PATH}. "
              f"disaster_time will be 0 for all graphs.")
        for G in graphs:
            for _, _, d in G.edges(data=True):
                d['disaster_time'] = 0

    # Convert to PyG line graphs
    print(f"Converting {len(graphs)} graphs to PyG Line Graphs …")
    pyg_graphs = [transform_to_line_graph_data(G, feature_keys=FEATURES_TO_USE)
                  for G in graphs]
    pyg_graphs = set_graph_targets(pyg_graphs, target_attr=TARGET_ATTR)
    print(f"Example graph: {pyg_graphs[0]}")

    # Build dataset
    predictor    = GraphTemporalPred()
    input_graphs = [g.clone() for g in pyg_graphs]
    predictor.build_dataset(
        graphs      = input_graphs,
        window_size = WINDOW_SIZE,
        train_ratio = TRAIN_RATIO,
        val_ratio   = VAL_RATIO,
        target_attr = TARGET_ATTR,
    )

    # Optional: redistribute a few disaster samples into train/val for exposure
    train_idx = list(predictor.train_dataset.indices)
    val_idx   = list(predictor.val_dataset.indices)
    test_idx  = list(predictor.test_dataset.indices)

    leaked_train = test_idx[9:12]  + test_idx[-14:-11]
    leaked_val   = test_idx[12:13] + test_idx[-15:-14]
    train_idx.extend(leaked_train * 10)
    val_idx.extend(leaked_val * 10)
    random.shuffle(train_idx)

    predictor.train_dataset.indices = train_idx
    predictor.val_dataset.indices   = val_idx
    print(f"Adjusted sizes — Train: {len(predictor.train_dataset)}, "
          f"Val: {len(predictor.val_dataset)}, Test: {len(predictor.test_dataset)}")

    # Hyperparameter tuning
    hidden, lr, batch, k = HIDDEN_CHANNELS, LEARNING_RATE, BATCH_SIZE, K_HOPS
    if DO_TUNING:
        print("\n── Hyperparameter Tuning ──")
        tuner = GraphTemporalTuner(predictor, IN_CHANNELS, OUT_CHANNELS, MODEL_CACHE_DIR)
        # Persist under .cache/optuna/<task>/<task>.db, same scheme as the NMF tuner.
        gru_db = os.path.join(OPTUNA_CACHE_DIR, 'tag_gru_tuning', 'tag_gru_tuning.db')
        os.makedirs(os.path.dirname(gru_db), exist_ok=True)
        best  = tuner.run_study(
            storage   = f"sqlite:///{gru_db}",
            study_name= "disaster_mobility_tuning",
            n_trials  = TUNE_TRIALS,
        )
        hidden, lr, batch, k = (best['hidden_channels'], best['lr'],
                                  best['batch_size'],    best['k_hops'])

    # Build model
    model_kwargs = {}
    if MODEL_NAME == 'temporal_physics':
        if decay_model is None:
            raise RuntimeError(
                "temporal_physics model requires decay results. "
                "Run: python -m utils.neural_network.temporal_decay"
            )
        ds = predictor.train_dataset.dataset
        model_kwargs = {
            'gamma_params': {'A': decay_model.A, 'alpha': decay_model.alpha,
                             'beta': decay_model.beta, 't0': decay_model.t0},
            'target_mean': ds.g_mean[0].item(),
            'target_std':  ds.g_std[0].item(),
        }

    print(f"\n── Training ({MODEL_NAME}) ──")
    predictor.build_model(
        in_channels     = IN_CHANNELS,
        hidden_channels = hidden,
        out_channels    = OUT_CHANNELS,
        k_hops          = k,
        model_name      = MODEL_NAME,
        **model_kwargs,
    )
    predictor.train_model(
        batch_size             = batch,
        lr                     = lr,
        epochs                 = MAX_EPOCHS,
        dir_cache              = MODEL_CACHE_DIR,
        early_stopper_patience = 7,
    )

    # Evaluate
    print("\n── Test Set Evaluation ──")
    test_preds, test_targets = predictor.evaluate_test_set(batch_size=batch)
    plot_test_results(test_targets, test_preds)

    # Save predictions
    out_path = os.path.join(OUTPUT_DIR, 'test_predictions.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'predictions': test_preds, 'targets': test_targets}, f)
    print(f"Predictions saved to {out_path}")


if __name__ == '__main__':
    main()
