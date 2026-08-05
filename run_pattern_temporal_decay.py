"""
Temporal decay analysis for Baton Rouge (Ida 2021).

Steps
-----
1. Load graphs.
2. Compute total flows.
3. Decompose the pre-disaster baseline with Prophet.
4. Forecast expected behaviour and measure % deviation.
5. Identify the impact period and fit a StepWise recovery curve.
6. Save the decay model parameters to disk for use in neural network preprocessing.

Run
---
    python run_pattern_temporal_decay.py

Retired from the production line 2026-07-12 but KEPT at the repo root
(2026-08-05): it is the sole producer of
outputs/archive/temporal_decay_results_br.pkl, which the GRU pipeline's
temporal_physics mode (run_prediction_training.py) unpickles.  Not part of
run_pattern_nmf.py's orbit.
"""
import os
import pickle

from config import (
    BR_GRAPH_PATH, OUTPUT_DIR,
    SLOT_PER_DAY, PROPHET_FREQ, PROPHET_START_DATE,
)
from utils.pattern_analysis.graph_io import load_graphs, calculate_total_flows
from utils.pattern_analysis.temporal import run_temporal_decay_analysis_pipeline


# ── Configuration ─────────────────────────────────────────────────────────────

# Post-disaster window = last N_DAYS_BACK days; the Prophet baseline is fit on
# everything BEFORE it, so the split must land before Ida (2021-08-29).  BR was
# extended to 2021-09-16; keeping the split at 2021-08-25 (4 days pre-Ida, as in
# the pre-extension config) while absorbing the longer recovery tail gives
# N_DAYS_BACK = 23 (Aug 25 – Sep 16 = Ida + 18 recovery days).  NOTE: unlike the
# window scripts, this script is intentionally NOT trimmed to Sep 12 — the
# recovery curve benefits from all available post-Ida data.
N_DAYS_BACK        = 23          # post-disaster window (Aug 25 – Sep 16, pre-Ida split)
ROLLING_WINDOW     = SLOT_PER_DAY  # moving-average window = 1 day of slots
                                   # (8 for 3h data, 12 for 2h data — auto from config)
SHOW_PLOTS         = True  # True: outputs Prophet decomposition, forecast comparison, decay fit plots
END_METHOD         = 'first_return'   # 'first_return' | 'consecutive' | 'end_of_series'

OUTPUT_PLOTS = os.path.join(OUTPUT_DIR, 'archive', 'temporal_decay')


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_PLOTS, exist_ok=True)

    graphs = load_graphs(BR_GRAPH_PATH)

    # Guard: this script expects the recovery-extended BR file (2021-04-15 ..
    # 2021-09-16 = 155 days).  N_DAYS_BACK=23 places the pre/disaster split at
    # 2021-08-25 (pre-Ida) only when the file reaches Sep 16; on the old Sep-3
    # file (142 days) that split would fall in mid-August and contaminate the
    # Prophet baseline with disaster slots.  Fail loudly rather than silently.
    _expected_days = 155
    _days = len(graphs) // SLOT_PER_DAY
    if _days < _expected_days:
        raise ValueError(
            f"Baton Rouge: temporal-decay expects the recovery-extended file "
            f"(~{_expected_days} days, ending 2021-09-16) but the pkl has only "
            f"{_days} days ({len(graphs)} graphs). The extended BR data may not be "
            f"rebuilt yet — rebuild the graph pkl, or adjust N_DAYS_BACK so the "
            f"pre/disaster split still lands before Ida (2021-08-29)."
        )

    flows  = calculate_total_flows(graphs)

    results = run_temporal_decay_analysis_pipeline(
        flows              = flows,
        n_days_back        = N_DAYS_BACK,
        start_date         = PROPHET_START_DATE,
        frequency          = PROPHET_FREQ,
        slots_per_day      = SLOT_PER_DAY,
        rolling_window_size= ROLLING_WINDOW,
        show_plots         = SHOW_PLOTS,
        output_dir         = OUTPUT_PLOTS,
        end_method         = END_METHOD,
    )

    decay_model   = results['recovery_model']
    impact_meta   = results['impact_metadata']

    print("\nImpact period summary:")
    print(f"  Start index : {impact_meta['impact_period']['start_index']}")
    print(f"  End index   : {impact_meta['impact_period']['end_index']}")
    print(f"  Duration    : {impact_meta['impact_period']['duration']} slots")
    print(f"  Status      : {impact_meta['impact_period']['status']}")

    if decay_model is not None:
        print(f"\nStepWise model parameters:")
        print(f"  A     = {decay_model.A:.4f}")
        print(f"  alpha = {decay_model.alpha:.4f}")
        print(f"  beta  = {decay_model.beta:.4f}")
        print(f"  t0    = {decay_model.t0:.4f}")

    # Persist results so the neural-network pipeline can read them
    out_path = os.path.join(OUTPUT_DIR, 'archive', 'temporal_decay_results_br.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
