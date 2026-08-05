"""
Temporal decay analysis for Baton Rouge (Ida 2021) — the GRU pipeline's
physics prior.

Produces the impact/recovery curve that run_prediction_training.py's
'temporal_physics' model needs: the city's TOTAL flow series is decomposed
against a Prophet baseline, the disaster-period deviation is measured, and a
StepWiseModel is fitted to it.  Its parameters (A, alpha, beta, t0) become the
gamma_params the physics-informed GRU is built with, and the fitted impact
window becomes the disaster_time node feature.

Steps
-----
1. Load graphs.
2. Compute total flows.
3. Decompose the pre-disaster baseline with Prophet.
4. Forecast expected behaviour and measure % deviation.
5. Identify the impact period and fit a StepWise recovery curve.
6. Save the decay model parameters to DECAY_RESULTS_PATH.

Run (from the repo root)
------------------------
    python -m utils.neural_network.temporal_decay

History: this began as a root-level exploration script (pre-NMF era), moved to
archive/ on 2026-07-12 when it left the research line, and moved here on
2026-08-05 — it is not exploration any more but the sole producer of a GRU
input, so it belongs to the GRU package.  It is NOT part of run_pattern_nmf.py's
orbit and shares nothing with it but generic graph loaders.

NOTE the analysis functions it calls still live in utils/pattern_analysis/
(temporal.py, and the flow time-series figures in that package's
visualization.py).  temporal.py in particular must stay importable at that path
regardless of who calls it: pickle resolves StepWiseModel from the module path
recorded when the results were written, so moving it would break the
deserialization of any results pkl produced before the move.
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
#
# CONSUMER MISMATCH, unresolved as of 2026-08-05: run_prediction_training.py
# maps the fitted impact indices back to the full series with a HARDCODED
# n_days_back = 10 and a comment claiming it must match this constant.  It does
# not — 10 is the pre-extension value, this became 23 when the BR file was
# extended to Sep 16.  Left as-is on both sides because changing either number
# changes GRU results, which is an owner decision; see that call site.
N_DAYS_BACK        = 23          # post-disaster window (Aug 25 – Sep 16, pre-Ida split)
ROLLING_WINDOW     = SLOT_PER_DAY  # moving-average window = 1 day of slots
                                   # (8 for 3h data, 12 for 2h data — auto from config)
SHOW_PLOTS         = True  # True: outputs Prophet decomposition, forecast comparison, decay fit plots
END_METHOD         = 'first_return'   # 'first_return' | 'consecutive' | 'end_of_series'

OUTPUT_PLOTS = os.path.join(OUTPUT_DIR, 'archive', 'temporal_decay')
# Where the fitted results land.  run_prediction_training.py imports this name
# instead of rebuilding the path, so producer and consumer cannot drift apart.
DECAY_RESULTS_PATH = os.path.join(OUTPUT_DIR, 'archive',
                                  'temporal_decay_results_br.pkl')


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_decay_results(save_path=DECAY_RESULTS_PATH, show_plots=SHOW_PLOTS):
    """Fit the impact/recovery curve and persist it; returns the results dict."""
    os.makedirs(OUTPUT_PLOTS, exist_ok=True)

    graphs = load_graphs(BR_GRAPH_PATH)

    # Guard: this analysis expects the recovery-extended BR file (2021-04-15 ..
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
        show_plots         = show_plots,
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {save_path}")
    return results


def main():
    build_decay_results()


if __name__ == '__main__':
    main()
