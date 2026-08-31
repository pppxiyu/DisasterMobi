"""
Production pipeline: NMF decomposition of disaster origin–destination (OD) flow
mobility for the hurricane city-events in CITY_EVENTS (13 units retained of the
17 registered; see EXCLUDED_CODES), per-component characterisation, and
cross-city resilience prediction.

Core decomposition idea
-----------------------
Each city-event (a "unit") is analysed on one trailing 33-day window whose
segments are aligned identically across all five units: 13 pre-disaster
("normal") days, then a 5-day pre-landfall buffer covering the preparation and
evacuation days, then a 15-day disaster segment consisting of the landfall day
plus 14 recovery days, which always closes the window.

On this window, each unit gets ONE decomposition (as opposed to decomposing
each period separately and matching components afterwards), and that
decomposition is a fit-then-project NMF: the spatial basis H is fitted on the
pre-disaster (normal + buffer) columns only, as selected by the unit's
`fit_segments` setting in CITY_EVENTS, and the FULL window — including the disaster
segment — is then projected onto that frozen basis to obtain W.  Two factors
result per unit; in the shapes below, k is the unit's number of components
(set in CITY_EVENTS), "time" counts the window's active time slots, and "OD"
counts the retained OD pairs:

    W : temporal factor  [time × k]   one activity-over-time curve per component
    H : spatial factor    [k × OD]    one OD-flow map      per component

Because every period is expressed on the same pre-disaster basis, each of the
k components keeps its identity across the normal/disaster boundary, so no
component-matching step is needed, and the disaster columns of W measure how
strongly each pre-disaster activity pattern persists through the disaster.
Disaster behaviour cannot reshape the basis itself.  The buffer columns are
part of the projection input, but they are excluded from the temporal features
and from the resilience baselines and curves.

Logical steps (mirrors main(); the same step numbers label the banners there
and the parameter and function groups below)
--------------------------------------------------------------------
STEP 1 — Load every unit and classify land use once.
    The pipeline loads every unit's geometry and then computes its ONE
    land-use classification: every block group of every unit is labelled with
    its dominant land-use category using a single TF-IWF weight vector pooled
    over all five units' block groups (TF-IWF is a TF-IDF-style reweighting
    that down-weights ubiquitous categories; because the weights are pooled, a
    functional share means the same thing in every city, and every later step
    reuses this same classification).  Since two events of one city are two
    units, Wilmington's block groups enter the pool once per event — the pool
    weights events, not distinct cities.
STEP 2 — Decompose every unit.
    For each unit, the pipeline loads the graph sequence, builds the
    [OD × time] flow matrix on the window described above (overnight slots
    are dropped, and low-activity OD pairs are filtered out when the unit's
    filter factor is positive), and runs the fit-then-project NMF described
    above at the unit's k and sparsity (both set in CITY_EVENTS), yielding
    W, H and the component importance weights.
STEP 3 — Characterise each unit's components (the within-city analyses).
    Four characteristic blocks quantify each component: its temporal
    signature; its loading-weighted flow distance ("loading-weighted" means
    averaged over the OD pairs with the component's row of H — its spatial
    loadings — as the weights, so OD pairs that carry more of the component
    count more); its loading-weighted ACS (American Community Survey) median
    household income; and its O×D functional cross-tab M, a [k × C × C] array
    (one C × C table per component) that distributes each component's H
    loading mass over origin-category × destination-category pairs, using the
    STEP-1 land-use labels (C categories).  The block
    results are assembled into one feature table with one row per component,
    holding the temporal, functional, spatial and socioeconomic predictors
    plus the two resilience targets — cum_loss, and the recovery rate
    recovery_alpha of the landfall-day-anchored logistic recovery model
    (higher = faster recovery back to baseline) — and correlation and
    Ridge-regression analyses relate those predictors to the targets.  Two per-event-constant
    LEVEL covariates are appended last, so that only the cross-city steps see
    them: hurricane_intensity, the storm's Saffir-Simpson intensity at
    arrival, and evac_level, the population-weighted strength of the event's
    official evacuation orders.
STEP 4 — Relate event severity to component resilience, pooled over all units.
    Every component of every city-event goes into one pooled table, and each
    resilience target is examined against hurricane_intensity (STEP 3).  This
    is a pooled diagnostic with no train/test split and no model.
STEP 5 — Build the cross-city feature tables.
    The predictor tables are rebuilt with the same STEP-1 classification
    (functional shares are therefore cross-city comparable by construction;
    the technical notes call this single-recipe design "config C").  Each
    unit gets two tables for historical reasons: the roles used to decompose the
    same matrix at different k (train at the unit's own k, test pinned at k = 10).
    Since the rank CV supplies a label-free k from the unit's own matrix, BOTH
    roles read that own-k decomposition and the two tables are the same object.
STEP 6 — Cross-city prediction.
    Each method runs in its enforced standardization frame (spearman in the
    rank frame, pearson in the raw-value frame) through three analyses on the
    STEP-5 tables: the leave-one-city-event-out transfer, the pairwise
    train-to-test comparison, and — for pearson only — the reconstruction of
    each city's cum_loss (the cumulative daily activity deficit over the
    disaster window, in day-equivalents — the resilience target)
    against the decomposition-free similarity baseline.
STEP 7 — Cross-city curve prediction.
    The output is upgraded from the cum_loss scalar to the whole mobility
    trajectory.  Each component's FITTED curve is the surge-plus-relaxation
    model r(t) = L/(1 + (L/r0−1)e^(−α·t)) + B·t·e^(−b·t), anchored at the
    OBSERVED landfall-day value r0 (Li, Wang & Chen 2024's problem setting:
    the initial post-disaster state and the normal level are the inputs, the
    recovery process is the prediction).  The end level L and the signed
    forced pulse (B, b) absorb the settle-away-from-baseline and hump/dip
    transients the data contains, so the jointly-fitted rate α is CLEAN.  The
    FORECAST is the PLATEAU INVERSION: the recovery RATE does not transfer
    across cities (its within-city Ridge R² is negative in all five units)
    but cum_loss does (up to 0.70), so the forecast holds the rate at the
    pooled-train mean and lets the ONE predictable quantity set the ONE free
    family parameter — each component's curve is the family logistic itself
    (B = 0) with the plateau L solved so the curve's net signed loss lands
    halfway between the backbone's own loss and the component's cross-city
    predicted cum_loss (halfway = fixed shrinkage against prediction noise).
    That predicted cum_loss is itself assembled by QUANTILE MAPPING (the
    comonotone one-dimensional optimal-transport assignment): the rank-path
    kNN orders the components within the city, the pooled-train cum_loss
    quantiles give the values their shape, a ratio of observable backbone-loss
    spreads rescales them to the held city's own dispersion, and an additive
    shift pins the weight_normal aggregate to the city total of STEP 6's
    aggr_denorm strategy — the component ridge's aggregate score s scored on
    [s, r0_city, GDP] and variance-matched (the same estimator the city_total/
    headline figure reports, shared through _city_total_from_scores).
    weight_normal aggregation turns the component curves into the city
    trajectory, compared against the observed curve alongside three reference
    lines: a city-wise forecast (one rate for the whole city, a single
    logistic — the decomposition-free counterpart), the component's own
    UNGATED full fit (the model family's ceiling), and the pooled-train mean
    rate alone (the L = 1, zero-shrinkage special case).

Output tree (outputs/)
--------------------------
Folder names carry a numeric prefix so a file browser lists them in pipeline
order; the prefixes are part of the paths below and of the OUTPUT_* constants.

    0-data/                      study-overview figures that use NO
        decomposition: city_mobility_curves.png (every unit's day-type-
        normalized total activity, minimum + landfall marked) + raw_data/.
        (The MSA cluster map is filed with its cluster instead, under
        2-cross_city_resi_pred/component_rank/.)
    1-decomposition/             the factorisation and what its components
                                 ARE, in two parts:
      1-decomposition_quality/   STEP 1 + STEP 3, everything about the
        factorisation itself.  nmf_rank_cv.png (STEP 1) sweeps k under
        held-out-entry cross-validation for EVERY registry unit and is what
        sets each unit's n_behaviors and what EXCLUDED_CODES drops; the units
        it drops keep their (greyed) panel, since this figure is the record of
        why.  The STEP-3 quality check then covers only the RETAINED units,
        FIT-window only (the data the basis is fit on; the disaster period is
        out of scope): the per-unit console report (figures retired
        2026-08-28) — per-slot distribution
        error, value-CDF overlay, component weights vs the NMF_MIN_COMP_FRAC
        health threshold) and the cross-city raw_data/nmf_quality_metrics.csv
        dashboard.  raw_data/ holds the rank-CV curves and selection table plus
        the quality metrics.
      2-component_characteristics/  per-component characteristics, by type:
        0-temporal/          signature heatmap + timeline (W)
        1-func/              O×D functional cross-tab heatmaps + entropy + raw CSVs
        2-resilience_curves/ per-function ordered line stacks: W timelines and
                             r(d) curves, where r(d) is the component's day-d
                             activity relative to its matched pre-disaster
                             baseline
        3-func_vs_temporal/  all_city/ (pooled over every unit: the time ×
                             function correlation heatmap plus, per temporal
                             feature, the scatter it summarises) and per_city/
                             (the same heatmap one unit at a time)
        4-func_vs_resi/      func/distance/income × cum_loss Spearman heatmaps
                             (rows = INTRA_RES_COLS, cum_loss only), per_city/ +
                             pooled all_city/ with raw_data/
        5-disaster_vs_resi/  pooled arrival-intensity vs cum_loss scatter
                             (intensity is a per-event constant, so this reads
                             between events) + raw_data/
        6-socioeconomic/     per-component median-income bars + raw CSV
        7-spatial/           per-component flow-distance bar (H) + raw CSV
    2-cross_city_resi_pred/      the cross-city outputs, split by what is predicted:
        city_total/     the CITY-LEVEL cum_loss prediction, one number per
                        city-event: bar_cross_city_resi_pred.png/.svg (observed
                        vs predicted + a leave-one-out R² panel), raw_data/, and
                        decomp_pred_aggr_denorm/<model>/ per predictor with its
                        own calibration scatter + raw_data/
        component_rank/ the per-COMPONENT rank transfer (cum_loss only, and
                        no income predictor), ridge only and NO per-model
                        subfolder (cosine-kNN retired 2026-08-28, the ridge/
                        level flattened away with it).  THE rank channel lives
                        here: rank_pred_vs_true.png (the leave-one-out sweep,
                        one panel per held-out unit, Spearman ρ), the
                        pairwise transfer heatmap (its Louvain boxes are a
                        DISPLAY ordering — nothing trains on them since
                        2026-08-31),
                        the mapping-direction PCA (mapping_direction_pca.png:
                        one panel of every component in within-city rank-z
                        space, each city's and the one pooled feature->loss
                        mapping direction as arrows) and the function
                        co-riding graph (cluster_function_graph.png: the
                        pooled average and each cluster as a distance layout
                        of the 6 functions, d = 1 - rho, edges gated by a
                        within-city permutation test with BH correction;
                        SECOND ROW = the same layouts greyed out as a base
                        map with every city-event's own layout drawn on top,
                        colour = the function)
    3-cross_city_curve_pred/     the STEP-7 outputs: bar_cross_city_curve_mae.png
        plus the mechanism figures (rank_to_cumloss_qm,
        rank_to_cumloss_scatter),
        (the city-level whole-curve error of each forecast line, per city-event),
        per-unit city_magnitude_curve_<code>.png and component_curves_<code>.png,
        curve_pred_metrics.csv, and raw_data/ (per-day city curves by method +
        the per-component α/L table + the plotted MAE table)
    4-cross_city_od_pred/        per city-event, one self-contained HTML slider map
        of the daily predicted / observed / difference OD flows

Adding a city-event means adding one CITY_EVENTS entry and providing its graph
pkl, its geo CSV, and its land-use (EPA Smart Location Database) and income
(ACS) caches under data/.  Geometry files are mandatory: loading raises
FileNotFoundError when a unit's geo CSV is absent.

Run
---
    python run_pattern_nmf.py
"""
import itertools
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares, minimize
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from config import (
    AGG_LEVEL, DATA_DIR, OUTPUT_DIR, SLOT_PER_DAY, SLOTS_ACTIVE,
)
from utils.pattern_analysis.graph_io import (
    load_graphs_trimmed, build_distance_array, build_income_array,
)
from utils.pattern_analysis.decomposition import select_segment_columns
from utils.pattern_analysis.nmf_pipeline import (
    build_city_matrices, decompose_city, decompose_city_context,
    nmf_quality_metrics, rank_cv_entry,
)
from utils.pattern_analysis.visualization import (
    vis_heatmap_temporal_signature,
    vis_city_mobility_curves, vis_mapping_pca,
    vis_cluster_function_graph,
    vis_cluster_function_heatmap,
    vis_nmf_rank_cv,
    vis_line_nmf_component_timeline, vis_heatmap_od_function,
    vis_line_resilience_curves,
    vis_heatmap_corr,
    vis_hist_function_entropy,
    vis_bar_component_distance, vis_bar_component_income,
    vis_heatmap_pair_transfer, vis_scatter_intensity_resilience,
    vis_exposure_vs_cumloss,
    vis_bar_cross_city_resi_pred, vis_scatter_city_pred, vis_bar_curve_mae,
    vis_curves_city_pred, vis_city_curves_grid,
    vis_component_curves_grid, vis_od_flow_slider_html,
    vis_rank_pred_vs_true, vis_rank_to_cumloss_qm, vis_qm_pred_vs_obs, vis_func_vs_time_distribution,
    vis_centered_spectrum_loo,
    vis_centered_distributions, vis_spread_vs_predictors,
    vis_centered_spectrum_schematic,
    vis_spread_concept, vis_mlb2_pca,
)
from utils.pattern_analysis.space_function import (
    share_lookup_from_landuse, build_od_function_matrix_soft,
)
from utils.pattern_analysis.component_features import (
    PERIOD_BANDS,
    temporal_features, functional_features, time_function_correlation,
    resilience_features, resilience_curves, component_function_entropy,
    spatial_features, socioeconomic_features, recovery_curve_features,
    daily_baselines,
)
from utils.pattern_analysis.ml_resilience import (
    cross_city_resilience,
)
from utils.data_processing.geo_loader import load_city_geo
from utils.data_processing.hurdat_exposure import load_track, city_exposure
from utils.data_processing.fetch_sld_landuse import (
    ensure_city_landuse_raw, classify_dominant_function, pooled_iwf_weights,
    CATEGORIES as SF_CATEGORIES,
)
from utils.data_processing.fetch_acs_income import (
    ensure_city_income_raw, load_city_income, ACS_DATA_DIR,
)

import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42,        # editable TrueType text in PDF exports
    'svg.fonttype': 'none',    # editable text in SVG exports
})


# ── Shared configuration — used by several steps ────────────────────────────────

# ── City-event registry (CITY_EVENTS) — one entry per unit.  Add a unit = add an entry. ──
# The single source of truth for per-unit params; the rest of the pipeline is global.
# Paths are block_group (AGG_LEVEL is the global resolution switch; SLD land-use only
# supports block_group).  Per-entry keys:
#   window/buffer/disaster — the trailing `window` days of the trimmed graph =
#       (window-buffer-disaster) normal + `buffer` pre-landfall + `disaster` (landfall +
#       recovery); `analysis_days` trims the raw sequence so landfall+14 sits at its end.
#       The 5-day buffer isolates preparation/evacuation behaviour from the clean normal
#       baseline (FM showed a +46% Saturday surge on Sep 24 and −19% on Tue Sep 27);
#       buffer slots stay in the NMF input but feed neither the temporal features nor
#       the resilience baselines/curves.
#   n_behaviors/l1_reg — per-unit NMF size and sparsity (tuned, see entry comments).
#   fit_segments — which segments FIT the NMF basis H; the full window is ALWAYS
#       projected onto that basis, so W/H shapes and n_nor/n_dis are unchanged
#       downstream.  ('normal','buffer') keeps disaster behaviour out of the basis.
#       When re-slicing, revisit n_behaviors/l1_reg and watch the near-zero-weight
#       component count in the projection diagnostics (fewer fit rows support fewer k).
#   context_aware/lambda_ctx — per-unit switch/strength for the context-aware solver
#       (OFF everywhere in production; even lambda 0 changes the solver when on).
#   ss_intensity/evac_level — the two per-event LEVEL covariates (see LEVEL_FEATURE_COLS).
#   key — the SLD/income filename prefix (city-level, shared across a city's events).
#   first_day_* — weekday of the normal-window start / of the landfall day (e.g. BR:
#       trailing window Aug 11 (Wed) – Sep 12 2021, disaster from Aug 29 (Sun, Ida);
#       FM: Sep 10 (Sat) – Oct 12 2022, disaster from Sep 28 (Wed, Ian)).
_WIN, _BUF, _DIS = 33, 5, 15          # shared 33-day window: 13 normal + 5 buffer + 15 disaster
CITY_EVENTS = [
    dict(code='BR_Ida', label='Baton Rouge', key='Baton_Rouge',
         graph='data/Baton_Rouge_Ida_2021_graph_intersection.pkl',
         geo={'block_group': 'data/Baton_Rouge_block_group_geo.csv'},
         analysis_days=151, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=12, l1_reg=0.5, filter_factor=0, ss_intensity=5,  # Ida @ BR ~Cat 2
         evac_level=0.091041,  # BG-pop-weighted HEvOD 3-level evacuation strength (data/evacuation_orders)
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Wednesday', first_day_disaster='Sunday'),
    dict(code='FM_Ian', label='Fort Myers', key='Fort_Myers',
         graph='data/Fort_Myers_Ian_2022_graph_intersection.pkl',
         geo={'block_group': 'data/Fort_Myers_block_group_geo.csv'},
         analysis_days=44, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=6, l1_reg=0.13, filter_factor=0, ss_intensity=7,  # Ian @ FM ~Cat 4
         evac_level=1.603633,  # BG-pop-weighted HEvOD 3-level evacuation strength (data/evacuation_orders)
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Saturday', first_day_disaster='Wednesday'),
    # New city-events: 87-day (L-56 -> L+30) block_group graphs, landfall at day 56
    # -> analysis_days = 71.  Per-city k / l1 are TUNED (cum_loss LOO-CV, 2026-06-29,
    # study nmf_loocv_cum_loss); filter_factor=0 = no OD filtering.  (The 2026-07-07
    # city-MAE re-tuning was tried and REVERTED — it overfit the n=4 validation.)
    dict(code='WM_Dorian', label='Wilmington', key='Wilmington',
         graph='data/Wilmington_Dorian_2019_graph_intersection.pkl',
         geo={'block_group': 'data/Wilmington_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=5, l1_reg=1.626, filter_factor=0, ss_intensity=5,  # Dorian @ Wilmington ~Cat 2 (true arrival)
         evac_level=0.814236,  # BG-pop-weighted HEvOD 3-level evacuation strength (data/evacuation_orders)
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Sunday', first_day_disaster='Thursday'),
    dict(code='WM_Isaias', label='Wilmington', key='Wilmington',
         graph='data/Wilmington_Isaias_2020_graph_intersection.pkl',
         geo={'block_group': 'data/Wilmington_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=5, l1_reg=0.315, filter_factor=0, ss_intensity=4,  # Isaias @ Wilmington ~Cat 1
         evac_level=0.018679,  # BG-pop-weighted HEvOD 3-level evacuation strength (data/evacuation_orders)
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Thursday', first_day_disaster='Monday'),
    dict(code='LC_Laura', label='Lake Charles', key='Lake_Charles',
         graph='data/Lake_Charles_Laura_2020_graph_intersection.pkl',
         geo={'block_group': 'data/Lake_Charles_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=10, l1_reg=0.549, filter_factor=0, ss_intensity=6,  # Laura @ Lake Charles ~Cat 3 (true arrival)
         evac_level=1.868990,  # BG-pop-weighted HEvOD 3-level evacuation strength (data/evacuation_orders)
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Sunday', first_day_disaster='Thursday'),
    # ── top-15 extension (2026-07-28): 12 city-events from the FEMA-ranked
    # disaster-MSA download (notebook main_20260727_download_top15_trajectories).
    # analysis_days / first_day_* derive from each storm's EVENT_BATCHES span
    # (graph day 0) and landfall date, verified programmatically; landfall+14
    # sits at the end as everywhere else.  n_behaviors=10 / l1_reg=0.5 are
    # UNTUNED defaults (l1 = median of the five tuned units; the per-city
    # LOO-CV tuner has not been run for these yet).  ss_intensity: at-arrival
    # SS category, NHC TCR best-track interpolation at closest approach +
    # local NWS obs (2026-07-28 research; same reading as the original five).
    # evac_level: BG-pop-weighted HEvOD 3-level strength — pipeline in
    # data/evacuation_orders, extended to the 12 events with the same
    # HEvOD-primary + verified-supplement discipline (Terrebonne/Ida is the
    # one supplement, tpcg.org release 1958).
    dict(code='HU_Ida', label='Houma', key='Houma',
         graph='data/Houma_Ida_2021_graph_intersection.pkl',
         geo={'block_group': 'data/Houma_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=10, l1_reg=0.5, filter_factor=0, ss_intensity=6,  # Ida @ Houma Cat 3 (borderline 4; eyewall)
         evac_level=2.000000,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Wednesday', first_day_disaster='Sunday'),
    dict(code='HM_Ida', label='Hammond', key='Hammond',
         graph='data/Hammond_Ida_2021_graph_intersection.pkl',
         geo={'block_group': 'data/Hammond_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=5, l1_reg=0.5, filter_factor=0, ss_intensity=4,  # Ida @ Hammond Cat 1 (borderline 2)
         evac_level=0.000000,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Wednesday', first_day_disaster='Sunday'),
    dict(code='SL_Ida', label='Slidell', key='Slidell',
         graph='data/Slidell_Ida_2021_graph_intersection.pkl',
         geo={'block_group': 'data/Slidell_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=6, l1_reg=0.5, filter_factor=0, ss_intensity=5,  # Ida @ Slidell Cat 2 (54 mi W of track)
         evac_level=0.000000,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Wednesday', first_day_disaster='Sunday'),
    dict(code='PG_Ian', label='Punta Gorda', key='Punta_Gorda',
         graph='data/Punta_Gorda_Ian_2022_graph_intersection.pkl',
         geo={'block_group': 'data/Punta_Gorda_block_group_geo.csv'},
         analysis_days=44, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=5, l1_reg=0.5, filter_factor=0, ss_intensity=7,  # Ian @ Punta Gorda Cat 4 (landfall hit)
         evac_level=1.365782,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Saturday', first_day_disaster='Wednesday'),
    dict(code='NA_Ian', label='Naples', key='Naples',
         graph='data/Naples_Ian_2022_graph_intersection.pkl',
         geo={'block_group': 'data/Naples_block_group_geo.csv'},
         analysis_days=44, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=5, l1_reg=0.5, filter_factor=0, ss_intensity=7,  # Ian @ Naples Cat 4 (40 mi WNW)
         evac_level=0.442624,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Saturday', first_day_disaster='Wednesday'),
    dict(code='NP_Ian', label='North Port', key='North_Port',
         graph='data/North_Port_Ian_2022_graph_intersection.pkl',
         geo={'block_group': 'data/North_Port_block_group_geo.csv'},
         analysis_days=44, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=7, l1_reg=0.5, filter_factor=0, ss_intensity=7,  # Ian @ North Port Cat 4 (weakening inland)
         evac_level=0.370944,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Saturday', first_day_disaster='Wednesday'),
    dict(code='DT_Ian', label='Deltona', key='Deltona',
         graph='data/Deltona_Ian_2022_graph_intersection.pkl',
         geo={'block_group': 'data/Deltona_block_group_geo.csv'},
         analysis_days=44, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=8, l1_reg=0.5, filter_factor=0, ss_intensity=3,  # Ian @ Deltona TS (inland crossing)
         evac_level=0.000000,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Saturday', first_day_disaster='Wednesday'),
    dict(code='CH_Dorian', label='Charleston', key='Charleston',
         graph='data/Charleston_Dorian_2019_graph_intersection.pkl',
         geo={'block_group': 'data/Charleston_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=5, l1_reg=0.5, filter_factor=0, ss_intensity=6,  # Dorian @ Charleston Cat 3 (borderline 2, offshore)
         evac_level=1.496924,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Sunday', first_day_disaster='Thursday'),
    dict(code='MB_Dorian', label='Myrtle Beach', key='Myrtle_Beach',
         graph='data/Myrtle_Beach_Dorian_2019_graph_intersection.pkl',
         geo={'block_group': 'data/Myrtle_Beach_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=7, l1_reg=0.5, filter_factor=0, ss_intensity=5,  # Dorian @ Myrtle Beach Cat 2 (offshore)
         evac_level=0.201570,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Sunday', first_day_disaster='Thursday'),
    dict(code='HH_Dorian', label='Hilton Head', key='Hilton_Head',
         graph='data/Hilton_Head_Dorian_2019_graph_intersection.pkl',
         geo={'block_group': 'data/Hilton_Head_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=10, l1_reg=0.5, filter_factor=0, ss_intensity=6,  # Dorian @ Hilton Head Cat 3 (offshore)
         evac_level=1.861403,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Sunday', first_day_disaster='Thursday'),
    dict(code='DA_Sally', label='Daphne', key='Daphne',
         graph='data/Daphne_Sally_2020_graph_intersection.pkl',
         geo={'block_group': 'data/Daphne_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=5, l1_reg=0.5, filter_factor=0, ss_intensity=5,  # Sally @ Daphne Cat 2 (landfall county)
         evac_level=0.325377,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Saturday', first_day_disaster='Wednesday'),
    dict(code='LC_Delta', label='Lake Charles', key='Lake_Charles',
         graph='data/Lake_Charles_Delta_2020_graph_intersection.pkl',
         geo={'block_group': 'data/Lake_Charles_block_group_geo.csv'},
         analysis_days=55, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=10, l1_reg=0.5, filter_factor=0, ss_intensity=4,  # Delta @ Lake Charles Cat 1 (weakening at closest approach)
         evac_level=2.000000,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Monday', first_day_disaster='Friday'),
]

OUTPUT_PLOTS = OUTPUT_DIR    # flat since 2026-08-05 (was outputs/nmf/)

# Per-city block-group space-function data (EPA Smart Location Database).
SPACE_FUNCTION_DIR = os.path.join(DATA_DIR, 'space_function')

# Socioeconomic block: per-component ACS median household income (table B19013,
# 5-year — block-group income needs the 5-year dataset).
# INCOME_ENDPOINT_MODES — per-flow income aggregations to compute; each becomes a
# median_income_<mode> column (a bar figure + a right-hand heatmap column):
#   'combined' = loading-weighted mean of the flow's origin+destination income,
#   'origin'   = the flow's origin block-group income only.
# ACS 5-year vintage: 2013–2019 use 2010-census block-group boundaries (which the
# mobility/SLD geo data uses → 100% match); 2020+ use 2020 boundaries (only ~64%
# match here).  2019 is the latest 2010-vintage 5-year.
ACS_INCOME_YEAR       = 2019
INCOME_ENDPOINT_MODES = ['combined', 'origin']

# Cross-tab axes are the functional categories plus 'Mix'.  Unknown and
# unmatched endpoints are dropped.  Order defines the heatmap rows (origin)
# and columns (destination).
# (AXIS_CATEGORIES = SF_CATEGORIES + ['Mix'] is gone: the SOFT shares keep
# every block group's full composition, so there is no Mix bucket to axis.)

# Functional features come from the O×D cross-tab with Mix and Unknown dropped
# and the six categories renormalised, split by flow direction.  Shares are
# full row/column sums, so same-function diagonal flow counts on both sides.
#   share_from_<cat> — outflow side, the fraction departing from function <cat>
#   share_to_<cat>   — inflow side, the fraction arriving at function <cat>
# These are the RAW columns; every model reads the MERGED func_<cat> below
# (the direction-split pair had no remaining reader once the rank channel was
# unified on 2026-08-28, so the FUNC_COLS list and its MERGE_FUNC_DIRECTIONS
# switch went with it).

# The MERGED functional shares the correlation blocks analyse: func_<cat> =
# share_from_<cat> + share_to_<cat>, the total share of a component's flow
# touching function <cat>.  The merge happens BEFORE the correlation, so each
# category yields ONE coefficient describing the whole functional exposure;
# correlating the two directions separately reported two coefficients of two
# halves of the same quantity and invited reading a direction effect into what
# is a single exposure.  Same definition the cross-city and Ridge blocks use
# under merge_func_directions=True, so `func_<cat>` means one thing repo-wide.
FUNC_MERGED_COLS = [f'func_{c}' for c in SF_CATEGORIES]


def _with_merged_func(feats):
    """Copy of a component feature table with the merged func_<cat> columns added."""
    out = feats.copy()
    for c in SF_CATEGORIES:
        out[f'func_{c}'] = out[f'share_from_{c}'] + out[f'share_to_{c}']
    return out


# ── The CITY-TOTAL estimator's COMPONENT (phi) feature set, defined ONCE ─────
# The city-total estimator has two stages.  phi (the component ridge) predicts
# each component's standardized cum_loss from the columns named below; its
# weight_normal aggregate is the city score s.  rho (_city_total_from_scores)
# then scores [s, r0_city, GDP] and variance-matches onto day-equivalents.  Two
# places run this estimator: STEP 6's aggr_denorm (the city_total/ headline
# figure) and STEP 7's _city_score/_city_total_prediction (the level the curve
# forecast is shifted onto).  They must be the SAME estimator — the quantile
# mapping shifts onto the number the headline figure reports — so both read the
# constant below for phi AND call _city_total_from_scores for rho; neither the
# feature set nor the rho math can drift between them.
#
# phi's set is the 2026-08-12 sandbox result: the 9 base predictors, the 15
# pairwise func-func products, and the CITY-level r0.
#   interactions — which functions lose activity together is not the sum of the
#     separate shares; adding the products moved the city LOO R2 from +0.17 to
#     +0.41 and survived a permutation test on the shuffled component target
#     (P = 0.003, 300 draws), so it is signal, not 24-features-on-13-cities.
#   r0_city — 86% of the component r0's variance is WITHIN city, and phi
#     aggregates the component predictions by weight_normal before scoring, so
#     that within-city signal is averaged away before it reaches the city
#     number; a city-constant r0 moves the whole city instead.  It enters BOTH
#     phi (here) and rho (as the [s, r0_city, GDP] second term).  With rho a
#     bare scalar-s variance match this scored honest LOO R2 +0.47; giving rho
#     the [r0_city, GDP] terms lifts it to +0.58 (MAE 0.88, jackknife floor
#     +0.50) — see _city_total_from_scores.
FUNC_X_COLS = [f'func_{a}_X_{b}'
               for i, a in enumerate(SF_CATEGORIES)
               for b in SF_CATEGORIES[i + 1:]]
CITY_TOTAL_FEATURE_COLS = (FUNC_MERGED_COLS
                           + ['mean_distance', 'median_income_combined']
                           + FUNC_X_COLS + ['r0_city'])
CITY_TOTAL_POOLED_COLS = list(CITY_TOTAL_FEATURE_COLS)   # all pooled-train z


def _with_city_total_feats(feats):
    """Component table extended with everything CITY_TOTAL_FEATURE_COLS names:
    the merged func_<cat> shares, their 15 pairwise products, and the
    weight_normal-weighted city mean of r0 (constant within the city).  The
    component-level r0 is left untouched for the channels that still read it."""
    out = _with_merged_func(feats)
    for i, a in enumerate(SF_CATEGORIES):
        for b in SF_CATEGORIES[i + 1:]:
            out[f'func_{a}_X_{b}'] = out[f'func_{a}'] * out[f'func_{b}']
    w = out['weight_normal'].to_numpy(dtype=float)
    wsum = float(w.sum())
    out['r0_city'] = (float(w @ out['r0'].to_numpy(dtype=float) / wsum)
                      if wsum > 0 else float(out['r0'].mean()))
    return out


# ── The city-total ρ stage: a city-level covariate (metro GDP) and the estimator
# The component ridge φ (on CITY_TOTAL_FEATURE_COLS) predicts each component's
# standardized cum_loss; the weight_normal aggregate of those predictions is the
# city score s.  The ρ stage below scores each unit on [s, r0_city, GDP] and
# variance-matches onto day-equivalents (see _city_total_from_scores).
#   GDP — the metro's 2019 real GDP in $thousands (BEA CAGDP2, LineCode 1 "All
#     industry total", summed over the CBSA's counties).  A PRE-EVENT normal-
#     period static (the hurricanes span 2019-2022), so no disaster information
#     enters it; it is an exposure/size proxy — small metros sit fully inside the
#     storm footprint, large ones mostly outside — and correlates negatively with
#     cum_loss (metro size, not wealth: per-capita GDP is not significant).
#     Frozen under data/msa_static/ (provenance: build_msa_features.py), gitignored
#     runtime data like the graphs and ACS caches.
MSA_STATIC_CSV = os.path.join('data', 'msa_static', 'msa_features.csv')
CITY_TOTAL_RIDGE_ALPHAS = np.logspace(-3, 3, 25)
_MSA_GDP = None


def _msa_gdp_table():
    """{code -> metro 2019 real GDP, $thousands}, read once from MSA_STATIC_CSV.
    Lazy so importing the module never touches the (gitignored) data tree."""
    global _MSA_GDP
    if _MSA_GDP is None:
        df = pd.read_csv(MSA_STATIC_CSV).set_index('code')
        _MSA_GDP = {c: float(df.loc[c, 'gdp_2019_kusd']) for c in df.index}
    return _MSA_GDP


def _city_total_from_scores(s_held, x_held, s_train, x_train, gt_train):
    """The ρ stage of the city-total estimator: map component-aggregate city
    scores to a city cum_loss in day-equivalents.  Shared by STEP 6 (the
    city_total/ headline figure) and STEP 7 (the level the curve forecast is
    shifted onto), so the two cannot drift.

    Inputs, for the held unit and every TRAINING unit: the weight_normal
    aggregate `s` of that unit's STANDARDIZED component cum_loss predictions
    (from the component ridge φ on CITY_TOTAL_FEATURE_COLS), the unit's city-level
    r0 and metro GDP (x = [r0_city, GDP]), and — training only — the observed city
    cum_loss.  The training `s` values come from an INNER leave-one-out so the
    held unit never enters its own calibration.

    Two steps:
      DIRECTION  a ridge (alpha by inner GCV) of city cum_loss on the standardized
                 [s, r0_city, GDP] gives the linear combination that tracks the
                 city total; ridge, not OLS, so the three partly collinear
                 predictors give a stable direction on 12 units.
      SCALE      the fit is then variance-matched (de-shrunk): its fitted scores
                 are rescaled so their spread equals the training cum_loss spread,
                 c_hat = g_mean + (z_hat - z_mean) * sigma_g / sigma_z.  Ridge
                 deliberately shrinks the fit toward the mean (sigma_z < sigma_g),
                 which compresses exactly the extreme cities; the rescale removes
                 that shrinkage while keeping ridge's direction.  With the single
                 predictor [s] this is the reduced-major-axis (RMA / Model-II)
                 slope sigma_g / sigma_s and reproduces the pre-2026-08-14 scalar-s
                 estimator (honest LOO R2 +0.47); adding [r0_city, GDP] as city-
                 level terms lifts it to +0.58 (MAE 0.88, jackknife floor +0.50).

    Fewer than two usable training units -> NaN (the variance match is undefined;
    cannot happen at 12 training units)."""
    if len(s_train) < 2 or not np.isfinite(s_held):
        return np.nan
    X = np.column_stack([np.asarray(s_train, dtype=float),
                         np.asarray(x_train, dtype=float)])
    y = np.asarray(gt_train, dtype=float)
    xh = np.concatenate([[float(s_held)], np.asarray(x_held, dtype=float)])
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
    Xz = (X - mu) / sd; xhz = (xh - mu) / sd
    m = RidgeCV(alphas=CITY_TOTAL_RIDGE_ALPHAS).fit(Xz, y)
    zt = m.predict(Xz)
    zh = float(m.predict(xhz.reshape(1, -1))[0])
    sdz = float(zt.std()) or 1.0
    return float(y.mean() + (zh - zt.mean()) * (float(y.std()) / sdz))

# The resilience target.  cum_loss is computed from the relative-activity curve
# r, where r(d) = the component's daily total on disaster day d divided by its
# weekday/weekend-matched pre-disaster baseline, 3-day smoothed (see
# component_features.resilience_features).  r = 1 means the normal level.
#   cum_loss        — Σ (1 − r), the NET unclipped cumulative deviation over the
#                     disaster window, in day-equivalents (above-baseline surges
#                     cancel drops, which makes it linear in r and additive
#                     across components — the property the city-level
#                     reconstruction relies on).  HIGHER = WORSE.
#   recovery_alpha  — the CLEAN recovery rate α of the surge-plus-relaxation
#                     model r(t) = L/(1 + (L/r0−1)e^(−α·t)) + B·t·e^(−b·t),
#                     anchored at the OBSERVED landfall-day value r(0) (the
#                     logistic base is the single-unit reduction of Li, Wang &
#                     Chen 2024's spatiotemporal decay dynamics; the pulse term
#                     is an externally-forced transient in the unit-hydrograph
#                     superposition tradition; component_features.
#                     recovery_curve_features).  A RATE: HIGHER = FASTER
#                     recovery = MORE resilient — the OPPOSITE direction to
#                     cum_loss.  "Clean" because the jointly-fitted companions
#                     absorb what used to contaminate the rate: recovery_level
#                     (L, the end level), surge_strength (B, the SIGNED
#                     transient amplitude: >0 above-baseline surge, <0
#                     late-deepening dip) and surge_rate (b, pulse peak at 1/b
#                     days).  The companions RIDE IN THE FEATURE TABLES for the
#                     STEP-7 oracle but are NOT RES_COLS metrics and are NOT
#                     forecast: the STEP-7 prediction synthesizes the monotone
#                     L = 1, B = 0 slice because neither L nor B transfers
#                     across cities (2026-07-14 sandbox tournaments).  NaN when
#                     day-0 is a total stop, the curve is constant, or the
#                     joint fit fails the ALPHA_MIN_FIT_R2 quality gate below.
# The city-level cum_loss reconstruction (STEP 6) stays cum_loss-only; the
# curve prediction (STEP 7) forecasts recovery_alpha at L = 1, B = 0.
# HISTORY: the analyses previously also carried drop_depth, early_collapse,
# recovery_day and recovery_deficit (retired 2026-07-12; resilience_features
# still COMPUTES all five — kept long-term, see the note in
# component_features.resilience_features — so restoring one is just adding its
# name back to this list) and recovery_lambda, an exponential-recovery rate
# (added 2026-07-12, REMOVED 2026-07-13: the owner judged its definition
# flawed; recovery_alpha is its successor).  recovery_alpha itself started as a
# FIXED-baseline logistic (capped at 1), became free-plateau 2026-07-13 (the
# cap misfit the above-baseline majority), and became the surge-model clean
# rate 2026-07-14 (humps contaminated the free-plateau rate; the forecast
# tournament showed the clean rate transfers best).
RES_COLS = ['cum_loss', 'recovery_alpha']

# The INTRA-city func × cum_loss heatmaps (1-.../4-func_vs_resi) run on this
# restriction of RES_COLS.
# recovery_alpha was dropped from it 2026-08-04: at 5-12 components per unit its
# within-unit correlations never reached significance and its LOO R² was
# negative throughout (that Ridge is now retired), so the row documented noise.
# The variable itself is NOT
# retired — the cross-city feature tables, the STEP-7 rate transfer and the
# intensity scatter all still carry recovery_alpha via RES_COLS.
INTRA_RES_COLS = ['cum_loss']

# recovery-curve fit knobs (shared by the STEP-3 and STEP-5 feature tables;
# see component_features.recovery_curve_features for the full NaN rules).
ALPHA_MAX_RATE = 5.0     # upper bound of the fitted rate α (1/days)
ALPHA_MIN_FIT_R2 = 0.0   # QUALITY GATE: a joint fit explaining less variance
                         # of the observed curve than a constant (R² < 0) is
                         # dropped (supplies no training row)
ALPHA_MIN_STD = 0.02     # a curve with std below this is constant -> the
                         # parameters are jointly unidentifiable -> NaN
LEVEL_BOUNDS = (0.05, 5.0)   # bounds of the fitted end level L (relative to
                             # the normal baseline; > 1 = settles above normal)
SURGE_BOUNDS = (-10.0, 10.0)     # bounds of the SIGNED surge strength B
SURGE_RATE_BOUNDS = (0.1, 3.0)   # bounds of the surge rate b (pulse peaks at
                                 # 1/b days: between 0.33 and 10 days)
CURVE_PRED_SHRINK = 0.5  # STEP-7 plateau inversion: the solved-L target is
                         # c_base + SHRINK·(ĉ − c_base) — 0 reproduces the
                         # train-mean line (L = 1), 1 trusts the cum_loss
                         # prediction fully.  0.5 is a fixed halfway-shrinkage
                         # convention: per-fold tuning of this weight overfits
                         # at n = 4 inner folds (2026-07-14 nested-menu test).

# STEP-7 'pred' line variants (2026-07-21 sandbox comparison; all share the full
# quantile-mapped cum_loss prediction and differ ONLY in how each component's
# (α, L) is set from the shrunk target):
#   CURVE_PRED_SOLVER  'solve_L'      α pinned at the backbone mean ᾱ, L solved
#                                     exactly by brentq (the pre-2026-07-21
#                                     production solver; 15-day city-MAE 0.0706)
#                      'joint_alphaL' α AND L jointly least-squares-fitted to the
#                                     target, optionally plus λ·(L − line(α))²
#                                     pulling L toward the training α-L
#                                     relationship (pooled OLS L = a0 + b·log10 α
#                                     over the TRAINING units' ungated fits,
#                                     bound-pinned included).  CURRENT SETTING
#                                     λ = 0: unregularized, so the fit is
#                                     under-determined and its 15-day 0.0646 is
#                                     ERROR CANCELLATION (true-cum_loss 0.0778),
#                                     not a real gain; λ = 0.1 scored 0.0752.
#   CURVE_JOINT_LAMBDA regularization strength, read only by 'joint_alphaL'.
#   CURVE_ALPHA_BACKBONE 'surge'      ᾱ = pooled-train mean of the surge-model
#                                     CLEAN rate (production)
#                        'no_surge'   ᾱ from refitting the training curves with
#                                     the pulse term REMOVED (plain free-plateau
#                                     logistic) — the surge-contaminated rate;
#                                     worse at 15 days (0.0740) but the best
#                                     backbone at extended windows (T ≥ 20).
CURVE_PRED_SOLVER = 'joint_alphaL'   # joint (α, L) solve; the α-L-line ridge is
                                     # switched OFF by λ = 0 below.
CURVE_JOINT_LAMBDA = 0.0             # α-L regularization REMOVED (owner choice
                                     # 2026-08-04; was 0.1, and λ-insensitive at
                                     # ~0.0752 across 1e-4..10).
                                     # READ BEFORE QUOTING THE NUMBER: at λ = 0
                                     # the solve is under-determined — only
                                     # cum_loss constrains (α, L), so the free
                                     # direction absorbs prediction error.  The
                                     # 2026-07-21 sweep measured 15-day city-MAE
                                     # 0.0646 here against 0.0778 under TRUE
                                     # cum_loss, i.e. the apparent gain is error
                                     # CANCELLATION, not a better forecast.  The
                                     # solved α and L are individually no longer
                                     # identified and must not be read as
                                     # estimates of the recovery rate or plateau.
CURVE_ALPHA_BACKBONE = 'surge'

# ── STEP 1 parameters — the global land-use classification ──────────────────────

# On-the-fly classification knobs for the space-function data (TF-IWF
# reweighting, see utils/data_processing/fetch_sld_landuse.py).  The IWF exponent
# itself is GLOBAL_IWF_SCALE below — the pipeline classifies with ONE weight
# vector pooled over all city-events (see the recipe comment at GLOBAL_IWF_SCALE).
LANDUSE_WEIGHTING          = 'tf_iwf'   # 'tf_iwf' down-weights ubiquitous residential, 'raw_share' does not
LANDUSE_RESIDENTIAL_WEIGHT = 1.0        # Housing-unit to job equivalence; 1.0 treats one housing unit as one job
LANDUSE_DOMINANT_THRESHOLD = 0.4        # Top-category share needed for a label, otherwise 'Mix'.
                                        # Lower gives fewer Mix

# The GLOBAL land-use recipe — the ONLY TF-IWF recipe in the pipeline (the technical
# notes call it "config C").  One IWF weight vector is pooled over ALL city-events'
# block groups and classifies every city, so a functional share means the same thing in
# every city; this makes `func_<cat>` cross-city comparable (which unlocked the
# cross-city severity signal, LOO cum_loss ranking corr ~0.88) and, since 2026-07-12,
# the SAME classification also feeds the within-city functional analyses (the earlier
# per-city recipe was removed).  Two consequences to keep in mind:
#   * the pooled weights are TRANSDUCTIVE over ALL units including the held-out test
#     city (land-use features only, never resilience labels; a stricter per-fold
#     train-only IWF is a future refinement);
#   * ALL functional outputs — within-city included — depend on the whole registry, so
#     adding a city-event shifts every unit's labels slightly.
# GLOBAL_IWF_SCALE is the exponent generating the pooled weights and the intended
# OPTUNA FINE-TUNING TARGET (fine-tuning is deferred).
GLOBAL_IWF_SCALE = 2.0     # 2026-08-04 sweep under the SOFT shares (0..3 on the two
                           # pooled all_city heatmaps): a flat optimum spans 1.52-2.5;
                           # 2.0 is its centre (headline cells 0.502 -> 0.526, industrial
                           # x cum_loss -0.44 -> -0.49 vs 1.52).  HISTORY: 1.52 was the
                           # untuned default; the downstream city-MAE tuning (2026-07-07)
                           # OVERFIT its n=5 validation and was reverted.

# ── STEP 2 parameters — per-unit NMF decomposition ──────────────────────────────

# Context-aware NMF feature mode (used only when a unit's CITY_EVENTS entry sets
# context_aware=True; all five production entries currently have it OFF, giving
# the exact sklearn baseline).  When on, the spatial factor H is regularised
# toward a per-flow POI feature built from the endpoints' TF-IWF land-use shares
# (the same classification the O×D func block uses); the per-event strength is
# the unit's `lambda_ctx` (relative weight, auto-scaled by ‖X‖²/‖Y‖²).  See
# docs/technical_notes/1 §3.2/§3.3 for the formulation and magnitude reasoning.
FLOW_FEATURE_MODE = 'sum'    # 'outer' = C² joint O×D type, 'sum' = C-dim combined

# ── STEP 3 parameters — within-city component analyses ──────────────────────────

# Derived from config.  Never hardcode the interval here.
_INTERVAL_HOURS = 24 // SLOT_PER_DAY   # 3 for 3h data and 2 for 2h data

# ── Per-component feature columns used by the correlation blocks ──────────────

# Temporal features are computed from the pre-disaster part of W only (see
# utils/pattern_analysis/component_features.py).
#   weekday_ratio — mean weekday daily total over mean weekend daily total
#                   (weekday from FIRST_DAY_*_NORMAL).  Above 1 means
#                   weekday-dominated (commute-like) and below 1 means
#                   weekend-dominated (leisure-like).  The only temporal
#                   feature in the rank correlations.
# Two CATEGORICAL temporal features go to bar charts instead of correlations
# (time-of-day is not a monotone scale).  Components with weekday_ratio below
# WEEKEND_RATIO_THRESHOLD form a separate 'weekend' category in both, with no
# within-day breakdown; all other components are profiled on WEEKDAY days only.
#   peak_slot   — the weekday profile's argmax slot ('6-8h' … '20-22h')
#   peak_period — the weekday profile's strongest day-period band, width-
#                 corrected (morning_peak 6-10h / midday 10-16h /
#                 evening_peak 16-20h / night 20-22h)
WEEKEND_RATIO_THRESHOLD = 1.0
TIME_COLS = ['weekday_ratio']

# Extra single-value features appended (one cell each, no from/to split) to the
# RIGHT of the functional resilience-correlation heatmap: the component's
# loading-weighted mean flow distance (spatial) and the median household income
# for each endpoint mode (socioeconomic).  All must already be columns of `feats`.
EXTRA_CORR_COLS = ['mean_distance'] + [f'median_income_{m}' for m in INCOME_ENDPOINT_MODES]


# All per-component characteristics live under component_characteristics, one
# subfolder per characteristic type.
#   temporal/          signature heatmap + timeline (temporal factor W)
#   spatial/           per-component flow-distance figure (spatial factor H)
#   func/              per-component O×D functional cross-tabs
#   func_vs_temporal/  time × function correlation figures
# ── NMF decomposition-quality check ──────────────────────────────────────────
# Runs right after every retained unit's decomposition, on the FIT window only
# (normal+buffer — the data the basis is fit on; the disaster period is out of
# scope, no predictive consideration): absolute reconstruction error, per-slot
# distribution (KS) error, and the component-count health rule.  One figure per
# unit + a cross-city summary + raw CSV.  These are DIAGNOSTICS only — nothing
# optimises them.  k comes from the rank CV plus a manual reading of it (see
# EXCLUDED_CODES and the k-policy notes above CITY_EVENTS); reconstruction
# error cannot choose k,
# since it falls monotonically with k.
# 0-data holds the raw study-overview figures (city mobility curves); the
# analysis sections are numbered from 1 (bumped +1 on 2026-08-05 to free 0).
OUTPUT_DATA         = os.path.join(OUTPUT_PLOTS, '0-data')
# Everything about the factorisation lives under ONE parent: what the
# decomposition IS (the quality/rank-CV evidence) and what its components ARE
# (their characteristics) are two views of the same object, and the top-level
# list reads as one step per stage of the pipeline.
OUTPUT_DECOMP       = os.path.join(OUTPUT_PLOTS, '1-decomposition')
OUTPUT_QUALITY      = os.path.join(OUTPUT_DECOMP, '1-decomposition_quality')
OUTPUT_QUALITY_RAW  = os.path.join(OUTPUT_QUALITY, 'raw_data')

# ── Rank cross-validation (STEP 1) and the k policy it sets ──────────────────
# Held-out-entry CV masks a random share of the fit matrix's entries, refits
# without them and scores the masked cells, so — unlike reconstruction error,
# which only falls with k — the score has an interior optimum.  It runs over
# EVERY registry unit (its own results stay complete) and reports, per unit, the
# k at the curve's minimum plus the range of k within one/two standard errors.
RANK_CV_ENTRY_FRAC    = 0.10   # entries masked per repeat
RANK_CV_REPEATS       = 5      # independent masks; standard error = sd / sqrt(this)
RANK_CV_K_MAX         = 40     # capped further by each unit's matrix dimensions
RANK_CV_SEED          = 42
# Tolerance bands as multiples of the spread at the minimum; (name, multiple,
# use_standard_error).  All three are written to the CSV; the figure captions
# carry RANK_CV_FIGURE_BAND.
RANK_CV_BANDS         = [('k_1se', 1.0, True), ('k_2se', 2.0, True),
                         ('k_2sd', 2.0, False)]
RANK_CV_FIGURE_BAND   = 'k_2se'
# The CV curves depend on the graphs, the window/fit segments and l1_reg — NOT
# on n_behaviors, which the sweep supplies itself.  So the cached curves stay
# valid across k changes and are reused whenever they already cover every
# registry unit; delete raw_data/nmf_rank_cv_curves.csv to force a recompute
# (it costs ~25 minutes over 17 units).
RANK_CV_CURVES_CSV    = os.path.join(OUTPUT_QUALITY_RAW, 'nmf_rank_cv_curves.csv')

# Units DROPPED from every step after the rank CV: their 2-standard-error band
# tops out at k <= 2, i.e. the data supports at most two components, too few to
# describe a city's behavioural structure or to correlate anything against.
# All four are the smallest matrices in the registry (14-61 OD pairs).
EXCLUDED_CODES = frozenset({'HH_Dorian', 'LC_Laura', 'LC_Delta', 'HU_Ida'})
# Retained units take k = the TOP of the 2-standard-error band, floored at 5 so
# the within-city correlations keep a usable number of observations, with two
# manual overrides (SL_Ida, FM_Ian -> 6).  The resulting values live in each
# CITY_EVENTS entry's n_behaviors; this comment records where they came from
# (nothing reads the floor programmatically).

# ── K-policy TRIAL switch ────────────────────────────────────────────────────
# 'manual'      — the registry n_behaviors (the hand-read values): the
#                 production setting.
# 'rank_cv_min' — TRIAL: every unit takes k = the rank-CV curve's own minimum
#                 (k_min), and units whose k_min falls below K_MIN_TRIAL_FLOOR
#                 are dropped for the run — too few components to characterise.
#                 Nothing in the registry is edited; flip back to 'manual' and
#                 rerun to restore the production outputs.
K_POLICY          = 'manual'
K_MIN_TRIAL_FLOOR = 4
# How far main() runs.  'full' is production; 'intra_city' stops after the
# within-city characterisation (folder 1- complete) — the trial only asks
# whether the component characteristics survive at the smaller k, so the
# cross-city machinery stays untouched.
PIPELINE_SCOPE    = 'full'
# The X% of the component-size health check: a component whose weight
# ‖W_i‖·‖H_i‖ falls below this fraction of the largest component's weight
# counts as a too-large-k symptom.  Reported per unit as a pass/fail check; it
# is not a constraint anything optimises against.
NMF_MIN_COMP_FRAC   = 0.05

OUTPUT_CHAR         = os.path.join(OUTPUT_DECOMP, '2-component_characteristics')
OUTPUT_TEMPORAL     = os.path.join(OUTPUT_CHAR, '0-temporal')
# The two temporal views are split by figure type: the per-component W heatmap
# (one image showing every component at once) and the per-component timeline
# curves, which are read differently and are easier to browse apart.
OUTPUT_TEMPORAL_HM  = os.path.join(OUTPUT_TEMPORAL, 'heatmap')
OUTPUT_TEMPORAL_CV  = os.path.join(OUTPUT_TEMPORAL, 'curve')
OUTPUT_SPATIAL      = os.path.join(OUTPUT_CHAR, '7-spatial')
# CSV raw-data for the per-component flow-distance figure (kept out of the figure folder).
OUTPUT_SPATIAL_DIST_RAW = os.path.join(OUTPUT_SPATIAL, 'component_distance_raw_data')
# Socioeconomic: per-component ACS median household income bar figure + raw CSV.
OUTPUT_SOCIO        = os.path.join(OUTPUT_CHAR, '6-socioeconomic')
OUTPUT_SOCIO_RAW    = os.path.join(OUTPUT_SOCIO, 'component_income_raw_data')
OUTPUT_FUNC         = os.path.join(OUTPUT_CHAR, '1-func')
# CSV raw-data for the func figures, each in its own subfolder so the tables stay
# out of the figure folder; the folder name says which figure the CSV backs.
OUTPUT_FUNC_HM_RAW  = os.path.join(OUTPUT_FUNC, 'heatmap_od_functionality_raw_data')
OUTPUT_FUNC_ENT_RAW = os.path.join(OUTPUT_FUNC, 'hist_function_entropy_raw_data')
OUTPUT_FUNC_VS_TEMP = os.path.join(OUTPUT_CHAR, '3-func_vs_temporal')
# Split by SAMPLE, not by figure type: all_city/ holds everything computed over
# every unit at once, per_city/ the one-figure-per-unit versions.  The pooled
# figures are the ones with the statistical power (81 components vs 5-12), so
# they should not be buried among thirteen per-unit heatmaps.
OUTPUT_FVT_ALL      = os.path.join(OUTPUT_FUNC_VS_TEMP, 'all_city')
OUTPUT_FVT_PER      = os.path.join(OUTPUT_FUNC_VS_TEMP, 'per_city')
# Numeric tables sit in their own raw_data/ so the folder itself holds only
# figures (same convention as the other output folders).
OUTPUT_FUNC_VS_TEMP_RAW = os.path.join(OUTPUT_FVT_ALL, 'raw_data')

# Function vs the resilience CURVES: per-function stacks of per-component line
# figures (one figure per functional category, components ordered by combined
# functional share) — the full-window W timelines and the disaster r(d) curves.
OUTPUT_CHAR_RESIL    = os.path.join(OUTPUT_CHAR, '2-resilience_curves')
# Function (+ distance / income) vs the scalar resilience metric (cum_loss,
# INTRA_RES_COLS): Spearman heatmaps, split by sample like 3-func_vs_temporal —
# per_city/ one heatmap per unit, all_city/ the pooled version under the
# within-unit rank normalisation (+ raw_data/).  The pearson twin and the
# within-unit LOO Ridge were retired 2026-08-04 with the rest of the old
# intra-city resilience-correlation folder.
OUTPUT_FUNC_VS_RESI       = os.path.join(OUTPUT_CHAR, '4-func_vs_resi')
OUTPUT_RESI_CORR_RANK_PER = os.path.join(OUTPUT_FUNC_VS_RESI, 'per_city')
OUTPUT_RESI_CORR_RANK_ALL = os.path.join(OUTPUT_FUNC_VS_RESI, 'all_city')
OUTPUT_RESI_CORR_RANK_RAW = os.path.join(OUTPUT_RESI_CORR_RANK_ALL, 'raw_data')
OUTPUT_TL_BY_FUNC    = os.path.join(OUTPUT_CHAR_RESIL, 'line_component_timeline_by_func')
OUTPUT_RC_BY_FUNC    = os.path.join(OUTPUT_CHAR_RESIL, 'line_component_resilience_curves_by_func')

# ── STEP 4 parameters — pooled severity-vs-resilience ───────────────────────────


# ── STEP 5 parameters — the cross-city feature tables ───────────────────────────

# (A K_LOO_TEST constant used to pin the held-out unit's k at a fixed 10, on the
# grounds that a brand-new city would have no tuned k.  The rank CV removes that
# argument — it derives a k from the city's own matrix without any labels — so
# the held-out unit is now decomposed at its own n_behaviors like every other,
# and the constant is gone.)

# ── STEP 6 parameters — cross-city prediction ───────────────────────────────────

# Cross-city resilience prediction reconstructed to the CITY level (predicted vs
# ground-truth cum_loss per city-event).  Only for pearson + multi_city_std.
OUTPUT_CROSS_CITY_RESI_PRED = os.path.join(OUTPUT_PLOTS, '2-cross_city_resi_pred')
# Two sibling views of the same STEP-6 transfer, each with its own raw_data/:
#   city_total/    the CITY-LEVEL cum_loss prediction (one number per unit),
#                  with decomp_pred_aggr_denorm/<model>/ per predictor
#   component_rank/ the per-COMPONENT rank transfer; the folder name comes from
#                  CROSS_CITY_METHOD_STD.  Two metrics live here and are NOT
#                  interchangeable: the leave-one-out scatters and the pairwise
#                  transfer heatmap report SPEARMAN rho (this channel predicts
#                  an ordering), while loo_cross_city_r2_baseline.csv stays R²
#                  as the pipeline-wide headline number.
OUTPUT_CITY_TOTAL = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED, 'city_total')
#   centered_distribution/ the third sibling: the CENTRED loss spectrum (the
#                  within-unit inequality of component cum_loss, level removed).
#                  That spectrum is what STEP 7's quantile mapping transfers
#                  across cities, so this folder measures that transfer on its
#                  own, before the ordering and the city-total anchor are
#                  applied.  raw/ holds its metrics table.
OUTPUT_CENTERED_DIST = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED,
                                    'centered_distribution')
# Disaster (Saffir-Simpson arrival intensity) vs cum_loss scatter, all
# city-events pooled (not split into train/test) — filed with the other
# per-component characteristic lenses.  The x axis is a per-EVENT constant, so
# what it can speak to is between-event variation; the panel carries that
# caveat in its reading, not in its location.
OUTPUT_DISASTER_VS_RESIL = os.path.join(OUTPUT_CHAR, '5-disaster_vs_resi')
# Three definitions of "how hard was this city hit" share that folder, because
# which definition one picks changes the answer and the figure should show that
# rather than settle it:
#   ss_intensity   the STORM's Saffir-Simpson category at closest approach —
#                  the registry value, compiled from NHC TCRs.  Overstates the
#                  exposure of every city the centre missed (its provenance note
#                  says so: Baton Rouge is coded Cat 2 and saw ~TS winds).
#   local_wind_kt  the wind band the city actually sat in, read off the best
#                  track's QUADRANT WIND RADII at closest approach.
#   track_dist_km  closest-approach distance, pure geometry — the only one that
#                  separates Ian's four cities, all of them "Cat 4".
#   evac_level     already in the registry: population-weighted evacuation
#                  strength, i.e. the exposure the authorities acted on.
# The last three are derived per run; HURDAT2 is the single source for the two
# track-based ones (see utils/data_processing/hurdat_exposure.py).
HURDAT2_BASIN_IDS = {'Ida': 'AL092021', 'Ian': 'AL092022', 'Dorian': 'AL052019',
                     'Isaias': 'AL092020', 'Laura': 'AL132020',
                     'Sally': 'AL192020', 'Delta': 'AL262020'}


# Cross-city transfer split — the ONE knob you control.  Lists of city-event codes
# (e.g. 'BR_Ida', 'FM_Ian'), which are the per-unit `code`/tag below.  The cross-city
# step goes ONLY pooled(train) -> each test unit.  If train and test are the SAME set
# it becomes a pooled leave-one-component-out instead.  A unit in neither list is
# still decomposed/characterized but excluded from the cross-city step.  Both sides
# are flexible.  None -> the cross-city step is skipped (with a warning).
# NOTE: there is no fixed train/test list.  Every cross-city step is
# leave-one-unit-out: main() builds split={'train': rest, 'test': [held]} per
# fold, so each retained unit trains every fold it is not the held-out one of.
# (A CROSS_CITY_SPLIT constant used to live here for the retired downstream
# tuner; main() never read it, and it was removed to stop it misleading.)

# Cross-city TARGET standardization is PAIRED one-to-one with the method — each method
# produces exactly ONE version, written to cross_city_resi_pred/<label>/:
#   'spearman' -> 'within_unit'  (single-city std) -> component_rank/: ranks are
#       computed within each unit, so only the level-robust within-unit standardization
#       is semantically consistent (a pooled z-score of per-unit ranks would fake a
#       cross-city level).
# The 'pooled_train' (multi-city std) mode has no entry here any more, but it is
# very much alive: raw values standardized on the pooled TRAINING units keep the
# absolute level, which the LEVEL / POOLED features, the city-level
# reconstruction (resi_pred) and the STEP-7 curve prediction all rely on — they
# request it directly rather than through this dict.
# Modes are cross_city_resilience's target_std values (see its docstring).  The pairing
# is ENFORCED: analysis_cross_city / analysis_cross_city_pairs raise on a mismatching
# explicit target_std and resolve target_std=None / subdir=None from this dict.
# Only the RANK path still produces a cross-city LOO output folder.  The
# raw-value counterpart (pearson + pooled_train -> cross_city_pred_raw_value)
# was retired on 2026-08-04: its per-fold scatters and R² matrices duplicated
# the rank path's story at a standardization the cross-city transfer does not
# rely on.  The pooled_train standardization ITSELF is untouched — it is still
# how analysis_cross_city_resi_pred (city-level cum_loss) and the STEP-7 curve
# prediction call cross_city_resilience; it simply no longer gets a folder of
# its own per-fold diagnostics.
CROSS_CITY_METHOD_STD = {
    'spearman': ('within_unit', 'component_rank'),
}

# LEVEL features: per-event CONSTANT covariates.  Two are carried: the Saffir-Simpson
# arrival intensity `hurricane_intensity` (8-level scale 1=Extratropical .. 3=Trop.Storm,
# 4=Cat1 .. 8=Cat5) and `evac_level`, the BG-population-weighted mean of the HEvOD 3-level
# ordinal evacuation strength (0 none / 1 voluntary / 2 mandatory) per city-event.  evac_level
# varies across same-city events (WM_Dorian 0.81 vs WM_Isaias 0.02) whereas the static POI /
# income covariates do not, so it carries a per-event severity signal those cannot; it cut the
# city cum_loss MAE 0.499 -> 0.270 of the then-current aggregate-then-denormalize
# reconstruction (sandbox-validated 2026-07-08; that PRE-r0 variant was removed from
# STEP 6 on 2026-08-04, so the number is provenance, not a property of today's
# features+r0 aggr_denorm strategy).  Within-
# unit standardization would zero a constant column, so both are standardized on the POOLED
# TRAIN instead, and ONLY enter the model in the 'pooled_train' (multi-city) mode.  In
# 'within_unit' (single-city) mode they are not used (single-city results are unchanged).
# The level covariates feed the COMPONENT-level path only (aggr_denorm, kNN(sigma), the
# LOO-R² and pairwise cross-city analyses); the decomposition-free city baseline does NOT
# take evac_level — over its low-dim cosine vector it is collinear with intensity and
# reshuffled the n=5 neighbors unhelpfully (sandbox: both worsened; the other affected
# path, the city-level "D+city" reconstruction, was removed 2026-08-04).  See
# cross_city_resilience's level_feature_cols.
LEVEL_FEATURE_COLS = ['hurricane_intensity', 'evac_level']

# POOLED features: func_<cat>, mean_distance AND median_income_combined are all
# standardized on the pooled TRAIN (cross-city) in 'pooled_train' mode, so they carry
# cross-city LEVEL.  Pooling func is only coherent BECAUSE func uses the
# global TF-IWF above (with a per-city IWF, pooling func injects each city's own
# normalization — that config was tried and is worse).  Income (USD, loading-weighted per component) is a
# raw magnitude comparable across cities, like mean_distance; its signal lives in the
# WITHIN-city variation across components (the static city means are nearly identical
# across these city-events), which only the decomposition exposes — adding it cut the
# city MAE 0.953 -> 0.499 of the then-current PRE-r0 reconstruction (sandbox-validated
# 2026-07-08; provenance only — that variant left STEP 6 on 2026-08-04).  In
# 'within_unit' (single-city) mode these are left within-unit.  See cross_city_resilience.
POOLED_FEATURE_COLS = ([f'func_{c}' for c in SF_CATEGORIES]
                       + ['mean_distance', 'median_income_combined'])

# The RANK channel's predictor.  It ran ridge and cosine-kNN side by side into
# component_rank/<model>/ from 2026-08-04 (a leave-one-out ablation put ridge
# ahead on average, mean test R2 +0.36 vs +0.28 and better on 8 of 13 units,
# but riskier on the largest unit: BR_Ida at k=12 went +0.10 -> -0.15 while
# cosine-kNN's bounded, shrunken predictions never blow up).  The cosine-kNN
# folder was RETIRED 2026-08-28 — the rank channel reports ridge alone.  The
# two-predictor comparison itself is NOT gone: it lives at the city level,
# where CITY_TOTAL_DECOMP_MODELS still runs both through the rho stage and the
# headline bar figure shows both.
#
# With one predictor left there is no per-model folder any more: the rank
# channel writes straight into component_rank/, and every downstream mechanism
# figure (the mapping PCA, the function graph, the cluster-restricted variant)
# reads that folder.  Reinstating a second predictor means reinstating the
# subfolder level with it.
#
# THE RANK CHANNEL, DEFINED ONCE (unified 2026-08-28).  Predicting a within-city
# cum_loss ORDERING used to exist twice: STEP 6 fitted 7 columns on every other
# city, STEP 7 fitted those 7 plus r0 on the held city's estimated cluster, and
# the two disagreed by +0.09 mean Spearman for reasons that had nothing to do
# with the question either was asking.  There is now ONE definition -- these
# three constants and rank_predict() below -- and every rank prediction in the
# pipeline goes through it: the LOO sweep, the pairwise transfer heatmap, the
# cluster-restricted diagnostic, the mapping-direction PCA and the STEP-7
# ordering channel.  Change the rank prediction here and it changes everywhere.
RANK_MODEL = 'ridge'
# 23 columns since 2026-08-31: the 15 pairwise func PRODUCTS joined the list.
# Products are computed on the RAW merged shares (rank_merge_feats) and only
# then ranked within the city -- the rank of a product is not the product of
# ranks.  Measured on the captured production tables, pooled all-12 training:
# mean test Spearman +0.587 -> +0.633, median +0.700 -> +0.786, 8 of 13 folds
# up and 2 down.  That ties the old cluster-restricted 8-feature channel
# (+0.637) without depending on the Louvain partition (modularity +0.026)
# that channel trained on -- which is why the cluster restriction retired
# with this adoption (see RANK_TRAIN_SCOPE).
RANK_FEATURE_COLS = ([f'func_{c}' for c in SF_CATEGORIES]
                     + ['mean_distance', 'r0'] + FUNC_X_COLS)

# (PARTITION_FEATURE_COLS retired 2026-08-31 with the cluster restriction it
# served: nothing trains on the Louvain partition any more, so the pairwise
# heatmap simply runs the rank channel's own list and its partition is a
# DISPLAY ordering.  The 2026-08-28 measurement that motivated the separate
# r0-free list -- r0 flattens the community structure, modularity +0.026 ->
# -0.043 -- still holds and is the reason the partition must not be read as a
# training recipe.)

# The predictors compared at the CITY level: aggregate each one's standardized
# component scores into s, then put s through the identical rho stage, so the
# figure compares the PREDICTOR and nothing else.  Independent of the rank
# channel above — this is why dropping cosine-kNN from component_rank/ leaves
# the headline bar figure's cosine-kNN bar standing.  (Model choices elsewhere
# are pinned at their own sites: the legacy kNN(sigma) reference and the raw
# pooled channel hardcode 'cosine_knn'; the STEP-7 channels read
# CURVE_PRED_CITY_MODEL, and the rank channel reads RANK_MODEL.)
CITY_TOTAL_DECOMP_MODELS = {'cos_KNN': 'cosine_knn', 'ridge': 'ridge'}


# ── STEP 7 parameters — cross-city curve prediction ─────────────────────────────

# Everything the STEP-7 curve prediction writes goes here.
OUTPUT_CURVE_PRED = os.path.join(OUTPUT_PLOTS, '3-cross_city_curve_pred')

# STEP-7 spatial view: per held city, an interactive slider map of the daily
# PREDICTED OD flows (forecast curves x component baselines x H), the observed
# flows, and their difference.  The spatial pattern per component is the test
# decomposition's own H (descriptive, not predicted); only the time dimension
# is forecast.  One self-contained HTML per city-event under
# cross_city_od_pred/<code>/.  OD_MAP_TOP_ARCS caps the arcs drawn per
# day-and-view (kept by |flow|) so the embedded data stays a few MB.
OUTPUT_OD_PRED = os.path.join(OUTPUT_PLOTS, '4-cross_city_od_pred')
CURVE_OD_MAPS = True
OD_MAP_TOP_ARCS = 600

# Which predictor each of STEP-7's two transfer channels runs (2026-08-04,
# when both consumers were re-wired to the improved STEP-6 recipes):
#   city total — the aggr_denorm strategy on pooled features + r0.  ridge is
#     the only variant that calibrates at 13 units (LOO R² +0.17, MAE 1.26 vs
#     -0.46 / 1.74 for cosine-kNN), so the quantile mapping's location shift
#     anchors on it.
#   rank ordering — the component_rank recipe (func + mean_distance, income
#     dropped 2026-08-04) plus STEP-7's own observed-r0 predictor.  ridge leads
#     the rank channel on average (Spearman mean 0.550 vs 0.496, median 0.600
#     vs 0.543; not significant at n=13, and cosine-kNN stays better on
#     SL_Ida, 0.543 vs 0.086) — chosen for consistency with the city channel.
CURVE_PRED_CITY_MODEL = 'ridge'
# (the rank channel's predictor is RANK_MODEL; see the block above)

# Training scope for the RANK channel (STEP-7 curve prediction and the
# component_rank headline): 'pooled' trains every fold on all 12 reference
# cities; 'cluster' would restrict to the held city's ESTIMATED transfer
# cluster (the machinery -- rank_cluster_context / rank_train_pool /
# _estimate_held_cluster -- is kept for a revert).  A revert is NOT one flag,
# though: the pairwise heatmap now fits the full 23-column list, so the
# partition on disk is cut WITH r0 -- the configuration measured on
# 2026-08-28 to flatten the community structure (assignment 10/13 -> 8/13
# already at the current partition).  Reinstating the 2026-08-12 behaviour
# means also restoring the r0-free partition list for the heatmap.  Back to 'pooled'
# 2026-08-31, together with the 23-column RANK_FEATURE_COLS: wide features
# and narrow training pools are SUBSTITUTES, not complements.  Measured on
# the captured production tables (23 features): pooled +0.633 mean Spearman,
# the transfer-Louvain restriction +0.290, co-riding k-means partitions at
# k=2/3/4 +0.482/+0.446/+0.133 -- a 23-column ridge on a 2-6-city pool
# (10-35 components) starves, and the smaller the cluster the worse.  The
# 8-feature cluster-restricted channel this replaces scored +0.637; the
# pooled 23-feature channel matches it (difference 0.004 at n=13) with one
# fewer fragile dependency.
RANK_TRAIN_SCOPE = 'pooled'

# Minimum usable rows per unit-and-metric inside the cross-city engine, for
# BOTH the STEP-6 analyses and the STEP-7 curve prediction.  The engine
# default (MIN_ROWS = 8) guards a WITHIN-city Ridge and is too high here for two
# separate reasons: recovery_alpha is NaN for every component whose curve starts
# at the baseline or never left it, so no unit keeps 8 usable α rows; and since
# the rank CV set k per unit (floor 5), most units no longer HAVE 8 components at
# all.  EVERY cross_city_resilience call must pass this explicitly — the one that
# did not (the city-level cum_loss fold) silently inherited 8 and dropped 11 of
# 13 units from that figure until 2026-08-03.
CROSS_CITY_MIN_ROWS = 5

# Rows of the LOO scatter grid.  13 units go 5/5/3 with the short row
# centred, which stays wider than tall (slide-shaped) while keeping the
# panels large enough that the type still reads once the figure is
# scaled to slide width -- a 2-row band puts one row's tick labels into
# the next row's title.
CROSS_CITY_SCATTER_GRID_ROWS = 3


# ── STEP 1 — global land-use classification ─────────────────────────────────────

def _global_landuse_classification(units):
    """The pipeline's ONE land-use classification, shared by every step (the technical
    notes call the recipe "config C"), so `func_<cat>` means the same thing in every
    city.  Pools the raw-SLD score mass over ALL units' block groups (transductive;
    land-use features only, no resilience labels), computes the global IWF weights at
    GLOBAL_IWF_SCALE, then classifies each city's block groups with those SAME weights.
    Returns (iwf_vector, {code: classified landuse DataFrame with share_<cat> +
    dominant_category}, {code: aggr_id -> CONTINUOUS share-vector lookup} — the
    SOFT scheme, 2026-08-04: components aggregate every endpoint's full TF-IWF
    composition instead of one dominant label, so no flow is lost to a Mix
    bucket (the hard labels dropped ~57% of it) and there is no 0.4-threshold
    cliff.  dominant_category is still computed for the per-city print).
    NOTE: two events of one city share a raw SLD file, so that city's block groups enter
    the pool once per event (matches the validated experiment).  Because the pool spans
    the whole registry, adding a city-event shifts every unit's labels slightly."""
    os.makedirs(SPACE_FUNCTION_DIR, exist_ok=True)
    dfs = {}
    for code, u in units.items():
        raw_csv = os.path.join(SPACE_FUNCTION_DIR, f"{u['cfg']['key']}_block_group_sld_raw.csv")
        assert ensure_city_landuse_raw(u['label'], u['gdf']['aggr_id'].tolist(), raw_csv) is not None
        dfs[code] = pd.read_csv(raw_csv)
    iwf = pooled_iwf_weights(list(dfs.values()), residential_weight=LANDUSE_RESIDENTIAL_WEIGHT,
                             iwf_scale=GLOBAL_IWF_SCALE)
    print("  [global IWF] pooled over all city-events (scale="
          f"{GLOBAL_IWF_SCALE}): "
          + ", ".join(f"{c}={w:.3f}" for c, w in zip(SF_CATEGORIES, iwf)))
    landuse_by_code, lookups = {}, {}
    for code, df in dfs.items():
        lu = classify_dominant_function(
            df, residential_weight=LANDUSE_RESIDENTIAL_WEIGHT, weighting=LANDUSE_WEIGHTING,
            dominant_threshold=LANDUSE_DOMINANT_THRESHOLD, iwf=iwf)
        landuse_by_code[code] = lu
        lookups[code] = share_lookup_from_landuse(lu, SF_CATEGORIES)
    return iwf, landuse_by_code, lookups


# ── STEP 3 — within-city analysis helpers (one per analysis block) ──────────────

def _lambda_tag(lambda_ctx):
    """Filename-safe context-strength tag, e.g. 'lambda0', 'lambda0.1', or
    'baseline' when context-aware NMF is off (lambda_ctx is None)."""
    return f"lambda{lambda_ctx:g}" if lambda_ctx is not None else "baseline"


def _rank_cv_fit_matrix(cfg):
    """A unit's FIT-window matrix [n_fit_slots × n_OD] — the only data the rank
    CV touches (no disaster columns, no downstream quantity)."""
    graphs = load_graphs_trimmed(cfg['graph'], cfg['analysis_days'],
                                 SLOT_PER_DAY, label=cfg['label'])
    X_all, n_nor, _m = build_city_matrices(
        graphs, cfg['window'], cfg['buffer'] + cfg['disaster'],
        cfg['filter_factor'])
    n_dis = n_nor + cfg['buffer'] * SLOTS_ACTIVE
    fit = (None if set(cfg['fit_segments']) == {'normal', 'buffer', 'disaster'}
           else select_segment_columns(cfg['fit_segments'], n_nor, n_dis,
                                       X_all.shape[1]))
    Xt = X_all.T
    return Xt[fit] if fit is not None else Xt


def _rank_cv_k_grid(X):
    """Candidate ranks: every k up to 12, then coarser.  Starts at k = 1 so a
    minimum sitting at the left edge is visible as such rather than being an
    artefact of where the grid begins."""
    k_max = int(min(RANK_CV_K_MAX, *X.shape))
    ks = list(range(1, min(13, k_max + 1)))
    ks += [k for k in (15, 18, 21, 25, 30, 35, 40) if k <= k_max]
    return sorted(set(ks))


def analysis_rank_cv(cfgs):
    """STEP 1 — how many components does each unit's data support?

    Sweeps k for every registry unit under held-out-entry cross-validation and
    writes the per-unit curves plus the selection table.  Runs over ALL units,
    including those EXCLUDED_CODES drops from the later steps: the exclusion is
    a CONSEQUENCE of this analysis, so its own record has to stay complete.

    Reuses the cached curves when they already cover every unit (the curves do
    not depend on n_behaviors — see the RANK_CV_CURVES_CSV comment); otherwise
    computes the missing units and rewrites the cache.
    """
    codes = [c['code'] for c in cfgs]
    cached = (pd.read_csv(RANK_CV_CURVES_CSV)
              if os.path.exists(RANK_CV_CURVES_CSV) else
              pd.DataFrame(columns=['code', 'n_od', 'k', 'mean', 'sd']))
    todo = [c for c in cfgs if c['code'] not in set(cached.get('code', []))]
    if todo:
        print(f"  computing rank CV for {len(todo)} unit(s) "
              f"({len(codes) - len(todo)} cached)")
        rows = []
        for cfg in todo:
            X = _rank_cv_fit_matrix(cfg)
            print(f"    {cfg['code']}  X_fit {X.shape}", flush=True)
            for k in _rank_cv_k_grid(X):
                # Identical seeds at every k, so the curve compares ranks on
                # the same held-out cells rather than differently-lucky masks.
                errs = np.array([rank_cv_entry(X, k, cfg['l1_reg'],
                                               np.random.default_rng(RANK_CV_SEED + r),
                                               frac=RANK_CV_ENTRY_FRAC)
                                 for r in range(RANK_CV_REPEATS)], dtype=float)
                rows.append(dict(code=cfg['code'], n_od=X.shape[1], k=k,
                                 mean=np.nanmean(errs), sd=np.nanstd(errs)))
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        os.makedirs(OUTPUT_QUALITY_RAW, exist_ok=True)
        cached.to_csv(RANK_CV_CURVES_CSV, index=False)
    else:
        print(f"  reusing cached rank-CV curves for all {len(codes)} units")

    df = cached[cached.code.isin(codes)]
    recs = {}
    for cfg in cfgs:
        s = df[df.code == cfg['code']].sort_values('k')
        k = s['k'].to_numpy(); m = s['mean'].to_numpy(); sd = s['sd'].to_numpy()
        i = int(np.nanargmin(m))
        rec = dict(n_od=int(s['n_od'].iloc[0]), k_current=cfg['n_behaviors'],
                   k_min=int(k[i]))
        for name, mult, use_se in RANK_CV_BANDS:
            tol = mult * (sd[i] / np.sqrt(RANK_CV_REPEATS) if use_se else sd[i])
            ok = np.where(m <= m[i] + tol)[0]
            rec[name + '_lo'] = int(k[ok].min())
            rec[name + '_hi'] = int(k[ok].max())
        rec['excluded'] = cfg['code'] in EXCLUDED_CODES
        recs[cfg['code']] = rec

    # Ordered by matrix size: the size effect is the pattern to read off.
    table = pd.DataFrame(recs).T.sort_values('n_od', ascending=False)
    curves = {c: (df[df.code == c].sort_values('k')['k'].to_numpy(),
                  df[df.code == c].sort_values('k')['mean'].to_numpy(),
                  df[df.code == c].sort_values('k')['sd'].to_numpy())
              for c in table.index}
    os.makedirs(OUTPUT_QUALITY, exist_ok=True)
    vis_nmf_rank_cv(curves, table, band=RANK_CV_FIGURE_BAND,
                    save_path=os.path.join(OUTPUT_QUALITY, 'nmf_rank_cv.png'))
    table.to_csv(os.path.join(OUTPUT_QUALITY_RAW, 'nmf_rank_cv_selected.csv'))
    print(f"  [rank CV] {len(table)} units -> {OUTPUT_QUALITY}; "
          f"dropped downstream: {', '.join(sorted(EXCLUDED_CODES))}")
    return dict(table['k_min'])


def analysis_decomposition_quality(label, code, X_all, W, H, fit_time_cols):
    """Per-unit decomposition-quality check (see the OUTPUT_QUALITY
    comment for the three metric families; FIT-window only, the disaster
    period is out of scope by design).  Saves the unit's figure and returns
    the summary row for the cross-city dashboard."""
    q = nmf_quality_metrics(X_all.T, W, H, fit_time_cols,
                            min_comp_frac=NMF_MIN_COMP_FRAC)
    os.makedirs(OUTPUT_QUALITY, exist_ok=True)
    # Figure retired 2026-08-28 — the CHECK is the console line below plus the
    # cross-city CSV, and neither needed a per-unit three-panel PNG.
    print(f"  Quality (fit window): dist_err={q['dist_err_mean']:.4f}  "
          f"rel_err={q['rel_err']:.4f}  "
          f"comps below {NMF_MIN_COMP_FRAC:.0%}: {q['n_below']}/{W.shape[1]}")
    return dict(code=code, k=W.shape[1],
                dist_err_mean=q['dist_err_mean'],
                rel_err=q['rel_err'], n_below=q['n_below'])



# analysis_cluster_function_graph: within-city permutation draws, the BH gate
# the figure's line style encodes, and the seed shared by the permutation and
# the MDS initial condition (the MDS solution is otherwise rotation-arbitrary).
CLUSTER_GRAPH_N_PERM = 10000
CLUSTER_GRAPH_FDR_Q = 0.05
CLUSTER_GRAPH_SEED = 0
# Function colours for the overlay companion figure, where colour encodes the
# FUNCTION (the cluster is already the panel).
_FUNC_COLOR = {'residential': '#0F4D92', 'commercial': '#B64342',
               'leisure': '#4C9F70', 'industrial': '#E28E2C',
               'health': '#9A4D8E', 'public': '#42949E'}


def analysis_mapping_pca(feats_by_code):
    """The rank channel's mapping directions, no clustering (2026-08-31):
    every component of every retained unit in ONE PCA of the within-city
    rank-z feature space (RANK_FEATURE_COLS, the 23 columns the transfer
    model is fed), one dashed per-city ridge direction (features ->
    within-city cum_loss rank) and ONE solid pooled direction fit on all 81
    components stacked.

    This replaces the per-cluster panels: nothing trains on the transfer
    partition any more (RANK_TRAIN_SCOPE='pooled'), and the figure's question
    is now how far each city's mapping rule sits from the pooled rule the
    channel actually fits.  A city whose arrow opposes the pooled one is a
    city the pooled model cannot serve -- Wilmington (Dorian) is the standing
    example.

    City arrows are COLOURED by the pair heatmap's Louvain communities when
    its clusters CSV is present -- presentation only, the same display
    partition the heatmap boxes show, entering no computation; absent CSV ->
    one colour.

    Fits its own ridge because it needs the COEFFICIENT VECTOR (the mapping
    direction), not a prediction -- the one thing rank_predict does not
    return.  Same features, same within-city rank-z frame, same RidgeCV."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV

    fcols = list(RANK_FEATURE_COLS)
    alphas = np.logspace(-3, 3, 13)
    rows, blocks, targets, w_city = [], [], [], {}
    for code, feats in feats_by_code.items():
        sub = rank_merge_feats(feats)[fcols + ['cum_loss']].dropna()
        A = sub[fcols].to_numpy(float)
        R = np.column_stack([rankdata(A[:, j]) for j in range(A.shape[1])])
        R = (R - R.mean(0)) / (R.std(0) + 1e-12)
        y = sub['cum_loss'].to_numpy(float)
        yr = rankdata(y)
        yr = (yr - yr.mean()) / (yr.std() + 1e-12)
        w_city[code] = RidgeCV(alphas=alphas).fit(R, yr).coef_
        # Within-city rank -> colormap fraction; the [0.30, 0.92] floor keeps
        # the lightest point visible on white.
        shade = rankdata(y, method='average')
        shade = 0.30 + 0.62 * (shade - 1) / max(len(y) - 1, 1)
        blocks.append(R)
        targets.append(yr)
        for i in range(len(sub)):
            rows.append(dict(code=code, cum_loss=float(y[i]),
                             shade=float(shade[i])))
    pts = pd.DataFrame(rows)
    Ar = np.vstack(blocks)
    Yr = np.concatenate(targets)
    w_pool = RidgeCV(alphas=alphas).fit(Ar, Yr).coef_

    pca = PCA(n_components=2).fit(Ar)
    emb = pca.transform(Ar)
    pts['pc1'], pts['pc2'] = emb[:, 0], emb[:, 1]

    def _unit_and_plane(v):
        # Unit coefficient vector (23-D) + its unit in-plane shadow; near-zero
        # shadows (direction almost orthogonal to the plane) stay zero rather
        # than blowing up to an arbitrary angle.
        v = v / (np.linalg.norm(v) + 1e-12)
        d = pca.components_ @ v
        n = np.linalg.norm(d)
        return v, (d / n if n > 1e-9 else d * 0.0)

    arrows = []
    for kind, items in (('city', w_city.items()),
                        ('pooled', [('ALL', w_pool)])):
        for name, wv in items:
            v, d = _unit_and_plane(wv)
            arrows.append(dict(
                kind=kind, name=name, dx=float(d[0]), dy=float(d[1]),
                **{f'coef_{c}': float(v[i]) for i, c in enumerate(fcols)}))
    arr = pd.DataFrame(arrows)

    out_dir = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED, 'component_rank')
    raw_dir = os.path.join(out_dir, 'raw_data')
    os.makedirs(raw_dir, exist_ok=True)
    cl_csv = os.path.join(raw_dir, 'cross_city_pair_clusters.csv')
    cluster_of = None
    if os.path.exists(cl_csv):
        _cl = pd.read_csv(cl_csv, index_col=0)['cluster']
        if all(c in _cl.index for c in feats_by_code):
            cluster_of = {c: int(_cl[c]) for c in feats_by_code}
    arr['cluster'] = [cluster_of.get(r['name'], 0) if cluster_of and
                      r['kind'] == 'city' else 0
                      for _, r in arr.iterrows()]
    pts.to_csv(os.path.join(raw_dir, 'mapping_pca_components.csv'),
               index=False)
    arr.to_csv(os.path.join(raw_dir, 'mapping_pca_arrows.csv'), index=False)
    vis_mapping_pca(
        pts, arr, evr=tuple(pca.explained_variance_ratio_),
        save_path=os.path.join(out_dir, 'mapping_direction_pca.png'))
    print(f"  [mapping pca] {len(pts)} components, {len(w_city)} city arrows "
          f"+ 1 pooled -> {out_dir}")


def analysis_cluster_function_graph(feats_by_code):
    """Do the transfer clusters differ in WHICH FUNCTIONS RIDE THE SAME
    COMPONENTS?  One panel for the pooled average and one per ridge cluster.

    Representation ("co-riding", the component-grain colocation).  The 6
    merged func_<cat> shares are CLOSED — share_from_* and share_to_* each sum
    to 1, so func_<cat> sums to exactly 2 per component.  Correlating closed
    parts directly is not admissible: a constant-sum vector forces
    sum_c cov(x_i, x_c) = 0, i.e. an average off-diagonal correlation of
    about -1/(p-1) = -0.2 for p = 6 parts under NO structure at all.  A
    2026-08-12 check confirmed the damage was live: on Dirichlet data with no
    co-riding whatsoever the earlier version of this function returned mean
    rho = -0.135 and about 2.5 "significant" (always negative) edges, against
    the -0.144 and 4 all-negative edges it was reporting on the real data.
    The shares are therefore CENTRED-LOG-RATIO transformed first
    (clr(x)_i = log x_i - mean_j log x_j; the shares are strictly positive, so
    no zero handling is needed), and only then rank-z'd WITHIN each city and
    pooled.  CLR removes the raw constant-sum constraint but is itself
    sum-to-zero, so a residual negative pedestal of about -1/(p-1) = -0.2
    survives: read ABSOLUTE negative edges with that in mind (the panel means
    run -0.08 to -0.15, i.e. above the pedestal, so there is positive
    structure on top of it), while strong POSITIVE edges cannot be produced by
    the constraint at all, and CONTRASTS BETWEEN PANELS are immune to it
    because the pedestal is identical in every panel.  The health-public sign
    flip the figure is about — C1 -0.60 (q = .009) against C3 +0.51
    (q = .013) — is such a contrast.  Because each city's
    column is then mean 0 / sd 1 by construction, the POOLED PEARSON of those
    columns equals the component-count-weighted mean of the within-city
    correlations exactly — so the statistic is what the sentence "the average
    within-city relation, no between-city composition difference" claims.
    (Re-ranking the pooled column instead, as a pooled Spearman would, mixes
    a k=5 unit's coarse rank grid with a k=12 unit's fine one and drifts off
    that identity.)

    Geometry: d = 1 - rho embedded by metric MDS, so a high correlation is a
    short distance.  Every panel is fit FREELY (n_init=16) and then rigidly
    aligned onto the average layout by Procrustes (rotation/reflection only —
    scaling would distort the fitted distances); the shared frame comes from
    the alignment, not from a constrained initial condition, which used to
    cost C3 about 10% of its achievable stress.  Six points cannot honour 15
    distances in a plane, so Stress-1 (Kruskal's normalised stress) and the
    worst per-edge distance error are both recorded: at Stress-1 ~ 0.17 the
    layout can misstate a rho by ~0.2, which is why the CSV, not the picture,
    is the record.

    Significance: components of one city are NOT independent draws, so the
    textbook Spearman p does not apply.  The null shuffles one function's
    values WITHIN each city (city structure preserved), CLUSTER_GRAPH_N_PERM
    draws.  The BH family is the three CLUSTER panels (45 tests); the pooled
    panel is the union of those same components, so including it would
    double-count the data and inflate m to 60 — its p-values are reported
    uncorrected as a descriptive overview and its q column is set to the raw
    p.  Power differs across panels (81 vs 18-35 components), so a
    non-significant edge in a small cluster is weak evidence, not evidence of
    absence.

    Filed under component_rank/ with the partition it explains.  Skips
    (never raises) when the clusters CSV is absent OR does not cover every
    unit — it is a downstream overview figure and must not be able to abort
    the city-level reconstruction and STEP 7-8 that follow it."""
    cl_csv = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED, 'component_rank',
                          'raw_data', 'cross_city_pair_clusters.csv')
    if not os.path.exists(cl_csv):
        print(f"  [cluster graph] {cl_csv} absent; skipping.")
        return
    cl = pd.read_csv(cl_csv, index_col=0)['cluster']
    missing = [c for c in feats_by_code if c not in cl.index]
    if missing:
        print(f"  [cluster graph] clusters CSV does not cover "
              f"{', '.join(missing)} (stale file?); skipping.")
        return

    from scipy.linalg import orthogonal_procrustes
    from sklearn.manifold import MDS

    cats = list(SF_CATEGORIES)
    rows, city_col, clus_col = [], [], []
    for code, feats in feats_by_code.items():
        sub = _with_merged_func(feats)[FUNC_MERGED_COLS].dropna()
        A = sub.to_numpy(float)
        # CLR first (removes the constant-sum constraint), then within-city
        # rank-z.  A constant column would divide by ~0; it is left as NaN so
        # the pair is excluded rather than fabricated as a measured zero.
        L = np.log(A)
        A = L - L.mean(axis=1, keepdims=True)
        R = np.column_stack([rankdata(A[:, j]) for j in range(A.shape[1])])
        sd = R.std(0)
        Zc = np.where(sd > 1e-9, (R - R.mean(0)) / np.where(sd > 1e-9, sd, 1.0),
                      np.nan)
        rows.append(Zc)
        city_col += [code] * len(sub)
        clus_col += [int(cl[code])] * len(sub)
    Z = np.vstack(rows)
    city_of, clus_of = np.array(city_col), np.array(clus_col)
    # Panels come from the clusters actually PRESENT in the stacked
    # components; an id that survives only in the CSV would otherwise render
    # an empty panel whose degenerate p-values enter the BH family.
    groups = [('average', np.ones(len(Z), bool))] + [
        (f'C{k}', clus_of == k) for k in sorted(np.unique(clus_of))]

    def _corr(rowsel):
        # Pooled Pearson of the within-city rank-z columns == the
        # component-count-weighted mean of the within-city correlations.
        M = np.eye(len(cats))
        n = int(np.sum(rowsel))
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                a, b = Z[rowsel, i], Z[rowsel, j]
                ok = np.isfinite(a) & np.isfinite(b)
                M[i, j] = M[j, i] = (float(a[ok] @ b[ok] / n)
                                     if ok.sum() >= 3 and n else np.nan)
        return M

    rng = np.random.default_rng(CLUSTER_GRAPH_SEED)

    def _perm_p(rowsel):
        # One shared set of within-city shuffles serves every pair: a
        # permutation only re-orders a column, so each pair's marginal null
        # stays a valid within-city shuffle (it only couples their Monte-Carlo
        # error).  Selections too small to correlate return NaN, which keeps
        # them out of the BH family instead of emitting a floor p-value.
        idx = np.flatnonzero(rowsel)
        P = np.full((len(cats), len(cats)), np.nan)
        if len(idx) < 3:
            return P
        blk = pd.factorize(city_of[idx])[0].astype(float)
        order = np.argsort(blk[None, :] + rng.random(
            (CLUSTER_GRAPH_N_PERM, len(idx))), axis=1)
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                a, b = Z[idx, i], Z[idx, j]
                if not (np.isfinite(a).all() and np.isfinite(b).all()):
                    continue
                obs = float(a @ b / len(idx))
                null = (b[order] @ a) / len(idx)
                P[i, j] = P[j, i] = float(
                    (1 + (np.abs(null) >= abs(obs)).sum())
                    / (CLUSTER_GRAPH_N_PERM + 1))
        return P

    corr = {name: _corr(sel) for name, sel in groups}
    pval = {name: _perm_p(sel) for name, sel in groups}

    # BH over the CLUSTER panels only (the pooled panel is the union of the
    # same components: including it would double-count the data).  NaN
    # p-values (undefined pairs) stay out of the family.
    iu = np.triu_indices(len(cats), 1)
    cl_names = [name for name, _ in groups[1:]]
    flat = np.concatenate([pval[name][iu] for name in cl_names])
    fin = np.isfinite(flat)
    qflat = np.full_like(flat, np.nan)
    o = np.argsort(flat[fin])
    m_fam = int(fin.sum())
    tmp = np.empty(m_fam)
    tmp[o] = np.minimum.accumulate(
        (flat[fin][o] * m_fam / (np.arange(m_fam) + 1))[::-1])[::-1]
    qflat[fin] = np.minimum(tmp, 1.0)
    qval, n_pair = {}, len(iu[0])
    for t, name in enumerate(cl_names):
        Q = np.full((len(cats), len(cats)), np.nan)
        Q[iu] = qflat[t * n_pair:(t + 1) * n_pair]
        qval[name] = np.fmin(Q, Q.T)
    qval['average'] = pval['average'].copy()      # descriptive, uncorrected

    def _fit(Cm, align_to=None):
        Dm = 1.0 - Cm
        np.fill_diagonal(Dm, 0.0)
        Dm = np.nan_to_num(Dm, nan=1.0)           # undefined pair -> rho 0
        m = MDS(n_components=2, dissimilarity='precomputed',
                random_state=CLUSTER_GRAPH_SEED, n_init=16, max_iter=1000)
        X = m.fit_transform(Dm)
        # Stress-1 (Kruskal) is the interpretable one; the raw stress sklearn
        # reports has no scale.  The worst per-edge error says how much rho
        # the picture can misstate.
        fit_d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        s1 = float(np.sqrt(m.stress_ / (np.sum(Dm[iu] ** 2) + 1e-12)))
        err = float(np.abs(fit_d[iu] - Dm[iu]).max())
        if align_to is not None:
            mu_a, mu_x = align_to.mean(0), X.mean(0)
            Rt, _sc = orthogonal_procrustes(X - mu_x, align_to - mu_a)
            X = (X - mu_x) @ Rt + mu_a
        return X, s1, err

    # Every panel is fit FREELY and then aligned; the shared frame comes from
    # the Procrustes step, so no panel pays an initial-condition penalty.
    pos, stress, maxerr = {}, {}, {}
    pos['average'], stress['average'], maxerr['average'] = _fit(corr['average'])
    for name, _sel in groups[1:]:
        pos[name], stress[name], maxerr[name] = _fit(corr[name],
                                                     align_to=pos['average'])

    # Per-city layouts for the overlay figure: each unit's OWN correlation
    # (5-12 components, so noisy by construction — the figure shows the cloud,
    # not any one graph) solved from the panel's base layout and aligned onto
    # it, once per panel the unit appears in.
    codes = list(dict.fromkeys(city_of))
    city_corr = {c: _corr(city_of == c) for c in codes}
    city_pos = {}
    for c in codes:
        Xc, _s1, _e = _fit(city_corr[c])
        for panel in ('average', f'C{cl[c]}'):
            mu_b, mu_x = pos[panel].mean(0), Xc.mean(0)
            Rt, _sc = orthogonal_procrustes(Xc - mu_x, pos[panel] - mu_b)
            city_pos[(panel, c)] = (Xc - mu_x) @ Rt + mu_b

    labels = [c.title() for c in cats]
    panels, over_panels, recs = [], [], []
    for name, sel in groups:
        n_comp = int(sel.sum())
        n_city = len(set(city_of[sel]))
        head = ('Average' if name == 'average'
                else f'Cluster {name[1:]}')
        title = f'{head}\n{n_city} Cities · {n_comp} Components'
        panels.append(dict(
            title=title,
            labels=labels, pos=pos[name], corr=corr[name], qval=qval[name]))
        members = [c for c in codes
                   if name == 'average' or f'C{cl[c]}' == name]
        over_panels.append(dict(
            title=title, cats=cats, labels=labels, pos=pos[name],
            members=[(c, city_pos[(name, c)]) for c in members]))
        for i, j in zip(*iu):
            q = qval[name][i, j]
            recs.append(dict(panel=name, n_components=n_comp, n_cities=n_city,
                             func_a=cats[i], func_b=cats[j],
                             rho=corr[name][i, j], p_perm=pval[name][i, j],
                             q_bh=q,
                             # 'average' carries a raw p in this column (it is
                             # outside the BH family), so its flag is
                             # descriptive only.
                             significant=(bool(q < CLUSTER_GRAPH_FDR_Q)
                                          if np.isfinite(q) else False),
                             stress_1=stress[name],
                             max_edge_error=maxerr[name]))

    out_dir = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED, 'component_rank')
    raw_dir = os.path.join(out_dir, 'raw_data')
    os.makedirs(raw_dir, exist_ok=True)
    pd.DataFrame(recs).to_csv(
        os.path.join(raw_dir, 'cluster_function_graph.csv'), index=False)
    pd.DataFrame({f'{name}_{ax}': pos[name][:, a]
                  for name, _ in groups for a, ax in ((0, 'x'), (1, 'y'))},
                 index=cats).to_csv(
        os.path.join(raw_dir, 'cluster_function_graph_layout.csv'))
    # One figure, two rows: significance on top, city-event spread below, a
    # column per group.  They were separate PNGs until 2026-08-21; stacked,
    # the group title and half the legend are written once instead of twice.
    vis_cluster_function_graph(
        panels, over_panels, _FUNC_COLOR, fdr_q=CLUSTER_GRAPH_FDR_Q,
        save_path=os.path.join(out_dir, 'cluster_function_graph.png'))
    # Heatmap companion: the pooled average and each cluster, all in
    # ABSOLUTE (CLR) Spearman rho — the same matrices the graph figure lays
    # out, read cell by cell instead of as a geometry.  Each panel is ordered
    # by its OWN Louvain communities (the pair-heatmap recipe), because a
    # cluster that groups its functions differently should show that in its
    # row order.
    heat_panels, hrecs = [], []
    for t, (name, sel) in enumerate(groups):
        M = corr[name]
        ordered, _lab, blocks = _transfer_communities(
            pd.DataFrame(M, index=cats, columns=cats))
        perm = [cats.index(c) for c in ordered]
        shown = M[np.ix_(perm, perm)]
        heat_panels.append(dict(
            title=panels[t]['title'], labels=[c.title() for c in ordered],
            mat=shown, blocks=blocks))
        for i, j in zip(*iu):
            # Own CSV: `recs` above carries the tested rho in a fixed cat
            # order; this one carries the same rho in each panel's Louvain
            # display order, so the figure can be reproduced from it.
            hrecs.append(dict(panel=name, n_components=int(sel.sum()),
                              func_a=ordered[i], func_b=ordered[j],
                              rho=float(shown[i, j])))
    pd.DataFrame(hrecs).to_csv(
        os.path.join(raw_dir, 'cluster_function_heatmap.csv'), index=False)
    # The BOXES, which the rho table above cannot carry: each panel's Louvain
    # community per function, in that panel's display order.  Without it a
    # restyle of the heat map needed a whole pipeline run just to recover the
    # partition it draws.
    pd.DataFrame(
        [{'panel': name, 'position': pos, 'func': ordered[pos],
          'community': int(cid)}
         for (name, _sel), hp in zip(groups, heat_panels)
         for start, size, cid in hp['blocks']
         for pos in range(start, start + size)
         for ordered in [[lab.lower() for lab in hp['labels']]]]
    ).to_csv(os.path.join(raw_dir, 'cluster_function_heatmap_blocks.csv'),
             index=False)
    vis_cluster_function_heatmap(
        heat_panels,
        save_path=os.path.join(out_dir, 'cluster_function_heatmap.png'))
    # How far each unit's own structure sits from the layouts it belongs to —
    # the number behind the overlay's visual spread.
    pd.DataFrame([
        dict(code=c, cluster=int(cl[c]), k=int((city_of == c).sum()),
             dev_from_average=float(np.linalg.norm(
                 city_pos[('average', c)] - pos['average'], axis=1).mean()),
             dev_from_own_cluster=float(np.linalg.norm(
                 city_pos[(f'C{cl[c]}', c)] - pos[f'C{cl[c]}'],
                 axis=1).mean()))
        for c in codes]).to_csv(
        os.path.join(raw_dir, 'cluster_function_city_deviation.csv'),
        index=False)
    n_sig = {name: int(np.nansum(qval[name][iu] < CLUSTER_GRAPH_FDR_Q))
             for name in cl_names}
    print(f"  [cluster graph] {len(Z)} components (CLR-transformed); "
          f"significant cluster edges {n_sig} (BH over {m_fam} tests; the "
          f"pooled panel is descriptive); Stress-1 "
          f"{({k: round(v, 2) for k, v in stress.items()})} -> {out_dir}")


def _clr_rankz(func_matrix):
    """Per-city CLR then WITHIN-city rank-z of one city's [n x 6] strictly
    positive function shares.  CLR removes the closure (the 6 shares sum to a
    constant); the rank-z matches the rank channel's representation.  Stacking
    several cities' outputs and correlating the columns is the pooled co-riding
    template."""
    L = np.log(func_matrix)
    A = L - L.mean(axis=1, keepdims=True)
    R = np.column_stack([rankdata(A[:, j]) for j in range(A.shape[1])])
    return (R - R.mean(0)) / (R.std(0) + 1e-12)


def _coriding_15(rankz):
    """The 15 upper-triangle correlations of a [n x 6] rank-z matrix."""
    iu = np.triu_indices(rankz.shape[1], 1)
    n = len(rankz)
    return np.array([float(rankz[:, i] @ rankz[:, j] / n)
                     for i, j in zip(*iu)])


def _estimate_held_cluster(held_func, ref_func_by_cluster):
    """Nearest cluster for a held-out city by the SHAPE of its co-riding
    matrix.  `held_func` is the city's [n x 6] func-share matrix;
    `ref_func_by_cluster` maps each cluster id to the LIST of the reference
    cities' func-share matrices in it (the held city's own components are never
    in there).  Each city is CLR+rank-z'd on its own, the cluster's cities are
    then stacked into one template, and the distance is 1 - Spearman between
    the two 15-cell vectors — the whole-matrix rule that assigned 10/13 in the
    2026-08-12 sandbox, chosen over any single cell because it needs no cell
    selection."""
    hv = _coriding_15(_clr_rankz(held_func))
    d = {k: 1.0 - spearmanr(
             hv, _coriding_15(np.vstack([_clr_rankz(F) for F in mats]))).statistic
         for k, mats in ref_func_by_cluster.items()}
    return min(d, key=d.get)


# -- THE rank channel: the one predictor every rank figure goes through -------

def rank_merge_feats(feats):
    """Add every derived column RANK_FEATURE_COLS names: the merged
    func_<cat> shares and their 15 pairwise products.

    The decomposition tables carry direction-split shares (share_from_<cat>
    and share_to_<cat>); the rank channel reads a component's involvement
    with a function regardless of direction, so the two are summed.  The
    products are taken on those RAW merged shares -- BEFORE any ranking,
    because the rank of a product is not the product of ranks -- with the
    same definition _with_city_total_feats uses, so a frame that already
    went through that helper survives this one unchanged."""
    m = feats.copy()
    for c in SF_CATEGORIES:
        m[f'func_{c}'] = feats[f'share_from_{c}'] + feats[f'share_to_{c}']
    for i, a in enumerate(SF_CATEGORIES):
        for b in SF_CATEGORIES[i + 1:]:
            m[f'func_{a}_X_{b}'] = m[f'func_{a}'] * m[f'func_{b}']
    return m


def rank_cluster_context(merged, codes):
    """The transfer-cluster context RANK_TRAIN_SCOPE='cluster' needs, or None.

    Clusters come from the pairwise transfer heatmap written earlier in the
    same run, which is why that step must run BEFORE any cluster-restricted
    rank prediction.  Returns (cluster Series, {code -> func-share matrix});
    the matrices are what re-estimates a held-out city's membership per fold,
    so no label of the held unit ever enters its own training pool.  None when
    the scope is pooled, or the clusters CSV is missing or does not cover
    `codes` -- the caller then trains on every reference city."""
    if RANK_TRAIN_SCOPE != 'cluster':
        return None
    csv = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED, 'component_rank',
                       'raw_data', 'cross_city_pair_clusters.csv')
    if not os.path.exists(csv):
        return None
    cl = pd.read_csv(csv, index_col=0)['cluster']
    if not all(c in cl.index for c in codes):
        return None
    fc = [f'func_{c}' for c in SF_CATEGORIES]
    return cl, {c: merged[c][fc].dropna().to_numpy(float) for c in codes}


def rank_train_pool(held, rest, ctx):
    """`rest` narrowed to held's ESTIMATED cluster (>=1 city), else all of
    `rest`.  Falls back to the full pool when there is no context, when the
    reference cities occupy a single cluster (nothing to choose between), or
    when the estimate would leave no peers."""
    if ctx is None:
        return rest
    cl, func_of = ctx
    ref = {}
    for c in rest:
        ref.setdefault(int(cl[c]), []).append(func_of[c])
    if len(ref) < 2:
        return rest
    est = _estimate_held_cluster(func_of[held], ref)
    kept = [c for c in rest if int(cl[c]) == est]
    return kept if kept else rest


def rank_predict(held, train_codes, merged, target='cum_loss',
                 min_rows=CROSS_CITY_MIN_ROWS, feature_cols=None):
    """THE rank prediction: held's components scored so that their ORDER is the
    predicted ordering of `target`.  Returns a Series indexed by component, or
    None when the engine could not fit the fold.

    Only the ORDER of the returned scores means anything -- each score comes
    from a ridge fit in a frame where every feature and the target were rank-
    transformed and then z-scored WITHIN each city-event, so a score sits on no
    city's loss scale.  That within-unit frame is exactly why this channel
    transfers: it is immune to the cross-city level and scale drift a pooled
    frame has to assume away.

    The held unit's target column is overwritten with a placeholder ramp so its
    true losses cannot reach the fit even by accident.  (The engine's transfer
    path never uses the test target for prediction anyway; the ramp makes that
    structural rather than a property one has to re-verify.)  Score the result
    against the OBSERVED target read separately from the table.

    train_codes == [held] is the one exception: that is the unit's own
    leave-one-component-out, there is no other city to borrow from, and the
    real target has to stay in place for the fold to mean anything.

    `feature_cols` defaults to RANK_FEATURE_COLS; no production caller
    overrides it since the partition-specific list retired (2026-08-31), the
    parameter stays for ablations."""
    self_fit = list(train_codes) == [held]
    te = merged[held].copy()
    fold = {held: te}
    if not self_fit:
        te[target] = np.arange(len(te), dtype=float)
        fold.update({c: merged[c] for c in train_codes})
    _r2, pred, _g = cross_city_resilience(
        fold, [target],
        list(RANK_FEATURE_COLS if feature_cols is None else feature_cols),
        rank=True,
        split={'train': list(train_codes), 'test': [held]},
        target_std='within_unit', model=RANK_MODEL, min_rows=min_rows)
    pm = pred.get('pooled_LOO' if self_fit else held, {}).get(target)
    if pm is None:
        return None
    _y, ypred, cidx = pm
    return pd.Series(np.asarray(ypred, dtype=float), index=cidx)


def analysis_rank_channel(merged, codes, target='cum_loss'):
    """The rank channel's leave-one-city-out sweep and its figure.

    One fold per city-event: every other unit is the reference pool, narrowed
    to the held unit's estimated cluster when RANK_TRAIN_SCOPE says so, and
    rank_predict scores the held unit's components.  The fold's skill is the
    Spearman between that predicted ordering and the OBSERVED one, which is the
    only thing an ordering can be scored on -- an R2 would additionally score a
    prediction amplitude these scores do not have.

    Writes component_rank/rank_pred_vs_true.png plus the per-fold points and
    the per-fold rho table under raw_data/, and returns the rho Series.

    Requires the pairwise heatmap to have run FIRST when the scope is
    'cluster': that step writes the partition this one reads."""
    out_dir = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED, 'component_rank')
    raw_dir = os.path.join(out_dir, 'raw_data')
    os.makedirs(raw_dir, exist_ok=True)
    ctx = rank_cluster_context(merged, codes)
    if RANK_TRAIN_SCOPE == 'cluster' and ctx is None:
        print("  [rank] RANK_TRAIN_SCOPE=cluster but the clusters CSV is "
              "absent/incomplete; falling back to pooled rank training.")

    rows, rho = [], {}
    for held in codes:
        rest = [c for c in codes if c != held]
        pool = rank_train_pool(held, rest, ctx)
        score = rank_predict(held, pool, merged, target=target)
        if score is None:
            print(f"  [rank] {held}: the engine returned no fold; skipped.")
            continue
        obs = merged[held].loc[score.index, target].astype(float)
        keep = obs.notna() & score.notna()
        if int(keep.sum()) < 3:
            print(f"  [rank] {held}: fewer than 3 scorable components; skipped.")
            continue
        r = float(spearmanr(obs[keep], score[keep]).statistic)
        rho[held] = r
        for comp in score.index[keep]:
            rows.append({'code': held, 'component': comp,
                         target: float(obs[comp]),
                         'rank_score': float(score[comp]),
                         'spearman': r, 'n_train': len(pool)})
    if not rows:
        print("  [rank] no usable folds; skipping the rank channel outputs.")
        return pd.Series(dtype=float)

    par = pd.DataFrame(rows)
    par.to_csv(os.path.join(raw_dir, 'rank_pred_vs_true.csv'), index=False)
    rser = pd.Series(rho, name='spearman').rename_axis('code').reindex(
        [c for c in codes if c in rho])
    rser.to_csv(os.path.join(raw_dir, 'rank_loo_spearman.csv'))
    labels = {c['code']: c['label'] for c in CITY_EVENTS}
    names = {c: f"{labels.get(c, c)} ({c.split('_', 1)[-1]})"
             for c in rser.index}
    vis_rank_pred_vs_true(
        par, names=names, obs_col=target,
        save_path=os.path.join(out_dir, 'rank_pred_vs_true.png'),
        footnote=(f'mean test Spearman ρ over the {len(rser)} held-out '
                  f'city-events = {rser.mean():+.2f}'))
    print(f"  [rank] {len(rser)} folds, mean Spearman ρ "
          f"{rser.mean():+.3f} -> {out_dir}")
    return rser


def city_total_score(held, rest, merged, feats_test, model,
                     feature_cols=None, pooled_cols=None,
                     min_rows=CROSS_CITY_MIN_ROWS):
    """THE phi stage of the city-total estimator: one unit's aggregate score s.

    Every component's cum_loss is predicted on the POOLED-TRAIN scale and the
    predictions are weight_normal-averaged WITHOUT ever being un-standardized.
    Keeping them standardized is the whole point: un-standardizing multiplies
    by the large component-to-component spread and mis-calibrates the
    aggregate, which is what the raw channel does.  The result feeds
    _city_total_from_scores (the rho stage) -- together those two are the city
    total, and both STEP 6's headline figure and STEP 7's quantile-mapping
    anchor now read this same pair.

    The held unit's cum_loss is overwritten with a placeholder ramp so its own
    losses cannot reach the fit even by accident.  `feature_cols` defaults to
    CITY_TOTAL_FEATURE_COLS; the one caller that overrides it is STEP 6's
    cosine-kNN arm, deliberately kept on the smaller pre-interaction set (a
    cosine over 24 mostly-collinear products dilutes the neighbourhood it
    depends on).  NaN when the engine cannot form the fold."""
    cols = list(CITY_TOTAL_FEATURE_COLS if feature_cols is None
                else feature_cols)
    pcols = list(CITY_TOTAL_POOLED_COLS if pooled_cols is None
                 else pooled_cols)
    te = merged[held].copy()
    te['cum_loss'] = np.arange(len(te), dtype=float)
    fold = {held: te}
    fold.update({c: merged[c] for c in rest})
    _r2, pred, _g = cross_city_resilience(
        fold, ['cum_loss'], cols, rank=False,
        split={'train': list(rest), 'test': [held]}, target_std='pooled_train',
        level_feature_cols=LEVEL_FEATURE_COLS, model=model,
        pooled_feature_cols=pcols, min_rows=min_rows)
    pm = pred.get(held, {}).get('cum_loss')
    if pm is None:
        return np.nan
    _y, ypred, cidx = pm
    if len(cidx) == 0:
        return np.nan
    w = feats_test[held].loc[cidx, 'weight_normal'].to_numpy(dtype=float)
    yp = np.asarray(ypred, dtype=float)
    wsum = float(w.sum())
    return float((w * yp).sum() / wsum) if wsum > 0 else float(yp.mean())


def analysis_city_mobility_curves(units):
    """Every retained unit's CITY-LEVEL mobility curve on one page + raw CSV.

    Filed with the decomposition diagnostics because it is the reference the
    decomposition is judged against: it shows, with NO decomposition and no
    smoothing, what each unit's disaster actually looked like — the pre-landfall
    preparation surge, the collapse, and the recovery shape.

    Daily total flow is divided by the day-type-matched normal-period mean
    (weekday days by the weekday mean, weekend days by the weekend mean), so
    the level is comparable across units and the weekly rhythm is removed.  The
    reported minimum is searched over the buffer + disaster stretch only; the
    normal half cannot win it.

    Deliberately NOT coloured by transfer cluster: those come from STEP 6 and
    this runs at STEP 3b, so the colours would encode something the reader has
    not been shown yet (and would make a diagnostic depend on a downstream
    result)."""
    _WD = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
           'Sunday']
    curves, rows = {}, []
    for code, u in units.items():
        cfg = u['cfg']
        total = u['X_all'].sum(axis=0)
        n_days = len(total) // SLOTS_ACTIVE
        daily = total[:n_days * SLOTS_ACTIVE].reshape(n_days,
                                                      SLOTS_ACTIVE).sum(axis=1)
        first_wd = _WD.index(u['first_day_nor'])
        is_we = np.array([(first_wd + i) % 7 >= 5 for i in range(n_days)])
        d_nor = u['n_nor'] // SLOTS_ACTIVE
        # Baselines from the NORMAL half only, one per day type.
        base_wd = daily[:d_nor][~is_we[:d_nor]].mean()
        base_we = daily[:d_nor][is_we[:d_nor]].mean()
        rel = daily / np.where(is_we, base_we, base_wd)
        day0 = d_nor + cfg['buffer']              # landfall day index
        i_min = d_nor + int(np.argmin(rel[d_nor:]))
        curves[code] = dict(rel=rel, day0=day0, d_nor=d_nor, i_min=i_min)
        rows.append(dict(code=code, k=cfg['n_behaviors'],
                         min_day_rel_landfall=int(i_min - day0),
                         min_value=float(rel[i_min]),
                         landfall_value=float(rel[day0]),
                         pre_peak=float(rel[d_nor:day0].max())
                         if day0 > d_nor else np.nan,
                         end_value=float(rel[-1])))
    raw = os.path.join(OUTPUT_DATA, 'raw_data')
    os.makedirs(raw, exist_ok=True)
    df = pd.DataFrame(rows).set_index('code')
    df.to_csv(os.path.join(raw, 'city_mobility_curves.csv'))
    # Full "City (Storm)" titles; the two Wilmington events stay distinct.
    names = {code: f"{u['label']} ({code.split('_', 1)[1]})"
             for code, u in units.items()}
    vis_city_mobility_curves(
        curves, names=names,
        save_path=os.path.join(OUTPUT_DATA, 'city_mobility_curves.png'))
    late = df[df['min_day_rel_landfall'] != 0]
    note = (f"{len(late)} unit(s) bottom out AFTER landfall day: "
            + ", ".join(f"{c} (+{int(d)})"
                        for c, d in late['min_day_rel_landfall'].items())
            if len(late) else "every unit bottoms out on landfall day")
    print(f"  [city curves] {note} -> {OUTPUT_DATA}")


def analysis_decomposition_quality_summary(rows):
    """Cross-city quality dashboard + raw CSV, after every unit is decomposed."""
    df = pd.DataFrame(rows).set_index('code')
    os.makedirs(OUTPUT_QUALITY_RAW, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_QUALITY_RAW, 'nmf_quality_metrics.csv'))
    bad = df[df['n_below'] > 0]
    flag = (f"{len(bad)} unit(s) VIOLATE the component-count rule: "
            + ", ".join(f"{c} ({int(n)})" for c, n in bad['n_below'].items())
            if len(bad) else "all units satisfy the component-count rule")
    print(f"\n  [decomposition quality] {flag} -> {OUTPUT_QUALITY}")


def analysis_component_signature(W, n_nor, n_dis, first_day_normal,
                                 first_day_disaster, tag):
    """Component temporal and spatial characteristics.  Plots the full-window W
    heatmap and the per-component timeline (normal blue, buffer amber, disaster
    red, black dashed line at landfall).  n_nor marks the end of the clean
    normal columns and n_dis the disaster start, so [n_nor, n_dis) is the
    buffer."""
    os.makedirs(OUTPUT_TEMPORAL_HM, exist_ok=True)
    os.makedirs(OUTPUT_TEMPORAL_CV, exist_ok=True)
    vis_heatmap_temporal_signature(
        W, first_day=first_day_normal, show_days=True,
        slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
        output_dir=OUTPUT_TEMPORAL_HM, tag=tag,
    )
    vis_line_nmf_component_timeline(
        W[:n_nor], W[n_dis:], W_buffer=W[n_nor:n_dis],
        first_day_normal=first_day_normal, first_day_disaster=first_day_disaster,
        slots_per_day=SLOTS_ACTIVE,
        output_dir=OUTPUT_TEMPORAL_CV, tag=tag,
    )


def analysis_spatial(label, H, mapping, gdf, weights, tag, lambda_ctx=None):
    """Per-component SPATIAL characteristics block (structurally parallel to the
    O×D functional block).  Builds the OD centroid distance array
    (graph_io.build_distance_array — same definition the distance-decay analysis
    uses), computes each component's loading-weighted flow distance (km) from H,
    saves a per-component bar figure under component_characteristics/spatial/ and
    the raw values as CSV.  lambda_ctx tags the filename (H, hence the distances,
    shift with context strength).

    Returns the OD centroid `distances` array — the reusable intermediate (the
    spatial analogue of the func cross-tab M returned by analysis_od_function), so
    the caller rebuilds the feature inline with spatial_features(H, distances)."""
    os.makedirs(OUTPUT_SPATIAL, exist_ok=True)
    lambda_tag = _lambda_tag(lambda_ctx)
    distances = build_distance_array(mapping, gdf)
    spatial_feats = spatial_features(H, distances)
    vis_bar_component_distance(
        spatial_feats, weights=weights,
        title=f'{label}: per-component flow distance',
        save_path=os.path.join(OUTPUT_SPATIAL,
                               f'component_distance_{lambda_tag}{tag}.png'),
    )
    os.makedirs(OUTPUT_SPATIAL_DIST_RAW, exist_ok=True)
    spatial_feats.to_csv(os.path.join(OUTPUT_SPATIAL_DIST_RAW,
                                      f'spatial_features_{lambda_tag}{tag}.csv'))
    return distances


def analysis_socioeconomic(label, key, tag, gdf, H, mapping, weights,
                           lambda_ctx=None, modes=INCOME_ENDPOINT_MODES,
                           year=ACS_INCOME_YEAR):
    """Per-component SOCIOECONOMIC characteristics block.  Ensures the raw ACS
    median household income cache (table B19013, `year` 5-year) and maps each block
    group's income to the flows; then for EACH endpoint mode in `modes` ('combined'
    = origin+destination nan-aware mean, 'origin' = origin only) computes each
    component's loading-weighted income from H and saves a per-mode bar figure
    under component_characteristics/socioeconomic/.  RETURNS a DataFrame with one
    median_income_<mode> column per mode (a raw CSV holds all of them) so the caller
    adds them to the feature table, where they appear as right-hand columns of the
    resilience-correlation heatmap; income is NOT a regression predictor.  RAISES if
    the ACS data cannot be obtained (e.g. a missing CENSUS_API_KEY) — loud, not a
    silent skip — so the run stops rather than producing the figure without income."""
    os.makedirs(ACS_DATA_DIR, exist_ok=True)
    raw_csv = os.path.join(ACS_DATA_DIR,
                           f'{key}_block_group_acs_income_{year}_raw.csv')
    ensure_city_income_raw(label, gdf['aggr_id'].tolist(), raw_csv, year=year)
    income = load_city_income(raw_csv)
    income_by_aggr = dict(zip(income['aggr_id'], income['median_household_income']))

    lambda_tag = _lambda_tag(lambda_ctx)
    os.makedirs(OUTPUT_SOCIO, exist_ok=True)
    per_mode = []
    for mode in modes:
        income_array = build_income_array(mapping, income_by_aggr, mode=mode)
        sf = socioeconomic_features(H, income_array, name=f'median_income_{mode}')
        per_mode.append(sf)
        vis_bar_component_income(
            sf.rename(columns={f'median_income_{mode}': 'median_income'}), weights=weights,
            title=f'{label}: per-component median household income ({mode})',
            save_path=os.path.join(OUTPUT_SOCIO,
                                   f'component_income_{mode}_{lambda_tag}{tag}.png'),
        )
    socio = pd.concat(per_mode, axis=1)             # one median_income_<mode> column per mode
    os.makedirs(OUTPUT_SOCIO_RAW, exist_ok=True)
    socio.to_csv(os.path.join(OUTPUT_SOCIO_RAW,
                              f'socioeconomic_features_{lambda_tag}{tag}.csv'))
    # The median_income_<mode> columns join `feats` and appear as right-hand
    # columns of the resilience-correlation heatmap.
    return socio


def analysis_od_function(label, tag, H, mapping, weights, landuse,
                         lambda_ctx=None):
    """O×D functionality block.  Takes the city's GLOBAL-recipe land-use
    classification (`landuse`, from the STEP-1 global classification pass),
    aggregates each component's OD flows into an origin×destination functional
    cross-tab, and saves the heatmap grid plus a per-component functional-share
    CSV.  Also saves the across-component entropy distribution of each
    component's outflow/inflow functional mix (the heatmap M marginals);
    lambda_ctx tags the filename so context-aware strengths can be compared.
    Returns M [k × C × C]."""
    os.makedirs(OUTPUT_FUNC, exist_ok=True)
    print(f"  {label}: dominant_category counts: "
          f"{landuse['dominant_category'].value_counts().to_dict()}")

    share_lookup = share_lookup_from_landuse(landuse, SF_CATEGORIES)
    M, retained = build_od_function_matrix_soft(H, mapping, share_lookup,
                                                SF_CATEGORIES)
    print(f"  {label}: O×D flow with land-use data per component: "
          + ", ".join(f"[{i}]={r:.2f}" for i, r in enumerate(retained)))
    vis_heatmap_od_function(
        M, SF_CATEGORIES,
        save_path=os.path.join(OUTPUT_FUNC, f'heatmap_od_functionality{tag}.png'),
    )
    # Per-component functional dimensions — the 12 from/to shares, one row per
    # component (index), for inspecting the raw values behind the heatmap.
    os.makedirs(OUTPUT_FUNC_HM_RAW, exist_ok=True)
    functional_features(M, SF_CATEGORIES).to_csv(
        os.path.join(OUTPUT_FUNC_HM_RAW, f'component_functionality{tag}.csv'))
    # The heat map's OWN input, long-form: the file above holds only M's
    # marginals, so the 36 cells per component had to be recomputed on every
    # run.  With this the figure can be redrawn straight from disk.
    pd.DataFrame(
        [{'component': int(i), 'origin': SF_CATEGORIES[a],
          'destination': SF_CATEGORIES[b], 'proportion': float(M[i, a, b])}
         for i in range(M.shape[0])
         for a in range(len(SF_CATEGORIES))
         for b in range(len(SF_CATEGORIES))]
    ).to_csv(os.path.join(OUTPUT_FUNC_HM_RAW,
                          f'od_function_matrix{tag}.csv'), index=False)
    # `retained` is printed above and was otherwise lost: the share of each
    # component's flow whose CBGs carried land-use data, i.e. how much of the
    # component the heat map actually represents.
    pd.DataFrame({'component': range(len(retained)),
                  'landuse_coverage': np.asarray(retained, dtype=float)}
                 ).to_csv(os.path.join(OUTPUT_FUNC_HM_RAW,
                                       f'landuse_coverage{tag}.csv'),
                          index=False)

    # Functional-entropy distribution across components.  `entropy` is the
    # Shannon entropy of the MERGED exposure (outflow row sums + inflow column
    # sums) of the SAME heatmap M (categories include 'Mix'); lower = more
    # functionally concentrated.  The x axis is fixed to [0, ln K] so every
    # city-event's histogram is on one scale.  λ is in the filename for
    # cross-strength comparison.
    lambda_tag = _lambda_tag(lambda_ctx)
    ent = component_function_entropy(M)
    os.makedirs(OUTPUT_FUNC_ENT_RAW, exist_ok=True)
    ent.to_csv(os.path.join(OUTPUT_FUNC_ENT_RAW,
                            f'component_function_entropy_{lambda_tag}{tag}.csv'))
    vis_hist_function_entropy(
        ent, lambda_ctx=lambda_ctx,
        max_entropy=float(np.log(len(SF_CATEGORIES))),
        title=f'{label}: functional entropy across components',
        save_path=os.path.join(OUTPUT_FUNC,
                               f'hist_function_entropy_{lambda_tag}{tag}.png'),
    )
    return M


def analysis_time_function_corr(feats, tag):
    """Time × function correlation block.  One heatmap: Spearman across the
    city's components between every CONTINUOUS temporal feature and the MERGED
    functional shares (FUNC_MERGED_COLS).  Rows are weekday_ratio, the four
    day-period band intensities (graded, replacing the old argmax-grouped bar
    charts), and the per-slot shares; no 'weekend' row — weekday_ratio already
    carries that axis.  At the 2 h slot width the night band equals the
    20-22h slot, so those two rows are identical by construction."""
    os.makedirs(OUTPUT_FVT_PER, exist_ok=True)
    slot_rows = [c for c in feats.columns if re.fullmatch(r'\d+-\d+h', c)]
    time_rows = TIME_COLS + [name for name, _, _ in PERIOD_BANDS] + slot_rows
    rho, pval = time_function_correlation(_with_merged_func(feats), time_rows,
                                          FUNC_MERGED_COLS)
    # Blank bands separate the three row families: the weekday/weekend ratio,
    # the day-period band intensities, and the per-slot shares.
    vis_heatmap_corr(
        rho, pval, time_cols=[], categories=SF_CATEGORIES,
        row_gaps=(len(TIME_COLS), len(TIME_COLS) + len(PERIOD_BANDS)),
        save_path=os.path.join(OUTPUT_FVT_PER,
                               f'heatmap_time_function_corr{tag}.png'),
    )


def analysis_time_function_corr_pooled(feats_by_city):
    """The time × function heatmap over EVERY city-event's components at once.

    Same rows, columns and layout as the per-city figure; only the sample
    changes.  Pooling raw values would let the cities with the largest feature
    or share magnitudes dominate every cell, so each column is first
    rank-transformed WITHIN its own city-event and the rank is normalised to
    (0, 1) by that unit's component count — the units carry k = 9, 10 or 11
    components, and without the normalisation a rank of 9 would mean
    'the largest' in one unit and 'second largest' in another.  After that
    transform every unit contributes the same uniform marginal on every column,
    so a cell reflects only the WITHIN-city co-ordering of the two features,
    pooled across units; the cross-city level and scale differences of both the
    features and the functional shares are gone by construction.  The reported
    coefficient is Spearman over the pooled normalised ranks.

    This is the same level-robust rank convention the cross-city transfer uses
    ('within_unit' standardization, see cross_city_resilience)."""
    codes = list(feats_by_city)
    if len(codes) < 2:
        print("  [pooled time×function] fewer than 2 units; skipping.")
        return None
    slot_rows = [c for c in feats_by_city[codes[0]].columns
                 if re.fullmatch(r'\d+-\d+h', c)]
    time_rows = TIME_COLS + [name for name, _, _ in PERIOD_BANDS] + slot_rows
    cols = time_rows + FUNC_MERGED_COLS

    ranked = []
    for c in codes:
        t = _with_merged_func(feats_by_city[c])[cols].astype(float)
        n = len(t)
        # Normalised within-unit rank in (0, 1); NaN stays NaN so a unit missing
        # a feature drops only that cell's rows, exactly as the per-city figure.
        ranked.append(t.rank(axis=0, na_option='keep')
                       .sub(0.5).div(n).assign(code=c))
    pooled = pd.concat(ranked, ignore_index=True)

    rho, pval = time_function_correlation(pooled, time_rows, FUNC_MERGED_COLS,
                                          method='spearman')
    os.makedirs(OUTPUT_FVT_ALL, exist_ok=True)
    os.makedirs(OUTPUT_FUNC_VS_TEMP_RAW, exist_ok=True)
    vis_heatmap_corr(
        rho, pval, time_cols=[], categories=SF_CATEGORIES,
        row_gaps=(len(TIME_COLS), len(TIME_COLS) + len(PERIOD_BANDS)),
        save_path=os.path.join(OUTPUT_FVT_ALL,
                               'heatmap_time_function_corr_ALL.png'),
    )
    # The scatter behind every row of that heatmap: raw values, one point per
    # component, coloured by city-event, so a pooled coefficient can be read
    # against the spread it came from and against per-unit agreement.
    pd.concat([_with_merged_func(feats_by_city[c])[time_rows + FUNC_MERGED_COLS]
               .assign(code=c) for c in codes], ignore_index=True).to_csv(
        os.path.join(OUTPUT_FUNC_VS_TEMP_RAW, 'func_vs_time_pooled_raw.csv'),
        index=False)
    pooled.to_csv(os.path.join(OUTPUT_FUNC_VS_TEMP_RAW,
                               'func_vs_time_pooled_rank.csv'), index=False)
    for t in time_rows:
        vis_func_vs_time_distribution(
            pooled, t, SF_CATEGORIES, rho_rank=rho.loc[t],
            save_path=os.path.join(OUTPUT_FVT_ALL,
                                   f'scatter_func_vs_{t.replace("-", "_")}_ALL.png'))
    rho.to_csv(os.path.join(OUTPUT_FUNC_VS_TEMP_RAW,
                            'heatmap_time_function_corr_ALL_rho.csv'))
    pval.to_csv(os.path.join(OUTPUT_FUNC_VS_TEMP_RAW,
                             'heatmap_time_function_corr_ALL_pval.csv'))
    print(f"  [pooled time×function] {len(pooled)} components from "
          f"{len(codes)} city-events -> {OUTPUT_FUNC_VS_TEMP}")
    return rho


def analysis_func_resi_corr_pooled(feats_by_city):
    """The func/distance/income × cum_loss heatmap over EVERY city-event's
    components at once — the resilience-metrics twin of
    analysis_time_function_corr_pooled, under the same within-unit rank
    normalisation (see that docstring for why raw pooling would let the
    large-magnitude units dominate).  Every column — the functional shares, the
    two single-cell extras AND cum_loss — is ranked within its own unit first,
    so a cell reads 'do the components leaning toward this feature also rank
    high on cum_loss within their own city', pooled across units.  Same layout
    as the per-unit figures in per_city/, so the two read side by side."""
    codes = list(feats_by_city)
    if len(codes) < 2:
        print("  [pooled func×resilience] fewer than 2 units; skipping.")
        return None
    cols = FUNC_MERGED_COLS + EXTRA_CORR_COLS + list(INTRA_RES_COLS)
    ranked = []
    for c in codes:
        t = _with_merged_func(feats_by_city[c])[cols].astype(float)
        ranked.append(t.rank(axis=0, na_option='keep').sub(0.5).div(len(t)))
    pooled = pd.concat(ranked, ignore_index=True)

    rho, pval = time_function_correlation(pooled, list(INTRA_RES_COLS),
                                          FUNC_MERGED_COLS + EXTRA_CORR_COLS,
                                          method='spearman')
    os.makedirs(OUTPUT_RESI_CORR_RANK_ALL, exist_ok=True)
    os.makedirs(OUTPUT_RESI_CORR_RANK_RAW, exist_ok=True)
    vis_heatmap_corr(
        rho, pval, time_cols=[], categories=SF_CATEGORIES,
        extra_cols=EXTRA_CORR_COLS,
        save_path=os.path.join(OUTPUT_RESI_CORR_RANK_ALL,
                               'heatmap_resilience_corr_ALL.png'),
    )
    rho.to_csv(os.path.join(OUTPUT_RESI_CORR_RANK_RAW,
                            'heatmap_resilience_corr_ALL_rho.csv'))
    pval.to_csv(os.path.join(OUTPUT_RESI_CORR_RANK_RAW,
                             'heatmap_resilience_corr_ALL_pval.csv'))
    print(f"  [pooled func×resilience] {len(pooled)} components from "
          f"{len(codes)} city-events -> {OUTPUT_RESI_CORR_RANK_ALL}")
    return rho


def analysis_resilience_corr(feats, tag, lambda_ctx=None):
    """One unit's func/distance/income × cum_loss Spearman heatmap, the
    per_city/ half of 4-func_vs_resi (the pooled all_city/ half is
    analysis_func_resi_corr_pooled).  One cell per MERGED functional share
    func_<cat>, plus single-cell mean_distance / median_income on the RIGHT;
    lambda_ctx tags the filename.  The per-function curve stacks are drawn
    separately by analysis_func_ordered_lines."""
    lambda_tag = _lambda_tag(lambda_ctx)
    merged = _with_merged_func(feats)
    feat_cols = TIME_COLS + FUNC_MERGED_COLS + EXTRA_CORR_COLS
    rho, pval = time_function_correlation(merged, INTRA_RES_COLS, feat_cols)
    vis_heatmap_corr(
        rho, pval, time_cols=[], categories=SF_CATEGORIES,
        extra_cols=EXTRA_CORR_COLS,
        save_path=os.path.join(OUTPUT_RESI_CORR_RANK_PER,
                               f'heatmap_resilience_corr_{lambda_tag}{tag}.png'),
    )


def analysis_func_ordered_lines(W, n_nor, n_dis, first_day_normal,
                                first_day_disaster, curves, feats, tag):
    """Per-function component line stacks.  For each functional category, two
    figures order the components top-to-bottom by descending combined
    functional share (share_from + share_to), so the components most tied to
    the function sit at the top (NaN share sorts to the bottom):
      - line_component_timeline_by_func/      the full-window W timeline
      - line_component_resilience_curves_by_func/  the disaster relative-
        activity curves r(d)
    Both use the same single-column stacked layout and label each row with its
    true component index."""
    os.makedirs(OUTPUT_TL_BY_FUNC, exist_ok=True)
    os.makedirs(OUTPUT_RC_BY_FUNC, exist_ok=True)
    for cat in SF_CATEGORIES:
        combined = feats[f'share_from_{cat}'] + feats[f'share_to_{cat}']
        order = combined.sort_values(ascending=False).index.tolist()
        vis_line_nmf_component_timeline(
            W[:n_nor], W[n_dis:], W_buffer=W[n_nor:n_dis],
            first_day_normal=first_day_normal, first_day_disaster=first_day_disaster,
            slots_per_day=SLOTS_ACTIVE, order=order,
            output_dir=OUTPUT_TL_BY_FUNC, tag=f'_by_{cat}{tag}',
        )
        vis_line_resilience_curves(
            curves, order=order,
            save_path=os.path.join(
                OUTPUT_RC_BY_FUNC,
                f'line_component_resilience_curves_by_{cat}{tag}.png'),
        )


# ── STEP 5 — cross-city feature tables ──────────────────────────────────────────

def _build_cross_city_feats(cfg, X_all, n_nor, n_dis, mapping, gdf, fit_time_cols, k,
                            share_lookup):
    """Decompose this city at `k` and assemble ONLY the cross-city predictor/target
    columns (share_from/to_<c>, mean_distance/std_distance, median_income_combined,
    RES_COLS) — no figures, no within-city side effects.  Used by the leave-one-city-
    event-out cross-city loop to build each unit's cross-city feats.  `k` is now
    always the unit's own rank-CV n_behaviors (it used to be a fixed k=10 for the
    held-out role, hence the explicit parameter).  l1_reg is the unit's CITY_EVENTS
    value.  `cat_lookup` is the unit's block-group
    -> category map from the STEP-1 global classification.
    Returns (feats, (W, H)) — the decomposition is reused by STEP 7."""
    W, H, weights = decompose_city(X_all, k, l1_reg=cfg['l1_reg'], fit_time_cols=fit_time_cols)
    M, _ = build_od_function_matrix_soft(H, mapping, share_lookup,
                                         SF_CATEGORIES)
    distances = build_distance_array(mapping, gdf)
    # Per-component loading-weighted median household income (ACS cache; 'combined' =
    # nan-aware mean of the origin+destination BG incomes per flow).  A cross-city
    # PREDICTOR (POOLED_FEATURE_COLS): its cross-city signal is the WITHIN-city spread
    # across components, which the static city-wide income does not carry.
    inc_csv = os.path.join(ACS_DATA_DIR,
                           f"{cfg['key']}_block_group_acs_income_{ACS_INCOME_YEAR}_raw.csv")
    ensure_city_income_raw(cfg['label'], gdf['aggr_id'].tolist(), inc_csv,
                           year=ACS_INCOME_YEAR)
    income = load_city_income(inc_csv)
    income_by_aggr = dict(zip(income['aggr_id'], income['median_household_income']))
    income_array = build_income_array(mapping, income_by_aggr, mode='combined')
    feats = pd.concat([
        functional_features(M, SF_CATEGORIES),
        spatial_features(H, distances),
        socioeconomic_features(H, income_array, name='median_income_combined'),
        resilience_features(W, n_nor, cfg['first_day_normal'], SLOTS_ACTIVE, n_dis=n_dis),
        recovery_curve_features(W, n_nor, cfg['first_day_normal'], SLOTS_ACTIVE,
                                n_dis=n_dis, max_rate=ALPHA_MAX_RATE,
                                min_fit_r2=ALPHA_MIN_FIT_R2,
                                min_std=ALPHA_MIN_STD, level_bounds=LEVEL_BOUNDS,
                                surge_bounds=SURGE_BOUNDS,
                                surge_rate_bounds=SURGE_RATE_BOUNDS),
    ], axis=1)
    if CURVE_ALPHA_BACKBONE == 'no_surge':
        # Ablation backbone (STEP-7 only): the same curves refit WITHOUT the
        # pulse term.  Guarded so the default configuration computes nothing new.
        feats['recovery_alpha_nosurge'] = recovery_curve_features(
            W, n_nor, cfg['first_day_normal'], SLOTS_ACTIVE, n_dis=n_dis,
            max_rate=ALPHA_MAX_RATE, min_fit_r2=ALPHA_MIN_FIT_R2,
            min_std=ALPHA_MIN_STD, level_bounds=LEVEL_BOUNDS,
            surge_bounds=SURGE_BOUNDS, surge_rate_bounds=SURGE_RATE_BOUNDS,
            include_surge=False)['recovery_alpha'].to_numpy()
    feats.insert(0, 'city', cfg['label'])
    # Observed landfall-day relative drop r0 = r(0) of each component's curve
    # (baseline-normalized, so cross-city comparable).  It is BOTH the curve
    # anchor AND — being the mechanically strongest, feature-independent
    # predictor of cum_loss — a STEP-7 predictor of cum_loss (leak-free: r0 is
    # the observed initial condition the paper's setting provides).
    feats['r0'] = resilience_curves(W, n_nor, cfg['first_day_normal'], SLOTS_ACTIVE,
                                    n_dis=n_dis).iloc[0].to_numpy(dtype=float)
    # Per-component NMF importance (‖W‖·‖H‖, full window) — kept for reference.
    feats['weight'] = weights
    # Normal-period baseline magnitude per component = (Σ over the normal slots of W) ×
    # (Σ over OD of H) = the component's share of the city's NORMAL baseline.  Because
    # the relative curves r_i are normalized by each component's normal baseline, this
    # baseline share is the weight that correctly reconstructs the city total curve
    # (r_total = Σ p_i·r_i) and hence city cum_loss = Σ p_i·cum_loss_i; the full-window
    # importance above is NOT this.  Used to aggregate component cum_loss -> city level.
    feats['weight_normal'] = W[:n_nor].sum(axis=0) * H.sum(axis=1)
    # Per-event-constant LEVEL features (only used by the cross-city pooled_train mode,
    # see LEVEL_FEATURE_COLS): Saffir-Simpson arrival intensity and the BG-pop-weighted
    # HEvOD evacuation strength.
    feats['hurricane_intensity'] = cfg['ss_intensity']
    feats['evac_level'] = cfg['evac_level']
    # The decomposition is returned alongside the table so the STEP-7 curve
    # prediction can rebuild the SAME components' observed curves without a
    # second (identical, deterministic) NMF fit.
    return feats, (W, H)


# ── STEP 6 — cross-city prediction analyses ─────────────────────────────────────

# Raw SLD columns that feed the per-category land-use score (counthu = housing proxy;
# the rest are the 2-digit employment sectors aggregated by FUNCTION_FIELD_MAP).
_SLD_SCORE_COLS = ['counthu', 'e8_ret', 'e8_off', 'e8_svc', 'e8_ent',
                   'e8_ind', 'e8_hlth', 'e8_ed', 'e8_pub']


def _city_poi_share(raw_csv, iwf):
    """City-WIDE POI-type share under the GLOBAL TF-IWF recipe.  Each block group's raw
    SLD category quantities are clamped to >=0 (matching the per-BG classification's
    sentinel handling) and SUMMED over the whole city; classify_dominant_function then
    runs on that single aggregated 'city' row with the passed pooled `iwf` vector, so
    the rarity weights match every other step of the pipeline and the TF becomes the
    city-wide proportion.  Returns a Series share_<cat> over SF_CATEGORIES (sums to 1)."""
    df = pd.read_csv(raw_csv)
    cols = [c for c in _SLD_SCORE_COLS if c in df.columns]
    summed = df[cols].apply(lambda s: s.astype(float).clip(lower=0).fillna(0.0)).sum(axis=0)
    out = classify_dominant_function(
        pd.DataFrame([summed]), residential_weight=LANDUSE_RESIDENTIAL_WEIGHT,
        weighting=LANDUSE_WEIGHTING, iwf=iwf,
        dominant_threshold=LANDUSE_DOMINANT_THRESHOLD)
    return pd.Series({c: float(out.iloc[0][f'share_{c}']) for c in SF_CATEGORIES})


CENTERED_QUANTILE_GRID = np.linspace(0.0005, 0.9995, 2000)

# One colour per hurricane, shared by the city-event map and the descriptive
# centred-distribution panels so a storm reads the same across the outputs.
STORM_COLOURS = {'Ida': '#0F4D92', 'Ian': '#B64342', 'Dorian': '#4C9F70',
                 'Isaias': '#9A4D8E', 'Sally': '#E28E2C'}

# ── ML-B2inc spread increment (adopted 2026-08-18) ───────────────────────────
# The centred-spectrum SCALE prediction.  B2 — the zero-parameter backbone
# ratio (spread scale in _quantile_mapped_chat) — is kept as the forced part,
# and ONE fitted increment is layered on top:
#     log σ_c = b0_B2 + 1·x_c + γ'PC_c ,   x_c = log s_bb,c
# γ (K=2 numbers) is the only fitted quantity; γ = 0 reproduces B2 exactly, so
# the increment is nested and L2-shrinks toward the current behaviour.  PC are
# the leading PCs of a 21-feature block per city: the 15 pairwise Pearson
# correlations of the six merged functional shares across the unit's
# components, PLUS the 6 log-variances — the diagonal of the same 6×6
# covariance matrix, which correlations normalise away — PLUS the
# log-variance of the components' median income (adopted 2026-08-18:
# repo-protocol LOO 1.486 -> 1.474, 8/13 units; flow-distance variance was
# tested in the same slot and HURT, 1.494, so it stays out).  The 13-unit LOO
# bake-off (2026-08): corr-only 1.402, corr+var 1.392, against B2 1.465 mean
# W1; covariances (corr × scale entangled in one number) 1.645, significantly
# WORSE than the ref (p = 0.017) — the diagonal helps only as separate columns
# next to the correlations, never multiplied into them.  γ is fitted by
# full-shape profile likelihood (residual density estimated by a Gaussian KDE
# on the pooled standardized residuals, alternated with the scale fit), and
# the two hyper-parameters — KDE bandwidth h and L2 strength λ — are chosen by
# an inner LOO over the training units, so the held unit touches nothing.
MLB2_FUNCS = ('residential', 'commercial', 'leisure', 'industrial',
              'health', 'public')
MLB2_K = 2
MLB2_H_GRID = (0.25, 0.40)
MLB2_LAM_GRID = (0.0, 1.0, 10.0, 100.0)


def _backbone_spread(r0_vec, days, mu_a):
    """Within-city sd of the BACKBONE-implied losses Σ_d (1 − logistic(d; r0_j,
    mu_a)) over the components with a usable anchor — the observable spread
    proxy (x = log of this) that both B2's ratio and the ML-B2inc fit rely on.
    NaN when fewer than two components qualify."""
    v = [float(np.sum(1.0 - 1.0 / (1.0 + (1.0 / r - 1.0)
                                   * np.exp(-mu_a * days))))
         for r in np.asarray(r0_vec, dtype=float)
         if np.isfinite(r) and r > 1e-6]
    return float(np.std(v)) if len(v) >= 2 else np.nan


def _mlb2_feature_vec(f):
    """A unit's 22-feature block: [15 pairwise correlations, 6 log-variances]
    of the six merged (from+to) functional shares across its cum_loss-usable
    components, plus the log-variance of the components' median income.
    Correlations carry the pairing structure, log-variances the per-function
    dispersion the correlations normalise away; they enter as separate columns
    (the covariance packaging of the same information tested significantly
    worse).  The income column earns its seat the same way: the income signal
    is WITHIN-city component spread, and its dispersion adds to the block
    (repo-protocol LOO 1.486 -> 1.474) where the flow-distance dispersion
    subtracts (1.494) — tested one at a time, 2026-08-18.  None when the unit
    has < 3 usable components or < 3 finite income values."""
    ok = np.isfinite(f['cum_loss'].to_numpy(dtype=float))
    if ok.sum() < 3:
        return None
    F = np.column_stack([(f[f'share_from_{k}'] + f[f'share_to_{k}'])
                         .to_numpy(dtype=float) for k in MLB2_FUNCS])[ok]
    sig = np.cov(F, rowvar=False, ddof=0)
    corr = np.zeros(15)
    for i, (a, b) in enumerate(itertools.combinations(range(6), 2)):
        u, v = F[:, a], F[:, b]
        if u.std() > 1e-12 and v.std() > 1e-12:
            corr[i] = float(np.corrcoef(u, v)[0, 1])
    if 'median_income_combined' not in f.columns:
        return None
    inc = f['median_income_combined'].to_numpy(dtype=float)[ok]
    inc = inc[np.isfinite(inc)]
    if len(inc) < 3:
        return None
    return np.concatenate([corr, np.log(np.diag(sig) + 1e-12),
                           [np.log(np.var(inc) + 1e-12)]])


def _mlb2_fit(train, cen_by, sbb_by, vec_by, h, lam, pg):
    """Fit γ on `train` at one (h, λ).  Returns (γ, project, b0_B2, x̄, bary):
    `project` maps a unit's 21-vector onto the K training-city PCs (z-scored,
    train-only), b0_B2 is B2's own implied intercept — so γ = 0 reproduces the
    plain backbone-ratio scaling exactly — and `bary` the training barycentre.
    γ maximises the full-shape profile likelihood: residuals y/σ share one
    density, estimated by a Gaussian KDE of bandwidth h on the standardized
    pooled residuals, with an L2 penalty λ‖γ‖² shrinking toward B2."""
    E = np.vstack([vec_by[c] for c in train])
    mu, sd = E.mean(0), E.std(0)
    sd[sd == 0] = 1.0
    ctr = ((E - mu) / sd).mean(0)
    P = np.linalg.svd((E - mu) / sd - ctr, full_matrices=False)[2][:MLB2_K]

    def project(code):
        return P @ ((vec_by[code] - mu) / sd - ctr)

    bary = np.mean([np.quantile(cen_by[c], pg) for c in train], axis=0)
    bary = bary - bary.mean()
    xb = float(np.mean([np.log(sbb_by[c]) for c in train]))
    b0 = xb - float(np.log(np.mean([sbb_by[c] for c in train]))) \
        + float(np.log(bary.std()))
    X = np.array([np.log(sbb_by[c]) - xb for c in train])
    G = np.vstack([project(c) for c in train])
    yv = np.concatenate([cen_by[c] for c in train])
    rep = np.concatenate([np.full(len(cen_by[c]), i)
                          for i, c in enumerate(train)]).astype(int)

    def negll(g):
        sig = np.exp(b0 + X + G @ g)
        eps = yv / sig[rep]
        centers = eps / eps.std()
        d = eps[:, None] - centers[None, :]
        dens = np.exp(-0.5 * (d / h) ** 2).sum(1) / (len(centers) * h
                                                     * np.sqrt(2 * np.pi))
        ll = -float(np.sum(np.log(sig[rep]))) \
            + float(np.sum(np.log(np.maximum(dens, 1e-300))))
        return -ll + lam * float(np.sum(g ** 2))

    g0 = np.zeros(MLB2_K)
    r = minimize(negll, g0, method='Nelder-Mead',
                 options=dict(xatol=1e-4, fatol=1e-7, maxiter=1200))
    g = r.x if r.fun <= negll(g0) else g0        # never worse than B2's γ = 0
    return g, project, b0, xb, bary


def _mlb2_spread_multiplier(held, rest, cen_by, sbb_by, vec_by,
                            pg=CENTERED_QUANTILE_GRID):
    """exp(γ'PC_held), the fitted correction ML-B2inc layers on B2's backbone-
    ratio spread scale.  (h, λ) are chosen by an inner LOO over the training
    units — each candidate is scored by the W1 between its scaled barycentre
    and the left-out training unit's true centred spectrum, the same metric
    the branch is judged by — then γ is refit on all training units at the
    winning pair.  Falls back to 1.0 (= plain B2) when the held unit's feature
    block or too many training ingredients are missing."""
    train = [c for c in rest
             if c in cen_by and c in vec_by
             and np.isfinite(sbb_by.get(c, np.nan))]
    if held not in vec_by or len(train) < 4:
        return 1.0
    best, cfg = None, None
    for h in MLB2_H_GRID:
        for lam in MLB2_LAM_GRID:
            errs = []
            for t in train:
                tr = [c for c in train if c != t]
                g, proj, b0, xb, bary = _mlb2_fit(tr, cen_by, sbb_by, vec_by,
                                                  h, lam, pg)
                sd_hat = float(np.exp(b0 + (np.log(sbb_by[t]) - xb)
                                      + proj(t) @ g))
                crv = bary * (sd_hat / bary.std())
                crv = crv - crv.mean()
                errs.append(float(np.abs(crv - np.quantile(cen_by[t], pg))
                                  .mean()))
            m = float(np.mean(errs))
            if best is None or m < best:
                best, cfg = m, (h, lam)
    g, proj, _, _, _ = _mlb2_fit(train, cen_by, sbb_by, vec_by,
                                 cfg[0], cfg[1], pg)
    return float(np.exp(proj(held) @ g))


# The backbone window the spread branch was DEVELOPED on.  Every ML-B2inc
# decision above -- the 21-feature block, K = 2, the h/lambda grids, the
# covariance rejection, the income column -- was scored by the mean W1 that
# analysis_centered_spectrum reports, and that ran this fixed 15-day grid: the
# adoption record's 1.486 -> 1.474 reproduces from centered_spectrum_metrics.csv
# to three decimals.  STEP 7 used to pass each held unit's FULL analysis horizon
# here instead (44/71/151 days, and fold-dependent, since the window came from
# whichever unit was held out), which applied a tuned component at coordinates
# it was never validated on.  Both callers now run this window (unified
# 2026-08-28).
SPREAD_BACKBONE_DAYS = np.arange(15.0)


def pooled_backbone_alpha(merged, codes):
    """The backbone recovery rate the spread proxy is defined at: the pooled
    mean of the training units' component recovery_alpha.  Kept separate from
    STEP 7's curve alpha-bar, which its own ablation switch may replace -- the
    spread proxy stays pinned to the rate ML-B2inc was fitted at."""
    a = [merged[c]['recovery_alpha'].dropna().to_numpy(dtype=float)
         for c in codes if c in merged]
    a = [v for v in a if len(v)]
    return float(np.mean(np.concatenate(a))) if a else np.nan


def centred_spectra(merged, codes):
    """{code -> its components' cum_loss on the unit's own mean}, the object
    the spread branch transfers: centring strips the city LEVEL so a spectrum
    describes only WITHIN-city inequality.  Units with fewer than two usable
    components are dropped."""
    out = {}
    for c in codes:
        if c not in merged:
            continue
        v = merged[c]['cum_loss'].dropna().to_numpy(dtype=float)
        if len(v) >= 2:
            out[c] = v - v.mean()
    return out


def spread_scale(held, rest, merged, mu_a):
    """THE spread scale: (base, mult), the held unit's centred-spectrum width
    relative to the training barycentre's.

    `base` is B2, the zero-parameter backbone ratio -- the held unit's backbone
    spread over the training units' mean, both on SPREAD_BACKBONE_DAYS.  `mult`
    is the fitted ML-B2inc increment exp(gamma'PC), which is 1.0 wherever the
    ingredients are missing, so base alone is always a valid answer.  The scale
    the caller applies is base * mult.

    Callers pass the SAME arguments and get the same number: STEP 6 scores it
    by W1 against the held unit's true spectrum, STEP 7 multiplies the
    barycentre by it before the location shift."""
    codes = [held] + [c for c in rest if c != held]
    cen = centred_spectra(merged, codes)
    sbb = {c: _backbone_spread(merged[c]['r0'].to_numpy(dtype=float),
                               SPREAD_BACKBONE_DAYS, mu_a)
           for c in codes if c in merged}
    s_train = [sbb[c] for c in rest
               if c in sbb and np.isfinite(sbb[c]) and sbb[c] > 0]
    base = (sbb[held] / float(np.mean(s_train))
            if s_train and np.isfinite(sbb.get(held, np.nan)) else 1.0)
    vec = {c: _mlb2_feature_vec(merged[c]) for c in codes if c in merged}
    vec = {c: v for c, v in vec.items() if v is not None}
    mult = _mlb2_spread_multiplier(held, rest, cen, sbb, vec,
                                   CENTERED_QUANTILE_GRID)
    return float(base), float(mult)


def analysis_centered_spectrum(feats_test, codes):
    """Score the CENTRED-SPECTRUM transfer on its own, leave-one-unit-out.

    STEP 7's quantile mapping assembles a unit's component cum_loss from three
    separable parts: an ORDERING (the rank channel), a LEVEL (the city-total
    anchor) and, between them, the shape-and-width of the unit's CENTRED loss
    spectrum -- how unequally its components fared once the unit's own mean is
    removed.  Only that middle part is measured here, so a gain or loss in it is
    not confounded with the other two.

    Estimator, the same one _quantile_mapped_chat uses: the WASSERSTEIN
    BARYCENTRE of the other units' centred spectra — their quantile functions
    averaged, not their samples pooled (see the `shape` note in
    _quantile_mapped_chat for why) — SCALED by the predicted spread: B2's
    backbone ratio times the fitted ML-B2inc increment exp(γ'PC) (see the
    MLB2_* block above).  Both spread ingredients are recomputed per fold from
    the training units only, with the fold's backbone rate mu_a taken as the
    pooled-train mean recovery_alpha — the same statistic STEP 7 uses — so the
    scored estimator is the one production applies.

    Metric W1, the 1-Wasserstein distance between the predicted and the true
    centred spectrum, evaluated as the mean |difference| of their quantile
    functions -- equivalently the area between their CDFs, which is what the
    figure shades.  Day-equivalents, lower better.  W1 is not an arbitrary
    choice: under the comonotone assignment the mapping performs, the
    per-component error is exactly W1 plus what the ordering gets wrong, so W1
    is precisely the part this transfer owns.

    SKILL normalises W1 against the do-nothing predictor -- forecast NO
    within-unit inequality at all, i.e. put every component at the unit mean, a
    point mass at 0 on the centred scale.  Its W1 against the truth is
    integral |Q_T(p)| dp, the average distance mass must travel from the unit
    mean to where the components actually are, so
    skill = 1 - W1 / W1_null:  1 perfect, 0 no better than predicting a flat
    spectrum, negative worse than that.  Note Q_T is the LINEARLY INTERPOLATED
    quantile function (np.quantile's default, the same representation the
    prediction and _quantile_mapped_chat's template use, so the comparison is
    like-for-like); that is NOT the same as the mean absolute centred loss the
    n atoms would give -- interpolating between few order statistics fills the
    gaps and pulls mass inward, e.g. CH_Dorian 1.123 against 1.434 over its 5
    components.  Being a ratio, skill is comparable across units, but a unit
    whose components genuinely all fared alike has a tiny denominator and can
    post a large negative skill on a small absolute error (CH_Dorian: W1 2.67,
    only the 12th largest of 13, but skill -1.38 because its W1_null is the
    smallest at 1.12) -- read skill and W1 together, never skill alone.

    Writes centered_spectrum_loo.png and raw/centered_spectrum_metrics.csv."""
    cen = {}
    for c in codes:
        y = feats_test[c]['cum_loss'].dropna().to_numpy(dtype=float)
        if len(y) >= 2:
            cen[c] = y - y.mean()
    usable = [c for c in codes if c in cen]
    if len(usable) < 3:
        print("  [centered_spectrum] fewer than 3 usable units; skipping.")
        return
    pg = CENTERED_QUANTILE_GRID
    vec_by = {c: _mlb2_feature_vec(feats_test[c]) for c in usable}
    vec_by = {c: v for c, v in vec_by.items() if v is not None}
    rows, pred = [], {}
    for held in usable:
        rest = [c for c in usable if c != held]
        bary = np.mean([np.quantile(cen[c], pg) for c in rest], axis=0)
        # The SHARED spread definition (spread_scale): the backbone ratio and
        # the ML-B2inc increment, both on SPREAD_BACKBONE_DAYS.  STEP 7 calls
        # the same function with the same arguments, so this figure and the
        # curve forecast's width are one number.
        mu_a = pooled_backbone_alpha(feats_test, rest)
        scale, mult = spread_scale(held, rest, feats_test, mu_a)
        est = bary * (scale * mult)
        est = est - est.mean()
        qt = np.quantile(cen[held], pg)
        w1 = float(np.abs(est - qt).mean())
        null = float(np.abs(qt).mean())          # all mass at the unit mean
        rows.append(dict(code=held, n=len(cen[held]), W1=w1, W1_null=null,
                         skill=(1.0 - w1 / null) if null > 0 else np.nan,
                         spread_scale=scale, mlb2_mult=mult))
        pred[held] = dict(bary=est, true=cen[held])
    M = pd.DataFrame(rows).set_index('code').reindex(usable)
    raw_dir = os.path.join(OUTPUT_CENTERED_DIST, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    M.to_csv(os.path.join(raw_dir, 'centered_spectrum_metrics.csv'))
    # Schematic twin of the LOO grid: ONE unlabelled panel showing what W1
    # measures.  The fold shown is the MEDIAN-W1 one, chosen by rule rather
    # than by eye so the illustration is not a flattering pick.
    mid = M['W1'].sort_values().index[len(M) // 2]
    vis_centered_spectrum_schematic(
        pred[mid]['bary'], pred[mid]['true'],
        save_path=os.path.join(OUTPUT_CENTERED_DIST,
                               'centered_spectrum_schematic.png'))
    labels = {c['code']: c['label'] for c in CITY_EVENTS}
    names = {code: f"{labels.get(code, code)} ({code.split('_', 1)[-1]})"
             for code in usable}
    # Descriptive companion: the OBSERVED centred spectra this branch transfers,
    # pooled and per unit.  ONE colour for every unit: the comparison each panel
    # asks for is unit against the pooled reference, and colouring by hurricane
    # invites a between-storm reading the panels do not support.
    storm_col = {code: '#0F4D92' for code in usable}
    vis_centered_distributions(
        {c: cen[c] for c in usable}, names=names, storms=storm_col,
        save_path=os.path.join(OUTPUT_CENTERED_DIST,
                               'centered_distributions.png'))

    # Spread-predictor diagnostics: what tracks the WIDTH the model must
    # predict?  x is that width (the SD of the centred spectrum, i.e. the
    # sigma of ML-B2inc); y is each candidate.  DESCRIPTIVE — computed on all
    # units at once, so they show association, not held-out skill.  The model
    # ingests the dispersion candidates as LOG-variances; SD is plotted because
    # it shares the target's units, and log is monotone so the sign and the
    # ordering of the association carry over unchanged.
    mu_all = np.concatenate([feats_test[c]['recovery_alpha'].dropna()
                             .to_numpy(dtype=float) for c in usable])
    mu_all = float(np.mean(mu_all)) if len(mu_all) else np.nan
    tgt = {c: float(np.std(cen[c])) for c in usable}
    # Every DISPERSION candidate in one figure — the backbone spread, the six
    # functional shares and income — since they answer the same question and
    # only differ in which quantity is being dispersed.  Flow distance is not
    # among them: it was tested in this slot and made the spread prediction
    # worse (1.486 -> 1.494), so it is out of the block and out of the figure.
    sdvar, fcorr = {}, {}

    def sd_of(v):
        v = v[np.isfinite(v)]
        return float(np.std(v)) if v.size else np.nan

    for c in usable:
        f = feats_test[c]
        ok = np.isfinite(f['cum_loss'].to_numpy(dtype=float))
        F = np.column_stack([(f[f'share_from_{k}'] + f[f'share_to_{k}'])
                             .to_numpy(dtype=float) for k in MLB2_FUNCS])[ok]
        # Row-major order chosen so the two NON-functional candidates hold
        # column 1 -- backbone on the top row, income directly below it -- and
        # the six functional shares fill columns 2-4; gap_after=0 then sets
        # that first column apart.
        sdvar.setdefault('backbone spread from $r_0$', {})[c] = (
            _backbone_spread(f['r0'].to_numpy(dtype=float),
                             SPREAD_BACKBONE_DAYS, mu_all))
        for i in range(3):
            sdvar.setdefault(f'SD of {MLB2_FUNCS[i]} share', {})[c] = (
                sd_of(F[:, i]))
        sdvar.setdefault('SD of median income', {})[c] = (
            sd_of(f['median_income_combined'].to_numpy(dtype=float)[ok]))
        for i in range(3, 6):
            sdvar.setdefault(f'SD of {MLB2_FUNCS[i]} share', {})[c] = (
                sd_of(F[:, i]))
        for a, b in itertools.combinations(range(6), 2):
            u, v = F[:, a], F[:, b]
            key = f'corr {MLB2_FUNCS[a]} - {MLB2_FUNCS[b]}'
            fcorr.setdefault(key, {})[c] = (
                float(np.corrcoef(u, v)[0, 1])
                if u.std() > 1e-12 and v.std() > 1e-12 else np.nan)
    vis_spread_vs_predictors(
        tgt, sdvar, storms=storm_col, ncol=4, gap_after=0,
        xlabel='candidate predictor',
        save_path=os.path.join(OUTPUT_CENTERED_DIST,
                               'spread_vs_predictors_sd.png'))
    vis_spread_vs_predictors(
        tgt, fcorr, storms=storm_col, ncol=5,
        xlabel='within-city correlation between the paired functional shares',
        save_path=os.path.join(OUTPUT_CENTERED_DIST,
                               'spread_vs_predictors_func_corr.png'))

    # The PCA step itself: which raw features the two components are built
    # from, and where each unit lands in the plane gamma operates on.
    pca_block = {c: _mlb2_feature_vec(feats_test[c]) for c in usable}
    pca_block = {c: v for c, v in pca_block.items() if v is not None}
    if len(pca_block) >= 4:
        fnames = ([f'corr {a} - {b}'
                   for a, b in itertools.combinations(MLB2_FUNCS, 2)]
                  + [f'log var {k}' for k in MLB2_FUNCS]
                  + ['log var median income'])
        vis_mlb2_pca(
            pca_block, fnames, names=names,
            save_path=os.path.join(OUTPUT_CENTERED_DIST,
                                   'spread_feature_pca.png'))

    # Conceptual pair: the one degree of freedom the spread channel controls.
    # units_per_inch is shared, so the two files' widths are to scale.
    pooled_all = np.concatenate([cen[c] for c in usable])
    for tag, sc in (('widened', 1.6), ('narrowed', 0.6)):
        # Type and height grow with the frame, but SUBLINEARLY (exponent 0.45):
        # matching the width ratio outright would leave the wide file with
        # labels taller than its own plotting area.
        g = (sc / 0.6) ** 0.45
        vis_spread_concept(
            pooled_all, sc, font_scale=g, height=5.2 * g,
            save_path=os.path.join(OUTPUT_CENTERED_DIST,
                                   f'spread_concept_{tag}.png'))
    vis_centered_spectrum_loo(
        pred, M, names=names,
        save_path=os.path.join(OUTPUT_CENTERED_DIST,
                               'centered_spectrum_loo.png'))
    print(f"  [centered_spectrum] mean W1 {M.W1.mean():.3f} day-eq, mean skill "
          f"{M.skill.mean():+.3f} over {len(M)} units -> {OUTPUT_CENTERED_DIST}")


def analysis_cross_city_resi_pred(feats_by_city, feats_test, units, codes, global_iwf,
                                  method='pearson', target_std='pooled_train'):
    """Reconstruct a CITY-LEVEL cum_loss from the cross-city component predictions (pooled
    func + distance + income, plus the two LEVEL covariates) and compare to GT.
    `global_iwf` is the pooled TF-IWF weight vector from the STEP-1 global classification,
    reused for the baseline's city-wide POI shares.
    ONLY runs for pearson + multi_city_std (target_std='pooled_train'), where the prediction
    has an absolute, day-equivalent meaning; any other setting is skipped.

    Per held-out city, ONE legacy reconstruction of the per-component cosine-kNN
    prediction is kept, CSV-only, as a reference point:
      * knn  — un-standardize each component with the fold's (mu, sigma) then
               weight_normal-average.  sigma ~ the large per-component cum_loss spread over-
               amplifies the mis-scaled prediction (good ranking, poor calibration).
    Two further reconstructions were removed 2026-08-04 (see the comment above the LOO
    loop): the pre-r0 D+aggr+denorm, superseded by the aggr_denorm strategy described
    below (whose only difference is the r0 predictor), and the city-level "D+city"
    cosine-kNN.
    Ground truth = cum_loss of the city's TOTAL activity curve (no decomposition).  A
    DECOMPOSITION-FREE BASELINE (cosine-weighted average of the OTHER cities' GT over
    [ss_intensity, city-wide POI shares, static city-wide mean BG income]) is emitted for
    reference.  The static income keeps the baseline decomposition-free (no flow weighting)
    and is the fair income-augmented control; it barely moves the baseline (its cross-city
    spread is tiny — two same-city events even share the value), which is what shows the
    income gain of the decompose methods comes from the flow-weighted WITHIN-city component
    spread, not from income as a city covariate.

    The RIDGE path additionally carries the 15 pairwise func-func interaction
    terms and replaces the component r0 with the CITY-level weighted-mean r0
    (both adopted 2026-08-12; see the comment above _merge for the evidence and
    the honest nested-LOO figure).  cosine-kNN stays on the original 9 features.

    The headline predictions are the aggr_denorm strategy — keep the standardized
    component predictions, AGGREGATE them to a city score s, then DENORMALIZE to
    day-equivalents through the shared rho stage (_city_total_from_scores): a ridge
    of city cum_loss on [s, r0_city, GDP] gives the direction, then a city-scale
    VARIANCE MATCH (de-shrink) sets the level, both learned NESTED-LOO on the other
    training cities (never the held city).  Run under BOTH predictors (cosine-kNN and
    RidgeCV), into decomp_pred_aggr_denorm/<model>/: the 13-point LOO calibration
    scatter (predicted vs actual city cum_loss, R² quantified) + raw_data/city_pred.csv.
    (A denorm_aggr sibling strategy was measured 2026-08-04 and removed — see the
    comment at _fold_ex.)

    Saves a two-panel figure (panel a: observed city cum_loss as bars with each
    prediction as a marker on an error stem, cities sorted by observed loss; panel b:
    leave-one-out R² per method with MAE annotated) as bar_cross_city_resi_pred.png/.svg,
    covering aggr_denorm+r0 under both predictors against the baseline — the legacy
    kNN(σ) stays CSV-only — plus the raw CSV (cum_loss_gt/_pred_knn/
    _pred_aggr_denorm_<model>/_baseline) under cross_city_resi_pred/city_total/."""
    if method != 'pearson' or target_std != 'pooled_train':
        print(f"  [cross_city_resi_pred] skipped: only pearson + multi_city_std "
              f"(got {method} + {target_std}).")
        return
    feature_cols = ([f'func_{c}' for c in SF_CATEGORIES]
                    + ['mean_distance', 'median_income_combined'])

    # The ridge path runs the shared CITY-TOTAL estimator (interactions + city
    # r0); see CITY_TOTAL_FEATURE_COLS for what it contains and why.  STEP 7's
    # _city_score reads the same constants, so the two cannot drift apart.
    def _merge(feats):
        return _with_city_total_feats(feats)

    train_merged = {c: _merge(feats_by_city[c]) for c in codes}
    test_merged  = {c: _merge(feats_test[c]) for c in codes}

    def _r0_city(c):
        """The unit's city-level r0 (constant within the city), the second
        predictor of the rho stage after the aggregate score s."""
        return float(train_merged[c]['r0_city'].iloc[0])

    # ── Decomposition-FREE baseline inputs ──
    # Per city (no NMF): the city-level GROUND-TRUTH cum_loss (total activity curve)
    # and a city feature vector = [ss_intensity, 6 city-wide POI-type shares, static
    # city-wide mean of the BG median household incomes].  The static income summary
    # (no flow/NMF weighting) keeps the baseline decomposition-free.  The baseline
    # (below) predicts a held-out city's cum_loss as a cosine-similarity-weighted
    # average of the OTHER cities' GT cum_loss.
    gt = {}
    feat = {}
    for c in codes:
        u = units[c]
        total = u['X_all'].sum(axis=0).reshape(-1, 1)
        gt[c] = float(resilience_features(
            total, u['n_nor'], u['first_day_nor'], SLOTS_ACTIVE,
            n_dis=u['n_dis'])['cum_loss'].iloc[0])
        raw_csv = os.path.join(SPACE_FUNCTION_DIR, f"{u['key']}_block_group_sld_raw.csv")
        assert ensure_city_landuse_raw(u['label'], u['gdf']['aggr_id'].tolist(),
                                       raw_csv) is not None
        inc_csv = os.path.join(ACS_DATA_DIR,
                               f"{u['key']}_block_group_acs_income_{ACS_INCOME_YEAR}_raw.csv")
        ensure_city_income_raw(u['label'], u['gdf']['aggr_id'].tolist(), inc_csv,
                               year=ACS_INCOME_YEAR)
        city_income = float(load_city_income(inc_csv)['median_household_income']
                            .dropna().mean())
        feat[c] = np.concatenate([[float(u['cfg']['ss_intensity'])],
                                  _city_poi_share(raw_csv, global_iwf).to_numpy(),
                                  [city_income]])
    # z-score each feature across the city-events (so intensity does not dominate the
    # cosine), then cosine similarity on the standardized 8-d vectors.
    F = np.vstack([feat[c] for c in codes])
    sd = F.std(axis=0); sd[sd == 0] = 1.0
    Fz = (F - F.mean(axis=0)) / sd
    zfeat = {c: Fz[i] for i, c in enumerate(codes)}

    def _cos(a, b):
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    # One held-out fold of the COMPONENT-level cosine-kNN transfer, returning the
    # sigma-un-standardized weight_normal city prediction (`pcity`, the legacy
    # kNN number).  Its aggregated STANDARDIZED score served the pre-r0
    # D+aggr+denorm reconstruction and went with it (2026-08-04); the surviving
    # aggr_denorm strategy computes its own in _fold_ex, on features + r0.
    def _fold(held, rest):
        fold = {held: test_merged[held]}
        fold.update({c: train_merged[c] for c in rest})
        _, pred, _ = cross_city_resilience(
            fold, RES_COLS, feature_cols, rank=False,
            split={'train': rest, 'test': [held]}, target_std=target_std,
            level_feature_cols=LEVEL_FEATURE_COLS, model='cosine_knn',
            pooled_feature_cols=POOLED_FEATURE_COLS,
            min_rows=CROSS_CITY_MIN_ROWS)
        pm = pred.get(held, {}).get('cum_loss')
        if pm is None:
            return None
        yte_std, ypred_std, cidx = pm
        yte_std = np.asarray(yte_std, dtype=float); ypred_std = np.asarray(ypred_std, dtype=float)
        raw_true = feats_test[held].loc[cidx, 'cum_loss'].to_numpy(dtype=float)
        s = float(np.std(yte_std)); sigma = (float(np.std(raw_true)) / s) if s > 0 else 1.0
        mu = float(np.mean(raw_true) - sigma * np.mean(yte_std))
        pred_raw = ypred_std * sigma + mu
        w = feats_test[held].loc[cidx, 'weight_normal'].to_numpy(dtype=float); wsum = w.sum()
        pcity = float((w * pred_raw).sum() / wsum) if wsum > 0 else float(pred_raw.mean())
        return dict(pcity=pcity)

    # ── The aggr_denorm STRATEGY, both predictors ──
    # Same component-level transfer as _fold, predictor selectable.  Returns
    # scity, the weight_normal aggregate of the STANDARDIZED component
    # predictions (= the city score s), which the caller feeds — with the unit's
    # r0_city and GDP — to the shared rho stage _city_total_from_scores (city-
    # level ridge on [s, r0_city, GDP] then a nested-LOO variance match).
    # (A denorm_aggr sibling — invert each component with the pooled-train
    # (mu, sigma) BEFORE aggregating — was measured 2026-08-04 and removed: an
    # affine denorm commutes with the weighted average, so it only swaps the
    # calibration frame to the component-level sigma, and both predictors
    # scored WORSE under it: cos-kNN R2 -0.27, ridge -0.80, vs +0.17 for
    # aggr_denorm/ridge here.)
    fcols_r0 = feature_cols + ['r0']
    pooled_r0 = POOLED_FEATURE_COLS + ['r0']
    # The RIDGE path runs on the extended set (interactions + city r0); the
    # cosine-kNN path keeps the original 9 so the figure's two predictors stay
    # a like-for-like comparison of the PREDICTOR, and the extension is read
    # against the predictor it was tuned on.  kNN gains nothing from the extra
    # columns anyway: a cosine over 24 mostly-collinear products dilutes the
    # neighbourhood it depends on (sandbox 2026-08-12).
    def _fold_ex(held, rest, model_id):
        """One fold's aggregate score s, through the SHARED phi stage
        (city_total_score) -- the same call STEP 7's quantile-mapping anchor
        makes, so the headline figure and the anchor cannot drift apart.  The
        cosine-kNN arm is the one deliberate override: it keeps the smaller
        pre-interaction feature set (see fcols_r0 above)."""
        ridge_arm = (model_id == 'ridge')
        sc = city_total_score(
            held, rest, {**train_merged, held: test_merged[held]}, feats_test,
            model_id,
            feature_cols=None if ridge_arm else fcols_r0,
            pooled_cols=None if ridge_arm else pooled_r0)
        return None if not np.isfinite(sc) else dict(scity=float(sc))

    # The legacy reconstruction of the component predictions into a city
    # day-equivalent, kept CSV-only as a reference point:
    #   knn — un-standardize each component with the fold's (mu, sigma) then
    #         weight_normal-average.  sigma ~ the huge per-component cum_loss
    #         spread over-amplifies the (mis-scaled) prediction -> good ranking,
    #         poor calibration.
    # (Two further reconstructions lived here and were removed 2026-08-04: the
    # pre-r0 D+aggr+denorm, superseded by the aggr_denorm strategy defined just
    # above, whose only difference is the r0 predictor (LOO R² -0.73 -> -0.46 on
    # cosine-kNN), and a city-level cosine-kNN over the weight_normal-aggregated
    # feature vector predicting GT directly ("D+city", LOO R² -0.15 — its low MAE
    # came from predicting near the training mean, its correlation with GT was
    # 0.00).)
    rows = []
    for held in codes:
        rest = [c for c in codes if c != held]
        # The legacy kNN(sigma) number must NOT gate the row: it is a CSV-only
        # reference, so a fold where it cannot be formed still has a usable
        # ground truth, baseline and aggr_denorm prediction.  NaN it and carry
        # on — the same failure semantics _fold_ex already uses below.
        outer = _fold(held, rest)
        if outer is None:
            print(f"  [cross_city_resi_pred] {held}: legacy kNN(sigma) reference "
                  f"unavailable; the row keeps its other predictions.")
        pred_knn = outer['pcity'] if outer is not None else np.nan

        # Baseline (no decomposition): cosine-similarity-weighted average of the OTHER
        # cities' GT cum_loss over city features [ss_intensity, city-wide POI shares].
        sims = np.array([max(_cos(zfeat[held], zfeat[t]), 0.0) for t in rest])
        gts = np.array([gt[t] for t in rest])
        base = (float((sims * gts).sum() / sims.sum()) if sims.sum() > 0 else float(gts.mean()))

        row = {'code': held, 'cum_loss_gt': gt[held],
               'cum_loss_pred_knn': pred_knn, 'cum_loss_baseline': base}

        # The aggr_denorm strategy x both city-level predictors: aggregate
        # the standardized component scores into s, then the
        # city-level rho stage (_city_total_from_scores) scores [s, r0_city, GDP]
        # and variance-matches onto day-equivalents.  The inner s values must
        # come from the SAME predictor and feature set, so the nested loop reruns
        # _fold_ex per fold.  Both predictors go through the identical rho stage,
        # so the figure stays a like-for-like comparison of the PREDICTOR.
        for model_dir, model_id in CITY_TOTAL_DECOMP_MODELS.items():
            outer_ex = _fold_ex(held, rest, model_id)
            if outer_ex is None:
                row[f'cum_loss_pred_aggr_denorm_{model_dir}'] = np.nan
                continue
            s_in2, x_in2, gt_in2 = [], [], []
            for s2 in rest:
                inn2 = _fold_ex(s2, [c for c in rest if c != s2], model_id)
                if inn2 is not None:
                    s_in2.append(inn2['scity'])
                    x_in2.append([_r0_city(s2), _msa_gdp_table()[s2]])
                    gt_in2.append(gt[s2])
            row[f'cum_loss_pred_aggr_denorm_{model_dir}'] = _city_total_from_scores(
                outer_ex['scity'], [_r0_city(held), _msa_gdp_table()[held]],
                s_in2, x_in2, gt_in2)

        rows.append(row)

    if not rows:
        print("  [cross_city_resi_pred] no usable city-events; skipping.")
        return
    res = pd.DataFrame(rows).set_index('code')
    res = res.reindex([c for c in codes if c in res.index])
    raw_dir = os.path.join(OUTPUT_CITY_TOTAL, 'raw_data')
    os.makedirs(raw_dir, exist_ok=True)
    res.to_csv(os.path.join(raw_dir, 'cross_city_resi_pred.csv'))

    # One folder per predictor: the 13-point LOO calibration scatter
    # (predicted vs actual city cum_loss, R² + MAE in the title) and its raw
    # data, for the aggr_denorm strategy on features+r0.
    _STRATEGY_TITLES = {
        'aggr_denorm': 'Decompose+aggr+denorm',
    }
    for strat, strat_title in _STRATEGY_TITLES.items():
        for model_dir in CITY_TOTAL_DECOMP_MODELS:
            col = f'cum_loss_pred_{strat}_{model_dir}'
            if col not in res.columns:
                continue
            sdir = os.path.join(OUTPUT_CITY_TOTAL, f'decomp_pred_{strat}',
                                model_dir)
            sraw = os.path.join(sdir, 'raw_data')
            os.makedirs(sraw, exist_ok=True)
            res[['cum_loss_gt', col]].rename(
                columns={col: 'cum_loss_pred'}).to_csv(
                os.path.join(sraw, 'city_pred.csv'))
            vis_scatter_city_pred(
                res, col,
                title=f'{strat_title} — {model_dir}, LOO over {len(res)} '
                      f'city-events',
                save_path=os.path.join(sdir, 'scatter_city_pred.png'))
            sub = res[['cum_loss_gt', col]].dropna()
            r2v = (r2_score(sub['cum_loss_gt'], sub[col])
                   if len(sub) >= 2 else np.nan)
            print(f"  [cross_city_resi_pred] decomp_pred_{strat}/{model_dir}: "
                  f"R2={r2v:+.3f} "
                  f"MAE={float((sub['cum_loss_gt'] - sub[col]).abs().mean()):.3f}")

    # The figure omits the legacy kNN(σ) prediction (still in the CSV); it shows
    # the aggr_denorm strategy under both predictors against the
    # decomposition-free baseline.  Labels are spelled out — this is the
    # reader-facing figure, so no internal codenames.
    # The series are BUILT from CITY_TOTAL_DECOMP_MODELS rather than hardcoded:
    # those keys name the columns written above, and vis_bar_cross_city_resi_pred
    # silently drops a column it cannot find, so a hardcoded name that drifted
    # out of sync would yield a headline figure showing only the baseline with
    # no error anywhere.  The assertion below makes that failure loud instead.
    _MODEL_LABELS = {'cos_KNN': 'cosine-kNN',
                     'ridge': 'ridge + func interactions + city r0'}
    bar_pred_cols = tuple(
        (f'Component transfer, {_MODEL_LABELS.get(m, m)}',
         f'cum_loss_pred_aggr_denorm_{m}') for m in CITY_TOTAL_DECOMP_MODELS)
    _missing = [c for _l, c in bar_pred_cols if c not in res.columns]
    if _missing:
        raise KeyError(f"analysis_cross_city_resi_pred: the headline figure's "
                       f"prediction columns {_missing} were never written; "
                       f"CITY_TOTAL_DECOMP_MODELS and the LOO loop disagree.")
    vis_bar_cross_city_resi_pred(
        res, save_path=os.path.join(OUTPUT_CITY_TOTAL,
                                    'bar_cross_city_resi_pred.png'),
        pred_cols=bar_pred_cols,
        baseline_label='City-similarity baseline (no decomposition)')
    # Which CITY-level quantities track the city total at all?  Same renderer
    # as the spread diagnostics, so the two figures read the same way.  These
    # are the covariates the rho stage may draw on: GDP and the day-0 anchor
    # are its actual predictors, the two event-level severity measures are the
    # obvious candidates a reader will ask about.  DESCRIPTIVE -- computed on
    # all units at once, so association, not held-out skill.
    def _r0_city_of(c):
        f = feats_test[c]
        w = f['weight_normal'].to_numpy(dtype=float)
        r = f['r0'].to_numpy(dtype=float)
        return float(w @ r / w.sum()) if w.sum() > 0 else float(np.nanmean(r))

    gdp = _msa_gdp_table()
    nl = chr(10)
    city_pred = {
        'landfall day $r_0$':
            {c: _r0_city_of(c) for c in codes},
        'hurricane intensity' + nl + 'Saffir-Simpson at arrival':
            {c: float(feats_test[c]['hurricane_intensity'].iloc[0])
             for c in codes},
        'evacuation order strength' + nl + 'population-weighted':
            {c: float(feats_test[c]['evac_level'].iloc[0]) for c in codes},
        'MSA real GDP, 2019':
            {c: gdp[c] for c in codes},
    }
    vis_spread_vs_predictors(
        {c: gt[c] for c in codes}, city_pred, ncol=4, title_fontsize=24,
        xlabel='candidate predictor', ylabel='city total loss',
        save_path=os.path.join(OUTPUT_CITY_TOTAL,
                               'city_total_vs_predictors.png'))

    print(f"  [cross_city_resi_pred] -> {OUTPUT_CITY_TOTAL} "
          f"({len(res)} city-events)")


# Seed for the pairwise-transfer Louvain partition.  Louvain's node sweep is
# order-randomised, so an unpinned seed would repartition on every run and the
# heatmap's boxes would move between otherwise identical pipeline runs.  The
# partition is barely seed-sensitive here (50 seeds at gamma=1.0: ridge 50/50
# identical, cosine-kNN 48/50) — the seed buys literal reproducibility, not
# stability.
PAIR_LOUVAIN_SEED = 0
# Modularity's resolution gamma scales the null-model term gamma*k_i*k_j/2m:
# larger gamma charges more for keeping two units together, so communities
# come out smaller and more numerous.  Louvain takes no cluster-COUNT
# argument, so gamma is the only dial on how many boxes appear.
#
# For the 13-UNIT PAIR HEATMAP gamma is AUTO-SELECTED per run (2026-08-31,
# _pair_louvain_auto_gamma) by the rule that originally chose the fixed 1.2
# on the 7-feature matrices: sweep PAIR_LOUVAIN_GAMMA_GRID at the pinned
# seed, find the plateau where everything collapses into <= 2 communities and
# the point where the partition shatters into >= 5, and take the smallest
# gamma strictly between them (no such gap -> fall back to the fixed value
# below).  The sweep is written to raw_data/pair_louvain_gamma_sweep.csv so
# the choice is auditable.  On the 23-feature matrix the rule lands on 1.3
# (3 communities, sizes [6, 5, 2]) — with the standing caveat that at that
# gamma modularity is already slightly negative and the partition varies
# across Louvain seeds, so the boxes are a DISPLAY ordering, not a finding;
# nothing computational consumes them (RANK_TRAIN_SCOPE='pooled').
#
# The fixed value below remains the default for OTHER _transfer_communities
# callers (the 6-function row-ordering inside cluster_function_heatmap),
# where the 13-unit sweep rule has no meaning, and the fallback of the auto
# rule.  Modularity's resolution limit (~sqrt(2m), m ~ 29-34 on the pair
# graph) sits at the scale of these communities, so gamma below ~1 cannot
# split them whatever the data says.
PAIR_LOUVAIN_RESOLUTION = 1.2
PAIR_LOUVAIN_GAMMA_GRID = (0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                           1.7, 2.0, 3.0, 5.0)


def _pair_graph(mat):
    """The undirected non-negative graph every Louvain step here runs on:
    diagonal dropped, negatives clipped to 0, directions averaged (the three
    reductions _transfer_communities documents)."""
    import networkx as nx
    codes = list(mat.index)
    A = np.nan_to_num(mat.reindex(index=codes, columns=codes)
                      .to_numpy(dtype=float), nan=0.0)
    np.fill_diagonal(A, 0.0)
    A = np.clip(A, 0.0, None)
    Wsym = (A + A.T) / 2.0
    G = nx.Graph()
    G.add_nodes_from(range(len(codes)))
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            if Wsym[i, j] > 0:
                G.add_edge(i, j, weight=float(Wsym[i, j]))
    return G, codes


def _pair_louvain_gamma_sweep(mat, seed=PAIR_LOUVAIN_SEED,
                              grid=PAIR_LOUVAIN_GAMMA_GRID):
    """One row per candidate gamma: community count and sizes at the pinned
    seed, modularity of that partition, and how many DISTINCT partitions 40
    seeds produce (1 = the partition is a property of the data, many = of the
    seed).  The auto rule reads k_seed0; the rest is the audit trail."""
    from networkx.algorithms.community import (louvain_communities,
                                               modularity)
    G, _codes = _pair_graph(mat)
    rows = []
    for g in grid:
        c0 = louvain_communities(G, weight='weight', seed=seed, resolution=g)
        q0 = modularity(G, c0, weight='weight', resolution=g)
        distinct = {tuple(sorted(tuple(sorted(x)) for x in
                                 louvain_communities(G, weight='weight',
                                                     seed=s2, resolution=g)))
                    for s2 in range(40)}
        rows.append(dict(gamma=g, k_seed0=len(c0),
                         sizes=str(sorted((len(x) for x in c0),
                                          reverse=True)),
                         modularity=round(float(q0), 4),
                         distinct_40seeds=len(distinct)))
    return pd.DataFrame(rows)


def _pair_louvain_auto_gamma(sweep):
    """The re-runnable form of the rule that chose 1.2 by hand in 2026-08:
    the smallest gamma strictly between the <= 2-community collapse plateau
    and the >= 5-community shatter point.  Falls back to the fixed
    PAIR_LOUVAIN_RESOLUTION when the grid has no such gap."""
    k = dict(zip(sweep['gamma'], sweep['k_seed0']))
    collapsed = [g for g in k if k[g] <= 2]
    shattered = [g for g in k if k[g] >= 5]
    lo = max(collapsed) if collapsed else min(k)
    hi = min(shattered) if shattered else max(k)
    cand = sorted(g for g in k if lo < g < hi)
    return float(cand[0]) if cand else float(PAIR_LOUVAIN_RESOLUTION)


def _transfer_communities(mat, seed=PAIR_LOUVAIN_SEED,
                          resolution=PAIR_LOUVAIN_RESOLUTION):
    """Louvain communities over the MEASURED pairwise transfer matrix, used only
    to order and box the heatmap.

    The matrix is directed (rows=train, cols=test) and Louvain needs an
    undirected non-negative graph, so three reductions are applied, in order:
      * the diagonal is dropped — it is a within-unit leave-one-component-out,
        not a transfer between units, and as a self-loop it would only inflate
        each node's own degree;
      * negative R² is clipped to 0 — a source that transfers worse than the
        test unit's own mean carries no affinity, and a negative edge weight is
        not a modularity input;
      * the two directions are averaged, w_ab = (r_ab + r_ba)/2, so a
        one-directional transfer still counts as half an edge.

    NOTE this partition is DESCRIPTIVE only.  Every edge is an R² measured
    against the test unit's real labels, so the clusters cannot be used to pick
    a source for a genuinely unseen city — that is the label leakage the
    2026-07 cluster experiment was rolled back for.  STEP 8 answers the
    source-selection question from features alone instead.

    Returns (ordered_codes, {code -> cluster_id}, [(start, size, cluster_id)]).
    """
    from networkx.algorithms.community import louvain_communities

    G, codes = _pair_graph(mat)
    comms = louvain_communities(G, weight='weight', seed=seed,
                                resolution=resolution)
    # Deterministic presentation: communities ordered by their first member's
    # original position, members kept in the caller's order.
    comms = sorted((sorted(c) for c in comms), key=lambda c: c[0])

    ordered, labels, blocks, start = [], {}, [], 0
    for cid, members in enumerate(comms, start=1):
        for i in members:
            ordered.append(codes[i])
            labels[codes[i]] = cid
        blocks.append((start, len(members), cid))
        start += len(members)
    return ordered, labels, blocks


def analysis_cross_city_pairs(feats_train, feats_test, codes, method='spearman',
                              target_std=None, subdir=None, level_feature_cols=(),
                              model='ridge', pooled_feature_cols=(),
                              min_rows=CROSS_CITY_MIN_ROWS):
    """Pairwise single-train -> single-test cross-city transfer for cum_loss.
    For each ORDERED pair (train=a, test=b): both a and b use their OWN rank-CV k
    (feats_train and feats_test are the same own-k table per unit — the separate
    parameters survive from when the test role was pinned at k = 10); the diagonal
    (a==b) is a's within-unit leave-one-component-out.  The target standardization is
    HARD-PAIRED to the method via CROSS_CITY_METHOD_STD (target_std=None resolves it,
    a mismatching explicit value raises ValueError).  Saves the |codes|x|codes| matrix
    (rows=train, cols=test, value = the pair's cum_loss Spearman ρ) as
    cross_city_pair_heatmap (.png +
    raw_data/.csv) under cross_city_resi_pred/<subdir>/ (subdir=None -> the method's
    paired label).  Returns the matrix."""
    # Defensive guard: same pairing rule as analysis_cross_city (std AND output label),
    # checked up front.
    if method not in CROSS_CITY_METHOD_STD:
        raise ValueError(
            f"analysis_cross_city_pairs: method '{method}' has no cross-city "
            f"output (CROSS_CITY_METHOD_STD lists "
            f"{sorted(CROSS_CITY_METHOD_STD)}); the raw-value path was retired")
    expected_std, paired_label = CROSS_CITY_METHOD_STD[method]
    std_label = subdir if subdir is not None else paired_label
    if target_std is not None and target_std != expected_std:
        raise ValueError(
            f"analysis_cross_city_pairs: method '{method}' is paired with target_std "
            f"'{expected_std}' (CROSS_CITY_METHOD_STD); got '{target_std}'")
    if not str(std_label).startswith(paired_label):
        raise ValueError(
            f"analysis_cross_city_pairs: method '{method}' writes under subdir "
            f"'{paired_label}' (CROSS_CITY_METHOD_STD); got '{std_label}'")
    target_std = expected_std
    subdir = std_label
    # method/target_std/model/level_feature_cols/pooled_feature_cols survive as
    # the pairing GUARD above only: what is actually fitted is rank_predict, so
    # the pair matrix cannot drift away from the rest of the rank channel.
    # THE rank channel (rank_predict), one training city at a time, on the
    # full RANK_FEATURE_COLS.  Nothing trains on this matrix's Louvain
    # partition any more (RANK_TRAIN_SCOPE='pooled'), so the boxes on the
    # figure are a display ordering, not a training recipe.
    train_merged = {c: rank_merge_feats(feats_train[c]) for c in codes}
    test_merged = {c: rank_merge_feats(feats_test[c]) for c in codes}

    # Cell = SPEARMAN rho of the pair's predicted vs actual cum_loss ordering,
    # not the engine's R².  This channel's target is a within-unit rank, so the
    # ordering is what a pair can lend; R² additionally scores the amplitude of
    # the prediction, which is nuisance here (rank-projecting the same
    # predictions leaves rho untouched and moves R² by ~0.3).  R² also has no
    # lower bound — the R² version of this matrix ran to -11.96 — so after the
    # negative clipping below, a handful of catastrophic pairs decided which
    # edges existed at all.  rho is bounded to [-1, 1] and needs no clipping in
    # the colour scale.  Invariant to the standardization, so it is read off the
    # engine's standardized arrays directly.
    mat = pd.DataFrame(index=codes, columns=codes, dtype=float)   # rows=train, cols=test
    for a in codes:
        for b in codes:
            feats_pair = ({a: test_merged[a]} if a == b           # self: within-a LOO
                          else {a: train_merged[a], b: test_merged[b]})
            score = rank_predict(b, [a], feats_pair, min_rows=min_rows)
            if score is None or len(score) < 3:
                mat.loc[a, b] = np.nan
                continue
            obs = feats_pair[b].loc[score.index, 'cum_loss'].astype(float)
            keep = obs.notna() & score.notna()
            mat.loc[a, b] = (float(spearmanr(obs[keep], score[keep]).statistic)
                             if int(keep.sum()) >= 3 else np.nan)

    out_dir = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED, subdir)
    raw_dir = os.path.join(out_dir, 'raw_data')
    os.makedirs(raw_dir, exist_ok=True)

    # Louvain over the off-diagonal transfer; the matrix is REORDERED into the
    # partition so each community is a contiguous block the figure can box, and
    # the CSV is written in the same order as the figure it backs.
    sweep = _pair_louvain_gamma_sweep(mat)
    gamma = _pair_louvain_auto_gamma(sweep)
    sweep.to_csv(os.path.join(raw_dir, 'pair_louvain_gamma_sweep.csv'),
                 index=False)
    ordered, cl_labels, blocks = _transfer_communities(mat, resolution=gamma)
    mat = mat.reindex(index=ordered, columns=ordered)
    mat.to_csv(os.path.join(raw_dir, 'cross_city_pair_heatmap.csv'))
    pd.Series(cl_labels, name='cluster').rename_axis('code').loc[ordered].to_csv(
        os.path.join(raw_dir, 'cross_city_pair_clusters.csv'))
    vis_heatmap_pair_transfer(
        mat, xlabel='test city-event', ylabel='train city-event',
        blocks=blocks, vmax=1.0, cbar_label='Spearman ρ',
        # Abbreviated ticks: 13 full city-event names on both axes crowd the
        # grid and shrink the cells; the code already carries the storm, which
        # is the part that has to stay (Wilmington appears under two storms).
        names={c['code']: c['code'].replace('_', ' (') + ')'
               for c in CITY_EVENTS},
        save_path=os.path.join(out_dir, 'cross_city_pair_heatmap.png'))
    print(f"  [pairwise] cum_loss Spearman ρ heatmap ({method}) -> "
          f"{os.path.join(out_dir, 'cross_city_pair_heatmap.png')}  "
          f"[auto gamma {gamma}; {len(blocks)} Louvain clusters: "
          f"{', '.join(f'C{c}={s}' for _, s, c in blocks)}]")
    return mat


# ── STEP 7 — cross-city curve prediction ────────────────────────────────────────

# Display labels of the method lines (dict order = plot / colour order).
# 'pred'  — component-wise plateau inversion: each test component's plateau L
#           solved from its QUANTILE-MAPPED cum_loss prediction (rank-path
#           ordering + pooled-train spread + raw-path city total), aggregated
#           by weight_normal.
# 'city'  — city-wise: ONE α for the whole city predicted from the OTHER
#           cities' city-level features (decomposition-free analog of 'pred').
# 'pred' carries NO model tag on purpose.  It used to read "(kNN)" from the
# days when the component channel WAS a cosine-kNN; it is now a three-branch
# composite -- ORDER from the ridge rank channel, SHAPE from the barycentre
# scaled by the backbone ratio times the ML-B2inc increment, LEVEL from the
# ridge-plus-de-shrink city total.  The cosine-kNN raw channel survives only as
# a fallback for components without a usable rank score, and at the current
# 81/81 coverage it contributes nothing, so naming the line after it was wrong.
# 'city' DOES stay tagged kNN: that baseline really is a cosine-kNN over city
# feature vectors predicting one city-level alpha.
_CURVE_METHOD_LABELS = {
    'pred':       'component-based prediction',
    'city':       'city-wise prediction (kNN)',
    'oracle':     'oracle',
    'train_mean': 'train-mean',
}

# Which method lines appear in the CITY-LEVEL figures (the magnitude overlay and
# the whole-curve MAE bar).  The oracle is deliberately absent: it reads each
# test component's own curve, so it is a descriptive ceiling rather than a
# forecast, and putting it beside three genuine forecasts invites reading it as
# a fourth.  It is NOT removed from the analysis — it still carries the
# component grid (where it shows what the model family can express), the metrics
# CSV, the raw per-day curves and the ungated columns of the parameter table.
_CITY_FIGURE_METHODS = tuple(m for m in _CURVE_METHOD_LABELS if m != 'oracle')


def analysis_cross_city_curve_pred(feats_by_city, feats_test, units, codes, dec_test):
    """Predict each held-out unit's ENTIRE disaster-window mobility curve, not just
    its cum_loss.  Leave-one-unit-out; the pearson/pooled_train frame throughout
    (absolute parameters are what the curve model needs).

    The fitted recovery curve is the SURGE-PLUS-RELAXATION model
    r(t) = L/(1 + (L/r0−1)e^(−α·t)) + B·t·e^(−b·t), anchored at the OBSERVED
    landfall-day value r0 (the pulse vanishes at t = 0).  L absorbs the end
    level, the signed pulse (B, b) absorbs the hump/dip transient, so the
    jointly-fitted α is the CLEAN recovery rate.  Every FORECAST line holds
    the rate at the POOLED-TRAIN MEAN and drops the pulse (B = 0), because
    neither the rate nor any shape parameter transfers across cities via
    features — only cum_loss does (its within-city Ridge R² reaches 0.70; the
    rate's is negative in all five units).  The component-wise forecast
    therefore predicts cum_loss and inverts it into the family's ONE remaining
    free parameter, the plateau L (see _plateau_inversion_curves), so the full
    fitted model serves the FIT (the recovery_alpha metric and the oracle
    line) while the forecast stays inside the same family.
      1. cum_loss of every TEST component is predicted cross-city by QUANTILE
         MAPPING over two kNN channels (features = the STEP-6 pooled
         predictors PLUS the observed r0 — the day-0 drop is the mechanically
         strongest predictor of cum_loss and already an allowed input as the
         anchor):
           total     the D+aggr+denorm city cum_loss of STEP 6 (standardized
                     component predictions aggregated by weight_normal, then
                     one city-scale variance match calibrated nested-LOO on
                     the training units): city-total MAE 0.270 against the
                     0.785 of the raw path's own aggregate, which stays as
                     the fallback;
           ordering  RANK path (within_unit frame: features and target
                     rankdata'd per city) — rank transfer is immune to
                     cross-city level/scale drift, so it carries the
                     within-city ordering better than the raw values do;
           shape     pooled-TRAIN cum_loss quantiles at (k−0.5)/n, assigned
                     in the predicted order;
           spread    those quantiles scaled by the held unit's backbone-loss
                     spread over the training units' mean backbone-loss
                     spread (observable proxy, ratio form, no fitted
                     parameter), then shifted to match the total.
         Either channel needs only the test unit's FEATURES, so the TEST
         table's target column is replaced by a finite placeholder ramp the
         prediction never reads.
      2. Four method lines synthesize the curves:
           pred        component-wise PLATEAU-INVERSION forecast: the family
                       logistic at the pooled-train mean rate, anchored at each
                       component's OWN r0, with the plateau L solved (brentq;
                       the net loss is monotone decreasing in L) so the curve's
                       signed loss equals c_base + CURVE_PRED_SHRINK·(ĉ −
                       c_base) (L < 1 = settles below baseline, L > 1 = above —
                       so the family reaches the NEGATIVE net losses cum_loss's
                       unclipped definition contains);
           city        a CITY-WISE baseline (decomposition-free): ONE α for the
                       whole city, predicted by a cosine-kNN over the training
                       cities' weight_normal-aggregated feature vectors and
                       their city-level α (both from the TRAIN role, own-k
                       tables — the same role convention as the component
                       transfer); the city curve is a SINGLE L = 1 logistic
                       anchored at the observed city day-0 value.  Tests whether
                       the component decomposition buys anything over predicting
                       the city's recovery from city-level features alone;
           oracle      the component's OWN UNGATED surge-model joint fit
                       (α, L, B, b) — the model family's ceiling, separating
                       family misfit from parameter-transfer error; where even
                       the ungated fit is undefined (day-0 stop, constant
                       curve) the α = 0 curve ("stays at r0") is used;
           train_mean  a COMPONENT-LEVEL baseline: the pooled-train mean α with
                       L = 1 for every component (each still anchored at its
                       own r0) — the zero-shrinkage (w = 0) special case of
                       'pred', so the pair isolates exactly what the cum_loss
                       transfer adds.
         Components with an unusable anchor (all-NaN curve, or a day-0 total
         stop r0 ≈ 0) emit r ≡ 1; their (near-)zero weight_normal makes the
         aggregation unaffected.
      3. City curves: the component curves are aggregated with the weight_normal
         shares (the exact reconstruction weights of the total curve), and the
         day-type-matched baseline of the TOTAL activity matrix scales the
         relative curves to absolute daily flow volume for the magnitude figure.

    Outputs under cross_city_curve_pred/:
      bar_cross_city_curve_mae.png    per city-event, the city-level whole-curve
                                      MAE of the 3 FORECAST lines side by side,
                                      comparing them on the quantity the forecast
                                      optimises (lower is better; the oracle's own
                                      MAE stays in curve_pred_metrics.csv)
      city_magnitude_curve_<code>.png observed vs the 3 FORECAST lines, absolute
                                      volume (the oracle is a ceiling, not a
                                      forecast, so it is not drawn here)
      component_curves_<code>.png     per-component grid: observed / oracle / pred
      curve_pred_metrics.csv          component- and city-level MAE/NRMSE/R² plus
                                      the curve-derived cum_loss, per method line
      raw_data/                       per-day city curves + the per-component
                                      α/L/B table + the plotted MAE table
                                      (everything above is recomputable)."""
    # Three feature sets, one per channel (2026-08-04 re-wiring):
    #   feature_cols — the pooled/raw channel base (income stays: it is a LEVEL
    #     predictor there); _param_prediction and the city-wise reference curve
    #     keep it.
    #   CITY_TOTAL_FEATURE_COLS — the city-total channel.  _city_score below
    #     IS STEP-6's aggr_denorm estimator, so it reads the shared module-level
    #     constants rather than re-deriving the list here (they drifted apart on
    #     2026-08-12 when only STEP 6 gained the interactions and the city r0).
    #   the RANK channel is not listed here: it reads RANK_FEATURE_COLS
    #     through rank_predict, shared with STEP 6 (unified 2026-08-28).
    feature_cols = ([f'func_{c}' for c in SF_CATEGORIES]
                    + ['mean_distance', 'median_income_combined'])

    def _merge(feats):
        # The shared city-total builder: adds the merged func_<cat> shares that
        # every channel uses, plus the interaction products and r0_city that
        # only _city_score reads.  Extra columns are inert for the channels
        # that do not name them.
        return _with_city_total_feats(feats)

    train_merged = {c: _merge(feats_by_city[c]) for c in codes}
    test_merged = {c: _merge(feats_test[c]) for c in codes}

    def _r0_city(c):
        """The unit's city-level r0 (constant within the city), the rho stage's
        second predictor after the aggregate score s."""
        return float(train_merged[c]['r0_city'].iloc[0])

    # Rank-channel training-pool context.  Under the production
    # RANK_TRAIN_SCOPE='pooled' (2026-08-31) this resolves to None and every
    # fold trains on all reference cities; under a 'cluster' revert it would
    # narrow each fold to the held city's ESTIMATED transfer cluster (see
    # rank_cluster_context for the leakage control and RANK_TRAIN_SCOPE for
    # why a plain revert no longer reproduces 2026-08-12).
    _rank_ctx = rank_cluster_context(test_merged, codes)
    if RANK_TRAIN_SCOPE == 'cluster' and _rank_ctx is None:
        print("  [curve pred] RANK_TRAIN_SCOPE=cluster but clusters CSV "
              "absent/incomplete; falling back to pooled rank training.")

    # Observed smoothed relative curves of every unit's TEST components (same
    # decomposition as feats_test, via dec_test), the unit's ungated own-fit
    # (α, L) for the oracle line, the observed TOTAL relative curve (the city
    # ground truth) and the total day-type baseline (relative -> absolute).
    curves_obs, city_gt_rel, city_base, oracle_par = {}, {}, {}, {}
    for c in codes:
        u = units[c]
        W, H = dec_test[c]
        curves_obs[c] = resilience_curves(W, u['n_nor'], u['first_day_nor'],
                                          SLOTS_ACTIVE, n_dis=u['n_dis'])
        # UNGATED own fit for the oracle line (min_fit_r2 = -inf disables the
        # quality gate; the metric columns in feats_test stay gated).
        oracle_par[c] = recovery_curve_features(
            W, u['n_nor'], u['first_day_nor'], SLOTS_ACTIVE, n_dis=u['n_dis'],
            max_rate=ALPHA_MAX_RATE, min_fit_r2=-np.inf,
            min_std=ALPHA_MIN_STD, level_bounds=LEVEL_BOUNDS,
            surge_bounds=SURGE_BOUNDS, surge_rate_bounds=SURGE_RATE_BOUNDS)
        total = u['X_all'].sum(axis=0).reshape(-1, 1)
        city_gt_rel[c] = resilience_curves(total, u['n_nor'], u['first_day_nor'],
                                           SLOTS_ACTIVE, n_dis=u['n_dis']).iloc[:, 0]
        city_base[c] = daily_baselines(total, u['n_nor'], u['first_day_nor'],
                                       SLOTS_ACTIVE, n_dis=u['n_dis']).iloc[:, 0]
    # City-level cum_loss of the TOTAL activity curve, identical to the STEP-6
    # ground truth because both integrate the same smoothed total curve.  Only
    # TRAINING units' values are ever read, by the location channel's nested
    # calibration in _city_total_prediction.
    gt_cum = {c: float(np.nansum(1.0 - city_gt_rel[c].to_numpy(dtype=float)))
              for c in codes}

    # City-wise baseline inputs (decomposition-free forecast of the whole city's
    # recovery shape): per city, the weight_normal-aggregated predictor vector +
    # the storm intensity (evac_level deliberately excluded — over a low-dim
    # cosine vector it is collinear with intensity; this mirrored the STEP-6
    # "D+city" reconstruction, which was removed 2026-08-04, so this is now the
    # only city-level aggregate in the pipeline), plus the observed city day-0
    # value that anchors the single city logistic.  TRAINING cities
    # contribute their vector and their city-level α from the TRAIN role (own-k
    # tables) — the same role convention as the component transfer — while the
    # held city enters only through its own-k vector and its day-0 anchor.
    def _city_vec(feats):
        w = feats['weight_normal'].to_numpy(dtype=float); sw = w.sum()
        w = w / sw if sw > 0 else np.full(len(w), 1.0 / len(w))
        v = [float(w @ feats[col].to_numpy(dtype=float)) for col in feature_cols]
        return np.array(v + [float(feats['hurricane_intensity'].iloc[0])])

    city_vec_tr = {c: _city_vec(train_merged[c]) for c in codes}
    city_vec_te = {c: _city_vec(test_merged[c]) for c in codes}
    city_alpha_tr, r0_city = {}, {}
    for c in codes:
        ft = train_merged[c]
        w = ft['weight_normal'].to_numpy(dtype=float)
        a = ft['recovery_alpha'].to_numpy(dtype=float)
        v = np.isfinite(a)
        city_alpha_tr[c] = (float((w[v] / w[v].sum()) @ a[v])
                            if v.any() and w[v].sum() > 0 else np.nan)
        r0_city[c] = float(city_gt_rel[c].iloc[0])

    def _city_wise_curve(held, rest, days_f):
        """City-wise baseline: ONE α for the whole city, predicted by a cosine-kNN
        over the training cities' aggregated feature vectors and their city-level
        α; the city curve is a SINGLE L = 1 logistic anchored at the observed city
        day-0 (like every forecast line, the plateau is pinned at the normal
        baseline).  Returns (curve|None, α̂); standardization uses TRAIN cities
        only (no leakage)."""
        ok = [c for c in rest if np.isfinite(city_alpha_tr[c])]
        r0c = r0_city[held]
        if len(ok) < 2 or not np.isfinite(r0c) or r0c <= 1e-6:
            return None, np.nan
        Xtr = np.vstack([city_vec_tr[c] for c in ok])
        mu_x, sd_x = Xtr.mean(axis=0), Xtr.std(axis=0); sd_x[sd_x == 0] = 1.0
        Ztr = (Xtr - mu_x) / sd_x
        zte = (city_vec_te[held] - mu_x) / sd_x

        def _unit(M):
            n = np.linalg.norm(M, axis=-1, keepdims=True)
            return M / np.where(n > 0, n, 1.0)

        sims = np.clip(_unit(Ztr) @ _unit(zte), 0.0, None)
        ssum = float(sims.sum())
        atr = np.array([city_alpha_tr[c] for c in ok])
        a_hat = float(sims @ atr / ssum) if ssum > 0 else float(atr.mean())
        curve = 1.0 / (1.0 + (1.0 / r0c - 1.0) * np.exp(-a_hat * days_f))
        return curve, a_hat

    def _param_prediction(held, rest, target, extra_pooled=()):
        """(param_hat Series over the held unit's component index, pooled-train
        mean) for one target column.  `extra_pooled` adds observed pooled
        predictors (r0 for the cum_loss transfer).  param_hat is finite
        everywhere (train-mean fallback for feature-incomplete components).
        None when no unit supplies rows for `target`."""
        feat_cols = feature_cols + list(extra_pooled)
        pooled_cols = POOLED_FEATURE_COLS + list(extra_pooled)
        tr_vals = []
        for t in rest:
            sub = train_merged[t][[target] + feat_cols
                                  + LEVEL_FEATURE_COLS].dropna()
            if len(sub) < CROSS_CITY_MIN_ROWS or sub[target].nunique() < 2:
                continue          # the engine skips this unit too (same rule)
            tr_vals.append(sub[target].to_numpy(dtype=float))
        if not tr_vals:
            return None, np.nan
        pooled = np.concatenate(tr_vals)
        mu, sd = float(pooled.mean()), float(pooled.std())
        sd = sd if sd > 0 else 1.0

        te = test_merged[held].copy()
        # Placeholder ramp: keeps every feature-complete row through the engine's
        # [target]+features dropna and its constant-target guard; the transfer
        # prediction reads train rows and test FEATURES only.
        te[target] = np.arange(len(te), dtype=float)
        fold = {held: te}
        fold.update({c: train_merged[c] for c in rest})
        _, pred, _ = cross_city_resilience(
            fold, [target], feat_cols, rank=False,
            split={'train': rest, 'test': [held]}, target_std='pooled_train',
            level_feature_cols=LEVEL_FEATURE_COLS, model='cosine_knn',
            pooled_feature_cols=pooled_cols,
            min_rows=CROSS_CITY_MIN_ROWS)
        pm = pred.get(held, {}).get(target)
        hat = pd.Series(mu, index=test_merged[held].index, dtype=float)
        if pm is not None:
            _, ypred_std, cidx = pm
            # Inverting with the pooled-TRAIN (mu, sd) recomputed above — the
            # engine standardized the target with exactly these statistics.
            hat.loc[cidx] = np.asarray(ypred_std, dtype=float) * sd + mu
        return hat, mu

    def _rank_frames(rest, held):
        """held's TEST table plus the reference cities' TRAIN tables, the frame
        dict rank_predict reads.  The two roles are the same object in
        production; they stay separate here as everywhere else in this
        function."""
        f = {held: test_merged[held]}
        f.update({c: train_merged[c] for c in rest})
        return f

    def _rank_score_prediction(held, rest, target):
        """Predicted within-city ORDERING of `target` over the held unit's
        components.  This is THE rank channel (rank_predict) -- the same call,
        the same features and the same training-pool rule the component_rank
        sweep and the pairwise heatmap run, so a change to the rank prediction
        cannot land here and nowhere else.  None when the engine returns no
        prediction (the caller falls back to the raw-path values)."""
        pool = rank_train_pool(held, rest, _rank_ctx)
        score = rank_predict(held, pool, _rank_frames(pool, held),
                             target=target, min_rows=CROSS_CITY_MIN_ROWS)
        if score is None:
            return None
        return score.reindex(test_merged[held].index)

    def _city_score(held, rest):
        """One unit's aggregate score s, through the SHARED phi stage
        (city_total_score) -- byte-for-byte the same call STEP 6's headline
        city-total figure makes, on the same CITY_TOTAL_FEATURE_COLS and the
        same predictor."""
        return city_total_score(
            held, rest, {**train_merged, held: test_merged[held]}, feats_test,
            CURVE_PRED_CITY_MODEL)

    def _city_total_prediction(held, rest):
        """Predicted CITY cum_loss in day-equivalents: the rho stage of the
        city-total estimator, shared verbatim with STEP 6 through
        _city_total_from_scores, so the city total the quantile mapping shifts
        onto and the city_total/ headline figure are the same estimator.

        The held unit's aggregate score s and every training unit's inner-LOO s
        (predictor CURVE_PRED_CITY_MODEL, the shared CITY_TOTAL_FEATURE_COLS) are
        scored on [s, r0_city, GDP] by a ridge whose fit is then variance-matched
        onto day-equivalents; the nested leave-one-out over the training units
        keeps the held unit out of its own calibration.  The variance match uses
        the SMALL city-to-city dispersion of cum_loss instead of the large per-
        component one, which is what makes an absolute city total meaningful.
        NaN when the calibration cannot be formed, and the caller then falls
        back to the raw aggregate."""
        outer = _city_score(held, rest)
        s_in, x_in, g_in = [], [], []
        for t in rest:
            v = _city_score(t, [c for c in rest if c != t])
            if np.isfinite(v):
                s_in.append(v)
                x_in.append([_r0_city(t), _msa_gdp_table()[t]])
                g_in.append(gt_cum[t])
        return _city_total_from_scores(
            outer, [_r0_city(held), _msa_gdp_table()[held]], s_in, x_in, g_in)

    def _quantile_mapped_chat(chat_raw, score, obs, wn, rest, mu_a, c_city,
                              scale_value=1.0):
        """Component cum_loss predictions assembled by QUANTILE MAPPING, the
        comonotone assignment, i.e. the optimal-transport map on the line.
        Four ingredients, each contributing what it transfers best:

          ordering  the rank-path score (who loses more, within the city);
          shape     the WASSERSTEIN BARYCENTRE of the training units' centred
                    cum_loss spectra — each unit's own quantile function is
                    evaluated at the (k−0.5)/n positions and those functions are
                    AVERAGED; components ranked k of n get that average.  This
                    replaces the raw kNN values' spread, which a kernel smoother
                    compresses toward the mean (it cannot extrapolate).
                    Averaging quantile functions, not pooling the samples, is
                    what makes the template a spectrum a CITY could have.  A
                    pooled sample is the marginal over components, and a mixture
                    of differently-spread distributions is wider than any of its
                    parts: pooling reports the ROOT-MEAN-SQUARE of the units'
                    spreads where the barycentre reports their MEAN (5.46 vs
                    5.11 over the 13 units; RMS >= mean by Jensen, with equality
                    only if every unit were equally spread, and the units differ
                    4.8-fold).  Pooling also weights a unit by its COMPONENT
                    COUNT (12/81 against 5/81), i.e. by how finely the NMF rank
                    happened to split it, whereas the barycentre weights the 13
                    units equally, matching the level the estimand lives at.  And
                    it is the mean under the very geometry this branch is judged
                    and consumed in: the assignment below is the 1-D optimal-
                    transport map and the error decomposition is W1, so the
                    Frechet mean under W1 is the self-consistent average.  The
                    barycentre is also invariant to the centring above (averaging
                    quantile functions turns a per-unit mean into one constant,
                    which the location ingredient absorbs), so the between-city
                    level can no longer leak into the shape by construction;
                    _centred is kept only so the template reads as centred.
          spread    those quantiles are SCALED by the ratio of the held unit's
                    backbone-loss spread to the training units' mean backbone-
                    loss spread, _backbone_spread), then multiplied by the
                    fitted ML-B2inc increment exp(γ'PC) (see the MLB2_* block;
                    1.0 when its ingredients are missing).  The true within-city cum_loss
                    spread differs almost twofold across units and correlates
                    at rank 0.90 with this observable proxy, so a single pooled
                    spread over- or under-disperses most cities.  The RATIO
                    form cancels the unknown proxy-to-truth constant, leaving
                    zero fitted parameters;
          location  an additive shift so the weight_normal aggregate equals
                    `c_city`, the D+aggr+denorm predicted city cum_loss
                    (_city_total_prediction).  Some city-total anchor is
                    essential: skipping the shift is WORSE than the train-mean
                    baseline, because the training world's loss LEVEL does not
                    fit every city even though its shape does.  The raw
                    channel's own aggregate is the fallback when the D+aggr+denorm
                    calibration cannot be formed.

        Error decomposition (exact, 1-D OT): with a correct ordering the mean
        absolute error equals W1(constructed margin, true margin); ordering
        swaps add the local quantile gaps.  Because the shift re-centres the
        whole vector afterwards, scaling needs no centring convention of its
        own.  Components without a usable anchor or rank score keep their raw
        values up to that common shift; a dead rank path returns the raw
        predictions unchanged (the pre-QM production line).  Returns the
        predictions and the applied spread scale (NaN when none was applied)."""
        out = chat_raw.to_numpy(dtype=float).copy()

        def _centred(v):
            """One training unit's component cum_loss on its own mean, so a
            spectrum describes only WITHIN-city inequality (see `shape` above)."""
            return v - v.mean() if len(v) else v

        spectra = [_centred(train_merged[c]['cum_loss'].dropna().to_numpy(dtype=float))
                   for c in rest]
        spectra = [v for v in spectra if len(v)]
        r0 = np.array([obs.iloc[0, j] if np.isfinite(obs.iloc[:, j]).any()
                       else np.nan for j in range(obs.shape[1])], dtype=float)
        days = obs.index.to_numpy(dtype=float)
        scale = np.nan
        if score is not None and spectra:
            sc = score.to_numpy(dtype=float)
            ok = np.isfinite(sc) & np.isfinite(r0) & (r0 > 1e-6)
            if ok.sum() >= 2:
                # The scale comes from the shared spread_scale (see the
                # caller); this function only APPLIES it.
                scale = float(scale_value)
                idx = np.where(ok)[0][np.argsort(sc[ok], kind='stable')]
                pos = (np.arange(len(idx)) + 0.5) / len(idx)
                # Wasserstein barycentre: average the training units' QUANTILE
                # FUNCTIONS at the plotting positions, not their samples.
                out[idx] = scale * np.mean(
                    [np.quantile(v, pos) for v in spectra], axis=0)
                if np.isfinite(out).all():
                    loc = (float(c_city) if np.isfinite(c_city)
                           else float(wn @ chat_raw.to_numpy(dtype=float)))
                    out = out + (loc - float(wn @ out))
        return pd.Series(out, index=chat_raw.index), scale

    def _curves_from_params(obs, alpha_vec, level_vec, surge_vec=None,
                            surge_rate_vec=None):
        """[days × k] curve frame: the day-0-anchored surge-plus-relaxation
        model per component.  α = NaN -> the α = 0 curve (stays at r0);
        L = NaN -> 1; B or b missing/NaN -> pulse omitted; unusable anchor
        (r0 ≈ 0 / all-NaN) -> r ≡ 1.  The forecast passes no surge vectors
        (the L = 1, B = 0 monotone logistic slice); the oracle passes all four."""
        days = obs.index.to_numpy(dtype=float)
        out = np.ones((len(days), obs.shape[1]))
        for j in range(obs.shape[1]):
            col = obs.iloc[:, j].to_numpy(dtype=float)
            r0 = col[0] if np.isfinite(col).any() else np.nan
            if not np.isfinite(r0) or r0 <= 1e-6:
                continue
            a = float(alpha_vec[j]) if np.isfinite(alpha_vec[j]) else 0.0
            L = float(level_vec[j]) if np.isfinite(level_vec[j]) else 1.0
            out[:, j] = L / (1.0 + (L / r0 - 1.0) * np.exp(-a * days))
            if surge_vec is not None and surge_rate_vec is not None:
                B = surge_vec[j]
                b = surge_rate_vec[j]
                if np.isfinite(B) and np.isfinite(b):
                    out[:, j] = out[:, j] + float(B) * days * np.exp(-float(b) * days)
        return pd.DataFrame(out, index=obs.index, columns=obs.columns)

    def _plateau_inversion_curves(obs, mu_a, c_hat):
        """[days × k] PLATEAU-INVERSION forecast curves.

        Each component's curve IS the fitted family's logistic (B = 0) with the
        plateau L SOLVED so the curve's net signed loss lands at the shrunk
        target between the L = 1 backbone's own loss and the component's
        cross-city predicted cum_loss `c_hat`:

            r(t) = L/(1 + (L/r0 − 1)e^(−ᾱ·t))
            Σ_d (1 − r(d)) = c_base + CURVE_PRED_SHRINK·(ĉ − c_base)

        The rate stays at the pooled-train mean ᾱ (it does not transfer); the
        ONE quantity carrying cross-city information is cum_loss, the only
        target with real feature signal.  The loss is monotone DECREASING in L
        (brentq over LEVEL_BOUNDS; an out-of-range target lands on the bound),
        and near-linear in L (dL/dĉ ≈ 0.1 over the 15-day window), so unlike
        the rate inversion (dα/dĉ ∝ 1/ĉ²) prediction error is ATTENUATED, not
        amplified.  L > 1 settles above baseline, reaching the NEGATIVE net
        losses cum_loss's unclipped definition contains, while r(0) = r0 keeps
        the observed anchor.  CURVE_PRED_SHRINK = 0 gives L = 1 everywhere —
        the train-mean forecast — so that line is nested.  The solved L is a
        LOSS-MATCHING device: it does NOT recover the component's own fitted
        plateau (spearman ≈ 0.16 pooled) — the skill lives in the city
        aggregate, not in per-component attribution."""
        days = obs.index.to_numpy(dtype=float)
        out = np.ones((len(days), obs.shape[1]))
        level_solved = np.full(obs.shape[1], np.nan)
        for j in range(obs.shape[1]):
            col = obs.iloc[:, j].to_numpy(dtype=float)
            r0 = col[0] if np.isfinite(col).any() else np.nan
            if not np.isfinite(r0) or r0 <= 1e-6:
                continue
            base = 1.0 / (1.0 + (1.0 / r0 - 1.0) * np.exp(-mu_a * days))
            if not np.isfinite(c_hat[j]):
                out[:, j] = base
                level_solved[j] = 1.0
                continue
            c_base = float(np.sum(1.0 - base))
            target = c_base + CURVE_PRED_SHRINK * (float(c_hat[j]) - c_base)

            def _loss_of(L, _r0=r0):
                return float(np.sum(
                    1.0 - L / (1.0 + (L / _r0 - 1.0) * np.exp(-mu_a * days))))

            if target >= _loss_of(LEVEL_BOUNDS[0]):
                L = LEVEL_BOUNDS[0]
            elif target <= _loss_of(LEVEL_BOUNDS[1]):
                L = LEVEL_BOUNDS[1]
            else:
                L = float(brentq(lambda v: _loss_of(v) - target,
                                 LEVEL_BOUNDS[0], LEVEL_BOUNDS[1], xtol=1e-9))
            level_solved[j] = L
            out[:, j] = L / (1.0 + (L / r0 - 1.0) * np.exp(-mu_a * days))
        return (pd.DataFrame(out, index=obs.index, columns=obs.columns),
                level_solved)

    def _nosurge_backbone_mean(rest):
        """Pooled-train mean of the NO-SURGE rate (recovery_alpha_nosurge), the
        same per-unit row filter as _param_prediction's mu (feature-complete
        rows, >= CROSS_CITY_MIN_ROWS, non-constant).  NaN when no unit
        qualifies (caller keeps the surge backbone)."""
        vals = []
        for t in rest:
            sub = train_merged[t][['recovery_alpha_nosurge'] + feature_cols
                                  + LEVEL_FEATURE_COLS].dropna()
            if (len(sub) < CROSS_CITY_MIN_ROWS
                    or sub['recovery_alpha_nosurge'].nunique() < 2):
                continue
            vals.append(sub['recovery_alpha_nosurge'].to_numpy(dtype=float))
        return float(np.concatenate(vals).mean()) if vals else np.nan

    def _alphaL_training_line(rest):
        """Pooled OLS L = a0 + b·log10(α) over the TRAINING units' UNGATED
        (α, L) fits (oracle_par), bound-pinned fits INCLUDED — dropping them
        rests the line on a handful of points and can flip its sign at
        boundary-heavy windows.  ONE pooled line per fold (per-city lines flip
        direction across units and are unusable as a cross-city constraint).
        (1, 0) — the flat L = 1 line — when fewer than 3 usable points."""
        log_a, lv = [], []
        for t in rest:
            a = oracle_par[t]['recovery_alpha'].to_numpy(dtype=float)
            L = oracle_par[t]['recovery_level'].to_numpy(dtype=float)
            m = np.isfinite(a) & np.isfinite(L)
            log_a += list(np.log10(np.clip(a[m], 1e-3, None)))
            lv += list(L[m])
        if len(log_a) < 3 or float(np.std(log_a)) < 1e-6:
            return 1.0, 0.0
        coef = np.polyfit(log_a, lv, 1)
        return float(coef[1]), float(coef[0])

    def _joint_alphaL_curves(obs, mu_a, c_hat, line_a0, line_b, lam):
        """[days × k] JOINT (α, L) forecast curves: per component, least-squares
        over the residual pair

            [ Σ_d (1 − r(d; α, L)) − target ,  √λ·(L − line(α)) ]

        with line(α) = clip(a0 + b·log10 α, LEVEL_BOUNDS), α ∈ [0,
        ALPHA_MAX_RATE], L ∈ LEVEL_BOUNDS, started at the backbone (ᾱ, 1);
        the same shrunk target as the solve_L path.  λ = 0 leaves the
        under-determined cum_loss-only solve — its free direction cancels
        prediction error on small samples (see the CURVE_PRED_SOLVER comment)
        — and λ > 0 pins the remaining DOF to the training α-L relationship.
        Missing ĉ falls back to the backbone curve like the solve_L path.
        Returns (curves, level_solved, alpha_solved)."""
        days = obs.index.to_numpy(dtype=float)
        out = np.ones((len(days), obs.shape[1]))
        level_solved = np.full(obs.shape[1], np.nan)
        alpha_solved = np.full(obs.shape[1], np.nan)
        for j in range(obs.shape[1]):
            col = obs.iloc[:, j].to_numpy(dtype=float)
            r0 = col[0] if np.isfinite(col).any() else np.nan
            if not np.isfinite(r0) or r0 <= 1e-6:
                continue
            base = 1.0 / (1.0 + (1.0 / r0 - 1.0) * np.exp(-mu_a * days))
            if not np.isfinite(c_hat[j]):
                out[:, j] = base
                level_solved[j] = 1.0
                alpha_solved[j] = mu_a
                continue
            c_base = float(np.sum(1.0 - base))
            target = c_base + CURVE_PRED_SHRINK * (float(c_hat[j]) - c_base)

            def _resid(p, _r0=r0, _t=target):
                a = min(max(p[0], 0.0), ALPHA_MAX_RATE)
                L = min(max(p[1], LEVEL_BOUNDS[0]), LEVEL_BOUNDS[1])
                loss = float(np.sum(
                    1.0 - L / (1.0 + (L / _r0 - 1.0) * np.exp(-a * days))))
                line = float(np.clip(line_a0 + line_b * np.log10(max(a, 1e-3)),
                                     LEVEL_BOUNDS[0], LEVEL_BOUNDS[1]))
                return [loss - _t, np.sqrt(lam) * (L - line)]

            res = least_squares(_resid, [mu_a, 1.0],
                                bounds=([0.0, LEVEL_BOUNDS[0]],
                                        [ALPHA_MAX_RATE, LEVEL_BOUNDS[1]]),
                                max_nfev=4000)
            a_j, L_j = float(res.x[0]), float(res.x[1])
            alpha_solved[j] = a_j
            level_solved[j] = L_j
            out[:, j] = L_j / (1.0 + (L_j / r0 - 1.0) * np.exp(-a_j * days))
        return (pd.DataFrame(out, index=obs.index, columns=obs.columns),
                level_solved, alpha_solved)

    os.makedirs(os.path.join(OUTPUT_CURVE_PRED, 'raw_data'), exist_ok=True)
    metric_rows, par_rows, curve_rows = [], [], []
    city_curve_page = {}          # code -> (days, gt, lines) for the one-page grid
    comp_curve_rows = []          # long-form component curves -> raw_data/
    for held in codes:
        rest = [c for c in codes if c != held]
        alpha_hat, mu_a = _param_prediction(held, rest, 'recovery_alpha')
        if alpha_hat is None:
            print(f"  [curve_pred] {held}: no usable α training rows; skipping.")
            continue
        if CURVE_ALPHA_BACKBONE == 'no_surge':
            # Ablation backbone: the pulse-free rate replaces the clean rate
            # EVERYWHERE mu_a is read (backbone curves, train_mean line, the
            # QM spread proxy) so the ablation is a single consistent swap.
            mu_ns = _nosurge_backbone_mean(rest)
            if np.isfinite(mu_ns):
                mu_a = mu_ns
        # Component cum_loss prediction, TWO kNN channels combined by quantile
        # mapping: the RAW pooled_train path supplies the city TOTAL, the RANK
        # within_unit path supplies the within-city ORDERING, the pooled-train
        # cum_loss quantiles supply the SPREAD (see _quantile_mapped_chat).
        cum_hat_raw, _ = _param_prediction(held, rest, 'cum_loss', extra_pooled=('r0',))
        rank_score = _rank_score_prediction(held, rest, 'cum_loss')
        obs = curves_obs[held]
        fit_a = feats_test[held]['recovery_alpha']           # gated (the metric)
        fit_L = feats_test[held]['recovery_level']
        orc = oracle_par[held]                               # ungated (ceiling)
        k = obs.shape[1]
        # The component-wise line ('pred') is the PLATEAU-INVERSION forecast —
        # the family logistic at the mean rate with L solved from the predicted
        # cum_loss, the ONE target with real cross-city feature signal (the
        # recovery rate has none: its within-city Ridge R² is negative in all
        # 5 units, against up to 0.70 for cum_loss; kNN-predicting L directly
        # was the catastrophic 2026-07-13 free-L failure — the integral
        # constraint is the transferable route to it).  'train_mean' is that
        # forecast's zero-shrinkage special case (L = 1 everywhere).  The
        # ORACLE keeps the full own fit: it describes what the model family
        # can express, not what is forecastable.
        ones = np.ones(k)
        w = feats_test[held]['weight_normal'].to_numpy(dtype=float)
        wsum = w.sum()
        wn = w / wsum if wsum > 0 else np.full(len(w), 1.0 / len(w))
        city_total_hat = _city_total_prediction(held, rest)
        # ML-B2inc spread increment, fitted on the training units only (see
        # the MLB2_* block).  The SHARED spread definition: the same call the
        # centred_distribution/ figure makes, on SPREAD_BACKBONE_DAYS and at
        # the backbone rate ML-B2inc was fitted at (pooled_backbone_alpha) --
        # NOT this fold's curve alpha-bar, which CURVE_ALPHA_BACKBONE may
        # replace and which would move the tuned increment off its validated
        # coordinates.
        _sp_merged = {**train_merged, held: test_merged[held]}
        _sp_mu_a = pooled_backbone_alpha(train_merged, rest)
        _sp_base, _sp_mult = spread_scale(held, rest, _sp_merged, _sp_mu_a)
        cum_hat, spread_scale_applied = _quantile_mapped_chat(
            cum_hat_raw, rank_score, obs, wn, rest, mu_a, city_total_hat,
            scale_value=_sp_base * _sp_mult)
        if CURVE_PRED_SOLVER == 'joint_alphaL':
            line_a0, line_b = _alphaL_training_line(rest)
            pred_curves, L_solved, alpha_solved = _joint_alphaL_curves(
                obs, mu_a, cum_hat.to_numpy(dtype=float),
                line_a0, line_b, CURVE_JOINT_LAMBDA)
        else:
            pred_curves, L_solved = _plateau_inversion_curves(
                obs, mu_a, cum_hat.to_numpy(dtype=float))
            alpha_solved = None
        lines = {
            'pred':       pred_curves,
            'oracle':     _curves_from_params(obs,
                                              orc['recovery_alpha'].to_numpy(dtype=float),
                                              orc['recovery_level'].to_numpy(dtype=float),
                                              orc['surge_strength'].to_numpy(dtype=float),
                                              orc['surge_rate'].to_numpy(dtype=float)),
            'train_mean': _curves_from_params(obs, np.full(k, mu_a), ones),
        }

        gt_rel = city_gt_rel[held].to_numpy(dtype=float)
        base = city_base[held].to_numpy(dtype=float)
        days = obs.index.to_numpy()
        city_rel = {lab: dfc.to_numpy(dtype=float) @ wn for lab, dfc in lines.items()}
        # City-wise baseline: a single L = 1 city logistic from the city-level
        # predicted α; adds the 'city' entry (a plain vector, no component curves).
        city_curve, ca_hat = _city_wise_curve(held, rest, days.astype(float))
        if city_curve is not None:
            city_rel['city'] = city_curve

        # Metrics: component level pooled over all component-days with a finite
        # observed value (only the per-component methods); city level on the
        # aggregated curve (paper-style NRMSE = RMSE / std of the truth; R² only
        # at city level, where the curve has real variance).
        obs_arr = obs.to_numpy(dtype=float)
        sd_comp = float(np.nanstd(obs_arr))
        sd_city = float(np.nanstd(gt_rel))
        cum_loss_gt = float(np.nansum(1.0 - gt_rel))
        for lab in city_rel:
            cr = city_rel[lab]
            cdiff = cr - gt_rel
            ok = np.isfinite(gt_rel) & np.isfinite(cr)
            ss_res = float(np.sum((cr[ok] - gt_rel[ok]) ** 2))
            ss_tot = float(np.sum((gt_rel[ok] - gt_rel[ok].mean()) ** 2))
            if lab in lines:                          # per-component methods only
                diff = lines[lab].to_numpy(dtype=float) - obs_arr
                mae_comp = float(np.nanmean(np.abs(diff)))
                nrmse_comp = (float(np.sqrt(np.nanmean(diff ** 2))) / sd_comp
                              if sd_comp > 0 else np.nan)
            else:                                     # 'city' has no component curve
                mae_comp = nrmse_comp = np.nan
            metric_rows.append({
                'code': held, 'method': lab,
                'mae_component': mae_comp, 'nrmse_component': nrmse_comp,
                'mae_city': float(np.nanmean(np.abs(cdiff))),
                'nrmse_city': (float(np.sqrt(np.nanmean(cdiff ** 2))) / sd_city
                               if sd_city > 0 else np.nan),
                'r2_city': (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                'cum_loss_from_curve': float(np.nansum(1.0 - cr)),
                'cum_loss_gt': cum_loss_gt,
            })

        # Fitted per-component α and L: the gated metric values, kept for the raw
        # parameter table below.  `val` also feeds the per-unit progress line.
        a_arr = fit_a.to_numpy(dtype=float)
        L_arr = fit_L.to_numpy(dtype=float)
        val = np.isfinite(a_arr)

        # Raw per-day city curves (relative + absolute volume), GT first.
        for i, d in enumerate(days):
            curve_rows.append({'code': held, 'day': int(d), 'method': 'ground_truth',
                               'r_rel': gt_rel[i], 'magnitude': gt_rel[i] * base[i]})
            for lab in city_rel:
                curve_rows.append({'code': held, 'day': int(d), 'method': lab,
                                   'r_rel': city_rel[lab][i],
                                   'magnitude': city_rel[lab][i] * base[i]})
        # Raw per-component parameter table.  fit = the gated metric; fit_ungated
        # = the oracle's family best fit; cum_loss_pred = the quantile-mapped
        # prediction the forecast consumes; cum_loss_pred_raw = the raw-path
        # kNN value (supplies only the city total); rank_score = the ordering
        # channel (only its order matters); level_solved = the plateau L the
        # inversion derives — a loss-matching device, NOT an estimate of
        # level_fit (alpha is NOT predicted: the forecast uses the pooled-train
        # mean rate).
        for j in range(k):
            row = {'code': held, 'component': int(obs.columns[j]),
                   'cum_loss_fit': float(feats_test[held]['cum_loss'].iloc[j]),
                   'cum_loss_pred': float(cum_hat.iloc[j]),
                   'cum_loss_pred_raw': float(cum_hat_raw.iloc[j]),
                   'rank_score': (float(rank_score.iloc[j])
                                  if rank_score is not None else np.nan),
                   'spread_scale': spread_scale_applied,
                   'city_total_pred': city_total_hat,
                   'level_solved': float(L_solved[j]),
                   'alpha_train_mean': mu_a,
                   'alpha_fit': a_arr[j],
                   'alpha_fit_ungated': float(orc['recovery_alpha'].iloc[j]),
                   'level_fit': L_arr[j],
                   'level_fit_ungated': float(orc['recovery_level'].iloc[j]),
                   'surge_fit': float(feats_test[held]['surge_strength'].iloc[j]),
                   'surge_fit_ungated': float(orc['surge_strength'].iloc[j]),
                   'surge_rate_fit_ungated': float(orc['surge_rate'].iloc[j]),
                   'weight_normal': w[j]}
            if alpha_solved is not None:      # joint solver only: the solved rate
                row['alpha_solved'] = float(alpha_solved[j])
            par_rows.append(row)

        # Figures: absolute-volume city curve overlay (the FORECAST lines only,
        # in label order, see _CITY_FIGURE_METHODS) + per-component grid (the
        # grid keeps observed / oracle / component-pred, because at component
        # level the oracle is the point of comparison; more lines per small
        # panel would be unreadable).
        city_lines = {_CURVE_METHOD_LABELS[lab]: city_rel[lab] * base
                      for lab in _CITY_FIGURE_METHODS if lab in city_rel}
        # LEGACY copy: every forecast line, original small-format styling.
        # Kept so the earlier figure stays reproducible, moved out of the way.
        vis_curves_city_pred(
            days, gt_rel * base, city_lines,
            ylabel='daily mobility (flow volume per day)',
            title=f'{held}: city mobility over the disaster window',
            save_path=os.path.join(OUTPUT_CURVE_PRED,
                                   'city_magnitude_curve_legacy',
                                   f'city_magnitude_curve_{held}.png'))
        # CURRENT figure: slide format, and the city-wise kNN line dropped --
        # it is the weakest of the three and crowds the panel.
        # No per-unit slide file any more: the combined page below carries all
        # 13, so a separate copy per city was redundant.
        slide_lines = {k: v for k, v in city_lines.items()
                       if k != _CURVE_METHOD_LABELS['city']}
        city_curve_page[held] = (days, gt_rel * base, slide_lines)
        # Persist the component-level curves: the city ones were already in
        # raw_data/, these were not, which forced a full pipeline run for any
        # restyle of the component figure.
        # Underscore-prefixed loop names on purpose: a bare `val` here shadows
        # the enclosing `val` that the fold's summary line reads afterwards.
        for _lab, _dfm in (('observed', obs), ('oracle', lines['oracle']),
                           ('pred', lines['pred']),
                           ('train_mean', lines['train_mean'])):
            for _j, _comp in enumerate(obs.columns):
                for _d, _v in zip(obs.index.to_numpy(),
                                  np.asarray(_dfm.iloc[:, _j], dtype=float)):
                    comp_curve_rows.append(
                        {'code': held, 'component': _comp, 'day': _d,
                         'method': _lab, 'value': _v,
                         'weight_normal': float(wn[_j])})
        vis_component_curves_grid(
            obs, {_CURVE_METHOD_LABELS['oracle']: lines['oracle'],
                  _CURVE_METHOD_LABELS['pred']: lines['pred'],
                  _CURVE_METHOD_LABELS['train_mean']: lines['train_mean']},
            title=f'{held}: component relative curves (observed vs modelled)',
            save_path=os.path.join(OUTPUT_CURVE_PRED,
                                   f'component_curves_{held}.png'),
            weights=wn)
        if CURVE_OD_MAPS:
            u = units[held]
            W_t, H_s = dec_test[held]
            # r̂(d)·b(d) turns the RELATIVE forecast back into daily component
            # magnitudes (b = day-type-matched normal baseline, observed
            # pre-landfall); @H spreads them over OD pairs — X ≈ W·H at day
            # grain.  A NaN baseline (component inactive pre-landfall) zeroes
            # that component's contribution rather than poisoning the sum.
            base_comp = daily_baselines(W_t, u['n_nor'], u['first_day_nor'],
                                        SLOTS_ACTIVE, n_dis=u['n_dis'])
            m_hat = (pred_curves.to_numpy(dtype=float)
                     * base_comp.to_numpy(dtype=float))
            pred_od = np.nan_to_num(m_hat) @ H_s
            X_c = u['X_all']
            spd = SLOTS_ACTIVE
            nd_all = X_c.shape[1] // spd
            gt_od = (X_c[:, :nd_all * spd]
                     .reshape(X_c.shape[0], nd_all, spd).sum(axis=2)
                     [:, u['n_dis'] // spd:].T)
            n_d = min(pred_od.shape[0], gt_od.shape[0])
            cent = {g.aggr_id: (g.centroid.x, g.centroid.y)
                    for g in u['gdf'].itertuples()}
            coords = np.array([cent.get(o, (np.nan, np.nan))
                               + cent.get(t, (np.nan, np.nan))
                               for o, t in u['mapping']], dtype=float)
            ok_pair = np.isfinite(coords).all(axis=1)
            frames = {}
            for name, M in (('Prediction', pred_od[:n_d]),
                            ('Ground truth', gt_od[:n_d]),
                            ('Difference', pred_od[:n_d] - gt_od[:n_d])):
                per_day = []
                for d in range(n_d):
                    v = np.where(ok_pair, M[d], 0.0)
                    keep = np.argsort(-np.abs(v))[:OD_MAP_TOP_ARCS]
                    per_day.append(
                        [[round(coords[i, 0], 4), round(coords[i, 1], 4),
                          round(coords[i, 2], 4), round(coords[i, 3], 4),
                          round(float(v[i]), 1)]
                         for i in keep if abs(v[i]) > 1e-9])
                frames[name] = per_day
            vis_od_flow_slider_html(
                frames, [int(d) for d in obs.index[:n_d]],
                os.path.join(OUTPUT_OD_PRED, held, 'od_flow_pred_vs_gt.html'),
                f'{held}: daily OD flows — forecast vs observed',
                note=('H spatial patterns are the test decomposition&#39;s own '
                      '(descriptive); the time dimension is the STEP-7 '
                      'forecast.'))
        variant = ('' if (CURVE_PRED_SOLVER == 'solve_L'
                          and CURVE_ALPHA_BACKBONE == 'surge')
                   else f" [{CURVE_PRED_SOLVER}"
                        + (f" λ={CURVE_JOINT_LAMBDA}"
                           if CURVE_PRED_SOLVER == 'joint_alphaL' else '')
                        + (f", {CURVE_ALPHA_BACKBONE}"
                           if CURVE_ALPHA_BACKBONE != 'surge' else '') + ']')
        print(f"  [curve_pred] {held}: backbone ᾱ {mu_a:.2f}; "
              f"city total {city_total_hat:.2f}; "
              f"spread ×{spread_scale_applied:.2f}; "
              f"solved L [{np.nanmin(L_solved):.2f},{np.nanmax(L_solved):.2f}]; "
              f"city-α̂ {ca_hat:.2f} (fitted {int(val.sum())}/{len(val)}){variant}")

    if not metric_rows:
        print("  [curve_pred] nothing to evaluate; skipping outputs.")
        return
    raw_dir = os.path.join(OUTPUT_CURVE_PRED, 'raw_data')
    pd.DataFrame(metric_rows).to_csv(
        os.path.join(OUTPUT_CURVE_PRED, 'curve_pred_metrics.csv'), index=False)
    pd.DataFrame(curve_rows).to_csv(
        os.path.join(raw_dir, 'city_curves_by_method.csv'), index=False)
    par_df = pd.DataFrame(par_rows)
    par_df.to_csv(os.path.join(raw_dir, 'component_params_gt_vs_pred.csv'),
                  index=False)
    # The three publication figures behind the forecast's mechanism, drawn from
    # that same table: the rank channel on its own, what the quantile mapping
    # makes of it, and the rate/level relation the plateau inversion exploits.
    if not par_df.empty:
        labels = {c['code']: c['label'] for c in CITY_EVENTS}
        cnames = {c: f"{labels.get(c, c)} ({c.split('_', 1)[-1]})"
                  for c in par_df['code'].drop_duplicates()}
        # rank_pred_vs_true is NOT drawn here any more: the ordering it shows
        # is the shared rank channel's, so it is drawn once, by
        # analysis_rank_channel, into component_rank/ (moved 2026-08-28).
        vis_rank_to_cumloss_qm(
            par_df, save_path=os.path.join(OUTPUT_CURVE_PRED,
                                           'rank_to_cumloss_qm.png'))
        # Plain predicted-vs-observed scatter; the rank-ordered figure above is
        # kept as it was, the two answer different questions.
        vis_qm_pred_vs_obs(
            par_df, names=cnames,
            save_path=os.path.join(OUTPUT_CURVE_PRED,
                                   'rank_to_cumloss_scatter.png'))
    # Accuracy bar: per city-event, the CITY-LEVEL curve error of every method line,
    # so the four lines are compared on the quantity the analysis actually optimises
    # (the whole-curve MAE) rather than on any single fitted parameter.  Methods are
    # the forecast lines of _CITY_FIGURE_METHODS, in that order; a method missing for
    # a unit leaves a gap.  The oracle's own MAE stays in curve_pred_metrics.csv.
    if comp_curve_rows:
        pd.DataFrame(comp_curve_rows).to_csv(
            os.path.join(raw_dir, 'component_curves_by_method.csv'),
            index=False)
    if city_curve_page:
        labels = {c['code']: c['label'] for c in CITY_EVENTS}
        vis_city_curves_grid(
            city_curve_page,
            names={c: f"{labels.get(c, c)} ({c.split('_', 1)[-1]})"
                   for c in city_curve_page},
            save_path=os.path.join(OUTPUT_CURVE_PRED,
                                   'city_magnitude_curve_all.png'))

    mae_df = (pd.DataFrame(metric_rows)
              .pivot(index='code', columns='method', values='mae_city')
              .reindex(index=[c for c in codes if c in {r['code'] for r in metric_rows}],
                       columns=[m for m in _CITY_FIGURE_METHODS if m in
                                {r['method'] for r in metric_rows}])
              .rename(columns=_CURVE_METHOD_LABELS))
    mae_df.to_csv(os.path.join(raw_dir, 'city_curve_mae.csv'))
    vis_bar_curve_mae(
        mae_df,
        ylabel='city-curve MAE (fraction of the normal baseline)',
        title='Whole-curve prediction error per city-event, by method '
              '(lower is better)',
        save_path=os.path.join(OUTPUT_CURVE_PRED, 'bar_cross_city_curve_mae.png'))
    print(f"  [curve_pred] -> {OUTPUT_CURVE_PRED} "
          f"({len({r['code'] for r in metric_rows})} city-events)")


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_step_7(cc_train, cc_test, units, all_codes, dec_test):
    """STEP 7: cross-city curve prediction + the OD maps."""
    print("\n── Cross-city curve prediction (clean-rate forecast, surge-model fit) ──")
    analysis_cross_city_curve_pred(cc_train, cc_test, units, all_codes, dec_test)


def main():
    # ── STEP 1 — How many components does each unit support? ───────────────────
    # Runs over EVERY registry unit; its verdict is what EXCLUDED_CODES encodes.
    print("\n── Rank cross-validation (all registry units) ──")
    k_min_by_code = analysis_rank_cv(CITY_EVENTS)

    # ── STEP 2 — Load the retained units and classify land use once ────────────
    # EXCLUDED_CODES drop out here, so nothing downstream — not the pooled
    # land-use classification, not the cross-city steps — sees them.
    active = [c for c in CITY_EVENTS if c['code'] not in EXCLUDED_CODES]
    if K_POLICY == 'rank_cv_min':
        # TRIAL: shallow-copied registry entries with k overridden to the CV
        # minimum — the registry itself stays untouched, so 'manual' restores
        # the production configuration with no other change.
        dropped = sorted(c['code'] for c in active
                         if k_min_by_code[c['code']] < K_MIN_TRIAL_FLOOR)
        active = [dict(c, n_behaviors=int(k_min_by_code[c['code']]))
                  for c in active
                  if k_min_by_code[c['code']] >= K_MIN_TRIAL_FLOOR]
        print(f"\n── K_POLICY=rank_cv_min: k = CV minimum; dropped "
              f"{len(dropped)} unit(s) with k_min < {K_MIN_TRIAL_FLOOR}: "
              f"{', '.join(dropped)} ──")
    print(f"\n── {len(active)} of {len(CITY_EVENTS)} units retained ──")

    # Load every unit's geometry first, because the global land-use
    # classification below pools over ALL units' block groups.
    units = {}   # code -> everything the later steps need, filled incrementally
    for cfg in active:
        label, code = cfg['label'], cfg['code']
        units[code] = dict(
            label=label, key=cfg['key'], tag='_' + code, cfg=cfg,
            gdf=load_city_geo(label, AGG_LEVEL, cfg['geo']),
            first_day_nor=cfg['first_day_normal'], first_day_dis=cfg['first_day_disaster'],
            ctx_lambda=(cfg['lambda_ctx'] if cfg['context_aware'] else None))

    # The pipeline's ONE land-use classification (global TF-IWF), reused by
    # the context-aware solver, the within-city functional block, the cross-city
    # feature tables and the baseline's city-wide POI shares.
    print("\n── Global land-use classification (pooled TF-IWF) ──")
    global_iwf, landuse_by_code, cc_lookups = _global_landuse_classification(units)

    # ── STEP 2 — Decompose every unit ───────────────────────────────────────────
    # Trimming keeps the FIRST analysis_days days,
    # which drops the late recovery tail and leaves landfall + 14 recovery days at
    # the sequence end for the trailing window taken in build_city_matrices.
    _ALL_SEGMENTS = {'normal', 'buffer', 'disaster'}
    for code, u in units.items():
        cfg, label = u['cfg'], u['label']
        print()
        print(f"── {label} [{code}] ──")
        graphs = load_graphs_trimmed(cfg['graph'], cfg['analysis_days'],
                                     SLOT_PER_DAY, label=label)
        X_all, n_nor, mapping = build_city_matrices(
            graphs, cfg['window'], cfg['buffer'] + cfg['disaster'], cfg['filter_factor'])
        n_dis = n_nor + cfg['buffer'] * SLOTS_ACTIVE
        # All three fit segments -> fit_time_cols=None (exact original full-window fit);
        # a subset fits the basis on those columns and projects the full window onto it.
        fit_time_cols = (None if set(cfg['fit_segments']) == _ALL_SEGMENTS
                         else select_segment_columns(cfg['fit_segments'],
                                                     n_nor, n_dis, X_all.shape[1]))
        print(f"── NMF: {label} [{code}] ──")
        if cfg['context_aware']:
            W, H, weights = decompose_city_context(
                X_all, cfg['n_behaviors'], mapping, landuse_by_code[code],
                lambda_ctx=cfg['lambda_ctx'], feature_mode=FLOW_FEATURE_MODE,
                l1_reg=cfg['l1_reg'], fit_time_cols=fit_time_cols)
        else:
            W, H, weights = decompose_city(
                X_all, cfg['n_behaviors'], l1_reg=cfg['l1_reg'], fit_time_cols=fit_time_cols)
        u.update(
            H=H, W=W, mapping=mapping, weights=weights, n_nor=n_nor, n_dis=n_dis,
            # Kept for the cross-city LOO, which re-decomposes each unit for its
            # cross-city feature table (at the unit's own rank-CV k).
            X_all=X_all, fit_time_cols=fit_time_cols)

    # ── STEP 3b — Decomposition-quality check, retained units + summary ───────
    quality_rows = [
        analysis_decomposition_quality(u['label'], code, u['X_all'], u['W'],
                                       u['H'], u['fit_time_cols'])
        for code, u in units.items()]
    analysis_decomposition_quality_summary(quality_rows)
    analysis_city_mobility_curves(units)

    # ── STEP 3 — Within-city analyses: characterise each unit's components ──

    feats_by_city = {}   # short code -> per-component feats, for the cross-city test

    # Each unit's per-component feats + within-unit LOO are keyed by its city-event
    # code ('BR_Ida', 'WM_Dorian', ...), the key the cross-city folds address.
    for code, u in units.items():
        label, key, tag = u['label'], u['key'], u['tag']
        gdf, H, mapping, weights, W = u['gdf'], u['H'], u['mapping'], u['weights'], u['W']
        n_nor, n_dis = u['n_nor'], u['n_dis']
        first_day_nor, first_day_dis, ctx_lambda = (
            u['first_day_nor'], u['first_day_dis'], u['ctx_lambda'])
        print(f"\n── {label}: analysis ──")
        analysis_component_signature(W, n_nor, n_dis, first_day_nor, first_day_dis, tag)
        
        distances = analysis_spatial(label, H, mapping, gdf, weights, tag,
                                     lambda_ctx=ctx_lambda)
        socio = analysis_socioeconomic(label, key, tag, gdf, H, mapping,
                                       weights, lambda_ctx=ctx_lambda)
        M = analysis_od_function(label, tag, H, mapping, weights,
                                 landuse_by_code[code], lambda_ctx=ctx_lambda)

        feats = pd.concat([
            # Temporal features read W[:n_nor].
            temporal_features(W, n_nor, first_day_nor, SLOTS_ACTIVE, _INTERVAL_HOURS,
                              weekend_ratio_threshold=WEEKEND_RATIO_THRESHOLD),

            # Functional profile reads M.
            functional_features(M, SF_CATEGORIES),

            # Spatial: mean_distance, std_distance (reads H + OD centroid distances).
            spatial_features(H, distances),

            # Socioeconomic: median_income per endpoint mode (loading-weighted, ACS).
            socio,
            
            # Resilience reads W[n_dis:] against a baseline built from W[:n_nor].
            # The buffer columns [n_nor, n_dis) feed neither the resilience features
            # nor the curves.
            resilience_features(W, n_nor, first_day_nor, SLOTS_ACTIVE, n_dis=n_dis),

            # Free-plateau logistic recovery params.  recovery_alpha (the rate)
            # is the RES_COLS metric; recovery_level (the equilibrium L) rides
            # along for STEP 7 but is not itself a RES_COLS metric.  NaN when
            # day-0 is a total stop, the curve is constant, or the joint fit
            # fails the quality gate (see the RES_COLS comment).
            recovery_curve_features(W, n_nor, first_day_nor, SLOTS_ACTIVE,
                                    n_dis=n_dis, max_rate=ALPHA_MAX_RATE,
                                    min_fit_r2=ALPHA_MIN_FIT_R2,
                                    min_std=ALPHA_MIN_STD, level_bounds=LEVEL_BOUNDS,
                                    surge_bounds=SURGE_BOUNDS,
                                    surge_rate_bounds=SURGE_RATE_BOUNDS),
        ], axis=1)
        feats.insert(0, 'city', label)
        feats.insert(1, 'weight', weights)
        # The AGGREGATION weight: the component's share of the city's
        # NORMAL-period activity, derived exactly as _build_cross_city_feats
        # derives it.  `weight` above is the full-window importance
        # ‖W_i‖·‖H_i‖; weight_normal is what the city-level reconstruction
        # actually sums with, so any figure that aggregates components to a
        # city statement has to use this one.
        feats.insert(2, 'weight_normal', W[:n_nor].sum(axis=0) * H.sum(axis=1))

        analysis_time_function_corr(feats, tag)
        
        analysis_resilience_corr(feats, tag, lambda_ctx=ctx_lambda)
        
        curves = resilience_curves(W, n_nor, first_day_nor, SLOTS_ACTIVE, n_dis=n_dis)
        analysis_func_ordered_lines(W, n_nor, n_dis, first_day_nor, first_day_dis,
                                    curves, feats, tag)

        # Per-event-constant LEVEL features, added AFTER the within-city analyses
        # (which select specific column groups) so only the cross-city step sees
        # them (used only by the pooled_train mode): Saffir-Simpson arrival
        # intensity and evacuation strength.
        feats = feats.copy()
        _reg = next(c for c in CITY_EVENTS if c['code'] == code)
        feats['hurricane_intensity'] = _reg['ss_intensity']
        feats['evac_level'] = _reg['evac_level']
        # Stash for the cross-city steps after the loop, keyed by the city-event
        # code.  EVERYTHING downstream of the loop gates on this dict — dropping
        # this assignment silently ends the pipeline after STEP 3 (it did,
        # 2026-08-04, when a deletion cut it together with the retired Ridge).
        feats_by_city[code] = feats

    # Cross-city pooled heatmaps: the same figures as the per-city ones,
    # computed on every unit's components at once under the within-unit rank
    # normalization (see analysis_time_function_corr_pooled).
    if feats_by_city:
        print("\n── Time × function correlation, ALL city-events pooled ──")
        analysis_time_function_corr_pooled(feats_by_city)
        print("\n── Function × resilience correlation, ALL city-events pooled ──")
        analysis_func_resi_corr_pooled(feats_by_city)

    # The intra-city scope ends here: folder 1- is complete and internally
    # consistent (its per-unit AND pooled figures all reflect this run's k
    # policy).  Everything after is cross-city machinery.
    if PIPELINE_SCOPE == 'intra_city':
        print(f"\n── PIPELINE_SCOPE=intra_city: stopping after the within-city "
              f"characterisation ──")
        return

    # ── STEP 4 — Disaster (arrival intensity) vs resilience, ALL city-events pooled ──
    # x = the component's event-level Saffir-Simpson arrival intensity
    # (ss_intensity), y = raw cum_loss; points coloured by city-event and SIZED
    # by weight_normal, which also weights the per-level summary and the
    # reported Spearman — components carry very unequal shares of a city's
    # activity, so an equal-weight reading of this panel would not describe the
    # city it is about.  Not
    # split into train/test.  recovery_alpha's panel was dropped 2026-08-04
    # with the rest of its intra-city analyses; the rate keeps its role in the
    # STEP-7 transfer, it just is not scattered against intensity here.
    if feats_by_city:
        print("\n── Disaster intensity vs resilience scatter (all components) ──")
        pooled = pd.concat(
            [feats_by_city[c][['hurricane_intensity', 'cum_loss',
                               'weight_normal']].assign(code=c)
             for c in feats_by_city], ignore_index=True)
        os.makedirs(OUTPUT_DISASTER_VS_RESIL, exist_ok=True)
        vis_scatter_intensity_resilience(
            pooled, 'hurricane_intensity', ['cum_loss'], group_col='code',
            weight_col='weight_normal',
            title='Saffir-Simpson arrival intensity vs cum_loss '
                  '(all components, all city-events; weighted by each '
                  "component's share of its city's normal activity)",
            save_path=os.path.join(OUTPUT_DISASTER_VS_RESIL,
                                   'scatter_intensity_vs_resilience.png'))
        # The other two exposure definitions, derived per unit from the NHC best
        # track: closest-approach distance and the wind band the city sat in.
        # evac_level rides along from the registry.  All four then go into one
        # comparison figure — the point is which DEFINITION tracks the loss.
        expo, tracks = {}, {}
        for code in feats_by_city:
            cfg = next(c for c in CITY_EVENTS if c['code'] == code)
            bid = HURDAT2_BASIN_IDS[code.split('_', 1)[1]]
            if bid not in tracks:
                tracks[bid] = load_track(bid)
            gdf = units[code]['gdf'].to_crs(4326)
            cen = gdf.geometry.union_all().centroid
            e = city_exposure(tracks[bid], cen.y, cen.x)
            expo[code] = dict(track_dist_km=e['dist_km'],
                              local_wind_kt=e['local_wind_kt'],
                              exposure_band=e['band'],
                              closest_approach=str(e['time']),
                              evac_level=cfg['evac_level'])
            print(f"  [exposure] {code}: {e['dist_km']:6.1f} km, "
                  f"{e['band']}, evac {cfg['evac_level']:.2f}")
        for col in ('track_dist_km', 'local_wind_kt', 'evac_level'):
            pooled[col] = pooled['code'].map(lambda c: expo[c][col])

        vis_exposure_vs_cumloss(
            pooled,
            [('hurricane_intensity', 'storm Saffir-Simpson category at closest '
                                     'approach (3=TS .. 7=Cat4)', 'discrete'),
             ('local_wind_kt', 'wind band the city sat in, from the best-track '
                               'radii (kt)', 'discrete'),
             ('track_dist_km', 'closest-approach distance to the storm centre (km)',
              'continuous'),
             ('evac_level', 'population-weighted evacuation strength', 'continuous')],
            metric='cum_loss', group_col='code', weight_col='weight_normal',
            title='Four definitions of exposure vs component cum_loss '
                  '(weighted by each component\'s share of its city)',
            save_path=os.path.join(OUTPUT_DISASTER_VS_RESIL,
                                   'scatter_exposure_vs_cum_loss.png'))

        raw_dir = os.path.join(OUTPUT_DISASTER_VS_RESIL, 'raw_data')
        os.makedirs(raw_dir, exist_ok=True)
        pooled.to_csv(os.path.join(raw_dir, 'intensity_vs_resilience_raw.csv'),
                      index=False)
        pd.DataFrame(expo).T.rename_axis('code').to_csv(
            os.path.join(raw_dir, 'city_exposure_measures.csv'))
        print(f"  -> {OUTPUT_DISASTER_VS_RESIL}")

    # ── STEP 5 — Build the cross-city feature tables ──────────────────────────
    # Each unit takes a turn as the held-out test and EVERY other retained unit
    # trains that fold (split={'train': rest, 'test': [held]}) — there is no
    # fixed train/test list.  Every unit, in either role, is decomposed at its
    # own rank-CV k, so the two roles now share one decomposition.
    # Each method runs its PAIRED target standardization (CROSS_CITY_METHOD_STD:
    # spearman->single-city, pearson->multi-city), into cross_city_resi_pred/<label>/.
    if feats_by_city:
        all_codes = list(units)
        # Built here rather than reused from the within-city step so the SAME
        # global classification as STEP 2 (cc_lookups) labels every unit.
        print("\n── Cross-city: building feats (every unit at its own rank-CV k) ──")
        cc_train, cc_test = {}, {}
        dec_test = {}   # code -> the (W, H) behind the feats, for STEP 7
        for code in all_codes:
            u = units[code]
            print(f"  [cross-city feats] {u['label']} [{code}] k={u['cfg']['n_behaviors']}")
            feats_cc, dec_test[code] = _build_cross_city_feats(
                u['cfg'], u['X_all'], u['n_nor'], u['n_dis'], u['mapping'],
                u['gdf'], u['fit_time_cols'], u['cfg']['n_behaviors'],
                share_lookup=cc_lookups[code])
            # Both roles now read the SAME decomposition: the test-role k used to
            # be pinned at a fixed K_LOO_TEST because a brand-new city had no
            # tuned k, but the rank CV supplies one from the city's own matrix
            # with no labels, so the held-out unit uses its own k like any other.
            cc_train[code] = cc_test[code] = feats_cc

        # ── STEP 6 — Cross-city prediction (pairwise, LOO transfer) ─────
        # The heatmap still runs FIRST: nothing trains on its partition under
        # RANK_TRAIN_SCOPE='pooled', but the cluster-restricted DIAGNOSTIC and
        # the function co-riding figures read the clusters CSV it writes, and
        # a 'cluster' revert would again depend on this ordering.
        cc_merged = {c: rank_merge_feats(cc_test[c]) for c in all_codes}
        for method, (std_mode, base_label) in CROSS_CITY_METHOD_STD.items():
            print(f"\n── Cross-city pairwise cum_loss heatmap [{base_label}] ──")
            analysis_cross_city_pairs(cc_train, cc_test, all_codes,
                                      method=method,
                                      target_std=std_mode, subdir=base_label,
                                      level_feature_cols=LEVEL_FEATURE_COLS,
                                      model=RANK_MODEL,
                                      pooled_feature_cols=POOLED_FEATURE_COLS)

        print(f"\n── Rank channel LOO ({len(all_codes)} folds, "
              f"model={RANK_MODEL}, train scope={RANK_TRAIN_SCOPE}) ──")
        analysis_rank_channel(cc_merged, all_codes)

        # Mechanism figures: the mapping-direction PCA (one panel, no
        # clustering — what rule each city fits against the pooled one) and
        # the function co-riding graph (which still panels by the heatmap's
        # DESCRIPTIVE partition, reading back the clusters CSV).
        analysis_mapping_pca(cc_test)
        analysis_cluster_function_graph(cc_test)

        # City-level reconstruction of the cross-city cum_loss prediction vs ground
        # truth (runs only for pearson + multi_city_std; guarded inside the function).
        print("\n── Cross-city city-level cum_loss: prediction vs ground truth ──")
        analysis_cross_city_resi_pred(cc_train, cc_test, units, all_codes, global_iwf)

        # The same transfer's middle stage on its own: the centred loss spectrum
        # STEP 7's quantile mapping carries across cities, scored before the
        # ordering and the city-total anchor enter.
        print("\n── Cross-city centred loss spectrum: barycentre transfer ──")
        analysis_centered_spectrum(cc_test, all_codes)

        # ── STEP 7 — curve prediction (+ OD maps) ────────────────────────
        _run_step_7(cc_train, cc_test, units, all_codes, dec_test)


if __name__ == '__main__':
    main()
