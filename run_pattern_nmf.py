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
    aggr_denorm strategy (pooled features + observed r0, ridge — the same
    estimator the city_total/ headline figure reports; re-wired 2026-08-04).
    weight_normal aggregation turns the component curves into the city
    trajectory, compared against the observed curve alongside three reference
    lines: a city-wise forecast (one rate for the whole city, a single
    logistic — the decomposition-free counterpart), the component's own
    UNGATED full fit (the model family's ceiling), and the pooled-train mean
    rate alone (the L = 1, zero-shrinkage special case).

Output tree (outputs/nmf/)
--------------------------
Folder names carry a numeric prefix so a file browser lists them in pipeline
order; the prefixes are part of the paths below and of the OUTPUT_* constants.

    0-decomposition_quality/     STEP 1 + STEP 3, everything about the
        factorisation itself.  nmf_rank_cv.png (STEP 1) sweeps k under
        held-out-entry cross-validation for EVERY registry unit and is what
        sets each unit's n_behaviors and what EXCLUDED_CODES drops; the units
        it drops keep their (greyed) panel, since this figure is the record of
        why.  The STEP-3 quality check then covers only the RETAINED units,
        FIT-window only (the data the basis is fit on; the disaster period is
        out of scope): per-unit nmf_quality_<code>.png (per-slot distribution
        error, value-CDF overlay, component weights vs the NMF_MIN_COMP_FRAC
        health threshold) and the cross-city nmf_quality_summary.png
        dashboard.  raw_data/ holds the rank-CV curves and selection table plus
        the quality metrics.
    1-component_characteristics/  per-component characteristics, by type:
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
                        no income predictor), one subfolder per predictor —
                        cos_KNN/ and ridge/ — each with its leave-one-out
                        scatters (Spearman ρ), the LOO R² matrix, the pairwise
                        transfer heatmap (Spearman ρ) and raw_data/
    3-cross_city_curve_pred/     the STEP-7 outputs: bar_cross_city_curve_mae.png
        plus the three mechanism figures (rank_pred_vs_true, rank_to_cumloss_qm,
        alphaL_relationship_softreg),
        (the city-level whole-curve error of each forecast line, per city-event),
        per-unit city_magnitude_curve_<code>.png and component_curves_<code>.png,
        curve_pred_metrics.csv, and raw_data/ (per-day city curves by method +
        the per-component α/L table + the plotted MAE table)
    4-cross_city_od_pred/        per city-event, one self-contained HTML slider map
        of the daily predicted / observed / difference OD flows
    5-transferability/           the STEP-8 outputs: transferability_map.png,
        w2_decomposition.png and raw_data/ (domain distances, transfer matrices,
        the per-feature-group W2 split)

Adding a city-event means adding one CITY_EVENTS entry and providing its graph
pkl, its geo CSV, and its land-use (EPA Smart Location Database) and income
(ACS) caches under data/.  Geometry files are mandatory: loading raises
FileNotFoundError when a unit's geo CSV is absent.

Run
---
    python run_pattern_nmf.py
"""
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import rankdata, spearmanr
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
    vis_heatmap_temporal_signature, vis_nmf_quality, vis_nmf_quality_summary,
    vis_nmf_rank_cv,
    vis_line_nmf_component_timeline, vis_heatmap_od_function,
    vis_line_resilience_curves,
    vis_heatmap_corr,
    vis_hist_function_entropy,
    vis_bar_component_distance, vis_bar_component_income, vis_scatter_reg_pred,
    vis_heatmap_pair_transfer, vis_scatter_intensity_resilience,
    vis_exposure_vs_cumloss,
    vis_bar_cross_city_resi_pred, vis_scatter_city_pred, vis_bar_curve_mae,
    vis_curves_city_pred,
    vis_component_curves_grid, vis_od_flow_slider_html, vis_w2_decomposition,
    vis_rank_pred_vs_true, vis_rank_to_cumloss_qm, vis_func_vs_time_distribution,
    vis_alpha_level_relationship,
    vis_transferability_map,
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

OUTPUT_PLOTS = os.path.join(OUTPUT_DIR, 'nmf')

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

# Resilience regression: when True, each function's
# outflow + inflow shares are merged into ONE feature
# when False the 12 directional shares are used as-is.
MERGE_FUNC_DIRECTIONS = True

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
FUNC_COLS = ([f'share_from_{c}' for c in SF_CATEGORIES]
             + [f'share_to_{c}' for c in SF_CATEGORIES])

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
OUTPUT_QUALITY      = os.path.join(OUTPUT_PLOTS, '0-decomposition_quality')
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

OUTPUT_CHAR         = os.path.join(OUTPUT_PLOTS, '1-component_characteristics')
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

# The RANK channel runs BOTH predictors side by side, each into its own folder
# under component_rank/, because neither dominates: a 2026-08-04 leave-one-out
# ablation put ridge ahead on average (mean test R2 +0.36 vs +0.28, better on 8
# of 13 units) but it is the riskier of the two exactly where it matters most —
# BR_Ida, the largest unit at k=12, goes +0.10 -> -0.15, while cosine-kNN's
# bounded, shrunken predictions never blow up.  Showing both is the honest
# reading.  (Model choices elsewhere are pinned at their own sites: the legacy
# kNN(sigma) reference and the raw pooled channel hardcode 'cosine_knn'; the
# STEP-7 channels read CURVE_PRED_CITY_MODEL / CURVE_PRED_RANK_MODEL.)
CROSS_CITY_RANK_MODELS = {'cos_KNN': 'cosine_knn', 'ridge': 'ridge'}


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
CURVE_PRED_RANK_MODEL = 'ridge'

# ── STEP 8 parameters — transferability (source selection) ─────────────────────

# Everything analysis_transferability writes goes here.
OUTPUT_TRANSFER = os.path.join(OUTPUT_PLOTS, '5-transferability')

# A unit joins the comparison only with this many usable components; below it
# neither the OT distance nor the pairwise spearman is meaningful.
TRANSFER_MIN_ROWS = 5

# Mantel permutations.  The permutation is over UNIT LABELS (n! is tiny at
# n = 5, so the achievable p-floor is 1/120 — read the statistic, not the
# decimals, and treat P as descriptive at this sample size).
TRANSFER_MANTEL_PERM = 5000

# Per-unit colours for the transferability map, applied in CITY_EVENTS order.
_TRANSFER_COLORS = ['#0F4D92', '#4C9F70', '#7B5EA7', '#E28E2C', '#B0413E']

# Colours for the FEATURE GROUPS of the W2 decomposition.  The six functional
# shares are treated as ONE group: they are compositional (they sum to a fixed
# total per component), so splitting them would attribute one signal across six
# mutually-constrained columns.
_TRANSFER_GROUP_COLORS = {'func': '#0F4D92', 'dist': '#4C9F70',
                          'r0': '#B0413E'}

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
    vis_nmf_quality(
        q,
        title=f'{label} [{code}]: NMF decomposition quality, fit window '
              f'(k = {W.shape[1]})',
        save_path=os.path.join(OUTPUT_QUALITY, f'nmf_quality_{code}.png'))
    print(f"  Quality (fit window): dist_err={q['dist_err_mean']:.4f}  "
          f"rel_err={q['rel_err']:.4f}  "
          f"comps below {NMF_MIN_COMP_FRAC:.0%}: {q['n_below']}/{W.shape[1]}")
    return dict(code=code, k=W.shape[1],
                dist_err_mean=q['dist_err_mean'],
                rel_err=q['rel_err'], n_below=q['n_below'])


def analysis_decomposition_quality_summary(rows):
    """Cross-city quality dashboard + raw CSV, after every unit is decomposed."""
    df = pd.DataFrame(rows).set_index('code')
    os.makedirs(OUTPUT_QUALITY_RAW, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_QUALITY_RAW, 'nmf_quality_metrics.csv'))
    vis_nmf_quality_summary(
        df, NMF_MIN_COMP_FRAC,
        save_path=os.path.join(OUTPUT_QUALITY, 'nmf_quality_summary.png'))
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
        M, SF_CATEGORIES, weights=weights, ncols=3,
        save_path=os.path.join(OUTPUT_FUNC, f'heatmap_od_functionality{tag}.png'),
    )
    # Per-component functional dimensions — the 12 from/to shares, one row per
    # component (index), for inspecting the raw values behind the heatmap.
    os.makedirs(OUTPUT_FUNC_HM_RAW, exist_ok=True)
    functional_features(M, SF_CATEGORIES).to_csv(
        os.path.join(OUTPUT_FUNC_HM_RAW, f'component_functionality{tag}.csv'))

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

def analysis_cross_city(feats_by_city, lambda_ctx=None,
                        merge_func_directions=True, res_cols=None,
                        methods=tuple(CROSS_CITY_METHOD_STD), split=None,
                        target_std=None, subdir=None, level_feature_cols=(),
                        model='ridge', pooled_feature_cols=(),
                        min_rows=CROSS_CITY_MIN_ROWS):
    """Cross-city rank-transfer generalisation driven by an explicit disjoint
    train/test `split` of city-event codes: pooled(train) -> predict each test
    unit.  Each method's target standardization is HARD-PAIRED via
    CROSS_CITY_METHOD_STD (only spearman -> within_unit survives; the pearson /
    pooled_train method and the pooled-LOO display mode were retired 2026-08-04
    with the raw-value channel): target_std=None resolves the pairing, an
    explicit mismatching value raises ValueError.  Per method into
    cross_city_resi_pred/<subdir>/ (subdir=None -> the method's paired label):
    writes the predicted-vs-actual scatter AND its raw data
    (raw_data/cross_city_scatter_<col>_*_raw.csv), and RETURNS
    {method -> r2_table} so the caller can aggregate (e.g. the LOO matrix).
    split=None -> skipped (returns {}).
    lambda_ctx tags the filenames.  res_cols=None -> the full RES_COLS."""
    res_cols = list(RES_COLS) if res_cols is None else list(res_cols)
    # Defensive guard: reject a method/std combo OR an output label that contradicts the
    # pairing BEFORE anything is computed or written (all methods validated up front; the
    # subdir check stops results being written under the other method's folder).
    for m in methods:
        expected_std, expected_label = CROSS_CITY_METHOD_STD[m]
        if target_std is not None and target_std != expected_std:
            raise ValueError(
                f"analysis_cross_city: method '{m}' is paired with target_std "
                f"'{expected_std}' (CROSS_CITY_METHOD_STD); got '{target_std}'")
        # The model folder nests INSIDE the paired label, so the check is a
        # prefix one: component_rank/cos_KNN is legal, another method's label
        # still is not.
        if subdir is not None and not str(subdir).startswith(expected_label):
            raise ValueError(
                f"analysis_cross_city: method '{m}' writes under subdir "
                f"'{expected_label}' (CROSS_CITY_METHOD_STD); got '{subdir}'")
    if split is None:
        print("  [cross-city] split is None -> skipping the cross-city step.")
        return {}
    lambda_tag = _lambda_tag(lambda_ctx)

    # Per-unit predictor tables (merged func + distance + income).  The
    # merge is method-independent (only the rank flag inside cross_city_resilience
    # differs), so build them once and reuse across methods.
    if merge_func_directions:
        func_cols = [f'func_{c}' for c in SF_CATEGORIES]
        cities = {}
        for code, city_feats in feats_by_city.items():
            merged_feats = city_feats.copy()
            for c in SF_CATEGORIES:
                merged_feats[f'func_{c}'] = city_feats[f'share_from_{c}'] + city_feats[f'share_to_{c}']
            cities[code] = merged_feats
    else:
        func_cols = FUNC_COLS
        cities = feats_by_city
    # income is NOT a rank-channel predictor (dropped 2026-08-04): the same
    # ablation found it carries nothing the func shares and distance do not
    # already say in rank space — removing it left every fold within +-0.04 and
    # nudged the mean up (+0.280 -> +0.292).  The pooled_train paths (city-level
    # reconstruction, STEP-7) keep it: there it is a LEVEL predictor, which is a
    # different job and was not part of this test.
    feature_cols = func_cols + ['mean_distance']

    train = [c for c in split.get('train', []) if c in cities]
    test  = [c for c in split.get('test', []) if c in cities]

    results = {}
    for method in methods:
        rank = (method == 'spearman')   # spearman = rank transfer, pearson = raw
        std_mode, std_label = CROSS_CITY_METHOD_STD[method]   # paired (guard passed)
        out_dir = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED,
                               std_label if subdir is None else subdir)
        raw_dir = os.path.join(out_dir, 'raw_data')

        r2_table, pred, _groups = cross_city_resilience(
            cities, res_cols, feature_cols, rank=rank, split=split,
            target_std=std_mode, level_feature_cols=level_feature_cols, model=model,
            # Only the pooled columns this call actually has: the rank channel
            # dropped income from feature_cols, and the engine indexes
            # pooled_feature_cols INTO feature_cols before it reaches the branch
            # that would have ignored them.
            pooled_feature_cols=[c for c in pooled_feature_cols
                                 if c in feature_cols],
            min_rows=min_rows)
        if r2_table.shape[1] == 0:
            continue                                    # skipped/empty (warning already emitted)
        results[method] = r2_table

        # Predicted-vs-actual scatter per output column (test unit, or the pool).
        for col, pred_data in pred.items():
            # The engine hands back the model frame (within-unit z-scored ranks);
            # the figure shows the RANKS themselves.  The z-score's constants are
            # the rank vector's own mean/sd, so inverting is exact and carries no
            # test-label information: for untied ranks mean=(n+1)/2 and
            # sd=sqrt((n^2-1)/12) depend on n alone.  rankdata on the surviving
            # rows (cidx) reproduces them and stays correct under ties.
            # Only `pred` is rewritten here — r2_table was computed upstream on
            # the standardized arrays and the other cross_city_resilience callers
            # (STEP-6 city reconstruction, STEP-7) read the engine directly, so
            # none of them sees this.
            plot_data = dict(pred_data)
            for m, pm in pred_data.items():
                if pm is None:
                    continue
                y_std, p_std, cidx = (np.asarray(pm[0], dtype=float),
                                      np.asarray(pm[1], dtype=float), pm[2])
                # Every point belongs to the (single) test unit `col`.
                r = rankdata(cities[col].loc[cidx, m].to_numpy(float))
                sd = r.std() or 1.0
                plot_data[m] = (y_std * sd + r.mean(), p_std * sd + r.mean(),
                                cidx)
            # Spearman replaces R² on the panel: the target IS a within-unit rank,
            # so ordering is what this channel claims to transfer, while R² also
            # scores prediction amplitude (a 2026-08-04 sandbox found the two
            # rank models tie on rho and differ on R² almost entirely through the
            # amplitude of their predictions).  Invariant to the inversion above.
            col_stat = {}
            for m in res_cols:
                pm = pred_data.get(m)
                col_stat[m] = (np.nan if pm is None or len(pm[0]) < 3
                               else float(spearmanr(pm[0], pm[1]).statistic))
            col_summary = pd.DataFrame({
                'stat': col_stat,
                'passed': {m: (bool(col_stat[m] > 0) if pd.notna(col_stat[m])
                               else False) for m in res_cols},
                'status': {m: ('ok' if pred_data[m] is not None else 'insufficient_data')
                           for m in res_cols},
            })
            # No suptitle: it named every training unit, which at 13 units wrapped
            # to four lines over a single panel.  The filename carries the test
            # unit and the training set is every other unit by construction.
            fname = f'cross_city_scatter_{col}_{lambda_tag}.png'
            vis_scatter_reg_pred(
                plot_data, col_summary, res_cols, stat_label='test Spearman ρ',
                unit='rank within unit', title=None,
                save_path=os.path.join(out_dir, fname))

            # Raw data behind this scatter (predicted vs actual per metric), so each
            # point stays recoverable for later analysis.
            rows = []
            for m in res_cols:
                pm = plot_data.get(m)
                if pm is None:
                    continue
                y_true, y_pred, comp_index = pm
                for i in range(len(y_true)):
                    # Same scale as the figure (within-unit ranks), plus the
                    # panel's rho so the number stays with its points.
                    rows.append({'metric': m, 'comp_index': comp_index[i],
                                 'y_true': float(y_true[i]), 'y_pred': float(y_pred[i]),
                                 'spearman': col_stat.get(m, np.nan),
                                 'unit': col})
            if rows:
                os.makedirs(raw_dir, exist_ok=True)
                pd.DataFrame(rows).to_csv(
                    os.path.join(raw_dir, f'cross_city_scatter_{col}_{lambda_tag}_raw.csv'),
                    index=False)
    return results


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

    The headline predictions are the aggr_denorm strategy — keep the standardized
    component predictions, AGGREGATE them to a city score, then DENORMALIZE to
    day-equivalents with a 1-parameter city-scale VARIANCE MATCH learned NESTED-LOO on
    the other training cities (never the held city); the name is those three steps in
    order — run on the pooled features EXTENDED by r0 (the component's observed day-0
    baseline-normalized activity), under BOTH predictors (cosine-kNN and RidgeCV), into
    decomp_pred_aggr_denorm/<model>/: the 13-point LOO calibration scatter (predicted vs
    actual city cum_loss, R² quantified) + raw_data/city_pred.csv.  (A denorm_aggr
    sibling strategy was measured 2026-08-04 and removed — see the comment at _fold_ex.)

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

    def _merge(feats):
        m = feats.copy()
        for c in SF_CATEGORIES:
            m[f'func_{c}'] = feats[f'share_from_{c}'] + feats[f'share_to_{c}']
        return m

    train_merged = {c: _merge(feats_by_city[c]) for c in codes}
    test_merged  = {c: _merge(feats_test[c]) for c in codes}

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

    # ── The aggr_denorm STRATEGY on features + r0, both predictors ──
    # Same component-level transfer as _fold but with r0 (each component's
    # observed day-0 baseline-normalized activity — leak-free initial condition,
    # cross-city comparable, already a STEP-7 pooled predictor) appended to the
    # pooled features, and the predictor selectable.  Returns scity, the
    # weight_normal aggregate of the STANDARDIZED component predictions, which
    # the caller rescales at CITY scale (nested-LOO variance match).
    # (A denorm_aggr sibling — invert each component with the pooled-train
    # (mu, sigma) BEFORE aggregating — was measured 2026-08-04 and removed: an
    # affine denorm commutes with the weighted average, so it only swaps the
    # calibration frame to the component-level sigma, and both predictors
    # scored WORSE under it: cos-kNN R2 -0.27, ridge -0.80, vs +0.17 for
    # aggr_denorm/ridge here.)
    fcols_r0 = feature_cols + ['r0']
    pooled_r0 = POOLED_FEATURE_COLS + ['r0']

    def _fold_ex(held, rest, model_id):
        fold = {held: test_merged[held]}
        fold.update({c: train_merged[c] for c in rest})
        _, pred, _ = cross_city_resilience(
            fold, RES_COLS, fcols_r0, rank=False,
            split={'train': rest, 'test': [held]}, target_std=target_std,
            level_feature_cols=LEVEL_FEATURE_COLS, model=model_id,
            pooled_feature_cols=pooled_r0,
            min_rows=CROSS_CITY_MIN_ROWS)
        pm = pred.get(held, {}).get('cum_loss')
        if pm is None:
            return None
        ypred_std = np.asarray(pm[1], dtype=float)
        cidx = pm[2]
        if len(cidx) == 0:
            return None
        w = feats_test[held].loc[cidx, 'weight_normal'].to_numpy(dtype=float)
        wsum = float(w.sum())
        if wsum <= 0:                       # degenerate weights -> plain mean
            w, wsum = np.ones(len(cidx)), float(len(cidx))
        return dict(scity=float((w * ypred_std).sum() / wsum))

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

        # The aggr_denorm strategy x the same two predictors as the rank
        # channel, on features + r0: aggregate the standardized scores, then
        # the nested-LOO city-scale variance match (the inner scitys must come
        # from the SAME predictor and feature set, so the nested loop reruns
        # _fold_ex per fold).  Fewer than 2 usable inner cities -> NaN (the
        # variance match is undefined; cannot happen at 12 training units).
        for model_dir, model_id in CROSS_CITY_RANK_MODELS.items():
            outer_ex = _fold_ex(held, rest, model_id)
            if outer_ex is None:
                row[f'cum_loss_pred_aggr_denorm_{model_dir}'] = np.nan
                continue
            s_in2, gt_in2 = [], []
            for s2 in rest:
                inn2 = _fold_ex(s2, [c for c in rest if c != s2], model_id)
                if inn2 is not None:
                    s_in2.append(inn2['scity']); gt_in2.append(gt[s2])
            if len(s_in2) >= 2:
                s_in2, gt_in2 = np.array(s_in2), np.array(gt_in2)
                sd_s2 = s_in2.std()
                row[f'cum_loss_pred_aggr_denorm_{model_dir}'] = float(
                    gt_in2.mean() + (outer_ex['scity'] - s_in2.mean())
                    * (gt_in2.std() / (sd_s2 if sd_s2 > 0 else 1.0)))
            else:
                row[f'cum_loss_pred_aggr_denorm_{model_dir}'] = np.nan

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
        'aggr_denorm': 'Decompose+aggr+denorm (+r0)',
    }
    for strat, strat_title in _STRATEGY_TITLES.items():
        for model_dir in CROSS_CITY_RANK_MODELS:
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
    # The series are BUILT from CROSS_CITY_RANK_MODELS rather than hardcoded:
    # those keys name the columns written above, and vis_bar_cross_city_resi_pred
    # silently drops a column it cannot find, so a hardcoded name that drifted
    # out of sync would yield a headline figure showing only the baseline with
    # no error anywhere.  The assertion below makes that failure loud instead.
    _MODEL_LABELS = {'cos_KNN': 'cosine-kNN', 'ridge': 'ridge'}
    bar_pred_cols = tuple(
        (f'Component transfer, {_MODEL_LABELS.get(m, m)}',
         f'cum_loss_pred_aggr_denorm_{m}') for m in CROSS_CITY_RANK_MODELS)
    _missing = [c for _l, c in bar_pred_cols if c not in res.columns]
    if _missing:
        raise KeyError(f"analysis_cross_city_resi_pred: the headline figure's "
                       f"prediction columns {_missing} were never written; "
                       f"CROSS_CITY_RANK_MODELS and the LOO loop disagree.")
    vis_bar_cross_city_resi_pred(
        res, save_path=os.path.join(OUTPUT_CITY_TOTAL,
                                    'bar_cross_city_resi_pred.png'),
        pred_cols=bar_pred_cols,
        baseline_label='City-similarity baseline (no decomposition)')
    print(f"  [cross_city_resi_pred] -> {OUTPUT_CITY_TOTAL} "
          f"({len(res)} city-events)")


# Seed for the pairwise-transfer Louvain partition.  Louvain's node sweep is
# order-randomised, so an unpinned seed would repartition on every run and the
# heatmap's boxes would move between otherwise identical pipeline runs.  The
# partition is barely seed-sensitive here (50 seeds at gamma=1.0: ridge 50/50
# identical, cosine-kNN 48/50) — the seed buys literal reproducibility, not
# stability.
PAIR_LOUVAIN_SEED = 0
# Modularity's resolution gamma, which scales the null-model term
# gamma*k_i*k_j/2m: larger gamma charges more for keeping two units together,
# so communities come out smaller and more numerous.  Louvain takes no
# cluster-COUNT argument, so this is the only dial on how many boxes appear.
# Measured on the 13-unit SPEARMAN transfer matrices (clusters as cosine-kNN /
# ridge): gamma 0.5 -> 1/1, 0.8 -> 2/2, 1.0 -> 2/2, 1.2 -> 2/3, 1.5 -> 5/5,
# 2.0 -> 7/9, 5.0 -> 13/13.  1.2 sits just above the plateau where everything
# collapses into one or two communities and below 1.5, where the partition
# shatters into five.  It does NOT make the two rank models agree — they return
# 2 and 3 communities — so the boxes are read per model.  (Under the retired R²
# version of this matrix 1.2 did make them agree; that agreement was a property
# of the unbounded metric, not of the data.)  What survives both models and the
# metric change is the {PG_Ian, NA_Ian, NP_Ian, DT_Ian, CH_Dorian} core.
# Modularity's resolution limit (~sqrt(2m), m ~ 29-34 here) sits at the scale of
# these communities, so gamma below ~1 cannot split them whatever the data says.
PAIR_LOUVAIN_RESOLUTION = 1.2


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
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

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
    rank = (method == 'spearman')
    # Same rank channel as analysis_cross_city, so the same predictors:
    # income is out (see that function).
    feature_cols = [f'func_{c}' for c in SF_CATEGORIES] + ['mean_distance']

    def _merge(feats):
        m = feats.copy()
        for c in SF_CATEGORIES:
            m[f'func_{c}'] = feats[f'share_from_{c}'] + feats[f'share_to_{c}']
        return m

    train_merged = {c: _merge(feats_train[c]) for c in codes}
    test_merged  = {c: _merge(feats_test[c]) for c in codes}

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
            _r2_t, pred, _ = cross_city_resilience(
                feats_pair, RES_COLS, feature_cols, rank=rank,
                split={'train': [a], 'test': [b]}, target_std=target_std,
                level_feature_cols=level_feature_cols, model=model,
                pooled_feature_cols=[c for c in pooled_feature_cols
                                     if c in feature_cols],
                min_rows=min_rows)
            col = 'pooled_LOO' if a == b else b
            pm = pred.get(col, {}).get('cum_loss')
            mat.loc[a, b] = (float(spearmanr(pm[0], pm[1]).statistic)
                             if pm is not None and len(pm[0]) >= 3 else np.nan)

    out_dir = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED, subdir)
    raw_dir = os.path.join(out_dir, 'raw_data')
    os.makedirs(raw_dir, exist_ok=True)

    # Louvain over the off-diagonal transfer; the matrix is REORDERED into the
    # partition so each community is a contiguous block the figure can box, and
    # the CSV is written in the same order as the figure it backs.
    ordered, cl_labels, blocks = _transfer_communities(mat)
    mat = mat.reindex(index=ordered, columns=ordered)
    mat.to_csv(os.path.join(raw_dir, 'cross_city_pair_heatmap.csv'))
    pd.Series(cl_labels, name='cluster').rename_axis('code').loc[ordered].to_csv(
        os.path.join(raw_dir, 'cross_city_pair_clusters.csv'))
    vis_heatmap_pair_transfer(
        mat, title=f'Pairwise cross-city cum_loss rank transfer, Spearman ρ '
                   f'(each unit at its own rank-CV k)\nLouvain transfer '
                   f'communities boxed (γ={PAIR_LOUVAIN_RESOLUTION}); diagonal '
                   f'(within-unit LOO) excluded',
        xlabel='test city-event', ylabel='train city-event', blocks=blocks,
        vmax=1.0, cbar_label='Spearman ρ',
        save_path=os.path.join(out_dir, 'cross_city_pair_heatmap.png'))
    print(f"  [pairwise] cum_loss Spearman ρ heatmap ({method}) -> "
          f"{os.path.join(out_dir, 'cross_city_pair_heatmap.png')}  "
          f"[{len(blocks)} Louvain clusters: "
          f"{', '.join(f'C{c}={s}' for _, s, c in blocks)}]")
    return mat


# ── STEP 8 — transferability: does domain proximity predict rank transfer? ─────

def analysis_transferability(feats_test, codes):
    """Which city-events can lend their cum_loss ORDERING to which others, and
    can that be read off the FEATURES alone (i.e. before any target label is
    seen)?  This is the source-selection question for a new city-event.

    Scope — RANK only.  The pipeline's cum_loss channel is a rank channel: the
    ordering is what transfers across units (the numeric level is not learned
    from features but borrowed from the pooled training distribution and
    re-anchored per unit, see _quantile_mapped_chat).  Measured against the same
    domain distances, the NUMERIC cum_loss transfer carries no relation at all
    (sandbox 2026-07-28: spearman +0.02, Mantel P = 0.99, against -0.74 /
    P = 0.017 for the ordering), so this analysis is deliberately confined to
    the ordering and its figure is labelled accordingly.

    Distance — entropic-OT (Sinkhorn) W2 between the two units' COMPONENT
    CLOUDS in the standardized feature space, each component carrying its
    weight_normal as mass.  The two per-event LEVEL covariates are excluded:
    being constant within a unit they are domain identifiers in disguise and
    would separate the units by construction, making the comparison circular.

    Transfer — Ridge fitted on the source unit's components, applied to the
    target's, scored by spearman(prediction, observed cum_loss).  Directed, so
    the matrix is asymmetric; panel A can only draw the symmetrized value.

    Writes transferability_map.png plus the three matrices under raw_data/.

    Features follow the rank channel they explain (re-aligned 2026-08-04, when
    component_rank/ dropped income): func + mean_distance + r0.  Keeping income
    here while the rank models no longer see it would measure domain proximity
    over a coordinate the transfer being explained cannot use."""
    feature_cols = ([f'func_{c}' for c in SF_CATEGORIES]
                    + ['mean_distance', 'r0'])

    def _unit(code):
        t = feats_test[code].copy()
        for c in SF_CATEGORIES:
            t[f'func_{c}'] = t[f'share_from_{c}'] + t[f'share_to_{c}']
        X = t[feature_cols].to_numpy(dtype=float)
        y = t['cum_loss'].to_numpy(dtype=float)
        w = t['weight_normal'].to_numpy(dtype=float)
        ok = (np.isfinite(X).all(axis=1) & np.isfinite(y)
              & np.isfinite(w) & (w > 0))
        return X[ok], y[ok], w[ok] / w[ok].sum() if ok.sum() else (X, y, w)

    U = {c: _unit(c) for c in codes}
    usable = [c for c in codes if len(U[c][1]) >= TRANSFER_MIN_ROWS]
    if len(usable) < 3:
        print("  [transferability] fewer than 3 usable units; skipping.")
        return None
    codes = usable
    n = len(codes)
    pooled = np.vstack([U[c][0] for c in codes])
    mu = pooled.mean(axis=0)
    sd = np.where(pooled.std(axis=0) < 1e-12, 1.0, pooled.std(axis=0))
    Zs = {c: (U[c][0] - mu) / sd for c in codes}

    def _sinkhorn_coupling(Xa, wa, Xb, wb, eps_frac=0.1, iters=400):
        """Entropic-OT coupling; eps scaled to the cost median so the
        regularization is comparable across pairs of different spread."""
        C = ((Xa[:, None, :] - Xb[None, :, :]) ** 2).sum(-1)
        K = np.exp(-C / (eps_frac * np.median(C)))
        u = np.ones_like(wa)
        for _ in range(iters):
            v = wb / (K.T @ u + 1e-300)
            u = wa / (K @ v + 1e-300)
        return u[:, None] * K * v[None, :], C

    def _ridge_fit(X, y):
        m, s = X.mean(axis=0), X.std(axis=0)
        s = np.where(s < 1e-12, 1.0, s)
        Xs = (X - m) / s
        A = Xs.T @ Xs + np.eye(Xs.shape[1])            # ridge, alpha = 1
        coef = np.linalg.solve(A, Xs.T @ (y - y.mean()))
        return lambda Q: ((Q - m) / s) @ coef + y.mean()

    # A squared-Euclidean cost is additive over dimensions, so ONE coupling
    # yields both the pair's W2 and its exact split over feature groups.
    groups = {'func': list(range(len(SF_CATEGORIES))),
              'dist': [len(SF_CATEGORIES)],
              'r0': [len(SF_CATEGORIES) + 1]}
    W2 = np.zeros((n, n))
    T = np.full((n, n), np.nan)
    pair_rows = []
    for i, a in enumerate(codes):
        pred_a = _ridge_fit(U[a][0], U[a][1])
        for j, b in enumerate(codes):
            if i < j:
                gam, _ = _sinkhorn_coupling(Zs[a], U[a][2], Zs[b], U[b][2])
                per_dim = np.array([
                    float((gam * (Zs[a][:, None, d] - Zs[b][None, :, d]) ** 2)
                          .sum())
                    for d in range(Zs[a].shape[1])])
                W2[i, j] = W2[j, i] = float(per_dim.sum())
                row = dict(
                    pair=f'{a.split("_", 1)[-1][:3]}–{b.split("_", 1)[-1][:3]}',
                    a=a, b=b, w2=W2[i, j])
                row.update({g: float(per_dim[dims].sum())
                            for g, dims in groups.items()})
                pair_rows.append(row)
            if i != j:
                T[i, j] = float(spearmanr(pred_a(U[b][0]), U[b][1]).statistic)
    Tsym = np.nan_to_num((T + T.T) / 2.0)
    dec = pd.DataFrame(pair_rows)
    dec['transfer'] = [Tsym[codes.index(r.a), codes.index(r.b)]
                       for r in dec.itertuples()]
    dec = dec.sort_values('w2').reset_index(drop=True)

    # Mantel: the pair entries are not independent (each unit appears in n-1 of
    # them), so significance comes from permuting UNIT LABELS, not pairs.
    iu = np.triu_indices(n, 1)
    rho = float(spearmanr(W2[iu], Tsym[iu]).statistic)
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(TRANSFER_MANTEL_PERM):
        p = rng.permutation(n)
        r = float(spearmanr(W2[np.ix_(p, p)][iu], Tsym[iu]).statistic)
        if abs(r) >= abs(rho):
            hits += 1
    pval = (hits + 1) / (TRANSFER_MANTEL_PERM + 1)

    # 2-D view of the SAME standardized space the distances live in; the map is
    # an approximation of it, which is why panel B repeats the claim in full
    # dimension.
    Zall = np.vstack([Zs[c] for c in codes])
    Zc = Zall - Zall.mean(axis=0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    var = (Zc @ vt.T).var(axis=0)
    var_ratio = (var / var.sum())[:2]
    emb = {c: (Zs[c] - Zall.mean(axis=0)) @ vt[:2].T for c in codes}
    centroids = {c: emb[c].mean(axis=0) for c in codes}

    os.makedirs(os.path.join(OUTPUT_TRANSFER, 'raw_data'), exist_ok=True)
    for name, M in (('domain_distance_w2', W2), ('rank_transfer_directed', T),
                    ('rank_transfer_symmetrized', Tsym)):
        pd.DataFrame(M, index=codes, columns=codes).to_csv(
            os.path.join(OUTPUT_TRANSFER, 'raw_data', f'{name}.csv'))
    dec.to_csv(os.path.join(OUTPUT_TRANSFER, 'raw_data',
                            'w2_group_decomposition.csv'), index=False)
    # Label by EVENT, not city: two of the units are the same city under
    # different hurricanes, and panel C exists precisely to show that they are
    # not each other's nearest neighbour.
    labels = {c: (c.split('_', 1)[1] if '_' in c else c) for c in codes}
    colors = {c: _TRANSFER_COLORS[i % len(_TRANSFER_COLORS)]
              for i, c in enumerate(codes)}
    out_png = os.path.join(OUTPUT_TRANSFER, 'transferability_map.png')
    vis_transferability_map(emb, centroids, W2, T, Tsym, labels, colors,
                            var_ratio, rho, pval, save_path=out_png)
    dec_png = os.path.join(OUTPUT_TRANSFER, 'w2_decomposition.png')
    vis_w2_decomposition(dec['pair'].tolist(), dec, dec['transfer'].to_numpy(),
                         list(groups), _TRANSFER_GROUP_COLORS,
                         save_path=dec_png)
    print(f"  [transferability] W2 vs rank transfer: spearman {rho:+.2f} "
          f"(Mantel P = {pval:.3f}, {len(iu[0])} pairs) -> {out_png}")
    print("  [transferability] W2 group contributions vs rank transfer: "
          + ", ".join(f"{g} {float(spearmanr(dec[g], dec['transfer']).statistic):+.2f}"
                      for g in groups)
          + f" -> {dec_png}")
    return pd.DataFrame(T, index=codes, columns=codes)


# ── STEP 7 — cross-city curve prediction ────────────────────────────────────────

# Display labels of the method lines (dict order = plot / colour order).
# 'pred'  — component-wise plateau inversion: each test component's plateau L
#           solved from its QUANTILE-MAPPED cum_loss prediction (rank-path
#           ordering + pooled-train spread + raw-path city total), aggregated
#           by weight_normal.
# 'city'  — city-wise: ONE α for the whole city predicted from the OTHER
#           cities' city-level features (decomposition-free analog of 'pred').
_CURVE_METHOD_LABELS = {
    'pred':       'Component-wise pred (kNN)',
    'city':       'City-wise pred (kNN)',
    'oracle':     'Own-fit (oracle)',
    'train_mean': 'Train-mean',
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
    #   fcols_r0/pooled_r0 — the city-total channel, matching STEP-6's
    #     aggr_denorm exactly (pooled features + observed r0).
    #   rank_feature_cols — the rank channel, matching component_rank/ exactly
    #     (income dropped) + STEP-7's own r0.
    feature_cols = ([f'func_{c}' for c in SF_CATEGORIES]
                    + ['mean_distance', 'median_income_combined'])
    fcols_r0 = feature_cols + ['r0']
    pooled_r0 = POOLED_FEATURE_COLS + ['r0']
    rank_feature_cols = [f'func_{c}' for c in SF_CATEGORIES] + ['mean_distance']

    def _merge(feats):
        m = feats.copy()
        for c in SF_CATEGORIES:
            m[f'func_{c}'] = feats[f'share_from_{c}'] + feats[f'share_to_{c}']
        return m

    train_merged = {c: _merge(feats_by_city[c]) for c in codes}
    test_merged = {c: _merge(feats_test[c]) for c in codes}

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

    def _rank_score_prediction(held, rest, target):
        """Predicted within-city ORDERING of `target` over the held unit's
        components — the engine's RANK path (every feature and the target
        rankdata'd WITHIN each unit, then within-unit z-scored; cosine-kNN in
        that frame).  Rank transfer is immune to the cross-city level/scale
        drift the pooled frame must assume away, which is why this channel
        transfers best.  Only the ORDER of the returned scores means anything
        (each score is a weighted mean of other cities' within-city ranks).
        None when the engine returns no prediction (caller falls back to the
        raw-path values)."""
        te = test_merged[held].copy()
        te[target] = np.arange(len(te), dtype=float)   # placeholder ramp
        fold = {held: te}
        fold.update({c: train_merged[c] for c in rest})
        _, pred, _ = cross_city_resilience(
            fold, [target], rank_feature_cols + ['r0'], rank=True,
            split={'train': rest, 'test': [held]}, target_std='within_unit',
            model=CURVE_PRED_RANK_MODEL, min_rows=CROSS_CITY_MIN_ROWS)
        pm = pred.get(held, {}).get(target)
        if pm is None:
            return None
        _, ypred_std, cidx = pm
        return pd.Series(np.asarray(ypred_std, dtype=float),
                         index=cidx).reindex(test_merged[held].index)

    def _city_score(held, rest):
        """weight_normal mean of the STANDARDIZED component cum_loss predictions
        for one unit, i.e. the unit's position on the pooled-train scale without
        ever un-standardizing a component.  Keeping the components standardized
        is what separates this from the raw channel, whose per-component
        un-standardization multiplies by the LARGE component-to-component spread
        and mis-calibrates the aggregate.  The test table's target is the usual
        placeholder ramp, so nothing about the held unit's own losses enters.
        Since 2026-08-04 this is STEP-6's aggr_denorm recipe verbatim: pooled
        features + observed r0, predictor CURVE_PRED_CITY_MODEL."""
        te = test_merged[held].copy()
        te['cum_loss'] = np.arange(len(te), dtype=float)
        fold = {held: te}
        fold.update({c: train_merged[c] for c in rest})
        _, pred, _ = cross_city_resilience(
            fold, ['cum_loss'], fcols_r0, rank=False,
            split={'train': rest, 'test': [held]}, target_std='pooled_train',
            level_feature_cols=LEVEL_FEATURE_COLS, model=CURVE_PRED_CITY_MODEL,
            pooled_feature_cols=pooled_r0, min_rows=CROSS_CITY_MIN_ROWS)
        pm = pred.get(held, {}).get('cum_loss')
        if pm is None:
            return np.nan
        _, ypred_std, cidx = pm
        w = feats_test[held].loc[cidx, 'weight_normal'].to_numpy(dtype=float)
        yp = np.asarray(ypred_std, dtype=float)
        return float((w * yp).sum() / w.sum()) if w.sum() > 0 else float(yp.mean())

    def _city_total_prediction(held, rest):
        """Predicted CITY cum_loss in day-equivalents: map the held unit's
        standardized city score onto the day-equivalent scale with a
        one-parameter variance match, whose mean and scale come from a NESTED
        leave-one-out over the training units only, so the held unit never
        enters its own calibration.

        Since 2026-08-04 this matches STEP 6's surviving aggr_denorm strategy —
        pooled features + observed r0, predictor CURVE_PRED_CITY_MODEL (ridge,
        the only variant that calibrates at 13 units: LOO R² +0.17, MAE 1.26)
        — so the city total the quantile mapping shifts onto and the city_total/
        headline figure are the same estimator again.  (Until that date this
        function kept the pre-r0 cosine-kNN recipe after STEP 6 dropped it.)

        This is the location the quantile mapping shifts onto.  It replaces the
        raw channel's weight_normal aggregate because the variance match uses
        the SMALL city-to-city dispersion of cum_loss instead of the large
        per-component one: measured on the five units, city-total MAE 0.270
        against 0.785 day-equivalents.  NaN when the calibration cannot be
        formed, and the caller then falls back to the raw aggregate."""
        outer = _city_score(held, rest)
        s_in, g_in = [], []
        for t in rest:
            v = _city_score(t, [c for c in rest if c != t])
            if np.isfinite(v):
                s_in.append(v); g_in.append(gt_cum[t])
        if not np.isfinite(outer) or len(s_in) < 2:
            return np.nan
        s_in, g_in = np.asarray(s_in), np.asarray(g_in)
        sd_s = float(s_in.std())
        return float(g_in.mean() + (outer - s_in.mean())
                     * (float(g_in.std()) / (sd_s if sd_s > 0 else 1.0)))

    def _cbase_spread(r0_vec, days, mu_a):
        """Within-city standard deviation of the BACKBONE-implied losses, i.e.
        of Σ_d (1 − logistic(d; r0_j, mu_a)) over the components with a usable
        anchor.  Every input is observed (each component's r0) or already
        pooled-train (the mean rate), so this quantity is available for the
        held-out unit as well; it is the observable proxy the spread scaling
        below relies on.  NaN when fewer than two components qualify."""
        v = [float(np.sum(1.0 - 1.0 / (1.0 + (1.0 / r - 1.0)
                                       * np.exp(-mu_a * days))))
             for r in np.asarray(r0_vec, dtype=float)
             if np.isfinite(r) and r > 1e-6]
        return float(np.std(v)) if len(v) >= 2 else np.nan

    def _quantile_mapped_chat(chat_raw, score, obs, wn, rest, mu_a, c_city):
        """Component cum_loss predictions assembled by QUANTILE MAPPING, the
        comonotone assignment, i.e. the optimal-transport map on the line.
        Four ingredients, each contributing what it transfers best:

          ordering  the rank-path score (who loses more, within the city);
          shape     the pooled-TRAIN cum_loss distribution — components ranked
                    k of n get its (k−0.5)/n quantiles.  This replaces the raw
                    kNN values' spread, which a kernel smoother compresses
                    toward the mean (it cannot extrapolate);
          spread    those quantiles are SCALED by the ratio of the held unit's
                    backbone-loss spread to the training units' mean backbone-
                    loss spread (_cbase_spread).  The true within-city cum_loss
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
        pool = np.concatenate(
            [train_merged[c]['cum_loss'].dropna().to_numpy(dtype=float)
             for c in rest] or [np.array([])])
        r0 = np.array([obs.iloc[0, j] if np.isfinite(obs.iloc[:, j]).any()
                       else np.nan for j in range(obs.shape[1])], dtype=float)
        days = obs.index.to_numpy(dtype=float)
        scale = np.nan
        if score is not None and len(pool):
            sc = score.to_numpy(dtype=float)
            ok = np.isfinite(sc) & np.isfinite(r0) & (r0 > 1e-6)
            if ok.sum() >= 2:
                s_test = _cbase_spread(r0, days, mu_a)
                s_train = [_cbase_spread(train_merged[c]['r0'].to_numpy(dtype=float),
                                         days, mu_a) for c in rest]
                s_train = [v for v in s_train if np.isfinite(v) and v > 0]
                scale = (s_test / float(np.mean(s_train))
                         if s_train and np.isfinite(s_test) else 1.0)
                idx = np.where(ok)[0][np.argsort(sc[ok], kind='stable')]
                pos = (np.arange(len(idx)) + 0.5) / len(idx)
                out[idx] = scale * np.quantile(pool, pos)
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
        cum_hat, spread_scale = _quantile_mapped_chat(
            cum_hat_raw, rank_score, obs, wn, rest, mu_a, city_total_hat)
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
                   'spread_scale': spread_scale,
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
        vis_curves_city_pred(
            days, gt_rel * base,
            {_CURVE_METHOD_LABELS[lab]: city_rel[lab] * base
             for lab in _CITY_FIGURE_METHODS if lab in city_rel},
            ylabel='daily mobility (flow volume per day)',
            title=f'{held}: city mobility over the disaster window',
            save_path=os.path.join(OUTPUT_CURVE_PRED,
                                   f'city_magnitude_curve_{held}.png'))
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
              f"city total {city_total_hat:.2f}; spread ×{spread_scale:.2f}; "
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
        vis_rank_pred_vs_true(
            par_df, save_path=os.path.join(OUTPUT_CURVE_PRED,
                                           'rank_pred_vs_true.png'))
        vis_rank_to_cumloss_qm(
            par_df, save_path=os.path.join(OUTPUT_CURVE_PRED,
                                           'rank_to_cumloss_qm.png'))
        vis_alpha_level_relationship(
            par_df, save_path=os.path.join(OUTPUT_CURVE_PRED,
                                           'alphaL_relationship_softreg.png'))
    # Accuracy bar: per city-event, the CITY-LEVEL curve error of every method line,
    # so the four lines are compared on the quantity the analysis actually optimises
    # (the whole-curve MAE) rather than on any single fitted parameter.  Methods are
    # the forecast lines of _CITY_FIGURE_METHODS, in that order; a method missing for
    # a unit leaves a gap.  The oracle's own MAE stays in curve_pred_metrics.csv.
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

def _run_steps_78(cc_train, cc_test, units, all_codes, dec_test):
    """STEP 7 (curve prediction + OD maps) and STEP 8 (transferability)."""
    print("\n── Cross-city curve prediction (clean-rate forecast, surge-model fit) ──")
    analysis_cross_city_curve_pred(cc_train, cc_test, units, all_codes, dec_test)
    print("\n── Transferability: domain proximity vs rank transfer ──")
    analysis_transferability(cc_test, all_codes)


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

        # ── STEP 6 — Cross-city prediction (LOO transfer, pairwise, reconstruction) ──
        for method, (std_mode, base_label) in CROSS_CITY_METHOD_STD.items():
          for model_dir, model_id in CROSS_CITY_RANK_MODELS.items():
            std_label = os.path.join(base_label, model_dir)
            print(f"\n── Cross-city LOO [{method}: {std_label}] "
                  f"({len(all_codes)} folds, model={model_id}) ──")
            loo_r2 = {}   # held_code -> r2 Series over RES_COLS
            for held in all_codes:
                rest = [c for c in all_codes if c != held]
                fold_feats = {held: cc_test[held]}
                fold_feats.update({c: cc_train[c] for c in rest})
                print(f"  fold: test [{held}] <- train [{'+'.join(rest)}]")
                res = analysis_cross_city(
                    fold_feats, lambda_ctx=None,
                    merge_func_directions=MERGE_FUNC_DIRECTIONS,
                    # component_rank/ carries cum_loss only (2026-08-04):
                    # recovery_alpha's rank transfer was never the analysis of
                    # record here, and STEP 7 predicts the rate through its own
                    # calls, not through this folder.
                    res_cols=['cum_loss'],
                    methods=[method],
                    split={'train': rest, 'test': [held]},
                    target_std=std_mode, subdir=std_label,
                    level_feature_cols=LEVEL_FEATURE_COLS, model=model_id,
                    pooled_feature_cols=POOLED_FEATURE_COLS)
                r2_table = (res or {}).get(method)
                if r2_table is not None and held in r2_table.columns:
                    loo_r2[held] = r2_table[held]

            # Combined LOO R² matrix (rows = metrics, cols = held-out unit).
            if loo_r2:
                mat = pd.DataFrame(loo_r2).reindex(index=['cum_loss'],
                                                   columns=all_codes)
                raw_dir = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED, std_label,
                                       'raw_data')
                os.makedirs(raw_dir, exist_ok=True)
                out_csv = os.path.join(raw_dir, 'loo_cross_city_r2_baseline.csv')
                mat.to_csv(out_csv)
                print(f"  [{model_dir}] LOO cross-city R² mean="
                      f"{mat.loc['cum_loss'].mean():+.3f} -> {out_csv}")

            # Pairwise single-train -> single-test cum_loss R² (ordered pairs);
            # the diagonal is the within-unit leave-one-component-out.
            print(f"── Cross-city pairwise cum_loss heatmap [{std_label}] ──")
            analysis_cross_city_pairs(cc_train, cc_test, all_codes,
                                      method=method,
                                      target_std=std_mode, subdir=std_label,
                                      level_feature_cols=LEVEL_FEATURE_COLS,
                                      model=model_id,
                                      pooled_feature_cols=POOLED_FEATURE_COLS)

        # City-level reconstruction of the cross-city cum_loss prediction vs ground
        # truth (runs only for pearson + multi_city_std; guarded inside the function).
        print("\n── Cross-city city-level cum_loss: prediction vs ground truth ──")
        analysis_cross_city_resi_pred(cc_train, cc_test, units, all_codes, global_iwf)

        # ── STEP 7-8 — curve prediction (+ OD maps) and transferability ──────
        _run_steps_78(cc_train, cc_test, units, all_codes, dec_test)


if __name__ == '__main__':
    main()
