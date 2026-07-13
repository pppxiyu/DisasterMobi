"""
Production pipeline: NMF decomposition of disaster origin–destination (OD) flow
mobility for the five hurricane city-events in CITY_EVENTS (BR_Ida, FM_Ian,
WM_Dorian, WM_Isaias, LC_Laura), per-component characterisation, and cross-city
resilience prediction.

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
    plus the two resilience targets (cum_loss, and the exponential-recovery
    rate recovery_lambda fitted from each component's lowest point onward —
    higher = faster recovery), and correlation and Ridge-regression
    analyses relate those predictors to the targets.  Two per-event-constant
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
    unit gets two tables, because the two prediction roles decompose the same
    matrix at different k: the train role at the unit's own k, and the test
    role re-decomposed at a fixed k = 10.
STEP 6 — Cross-city prediction.
    Each method runs in its enforced standardization frame (spearman in the
    rank frame, pearson in the raw-value frame) through three analyses on the
    STEP-5 tables: the leave-one-city-event-out transfer, the pairwise
    train-to-test comparison, and — for pearson only — the reconstruction of
    each city's cum_loss (the cumulative daily activity deficit over the
    disaster window, in day-equivalents — the resilience target)
    against the decomposition-free similarity baseline.

Output tree (outputs/nmf/)
--------------------------
    component_characteristics/   per-component characteristics, by type:
        temporal/           signature heatmap + timeline (W)
        spatial/            per-component flow-distance bar (H) + raw CSV
        socioeconomic/      per-component median-income bars + raw CSV
        func/               O×D functional cross-tab heatmaps + entropy + raw CSVs
        func_vs_temporal/   time × function correlation figures
        resilience/         per-function ordered line stacks: W timelines and r(d)
                            curves, where r(d) is the component's day-d activity
                            relative to its matched pre-disaster baseline
    resilience_corr/
        func_vs_resilience/      correlation heatmaps per method, top-pair scatters,
                                 intra_city_loss_reg_<method>/ (Ridge leave-one-out
                                 diagnostics)
        temporal_vs_resilience/  weekday_ratio heatmap + peak-slot / peak-period bars
        disaster_vs_resilience/  pooled intensity-vs-resilience scatter
    cross_city_resi_pred/        the STEP-6 outputs: bar_cross_city_resi_pred.png +
        raw_data/ at the root; per-method leave-one-out scatters, R² matrix and
        pairwise heatmap under cross_city_pred_rank/ (spearman) and
        cross_city_pred_raw_value/ (pearson)

Adding a city-event means adding one CITY_EVENTS entry and providing its graph
pkl, its geo CSV, and its land-use (EPA Smart Location Database) and income
(ACS) caches under data/.  Geometry files are mandatory: loading raises
FileNotFoundError when a unit's geo CSV is absent.

Run
---
    python run_pattern_nmf.py
"""
import os

import numpy as np
import pandas as pd

from config import (
    BR_GRAPH_PATH, FM_GRAPH_PATH, BR_ANALYSIS_DAYS, FM_ANALYSIS_DAYS,
    BR_GEO_CSV, FM_GEO_CSV, AGG_LEVEL,
    DATA_DIR, OUTPUT_DIR, SLOT_PER_DAY, SLOTS_ACTIVE,
)
from utils.pattern_analysis.graph_io import (
    load_graphs_trimmed, build_distance_array, build_income_array,
)
from utils.pattern_analysis.decomposition import select_segment_columns
from utils.pattern_analysis.nmf_pipeline import (
    build_city_matrices, decompose_city, decompose_city_context,
)
from utils.pattern_analysis.visualization import (
    vis_heatmap_temporal_signature,
    vis_line_nmf_component_timeline, vis_heatmap_od_function,
    vis_scatter_component_features,
    vis_bar_function_by_peakslot, vis_line_resilience_curves,
    vis_bar_resilience_by_peakslot, vis_heatmap_corr_split,
    vis_hist_function_entropy,
    vis_bar_component_distance, vis_bar_component_income, vis_scatter_reg_pred,
    vis_heatmap_pair_r2, vis_scatter_intensity_resilience,
    vis_bar_cross_city_resi_pred,
)
from utils.pattern_analysis.space_function import (
    category_lookup_from_landuse, build_od_function_matrix,
)
from utils.pattern_analysis.component_features import (
    temporal_features, functional_features, time_function_correlation,
    resilience_features, resilience_curves, component_function_entropy,
    spatial_features, socioeconomic_features, recovery_rate_features,
)
from utils.pattern_analysis.ml_resilience import (
    run_city_resilience_linear, cross_city_resilience,
)
from utils.data_processing.geo_loader import load_city_geo
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
         n_behaviors=11, l1_reg=0.5, filter_factor=0, ss_intensity=5,  # Ida @ BR ~Cat 2
         evac_level=0.091041,  # BG-pop-weighted HEvOD 3-level evacuation strength (data/evacuation_orders)
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Wednesday', first_day_disaster='Sunday'),
    dict(code='FM_Ian', label='Fort Myers', key='Fort_Myers',
         graph='data/Fort_Myers_Ian_2022_graph_intersection.pkl',
         geo={'block_group': 'data/Fort_Myers_block_group_geo.csv'},
         analysis_days=44, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=11, l1_reg=0.13, filter_factor=0, ss_intensity=7,  # Ian @ FM ~Cat 4
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
         n_behaviors=9, l1_reg=1.626, filter_factor=0, ss_intensity=5,  # Dorian @ Wilmington ~Cat 2 (true arrival)
         evac_level=0.814236,  # BG-pop-weighted HEvOD 3-level evacuation strength (data/evacuation_orders)
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Sunday', first_day_disaster='Thursday'),
    dict(code='WM_Isaias', label='Wilmington', key='Wilmington',
         graph='data/Wilmington_Isaias_2020_graph_intersection.pkl',
         geo={'block_group': 'data/Wilmington_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=9, l1_reg=0.315, filter_factor=0, ss_intensity=4,  # Isaias @ Wilmington ~Cat 1
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
AXIS_CATEGORIES = list(SF_CATEGORIES) + ['Mix']

# Functional features come from the O×D cross-tab with Mix and Unknown dropped
# and the six categories renormalised, split by flow direction.  Shares are
# full row/column sums, so same-function diagonal flow counts on both sides.
#   share_from_<cat> — outflow side, the fraction departing from function <cat>
#   share_to_<cat>   — inflow side, the fraction arriving at function <cat>
FUNC_COLS = ([f'share_from_{c}' for c in SF_CATEGORIES]
             + [f'share_to_{c}' for c in SF_CATEGORIES])

# The resilience target.  cum_loss is computed from the relative-activity curve
# r, where r(d) = the component's daily total on disaster day d divided by its
# weekday/weekend-matched pre-disaster baseline, 3-day smoothed (see
# component_features.resilience_features).  r = 1 means the normal level.
#   cum_loss        — Σ (1 − r), the NET unclipped cumulative deviation over the
#                     disaster window, in day-equivalents (above-baseline surges
#                     cancel drops, which makes it linear in r and additive
#                     across components — the property the city-level
#                     reconstruction relies on).  HIGHER = WORSE.
#   recovery_lambda — the exponential-recovery rate λ of the deficit
#                     D(d) = D0·e^(−λ·(d − d_min)) fitted from each component's
#                     lowest point onward (component_features.
#                     recovery_rate_features; added 2026-07-12).  A RATE:
#                     HIGHER = FASTER recovery = MORE resilient — the OPPOSITE
#                     reading direction to cum_loss.  NaN for components that
#                     never fell below baseline (nothing to recover).
# The city-level reconstruction (STEP 6) stays cum_loss-only.
# HISTORY (2026-07-12): the analyses previously also carried drop_depth,
# early_collapse, recovery_day and recovery_deficit.  They were retired from
# every analysis, figure and output; resilience_features still COMPUTES all
# five (kept long-term, see the note in component_features.resilience_features),
# so restoring one is just adding its name back to this list.
RES_COLS = ['cum_loss', 'recovery_lambda']

# Resilience REGRESSION (prediction) method(s); intra-city -> intra_city_loss_reg_<method>/,
# inter-city -> cross_city_resi_pred/<label>/ (CROSS_CITY_METHOD_STD).
# 'spearman' RANK-regresses (features AND the metric rank-transformed before the
# Ridge fit -> a multivariate partial Spearman, matching the correlation heatmap);
# 'pearson' regresses the RAW standardized values.  The rank flag fed to the Ridge
# helpers is simply (method == 'spearman'); the scatter axis unit follows suit.
RESIL_REG_METHODS = ['spearman', 'pearson']

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
GLOBAL_IWF_SCALE = 1.52    # untuned default; the city-MAE tuning (2026-07-07) OVERFIT the
                           # 4-city validation (n=5) and WORSENED the 5-city production MAE
                           # (0.897 -> 1.46), so it was reverted.  See nmf_cityMAE_* studies.

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

# Correlation method(s) for the resilience-vs-feature heatmap.  Each method is
# drawn into its own heatmap_resilience_corr_<method> subfolder.  'spearman' =
# rank correlation (robust to non-linearity); 'pearson' = linear on raw values.
RESIL_CORR_METHODS = ['spearman', 'pearson']

# All per-component characteristics live under component_characteristics, one
# subfolder per characteristic type.
#   temporal/          signature heatmap + timeline (temporal factor W)
#   spatial/           per-component flow-distance figure (spatial factor H)
#   func/              per-component O×D functional cross-tabs
#   func_vs_temporal/  time × function correlation figures
OUTPUT_CHAR         = os.path.join(OUTPUT_PLOTS, 'component_characteristics')
OUTPUT_TEMPORAL     = os.path.join(OUTPUT_CHAR, 'temporal')
OUTPUT_SPATIAL      = os.path.join(OUTPUT_CHAR, 'spatial')
# CSV raw-data for the per-component flow-distance figure (kept out of the figure folder).
OUTPUT_SPATIAL_DIST_RAW = os.path.join(OUTPUT_SPATIAL, 'component_distance_raw_data')
# Socioeconomic: per-component ACS median household income bar figure + raw CSV.
OUTPUT_SOCIO        = os.path.join(OUTPUT_CHAR, 'socioeconomic')
OUTPUT_SOCIO_RAW    = os.path.join(OUTPUT_SOCIO, 'component_income_raw_data')
OUTPUT_FUNC         = os.path.join(OUTPUT_CHAR, 'func')
# CSV raw-data for the func figures, each in its own subfolder so the tables stay
# out of the figure folder; the folder name says which figure the CSV backs.
OUTPUT_FUNC_HM_RAW  = os.path.join(OUTPUT_FUNC, 'heatmap_od_functionality_raw_data')
OUTPUT_FUNC_ENT_RAW = os.path.join(OUTPUT_FUNC, 'hist_function_entropy_raw_data')
OUTPUT_FUNC_VS_TEMP = os.path.join(OUTPUT_CHAR, 'func_vs_temporal')
# Per-function stacks of per-component line figures (one figure per functional
# category, components ordered by combined functional share): the full-window W
# timelines and the disaster r(d) resilience curves.  These describe individual
# components, so they live under component_characteristics/resilience/.
OUTPUT_CHAR_RESIL   = os.path.join(OUTPUT_CHAR, 'resilience')

# Resilience correlation has two subfolders.  func_vs_resilience holds the
# functional-share correlation heatmap, the top-pair scatter, and the resilience
# REGRESSION figures.  temporal_vs_resilience holds the temporal-feature figures
# (weekday_ratio heatmap + peak-slot / peak-period bar charts).  The per-function
# line stacks (W timelines + r(d) resilience curves) now live under
# component_characteristics/resilience/ (OUTPUT_CHAR_RESIL).
OUTPUT_RESIL         = os.path.join(OUTPUT_PLOTS, 'resilience_corr')
OUTPUT_FUNC_VS_RESIL = os.path.join(OUTPUT_RESIL, 'func_vs_resilience')
# The resilience-correlation heatmap lives in a per-method subfolder
# (heatmap_resilience_corr_<spearman|pearson>) so each method + the lambda sweep
# collect separately.  Per-method path = f'{OUTPUT_RESIL_CORR_HM_BASE}_{method}'.
OUTPUT_RESIL_CORR_HM_BASE = os.path.join(OUTPUT_FUNC_VS_RESIL, 'heatmap_resilience_corr')
# Resilience REGRESSION (prediction) outputs ('spearman' = RANK regression, 'pearson' =
# RAW-value regression; see RESIL_REG_METHODS).  INTRA-city (within-unit) LOO scatters + CSVs
# live in func_vs_resilience/intra_city_loss_reg_<method>/; INTER-city (cross-city) LOO
# scatters, R² matrices and pairwise heatmaps live under cross_city_resi_pred/<label>/
# (label = the method's paired folder, see CROSS_CITY_METHOD_STD).
# (Paths built inline at each write site.)
OUTPUT_TL_BY_FUNC    = os.path.join(OUTPUT_CHAR_RESIL, 'line_component_timeline_by_func')
OUTPUT_RC_BY_FUNC    = os.path.join(OUTPUT_CHAR_RESIL, 'line_component_resilience_curves_by_func')
OUTPUT_TEMP_VS_RESIL = os.path.join(OUTPUT_RESIL, 'temporal_vs_resilience')

# ── STEP 4 parameters — pooled severity-vs-resilience ───────────────────────────

# Disaster (Saffir-Simpson arrival intensity) vs resilience scatter, all city-events
# pooled (not split into train/test).
OUTPUT_DISASTER_VS_RESIL = os.path.join(OUTPUT_RESIL, 'disaster_vs_resilience')

# ── STEP 5 parameters — the cross-city feature tables ───────────────────────────

# The held-out (test-role) city-event in every cross-city LOO fold is re-decomposed at
# this fixed k; train units keep their own n_behaviors.  Module-level so the tuner
# (tune_nmf_optuna) mirrors the same test-role convention.
K_LOO_TEST = 10

# ── STEP 6 parameters — cross-city prediction ───────────────────────────────────

# Cross-city resilience prediction reconstructed to the CITY level (predicted vs
# ground-truth cum_loss per city-event).  Only for pearson + multi_city_std.
OUTPUT_CROSS_CITY_RESI_PRED = os.path.join(OUTPUT_PLOTS, 'cross_city_resi_pred')

# Cross-city transfer split — the ONE knob you control.  Lists of city-event codes
# (e.g. 'BR_Ida', 'FM_Ian'), which are the per-unit `code`/tag below.  The cross-city
# step goes ONLY pooled(train) -> each test unit.  If train and test are the SAME set
# it becomes a pooled leave-one-component-out instead.  A unit in neither list is
# still decomposed/characterized but excluded from the cross-city step.  Both sides
# are flexible.  None -> the cross-city step is skipped (with a warning).
CROSS_CITY_SPLIT = {'train': ['FM_Ian', 'WM_Dorian', 'WM_Isaias', 'LC_Laura'],
                    'test':  ['BR_Ida']}

# Cross-city TARGET standardization is PAIRED one-to-one with the method — each method
# produces exactly ONE version, written to cross_city_resi_pred/<label>/:
#   'spearman' -> 'within_unit'  (single-city std) -> cross_city_pred_rank/: ranks are
#       computed within each unit, so only the level-robust within-unit standardization
#       is semantically consistent (a pooled z-score of per-unit ranks would fake a
#       cross-city level).
#   'pearson'  -> 'pooled_train' (multi-city std) -> cross_city_pred_raw_value/: raw
#       values standardized on the pooled TRAINING units keep the absolute level, which
#       the raw-value transfer, the LEVEL / POOLED features and the city-level
#       reconstruction (resi_pred) all rely on.
# Modes are cross_city_resilience's target_std values (see its docstring).  The pairing
# is ENFORCED: analysis_cross_city / analysis_cross_city_pairs raise on a mismatching
# explicit target_std and resolve target_std=None / subdir=None from this dict.
CROSS_CITY_METHOD_STD = {
    'spearman': ('within_unit', 'cross_city_pred_rank'),
    'pearson':  ('pooled_train', 'cross_city_pred_raw_value'),
}

# LEVEL features: per-event CONSTANT covariates.  Two are carried: the Saffir-Simpson
# arrival intensity `hurricane_intensity` (8-level scale 1=Extratropical .. 3=Trop.Storm,
# 4=Cat1 .. 8=Cat5) and `evac_level`, the BG-population-weighted mean of the HEvOD 3-level
# ordinal evacuation strength (0 none / 1 voluntary / 2 mandatory) per city-event.  evac_level
# varies across same-city events (WM_Dorian 0.81 vs WM_Isaias 0.02) whereas the static POI /
# income covariates do not, so it carries a per-event severity signal those cannot; it cut the
# component-wise (M1) city cum_loss MAE 0.499 -> 0.270 (sandbox-validated 2026-07-08).  Within-
# unit standardization would zero a constant column, so both are standardized on the POOLED
# TRAIN instead, and ONLY enter the model in the 'pooled_train' (multi-city) mode.  In
# 'within_unit' (single-city) mode they are not used (single-city results are unchanged).
# The level covariates feed the COMPONENT-level path only (M1, kNN(sigma), the LOO-R² and
# pairwise cross-city analyses); the city-level cosine-kNN reconstruction and the decomposition-
# free baseline do NOT take evac_level — over their low-dim cosine vector it is collinear with
# intensity and reshuffled the n=5 neighbors unhelpfully (sandbox: both worsened).  See
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
# component-wise (M1) city MAE 0.953 -> 0.499 (sandbox-validated 2026-07-08).  In
# 'within_unit' (single-city) mode these are left within-unit.  See cross_city_resilience.
POOLED_FEATURE_COLS = ([f'func_{c}' for c in SF_CATEGORIES]
                       + ['mean_distance', 'median_income_combined'])

# Cross-city predictor for the resilience regression: 'ridge' (global linear, can
# extrapolate) or 'cosine_knn' (Nadaraya-Watson with a cosine kernel = similarity-
# weighted mean of the training components' target; bounded to the training range).
# Applies to ALL cross-city outputs (LOO scatters/matrix, pairwise, city-level pred).
CROSS_CITY_MODEL = 'cosine_knn'


# ── STEP 1 — global land-use classification ─────────────────────────────────────

def _global_landuse_classification(units):
    """The pipeline's ONE land-use classification, shared by every step (the technical
    notes call the recipe "config C"), so `func_<cat>` means the same thing in every
    city.  Pools the raw-SLD score mass over ALL units' block groups (transductive;
    land-use features only, no resilience labels), computes the global IWF weights at
    GLOBAL_IWF_SCALE, then classifies each city's block groups with those SAME weights.
    Returns (iwf_vector, {code: classified landuse DataFrame with share_<cat> +
    dominant_category}, {code: aggr_id -> category lookup}).
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
        lookups[code] = category_lookup_from_landuse(lu)
    return iwf, landuse_by_code, lookups


# ── STEP 3 — within-city analysis helpers (one per analysis block) ──────────────

def _lambda_tag(lambda_ctx):
    """Filename-safe context-strength tag, e.g. 'lambda0', 'lambda0.1', or
    'baseline' when context-aware NMF is off (lambda_ctx is None)."""
    return f"lambda{lambda_ctx:g}" if lambda_ctx is not None else "baseline"


def analysis_component_signature(W, n_nor, n_dis, first_day_normal,
                                 first_day_disaster, tag):
    """Component temporal and spatial characteristics.  Plots the full-window W
    heatmap and the per-component timeline (normal blue, buffer amber, disaster
    red, black dashed line at landfall).  n_nor marks the end of the clean
    normal columns and n_dis the disaster start, so [n_nor, n_dis) is the
    buffer."""
    os.makedirs(OUTPUT_TEMPORAL, exist_ok=True)
    vis_heatmap_temporal_signature(
        W, first_day=first_day_normal, show_days=True,
        slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
        output_dir=OUTPUT_TEMPORAL, tag=tag,
    )
    vis_line_nmf_component_timeline(
        W[:n_nor], W[n_dis:], W_buffer=W[n_nor:n_dis],
        first_day_normal=first_day_normal, first_day_disaster=first_day_disaster,
        slots_per_day=SLOTS_ACTIVE,
        output_dir=OUTPUT_TEMPORAL, tag=tag,
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

    cat_lookup = category_lookup_from_landuse(landuse)
    M, retained = build_od_function_matrix(H, mapping, cat_lookup, AXIS_CATEGORIES)
    print(f"  {label}: O×D flow retained in-category per component: "
          + ", ".join(f"[{i}]={r:.2f}" for i, r in enumerate(retained)))
    vis_heatmap_od_function(
        M, AXIS_CATEGORIES, weights=weights, ncols=3,
        save_path=os.path.join(OUTPUT_FUNC, f'heatmap_od_functionality{tag}.png'),
    )
    # Per-component functional dimensions — the 12 from/to shares, one row per
    # component (index), for inspecting the raw values behind the heatmap.
    os.makedirs(OUTPUT_FUNC_HM_RAW, exist_ok=True)
    functional_features(M, AXIS_CATEGORIES).to_csv(
        os.path.join(OUTPUT_FUNC_HM_RAW, f'component_functionality{tag}.csv'))

    # Functional-entropy distribution across components.  entropy_from / _to are
    # the Shannon entropy of each component's outflow (row sums) and inflow
    # (column sums) of the SAME heatmap M (categories include 'Mix'); lower =
    # more functionally concentrated.  λ is in the filename for cross-strength
    # comparison.
    lambda_tag = _lambda_tag(lambda_ctx)
    ent = component_function_entropy(M)
    os.makedirs(OUTPUT_FUNC_ENT_RAW, exist_ok=True)
    ent.to_csv(os.path.join(OUTPUT_FUNC_ENT_RAW,
                            f'component_function_entropy_{lambda_tag}{tag}.csv'))
    vis_hist_function_entropy(
        ent, lambda_ctx=lambda_ctx,
        max_entropy=float(np.log(len(AXIS_CATEGORIES))),
        title=f'{label}: functional entropy across components',
        save_path=os.path.join(OUTPUT_FUNC,
                               f'hist_function_entropy_{lambda_tag}{tag}.png'),
    )
    return M


def analysis_time_function_corr(feats, tag):
    """Time × function correlation block.  Computes Spearman between TIME_COLS
    and FUNC_COLS across one city's components, then plots the heatmap, the
    top-pair scatter, and the categorical peak-slot bar chart."""
    os.makedirs(OUTPUT_FUNC_VS_TEMP, exist_ok=True)
    rho, pval = time_function_correlation(feats, TIME_COLS, FUNC_COLS)
    # Split-cell heatmap, same style as the resilience block.  No single-cell
    # time columns here (weekday_ratio is the row).
    vis_heatmap_corr_split(
        rho, pval, time_cols=[], categories=SF_CATEGORIES,
        save_path=os.path.join(OUTPUT_FUNC_VS_TEMP, f'heatmap_time_function_corr{tag}.png'),
    )
    pairs = rho.abs().stack().sort_values(ascending=False).index[:4].tolist()
    vis_scatter_component_features(
        feats, pairs,
        save_path=os.path.join(OUTPUT_FUNC_VS_TEMP, f'scatter_time_function_top_pairs{tag}.png'),
    )
    vis_bar_function_by_peakslot(
        feats, SF_CATEGORIES,
        save_path=os.path.join(OUTPUT_FUNC_VS_TEMP, f'bar_function_by_peakslot{tag}.png'),
    )
    vis_bar_function_by_peakslot(
        feats, SF_CATEGORIES,
        group_col='peak_period', label_col='peak_period_label',
        save_path=os.path.join(OUTPUT_FUNC_VS_TEMP, f'bar_function_by_peakperiod{tag}.png'),
    )


def analysis_resilience_corr(feats, tag, lambda_ctx=None, methods=RESIL_CORR_METHODS):
    """Resilience correlation block.  Draws the RES_COLS × feature heatmap once per
    correlation method in `methods`, each into its own
    heatmap_resilience_corr_<method> subfolder of func_vs_resilience (cell =
    share_from/share_to split per category, plus single-cell mean_distance /
    median_income on the RIGHT; lambda_ctx tags the filename).  The top-pair scatter
    and the temporal_vs_resilience figures (weekday_ratio heatmap, peak-slot /
    peak-period bars) use the FIRST method (Spearman by default).  The per-function
    curve stacks are drawn separately by analysis_func_ordered_lines."""
    os.makedirs(OUTPUT_FUNC_VS_RESIL, exist_ok=True)
    lambda_tag = _lambda_tag(lambda_ctx)
    feat_cols = TIME_COLS + FUNC_COLS + EXTRA_CORR_COLS

    # Per correlation method, into its own heatmap_resilience_corr_<method> folder:
    #   (1) the functional-share heatmap — each category cell stacks share_from
    #       (upper) / share_to (lower); mean_distance and median_income are single
    #       cells on the RIGHT; rows coloured by their own max |rho|;
    #   (2) the top-|corr| pair scatter — RANK-transformed for spearman, RAW values
    #       for pearson, so each scatter matches its heatmap's correlation.
    rho_by_method = {}
    for method in methods:
        rho, pval = time_function_correlation(feats, RES_COLS, feat_cols, method=method)
        rho_by_method[method] = (rho, pval)
        out_dir = f'{OUTPUT_RESIL_CORR_HM_BASE}_{method}'
        vis_heatmap_corr_split(
            rho, pval, time_cols=[], categories=SF_CATEGORIES,
            extra_cols=EXTRA_CORR_COLS,
            save_path=os.path.join(out_dir,
                                   f'heatmap_resilience_corr_{lambda_tag}{tag}.png'),
        )
        pairs = rho.abs().stack().sort_values(ascending=False).index[:4].tolist()
        vis_scatter_component_features(
            feats, pairs, rank=(method == 'spearman'),
            save_path=os.path.join(
                out_dir, f'scatter_resilience_top_pairs_{lambda_tag}{tag}.png'),
        )

    # The temporal_vs_resilience figures (weekday_ratio heatmap + peak bars) use
    # the first method's correlation (Spearman by default).
    rho, pval = rho_by_method[methods[0]]
    os.makedirs(OUTPUT_TEMP_VS_RESIL, exist_ok=True)
    # The weekday_ratio column of the same correlation, on its own as a single
    # temporal feature against the resilience metrics.
    vis_heatmap_corr_split(
        rho, pval, time_cols=TIME_COLS, categories=[],
        save_path=os.path.join(OUTPUT_TEMP_VS_RESIL, f'heatmap_weekday_ratio_resilience{tag}.png'),
    )
    vis_bar_resilience_by_peakslot(
        feats, RES_COLS,
        save_path=os.path.join(OUTPUT_TEMP_VS_RESIL, f'bar_resilience_by_peakslot{tag}.png'),
    )
    vis_bar_resilience_by_peakslot(
        feats, RES_COLS,
        group_col='peak_period', label_col='peak_period_label',
        save_path=os.path.join(OUTPUT_TEMP_VS_RESIL, f'bar_resilience_by_peakperiod{tag}.png'),
    )


def analysis_resilience_linear(feats, tag, lambda_ctx=None,
                               merge_func_directions=True,
                               methods=RESIL_REG_METHODS):
    """Multivariate Ridge counterpart to analysis_resilience_corr.  The functional
    predictors are either the 6 MERGED per-function shares (func_c = share_from_c +
    share_to_c, total share of the component's flow touching function c) when
    merge_func_directions is True, or the 12 directional shares (share_from_<c>,
    share_to_<c>) as-is when False; either way the component's loading-weighted
    mean flow distance (mean_distance) is appended as the final SPATIAL feature
    (7 predictors merged, 13 unmerged).  One ridge regression per resilience metric
    predicts that metric from those features, so each standardized coefficient is
    that feature's effect CONTROLLING FOR the others (vs the correlation block's
    one-at-a-time).

    Run ONCE PER METHOD in `methods`, each into its own intra_city_loss_reg_<method>
    subfolder of func_vs_resilience: 'spearman' RANK-regresses (features AND the
    metric rank-transformed, a multivariate PARTIAL Spearman matching the heatmap),
    'pearson' regresses the RAW standardized values.  Each method writes the
    per-metric LOO predicted-vs-actual scatter (each panel titled with its
    leave-one-out R² and PASS/FAIL) plus the coefficient + LOO-summary CSVs (under
    that folder's raw_data/).  Returns {method: summary} (the per-metric LOO R²
    table per method) for the cross-city within-city comparison.  lambda_ctx tags
    every filename."""
    lambda_tag = _lambda_tag(lambda_ctx)

    if merge_func_directions:
        # Merge same-category outflow + inflow into ONE feature per function:
        # func_c = share_from_c + share_to_c (total share of flow touching c).
        # 6 functional features instead of 12 — fewer parameters for the tiny n.
        feats_m = feats.copy()
        func_cols = [f'func_{c}' for c in SF_CATEGORIES]
        for c in SF_CATEGORIES:
            feats_m[f'func_{c}'] = feats[f'share_from_{c}'] + feats[f'share_to_{c}']
    else:
        # Keep outflow and inflow as the 12 separate directional shares.
        feats_m = feats
        func_cols = FUNC_COLS
    # mean_distance (the spatial feature, already in feats) is the final predictor.
    feature_cols = func_cols + ['mean_distance']

    summaries = {}
    for method in methods:
        rank = (method == 'spearman')   # spearman = rank regression, pearson = raw
        # Intra-city (within-unit) regression outputs: one folder per method directly under
        # func_vs_resilience (intra_city_loss_reg_<method>), separate from the cross-city
        # (inter-city) outputs which now live under cross_city_resi_pred/<label>/.
        out_dir = os.path.join(OUTPUT_FUNC_VS_RESIL, f'intra_city_loss_reg_{method}')
        raw_dir = os.path.join(out_dir, 'raw_data')
        os.makedirs(out_dir, exist_ok=True)
        coef_mat, summary, pred_data = run_city_resilience_linear(
            feats_m, RES_COLS, feature_cols, rank=rank)
        summaries[method] = summary

        # Per-city regression diagnostic: LOO predicted vs actual, one panel/metric.
        vis_scatter_reg_pred(
            pred_data, summary, RES_COLS,
            unit=('std rank' if rank else 'std value'),
            title=f'{feats["city"].iloc[0]}: regression predicted vs actual '
                  f'(LOO, {method})',
            save_path=os.path.join(out_dir,
                                   f'scatter_resilience_reg_{lambda_tag}{tag}.png'),
        )
        os.makedirs(raw_dir, exist_ok=True)
        coef_mat.to_csv(os.path.join(raw_dir, f'linear_coef_{lambda_tag}{tag}.csv'))
        summary.to_csv(os.path.join(raw_dir,
                                    f'linear_loo_summary_{lambda_tag}{tag}.csv'))
    return summaries


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
                            cat_lookup):
    """Decompose this city at `k` and assemble ONLY the cross-city predictor/target
    columns (share_from/to_<c>, mean_distance/std_distance, median_income_combined,
    RES_COLS) — no figures, no within-city side effects.  Used by the leave-one-city-
    event-out cross-city loop to build the HELD-OUT (test) unit's feats at a fixed k
    (k=10), independent of the unit's own n_behaviors used elsewhere.  l1_reg is
    the unit's CITY_EVENTS value (only k differs).  `cat_lookup` is the unit's block-group
    -> category map from the STEP-1 global classification."""
    W, H, weights = decompose_city(X_all, k, l1_reg=cfg['l1_reg'], fit_time_cols=fit_time_cols)
    M, _ = build_od_function_matrix(H, mapping, cat_lookup, AXIS_CATEGORIES)
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
        functional_features(M, AXIS_CATEGORIES),
        spatial_features(H, distances),
        socioeconomic_features(H, income_array, name='median_income_combined'),
        resilience_features(W, n_nor, cfg['first_day_normal'], SLOTS_ACTIVE, n_dis=n_dis),
        recovery_rate_features(W, n_nor, cfg['first_day_normal'], SLOTS_ACTIVE,
                               n_dis=n_dis),
    ], axis=1)
    feats.insert(0, 'city', cfg['label'])
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
    return feats


# ── STEP 6 — cross-city prediction analyses ─────────────────────────────────────

def analysis_cross_city(feats_by_city, loo_by_city, lambda_ctx=None,
                        merge_func_directions=True,
                        methods=RESIL_REG_METHODS, split=None,
                        target_std=None, subdir=None, level_feature_cols=(),
                        model='ridge', pooled_feature_cols=()):
    """Cross-city resilience generalisation driven by an explicit train/test `split`
    of city-event codes.  Each method's target standardization is HARD-PAIRED via
    CROSS_CITY_METHOD_STD (spearman->within_unit, pearson->pooled_train): target_std=None
    resolves the pairing per method, an explicit mismatching value raises ValueError.
    Two modes by `split`:
      - train & test DISJOINT  -> TRANSFER: pooled(train) -> predict each test unit.
      - train & test the SAME set -> POOLED-LOO across the pool's components.
    Per method into cross_city_resi_pred/<subdir>/ (subdir=None -> the method's paired
    label, e.g. cross_city_pred_raw_value): writes the predicted-vs-actual scatter AND
    its raw data (raw_data/cross_city_scatter_<col>_*_raw.csv), and RETURNS
    {method -> r2_table} so the caller can aggregate (e.g. the LOO matrix).
    split=None -> skipped (returns {}).  loo_by_city is accepted but unused.
    lambda_ctx tags the filenames."""
    # Defensive guard: reject a method/std combo OR an output label that contradicts the
    # pairing BEFORE anything is computed or written (all methods validated up front; the
    # subdir check stops results being written under the other method's folder).
    for m in methods:
        expected_std, expected_label = CROSS_CITY_METHOD_STD[m]
        if target_std is not None and target_std != expected_std:
            raise ValueError(
                f"analysis_cross_city: method '{m}' is paired with target_std "
                f"'{expected_std}' (CROSS_CITY_METHOD_STD); got '{target_std}'")
        if subdir is not None and subdir != expected_label:
            raise ValueError(
                f"analysis_cross_city: method '{m}' writes to subdir "
                f"'{expected_label}' (CROSS_CITY_METHOD_STD); got '{subdir}'")
    if split is None:
        print("  [cross-city] split is None -> skipping the cross-city step.")
        return {}
    lambda_tag = _lambda_tag(lambda_ctx)

    # Per-unit predictor tables (same recipe as analysis_resilience_linear).  The
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
    feature_cols = func_cols + ['mean_distance', 'median_income_combined']

    train = [c for c in split.get('train', []) if c in cities]
    test  = [c for c in split.get('test', []) if c in cities]
    pooled_loo = bool(train) and (set(train) == set(test))
    train_lbl  = '+'.join(train)

    results = {}
    for method in methods:
        rank = (method == 'spearman')   # spearman = rank transfer, pearson = raw
        std_mode, std_label = CROSS_CITY_METHOD_STD[method]   # paired (guard passed)
        out_dir = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED,
                               std_label if subdir is None else subdir)
        raw_dir = os.path.join(out_dir, 'raw_data')

        r2_table, pred, groups = cross_city_resilience(
            cities, RES_COLS, feature_cols, rank=rank, split=split,
            target_std=std_mode, level_feature_cols=level_feature_cols, model=model,
            pooled_feature_cols=pooled_feature_cols)
        if r2_table.shape[1] == 0:
            continue                                    # skipped/empty (warning already emitted)
        results[method] = r2_table

        # Predicted-vs-actual scatter per output column (test unit, or the pool).
        for col, pred_data in pred.items():
            col_summary = pd.DataFrame({
                'loo_r2': {m: r2_table.loc[m, col] for m in RES_COLS},
                'passed': {m: (bool(r2_table.loc[m, col] > 0)
                               if pd.notna(r2_table.loc[m, col]) else False)
                           for m in RES_COLS},
                'status': {m: ('ok' if pred_data[m] is not None else 'insufficient_data')
                           for m in RES_COLS},
            })
            if pooled_loo:
                title = f'Pooled-LOO ({method}): {train_lbl}'
                fname = f'cross_city_scatter_pooledLOO_{lambda_tag}.png'
            else:
                title = f'Cross-city ({method}): train [{train_lbl}] -> test {col}'
                fname = f'cross_city_scatter_{col}_{lambda_tag}.png'
            vis_scatter_reg_pred(
                pred_data, col_summary, RES_COLS, r2_label='test R²',
                unit=('std rank' if rank else 'std value'),
                groups=groups.get(col),         # colours pooled-LOO points by city-event
                title=title,
                save_path=os.path.join(out_dir, fname))

            # Raw data behind this scatter (predicted vs actual per metric), so each
            # point stays recoverable for later analysis.  groups[col] is keyed by
            # METRIC -> per-point unit array (non-None only in pooled-LOO mode).
            col_groups = groups.get(col) or {}
            rows = []
            for m in RES_COLS:
                pm = pred_data.get(m)
                if pm is None:
                    continue
                y_true, y_pred, comp_index = pm
                grp_m = col_groups.get(m)
                for i in range(len(y_true)):
                    rows.append({'metric': m, 'comp_index': comp_index[i],
                                 'y_true': float(y_true[i]), 'y_pred': float(y_pred[i]),
                                 'unit': (grp_m[i] if grp_m is not None else col)})
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

    Per held-out city, THREE reconstructions of the same per-component cosine-kNN prediction:
      * knn  — legacy: un-standardize each component with the fold's (mu, sigma) then
               weight_normal-average.  sigma ~ the large per-component cum_loss spread over-
               amplifies the mis-scaled prediction (good ranking, poor calibration).
      * M1   — keep the standardized component predictions, aggregate to a city score, then
               rescale to day-equivalents with a 1-parameter city-scale VARIANCE MATCH whose
               scale is learned NESTED-LOO on the other training cities (never the held city).
               Component-level prediction + city-scale calibration; avoids the per-component
               sigma.  (n=5-fragile: the scale is estimated from 4 outlier-driven cities.)
      * city — a city-level cosine-kNN over the weight_normal-aggregated feature vector,
               predicting GT directly (bounded to the training cities' GT; no sigma).
    Ground truth = cum_loss of the city's TOTAL activity curve (no decomposition).  A
    DECOMPOSITION-FREE BASELINE (cosine-weighted average of the OTHER cities' GT over
    [ss_intensity, city-wide POI shares, static city-wide mean BG income]) is emitted for
    reference.  The static income keeps the baseline decomposition-free (no flow weighting)
    and is the fair income-augmented control; it barely moves the baseline (its cross-city
    spread is tiny — two same-city events even share the value), which is what shows the
    income gain of the decompose methods comes from the flow-weighted WITHIN-city component
    spread, not from income as a city covariate.  Saves a grouped bar
    (GT / kNN / M1 / city-kNN / baseline) + raw CSV
    (cum_loss_gt/_pred_knn/_pred_m1/_pred_city/_baseline) under cross_city_resi_pred/."""
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

    def _unit_rows(M):
        n = np.linalg.norm(M, axis=-1, keepdims=True)
        return M / np.where(n > 0, n, 1.0)

    # City-level aggregated feature vector = weight_normal-weighted mean of the predictor
    # columns (config-C func + mean_distance) plus the per-event-constant intensity — used
    # by method (3) (city-level kNN) and computable for any unit's role table.
    def _city_vec(feats):
        w = feats['weight_normal'].to_numpy(dtype=float); sw = w.sum()
        w = w / sw if sw > 0 else np.full(len(w), 1.0 / len(w))
        v = [float(w @ feats[col].to_numpy(dtype=float)) for col in feature_cols]
        return np.array(v + [float(feats['hurricane_intensity'].iloc[0])])

    # One held-out fold of the COMPONENT-level cosine-kNN transfer.  Returns both the
    # sigma-un-standardized weight_normal city prediction (`pcity`, the legacy kNN number)
    # and the aggregated STANDARDIZED score (`scity` = Σ w·ŷ_std / Σ w) that method M1
    # rescales at city scale.  Reused for the outer folds AND the nested-LOO inner folds.
    def _fold(held, rest):
        fold = {held: test_merged[held]}
        fold.update({c: train_merged[c] for c in rest})
        _, pred, _ = cross_city_resilience(
            fold, RES_COLS, feature_cols, rank=False,
            split={'train': rest, 'test': [held]}, target_std=target_std,
            level_feature_cols=LEVEL_FEATURE_COLS, model='cosine_knn',
            pooled_feature_cols=POOLED_FEATURE_COLS)
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
        scity = float((w * ypred_std).sum() / wsum) if wsum > 0 else float(ypred_std.mean())
        return dict(pcity=pcity, scity=scity)

    # Three reconstructions of the component predictions into a city day-equivalent:
    #   knn  — legacy: un-standardize each component with the fold's (mu, sigma) then
    #          weight_normal-average.  sigma ~ the huge per-component cum_loss spread over-
    #          amplifies the (mis-scaled) prediction -> good ranking, poor calibration.
    #   M1   — component-level: keep the standardized predictions, aggregate to `scity`, then
    #          rescale to day-equivalents with a 1-parameter city-scale VARIANCE MATCH whose
    #          scale is learned nested-LOO on the other training cities (never the held city).
    #   city — method (3): a city-level cosine-kNN over the aggregated `_city_vec`, predicting
    #          GT directly (bounded to the training cities' GT; no per-component sigma).
    # NOTE: the M1 calibration is estimated from only |rest| (=4) cities, so it is unstable
    # (its std is outlier-driven) — the numbers beat the baseline but stay n=5-fragile.
    rows = []
    for held in codes:
        rest = [c for c in codes if c != held]
        outer = _fold(held, rest)
        if outer is None:
            print(f"  [cross_city_resi_pred] {held}: no cum_loss prediction; skipping.")
            continue
        pred_knn = outer['pcity']

        # M1: nested LOO on the training cities -> (scity, GT) pairs, fit the 1-param scale.
        s_in, gt_in = [], []
        for s2 in rest:
            inn = _fold(s2, [c for c in rest if c != s2])
            if inn is not None:
                s_in.append(inn['scity']); gt_in.append(gt[s2])
        if len(s_in) >= 2:
            s_in, gt_in = np.array(s_in), np.array(gt_in)
            sd_s = s_in.std()
            pred_m1 = float(gt_in.mean() + (outer['scity'] - s_in.mean())
                            * (gt_in.std() / (sd_s if sd_s > 0 else 1.0)))
        else:
            pred_m1 = pred_knn

        # method (3): city-level cosine-kNN over aggregated features -> GT.
        Xtr = np.vstack([_city_vec(train_merged[c]) for c in rest])
        ytr = np.array([gt[c] for c in rest])
        mu_x, sd_x = Xtr.mean(axis=0), Xtr.std(axis=0); sd_x[sd_x == 0] = 1.0
        Ztr = (Xtr - mu_x) / sd_x; zte = (_city_vec(test_merged[held]) - mu_x) / sd_x
        sims3 = np.clip(_unit_rows(Ztr) @ _unit_rows(zte), 0.0, None)
        pred_city = (float(sims3 @ ytr / sims3.sum()) if sims3.sum() > 0 else float(ytr.mean()))

        # Baseline (no decomposition): cosine-similarity-weighted average of the OTHER
        # cities' GT cum_loss over city features [ss_intensity, city-wide POI shares].
        sims = np.array([max(_cos(zfeat[held], zfeat[t]), 0.0) for t in rest])
        gts = np.array([gt[t] for t in rest])
        base = (float((sims * gts).sum() / sims.sum()) if sims.sum() > 0 else float(gts.mean()))

        rows.append({'code': held, 'cum_loss_gt': gt[held],
                     'cum_loss_pred_knn': pred_knn, 'cum_loss_pred_m1': pred_m1,
                     'cum_loss_pred_city': pred_city, 'cum_loss_baseline': base})

    if not rows:
        print("  [cross_city_resi_pred] no usable city-events; skipping.")
        return
    res = pd.DataFrame(rows).set_index('code')
    res = res.reindex([c for c in codes if c in res.index])
    raw_dir = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED, 'raw_data')
    os.makedirs(raw_dir, exist_ok=True)
    res.to_csv(os.path.join(raw_dir, 'cross_city_resi_pred.csv'))
    # The bar omits the legacy kNN(σ) prediction (still in the CSV); it shows the two
    # decomposition-based predictions and the decomposition-free baseline.
    vis_bar_cross_city_resi_pred(
        res, save_path=os.path.join(OUTPUT_CROSS_CITY_RESI_PRED,
                                    'bar_cross_city_resi_pred.png'),
        pred_cols=(('Decompose + component-wise', 'cum_loss_pred_m1', 'D+comp'),
                   ('Decompose + city-wise', 'cum_loss_pred_city', 'D+city')),
        baseline_label='City-wise',
        title='City-level cum_loss: ground truth vs decomposition-based predictions '
              'vs city-wise baseline (global-IWF features, pearson)')
    print(f"  [cross_city_resi_pred] -> {OUTPUT_CROSS_CITY_RESI_PRED} "
          f"({len(res)} city-events)")


def analysis_cross_city_pairs(feats_train, feats_test, codes, method='pearson',
                              target_std=None, subdir=None, level_feature_cols=(),
                              model='ridge', pooled_feature_cols=()):
    """Pairwise single-train -> single-test cross-city transfer for cum_loss.
    For each ORDERED pair (train=a, test=b): train a uses its own-k feats
    (feats_train), test b uses the k=10 feats (feats_test); the diagonal (a==b) is
    a's within-unit leave-one-component-out at k=10.  The target standardization is
    HARD-PAIRED to the method via CROSS_CITY_METHOD_STD (target_std=None resolves it,
    a mismatching explicit value raises ValueError).  Saves the |codes|x|codes| matrix
    (rows=train, cols=test, value = cum_loss R²) as cross_city_pair_heatmap (.png +
    raw_data/.csv) under cross_city_resi_pred/<subdir>/ (subdir=None -> the method's
    paired label).  Returns the matrix."""
    # Defensive guard: same pairing rule as analysis_cross_city (std AND output label),
    # checked up front.
    expected_std, std_label = CROSS_CITY_METHOD_STD[method]
    if target_std is not None and target_std != expected_std:
        raise ValueError(
            f"analysis_cross_city_pairs: method '{method}' is paired with target_std "
            f"'{expected_std}' (CROSS_CITY_METHOD_STD); got '{target_std}'")
    if subdir is not None and subdir != std_label:
        raise ValueError(
            f"analysis_cross_city_pairs: method '{method}' writes to subdir "
            f"'{std_label}' (CROSS_CITY_METHOD_STD); got '{subdir}'")
    target_std = expected_std
    subdir = std_label
    rank = (method == 'spearman')
    feature_cols = ([f'func_{c}' for c in SF_CATEGORIES]
                    + ['mean_distance', 'median_income_combined'])

    def _merge(feats):
        m = feats.copy()
        for c in SF_CATEGORIES:
            m[f'func_{c}'] = feats[f'share_from_{c}'] + feats[f'share_to_{c}']
        return m

    train_merged = {c: _merge(feats_train[c]) for c in codes}
    test_merged  = {c: _merge(feats_test[c]) for c in codes}

    mat = pd.DataFrame(index=codes, columns=codes, dtype=float)   # rows=train, cols=test
    for a in codes:
        for b in codes:
            feats_pair = ({a: test_merged[a]} if a == b           # self: within-a LOO at k=10
                          else {a: train_merged[a], b: test_merged[b]})
            r2_t, _, _ = cross_city_resilience(
                feats_pair, RES_COLS, feature_cols, rank=rank,
                split={'train': [a], 'test': [b]}, target_std=target_std,
                level_feature_cols=level_feature_cols, model=model,
                pooled_feature_cols=pooled_feature_cols)
            col = 'pooled_LOO' if a == b else b
            mat.loc[a, b] = (float(r2_t.loc['cum_loss', col])
                             if col in r2_t.columns and pd.notna(r2_t.loc['cum_loss', col])
                             else np.nan)

    out_dir = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED, subdir)
    raw_dir = os.path.join(out_dir, 'raw_data')
    os.makedirs(raw_dir, exist_ok=True)
    mat.to_csv(os.path.join(raw_dir, 'cross_city_pair_heatmap.csv'))
    vis_heatmap_pair_r2(
        mat, title=f'Pairwise cross-city cum_loss R² ({method}; test k=10)',
        xlabel='test city-event', ylabel='train city-event',
        save_path=os.path.join(out_dir, 'cross_city_pair_heatmap.png'))
    print(f"  [pairwise] cum_loss R² heatmap ({method}) -> "
          f"{os.path.join(out_dir, 'cross_city_pair_heatmap.png')}")
    return mat


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── STEP 1 — Load every unit and classify land use once ────────────────────

    # Load every unit's geometry first, because the global land-use
    # classification below pools over ALL units' block groups.
    units = {}   # code -> everything the later steps need, filled incrementally
    for cfg in CITY_EVENTS:
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
            # Kept for the cross-city LOO, which re-decomposes the held-out unit at k=10.
            X_all=X_all, fit_time_cols=fit_time_cols)

    # ── STEP 3 — Within-city analyses: characterise each unit's components ──

    feats_by_city = {}   # short code -> per-component feats, for the cross-city test
    loo_by_city   = {}   # short code -> within-city LOO R² (for the comparison)

    # Each unit's per-component feats + within-unit LOO are keyed by its city-event
    # code ('BR_Ida', 'WM_Dorian', ...), which is what CROSS_CITY_SPLIT lists.
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
            functional_features(M, AXIS_CATEGORIES),

            # Spatial: mean_distance, std_distance (reads H + OD centroid distances).
            spatial_features(H, distances),

            # Socioeconomic: median_income per endpoint mode (loading-weighted, ACS).
            socio,
            
            # Resilience reads W[n_dis:] against a baseline built from W[:n_nor].
            # The buffer columns [n_nor, n_dis) feed neither the resilience features
            # nor the curves.
            resilience_features(W, n_nor, first_day_nor, SLOTS_ACTIVE, n_dis=n_dis),

            # Exp-recovery rate lambda, the second resilience target (higher =
            # faster recovery; NaN when the component never fell below baseline).
            recovery_rate_features(W, n_nor, first_day_nor, SLOTS_ACTIVE,
                                   n_dis=n_dis),
        ], axis=1)
        feats.insert(0, 'city', label)
        feats.insert(1, 'weight', weights)

        analysis_time_function_corr(feats, tag)
        
        analysis_resilience_corr(feats, tag, lambda_ctx=ctx_lambda)
        
        curves = resilience_curves(W, n_nor, first_day_nor, SLOTS_ACTIVE, n_dis=n_dis)
        analysis_func_ordered_lines(W, n_nor, n_dis, first_day_nor, first_day_dis,
                                    curves, feats, tag)

        reg_summaries = analysis_resilience_linear(
            feats, tag, lambda_ctx=ctx_lambda,
            merge_func_directions=MERGE_FUNC_DIRECTIONS)

        # Per-event-constant LEVEL features, added AFTER the within-city analyses (which
        # select specific column groups) so only the cross-city step sees them (used only
        # by the pooled_train mode): Saffir-Simpson arrival intensity and evacuation strength.
        feats = feats.copy()
        _reg = next(c for c in CITY_EVENTS if c['code'] == code)
        feats['hurricane_intensity'] = _reg['ss_intensity']
        feats['evac_level'] = _reg['evac_level']
        # Stash for the cross-city test after the loop, keyed by the city-event code.
        feats_by_city[code] = feats
        loo_by_city[code]   = {m: s['loo_r2'] for m, s in reg_summaries.items()}

    # ── STEP 4 — Disaster (arrival intensity) vs resilience, ALL city-events pooled ──
    # One scatter panel per RES_COLS metric: x = the component's event-level
    # Saffir-Simpson arrival intensity (ss_intensity), y = the raw metric; points
    # coloured by city-event.  Not split into train/test.
    if feats_by_city:
        print("\n── Disaster intensity vs resilience scatter (all components) ──")
        pooled = pd.concat(
            [feats_by_city[c][['hurricane_intensity'] + list(RES_COLS)].assign(code=c)
             for c in feats_by_city], ignore_index=True)
        os.makedirs(OUTPUT_DISASTER_VS_RESIL, exist_ok=True)
        vis_scatter_intensity_resilience(
            pooled, 'hurricane_intensity', RES_COLS, group_col='code',
            title='Saffir-Simpson arrival intensity vs resilience '
                  '(all components, all city-events)',
            save_path=os.path.join(OUTPUT_DISASTER_VS_RESIL,
                                   'scatter_intensity_vs_resilience.png'))
        pooled.to_csv(os.path.join(OUTPUT_DISASTER_VS_RESIL,
                                   'intensity_vs_resilience_raw.csv'), index=False)
        print(f"  -> {OUTPUT_DISASTER_VS_RESIL}")

    # ── STEP 5 — Build the cross-city feature tables (two roles per unit) ──
    # Each unit takes a turn as the held-out test (pooled others -> predict it).  The
    # HELD-OUT (test) unit is re-decomposed at k=K_LOO_TEST=10 (module constant, shared
    # with the tuner; train units keep their own n_behaviors via feats_by_city).
    # Each method runs its PAIRED target standardization (CROSS_CITY_METHOD_STD:
    # spearman->single-city, pearson->multi-city), into cross_city_resi_pred/<label>/.
    if feats_by_city:
        all_codes = list(units)
        # Build the cross-city feats with the SAME global classification as STEP 1
        # (cc_lookups).  Both roles are rebuilt here — train role at each unit's
        # own k, test role at k=K_LOO_TEST — because the two roles decompose the
        # same matrix at different k (the within-city feats keep the unit's k only).
        print(f"\n── Cross-city: building feats (train @ per-unit k, test @ k={K_LOO_TEST}) ──")
        cc_train, cc_test = {}, {}
        for code in all_codes:
            u = units[code]
            print(f"  [cross-city feats] {u['label']} [{code}]")
            cc_train[code] = _build_cross_city_feats(
                u['cfg'], u['X_all'], u['n_nor'], u['n_dis'], u['mapping'],
                u['gdf'], u['fit_time_cols'], u['cfg']['n_behaviors'], cat_lookup=cc_lookups[code])
            cc_test[code] = _build_cross_city_feats(
                u['cfg'], u['X_all'], u['n_nor'], u['n_dis'], u['mapping'],
                u['gdf'], u['fit_time_cols'], K_LOO_TEST, cat_lookup=cc_lookups[code])

        # ── STEP 6 — Cross-city prediction (LOO transfer, pairwise, reconstruction) ──
        for method, (std_mode, std_label) in CROSS_CITY_METHOD_STD.items():
            print(f"\n── Cross-city LOO [{method}: {std_label}] ({len(all_codes)} folds) ──")
            loo_r2 = {}   # held_code -> r2 Series over RES_COLS
            for held in all_codes:
                rest = [c for c in all_codes if c != held]
                fold_feats = {held: cc_test[held]}
                fold_feats.update({c: cc_train[c] for c in rest})
                print(f"  fold: test [{held}] <- train [{'+'.join(rest)}]")
                res = analysis_cross_city(
                    fold_feats, loo_by_city, lambda_ctx=None,
                    merge_func_directions=MERGE_FUNC_DIRECTIONS,
                    methods=[method],
                    split={'train': rest, 'test': [held]},
                    target_std=std_mode, subdir=std_label,
                    level_feature_cols=LEVEL_FEATURE_COLS, model=CROSS_CITY_MODEL,
                    pooled_feature_cols=POOLED_FEATURE_COLS)
                r2_table = (res or {}).get(method)
                if r2_table is not None and held in r2_table.columns:
                    loo_r2[held] = r2_table[held]

            # Combined LOO R² matrix (rows = metrics, cols = held-out unit).
            if loo_r2:
                mat = pd.DataFrame(loo_r2).reindex(index=RES_COLS, columns=all_codes)
                raw_dir = os.path.join(OUTPUT_CROSS_CITY_RESI_PRED, std_label,
                                       'raw_data')
                os.makedirs(raw_dir, exist_ok=True)
                out_csv = os.path.join(raw_dir, 'loo_cross_city_r2_baseline.csv')
                mat.to_csv(out_csv)
                print(f"  [{method}] LOO cross-city R² matrix -> {out_csv}")

            # Pairwise single-train -> single-test cum_loss R² (ordered pairs);
            # test at k=10, train at its own k; diagonal = within-unit LOO.
            print(f"── Cross-city pairwise cum_loss heatmap [{method}: {std_label}] ──")
            analysis_cross_city_pairs(cc_train, cc_test, all_codes,
                                      method=method,
                                      target_std=std_mode, subdir=std_label,
                                      level_feature_cols=LEVEL_FEATURE_COLS,
                                      model=CROSS_CITY_MODEL,
                                      pooled_feature_cols=POOLED_FEATURE_COLS)

        # City-level reconstruction of the cross-city cum_loss prediction vs ground
        # truth (runs only for pearson + multi_city_std; guarded inside the function).
        print("\n── Cross-city city-level cum_loss: prediction vs ground truth ──")
        analysis_cross_city_resi_pred(cc_train, cc_test, units, all_codes, global_iwf)


if __name__ == '__main__':
    main()
