"""
NMF decomposition of mobility patterns (paper-style, single factorization).

Cities: #1 = Baton Rouge (Hurricane Ida, 2021), #2 = Fort Myers (Hurricane Ian, 2022).

Idea
----
For each city, the pre-disaster ("normal") period, a short pre-landfall buffer
(preparation/evacuation days, excluded from baselines/features) and the disaster
period are kept in ONE trailing window and a SINGLE non-negative matrix
factorisation is run on that whole window, producing two factors:

    W : temporal factor  [time × k]   one activity-over-time curve per component
    H : spatial factor    [k × OD]    one OD-flow map      per component

(k = N_BEHAVIORS components; time = DAYS_WINDOW × SLOTS_ACTIVE active slots.)
Because both periods share one factorisation, the k components keep the same
identity across the normal/disaster boundary.
The boundary is used only to colour the W timeline, never inside the factorisation.

Each city's graphs are trimmed so its disaster sits at the window end,
giving both cities a matched "landfall + 14-day recovery" window.

Pipeline
--------
1. Load both cities' graphs and trim to the analysis window.
2. Take the last DAYS_WINDOW days, drop overnight slots (periodic_trim), and
   flatten to one flow matrix per city.
3. Optionally drop low-activity OD pairs (threshold ∝ FILTER_FACTOR; 0 = keep all).
4. Run one NMF per city → temporal factor W, spatial factor H.
5. Save to outputs/nmf/:
     - component_characteristics/  per-component characteristics, by type:
         temporal/          signature heatmap + timeline (temporal factor W)
         spatial/           per-component OD arc maps (H), per-city subfolders
         func/              per-component O×D functional cross-tabs
         func_vs_temporal/  time × function correlation figures
     - resilience_corr/
         func_vs_resilience/      functional-share heatmap, top-pair scatter,
           and two per-function line stacks (one figure per category,
           components ordered by combined functional share):
             line_component_timeline_by_func/           full-window W timeline
             line_component_resilience_curves_by_func/  disaster r(d) curves
         temporal_vs_resilience/  weekday_ratio heatmap + peak-slot /
           peak-period bar charts

Geometry files are mandatory.  Loading raises FileNotFoundError when a city's
geo CSV is absent.

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
from utils_pattern_analysis.graph_io import (
    load_graphs_trimmed, build_distance_array, build_income_array,
)
from utils_pattern_analysis.decomposition import (
    h_slice_to_od_matrix, select_segment_columns,
)
from utils_pattern_analysis.nmf_pipeline import (
    build_city_matrices, decompose_city, decompose_city_context,
)
from utils_pattern_analysis.visualization import (
    vis_heatmap_temporal_signature, vis_map_od_flow,
    vis_line_nmf_component_timeline, vis_heatmap_od_function,
    vis_scatter_component_features,
    vis_bar_function_by_peakslot, vis_line_resilience_curves,
    vis_bar_resilience_by_peakslot, vis_heatmap_corr_split,
    vis_hist_function_entropy,
    vis_bar_component_distance, vis_bar_component_income, vis_scatter_reg_pred,
)
from utils_pattern_analysis.space_function import (
    category_lookup_from_landuse, build_od_function_matrix,
)
from utils_pattern_analysis.component_features import (
    temporal_features, functional_features, time_function_correlation,
    resilience_features, resilience_curves, component_function_entropy,
    spatial_features, socioeconomic_features,
)
from utils_pattern_analysis.ml_resilience import (
    run_city_resilience_linear, cross_city_resilience,
)
from utils_data_processing.build_graphs import load_city_geo
from utils_data_processing.fetch_sld_landuse import (
    ensure_city_landuse_raw, load_city_landuse, CATEGORIES as SF_CATEGORIES,
)
from utils_data_processing.fetch_acs_income import (
    ensure_city_income_raw, load_city_income, ACS_DATA_DIR,
)

import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42,        # editable TrueType text in PDF exports
    'svg.fonttype': 'none',    # editable text in SVG exports
})


# ── Configuration ─────────────────────────────────────────────────────────────

# Derived from config.  Never hardcode the interval here.
_INTERVAL_HOURS = 24 // SLOT_PER_DAY   # 3 for 3h data and 2 for 2h data

# Trailing window taken from the end of the graph sequence
# (normal + buffer + disaster).  The 5-day pre-landfall buffer isolates
# preparation and evacuation behaviour from the clean normal baseline
# (FM showed a +46% Saturday surge on Sep 24 and −19% on Tue Sep 27).
# Buffer slots stay in the NMF input but are excluded from the temporal
# features and from the resilience baselines and curves.
DAYS_WINDOW_BR          = 33   # 13 normal + 5 buffer + 15 disaster (Ida Aug 29 + 14 recovery)
DAYS_BUFFER_BR          = 5    # Aug 24–28 (mild pre-Ida prep, Fri Aug 27 ≈ +14%)
DAYS_DISASTER_IN_WIN_BR = 15
DAYS_WINDOW_FM          = 33   # 13 normal + 5 buffer + 15 disaster (Ian Sep 28 + 14 recovery)
DAYS_BUFFER_FM          = 5    # Sep 23–27 (Ian prep and evacuation contamination)
DAYS_DISASTER_IN_WIN_FM = 15

N_BEHAVIORS_BR = 11
N_BEHAVIORS_FM = 13
L1_REG_BR      = 0.5
L1_REG_FM      = 1.1

# Context-aware NMF (shared-factor / Chen et al. 2018, IEEE MIPR).  When on, the
# spatial factor H is regularised toward a per-flow POI feature built from the
# endpoints' TF-IWF land-use shares (the same classification the O×D func block
# uses).  The spatial unit is the flow: each flow gets ONE feature combining its
# origin and destination.  False -> the exact sklearn baseline (H unchanged).
# See docs/technical_notes §3.2/§3.3 for the formulation and the magnitude
# (co-scaling) reasoning behind FLOW_FEATURE_MODE / LAMBDA_CTX.
CONTEXT_AWARE_BR  = False     # Controls the solver: even when LAMBDA_CTX is 0 the solver still changes if this is on
CONTEXT_AWARE_FM  = False
LAMBDA_CTX_BR     = 0.1       # relative context weight (auto-scaled by ‖X‖²/‖Y‖²)
LAMBDA_CTX_FM     = 0.1
FLOW_FEATURE_MODE = 'sum'    # 'outer' = C² joint O×D type, 'sum' = C-dim combined

# Which time segments FIT the NMF basis H.  The full window is ALWAYS projected
# onto that basis, so W/H shapes and n_nor/n_dis are unchanged downstream.  All
# three segments = the original single-fit pipeline (exact).  A smaller fit
# segment has fewer time samples (normal=104, disaster=120, buffer=40, full=264
# rows at 8 slots/day) — revisit N_BEHAVIORS (and possibly L1_REG) when slicing
# and watch the near-zero-weight component count in the projection diagnostics.
NMF_FIT_SEGMENTS_BR = ('normal', 'buffer')
NMF_FIT_SEGMENTS_FM = ('normal', 'buffer')

FILTER_FACTOR_BR = 1
FILTER_FACTOR_FM = 2

# Resilience regression: when True, each function's
# outflow + inflow shares are merged into ONE feature
# when False the 12 directional shares are used as-is.
MERGE_FUNC_DIRECTIONS = True

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

# Baton Rouge runs Apr 15 (Thu) → Sep 16 and is trimmed to Sep 12 (Ida + 14,
# see BR_ANALYSIS_DAYS).  The last 33 days span Aug 11 (Wed) → Sep 12, with
# normal Aug 11–23, buffer Aug 24–28, and disaster Aug 29 (Sun, Ida) → Sep 12.
FIRST_DAY_BR_NORMAL   = 'Wednesday'   # First day of BR's 33-day window (Aug 11)
FIRST_DAY_BR_DISASTER = 'Sunday'      # First day of BR's disaster portion (Aug 29, Ida)

# Fort Myers is trimmed to Aug 30 – Oct 12 2022 (44 days, see FM_ANALYSIS_DAYS).
# The trailing 33-day window spans Sep 10 – Oct 12, with normal Sep 10–22,
# buffer Sep 23–27, and disaster Sep 28 – Oct 12 (Ian + 14 days recovery).
FIRST_DAY_FM_NORMAL   = 'Saturday'    # Sep 10 2022 (normal-segment start)
FIRST_DAY_FM_DISASTER = 'Wednesday'   # Sep 28 2022 (disaster start, Ian landfall)

# ── City-event registry — one entry per (city, disaster).  Add a city = add an entry. ──
# This is the single source of truth for per-unit params and SUPERSEDES the per-city
# BR_/FM_ constants above (DAYS_WINDOW_*, N_BEHAVIORS_*, CONTEXT_AWARE_*, FILTER_FACTOR_*,
# FIRST_DAY_* ...), which main() no longer reads.  Only per-unit params live here; the rest
# of the pipeline is global.  Paths are block_group (AGG_LEVEL is the global resolution
# switch; SLD land-use only supports block_group).  Window = the trailing `window` days of
# the trimmed graph = (window-buffer-disaster) normal + `buffer` pre-landfall + `disaster`
# (landfall + recovery); `analysis_days` trims so landfall+14 sits at the sequence end.
# `key` is the SLD/income filename prefix (city-level, shared across a city's events).
# first_day_* are the weekday of the normal-window start / of the landfall day.
_WIN, _BUF, _DIS = 33, 5, 15          # shared 33-day window: 13 normal + 5 buffer + 15 disaster
CITY_EVENTS = [
    dict(code='BR_Ida', label='Baton Rouge', key='Baton_Rouge',
         graph='data/Baton_Rouge_Ida_2021_graph_intersection.pkl',
         geo={'block_group': 'data/Baton_Rouge_block_group_geo.csv'},
         analysis_days=151, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=11, l1_reg=0.5, filter_factor=0,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Wednesday', first_day_disaster='Sunday'),
    dict(code='FM_Ian', label='Fort Myers', key='Fort_Myers',
         graph='data/Fort_Myers_Ian_2022_graph_intersection.pkl',
         geo={'block_group': 'data/Fort_Myers_block_group_geo.csv'},
         analysis_days=44, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=11, l1_reg=0.13, filter_factor=0,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Saturday', first_day_disaster='Wednesday'),
    # New city-events: 87-day (L-56 -> L+30) block_group graphs, landfall at day 56
    # -> analysis_days = 71.  Per-city k / l1 are TUNED (cum_loss LOO-CV, 2026-06-29,
    # study nmf_loocv_cum_loss); filter_factor=0 = no OD filtering.
    dict(code='WM_Dorian', label='Wilmington', key='Wilmington',
         graph='data/Wilmington_Dorian_2019_graph_intersection.pkl',
         geo={'block_group': 'data/Wilmington_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=9, l1_reg=1.626, filter_factor=0,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Sunday', first_day_disaster='Thursday'),
    dict(code='WM_Isaias', label='Wilmington', key='Wilmington',
         graph='data/Wilmington_Isaias_2020_graph_intersection.pkl',
         geo={'block_group': 'data/Wilmington_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=9, l1_reg=0.315, filter_factor=0,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Thursday', first_day_disaster='Monday'),
    dict(code='PC_Sally', label='Panama City', key='Panama_City',
         graph='data/Panama_City_Sally_2020_graph_intersection.pkl',
         geo={'block_group': 'data/Panama_City_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=8, l1_reg=1.469, filter_factor=0,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Saturday', first_day_disaster='Wednesday'),
    dict(code='LC_Laura', label='Lake Charles', key='Lake_Charles',
         graph='data/Lake_Charles_Laura_2020_graph_intersection.pkl',
         geo={'block_group': 'data/Lake_Charles_block_group_geo.csv'},
         analysis_days=71, window=_WIN, buffer=_BUF, disaster=_DIS,
         n_behaviors=10, l1_reg=0.549, filter_factor=0,
         context_aware=False, lambda_ctx=0.1, fit_segments=('normal', 'buffer'),
         first_day_normal='Sunday', first_day_disaster='Thursday'),
]

OUTPUT_PLOTS = os.path.join(OUTPUT_DIR, 'nmf')

# All per-component characteristics live under component_characteristics, one
# subfolder per characteristic type.
#   temporal/          signature heatmap + timeline (temporal factor W)
#   spatial/           per-city OD arc-map subfolders (spatial factor H)
#   func/              per-component O×D functional cross-tabs
#   func_vs_temporal/  time × function correlation figures
OUTPUT_CHAR         = os.path.join(OUTPUT_PLOTS, 'component_characteristics')
OUTPUT_TEMPORAL     = os.path.join(OUTPUT_CHAR, 'temporal')
OUTPUT_SPATIAL      = os.path.join(OUTPUT_CHAR, 'spatial')
# CSV raw-data for the per-component flow-distance figure (kept out of the figure folder).
OUTPUT_SPATIAL_DIST_RAW = os.path.join(OUTPUT_SPATIAL, 'component_distance_raw_data')
OUTPUT_NMF_BR       = os.path.join(OUTPUT_SPATIAL, 'component_spatial_characteristics_br')
OUTPUT_NMF_FM       = os.path.join(OUTPUT_SPATIAL, 'component_spatial_characteristics_fm')
# Socioeconomic: per-component ACS median household income bar figure + raw CSV.
OUTPUT_SOCIO        = os.path.join(OUTPUT_CHAR, 'socioeconomic')
OUTPUT_SOCIO_RAW    = os.path.join(OUTPUT_SOCIO, 'component_income_raw_data')
OUTPUT_FUNC         = os.path.join(OUTPUT_CHAR, 'func')
# CSV raw-data for the func figures, each in its own subfolder so the tables stay
# out of the figure folder; the folder name says which figure the CSV backs.
OUTPUT_FUNC_HM_RAW  = os.path.join(OUTPUT_FUNC, 'heatmap_od_functionality_raw_data')
OUTPUT_FUNC_ENT_RAW = os.path.join(OUTPUT_FUNC, 'hist_function_entropy_raw_data')
OUTPUT_FUNC_VS_TEMP = os.path.join(OUTPUT_CHAR, 'func_vs_temporal')

# Per-city block-group space-function data (EPA Smart Location Database).
SPACE_FUNCTION_DIR = os.path.join(DATA_DIR, 'space_function')

# On-the-fly classification knobs for the space-function data (TF-IWF
# reweighting, see utils_data_processing/fetch_sld_landuse.py).
LANDUSE_WEIGHTING          = 'tf_iwf'   # 'tf_iwf' down-weights ubiquitous residential, 'raw_share' does not
LANDUSE_IWF_SCALE          = 1.52       # IWF exponent (tuned: cum_loss LOO-CV 2026-06-29).  Higher pulls labels toward rare functions
LANDUSE_RESIDENTIAL_WEIGHT = 1.0        # Housing-unit to job equivalence; 1.0 treats one housing unit as one job
LANDUSE_DOMINANT_THRESHOLD = 0.4        # Top-category share needed for a label, otherwise 'Mix'.
                                        # Lower gives fewer Mix

# Cross-tab axes are the functional categories plus 'Mix'.  Unknown and
# unmatched endpoints are dropped.  Order defines the heatmap rows (origin)
# and columns (destination).
AXIS_CATEGORIES = list(SF_CATEGORIES) + ['Mix']

# Resilience correlation has two subfolders.  func_vs_resilience holds the
# functional-share heatmap, the top-pair scatter, and two per-function stacks
# of line figures (one figure per category, components ordered by combined
# functional share): the W timelines and the resilience curves.
# temporal_vs_resilience holds the temporal-feature figures (weekday_ratio
# heatmap + peak-slot / peak-period bar charts).
OUTPUT_RESIL         = os.path.join(OUTPUT_PLOTS, 'resilience_corr')
OUTPUT_FUNC_VS_RESIL = os.path.join(OUTPUT_RESIL, 'func_vs_resilience')
# The resilience-correlation heatmap lives in a per-method subfolder
# (heatmap_resilience_corr_<spearman|pearson>) so each method + the lambda sweep
# collect separately.  Per-method path = f'{OUTPUT_RESIL_CORR_HM_BASE}_{method}'.
OUTPUT_RESIL_CORR_HM_BASE = os.path.join(OUTPUT_FUNC_VS_RESIL, 'heatmap_resilience_corr')
# Resilience REGRESSION (prediction) figures + raw CSVs, split by method into
# resilience_reg_<method> subfolders (figures at top, CSVs under raw_data/), exactly
# like the correlation heatmap above.  'spearman' = RANK regression, 'pearson' =
# RAW-value regression (see RESIL_REG_METHODS).  Per-method dir = f'{BASE}_{method}'.
OUTPUT_RESIL_REG_BASE = os.path.join(OUTPUT_FUNC_VS_RESIL, 'resilience_reg')
OUTPUT_TL_BY_FUNC    = os.path.join(OUTPUT_FUNC_VS_RESIL, 'line_component_timeline_by_func')
OUTPUT_RC_BY_FUNC    = os.path.join(OUTPUT_FUNC_VS_RESIL, 'line_component_resilience_curves_by_func')
OUTPUT_TEMP_VS_RESIL = os.path.join(OUTPUT_RESIL, 'temporal_vs_resilience')

# ── Per-component feature columns used by the correlation blocks ──────────────

# Temporal features are computed from the pre-disaster part of W only (see
# utils_pattern_analysis/component_features.py).
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

# Functional features come from the O×D cross-tab with Mix and Unknown dropped
# and the six categories renormalised, split by flow direction.  Shares are
# full row/column sums, so same-function diagonal flow counts on both sides.
#   share_from_<cat> — outflow side, the fraction departing from function <cat>
#   share_to_<cat>   — inflow side, the fraction arriving at function <cat>
FUNC_COLS = ([f'share_from_{c}' for c in SF_CATEGORIES]
             + [f'share_to_{c}' for c in SF_CATEGORIES])

# Resilience features are computed from the relative-activity curve r, where
# r(d) = the component's daily total on disaster day d divided by its
# weekday/weekend-matched pre-disaster baseline, 3-day smoothed (see
# component_features.resilience_features).  r = 1 means the normal level.
# Every metric reads HIGHER = WORSE; resilience_features returns all five
# directly (early_collapse and recovery_deficit are inverted inside it).
#   drop_depth       — 1 − min(r), how deep activity fell.  1 means a total
#                      stop and negative means it rose above baseline
#                      (a disaster-emergent pattern)
#   early_collapse   — (disaster days − 1) − lowest_day, how soon the curve
#                      bottomed out.  14 means the lowest point was on
#                      landfall day, 0 means it was on the final day
#   recovery_day     — days until r reaches 0.9 AND stays there (the clock
#                      resets on any later dip below 0.9, so oscillating
#                      recoveries are not credited early).  0 means never
#                      below the threshold.  A value equal to the disaster
#                      window length means not recovered within the window
#                      (right-censored, kept numeric so the hardest-hit
#                      components stay in the rank correlation)
#   recovery_deficit — 1 − mean r over the last 3 disaster days, the shortfall
#                      still left at the window end.  0 means fully recovered
#                      and negative means overshoot above normal
#   cum_loss         — Σ max(0, 1−r), the resilience-triangle area in
#                      day-equivalents.  Smaller means more resilient
RES_COLS = ['drop_depth', 'early_collapse', 'recovery_day', 'recovery_deficit',
            'cum_loss']

# Extra single-value features appended (one cell each, no from/to split) to the
# RIGHT of the functional resilience-correlation heatmap: the component's
# loading-weighted mean flow distance (spatial) and the median household income
# for each endpoint mode (socioeconomic).  All must already be columns of `feats`.
EXTRA_CORR_COLS = ['mean_distance'] + [f'median_income_{m}' for m in INCOME_ENDPOINT_MODES]

# Correlation method(s) for the resilience-vs-feature heatmap.  Each method is
# drawn into its own heatmap_resilience_corr_<method> subfolder.  'spearman' =
# rank correlation (robust to non-linearity); 'pearson' = linear on raw values.
RESIL_CORR_METHODS = ['spearman', 'pearson']

# Resilience REGRESSION (prediction) method(s) -> resilience_reg_<method> subfolders.
# 'spearman' RANK-regresses (features AND the metric rank-transformed before the
# Ridge fit -> a multivariate partial Spearman, matching the correlation heatmap);
# 'pearson' regresses the RAW standardized values.  The rank flag fed to the Ridge
# helpers is simply (method == 'spearman'); the scatter axis unit follows suit.
RESIL_REG_METHODS = ['spearman', 'pearson']

# Cross-city transfer split — the ONE knob you control.  Lists of city-event codes
# (e.g. 'BR_Ida', 'FM_Ian'), which are the per-unit `code`/tag below.  The cross-city
# step goes ONLY pooled(train) -> each test unit.  If train and test are the SAME set
# it becomes a pooled leave-one-component-out instead.  A unit in neither list is
# still decomposed/characterized but excluded from the cross-city step.  Both sides
# are flexible.  None -> the cross-city step is skipped (with a warning).
CROSS_CITY_SPLIT = {'train': ['FM_Ian', 'WM_Dorian', 'WM_Isaias', 'PC_Sally', 'LC_Laura'],
                    'test':  ['BR_Ida']}


# ── Analysis helpers (one per analysis block, called once per city) ──────────

def _lambda_tag(lambda_ctx):
    """Filename-safe context-strength tag, e.g. 'lambda0', 'lambda0.1', or
    'baseline' when context-aware NMF is off (lambda_ctx is None)."""
    return f"lambda{lambda_ctx:g}" if lambda_ctx is not None else "baseline"


def load_landuse_for_context(label, key, gdf):
    """Ensure the raw SLD cache and load the on-the-fly TF-IWF classification for
    the context-aware NMF, using the same LANDUSE_RESIDENTIAL_WEIGHT /
    LANDUSE_WEIGHTING / LANDUSE_IWF_SCALE / LANDUSE_DOMINANT_THRESHOLD knobs as the
    O×D func block.  Returns the landuse DataFrame with `share_<cat>` columns
    keyed by `aggr_id`."""
    os.makedirs(SPACE_FUNCTION_DIR, exist_ok=True)
    raw_csv = os.path.join(SPACE_FUNCTION_DIR, f'{key}_block_group_sld_raw.csv')
    assert ensure_city_landuse_raw(label, gdf['aggr_id'].tolist(), raw_csv) is not None
    return load_city_landuse(
        raw_csv,
        residential_weight=LANDUSE_RESIDENTIAL_WEIGHT,
        weighting=LANDUSE_WEIGHTING,
        iwf_scale=LANDUSE_IWF_SCALE,
        dominant_threshold=LANDUSE_DOMINANT_THRESHOLD,
    )


def analysis_component_signature(W, n_nor, n_dis, first_day_normal,
                                 first_day_disaster, tag,
                                 gdf=None, H=None, mapping=None):
    """Component temporal and spatial characteristics.  Plots the full-window W
    heatmap and the per-component timeline (normal blue, buffer amber, disaster
    red, black dashed line at landfall).  n_nor marks the end of the clean
    normal columns and n_dis the disaster start, so [n_nor, n_dis) is the
    buffer.  Interactive OD arc maps are available below (commented) — pass
    gdf, H and mapping, then uncomment to enable."""
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

    # DISABLED arc-map path: the gdf/H/mapping params and the vis_map_od_flow /
    # h_slice_to_od_matrix imports exist only for this commented block.
    # # Interactive HTML arc maps, one file per spatial component.  The
    # # centroid column from load_city_geo is already EPSG:4326.
    # out_dir = os.path.join(OUTPUT_SPATIAL,
    #                        f'component_spatial_characteristics{tag}')  # = OUTPUT_NMF_BR/_FM
    # os.makedirs(out_dir, exist_ok=True)
    # gdf['lon'] = gdf['centroid'].x
    # gdf['lat'] = gdf['centroid'].y
    # for i in range(H.shape[0]):
    #     vis_map_od_flow(
    #         [h_slice_to_od_matrix(H[i: i+1, :], mapping)],
    #         gdfs=gdf, id_col='aggr_id', min_flow=0.5,
    #         max_line_width=20, alpha_range=(0.05, 0.95), curve_rad=0.3, vmax=5,
    #         save_dir=os.path.join(out_dir, f'component_{i}.html'),
    #     )


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


def analysis_od_function(label, key, tag, gdf, H, mapping, weights,
                         lambda_ctx=None):
    """O×D functionality block.  Ensures the raw SLD cache,
    classifies block groups on the fly, aggregates each component's OD flows
    into an origin×destination functional cross-tab, and saves the heatmap
    grid plus a per-component functional-share CSV.  Also saves the across-
    component entropy distribution of each component's outflow/inflow functional
    mix (the heatmap M marginals); lambda_ctx tags the filename so context-aware
    strengths can be compared.  Returns M [k × C × C]."""
    os.makedirs(SPACE_FUNCTION_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FUNC, exist_ok=True)
    raw_csv = os.path.join(SPACE_FUNCTION_DIR, f'{key}_block_group_sld_raw.csv')
    assert ensure_city_landuse_raw(label, gdf['aggr_id'].tolist(), raw_csv) is not None
    landuse = load_city_landuse(
        raw_csv,
        residential_weight=LANDUSE_RESIDENTIAL_WEIGHT,
        weighting=LANDUSE_WEIGHTING,
        iwf_scale=LANDUSE_IWF_SCALE,
        dominant_threshold=LANDUSE_DOMINANT_THRESHOLD,
    )
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

    Run ONCE PER METHOD in `methods`, each into its own resilience_reg_<method>
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
        out_dir = f'{OUTPUT_RESIL_REG_BASE}_{method}'
        raw_dir = os.path.join(out_dir, 'raw_data')
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


def analysis_cross_city(feats_by_city, loo_by_city, lambda_ctx=None,
                        merge_func_directions=True,
                        methods=RESIL_REG_METHODS, split=None):
    """Cross-city resilience generalisation driven by an explicit train/test `split`
    (CROSS_CITY_SPLIT) of city-event codes.  Each unit is rank/standardized within
    itself (Option A), then pooled; see cross_city_resilience for the two modes:
      - train & test DISJOINT  -> TRANSFER: pooled(train) -> predict each test unit
        (r2 columns = test units; LOO comparison column = that test unit's own
        within-unit LOO).
      - train & test the SAME set -> POOLED-LOO across the pool's components (one
        'pooled_LOO' column; the scatter colours points by city-event).
    Runs once per method (spearman=rank / pearson=raw) into resilience_reg_<method>/.
    split=None -> the step is skipped with a warning.  loo_by_city maps city-event
    code -> {method -> within-unit LOO R² Series}.  lambda_ctx tags the filenames."""
    if split is None:
        print("  [cross-city] CROSS_CITY_SPLIT is None -> skipping the cross-city step.")
        return
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
    feature_cols = func_cols + ['mean_distance']

    train = [c for c in split.get('train', []) if c in cities]
    test  = [c for c in split.get('test', []) if c in cities]
    pooled_loo = bool(train) and (set(train) == set(test))
    train_lbl  = '+'.join(train)

    for method in methods:
        rank = (method == 'spearman')   # spearman = rank transfer, pearson = raw
        out_dir = f'{OUTPUT_RESIL_REG_BASE}_{method}'

        r2_table, pred, groups = cross_city_resilience(
            cities, RES_COLS, feature_cols, rank=rank, split=split)
        if r2_table.shape[1] == 0:
            continue                                    # skipped/empty (warning already emitted)

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Data preprocessing ─────────────────────────────────────────────────────

    # Trimming keeps the FIRST analysis_days days, which drops the late
    # recovery tail and leaves landfall + 14 recovery days at the sequence end
    # for the trailing window taken in build_city_matrices.
    _ALL_SEGMENTS = {'normal', 'buffer', 'disaster'}
    units = {}   # code -> everything the per-unit analysis loop needs
    for cfg in CITY_EVENTS:
        label, code = cfg['label'], cfg['code']
        print()
        print(f"── {label} [{code}] ──")
        gdf = load_city_geo(label, AGG_LEVEL, cfg['geo'])
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
            landuse = load_landuse_for_context(label, cfg['key'], gdf)
            W, H, weights = decompose_city_context(
                X_all, cfg['n_behaviors'], mapping, landuse,
                lambda_ctx=cfg['lambda_ctx'], feature_mode=FLOW_FEATURE_MODE,
                l1_reg=cfg['l1_reg'], fit_time_cols=fit_time_cols)
        else:
            W, H, weights = decompose_city(
                X_all, cfg['n_behaviors'], l1_reg=cfg['l1_reg'], fit_time_cols=fit_time_cols)
        units[code] = dict(
            label=label, key=cfg['key'], tag='_' + code, gdf=gdf, H=H, W=W,
            mapping=mapping, weights=weights, n_nor=n_nor, n_dis=n_dis,
            first_day_nor=cfg['first_day_normal'], first_day_dis=cfg['first_day_disaster'],
            ctx_lambda=(cfg['lambda_ctx'] if cfg['context_aware'] else None))

    # ── Analysis ──

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
        analysis_component_signature(W, n_nor, n_dis, first_day_nor, first_day_dis, tag,
                                     gdf=gdf, H=H, mapping=mapping)
        
        distances = analysis_spatial(label, H, mapping, gdf, weights, tag,
                                     lambda_ctx=ctx_lambda)
        socio = analysis_socioeconomic(label, key, tag, gdf, H, mapping,
                                       weights, lambda_ctx=ctx_lambda)
        M = analysis_od_function(label, key, tag, gdf, H, mapping, weights,
                                 lambda_ctx=ctx_lambda)

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

        # Stash for the cross-city test after the loop, keyed by the city-event code.
        # Keep the per-method LOO R² so the cross-city heatmap's within-unit columns
        # match each method.
        feats_by_city[code] = feats.copy()
        loo_by_city[code]   = {m: s['loo_r2'] for m, s in reg_summaries.items()}

    # ── Cross-city generalisation (explicit train-set -> test-set, CROSS_CITY_SPLIT) ──
    if feats_by_city:
        print("\n── Cross-city resilience test ──")
        analysis_cross_city(feats_by_city, loo_by_city, lambda_ctx=None,
                            merge_func_directions=MERGE_FUNC_DIRECTIONS,
                            split=CROSS_CITY_SPLIT)


if __name__ == '__main__':
    main()
