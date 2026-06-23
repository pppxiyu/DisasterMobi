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

# Derived from config.  Never hardcode the interval here.
_INTERVAL_HOURS = 24 // SLOT_PER_DAY   # 3 for 3h data and 2 for 2h data
from utils_pattern_analysis.graph_io import load_graphs_trimmed, build_distance_array
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
    vis_hist_function_entropy, vis_heatmap_corr_merged,
    vis_bar_component_distance, vis_scatter_reg_pred,
    vis_heatmap_cross_city_r2,
)
from utils_pattern_analysis.space_function import (
    category_lookup_from_landuse, build_od_function_matrix,
)
from utils_pattern_analysis.component_features import (
    temporal_features, functional_features, time_function_correlation,
    resilience_features, resilience_curves, component_function_entropy,
    spatial_features,
)
from utils_pattern_analysis.ml_resilience import (
    run_city_resilience_linear, cross_city_resilience,
)
from utils_data_processing.build_graphs import load_city_geo
from utils_data_processing.fetch_sld_landuse import (
    ensure_city_landuse_raw, load_city_landuse, CATEGORIES as SF_CATEGORIES,
)

import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42,        # editable TrueType text in PDF exports
    'svg.fonttype': 'none',    # editable text in SVG exports
})


# ── Configuration ─────────────────────────────────────────────────────────────

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

N_BEHAVIORS_BR = 10
N_BEHAVIORS_FM = 15
L1_REG_BR      = 0.5
L1_REG_FM      = 0.5

# Context-aware NMF (shared-factor / Chen et al. 2018, IEEE MIPR).  When on, the
# spatial factor H is regularised toward a per-flow POI feature built from the
# endpoints' TF-IWF land-use shares (the same classification the O×D func block
# uses).  The spatial unit is the flow: each flow gets ONE feature combining its
# origin and destination.  False -> the exact sklearn baseline (H unchanged).
# See docs/technical_notes §3.2/§3.3 for the formulation and the magnitude
# (co-scaling) reasoning behind FLOW_FEATURE_MODE / LAMBDA_CTX.
CONTEXT_AWARE_BR  = False     # Controls solver, even if lambada is 0, the solver chages if its on
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

FILTER_FACTOR_BR = 3
FILTER_FACTOR_FM = 1

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
LANDUSE_IWF_SCALE          = 1.0        # IWF exponent.  Higher pulls labels toward rare functions
LANDUSE_RESIDENTIAL_WEIGHT = 1.0        # Housing-unit to job equivalence.  Fix this
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
# The Spearman (linear) heatmap now lives in its own subfolder so the lambda
# sweep collects there; the Ridge heatmap sits directly in func_vs_resilience.
OUTPUT_RESIL_CORR_HM = os.path.join(OUTPUT_FUNC_VS_RESIL, 'heatmap_resilience_corr')
# CSV raw-data for the Ridge heatmap, kept out of the figure folder.
OUTPUT_RESIL_REG_RAW = os.path.join(OUTPUT_FUNC_VS_RESIL, 'heatmap_resilience_reg_raw_data')
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
# Every metric reads HIGHER = WORSE.  recovery_deficit and early_collapse are
# script-level inversions of resilience_features' recovery_level / lowest_day.
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


# ── Analysis helpers (one per analysis block, called once per city) ──────────

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
    temporal / functional / resilience blocks).  Computes each component's
    loading-weighted flow distance (km) from H and the OD centroid distances
    (graph_io.build_distance_array — same definition the distance-decay analysis
    uses), saves a per-component bar figure under component_characteristics/
    spatial/ and the raw values as CSV.  NOT yet fed into the regression — this
    is for inspecting the visualisation first.  lambda_ctx tags the filename
    (H, hence the distances, shift with context strength)."""
    os.makedirs(OUTPUT_SPATIAL, exist_ok=True)
    ltag = _lambda_tag(lambda_ctx)
    distances = build_distance_array(mapping, gdf)
    sf = spatial_features(H, distances)
    vis_bar_component_distance(
        sf, weights=weights,
        title=f'{label}: per-component flow distance',
        save_path=os.path.join(OUTPUT_SPATIAL,
                               f'component_distance_{ltag}{tag}.png'),
    )
    os.makedirs(OUTPUT_SPATIAL_DIST_RAW, exist_ok=True)
    sf.to_csv(os.path.join(OUTPUT_SPATIAL_DIST_RAW,
                           f'spatial_features_{ltag}{tag}.csv'))
    return sf


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
    ltag = _lambda_tag(lambda_ctx)
    ent = component_function_entropy(M)
    os.makedirs(OUTPUT_FUNC_ENT_RAW, exist_ok=True)
    ent.to_csv(os.path.join(OUTPUT_FUNC_ENT_RAW,
                            f'component_function_entropy_{ltag}{tag}.csv'))
    vis_hist_function_entropy(
        ent, lambda_ctx=lambda_ctx,
        max_entropy=float(np.log(len(AXIS_CATEGORIES))),
        title=f'{label}: functional entropy across components',
        save_path=os.path.join(OUTPUT_FUNC,
                               f'hist_function_entropy_{ltag}{tag}.png'),
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


def analysis_resilience_corr(feats, tag, lambda_ctx=None):
    """Resilience correlation block.  Splits its figures across two subfolders
    of resilience_corr — func_vs_resilience gets the Spearman heatmap between
    RES_COLS and the functional shares plus the top-pair scatter;
    temporal_vs_resilience gets the weekday_ratio heatmap and the peak-slot /
    peak-period bar charts.  The per-function curve stacks are drawn separately
    by analysis_func_ordered_lines.  lambda_ctx tags the functional-share heatmap
    filename so context-aware strengths can be compared."""
    os.makedirs(OUTPUT_FUNC_VS_RESIL, exist_ok=True)
    rho, pval = time_function_correlation(feats, RES_COLS, TIME_COLS + FUNC_COLS)
    # Functional-share heatmap — the 6 category columns only, each cell stacking
    # share_from (upper) and share_to (lower).  weekday_ratio is split off into
    # its own heatmap below.  Rows are coloured by their own max |rho|.
    ltag = _lambda_tag(lambda_ctx)
    vis_heatmap_corr_split(
        rho, pval, time_cols=[], categories=SF_CATEGORIES,
        save_path=os.path.join(OUTPUT_RESIL_CORR_HM,
                               f'heatmap_resilience_corr_{ltag}{tag}.png'),
    )

    pairs = rho.abs().stack().sort_values(ascending=False).index[:4].tolist()
    vis_scatter_component_features(
        feats, pairs,
        save_path=os.path.join(OUTPUT_FUNC_VS_RESIL, f'scatter_resilience_top_pairs{tag}.png'),
    )
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


def analysis_resilience_linear(feats, tag, lambda_ctx=None):
    """Multivariate Ridge counterpart to analysis_resilience_corr (ADDED
    ALONGSIDE; the Spearman block is unchanged, only relocated to its own
    subfolder).  Each function's OUTFLOW and INFLOW shares are MERGED into one
    feature (func_c = share_from_c + share_to_c = total share of the component's
    flow touching function c), so there are 6 functional features (not 12), PLUS
    the component's loading-weighted mean flow distance (mean_distance) as a 7th
    SPATIAL feature.  One RANK ridge regression per resilience metric (features
    AND the metric are rank-transformed, so it is Spearman-aligned — a
    multivariate PARTIAL Spearman) predicts that metric's rank from those 7, so
    each standardized coefficient is that feature's rank-effect CONTROLLING FOR
    the others (vs the Spearman block's one-at-a-time).  Outputs the per-metric
    LOO predicted-vs-actual scatter (each panel titled with its leave-one-out R²
    and PASS/FAIL) plus the coefficient + LOO-summary CSVs; the coefficient
    heatmap is no longer drawn.  lambda_ctx tags every filename."""
    os.makedirs(OUTPUT_FUNC_VS_RESIL, exist_ok=True)
    ltag = _lambda_tag(lambda_ctx)

    # Merge same-category outflow + inflow into ONE feature per function:
    # func_c = share_from_c + share_to_c (total share of flow touching function c).
    # 6 functional features instead of 12 — fewer parameters for the tiny n —
    # plus mean_distance (the spatial feature, already in feats) as the 7th predictor.
    feats_m = feats.copy()
    merged_cols = [f'func_{c}' for c in SF_CATEGORIES]
    for c in SF_CATEGORIES:
        feats_m[f'func_{c}'] = feats[f'share_from_{c}'] + feats[f'share_to_{c}']
    feature_cols = merged_cols + ['mean_distance']
    coef_mat, summary, pred_data = run_city_resilience_linear(
        feats_m, RES_COLS, feature_cols)

    # Per-city regression diagnostic: LOO predicted vs actual, one panel/metric.
    vis_scatter_reg_pred(
        pred_data, summary, RES_COLS,
        title=f'{feats["city"].iloc[0]}: regression predicted vs actual (LOO)',
        save_path=os.path.join(OUTPUT_FUNC_VS_RESIL,
                               f'scatter_resilience_reg_{ltag}{tag}.png'),
    )
    os.makedirs(OUTPUT_RESIL_REG_RAW, exist_ok=True)
    coef_mat.to_csv(os.path.join(OUTPUT_RESIL_REG_RAW,
                                 f'linear_coef_{ltag}{tag}.csv'))
    summary.to_csv(os.path.join(OUTPUT_RESIL_REG_RAW,
                                f'linear_loo_summary_{ltag}{tag}.csv'))
    return summary


def analysis_cross_city(feats_by_city, loo_by_city, lambda_ctx=None):
    """Cross-city (leave-one-city-out) generalisation of the resilience
    regression: train on one city, TEST on the other (option A — each city
    rank-transformed + standardized within itself, coefficients transferred, so
    absolute-level differences between the two disasters are normalised away).
    With 2 cities this is a single hard probe (both directions), not a stable
    estimate.  Saves a cross-city test-R² heatmap (with within-city LOO columns
    for comparison) + a predicted-vs-actual scatter per direction, in
    func_vs_resilience.  lambda_ctx tags the filenames."""
    os.makedirs(OUTPUT_FUNC_VS_RESIL, exist_ok=True)
    ltag = _lambda_tag(lambda_ctx)

    # Merge func features per city (same recipe as analysis_resilience_linear).
    feature_cols = [f'func_{c}' for c in SF_CATEGORIES] + ['mean_distance']
    merged = {}
    for code, fc in feats_by_city.items():
        fm = fc.copy()
        for c in SF_CATEGORIES:
            fm[f'func_{c}'] = fc[f'share_from_{c}'] + fc[f'share_to_{c}']
        merged[code] = fm

    r2_table, pred = cross_city_resilience(merged, RES_COLS, feature_cols)
    loo_table = pd.DataFrame(loo_by_city).reindex(index=RES_COLS)

    vis_heatmap_cross_city_r2(
        r2_table, loo=loo_table,
        title='Cross-city resilience prediction (test R²; LOO cols = within-city)',
        save_path=os.path.join(OUTPUT_FUNC_VS_RESIL, f'cross_city_r2_{ltag}.png'))
    os.makedirs(OUTPUT_RESIL_REG_RAW, exist_ok=True)
    pd.concat([loo_table.add_prefix('LOO_'), r2_table], axis=1).to_csv(
        os.path.join(OUTPUT_RESIL_REG_RAW, f'cross_city_r2_{ltag}.csv'))

    # Predicted-vs-actual scatter per direction (reuse the within-city scatter).
    for d, pdat in pred.items():
        s = pd.DataFrame({
            'loo_r2': {m: r2_table.loc[m, d] for m in RES_COLS},
            'passed': {m: (bool(r2_table.loc[m, d] > 0)
                           if pd.notna(r2_table.loc[m, d]) else False)
                       for m in RES_COLS},
            'status': {m: ('ok' if pdat[m] is not None else 'insufficient_data')
                       for m in RES_COLS},
        })
        train, test = d.split('->')
        vis_scatter_reg_pred(
            pdat, s, RES_COLS, r2_label='test R²',
            title=f'Cross-city: train {train} -> test {test}',
            save_path=os.path.join(
                OUTPUT_FUNC_VS_RESIL,
                f'cross_city_scatter_{d.replace("->", "_to_")}_{ltag}.png'))


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


# ── Main ──────────────────────────────────────────────────────────────────────

def _lambda_tag(lambda_ctx):
    """Filename-safe context-strength tag, e.g. 'lambda0', 'lambda0.1', or
    'baseline' when context-aware NMF is off (lambda_ctx is None)."""
    return f"lambda{lambda_ctx:g}" if lambda_ctx is not None else "baseline"


def load_landuse_for_context(label, key, gdf):
    """Ensure the raw SLD cache and load the on-the-fly TF-IWF classification
    (same knobs as the O×D func block) for the context-aware NMF.  Returns the
    landuse DataFrame with `share_<cat>` columns keyed by `aggr_id`."""
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


def main():
    # ── Data preprocessing ─────────────────────────────────────────────────────

    # Trimming keeps the FIRST analysis_days days, which drops the late
    # recovery tail and leaves landfall + 14 recovery days at the sequence end
    # for the trailing window taken in build_city_matrices.
    br_gdf = load_city_geo('Baton Rouge', AGG_LEVEL, BR_GEO_CSV)
    fm_gdf = load_city_geo('Fort Myers',  AGG_LEVEL, FM_GEO_CSV)

    br_graphs = load_graphs_trimmed(BR_GRAPH_PATH, BR_ANALYSIS_DAYS,
                                    SLOT_PER_DAY, label='Baton Rouge')
    fm_graphs = load_graphs_trimmed(FM_GRAPH_PATH, FM_ANALYSIS_DAYS,
                                    SLOT_PER_DAY, label='Fort Myers')

    # ── Compute ────────────────────────────────────────────────────────────────

    # The disaster argument of build_city_matrices covers everything after the
    # clean normal segment (buffer plus disaster), so the returned n_nor marks
    # the end of the clean normal columns.  The landfall column n_dis is n_nor
    # plus the buffer slots.
    print("\n── Baton Rouge ──")
    X_br_all, n_nor_br, mapping_br = build_city_matrices(
        br_graphs, DAYS_WINDOW_BR,
        DAYS_BUFFER_BR + DAYS_DISASTER_IN_WIN_BR, FILTER_FACTOR_BR,
    )
    n_dis_br = n_nor_br + DAYS_BUFFER_BR * SLOTS_ACTIVE

    print("\n── Fort Myers ──")
    X_fm_all, n_nor_fm, mapping_fm = build_city_matrices(
        fm_graphs, DAYS_WINDOW_FM,
        DAYS_BUFFER_FM + DAYS_DISASTER_IN_WIN_FM, FILTER_FACTOR_FM,
    )
    n_dis_fm = n_nor_fm + DAYS_BUFFER_FM * SLOTS_ACTIVE

    # All three segments selected -> fit_time_cols=None -> the exact original
    # full-window fit_transform path.  A subset fits the basis on those columns
    # only and projects the full window onto it.
    _ALL_SEGMENTS = {'normal', 'buffer', 'disaster'}
    fit_time_cols_br = (None if set(NMF_FIT_SEGMENTS_BR) == _ALL_SEGMENTS
                        else select_segment_columns(NMF_FIT_SEGMENTS_BR,
                                                    n_nor_br, n_dis_br, X_br_all.shape[1]))
    fit_time_cols_fm = (None if set(NMF_FIT_SEGMENTS_FM) == _ALL_SEGMENTS
                        else select_segment_columns(NMF_FIT_SEGMENTS_FM,
                                                    n_nor_fm, n_dis_fm, X_fm_all.shape[1]))

    print("\n── NMF: Baton Rouge ──")
    if CONTEXT_AWARE_BR:
        landuse_br = load_landuse_for_context('Baton Rouge', 'Baton_Rouge', br_gdf)
        W_br, H_br, weights_br = decompose_city_context(
            X_br_all, N_BEHAVIORS_BR, mapping_br, landuse_br,
            lambda_ctx=LAMBDA_CTX_BR, feature_mode=FLOW_FEATURE_MODE,
            l1_reg=L1_REG_BR, fit_time_cols=fit_time_cols_br)
    else:
        W_br, H_br, weights_br = decompose_city(
            X_br_all, N_BEHAVIORS_BR, l1_reg=L1_REG_BR, fit_time_cols=fit_time_cols_br)

    print("\n── NMF: Fort Myers ──")
    if CONTEXT_AWARE_FM:
        landuse_fm = load_landuse_for_context('Fort Myers', 'Fort_Myers', fm_gdf)
        W_fm, H_fm, weights_fm = decompose_city_context(
            X_fm_all, N_BEHAVIORS_FM, mapping_fm, landuse_fm,
            lambda_ctx=LAMBDA_CTX_FM, feature_mode=FLOW_FEATURE_MODE,
            l1_reg=L1_REG_FM, fit_time_cols=fit_time_cols_fm)
    else:
        W_fm, H_fm, weights_fm = decompose_city(
            X_fm_all, N_BEHAVIORS_FM, l1_reg=L1_REG_FM, fit_time_cols=fit_time_cols_fm)

    # ── Analysis ──────────────────────────

    # Per-city context strength tags the func entropy figure/CSV; None (baseline)
    # when context-aware NMF is off, so the filename still distinguishes runs.
    ctx_lambda_br = LAMBDA_CTX_BR if CONTEXT_AWARE_BR else None
    ctx_lambda_fm = LAMBDA_CTX_FM if CONTEXT_AWARE_FM else None

    feats_by_city = {}   # short code -> per-component feats, for the cross-city test
    loo_by_city   = {}   # short code -> within-city LOO R² (for the comparison)

    for label, key, tag, gdf, H, mapping, weights, W, n_nor, n_dis, fd_nor, fd_dis, ctx_lambda in (
        ('Baton Rouge', 'Baton_Rouge', '_br', br_gdf, H_br, mapping_br, weights_br,
         W_br, n_nor_br, n_dis_br, FIRST_DAY_BR_NORMAL, FIRST_DAY_BR_DISASTER, ctx_lambda_br),
        ('Fort Myers',  'Fort_Myers',  '_fm', fm_gdf, H_fm, mapping_fm, weights_fm,
         W_fm, n_nor_fm, n_dis_fm, FIRST_DAY_FM_NORMAL, FIRST_DAY_FM_DISASTER, ctx_lambda_fm),
    ):
        print(f"\n── {label}: analysis ──")
        analysis_component_signature(W, n_nor, n_dis, fd_nor, fd_dis, tag,
                                     gdf=gdf, H=H, mapping=mapping)
        sf = analysis_spatial(label, H, mapping, gdf, weights, tag,
                              lambda_ctx=ctx_lambda)

        M = analysis_od_function(label, key, tag, gdf, H, mapping, weights,
                                 lambda_ctx=ctx_lambda)

        # Temporal features read W[:n_nor], resilience reads W[n_dis:] against
        # a baseline built from W[:n_nor], and the functional profile reads M.
        # The buffer columns [n_nor, n_dis) feed none of them.
        # recovery_level and lowest_day are inverted here so every resilience
        # metric reads higher = worse.  Spearman is invariant to monotone
        # transforms, so only the sign of these correlation rows changes.
        res = resilience_features(W, n_nor, fd_nor, SLOTS_ACTIVE, n_dis=n_dis)
        n_dd = (W.shape[0] - n_dis) // SLOTS_ACTIVE
        res['recovery_deficit'] = 1.0 - res.pop('recovery_level')
        res['early_collapse']   = (n_dd - 1) - res.pop('lowest_day')
        feats = pd.concat([
            temporal_features(W, n_nor, fd_nor, SLOTS_ACTIVE, _INTERVAL_HOURS,
                              weekend_ratio_threshold=WEEKEND_RATIO_THRESHOLD),
            functional_features(M, AXIS_CATEGORIES),
            sf,                                   # spatial: mean_distance, std_distance
            res,
        ], axis=1)
        feats.insert(0, 'city', label)
        feats.insert(1, 'weight', weights)
        curves = resilience_curves(W, n_nor, fd_nor, SLOTS_ACTIVE, n_dis=n_dis)

        analysis_time_function_corr(feats, tag)
        analysis_resilience_corr(feats, tag, lambda_ctx=ctx_lambda)
        summary_reg = analysis_resilience_linear(feats, tag, lambda_ctx=ctx_lambda)
        analysis_func_ordered_lines(W, n_nor, n_dis, fd_nor, fd_dis,
                                    curves, feats, tag)

        # Stash for the cross-city (leave-one-city-out) test after the loop.
        feats_by_city[tag.strip('_').upper()] = feats.copy()
        loo_by_city[tag.strip('_').upper()]   = summary_reg['loo_r2']

    # ── Cross-city generalisation (train one city, test the other) ──
    if len(feats_by_city) >= 2:
        print("\n── Cross-city resilience test ──")
        analysis_cross_city(feats_by_city, loo_by_city, lambda_ctx=ctx_lambda_br)


if __name__ == '__main__':
    main()
