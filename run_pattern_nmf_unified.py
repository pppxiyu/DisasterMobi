"""
NMF decomposition of mobility patterns — UNIFIED (paper-style) approach.

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
5. Save to outputs/nmf_unified/:
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
    python run_pattern_nmf_unified.py
"""
import os

import pandas as pd

from config import (
    BR_GRAPH_PATH, FM_GRAPH_PATH, BR_ANALYSIS_DAYS, FM_ANALYSIS_DAYS,
    BR_GEO_CSV, FM_GEO_CSV, AGG_LEVEL,
    DATA_DIR, OUTPUT_DIR, SLOT_PER_DAY, SLOTS_ACTIVE,
)

# Derived from config.  Never hardcode the interval here.
_INTERVAL_HOURS = 24 // SLOT_PER_DAY   # 3 for 3h data and 2 for 2h data
from utils_pattern_analysis.graph_io import load_graphs_trimmed
from utils_pattern_analysis.decomposition import h_slice_to_od_matrix
from utils_pattern_analysis.nmf_pipeline import (
    build_city_matrices, decompose_city,
)
from utils_pattern_analysis.visualization import (
    vis_heatmap_temporal_signature, vis_map_od_flow,
    vis_line_nmf_component_timeline, vis_heatmap_od_function,
    vis_scatter_component_features,
    vis_bar_function_by_peakslot, vis_line_resilience_curves,
    vis_bar_resilience_by_peakslot, vis_heatmap_corr_split,
)
from utils_pattern_analysis.space_function import (
    category_lookup_from_landuse, build_od_function_matrix,
)
from utils_pattern_analysis.component_features import (
    temporal_features, functional_features, time_function_correlation,
    resilience_features, resilience_curves,
)
from utils_data_processing.build_graphs import load_city_geo
from utils_data_processing.fetch_sld_landuse import (
    ensure_city_landuse_raw, load_city_landuse, CATEGORIES as SF_CATEGORIES,
)


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

N_BEHAVIORS_BR = 20
N_BEHAVIORS_FM = 25
L1_REG_BR      = 0.5
L1_REG_FM      = 0.5

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

OUTPUT_PLOTS = os.path.join(OUTPUT_DIR, 'nmf_unified')

# All per-component characteristics live under component_characteristics, one
# subfolder per characteristic type.
#   temporal/          signature heatmap + timeline (temporal factor W)
#   spatial/           per-city OD arc-map subfolders (spatial factor H)
#   func/              per-component O×D functional cross-tabs
#   func_vs_temporal/  time × function correlation figures
OUTPUT_CHAR         = os.path.join(OUTPUT_PLOTS, 'component_characteristics')
OUTPUT_TEMPORAL     = os.path.join(OUTPUT_CHAR, 'temporal')
OUTPUT_SPATIAL      = os.path.join(OUTPUT_CHAR, 'spatial')
OUTPUT_NMF_BR       = os.path.join(OUTPUT_SPATIAL, 'component_spatial_characteristics_br')
OUTPUT_NMF_FM       = os.path.join(OUTPUT_SPATIAL, 'component_spatial_characteristics_fm')
OUTPUT_FUNC         = os.path.join(OUTPUT_CHAR, 'func')
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

    # Interactive HTML arc maps, one file per spatial component.  The
    # centroid column from load_city_geo is already EPSG:4326.
    out_dir = os.path.join(OUTPUT_SPATIAL,
                           f'component_spatial_characteristics{tag}')  # = OUTPUT_NMF_BR/_FM
    os.makedirs(out_dir, exist_ok=True)
    gdf['lon'] = gdf['centroid'].x
    gdf['lat'] = gdf['centroid'].y
    for i in range(H.shape[0]):
        vis_map_od_flow(
            [h_slice_to_od_matrix(H[i: i+1, :], mapping)],
            gdfs=gdf, id_col='aggr_id', min_flow=0.5,
            max_line_width=20, alpha_range=(0.05, 0.95), curve_rad=0.3, vmax=5,
            save_dir=os.path.join(out_dir, f'component_{i}.html'),
        )


def analysis_od_function(label, key, tag, gdf, H, mapping, weights):
    """O×D functionality block.  Ensures the raw SLD cache,
    classifies block groups on the fly, aggregates each component's OD flows
    into an origin×destination functional cross-tab, and saves the heatmap
    grid plus a per-component functional-share CSV.  Returns M [k × C × C]."""
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
    functional_features(M, AXIS_CATEGORIES).to_csv(
        os.path.join(OUTPUT_FUNC, f'component_functionality{tag}.csv'))
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


def analysis_resilience_corr(feats, tag):
    """Resilience correlation block.  Splits its figures across two subfolders
    of resilience_corr — func_vs_resilience gets the Spearman heatmap between
    RES_COLS and the functional shares plus the top-pair scatter;
    temporal_vs_resilience gets the weekday_ratio heatmap and the peak-slot /
    peak-period bar charts.  The per-function curve stacks are drawn separately
    by analysis_func_ordered_lines."""
    os.makedirs(OUTPUT_FUNC_VS_RESIL, exist_ok=True)
    rho, pval = time_function_correlation(feats, RES_COLS, TIME_COLS + FUNC_COLS)
    # Functional-share heatmap — the 6 category columns only, each cell stacking
    # share_from (upper) and share_to (lower).  weekday_ratio is split off into
    # its own heatmap below.  Rows are coloured by their own max |rho|.
    vis_heatmap_corr_split(
        rho, pval, time_cols=[], categories=SF_CATEGORIES,
        save_path=os.path.join(OUTPUT_FUNC_VS_RESIL, f'heatmap_resilience_corr{tag}.png'),
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

    print("\n── NMF: Baton Rouge ──")
    W_br, H_br, weights_br = decompose_city(X_br_all, N_BEHAVIORS_BR, l1_reg=L1_REG_BR)

    print("\n── NMF: Fort Myers ──")
    W_fm, H_fm, weights_fm = decompose_city(X_fm_all, N_BEHAVIORS_FM, l1_reg=L1_REG_FM)

    # ── Analysis ──────────────────────────

    for label, key, tag, gdf, H, mapping, weights, W, n_nor, n_dis, fd_nor, fd_dis in (
        ('Baton Rouge', 'Baton_Rouge', '_br', br_gdf, H_br, mapping_br, weights_br,
         W_br, n_nor_br, n_dis_br, FIRST_DAY_BR_NORMAL, FIRST_DAY_BR_DISASTER),
        ('Fort Myers',  'Fort_Myers',  '_fm', fm_gdf, H_fm, mapping_fm, weights_fm,
         W_fm, n_nor_fm, n_dis_fm, FIRST_DAY_FM_NORMAL, FIRST_DAY_FM_DISASTER),
    ):
        print(f"\n── {label}: analysis ──")
        analysis_component_signature(W, n_nor, n_dis, fd_nor, fd_dis, tag,
                                     gdf=gdf, H=H, mapping=mapping)

        M = analysis_od_function(label, key, tag, gdf, H, mapping, weights)

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
            res,
        ], axis=1)
        feats.insert(0, 'city', label)
        feats.insert(1, 'weight', weights)
        curves = resilience_curves(W, n_nor, fd_nor, SLOTS_ACTIVE, n_dis=n_dis)

        analysis_time_function_corr(feats, tag)
        analysis_resilience_corr(feats, tag)
        analysis_func_ordered_lines(W, n_nor, n_dis, fd_nor, fd_dis,
                                    curves, feats, tag)



if __name__ == '__main__':
    main()
