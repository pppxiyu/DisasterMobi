"""
NMF decomposition of mobility patterns — UNIFIED (paper-style) approach.

Cities: #1 = Baton Rouge (Hurricane Ida, 2021), #2 = Fort Myers (Hurricane Ian, 2022).

Idea
----
For each city, the pre-disaster ("normal") period and the disaster period are kept
in ONE trailing window and a SINGLE non-negative matrix factorisation is run on
that whole window, producing two factors:

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
     - temporal-signature heatmap (W), per city
     - per-component timeline (W): normal in blue, disaster in red
     - per-component OD arc map (H): one HTML per component, both cities

Geometry is loaded per city and is missing-safe: a city without a geo file keeps
its NMF and temporal plots but skips the OD arc maps.

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

# Derived from config — never hardcode 3 or 5 here
_INTERVAL_HOURS = 24 // SLOT_PER_DAY   # 3 for 3h data, 2 for 2h data
from utils_pattern_analysis.graph_io import load_graphs
from utils_pattern_analysis.decomposition import h_slice_to_od_matrix
from utils_pattern_analysis.nmf_pipeline import (
    build_city_matrices, decompose_city,
)
from utils_pattern_analysis.visualization import (
    vis_heatmap_temporal_signature, vis_map_od_flow,
    vis_line_nmf_component_timeline, vis_heatmap_od_function,
    vis_heatmap_time_function_corr, vis_scatter_component_features,
    vis_bar_function_by_peakslot, vis_line_resilience_curves,
    vis_bar_resilience_by_peakslot,
)
from utils_pattern_analysis.space_function import (
    category_lookup_from_landuse, build_od_function_matrix,
    od_function_to_dataframe,
)
from utils_pattern_analysis.component_features import (
    temporal_features, functional_features, time_function_correlation,
    resilience_features, resilience_curves,
)
from utils_data_processing.build_graphs import load_city_geo_or_warn
from utils_data_processing.fetch_sld_landuse import (
    ensure_city_landuse_raw, load_city_landuse, CATEGORIES as SF_CATEGORIES,
)


# ── Configuration ─────────────────────────────────────────────────────────────

# Trailing window taken from the end of the graph sequence (normal + disaster).
DAYS_WINDOW_BR          = 28   # 13 normal + 15 disaster (Ida Aug 29 + 14 recovery)
DAYS_DISASTER_IN_WIN_BR = 15
DAYS_WINDOW_FM          = 28   # 13 normal + 15 disaster (Ian Sep 28 + 14 recovery)
DAYS_DISASTER_IN_WIN_FM = 15

N_BEHAVIORS_BR = 15
N_BEHAVIORS_FM = 15
L1_REG_BR      = 0.5
L1_REG_FM      = 0.5

FILTER_FACTOR_BR = 3
FILTER_FACTOR_FM = 1

# Baton Rouge runs Apr 15 (Thu) → Sep 16, trimmed to Sep 12 (Ida + 14; see
# BR_ANALYSIS_DAYS).  Last 28 days = Aug 16 (Mon) → Sep 12; disaster portion
# = Aug 29 (Sun, Ida landfall) → Sep 12.
FIRST_DAY_BR_NORMAL   = 'Monday'   # first day of BR's 28-day window   (Aug 16)
FIRST_DAY_BR_DISASTER = 'Sunday'   # first day of BR's disaster portion (Aug 29, Ida)

# Fort Myers trimmed to Aug 30 – Oct 12 2022 (44 days; see FM_ANALYSIS_DAYS in
# config).  Trailing 28-day window = Sep 15–Oct 12; disaster half = Sep 28–Oct 12
# (Ian + 14 days recovery).  Weekdays are the real first-day-of-window dates.
FIRST_DAY_FM_NORMAL   = 'Thursday'    # Sep 15 2022 (normal-half start)
FIRST_DAY_FM_DISASTER = 'Wednesday'   # Sep 28 2022 (disaster-half / Ian landfall)

OUTPUT_PLOTS  = os.path.join(OUTPUT_DIR, 'nmf_unified')
OUTPUT_NMF_BR = os.path.join(OUTPUT_DIR, 'nmf_unified', 'components_br')
OUTPUT_NMF_FM = os.path.join(OUTPUT_DIR, 'nmf_unified', 'components_fm')

# Per-city block-group space-function data (EPA Smart Location Database). 
SPACE_FUNCTION_DIR = os.path.join(DATA_DIR, 'space_function')

# On-the-fly land-use classification knobs (TF-IWF reweighting; see
# utils_data_processing/fetch_sld_landuse.py for the category definitions).
LANDUSE_WEIGHTING          = 'tf_iwf'   # 'tf_iwf' (down-weight ubiquitous residential) | 'raw_share'
LANDUSE_IWF_SCALE          = 1.0        # IWF exponent: higher → stronger pull to rare/distinctive functions
LANDUSE_RESIDENTIAL_WEIGHT = 1.0        # housing-unit ↔ job equivalence, fix this
LANDUSE_DOMINANT_THRESHOLD = 0.4        # a BG is labelled by its top category only if its
                                        # (reweighted) share exceeds this; else 'Mix'. Lower → fewer Mix

# Figure-9 axes: the functional categories + 'Mix'; 'Unknown'/unmatched endpoints
# are dropped.  Order defines the heatmap rows (origin) and columns (destination).
AXIS_CATEGORIES = list(SF_CATEGORIES) + ['Mix']

OUTPUT_FIG9 = os.path.join(OUTPUT_PLOTS, 'od_function')      # O×D cross-tab outputs
OUTPUT_CORR = os.path.join(OUTPUT_PLOTS, 'component_corr')   # feature/correlation outputs

# ── Per-component feature columns used by the correlation blocks ──────────────
# Temporal features (computed from the PRE-disaster part of W only; see
# utils_pattern_analysis/component_features.py):
#   am_pm         — morning minus evening share of the within-day profile:
#                   sum of 6-12h slot shares − sum of 16-22h slot shares;
#                   >0 morning-type, <0 evening-type, ≈0 flat/symmetric
#                   (midday 12-16h is a buffer, counted for neither).
#   weekday_ratio — mean weekday daily total / mean weekend daily total
#                   (weekday from FIRST_DAY_*_NORMAL); >1 weekday-dominated
#                   (commute-like), <1 weekend-dominated (leisure-like).
# peak_slot (argmax of the within-day profile) is NOT in the rank correlation —
# time-of-day is not a monotone scale; it is treated as CATEGORICAL instead,
# via the per-peak-slot bar charts.
TIME_COLS = ['am_pm', 'weekday_ratio']

# Functional features (from the O×D cross-tab, Mix/Unknown dropped and the
# 5 categories renormalised), split by flow direction:
#   share_from_<cat> — outflow side: fraction departing FROM function <cat>
#   share_to_<cat>   — inflow side: fraction arriving AT function <cat>
#   diag_share       — same-function flow fraction (trace).
FUNC_COLS = ([f'share_from_{c}' for c in SF_CATEGORIES]
             + [f'share_to_{c}' for c in SF_CATEGORIES]
             + ['diag_share'])

# Resilience features (disaster-period daily activity relative to a weekday/
# weekend-matched pre-disaster baseline, 3-day smoothed; see
# component_features.resilience_features):
#   drop_depth     — 1 − min(r): how deep activity fell (1 = total stop;
#                    NEGATIVE = rose above baseline → disaster-emergent pattern)
#   trough_day     — days from landfall to the curve minimum
#   recovery_level — mean r over the last 3 disaster days (≈1 recovered,
#                    <1 not recovered, >1 overshoot)
#   cum_loss       — Σ max(0, 1−r): resilience-triangle area = total lost
#                    activity in day-equivalents (smaller = more resilient)
RES_COLS = ['drop_depth', 'trough_day', 'recovery_level', 'cum_loss']


# ── Analysis helpers (one per analysis block; called once per city) ──────────

def analysis_component_signature(W, n_nor, first_day_normal, first_day_disaster, tag,
                                 gdf=None, H=None, mapping=None):
    """Component temporal signatures: full-window W heatmap + per-component
    timeline (normal half blue, disaster half red, dashed boundary).
    Optionally (commented below) interactive HTML OD arc maps per component —
    pass gdf/H/mapping and uncomment to enable."""
    vis_heatmap_temporal_signature(
        W, first_day=first_day_normal, show_days=True,
        slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
        output_dir=OUTPUT_PLOTS, tag=tag,
    )
    vis_line_nmf_component_timeline(
        W[:n_nor], W[n_nor:],
        first_day_normal=first_day_normal, first_day_disaster=first_day_disaster,
        slots_per_day=SLOTS_ACTIVE,
        output_dir=OUTPUT_PLOTS, tag=tag,
    )

    # # Interactive HTML arc maps: one file per spatial component.
    # out_dir = os.path.join(OUTPUT_PLOTS, f'components{tag}')  # = OUTPUT_NMF_BR/_FM
    # assert gdf is not None, 'Geo file missing.'
    # gdf['lon'] = gdf['centroid'].to_crs(epsg=4326).x
    # gdf['lat'] = gdf['centroid'].to_crs(epsg=4326).y
    # for i in range(H.shape[0]):
    #     vis_map_od_flow(
    #         [h_slice_to_od_matrix(H[i: i+1, :], mapping)],
    #         gdfs=gdf, id_col='aggr_id', min_flow=0.5,
    #         max_line_width=20, alpha_range=(0.05, 0.95), curve_rad=0.3, vmax=5,
    #         save_dir=os.path.join(out_dir, f'component_{i}.html'),
    #     )


def analysis_od_function(label, key, tag, gdf, H, mapping, weights):
    """Paper-Fig.9 block: ensure the raw SLD cache, classify block groups
    on-the-fly, aggregate each component's OD flows into an origin×destination
    functional cross-tab, save CSV + heatmap grid.  Returns M [k × C × C]."""
    raw_csv = os.path.join(SPACE_FUNCTION_DIR, f'{key}_block_group_sld_raw.csv')
    assert gdf is not None, 'Geo file missing.'
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
    od_function_to_dataframe(M, AXIS_CATEGORIES, weights).to_csv(
        os.path.join(OUTPUT_FIG9, f'od_function{tag}.csv'), index=False)
    vis_heatmap_od_function(
        M, AXIS_CATEGORIES, weights=weights, ncols=3,
        save_path=os.path.join(OUTPUT_FIG9, f'heatmap_od_function{tag}.png'),
    )
    return M


def analysis_time_function_corr(feats, tag):
    """Time × function block: Spearman between TIME_COLS and FUNC_COLS across
    one city's components — heatmap, top-|rho| pair scatter, plus the
    categorical peak-slot view (mean from/to shares per peak slot)."""
    rho, pval = time_function_correlation(feats, TIME_COLS, FUNC_COLS)
    rho.to_csv(os.path.join(OUTPUT_CORR, f'spearman_rho{tag}.csv'))
    pval.to_csv(os.path.join(OUTPUT_CORR, f'spearman_pval{tag}.csv'))
    vis_heatmap_time_function_corr(
        rho, pval,
        save_path=os.path.join(OUTPUT_CORR, f'heatmap_time_function_corr{tag}.png'),
    )
    pairs = rho.abs().stack().sort_values(ascending=False).index[:4].tolist()
    vis_scatter_component_features(
        feats, pairs,
        save_path=os.path.join(OUTPUT_CORR, f'scatter_time_function_top_pairs{tag}.png'),
    )
    vis_bar_function_by_peakslot(
        feats, SF_CATEGORIES,
        save_path=os.path.join(OUTPUT_CORR, f'bar_function_by_peakslot{tag}.png'),
    )


def analysis_resilience_corr(feats, curves, tag):
    """Resilience block: per-component drop-and-recovery curves (QC view),
    Spearman between RES_COLS and the temporal+functional features, ranked
    |rho| pair list, top-pair scatter, plus the categorical peak-slot view."""
    vis_line_resilience_curves(
        curves,
        save_path=os.path.join(OUTPUT_CORR, f'line_resilience_curves{tag}.png'),
    )

    rho, pval = time_function_correlation(feats, RES_COLS, TIME_COLS + FUNC_COLS)
    rho.to_csv(os.path.join(OUTPUT_CORR, f'spearman_resilience_rho{tag}.csv'))
    pval.to_csv(os.path.join(OUTPUT_CORR, f'spearman_resilience_pval{tag}.csv'))
    vis_heatmap_time_function_corr(
        rho, pval,
        save_path=os.path.join(OUTPUT_CORR, f'heatmap_resilience_corr{tag}.png'),
    )

    # Ranked pair list — answers "which pairs are strongest" in one CSV.
    ranked = (pd.concat([rho.stack().rename('rho'),
                         pval.stack().rename('pval')], axis=1)
                .reset_index()
                .rename(columns={'level_0': 'resilience_feature',
                                 'level_1': 'feature'}))
    ranked = ranked.reindex(ranked['rho'].abs().sort_values(ascending=False).index)
    ranked.to_csv(os.path.join(OUTPUT_CORR, f'resilience_top_pairs{tag}.csv'),
                  index=False)

    pairs = rho.abs().stack().sort_values(ascending=False).index[:4].tolist()
    vis_scatter_component_features(
        feats, pairs,
        save_path=os.path.join(OUTPUT_CORR, f'scatter_resilience_top_pairs{tag}.png'),
    )
    vis_bar_resilience_by_peakslot(
        feats, RES_COLS,
        save_path=os.path.join(OUTPUT_CORR, f'bar_resilience_by_peakslot{tag}.png'),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_NMF_BR, exist_ok=True)
    os.makedirs(OUTPUT_NMF_FM, exist_ok=True)

    # geo
    br_gdf = load_city_geo_or_warn('Baton Rouge', AGG_LEVEL, BR_GEO_CSV)
    fm_gdf = load_city_geo_or_warn('Fort Myers',  AGG_LEVEL, FM_GEO_CSV)

    for gdf in (br_gdf, fm_gdf):
        if gdf is not None:
            gdf['centroid'] = gdf.geometry.to_crs(epsg=3857).centroid.to_crs(epsg=4326)

    # trim data on temporal dim
    print("Loading Baton Rouge graphs …")
    br_graphs = load_graphs(BR_GRAPH_PATH)
    if BR_ANALYSIS_DAYS is not None:
        # Trim BR post-Ida recovery tail
        need = BR_ANALYSIS_DAYS * SLOT_PER_DAY
        if len(br_graphs) < need:
            raise ValueError(
                f"Baton Rouge: BR_ANALYSIS_DAYS={BR_ANALYSIS_DAYS} needs {need} "
                f"graphs ({BR_ANALYSIS_DAYS} days) but the pkl has only "
                f"{len(br_graphs)} ({len(br_graphs) // SLOT_PER_DAY} days)."
            )
        br_graphs = br_graphs[:need]
    print("Loading Fort Myers graphs …")
    fm_graphs = load_graphs(FM_GRAPH_PATH)
    if FM_ANALYSIS_DAYS is not None:
        fm_graphs = fm_graphs[:FM_ANALYSIS_DAYS * SLOT_PER_DAY]

    # sanity check
    _expected_interval = f'{_INTERVAL_HOURS}h'
    for city, graphs in [('Baton Rouge', br_graphs), ('Fort Myers', fm_graphs)]:
        n = len(graphs)
        # Metadata check (graphs built by our pipeline store interval_duration)
        actual_interval = graphs[0].graph.get('interval_duration')
        if actual_interval is not None and actual_interval != _expected_interval:
            raise ValueError(
                f"{city}: pkl contains {actual_interval} graphs but config expects "
                f"{_expected_interval} (SLOT_PER_DAY={SLOT_PER_DAY}). "
            )
        # Divisibility check (catches 3h file with 2h config even without metadata)
        if n % SLOT_PER_DAY != 0:
            raise ValueError(
                f"{city}: {n} graphs is not divisible by SLOT_PER_DAY={SLOT_PER_DAY}. "
                f"The loaded pkl was likely built with a different time resolution. "
            )

    # compute
    print("\n── Baton Rouge ──")
    X_br_all, n_nor_br, mapping_br = build_city_matrices(
        br_graphs, DAYS_WINDOW_BR, DAYS_DISASTER_IN_WIN_BR, FILTER_FACTOR_BR,
    )

    print("\n── Fort Myers ──")
    X_fm_all, n_nor_fm, mapping_fm = build_city_matrices(
        fm_graphs, DAYS_WINDOW_FM, DAYS_DISASTER_IN_WIN_FM, FILTER_FACTOR_FM,
    )

    print("\n── NMF: Baton Rouge ──")
    W_br, H_br, weights_br = decompose_city(X_br_all, N_BEHAVIORS_BR, l1_reg=L1_REG_BR)

    print("\n── NMF: Fort Myers ──")
    W_fm, H_fm, weights_fm = decompose_city(X_fm_all, N_BEHAVIORS_FM, l1_reg=L1_REG_FM)

    # ── Per-city analysis (helpers above, one per analysis block) ─────────────
    os.makedirs(SPACE_FUNCTION_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FIG9, exist_ok=True)
    os.makedirs(OUTPUT_CORR, exist_ok=True)

    feat_frames = []
    for label, key, tag, gdf, H, mapping, weights, W, n_nor, fd_nor, fd_dis in (
        ('Baton Rouge', 'Baton_Rouge', '_br', br_gdf, H_br, mapping_br, weights_br,
         W_br, n_nor_br, FIRST_DAY_BR_NORMAL, FIRST_DAY_BR_DISASTER),
        ('Fort Myers',  'Fort_Myers',  '_fm', fm_gdf, H_fm, mapping_fm, weights_fm,
         W_fm, n_nor_fm, FIRST_DAY_FM_NORMAL, FIRST_DAY_FM_DISASTER),
    ):
        print(f"\n── {label}: analysis ──")
        # 1) component temporal signatures (heatmap + normal/disaster timeline;
        #    optional OD arc maps commented inside the helper)
        analysis_component_signature(W, n_nor, fd_nor, fd_dis, tag,
                                     gdf=gdf, H=H, mapping=mapping)

        # 2) O×D functional cross-tab per component (paper Fig. 9)
        M = analysis_od_function(label, key, tag, gdf, H, mapping, weights)

        # per-component features (pure computation): temporal rhythm from the
        # pre-disaster W, functional profile from M, resilience from the
        # disaster-period relative-activity curve
        feats = pd.concat([
            temporal_features(W, n_nor, fd_nor, SLOTS_ACTIVE, _INTERVAL_HOURS),
            functional_features(M, AXIS_CATEGORIES),
            resilience_features(W, n_nor, fd_nor, SLOTS_ACTIVE),
        ], axis=1)
        feats.insert(0, 'city', label)
        feats.insert(1, 'weight', weights)
        feat_frames.append(feats)
        curves = resilience_curves(W, n_nor, fd_nor, SLOTS_ACTIVE)

        # 3) time × function correlation
        analysis_time_function_corr(feats, tag)

        # 4) resilience: curves + correlation vs time & function features
        analysis_resilience_corr(feats, curves, tag)

    pd.concat(feat_frames).to_csv(os.path.join(OUTPUT_CORR, 'component_features.csv'))



if __name__ == '__main__':
    main()
