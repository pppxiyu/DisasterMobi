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
)
from utils_pattern_analysis.space_function import (
    category_lookup_from_landuse, build_od_function_matrix,
    od_function_to_dataframe,
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

    # vis
    vis_heatmap_temporal_signature(W_br, first_day=FIRST_DAY_BR_NORMAL, show_days=True,
                                    slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
                                    output_dir=OUTPUT_PLOTS, tag='_br')
    vis_heatmap_temporal_signature(W_fm, first_day=FIRST_DAY_FM_NORMAL, show_days=True,
                                    slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
                                    output_dir=OUTPUT_PLOTS, tag='_fm')

    # Timeline: split W at n_nor only for colouring (blue = pre-disaster, red =
    # disaster); the dashed vertical line marks where the disaster period begins.
    W_br_nor, W_br_dis = W_br[:n_nor_br], W_br[n_nor_br:]
    W_fm_nor, W_fm_dis = W_fm[:n_nor_fm], W_fm[n_nor_fm:]
    vis_line_nmf_component_timeline(
        W_br_nor, W_br_dis,
        first_day_normal=FIRST_DAY_BR_NORMAL, first_day_disaster=FIRST_DAY_BR_DISASTER,
        slots_per_day=SLOTS_ACTIVE,
        output_dir=OUTPUT_PLOTS, tag='_br',
    )
    vis_line_nmf_component_timeline(
        W_fm_nor, W_fm_dis,
        first_day_normal=FIRST_DAY_FM_NORMAL, first_day_disaster=FIRST_DAY_FM_DISASTER,
        slots_per_day=SLOTS_ACTIVE,
        output_dir=OUTPUT_PLOTS, tag='_fm',
    )

    # # Interactive HTML arc maps: one file per spatial component, for both cities
    # for label, gdf, H, mapping, out_dir in (
    #     ('Baton Rouge', br_gdf, H_br, mapping_br, OUTPUT_NMF_BR),
    #     ('Fort Myers',  fm_gdf, H_fm, mapping_fm, OUTPUT_NMF_FM),
    # ):
    #     assert gdf is not None, 'Geo file missing.'
    #     gdf['lon'] = gdf['centroid'].to_crs(epsg=4326).x
    #     gdf['lat'] = gdf['centroid'].to_crs(epsg=4326).y
    #     for i in range(H.shape[0]):
    #         vis_map_od_flow(
    #             [h_slice_to_od_matrix(H[i: i+1, :], mapping)],
    #             gdfs=gdf, id_col='aggr_id', min_flow=0.5,
    #             max_line_width=20, alpha_range=(0.05, 0.95), curve_rad=0.3, vmax=5,
    #             save_dir=os.path.join(out_dir, f'component_{i}.html'),
    #         )

    # Cache RAW SLD data once (skip if present), classify on-the-fly, then build
    # the per-component origin×destination functional cross-tabs (paper Fig. 9).
    print("\n── Space-function data (EPA SLD) ──")
    os.makedirs(SPACE_FUNCTION_DIR, exist_ok=True)
    OUTPUT_FIG9 = os.path.join(OUTPUT_PLOTS, 'od_function')
    os.makedirs(OUTPUT_FIG9, exist_ok=True)

    landuse = {}
    for label, key, tag, gdf, H, mapping, weights in (
        ('Baton Rouge', 'Baton_Rouge', '_br', br_gdf, H_br, mapping_br, weights_br),
        ('Fort Myers',  'Fort_Myers',  '_fm', fm_gdf, H_fm, mapping_fm, weights_fm),
    ):
        raw_csv = os.path.join(SPACE_FUNCTION_DIR, f'{key}_block_group_sld_raw.csv')
        assert gdf is not None, 'Geo file missing.'
        assert ensure_city_landuse_raw(label, gdf['aggr_id'].tolist(), raw_csv) is not None
        landuse[key] = load_city_landuse(
            raw_csv,
            residential_weight=LANDUSE_RESIDENTIAL_WEIGHT,
            weighting=LANDUSE_WEIGHTING,
            iwf_scale=LANDUSE_IWF_SCALE,
            dominant_threshold=LANDUSE_DOMINANT_THRESHOLD,
        )
        dist = landuse[key]['dominant_category'].value_counts().to_dict()
        print(f"  {label}: dominant_category counts: {dist}")

        # Figure 9: O×D functional cross-tab per component (hard dominant labels).
        cat_lookup = category_lookup_from_landuse(landuse[key])
        M, retained = build_od_function_matrix(H, mapping, cat_lookup, AXIS_CATEGORIES)
        print(f"  {label}: O×D flow retained in-category per component: "
              + ", ".join(f"[{i}]={r:.2f}" for i, r in enumerate(retained)))
        od_function_to_dataframe(M, AXIS_CATEGORIES, weights).to_csv(
            os.path.join(OUTPUT_FIG9, f'od_function{tag}.csv'), index=False)
        vis_heatmap_od_function(
            M, AXIS_CATEGORIES, weights=weights, ncols=3,
            save_path=os.path.join(OUTPUT_FIG9, f'od_function_grid{tag}.png'),
        )


if __name__ == '__main__':
    main()
