"""
Tucker tensor decomposition of OD mobility patterns.

Cities: #1 = Baton Rouge (Ida 2021), #2 = Fort Myers (Ian 2022).

For each city:
  1. Convert graphs to a 3-D (Origin × Destination × Time) tensor.
  2. Average the normal period into a 7-day template.
  3. Run non-negative Tucker decomposition on the normal template.
  4. Fix U1/U2 (spatial factors) and re-decompose the disaster tensor with
     a weekday-aligned U3 warm start and cosine-similarity temporal regularisation.
  5. Compute temporal and spatiotemporal inter-city component mappings.
  6. Optionally extract OSM land-use features per component (requires srai).

Note on Fort Myers
------------------
Spatial choropleths are produced for Baton Rouge only; the OSM-feature block is
guarded so it skips a city when geometry is unavailable.  Fort Myers is trimmed
to Aug 30 – Oct 12 2022 (44 days; see FM_ANALYSIS_DAYS in config) so Ian's
landfall (Sep 28) + ~2 weeks of recovery sit at the window end.

Run
---
    python run_pattern_tucker.py
"""
import os
import pickle

import numpy as np

from config import (
    BR_GRAPH_PATH, FM_GRAPH_PATH, BR_ANALYSIS_DAYS, FM_ANALYSIS_DAYS,
    BR_GEO_CSV, FM_GEO_CSV, AGG_LEVEL,
    OUTPUT_DIR, SLOT_PER_DAY, SLOT_TRIM_START, SLOT_TRIM_END, SLOTS_ACTIVE,
)

# Derived from config — never hardcode 3 or 5 here
_INTERVAL_HOURS = 24 // SLOT_PER_DAY   # 3 for 3h data, 2 for 2h data
from utils_pattern_analysis.graph_io import (
    load_graphs, periodic_trim,
    graphs_to_3d_tensor, filter_inactive_locations,
    calculate_segment_average, print_matrix_diagnostics,
)
from utils_pattern_analysis.decomposition import (
    non_negative_tucker_decomposition, non_negative_tucker_hals,
    check_reconstruction_error_detailed,
)
from utils_pattern_analysis.matching import (
    calculate_correlation_matrix, process_weight_matrix,
    reconstruct_v_no_with_coor, broadcast_matrix_by_weekday,
    get_global_temporal_trends,
)
from utils_pattern_analysis.visualization import (
    vis_heatmap_temporal_signature, vis_heatmap_spatial_profiles,
    vis_map_spatial_factors,
    vis_heatmap_core_interaction, vis_heatmap_component_mapping,
)
from utils_data_processing.build_graphs import load_city_geo_or_warn


# ── Configuration ─────────────────────────────────────────────────────────────

DAYS_NORMAL        = 28
DAYS_DISASTER_BR   = 15   # BR disaster window: last 15 days (Aug 29–Sep 12, Ida + 14d)
DAYS_DISASTER_FM   = 15   # FM disaster window: last 15 days (Sep 28–Oct 12, Ian + 14d)

RANKS_BR = [15, 15, 8]   # BR keeps ~22 nodes at ff=20 (15 ≤ 22, runs)
RANKS_FM = [12, 12, 8]   # FM sparser → fewer nodes; ranks lowered to fit (12 ≤ 46)

FILTER_FACTOR_BR   = 20
# FM is ~3× sparser than BR: ff=20 leaves only ~8 nodes (< rank 15 → crash), so
# it is lowered to retain ~46 nodes (≥ RANKS_FM).  Retune once FM is profiled.
FILTER_FACTOR_FM   = 3

TEMPORAL_LAMBDA_BR = 1500
TEMPORAL_LAMBDA_FM = 1500  # conservative: matched to BR; retune later

# Baton Rouge: PROPHET_START_DATE 2021-04-15 is a Thursday.  Disaster = last 15
# days of the trimmed file = Aug 29 (Sun, Ida landfall) → Sep 12.
FIRST_DAY_BR_NORMAL    = 'Thursday'   # first-28d normal start (Apr 15)
FIRST_DAY_BR_DISASTER  = 'Sunday'     # last-15d disaster start (Aug 29, Ida)

# Fort Myers trimmed to Aug 30 – Oct 12 2022 (44 days; see FM_ANALYSIS_DAYS in
# config).  Normal = first 28 days (Aug 30); disaster = last 15 days (Sep 28–Oct
# 12, Ian + 14d recovery).  Weekdays are the real first-day-of-window dates.
FIRST_DAY_FM_NORMAL    = 'Tuesday'     # Aug 30 2022 (first-28d normal start)
FIRST_DAY_FM_DISASTER  = 'Wednesday'   # Sep 28 2022 (last-15d disaster start / Ian)

TEMPERATURE = 0.20

DECOMP_TOL      = 1e-5
DECOMP_ITER_MAX = 1000

RUN_SPATIAL_FEATURES = False  # set True to extract OSM features (requires srai + internet)

OUTPUT_PLOTS = os.path.join(OUTPUT_DIR, 'tucker')


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_city_tensors(graphs, days_normal, days_disaster, filter_factor):
    slot_active = SLOTS_ACTIVE

    normal_graphs = periodic_trim(
        graphs[: SLOT_PER_DAY * days_normal],
        cycle_len=SLOT_PER_DAY, trim_start=SLOT_TRIM_START, trim_end=SLOT_TRIM_END,
    )
    X, mapping = graphs_to_3d_tensor(normal_graphs)
    X = calculate_segment_average(X, segment_len=slot_active * 7)

    disaster_graphs = periodic_trim(
        graphs[-SLOT_PER_DAY * days_disaster:],
        cycle_len=SLOT_PER_DAY, trim_start=SLOT_TRIM_START, trim_end=SLOT_TRIM_END,
    )
    X_imp, _ = graphs_to_3d_tensor(disaster_graphs)

    threshold = slot_active * 7 * filter_factor
    X, X_imp, mapping = filter_inactive_locations(X, X_imp, mapping, threshold)

    print(f"  Normal tensor: {X.shape}, Disaster tensor: {X_imp.shape}")
    print_matrix_diagnostics(X,     mapping, label='normal period (7-day avg)')
    print_matrix_diagnostics(X_imp, mapping, label='disaster period (raw)')
    return X, X_imp, mapping


def decompose_city(X, X_imp, ranks, days_disaster,
                   first_day_normal, first_day_disaster, temporal_lambda):
    slot_active = SLOTS_ACTIVE

    core, factors = non_negative_tucker_decomposition(
        X, ranks, tol=DECOMP_TOL, n_iter_max=DECOMP_ITER_MAX, verbose=True,
    )
    check_reconstruction_error_detailed(X, core, factors)

    U3_init = broadcast_matrix_by_weekday(
        factors[2], first_day_normal, first_day_disaster,
        days_disaster * slot_active,
    )

    core_imp, factors_imp = non_negative_tucker_hals(
        X_imp, rank=ranks,
        n_iter_max=DECOMP_ITER_MAX,
        init=(core, [factors[0], factors[1], U3_init]),
        tol=DECOMP_TOL,
        verbose=True,
        normalize_factors=True,
        fixed_modes=[0, 1],
        temporal_lambda=temporal_lambda,
    )
    check_reconstruction_error_detailed(X_imp, core_imp, factors_imp)

    return core, factors, core_imp, factors_imp


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Geometry (AGG_LEVEL-aware, missing-safe).  Only Baton Rouge is plotted
    # spatially here (only Baton Rouge gets choropleths in this script).
    br_gdf = load_city_geo_or_warn('Baton Rouge', AGG_LEVEL, BR_GEO_CSV)
    fm_gdf = load_city_geo_or_warn('Fort Myers',  AGG_LEVEL, FM_GEO_CSV)

    if br_gdf is not None:
        br_gdf['centroid'] = br_gdf.geometry.to_crs(epsg=3857).centroid.to_crs(epsg=4326)

    print("Loading graphs …")
    br_graphs = load_graphs(BR_GRAPH_PATH)
    if BR_ANALYSIS_DAYS is not None:
        # Trim BR post-Ida recovery tail so the disaster sits at the window end
        # (2021-04-15 .. 2021-09-12 = Ida + 14 days recovery; see config).  Guard:
        # error loudly if the pkl is shorter than the trim target (e.g. the
        # recovery-extended data through 2021-09-16 is not rebuilt yet) instead of
        # silently analysing a mis-placed disaster window.
        need = BR_ANALYSIS_DAYS * SLOT_PER_DAY
        if len(br_graphs) < need:
            raise ValueError(
                f"Baton Rouge: BR_ANALYSIS_DAYS={BR_ANALYSIS_DAYS} needs {need} "
                f"graphs ({BR_ANALYSIS_DAYS} days) but the pkl has only "
                f"{len(br_graphs)} ({len(br_graphs) // SLOT_PER_DAY} days). The "
                f"recovery-extended BR data (through 2021-09-16) may not be rebuilt "
                f"yet — rebuild the graph pkl or lower BR_ANALYSIS_DAYS in config.py."
            )
        br_graphs = br_graphs[:need]
    fm_graphs = load_graphs(FM_GRAPH_PATH)
    if FM_ANALYSIS_DAYS is not None:
        # Trim Fort Myers post-Ian recovery tail so the disaster sits near the
        # window end (Aug 30 – Oct 12 2022 = Ian + 2 weeks recovery; see config).
        fm_graphs = fm_graphs[:FM_ANALYSIS_DAYS * SLOT_PER_DAY]

    # ── Sanity-check: graph resolution must match SLOT_PER_DAY in config ──────
    _expected_interval = f'{_INTERVAL_HOURS}h'
    for city, graphs in [('Baton Rouge', br_graphs), ('Fort Myers', fm_graphs)]:
        n = len(graphs)
        actual_interval = graphs[0].graph.get('interval_duration')
        if actual_interval is not None and actual_interval != _expected_interval:
            raise ValueError(
                f"{city}: pkl contains {actual_interval} graphs but config expects "
                f"{_expected_interval} (SLOT_PER_DAY={SLOT_PER_DAY}). "
                f"Rebuild the graph pkl (notebook_drafts/Cubique/main_20260607.ipynb) or update config.py."
            )
        if n % SLOT_PER_DAY != 0:
            raise ValueError(
                f"{city}: {n} graphs is not divisible by SLOT_PER_DAY={SLOT_PER_DAY}. "
                f"Rebuild the graph pkl (notebook_drafts/Cubique/main_20260607.ipynb) or update config.py."
            )
        days_in_file = n // SLOT_PER_DAY
        meta_note = f" [metadata: {actual_interval}]" if actual_interval else " [no metadata]"
        print(f"  {city}: {n} graphs = {days_in_file} days × {SLOT_PER_DAY} slots "
              f"({_INTERVAL_HOURS}h each){meta_note} ✓")

    # ── Baton Rouge ──────────────────────────────────────────────────────────
    print("\n── Baton Rouge: tensors ──")
    X_br, X_br_imp, mapping_br = build_city_tensors(
        br_graphs, DAYS_NORMAL, DAYS_DISASTER_BR, FILTER_FACTOR_BR,
    )

    print("\n── Baton Rouge: Tucker decomposition ──")
    core_br, factors_br, core_br_imp, factors_br_imp = decompose_city(
        X_br, X_br_imp, RANKS_BR, DAYS_DISASTER_BR,
        FIRST_DAY_BR_NORMAL, FIRST_DAY_BR_DISASTER, TEMPORAL_LAMBDA_BR,
    )

    # Histogram of weight distributions across spatial components (U1/U2):
    # checks whether components are well-spread or dominated by a few zones.
    vis_heatmap_spatial_profiles(factors_br, output_dir=OUTPUT_PLOTS, tag='_br')
    # Choropleth maps: spatial loading of each destination component (U2)
    # across zones, one subplot per component (Baton Rouge only — needs geo).
    if br_gdf is not None:
        vis_map_spatial_factors(
            factors_br[1], br_gdf, mapping_br,
            component_indices=list(range(RANKS_BR[1])), ncols=4,
            output_dir=OUTPUT_PLOTS, tag='_br',
        )
    else:
        print("  ⚠ Baton Rouge geometry unavailable — skipping spatial choropleths.")
    # Heatmap of the temporal factor U3 (normal period): shows the 7-day
    # weekly template learned by Tucker for each temporal component.
    vis_heatmap_temporal_signature(
        factors_br[2], first_day=FIRST_DAY_BR_NORMAL, show_days=True,
        slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
        index_range=(0, SLOTS_ACTIVE * 7),
        output_dir=OUTPUT_PLOTS, tag='_br_normal',
    )
    # Heatmap slices of the core tensor G[:, :, ts]: interaction strength
    # between origin and destination components at each temporal component.
    for ts in [2, 3, 4, 5]:
        vis_heatmap_core_interaction(core_br, time_slice=ts, threshold=1,
                                      output_dir=OUTPUT_PLOTS)
    # Heatmap of the temporal factor U3 (disaster period): compare against
    # the normal-period template to identify disruption in travel rhythms.
    vis_heatmap_temporal_signature(
        factors_br_imp[2], first_day=FIRST_DAY_BR_DISASTER, show_days=True,
        slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
        index_range=(0, DAYS_DISASTER_BR * SLOTS_ACTIVE),
        output_dir=OUTPUT_PLOTS, tag='_br_disaster',
    )

    # ── Fort Myers ───────────────────────────────────────────────────────────
    print("\n── Fort Myers: tensors ──")
    X_fm, X_fm_imp, mapping_fm = build_city_tensors(
        fm_graphs, DAYS_NORMAL, DAYS_DISASTER_FM, FILTER_FACTOR_FM,
    )

    print("\n── Fort Myers: Tucker decomposition ──")
    core_fm, factors_fm, core_fm_imp, factors_fm_imp = decompose_city(
        X_fm, X_fm_imp, RANKS_FM, DAYS_DISASTER_FM,
        FIRST_DAY_FM_NORMAL, FIRST_DAY_FM_DISASTER, TEMPORAL_LAMBDA_FM,
    )

    # Histogram of spatial component weight distributions for Fort Myers.
    vis_heatmap_spatial_profiles(factors_fm, output_dir=OUTPUT_PLOTS, tag='_fm')
    # Heatmap of U3 temporal factor (FM normal period): weekly template.
    vis_heatmap_temporal_signature(
        factors_fm[2], first_day=FIRST_DAY_FM_NORMAL, show_days=True,
        slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
        index_range=(0, SLOTS_ACTIVE * 7),
        output_dir=OUTPUT_PLOTS, tag='_fm_normal',
    )
    # Heatmap of U3 temporal factor (FM disaster period): disruption pattern.
    vis_heatmap_temporal_signature(
        factors_fm_imp[2], first_day=FIRST_DAY_FM_DISASTER, show_days=True,
        slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
        index_range=(0, DAYS_DISASTER_FM * SLOTS_ACTIVE),
        output_dir=OUTPUT_PLOTS, tag='_fm_disaster',
    )

    # ── Temporal inter-city matching ─────────────────────────────────────────
    print("\n── Temporal inter-city matching ──")
    t_sig_br     = get_global_temporal_trends(core_br,     factors_br,     de_scale=False)
    t_sig_fm     = get_global_temporal_trends(core_fm,     factors_fm,     de_scale=False)
    t_sig_br_imp = get_global_temporal_trends(core_br_imp, factors_br_imp, de_scale=False)
    t_sig_fm_imp = get_global_temporal_trends(core_fm_imp, factors_fm_imp, de_scale=False)

    # Cross-city NORMAL matching: both normal periods are 7-day templates →
    # same length, comparable.
    coor_normal  = process_weight_matrix(
        calculate_correlation_matrix(t_sig_br, t_sig_fm), temperature=TEMPERATURE,
    )
    # Cross-city DISASTER matching only makes sense when the disaster windows are
    # the same length AND phase.  BR and FM now use a matched landfall + 14-day
    # recovery window (both 15d), so the temporal trends are length-aligned and
    # this comparison runs; the length guard below stays as a safety net.
    if t_sig_br_imp.shape[0] == t_sig_fm_imp.shape[0]:
        coor_impact = process_weight_matrix(
            calculate_correlation_matrix(t_sig_br_imp, t_sig_fm_imp),
            temperature=TEMPERATURE,
        )
    else:
        coor_impact = None
        print(f"  ⚠ cross-city disaster matching skipped: BR disaster window "
              f"({t_sig_br_imp.shape[0]} slots) ≠ FM ({t_sig_fm_imp.shape[0]} "
              f"slots) — different length/phase.")
    coor_br_p    = process_weight_matrix(
        calculate_correlation_matrix(
            broadcast_matrix_by_weekday(t_sig_br, FIRST_DAY_BR_NORMAL, FIRST_DAY_BR_DISASTER,
                                         DAYS_DISASTER_BR * SLOTS_ACTIVE),
            t_sig_br_imp,
        ), temperature=TEMPERATURE,
    )
    coor_fm_p    = process_weight_matrix(
        calculate_correlation_matrix(
            broadcast_matrix_by_weekday(t_sig_fm, FIRST_DAY_FM_NORMAL, FIRST_DAY_FM_DISASTER,
                                         DAYS_DISASTER_FM * SLOTS_ACTIVE),
            t_sig_fm_imp,
        ), temperature=TEMPERATURE,
    )

    t_sig_fm_imp_hat = reconstruct_v_no_with_coor(t_sig_br_imp, coor_normal)

    # Heatmap of component mapping weights: how each normal-period component
    # maps onto disaster-period components, for BR and FM independently.
    vis_heatmap_component_mapping(coor_br_p,   basis_name='Baton Rouge', target_name='Baton Rouge',
                                   output_dir=OUTPUT_PLOTS, tag='_period')
    vis_heatmap_component_mapping(coor_fm_p,   basis_name='Fort Myers', target_name='Fort Myers',
                                   output_dir=OUTPUT_PLOTS, tag='_period')
    # Heatmap of cross-city component mapping weights (BR → FM):
    # normal period shows structural similarity; the disaster heatmap is only
    # produced when the two disaster windows are the same length/phase.
    vis_heatmap_component_mapping(coor_normal, basis_name='Baton Rouge', target_name='Fort Myers',
                                   output_dir=OUTPUT_PLOTS, tag='_normal')
    if coor_impact is not None:
        vis_heatmap_component_mapping(coor_impact, basis_name='Baton Rouge', target_name='Fort Myers',
                                       output_dir=OUTPUT_PLOTS, tag='_disaster')

    # ── Spatiotemporal features (optional) ───────────────────────────────────
    if RUN_SPATIAL_FEATURES:
        from utils_pattern_analysis.spatial_features import (
            get_comp_features_df, vis_heatmap_component_similarity,
        )
        print("\n── Extracting OSM features ──")
        if br_gdf is None:
            print("  ⚠ Baton Rouge geometry unavailable — skipping OSM features.")
        else:
            cf_br_ori  = get_comp_features_df(br_gdf.to_crs(epsg=4326).copy(), factors_br[0] ** 2, mapping_br)
            cf_br_dest = get_comp_features_df(br_gdf.to_crs(epsg=4326).copy(), factors_br[1] ** 2, mapping_br)
            if fm_gdf is None:
                print("  ⚠ Fort Myers geometry unavailable — skipping FM OSM features "
                      "and cross-city similarity.")
            else:
                cf_fm_ori  = get_comp_features_df(fm_gdf.to_crs(epsg=4326).copy(), factors_fm[0] ** 2, mapping_fm)
                cf_fm_dest = get_comp_features_df(fm_gdf.to_crs(epsg=4326).copy(), factors_fm[1] ** 2, mapping_fm)
                # Heatmap of cosine similarity between BR and FM origin components
                # based on OSM land-use features: reveals which components represent
                # the same land-use type across the two cities.
                vis_heatmap_component_similarity(cf_br_ori, cf_fm_ori, output_dir=OUTPUT_PLOTS)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, 'tucker_results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({
            'br': {
                'core': core_br, 'factors': factors_br, 'mapping': mapping_br,
                'core_imp': core_br_imp, 'factors_imp': factors_br_imp,
            },
            'fm': {
                'core': core_fm, 'factors': factors_fm, 'mapping': mapping_fm,
                'core_imp': core_fm_imp, 'factors_imp': factors_fm_imp,
            },
            'temporal_matching': {
                'coor_normal': coor_normal, 'coor_impact': coor_impact,
                'coor_br_period': coor_br_p, 'coor_fm_period': coor_fm_p,
                't_sig_fm_imp_hat': t_sig_fm_imp_hat,
            },
        }, f)
    print(f"\nTucker results saved to {out_path}")


if __name__ == '__main__':
    main()
