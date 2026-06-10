"""
NMF (Matrix Factorization) decomposition of mobility patterns.

Cities: #1 = Baton Rouge (Ida 2021), #2 = Fort Myers (Ian 2022).

Performs NMF on each city independently, then computes inter-city component
mapping via correlation matrices.

Steps
-----
1. Load graphs for both cities.
2. Convert to 2-D flow matrices (OD-pairs × time).
3. Segment-average the normal period into a weekly template.
4. Filter low-activity OD pairs.
5. Decompose with NMF (normal period).
6. Re-decompose the disaster period with a warm-started W / H.
7. Compute inter-city and inter-period component mappings.
8. Visualise temporal signatures and spatial OD maps.
9. Save outputs.

Note on Fort Myers
------------------
This script plots OD arc maps for Baton Rouge (city #1) only.  Geometry files
are mandatory (loading raises if a geo CSV is absent).  Fort Myers
is trimmed to Aug 30 – Oct 12 2022 (44 days; see FM_ANALYSIS_DAYS in config) so
Ian's landfall (Sep 28) + ~2 weeks of recovery sit at the window end.

Run
---
    python run_pattern_nmf.py
"""
import os

from config import (
    BR_GRAPH_PATH, FM_GRAPH_PATH, BR_ANALYSIS_DAYS, FM_ANALYSIS_DAYS,
    BR_GEO_CSV, FM_GEO_CSV, AGG_LEVEL,
    OUTPUT_DIR, SLOT_PER_DAY, SLOT_TRIM_START, SLOT_TRIM_END, SLOTS_ACTIVE,
)

# Derived from config — never hardcode 3 or 5 here
_INTERVAL_HOURS = 24 // SLOT_PER_DAY   # 3 for 3h data, 2 for 2h data
from utils_pattern_analysis.graph_io import (
    load_graphs, periodic_trim,
    graphs_to_2d_matrix, filter_inactive_locations_2d,
    calculate_segment_average_2d, print_matrix_diagnostics,
)
from utils_pattern_analysis.decomposition import (
    decompose_mobility_patterns, decompose_mobility_patterns_with_init,
    generate_W_init_by_weekday, h_slice_to_od_matrix,
)
from utils_pattern_analysis.matching import (
    calculate_correlation_matrix, process_weight_matrix,
    reconstruct_v_no_with_coor, broadcast_matrix_by_weekday,
)
from utils_pattern_analysis.visualization import (
    vis_heatmap_temporal_signature, vis_map_od_flow,
    vis_heatmap_component_mapping, vis_line_nmf_component_timeline,
)
from utils_data_processing.build_graphs import load_city_geo


# ── Configuration ─────────────────────────────────────────────────────────────

DAYS_NORMAL      = 28   # first-N-days normal baseline (both cities)
# Disaster window = last-N days, now MATCHED: BR 15d (Aug 29–Sep 12, Ida + 14d)
# and FM 15d (Sep 28–Oct 12, Ian + 14d recovery).  BR is trimmed to 2021-09-12
# (see BR_ANALYSIS_DAYS in config) so its last 15 days are Ida landfall + recovery.
DAYS_DISASTER_BR = 15
DAYS_DISASTER_FM = 15

N_BEHAVIORS   = 7
L1_REG_BR     = 12  # sparsity q for Baton Rouge  (paper's Eq. 3; tune to ~0.1×mean(X_br))
L1_REG_FM     = 12  # conservative: matched to BR (FM ≈ BR scale); retune later
                    # set either to 0.0 to disable L1 regularisation for that city

# Block-group resolution: the census-tract-era value (50) over-filters here —
# at ff=50 BR keeps only ~4 OD pairs (< k), so it was lowered.  FM is ~3× sparser
# than BR, so it needs a still-lower factor to retain enough OD pairs for k=7.
FILTER_FACTOR_BR = 5    # → ~329 OD pairs (profiled, block group)
FILTER_FACTOR_FM = 1    # → ~146 OD pairs (FM sparser; retune later)

# Baton Rouge: PROPHET_START_DATE 2021-04-15 is a Thursday.  Disaster = last 15
# days of the trimmed file = Aug 29 (Sun, Ida landfall) → Sep 12.
FIRST_DAY_BR_NORMAL   = 'Thursday'   # first-28d normal start (Apr 15)
FIRST_DAY_BR_DISASTER = 'Sunday'     # last-15d disaster start (Aug 29, Ida)

# Fort Myers trimmed to Aug 30 – Oct 12 2022 (44 days; see FM_ANALYSIS_DAYS in
# config).  Normal = first 28 days (Aug 30); disaster = last 15 days (Sep 28–Oct
# 12, Ian + 14d recovery).  Weekdays are the real first-day-of-window dates.
FIRST_DAY_FM_NORMAL   = 'Tuesday'     # Aug 30 2022 (first-28d normal start)
FIRST_DAY_FM_DISASTER = 'Wednesday'   # Sep 28 2022 (last-15d disaster start / Ian)

TEMPERATURE = 0.20

OUTPUT_PLOTS  = os.path.join(OUTPUT_DIR, 'nmf')
OUTPUT_NMF_BR = os.path.join(OUTPUT_DIR, 'nmf', 'components_br')
OUTPUT_NMF_FM = os.path.join(OUTPUT_DIR, 'nmf', 'components_fm')


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_city_matrices(graphs, days_normal, days_disaster, filter_factor):
    slot_active = SLOTS_ACTIVE

    wh = periodic_trim(
        graphs[: SLOT_PER_DAY * days_normal],
        cycle_len=SLOT_PER_DAY, trim_start=SLOT_TRIM_START, trim_end=SLOT_TRIM_END,
    )
    X, mapping = graphs_to_2d_matrix(wh)
    X = calculate_segment_average_2d(X, slot_active * 7)

    wh_imp = periodic_trim(
        graphs[-SLOT_PER_DAY * days_disaster:],
        cycle_len=SLOT_PER_DAY, trim_start=SLOT_TRIM_START, trim_end=SLOT_TRIM_END,
    )
    X_imp, _ = graphs_to_2d_matrix(wh_imp)

    threshold = slot_active * 7 * filter_factor
    X, X_imp, mapping = filter_inactive_locations_2d(X, X_imp, mapping, threshold)

    print(f"  X shape: {X.shape}, X_impact shape: {X_imp.shape}")
    print_matrix_diagnostics(X,     mapping, label='normal period (7-day avg)')
    print_matrix_diagnostics(X_imp, mapping, label='disaster period (raw)')
    return X, X_imp, mapping


def decompose_city(X, X_impact, mapping, first_day_normal, first_day_disaster,
                   n_behaviors, days_disaster, l1_reg=0.0):
    slot_active = SLOTS_ACTIVE

    # No pre-normalisation: matches the paper's approach (raw flow counts input to NMF).
    # l1_reg (= paper's q) must be calibrated to your data's flow magnitude;
    # a useful starting point is  q ≈ 0.1 × mean(X).
    W, H = decompose_mobility_patterns(X.T, n_behaviors=n_behaviors, l1_reg=l1_reg)

    W_init = generate_W_init_by_weekday(
        W, first_day_normal, first_day_disaster,
        target_len=slot_active * days_disaster,
        slots_per_day=slot_active,
    )
    W_imp, H_imp = decompose_mobility_patterns_with_init(
        X_impact.T, W_init=W_init, H_init=H, n_behaviors=n_behaviors, l1_reg=l1_reg,
    )
    return W, H, W_imp, H_imp


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_NMF_BR, exist_ok=True)
    os.makedirs(OUTPUT_NMF_FM, exist_ok=True)

    # Geometry (AGG_LEVEL-aware, mandatory).  Only Baton Rouge is plotted
    # spatially here (this script maps city #1 only).
    br_gdf = load_city_geo('Baton Rouge', AGG_LEVEL, BR_GEO_CSV)
    fm_gdf = load_city_geo('Fort Myers',  AGG_LEVEL, FM_GEO_CSV)

    print("Loading Baton Rouge graphs …")
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
    print("Loading Fort Myers graphs …")
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

    print("\n── Baton Rouge ──")
    X_br, X_br_imp, mapping_br = build_city_matrices(
        br_graphs, DAYS_NORMAL, DAYS_DISASTER_BR, FILTER_FACTOR_BR,
    )

    print("\n── Fort Myers ──")
    X_fm, X_fm_imp, mapping_fm = build_city_matrices(
        fm_graphs, DAYS_NORMAL, DAYS_DISASTER_FM, FILTER_FACTOR_FM,
    )

    print("\n── NMF: Baton Rouge ──")
    W_br, H_br, W_br_imp, H_br_imp = decompose_city(
        X_br, X_br_imp, mapping_br,
        FIRST_DAY_BR_NORMAL, FIRST_DAY_BR_DISASTER, N_BEHAVIORS, DAYS_DISASTER_BR,
        l1_reg=L1_REG_BR,
    )

    print("\n── NMF: Fort Myers ──")
    W_fm, H_fm, W_fm_imp, H_fm_imp = decompose_city(
        X_fm, X_fm_imp, mapping_fm,
        FIRST_DAY_FM_NORMAL, FIRST_DAY_FM_DISASTER, N_BEHAVIORS, DAYS_DISASTER_FM,
        l1_reg=L1_REG_FM,
    )

    print("\n── Component matching ──")
    # Cross-city NORMAL matching: both cities' normal periods are averaged to a
    # 7-day template, so their temporal factors are the same length → comparable.
    coor_normal = process_weight_matrix(
        calculate_correlation_matrix(W_br, W_fm), temperature=TEMPERATURE,
    )
    # Cross-city DISASTER matching only makes sense when the two disaster windows
    # have the same length AND represent the same phase.  BR and FM now use a
    # matched landfall + 14-day recovery window (both 15d), so the temporal
    # factors are length-aligned and this comparison runs; the length guard below
    # stays as a safety net in case the per-city windows are de-synced again.
    if W_br_imp.shape[0] == W_fm_imp.shape[0]:
        coor_impact  = process_weight_matrix(
            calculate_correlation_matrix(W_br_imp, W_fm_imp), temperature=TEMPERATURE,
        )
        W_fm_imp_hat = reconstruct_v_no_with_coor(W_br_imp, coor_impact)
    else:
        coor_impact, W_fm_imp_hat = None, None
        print(f"  ⚠ cross-city disaster matching skipped: BR disaster window "
              f"({W_br_imp.shape[0]} slots) ≠ FM ({W_fm_imp.shape[0]} slots) "
              f"— different length/phase.")
    # Within-city normal→disaster matching (each city broadcasts its own 7-day
    # normal template to its own disaster length — always valid per city).
    coor_br_period = process_weight_matrix(
        calculate_correlation_matrix(
            broadcast_matrix_by_weekday(
                W_br, FIRST_DAY_BR_NORMAL, FIRST_DAY_BR_DISASTER,
                DAYS_DISASTER_BR * SLOTS_ACTIVE,
            ), W_br_imp,
        ), temperature=TEMPERATURE,
    )
    coor_fm_period = process_weight_matrix(
        calculate_correlation_matrix(
            broadcast_matrix_by_weekday(
                W_fm, FIRST_DAY_FM_NORMAL, FIRST_DAY_FM_DISASTER,
                DAYS_DISASTER_FM * SLOTS_ACTIVE,
            ), W_fm_imp,
        ), temperature=TEMPERATURE,
    )

    # # Heatmap of component mapping weights: how each normal-period component
    # # shifts into disaster-period components, for BR and FM independently.
    # vis_heatmap_component_mapping(coor_br_period,  basis_name='Baton Rouge',  target_name='Baton Rouge',
    #                                output_dir=OUTPUT_PLOTS, tag='_period')
    # vis_heatmap_component_mapping(coor_fm_period,  basis_name='Fort Myers',  target_name='Fort Myers',
    #                                output_dir=OUTPUT_PLOTS, tag='_period')
    # # Heatmap of cross-city component mapping weights (BR → FM):
    # # normal period shows structural similarity; disaster period shows how
    # # disruption propagates across cities.
    # vis_heatmap_component_mapping(coor_normal,     basis_name='Baton Rouge',  target_name='Fort Myers',
    #                                output_dir=OUTPUT_PLOTS, tag='_normal')
    # vis_heatmap_component_mapping(coor_impact,     basis_name='Baton Rouge',  target_name='Fort Myers',
    #                                output_dir=OUTPUT_PLOTS, tag='_disaster')

    # Heatmap of temporal signatures (time slots × components): shows when
    # each behavior component is active across the week.
    vis_heatmap_temporal_signature(W_br,     first_day=FIRST_DAY_BR_NORMAL,   show_days=True,
                                    slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
                                    output_dir=OUTPUT_PLOTS, tag='_br_normal')   # BR normal
    vis_heatmap_temporal_signature(W_br_imp, first_day=FIRST_DAY_BR_DISASTER, show_days=True,
                                    slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
                                    output_dir=OUTPUT_PLOTS, tag='_br_disaster') # BR disaster
    vis_heatmap_temporal_signature(W_fm,     first_day=FIRST_DAY_FM_NORMAL,   show_days=True,
                                    slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
                                    output_dir=OUTPUT_PLOTS, tag='_fm_normal')   # FM normal
    vis_heatmap_temporal_signature(W_fm_imp, first_day=FIRST_DAY_FM_DISASTER, show_days=True,
                                    slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
                                    output_dir=OUTPUT_PLOTS, tag='_fm_disaster') # FM disaster

    # Line subplots of each NMF component over the full timeline (normal template
    # followed by disaster period), styled after Fig. 5 of the reference paper:
    # blue = normal, red = disaster, grey background = weekend slots.
    vis_line_nmf_component_timeline(
        W_br, W_br_imp,
        first_day_normal=FIRST_DAY_BR_NORMAL, first_day_disaster=FIRST_DAY_BR_DISASTER,
        slots_per_day=SLOTS_ACTIVE,
        output_dir=OUTPUT_PLOTS, tag='_br',
    )
    vis_line_nmf_component_timeline(
        W_fm, W_fm_imp,
        first_day_normal=FIRST_DAY_FM_NORMAL, first_day_disaster=FIRST_DAY_FM_DISASTER,
        slots_per_day=SLOTS_ACTIVE,
        output_dir=OUTPUT_PLOTS, tag='_fm',
    )

    # Interactive HTML arc maps: one file per spatial component (Baton Rouge
    # only — this script maps city #1).
    if br_gdf is not None:
        br_gdf['lon'] = br_gdf['centroid'].to_crs(epsg=4326).x
        br_gdf['lat'] = br_gdf['centroid'].to_crs(epsg=4326).y
        for i in range(H_br.shape[0]):
            vis_map_od_flow(
                [h_slice_to_od_matrix(H_br[i: i+1, :], mapping_br)],
                gdfs=br_gdf, id_col='aggr_id', min_flow=0.5,
                max_line_width=20, alpha_range=(0.05, 0.95), curve_rad=0.3, vmax=5,
                save_dir=os.path.join(OUTPUT_NMF_BR, f'component_{i}.html'),
            )
    else:
        print("  ⚠ Baton Rouge geometry unavailable — skipping OD arc maps.")


if __name__ == '__main__':
    main()
