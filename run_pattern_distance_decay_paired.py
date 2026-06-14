"""
Paired-NMF distance-decay analysis.

Goal
----
Answer the question that single-NMF + Figure 7 cannot:

    "Does each NMF component's distance-decay α change from the normal
     period to the disaster period?"

In the single-NMF framework of run_pattern_nmf.py, every component's
α is structurally time-invariant (H[k, :] is shared across the whole window).
To get a per-component, period-aware α we must let H itself vary by period,
i.e. run TWO NMFs — one on each half — and compare the resulting H rows.

Method
------
1. Take the last DAYS_WINDOW_<city> days of the graph sequence (trailing
   window convention — the last DAYS_WINDOW days).
2. Split the window into normal (first half) + disaster (second half).
3. Build two OD-aligned matrices (same row ordering, joint low-activity filter).
4. NMF on the normal half and NMF on the disaster half — INDEPENDENTLY
   (cold-start, NNDSVD).  A warm-start seeded with H_normal would freeze H's
   direction and only rescale it, forcing α_disaster ≡ α_normal (Δα ≡ 0); two
   independent fits let the disaster period learn its own spatial patterns.
5. Match disaster components to normal components by spatial (H-row) cosine via
   optimal assignment, and reorder the disaster factors so component k means the
   same behaviour in both periods.  The match cosine is a real 0–1 diagnostic
   (low cosine → no close spatial counterpart → its Δα is not like-for-like).
6. For each k, fit a truncated power-law on H_normal[k, :] AND H_disaster[k, :],
   giving (α_normal_k, α_disaster_k).
7. Visualise:
     - Per-period Figure-7 grid (one per half, per city)
     - Slope graph (each component a line from α_normal to α_disaster)
     - CSV: full parameter table including Δα and match cosine.

Methodology alignment with run_pattern_nmf.py
---------------------------------------------
When the two-stage approach conflicts with run_pattern_nmf.py's conventions, we
follow run_pattern_nmf.py:

  - Trailing window (both cities 22d = 7 normal + 15 disaster), not the 28+14 split
  - No segment-averaging of the normal period
  - Per-city FIRST_DAY_* (Baton Rouge = Sunday; Fort Myers = Wednesday)
  - Same FILTER_FACTOR / L1_REG / N_BEHAVIORS as run_pattern_nmf.py

This script matches components on the *spatial* factor H (not the temporal
factor), so it decomposes each period independently and aligns components
post-hoc.

Run
---
    python run_pattern_distance_decay_paired.py

Outputs (under outputs/distance_decay_paired/)
---------------------------------------------
    paired_<city>.csv                          full parameter table per component
    temporal_signature_<city>_normal.png       heatmap of W_normal
    temporal_signature_<city>_disaster.png     heatmap of W_disaster
    nmf_component_timeline_<city>.png          Figure-5 style timeline,
                                               normal (blue) + disaster (red)
    distance_decay_normal_<city>.png           Figure-7 grid, normal half
    distance_decay_disaster_<city>.png         Figure-7 grid, disaster half
    slope_<city>.png                           paired-α slope graph (key plot)
"""
import os

import numpy as np
import pandas as pd

from config import (
    BR_GRAPH_PATH, FM_GRAPH_PATH, BR_ANALYSIS_DAYS, FM_ANALYSIS_DAYS,
    BR_GEO_CSV, FM_GEO_CSV, AGG_LEVEL,
    OUTPUT_DIR, SLOT_PER_DAY, SLOTS_ACTIVE,
)

_INTERVAL_HOURS = 24 // SLOT_PER_DAY

from utils_pattern_analysis.graph_io import load_graphs
from utils_pattern_analysis.nmf_pipeline import (
    build_paired_matrices, decompose_city_paired,
)
from utils_pattern_analysis.visualization import (
    vis_grid_distance_decay, vis_slope_paired_alpha,
    vis_heatmap_temporal_signature, vis_line_nmf_component_timeline,
)
from utils_data_processing.build_graphs import load_city_geo

try:
    import powerlaw
except ImportError as e:
    raise ImportError(
        "The 'powerlaw' package (Alstott et al. 2014) is required.\n"
        "Install with:  pip install powerlaw"
    ) from e


# ── Configuration (mirror run_pattern_nmf.py where applicable) ──────────────

# Trailing window — now MATCHED across cities: both use 22d (7 normal + 15
# disaster = landfall day + 14 recovery days).  BR's data was extended through
# 2021-09-16, so BR is trimmed to 2021-09-12 (Ida + 14; see BR_ANALYSIS_DAYS in
# config) to mirror FM's Ian + 14 window.  The normal half is a 7-day weekly
# baseline for both cities.
DAYS_WINDOW_BR          = 22
DAYS_DISASTER_BR        = 15
DAYS_WINDOW_FM          = 22
DAYS_DISASTER_FM        = 15

N_BEHAVIORS_BR = 7
N_BEHAVIORS_FM = 7
L1_REG_BR      = 0.15
L1_REG_FM      = 0.15   # conservative: matched to BR (FM ≈ BR scale); retune later

FILTER_FACTOR_BR = 0
# FM's disaster window is sparse relative to BR; ff=1 keeps enough OD pairs.
# Note: FM still yields only ~5 usable components (post-Ian mobility is heavily
# disrupted); the extra components come back as "dead" and are flagged in the CSV.
FILTER_FACTOR_FM = 0

# Baton Rouge trimmed to 2021-09-12 (Ida + 14; see BR_ANALYSIS_DAYS).  Trailing
# 22-day window = Aug 22 (Sun) → Sep 12; disaster half = Aug 29 (Sun, Ida landfall)
# → Sep 12.
FIRST_DAY_BR_NORMAL   = 'Sunday'
FIRST_DAY_BR_DISASTER = 'Sunday'

# Fort Myers trimmed to Aug 30 – Oct 12 2022 (44 days; see FM_ANALYSIS_DAYS in
# config).  Trailing 22-day window = Sep 21–Oct 12; disaster half = Sep 28–Oct 12
# (Ian + 14 days recovery).  Weekdays are the real first-day-of-window dates.
FIRST_DAY_FM_NORMAL   = 'Wednesday'   # Sep 21 2022 (normal-half start)
FIRST_DAY_FM_DISASTER = 'Wednesday'   # Sep 28 2022 (disaster-half / Ian landfall)

# Fit knobs
N_SAMPLES_FIT = 50_000
SEED          = 42
NCOLS_GRID    = 3

OUTPUT_DD = os.path.join(OUTPUT_DIR, 'distance_decay_paired')


# ── Distance computation (same as the static add-on) ─────────────────────────

def build_distance_array(mapping, gdf, id_col='aggr_id'):
    """OD-pair centroid distances (km) aligned to NMF row order."""
    proj   = gdf.to_crs(epsg=3857)
    cents  = proj.geometry.centroid
    coords = pd.DataFrame({
        'cx': cents.x.values,
        'cy': cents.y.values,
    }, index=gdf[id_col].astype(str).values)

    origins = np.array([o for o, _ in mapping])
    dests   = np.array([d for _, d in mapping])
    missing = (~pd.Index(origins).isin(coords.index)) | \
              (~pd.Index(dests).isin(coords.index))
    if missing.any():
        raise KeyError(
            f"{int(missing.sum())} OD pairs reference aggr_id values not "
            f"found in the geo-dataframe."
        )

    co = coords.loc[origins][['cx', 'cy']].to_numpy()
    cd = coords.loc[dests  ][['cx', 'cy']].to_numpy()
    return np.linalg.norm(co - cd, axis=1) / 1000.0


# ── Truncated power-law fitting ──────────────────────────────────────────────

def weighted_sample(values, weights, n_samples, rng, jitter_zero=1e-3):
    """Importance-sample distances by spatial weights; jitter zeros."""
    w = np.clip(np.asarray(weights, dtype=float), 0, None)
    if w.sum() <= 0:
        return np.array([])
    p   = w / w.sum()
    idx = rng.choice(len(values), size=n_samples, replace=True, p=p)
    samples = values[idx].astype(float)
    zero_mask = samples <= 0
    if zero_mask.any():
        samples[zero_mask] = rng.uniform(jitter_zero / 2, jitter_zero,
                                          size=int(zero_mask.sum()))
    return samples


def fit_truncated_power_law(samples, min_samples=100):
    """
    Fit truncated PL via powerlaw package.  Returns None for empty / too-small
    inputs so callers can record a NaN α (typical when sparse NMF kills a
    component — its H row is all zeros, weighted_sample returns []).
    """
    if samples is None or len(samples) < min_samples:
        return None
    fit = powerlaw.Fit(samples, discrete=False, verbose=False)
    tpl = fit.truncated_power_law
    return {
        'alpha':   float(tpl.alpha),
        'lambda':  float(tpl.Lambda),
        'xmin':    float(fit.xmin),
        'ks_dist': float(tpl.D),
        '_fit':    fit,
    }


# ── Per-city driver ──────────────────────────────────────────────────────────

def run_city(city_label, graphs, gdf, n_behaviors, l1_reg, filter_factor,
             first_day_normal, first_day_disaster, days_window, days_disaster):
    """
    Run paired NMF + per-period distance-decay for one city.

    days_window / days_disaster are now matched across cities: both use 22/15
    (7-day normal baseline + landfall day + 14 recovery days).  The normal half
    is a 7-day weekly baseline by convention.

    `gdf` must be a valid GeoDataFrame — geometry is mandatory here (distance
    decay needs centroid distances); main() loads it with the strict loader.
    """
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  {city_label}")
    print(f"══════════════════════════════════════════════════════════════")

    # 1. Build the two aligned matrices.
    X_normal, X_disaster, mapping = build_paired_matrices(
        graphs, days_window, days_disaster, filter_factor,
    )

    # 2. Two independent NMFs + spatial component matching (disaster components
    #    are reordered so component k means the same behaviour in both periods).
    print(f"\n── Paired NMF: {city_label} ──")
    (W_n, H_n, weights_n,
     W_d, H_d, weights_d, match_corr) = decompose_city_paired(
        X_normal, X_disaster, n_behaviors, l1_reg=l1_reg,
    )

    # 3. Identity diagnostic: spatial cosine of each matched (normal, disaster)
    #    pair.  Low cosine = the disaster period has no close spatial counterpart
    #    for that normal component, so its Δα is not a like-for-like comparison.
    print(f"  Per-component matched spatial cosine (normal ↔ disaster):")
    for k_i, c in enumerate(match_corr):
        flag = '✓' if c > 0.8 else ('⚠' if c > 0.5 else '✗')
        print(f"    Comp {k_i}: {c:.3f}  {flag}")

    # 4. OD-pair distances (geometry is guaranteed by main()).
    distances_km = build_distance_array(mapping, gdf)
    print(f"\n  {len(distances_km):,} OD pairs;  "
          f"d ∈ [{distances_km.min():.2f}, {distances_km.max():.2f}] km, "
          f"median {np.median(distances_km):.2f} km")

    # 5. Truncated-PL fit per component, per period.  Sparse NMF can still leave
    #    a "dead" component (H row all zeros) → its α is recorded as NaN.
    fits_n, fits_d, rows = [], [], []
    print(f"\n── Fitting truncated power-law per component, per period ──")
    rng = np.random.default_rng(SEED)
    for k_i in range(H_n.shape[0]):
        n_alive = H_n[k_i, :].sum() > 0
        d_alive = H_d[k_i, :].sum() > 0
        f_n = f_d = None
        if n_alive:
            f_n = fit_truncated_power_law(
                weighted_sample(distances_km, H_n[k_i, :], N_SAMPLES_FIT, rng))
        if d_alive:
            f_d = fit_truncated_power_law(
                weighted_sample(distances_km, H_d[k_i, :], N_SAMPLES_FIT, rng))
        fits_n.append(f_n)
        fits_d.append(f_d)

        a_n = f_n['alpha'] if f_n else np.nan
        a_d = f_d['alpha'] if f_d else np.nan
        d_a = (a_d - a_n) if (f_n and f_d) else np.nan
        rows.append({
            'component':         k_i,
            'alpha_normal':      a_n,
            'alpha_disaster':    a_d,
            'delta_alpha':       d_a,
            'lambda_normal':     f_n['lambda']  if f_n else np.nan,
            'lambda_disaster':   f_d['lambda']  if f_d else np.nan,
            'xmin_normal':       f_n['xmin']    if f_n else np.nan,
            'xmin_disaster':     f_d['xmin']    if f_d else np.nan,
            'ks_normal':         f_n['ks_dist'] if f_n else np.nan,
            'ks_disaster':       f_d['ks_dist'] if f_d else np.nan,
            'weight_normal':     weights_n[k_i],
            'weight_disaster':   weights_d[k_i],
            'match_correlation': float(match_corr[k_i]),
            'alive_normal':      bool(n_alive),
            'alive_disaster':    bool(d_alive),
        })

        if f_n and f_d:
            print(f"  Comp {k_i}:  α_pre={a_n:.3f}  →  "
                  f"α_dis={a_d:.3f}  "
                  f"(Δα={d_a:+.3f}, corr={match_corr[k_i]:.2f})")
        else:
            dead_in = []
            if not f_n: dead_in.append('normal')
            if not f_d: dead_in.append('disaster')
            print(f"  Comp {k_i}:  ⚠ dead in {'+'.join(dead_in)} half "
                  f"(H row ≈ 0); α set to NaN")

    # 6. Save outputs.
    tag = city_label.lower().replace(' ', '_')
    os.makedirs(OUTPUT_DD, exist_ok=True)

    # 6a. Per-component parameter table.
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DD, f'paired_{tag}.csv'), index=False)

    # 6b. Temporal-component visualisations.
    vis_heatmap_temporal_signature(
        W_n, first_day=first_day_normal, show_days=True,
        slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
        output_dir=OUTPUT_DD, tag=f'_{tag}_normal',
    )
    vis_heatmap_temporal_signature(
        W_d, first_day=first_day_disaster, show_days=True,
        slots_per_day=SLOTS_ACTIVE, interval_hours=_INTERVAL_HOURS,
        output_dir=OUTPUT_DD, tag=f'_{tag}_disaster',
    )
    vis_line_nmf_component_timeline(
        W_n, W_d,
        first_day_normal=first_day_normal,
        first_day_disaster=first_day_disaster,
        slots_per_day=SLOTS_ACTIVE,
        output_dir=OUTPUT_DD, tag=f'_{tag}',
    )

    # 6c. Distance-decay plots.
    vis_grid_distance_decay(
        fits_n, weights_n, f'{city_label} (normal half)',
        os.path.join(OUTPUT_DD, f'distance_decay_normal_{tag}.png'),
        ncols=NCOLS_GRID,
    )
    vis_grid_distance_decay(
        fits_d, weights_d, f'{city_label} (disaster half)',
        os.path.join(OUTPUT_DD, f'distance_decay_disaster_{tag}.png'),
        ncols=NCOLS_GRID,
    )
    vis_slope_paired_alpha(
        rows, city_label,
        os.path.join(OUTPUT_DD, f'slope_{tag}.png'),
    )

    print(f"\n  Saved under {OUTPUT_DD}/:")
    print(f"    paired_{tag}.csv")
    print(f"    temporal_signature_{tag}_normal.png      (heatmap, normal NMF)")
    print(f"    temporal_signature_{tag}_disaster.png    (heatmap, disaster NMF)")
    print(f"    nmf_component_timeline_{tag}.png         (Figure-5 style)")
    print(f"    distance_decay_normal_{tag}.png          (Figure-7 grid, normal)")
    print(f"    distance_decay_disaster_{tag}.png        (Figure-7 grid, disaster)")
    print(f"    slope_{tag}.png                          (paired-α slope graph)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DD, exist_ok=True)

    # Geometry is MANDATORY for distance-decay (no geometry → no centroid
    # distances → no α).  load_city_geo raises FileNotFoundError if a city's
    # geo file is absent, so the run fails fast rather than producing empty output.
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

    # Same sanity-check as run_pattern_nmf.py.
    expected_interval = f'{_INTERVAL_HOURS}h'
    for city, gs in [('Baton Rouge', br_graphs), ('Fort Myers', fm_graphs)]:
        n = len(gs)
        actual = gs[0].graph.get('interval_duration')
        if actual is not None and actual != expected_interval:
            raise ValueError(
                f"{city}: pkl contains {actual} graphs but config expects "
                f"{expected_interval} (SLOT_PER_DAY={SLOT_PER_DAY})."
            )
        if n % SLOT_PER_DAY != 0:
            raise ValueError(
                f"{city}: {n} graphs not divisible by SLOT_PER_DAY={SLOT_PER_DAY}."
            )

    run_city('Baton Rouge', br_graphs, br_gdf,
             N_BEHAVIORS_BR, L1_REG_BR, FILTER_FACTOR_BR,
             FIRST_DAY_BR_NORMAL, FIRST_DAY_BR_DISASTER,
             DAYS_WINDOW_BR, DAYS_DISASTER_BR)
    run_city('Fort Myers', fm_graphs, fm_gdf,
             N_BEHAVIORS_FM, L1_REG_FM, FILTER_FACTOR_FM,
             FIRST_DAY_FM_NORMAL, FIRST_DAY_FM_DISASTER,
             DAYS_WINDOW_FM, DAYS_DISASTER_FM)


if __name__ == '__main__':
    main()
