"""
Visualisation helpers for pattern analysis.

Most functions serve the production NMF pipeline (run_pattern_nmf.py).  The
functions marked "only used by archive/..." in their docstrings serve the
archived exploration scripts and can be skipped by production readers.
contextily is imported for the archive-only vis_map_spatial_factors basemap.
"""
import os
import numpy as np
from scipy.stats import rankdata, spearmanr, pearsonr
import pandas as pd
import matplotlib
# Headless-safe: every figure here is written to disk, never shown.  Without
# this, a detached run lazily imports the Qt GUI backend at the first
# plt.subplots and can die on a Windows access violation inside Qt.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import seaborn as sns
import contextily as cx


# ── Flow time-series (only used by archive/run_pattern_temporal_decay.py) ─────

def vis_line_total_flows(flow_list, output_dir='outputs', tag=''):
    """Line plot of a single flow time-series.
    Saves to: <output_dir>/flow_series<tag>.png
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(flow_list, marker='o', markersize=3, linestyle='-', color='#1976D2', alpha=0.8)
    plt.title('Total Network Flow over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Graph Index (Temporal Sequence)', fontsize=12)
    plt.ylabel('Total Flow Value', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'flow_series{tag}.png'), bbox_inches='tight', dpi=150)
    plt.close()


def vis_line_pct_change(df, window=1, output_dir='outputs', tag=''):
    """Percentage-change line plot with uncertainty bounds and optional moving average.
    Saves to: <output_dir>/flow_pct_change<tag>.png
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(14, 7))
    plot_df = df.copy()

    if window > 1:
        raw = plot_df['pct_change'].copy()
        for col in ('pct_change', 'pct_lower', 'pct_upper'):
            plot_df[col] = plot_df[col].rolling(window=window, min_periods=1).mean()
        plt.plot(df.index, raw, color='#1976D2', alpha=0.15, linewidth=1,
                 label='Raw (Interval)')

    label = 'Percentage Change' if window <= 1 else f'Trend ({window}-step MA)'
    plt.plot(plot_df.index, plot_df['pct_change'], label=label,
             color='#1976D2', marker='o', markersize=4, linewidth=2.5)
    plt.fill_between(plot_df.index, plot_df['pct_lower'], plot_df['pct_upper'],
                     color='#1976D2', alpha=0.2, label='Uncertainty Interval')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title('Percentage Change from Baseline' + (f' (Smoothed w/ Window={window})' if window > 1 else ''),
              fontsize=16, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Percentage Change (Decimal)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'flow_pct_change{tag}.png'), bbox_inches='tight', dpi=150)
    plt.close()
    return plot_df


def vis_line_recovery_fit(original_data, model, output_dir='outputs', tag=''):
    """Scatter of the impact series overlaid with the fitted StepWise recovery curve.
    Saves to: <output_dir>/decay_recovery_fit<tag>.png
    """
    if model is None:
        print("Model is None, cannot plot.")
        return
    os.makedirs(output_dir, exist_ok=True)
    t_orig = np.arange(len(original_data))
    t_smooth = np.linspace(0, len(original_data) - 1, 100)
    plt.figure(figsize=(10, 5))
    plt.scatter(t_orig, original_data, color='#D81B60', alpha=0.6,
                label='Original Data', zorder=2)
    plt.plot(t_smooth, model.predict(t_smooth), color='#1976D2', linewidth=2, zorder=3)
    plt.title('Impact Period: Original vs. Fitted Reconstruction', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps since Impact Start', fontsize=12)
    plt.ylabel('Percentage Change', fontsize=12)
    plt.legend(frameon=True); plt.grid(True, linestyle=':', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'decay_recovery_fit{tag}.png'), bbox_inches='tight', dpi=150)
    plt.close()


# ── Temporal signatures ───────────────────────────────────────────────────────

def vis_heatmap_temporal_signature(
    matrix, first_day="Monday", show_days=True, show_components=True,
    index_range=None, slots_per_day=8, interval_hours=2,
    output_dir='outputs', tag='',
):
    """
    Heatmap of a temporal factor matrix (rows=time slots, cols=components)
    with day-of-week and time-range annotations on the left axis.
    Saves to: <output_dir>/heatmap_temporal_signature<tag>.png

    Parameters
    ----------
    slots_per_day  : number of ACTIVE slots per day (= SLOTS_ACTIVE from config)
    interval_hours : hours per slot (= 24 // SLOT_PER_DAY from config)
                     used to generate time-range labels like "6-8", "8-10", …
    """
    if index_range is not None:
        s, e = index_range
        matrix = matrix[max(0, s): min(matrix.shape[0], e), :]
        offset = max(0, s)
    else:
        offset = 0

    n_steps, n_comps = matrix.shape
    # Slot labels computed from resolution — active slots always start at 06:00
    start_hour  = 6
    slot_labels = [
        f"{start_hour + i * interval_hours}-{start_hour + (i + 1) * interval_hours}"
        for i in range(slots_per_day)
    ]
    days        = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    start_di    = days.index(first_day.capitalize())

    plt.figure(figsize=(n_comps * 1.5, max(4, n_steps * 0.2)))
    ax = sns.heatmap(
        matrix, cbar=True, cmap='viridis',
        xticklabels=[f"Comp {i}" if show_components else "" for i in range(n_comps)],
        yticklabels=False,
    )

    if show_days:
        for i in range(n_steps):
            orig   = i + offset
            day_n  = orig // slots_per_day
            slot_i = orig % slots_per_day
            if slot_i == 0 or i == 0:
                ax.text(-0.8, i + 0.5,
                        days[(start_di + day_n) % 7],
                        va='center', ha='right', fontsize=9)
            ax.text(-0.05, i + 0.5, slot_labels[slot_i],
                    va='center', ha='right', fontsize=8, color='gray')
            if slot_i == 0 and i > 0:
                ax.axhline(i, color='white', lw=1.5, alpha=0.7)

    if show_components:
        ax.xaxis.tick_top(); plt.xticks(fontsize=9)
    else:
        ax.set_xticklabels([])

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_temporal_signature{tag}.png'), bbox_inches='tight', dpi=150)
    plt.close()


# ── Tucker factor plots (only used by archive/run_pattern_tucker.py) ──────────

def vis_heatmap_spatial_profiles(factors, output_dir='outputs', tag=''):
    """Side-by-side heatmaps of origin (U1) and destination (U2) spatial factor matrices.
    Saves to: <output_dir>/spatial_profiles<tag>.png
    """
    os.makedirs(output_dir, exist_ok=True)
    U_ori, U_dest, _ = factors
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(U_ori,  ax=axes[0], cmap='YlGnBu')
    axes[0].set_title("Origin Profiles (U1)")
    sns.heatmap(U_dest, ax=axes[1], cmap='YlOrRd')
    axes[1].set_title("Destination Profiles (U2)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'spatial_profiles{tag}.png'), bbox_inches='tight', dpi=150)
    plt.close()


def vis_map_spatial_factors(U_ori, gdf, spatial_mapping,
                             component_indices=None, ncols=3,
                             output_dir='outputs', tag=''):
    """Choropleth maps for selected columns of U_ori overlaid on a contextily basemap.
    Saves to: <output_dir>/spatial_factors<tag>.png
    """
    if component_indices is None:
        component_indices = list(range(U_ori.shape[1]))
    plot_gdf = gdf.copy()
    n_comps  = len(component_indices)
    nrows    = int(np.ceil(n_comps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes = np.array([axes]).flatten() if n_comps == 1 else axes.flatten()
    cmaps = ['viridis','plasma','inferno','magma','cividis','YlGnBu']

    for i, comp_idx in enumerate(component_indices):
        ax = axes[i]
        col = f'factor_{comp_idx}'
        fv  = {spatial_mapping[k]: U_ori[k, comp_idx] for k in range(len(spatial_mapping))}
        plot_gdf[col] = plot_gdf['aggr_id'].map(fv)
        gdf_3857 = plot_gdf.dropna(subset=[col]).to_crs(epsg=3857)
        gdf_3857.plot(column=col, ax=ax, cmap=cmaps[i % len(cmaps)],
                      alpha=0.6, legend=True,
                      legend_kwds={'shrink': 0.5, 'label': 'Factor Loading'})
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        ax.set_title(f"Spatial Factor {comp_idx}", fontsize=14, fontweight='bold')
        ax.set_axis_off()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'spatial_factors{tag}.png'), bbox_inches='tight', dpi=150)
    plt.close()


def vis_heatmap_core_interaction(core, time_slice=0, sum_temporal=False, threshold=0.1,
                                  output_dir='outputs', tag=''):
    """Heatmap of the Tucker core tensor G[:, :, time_slice]: origin-component ×
    destination-component interaction strength for a single temporal component.
    Saves to: <output_dir>/core_interaction_ts<time_slice><tag>.png
              (or core_interaction_summed<tag>.png when sum_temporal=True)
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(7, 6))
    if sum_temporal:
        data  = np.sum(core, axis=2)
        title = "Core Interaction (G) – Summed"
        fname = f'core_interaction_summed{tag}.png'
    else:
        data  = core[:, :, time_slice]
        title = f"Core Interaction (G) for Temporal Component {time_slice}"
        fname = f'core_interaction_ts{time_slice}{tag}.png'
    labels = np.where(np.abs(data) > threshold,
                      np.around(data, 2).astype(str), "")
    sns.heatmap(data, annot=labels, fmt="", cmap='viridis',
                annot_kws={"color": "white"})
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel("Destination Factor (U2)"); plt.ylabel("Origin Factor (U1)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
    plt.close()


def vis_heatmap_component_mapping(weights, basis_name, target_name,
                                   output_dir='outputs', tag=''):
    """Annotated heatmap of the softmax-weighted component mapping matrix
    (rows=basis components, cols=target components).
    Saves to: <output_dir>/component_mapping_<basis>_to_<target><tag>.png
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        weights, annot=True, fmt=".3f", cmap='YlGnBu',
        xticklabels=[f'{target_name} Comp {i}' for i in range(weights.shape[1])],
        yticklabels=[f'{basis_name} Comp {i}' for i in range(weights.shape[0])],
    )
    plt.title(f'Component Mapping: {basis_name} → {target_name}',
              fontsize=14, pad=20, fontweight='bold')
    plt.xlabel(f'{target_name} (Target)', fontsize=12)
    plt.ylabel(f'{basis_name} (Basis)',   fontsize=12)
    plt.xticks(rotation=45); plt.yticks(rotation=0)
    fname = (f"component_mapping"
             f"_{basis_name.replace(' ', '_')}"
             f"_to_{target_name.replace(' ', '_')}"
             f"{tag}.png")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
    plt.close()


# ── NMF component timeline (Figure-5 style) ───────────────────────────────────

def vis_line_nmf_component_timeline(
    W_normal, W_imp, W_buffer=None,
    first_day_normal='Thursday', first_day_disaster='Saturday',
    slots_per_day=None,
    output_dir='outputs', tag='', order=None,
):
    """
    Line subplots (one per NMF component) showing temporal factors across the
    normal weekly template, an optional pre-landfall buffer, and the disaster
    period.

    - Normal period plotted in blue; buffer (if given) in amber; disaster in red.
    - Weekend slots shaded light grey; weekday slots left white.
    - Dashed vertical lines mark the segment boundaries (black = landfall).
    - X-axis labelled with three-letter day abbreviations at the start of each day.
    - W_buffer sits between the two segments in calendar time; its weekdays
      continue from first_day_normal (the window is contiguous).

    Parameters
    ----------
    slots_per_day : int
        Number of ACTIVE slots per day (= SLOTS_ACTIVE from config).
        Required — pass SLOTS_ACTIVE explicitly so the day boundaries are
        correct for both 2h (8 slots) and 3h (5 slots) resolution.
    order : list[int] | None
        Component indices in top-to-bottom plotting order.  None keeps the
        natural 0..k-1 order.  Each subplot is labelled with its TRUE
        component index, so a custom order (e.g. sorted by a functional
        share) stays readable.

    Saves to: <output_dir>/line_component_timeline<tag>.png
    """
    if slots_per_day is None:
        raise ValueError(
            "slots_per_day is required. Pass SLOTS_ACTIVE from config "
            "(5 for 3h resolution, 8 for 2h resolution)."
        )
    os.makedirs(output_dir, exist_ok=True)

    DAYS       = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                  'Friday', 'Saturday', 'Sunday']
    WEEKEND    = {5, 6}           # Saturday, Sunday indices
    COLOR_NOR  = '#1976D2'        # blue  – normal period
    COLOR_BUF  = '#F9A825'        # amber – pre-landfall buffer/alert period
    COLOR_DIS  = '#D32F2F'        # red   – disaster period
    COLOR_WKD  = '#EBEBEB'        # light grey – weekend background

    n_nor   = W_normal.shape[0]
    n_buf   = 0 if W_buffer is None else W_buffer.shape[0]
    n_dis   = W_imp.shape[0]
    n_total = n_nor + n_buf + n_dis
    k       = W_normal.shape[1]
    order   = list(range(k)) if order is None else list(order)

    parts = [W_normal] + ([W_buffer] if n_buf else []) + [W_imp]
    W_all = np.concatenate(parts, axis=0)

    # Day-of-week index for every time slot; the buffer continues from the
    # normal anchor (the window is contiguous in calendar time).
    si_nor = DAYS.index(first_day_normal.capitalize())
    si_buf = (si_nor + n_nor // slots_per_day) % 7
    si_dis = DAYS.index(first_day_disaster.capitalize())
    day_idx = (
        [(si_nor + t // slots_per_day) % 7 for t in range(n_nor)] +
        [(si_buf + t // slots_per_day) % 7 for t in range(n_buf)] +
        [(si_dis + t // slots_per_day) % 7 for t in range(n_dis)]
    )

    # X-tick positions and labels (first slot of each day)
    tick_pos, tick_lbl = [], []
    for t in range(n_total):
        if t < n_nor:
            slot_in_day = t % slots_per_day
        elif t < n_nor + n_buf:
            slot_in_day = (t - n_nor) % slots_per_day
        else:
            slot_in_day = (t - n_nor - n_buf) % slots_per_day
        if slot_in_day == 0:
            tick_pos.append(t)
            tick_lbl.append(DAYS[day_idx[t]][:3])

    fig, axes = plt.subplots(k, 1, figsize=(14, 2.5 * k), sharex=True)
    if k == 1:
        axes = [axes]

    def _shade_weekends(ax):
        """Fill continuous weekend blocks with light grey."""
        in_wkd, start = False, None
        for t in range(n_total):
            if day_idx[t] in WEEKEND:
                if not in_wkd:
                    start, in_wkd = t, True
            else:
                if in_wkd:
                    ax.axvspan(start - 0.5, t - 0.5,
                               color=COLOR_WKD, zorder=0, linewidth=0)
                    in_wkd = False
        if in_wkd:
            ax.axvspan(start - 0.5, n_total - 0.5,
                       color=COLOR_WKD, zorder=0, linewidth=0)

    for row, comp in enumerate(order):
        ax = axes[row]
        _shade_weekends(ax)
        # Normal period line
        ax.plot(range(n_nor), W_all[:n_nor, comp],
                color=COLOR_NOR, linewidth=1.6, zorder=2)
        # Buffer (alert) period line
        if n_buf:
            ax.plot(range(n_nor, n_nor + n_buf), W_all[n_nor:n_nor + n_buf, comp],
                    color=COLOR_BUF, linewidth=1.6, zorder=2)
        # Disaster period line
        ax.plot(range(n_nor + n_buf, n_total), W_all[n_nor + n_buf:, comp],
                color=COLOR_DIS, linewidth=1.6, zorder=2)
        # Boundaries: buffer start (grey, if any) and landfall (black)
        if n_buf:
            ax.axvline(n_nor - 0.5, color='grey', linestyle='--',
                       linewidth=1.0, alpha=0.6, zorder=3)
        ax.axvline(n_nor + n_buf - 0.5, color='black', linestyle='--',
                   linewidth=1.1, alpha=0.6, zorder=3)
        ax.set_ylabel(f'Comp {comp}', fontsize=10, fontweight='bold', labelpad=4)
        ax.set_xlim(-0.5, n_total - 0.5)
        ax.grid(axis='y', linestyle=':', alpha=0.35)
        ax.tick_params(axis='y', labelsize=8)

    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_lbl, fontsize=8)
    axes[-1].set_xlabel('Day', fontsize=11, labelpad=6)

    # Region labels above the top subplot
    axes[0].text((n_nor / 2) / n_total, 1.04, 'Normal (weekly template)',
                 transform=axes[0].transAxes, ha='center', va='bottom',
                 fontsize=9, color=COLOR_NOR, fontweight='bold')
    if n_buf:
        axes[0].text((n_nor + n_buf / 2) / n_total, 1.04, 'Buffer',
                     transform=axes[0].transAxes, ha='center', va='bottom',
                     fontsize=9, color=COLOR_BUF, fontweight='bold')
    axes[0].text((n_nor + n_buf + n_dis / 2) / n_total, 1.04, 'Disaster period',
                 transform=axes[0].transAxes, ha='center', va='bottom',
                 fontsize=9, color=COLOR_DIS, fontweight='bold')

    # Legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=COLOR_NOR,  label='Normal period'),
        *([mpatches.Patch(color=COLOR_BUF, label='Buffer (pre-landfall)')]
          if n_buf else []),
        mpatches.Patch(color=COLOR_DIS,  label='Disaster period'),
        mpatches.Patch(color=COLOR_WKD,  label='Weekend'),
    ]
    axes[0].legend(handles=handles, loc='upper right', fontsize=8,
                   frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'line_component_timeline{tag}.png'),
                bbox_inches='tight', dpi=150)
    plt.close()


# ── Distance decay (only used by archive/run_pattern_distance_decay_paired.py) ─
#
# Pure plotting helper for per-component distance-decay fits.  The truncated
# power-law fitting itself happens upstream (see run_pattern_nmf.py's
# add-on block) and the result is passed in.  No `powerlaw` import is needed
# here — `fits[i]['_fit']` carries the powerlaw.Fit object whose `.plot_pdf`
# method we call.

def vis_grid_distance_decay(fits, weights, city_label, output_path, ncols=3):
    """
    Figure-7 style grid: one log-log panel per NMF component showing the
    empirical OD-distance PDF (above x_min) and the fitted truncated
    power-law.

    Parameters
    ----------
    fits : list of dict
        Each entry is the output of fit_truncated_power_law() — must contain
        'alpha', 'lambda', 'xmin', and '_fit' (a powerlaw.Fit object whose
        `plot_pdf` and `truncated_power_law.plot_pdf` methods are called).
    weights : 1-D array
        Per-component importance (from normalize_nmf_components); used only
        in the panel title.
    city_label : str
        Title prefix (e.g. 'Baton Rouge').
    output_path : str
        Full destination PNG path; parent directory is auto-created.
    ncols : int, optional
        Subplots per row (default 3).
    """
    k     = len(fits)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.8 * ncols, 4 * nrows),
                              squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= k:
            ax.axis('off'); continue
        fit = fits[i]
        if fit is None:
            # Dead component (sparse NMF set its H row to zero)
            ax.axis('off')
            ax.text(0.5, 0.5,
                    f'Comp {i}\n(dead — H row ≈ 0)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, alpha=0.5)
            continue
        fit['_fit'].plot_pdf(ax=ax, color='#1976D2', linewidth=1.6,
                              original_data=False, label='Empirical (≥ x_min)')
        fit['_fit'].truncated_power_law.plot_pdf(
            ax=ax, color='#D32F2F', linestyle='--', linewidth=1.6,
            label='Truncated PL fit',
        )
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'Comp {i}  (weight={weights[i]:.2g})', fontsize=11)
        ax.text(
            0.97, 0.95,
            f"α = {fit['alpha']:.2f}\n"
            f"λ = {fit['lambda']:.3f}\n"
            f"x_min = {fit['xmin']:.2f} km",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
        )
        ax.legend(fontsize=8, loc='lower left')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('P(x)')
        ax.grid(True, which='both', linestyle=':', alpha=0.4)

    fig.suptitle(f'{city_label} — per-component distance decay '
                  '(truncated power law)',
                  fontsize=13, fontweight='bold')
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def vis_slope_paired_alpha(rows, city_label, output_path):
    """
    Paired-NMF slope graph: each NMF component shown as a line from its
    normal-half α to its disaster-half α.

    Lines are colour-coded by the sign and magnitude of Δα = α_dis − α_pre:
        red    → Δα > 0  (distance friction intensified)
        blue   → Δα < 0  (distance friction relaxed)
        grey   → |Δα| < 0.1 (essentially unchanged)
    Line thickness ∝ |Δα|.

    Each component is annotated with its index, weight, and match correlation
    (how well the warm-started disaster component preserved its identity).

    Parameters
    ----------
    rows : list of dict
        One entry per component, with keys at least:
            component, alpha_normal, alpha_disaster,
            weight_normal, weight_disaster, match_correlation
    city_label : str
    output_path : str
        Full destination PNG path; parent directory is auto-created.
    """
    fig, ax = plt.subplots(figsize=(8, max(5, 0.6 * len(rows) + 3)))
    x_left, x_right = 0.0, 1.0
    cmap_red  = '#D32F2F'
    cmap_blue = '#1976D2'
    cmap_grey = '#9E9E9E'
    cmap_dead = '#BDBDBD'

    alphas_n = np.array([r['alpha_normal']   for r in rows], dtype=float)
    alphas_d = np.array([r['alpha_disaster'] for r in rows], dtype=float)
    deltas   = alphas_d - alphas_n
    valid    = ~np.isnan(deltas)
    max_abs  = max(np.abs(deltas[valid]).max(), 1e-9) if valid.any() else 1.0

    for r, dn, dd, dlt in zip(rows, alphas_n, alphas_d, deltas):
        if np.isnan(dn) and np.isnan(dd):
            # Dead in both halves — skip entirely
            continue
        if np.isnan(dn) or np.isnan(dd):
            # Dead in one half — show a single dot at the surviving end
            xy = (x_left, dn) if not np.isnan(dn) else (x_right, dd)
            ax.plot(*xy, 'x', color=cmap_dead, markersize=10,
                    markeredgewidth=2)
            ax.annotate(
                f"C{r['component']} (dead in "
                f"{'disaster' if np.isnan(dd) else 'normal'})",
                xy=xy,
                xytext=(8, 0), textcoords='offset points',
                ha='left', va='center', fontsize=9, color=cmap_dead,
            )
            continue
        colour = cmap_grey
        if abs(dlt) >= 0.1:
            colour = cmap_red if dlt > 0 else cmap_blue
        lw = 1.0 + 4.0 * abs(dlt) / max_abs
        ax.plot([x_left, x_right], [dn, dd],
                '-o', color=colour, linewidth=lw, markersize=6, alpha=0.85)
        ax.annotate(f"C{r['component']}", xy=(x_left - 0.02, dn),
                    ha='right', va='center', fontsize=9, color=colour)
        ax.annotate(
            f"C{r['component']}  Δα={dlt:+.2f}  "
            f"(corr={r['match_correlation']:.2f})",
            xy=(x_right + 0.02, dd),
            ha='left', va='center', fontsize=9, color=colour,
        )

    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(['Normal half', 'Disaster half'])
    ax.set_xlim(-0.3, 1.6)
    ax.set_ylabel(r'Distance-decay $\alpha$  (truncated PL)')
    ax.set_title(f'{city_label} — per-component α: normal → disaster')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.tick_params(axis='x', labelsize=11)

    # Legend
    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color=cmap_red,  linewidth=2,
                       label='Δα > 0  distance friction ↑'),
        mlines.Line2D([], [], color=cmap_blue, linewidth=2,
                       label='Δα < 0  distance friction ↓'),
        mlines.Line2D([], [], color=cmap_grey, linewidth=2,
                       label='|Δα| < 0.1  ~unchanged'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(0, 1.0), fontsize=9, frameon=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Origin × Destination functional heatmaps (paper Fig. 9) ───────────────────

def vis_heatmap_od_function(M, categories, weights=None, ncols=3,
                            title=None, save_path=None, cmap='YlGnBu'):
    """
    Grid of per-component origin×destination functional heatmaps (paper Fig. 9).

    Parameters
    ----------
    M          : ndarray [k × C × C]  per-component proportions (rows = origin
                 function, cols = destination function).
    categories : list[str]            axis labels (length C).
    weights    : ndarray [k] | None   component importances; if given, panels are
                 ordered by descending weight and the weight is shown per panel.
    ncols      : int                  panels per row in the grid.
    title      : str | None           figure suptitle.
    save_path  : str | None           PNG path; created if its directory is absent.

    Returns the Matplotlib Figure.
    """
    import math
    M = np.asarray(M, dtype=float)
    k, C = M.shape[0], len(categories)
    ncols = max(1, min(ncols, k))
    nrows = math.ceil(k / ncols)
    order = list(range(k))   # Panel layout follows the component index
    vmax = M.max() if M.size and M.max() > 0 else 1.0

    # Font sizes.
    FS_ANNOT, FS_TICK, FS_AXLABEL, FS_TITLE, FS_CBAR = 11, 12, 13, 15, 12

    # constrained_layout guarantees panels / rotated tick labels / colorbar never
    # overlap; tightened pads below keep the grid compact.
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 5.0 * nrows),
                             squeeze=False, constrained_layout=True)
    try:
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.02, hspace=0.04)
    except Exception:
        pass

    im = None
    for pos, comp in enumerate(order):
        ax = axes[pos // ncols][pos % ncols]
        im = ax.imshow(M[comp], cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
        for a in range(C):
            for b in range(C):
                v = M[comp, a, b]
                if v > 0:
                    ax.text(b, a, f'{v:.2f}', ha='center', va='center', fontsize=FS_ANNOT,
                            color='white' if v > 0.6 * vmax else 'black')
        ax.set_xticks(range(C)); ax.set_yticks(range(C))
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=FS_TICK)
        ax.set_yticklabels(categories, fontsize=FS_TICK)
        ax.set_xlabel('destination', fontsize=FS_AXLABEL)
        ax.set_ylabel('origin', fontsize=FS_AXLABEL)
        ttl = f'Component {comp}'
        if weights is not None:
            ttl += f'  (w={weights[comp]:.0f})'
        ax.set_title(ttl, fontsize=FS_TITLE)

    for pos in range(k, nrows * ncols):                  # hide unused panels
        axes[pos // ncols][pos % ncols].axis('off')

    if title:
        fig.suptitle(title, fontsize=FS_TITLE + 2)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        cbar.set_label('within-component flow proportion', fontsize=FS_AXLABEL)
        cbar.ax.tick_params(labelsize=FS_CBAR)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)            # constrained_layout handles margins
        plt.close(fig)
    return fig


def vis_hist_function_entropy(entropy_df, lambda_ctx=None, max_entropy=None,
                              title=None, save_path=None, nbins=12):
    """
    Distribution, ACROSS components, of per-component functional entropy — one
    histogram of the MERGED exposure entropy (component_function_entropy,
    computed from the functionality-heatmap M with the outflow and inflow
    marginals summed before H is taken, matching func_<cat> = share_from_<cat>
    + share_to_<cat>).  The spread of those values over the components is the
    plotted distribution.

    LOWER entropy = the component's flow is concentrated on fewer functions (more
    functionally specialised); HIGHER = spread across many.  A dashed line marks
    the mean, a rug shows the individual components, and a grey dotted line
    marks the theoretical maximum ln K.

    The x axis is FIXED to the full theoretical range [0, max_entropy] rather
    than each unit's own data range, so the figures of different city-events sit
    on one scale and can be compared directly.  Without max_entropy it falls
    back to the observed range (single-unit use only).

    lambda_ctx (context-aware strength) is shown in the title; the caller encodes
    it in save_path so runs at different strengths can be compared.

    Returns the Matplotlib Figure.
    """
    vals = np.asarray(entropy_df['entropy'].dropna(), dtype=float)
    if max_entropy is not None:
        lo, hi = 0.0, float(max_entropy)
    elif vals.size and np.ptp(vals) > 0:
        lo, hi = float(vals.min()), float(vals.max())
    else:
        lo, hi = 0.0, 1.0
    bins = np.linspace(lo, hi, nbins + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    color = '#1976D2'
    if vals.size:
        ax.hist(vals, bins=bins, color=color, alpha=0.55, edgecolor='white',
                linewidth=0.6,
                label=f'merged exposure   mean={vals.mean():.2f}, n={vals.size}')
        sns.rugplot(x=vals, ax=ax, color=color, height=0.06, alpha=0.8, lw=1.4)
        ax.axvline(vals.mean(), color=color, linestyle='--', linewidth=1.6)
    ax.set_xlim(lo, hi)

    if max_entropy is not None:
        ax.axvline(max_entropy, color='grey', linestyle=':', linewidth=1.4)
        ax.text(max_entropy, ax.get_ylim()[1] * 0.98,
                f'  max = ln K = {max_entropy:.2f}', color='grey', fontsize=9,
                va='top', ha='right', rotation=90)

    ttl = title or 'Functional entropy distribution'
    if lambda_ctx is not None:
        ttl += f'  (context-aware λ = {lambda_ctx:g})'
    ax.set_title(ttl, fontsize=14)
    ax.set_xlabel('Shannon entropy of functional distribution (nats) '
                  '— lower = more concentrated', fontsize=11)
    ax.set_ylabel('number of components', fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    ax.margins(x=0.02)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# ── Feature correlation heatmap (split from/to cells) ─────────────────────────


def vis_bar_component_distance(spatial_df, weights=None, title=None,
                               save_path=None):
    """
    Per-component loading-weighted flow distance: one horizontal bar per
    component (bar = mean distance, whisker = ±1 weighted SD of the component's
    flow distances), sorted shortest→longest.  If weights are given, bars are
    coloured by component importance so you can see whether long- or short-range
    components carry the weight.  Components with NaN distance (zero loading mass)
    are dropped.
    """
    df = spatial_df.dropna(subset=['mean_distance']).copy()
    df = df.loc[df['mean_distance'].sort_values().index]      # shortest at bottom
    n = len(df)
    ypos = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.34 * n + 1.6)))
    ekw = dict(ecolor='grey', lw=1, alpha=0.6)
    if weights is not None and n:
        w = np.asarray(weights, dtype=float)[df.index.to_numpy()]
        norm = mcolors.Normalize(vmin=float(np.nanmin(w)), vmax=float(np.nanmax(w)))
        ax.barh(ypos, df['mean_distance'], xerr=df['std_distance'],
                color=cm.viridis(norm(w)), error_kw=ekw)
        sm = cm.ScalarMappable(norm=norm, cmap='viridis'); sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.02).set_label('component weight', fontsize=10)
    else:
        ax.barh(ypos, df['mean_distance'], xerr=df['std_distance'],
                color='#1976D2', error_kw=ekw)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f'comp {i}' for i in df.index], fontsize=9)
    ax.set_xlabel('loading-weighted flow distance (km)   '
                  '[bar = mean, whisker = ±1 weighted SD]', fontsize=10)
    ax.set_title(title or 'Per-component flow distance', fontsize=13)
    ax.margins(y=0.01)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def vis_bar_component_income(income_df, weights=None, title=None, save_path=None):
    """
    Per-component loading-weighted median household income: one horizontal bar per
    component (USD), sorted lowest→highest.  If weights are given, bars are
    coloured by component importance so you can see whether high- or low-income
    components carry the weight.  Components with NaN income (no valid-income
    loading mass) are dropped.
    """
    df = income_df.dropna(subset=['median_income']).copy()
    df = df.loc[df['median_income'].sort_values().index]      # lowest at bottom
    n = len(df)
    ypos = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.34 * n + 1.6)))
    if weights is not None and n:
        w = np.asarray(weights, dtype=float)[df.index.to_numpy()]
        norm = mcolors.Normalize(vmin=float(np.nanmin(w)), vmax=float(np.nanmax(w)))
        ax.barh(ypos, df['median_income'], color=cm.viridis(norm(w)))
        sm = cm.ScalarMappable(norm=norm, cmap='viridis'); sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.02).set_label('component weight', fontsize=10)
    else:
        ax.barh(ypos, df['median_income'], color='#1976D2')
    ax.set_yticks(ypos)
    ax.set_yticklabels([f'comp {i}' for i in df.index], fontsize=9)
    ax.set_xlabel('loading-weighted median household income (USD)', fontsize=10)
    ax.set_title(title or 'Per-component median household income', fontsize=13)
    ax.margins(y=0.01)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def vis_heatmap_corr(rho, pval=None, time_cols=None,
                     categories=None, save_path=None,
                     cmap='RdBu_r', annot_fs=17, extra_cols=None,
                     row_gaps=(), row_gap_size=0.45):
    """
    Row-scaled correlation heatmap.  Rows are any metric set (resilience
    metrics, weekday_ratio, …); every column renders as ONE solid cell.

    Functional categories are read from the MERGED column func_<cat>
    (= share_from_<cat> + share_to_<cat>, the total share of the component's
    flow touching that function).  The merge happens upstream, before the
    correlation is computed, so one number per category describes the whole
    relationship — the earlier from/to split reported two correlations of two
    halves of the same quantity, which invited reading a direction difference
    into what is one functional exposure.  Single-value columns sit on the LEFT
    (time_cols) or on the RIGHT after the categories (extra_cols, e.g. a spatial
    or socioeconomic feature).  Each row is coloured by its own max |rho| over
    its displayed cells, so rows are independent and no colorbar is drawn —
    colours compare within a row only, numbers compare everywhere.

    Parameters
    ----------
    rho, pval  : DataFrames [metric rows × feature columns]; columns must contain
                 time_cols, extra_cols, and func_<cat> for every category.
    time_cols  : list[str] single-cell columns shown on the LEFT (may be empty).
    categories : list[str] functional category names (e.g. SF_CATEGORIES); each
                 reads the merged func_<cat> column.
    extra_cols : list[str] single-cell columns shown on the RIGHT, after the
                 categories (e.g. mean_distance, median_income); may be empty.
    row_gaps   : iterable[int] row indices BEFORE which a blank band is left, to
                 separate groups of rows that are read as different families
                 (e.g. a ratio, then day-period bands, then per-slot shares).
                 Purely visual — the values and the per-row colour scaling are
                 untouched.
    row_gap_size : float height of that blank band, in row units.
    """
    time_cols  = time_cols or []
    extra_cols = extra_cols or []
    categories = categories or []
    rows = list(rho.index)
    n_r = len(rows)
    n_cols = len(time_cols) + len(categories) + len(extra_cols)

    # One value per display cell; the column a category reads is the merged
    # func_<cat>.
    V = np.full((n_r, n_cols), np.nan)
    names = list(time_cols) + [f'func_{c}' for c in categories] + list(extra_cols)
    for i, rname in enumerate(rows):
        for j, col in enumerate(names):
            V[i, j] = rho.loc[rname, col]

    # Per-row colour scale over every displayed value.
    scale = np.nanmax(np.abs(V), axis=1, keepdims=True)
    scale = np.where(np.isnan(scale) | (scale == 0), 1.0, scale)
    C = V / scale

    # Top edge of each row, shifted down by row_gap_size at every requested gap.
    gaps = set(int(g) for g in (row_gaps or ()))
    ytop, y = np.empty(n_r), 0.0
    for i in range(n_r):
        if i in gaps:
            y += row_gap_size
        ytop[i] = y
        y += 1.0
    total_h = y

    fig, ax = plt.subplots(figsize=(2.0 * n_cols + 2.4, 1.75 * total_h + 2.2),
                           constrained_layout=True)
    # One imshow per row so the blank bands stay empty (a single image cannot
    # skip rows); each row still spans the full column range.
    for i in range(n_r):
        ax.imshow(C[i:i + 1], cmap=cmap, vmin=-1, vmax=1, aspect='auto',
                  extent=[0, n_cols, ytop[i] + 1.0, ytop[i]],
                  interpolation='nearest')
        for x in range(n_cols + 1):                       # inner cell borders
            ax.plot([x, x], [ytop[i], ytop[i] + 1.0], color='white', lw=2,
                    zorder=3)
        for yy in (ytop[i], ytop[i] + 1.0):
            ax.plot([0, n_cols], [yy, yy], color='white', lw=2, zorder=3)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(total_h, 0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    def _stars(rname, col):
        if pval is None or np.isnan(pval.loc[rname, col]):
            return ''
        p = pval.loc[rname, col]
        return '**' if p < 0.01 else ('*' if p < 0.05 else '')

    for i, rname in enumerate(rows):
        for j, col in enumerate(names):
            v = V[i, j]
            if np.isnan(v):
                continue
            ax.text(j + 0.5, ytop[i] + 0.5, f'{v:.2f}{_stars(rname, col)}',
                    ha='center', va='center', fontsize=annot_fs, zorder=4,
                    color='white' if abs(C[i, j]) > 0.6 else 'black')

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(list(time_cols) + list(categories) + list(extra_cols),
                       rotation=45, ha='right', fontsize=18)
    ax.set_yticks(ytop + 0.5)
    ax.set_yticklabels(rows, fontsize=18)
    ax.tick_params(length=0)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def vis_bar_cross_city_resi_pred(df, gt_col='cum_loss_gt',
                                 pred_cols=(('Prediction (kNN)', 'cum_loss_pred_knn', 'kNN'),
                                            ('Prediction (ridge)', 'cum_loss_pred_ridge', 'ridge')),
                                 baseline_col='cum_loss_baseline', baseline_label='Baseline',
                                 save_path=None, title=None,
                                 ylabel='cum_loss (day-equivalents)'):
    """Publication (Nature-style) grouped bar: per city-event (x), the CITY-LEVEL
    cum_loss as side-by-side bars — Ground truth, one bar per prediction in `pred_cols`
    (each a (legend_label, column, mae_tag); columns absent from df are skipped), and (if
    `baseline_col` is present) a baseline labelled `baseline_label` — y in day-equivalents.
    Nature styling: Arial sans, no top/right spines, restrained palette, no legend frame.
    Per-bar values are printed VERTICALLY (no horizontal collisions) and the legend sits
    BELOW the axes (long labels do not overlap the bars); MAE vs GT annotated.  PNG >=300 dpi."""
    codes = list(df.index)
    _PRED_COLORS = ['#0F4D92', '#4C9F70', '#7B5EA7']       # blue, green, purple
    # (legend_label, mae_tag, values, color); GT has no MAE tag.
    series = [('Ground truth', None, df[gt_col].to_numpy(dtype=float), '#767676')]
    for i, (lab, col, tag) in enumerate(pred_cols):
        if col in df.columns:
            series.append((lab, tag, df[col].to_numpy(dtype=float),
                           _PRED_COLORS[i % len(_PRED_COLORS)]))
    if baseline_col in df.columns:
        series.append((baseline_label, baseline_label, df[baseline_col].to_numpy(dtype=float),
                       '#E28E2C'))
    gt = series[0][2]
    mae_txt = "    ".join(f"MAE({tag})={np.nanmean(np.abs(gt - vals)):.3f}"
                          for lab, tag, vals, _ in series[1:])
    nature_rc = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 8, 'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.8, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    }
    with plt.rc_context(nature_rc):
        n = len(series)
        x = np.arange(len(codes))
        w = 0.8 / n
        fig, ax = plt.subplots(figsize=(1.35 * len(codes) + 1.4, 3.9))
        for i, (lab, _tag, vals, color) in enumerate(series):
            offset = (i - (n - 1) / 2) * w
            bars = ax.bar(x + offset, vals, width=w * 0.92, label=lab, color=color,
                          edgecolor='black', linewidth=0.7)
            for bar in bars:
                h = bar.get_height()
                if np.isnan(h):
                    continue
                # Vertical labels above each bar -> no horizontal overlap between neighbours.
                ax.annotate(f'{h:.2f}', (bar.get_x() + bar.get_width() / 2, max(h, 0)),
                            xytext=(0, 2), textcoords='offset points', rotation=90,
                            ha='center', va='bottom', fontsize=5.5, color='#3a3a3a')
        ax.set_xticks(x)
        ax.set_xticklabels(codes, rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('city-event')
        ax.margins(y=0.22)                                  # headroom for vertical labels
        # Legend BELOW the axes so the long labels never overlap the bars.
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=7,
                  columnspacing=1.4, handlelength=1.3)
        ax.text(0.99, 0.99, mae_txt, transform=ax.transAxes, ha='right', va='top',
                fontsize=6.5, color='#4D4D4D')
        if title:
            ax.set_title(title, fontsize=8.5)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_bar_curve_mae(df, save_path=None, title=None, colors=None,
                      ylabel='city-curve MAE (fraction of the normal baseline)'):
    """Publication (Nature-style) grouped bar: per city-event (x), the ERROR of each
    curve-prediction method as side-by-side bars, one bar per column of `df` in column
    order, y = the mean absolute deviation between the predicted and observed city
    relative curve over the disaster window.  Unlike vis_bar_cross_city_resi_pred there
    is no ground-truth bar, because every bar IS an error and lower is better; the
    five-unit mean of each method is annotated in the corner instead of an MAE-vs-GT
    line.  Bar values are printed VERTICALLY and the legend sits BELOW the axes, so
    neither collides with a neighbour.  PNG >= 300 dpi."""
    codes = list(df.index)
    methods = list(df.columns)
    palette = colors or ['#0F4D92', '#E28E2C', '#7B5EA7', '#767676', '#4C9F70']
    mean_txt = "    ".join(f"mean({m})={np.nanmean(df[m].to_numpy(dtype=float)):.4f}"
                           for m in methods)
    nature_rc = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 8, 'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.8, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    }
    with plt.rc_context(nature_rc):
        n = max(len(methods), 1)
        x = np.arange(len(codes))
        w = 0.8 / n
        fig, ax = plt.subplots(figsize=(1.35 * len(codes) + 1.4, 3.9))
        for i, m in enumerate(methods):
            vals = df[m].to_numpy(dtype=float)
            offset = (i - (n - 1) / 2) * w
            bars = ax.bar(x + offset, vals, width=w * 0.92, label=m,
                          color=palette[i % len(palette)],
                          edgecolor='black', linewidth=0.7)
            for bar in bars:
                h = bar.get_height()
                if np.isnan(h):
                    continue
                ax.annotate(f'{h:.3f}', (bar.get_x() + bar.get_width() / 2, max(h, 0)),
                            xytext=(0, 2), textcoords='offset points', rotation=90,
                            ha='center', va='bottom', fontsize=5.5, color='#3a3a3a')
        ax.set_xticks(x)
        ax.set_xticklabels(codes, rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('city-event')
        ax.margins(y=0.22)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=7,
                  columnspacing=1.4, handlelength=1.3)
        ax.text(0.99, 0.99, mean_txt, transform=ax.transAxes, ha='right', va='top',
                fontsize=6.5, color='#4D4D4D')
        if title:
            ax.set_title(title, fontsize=8.5)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_curves_city_pred(days, gt, method_curves, save_path=None, title=None,
                         ylabel='daily mobility (flow volume per day)'):
    """Publication-style overlay of ONE city-event's mobility curve over the
    disaster window: the ground-truth curve (solid dark) plus one line per
    prediction method.  `days` is the x vector (days since landfall), `gt` the
    ground-truth values, `method_curves` an ORDERED dict label -> same-length
    array (NaNs allowed).  Each method's MAE vs the ground truth is appended to
    its LEGEND label (the legend sits below the axes), so no text is ever drawn
    over the curves."""
    _COLORS = ['#0F4D92', '#4C9F70', '#7B5EA7', '#E28E2C', '#B0413E']
    nature_rc = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 8, 'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.8, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    }
    gt = np.asarray(gt, dtype=float)
    with plt.rc_context(nature_rc):
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(days, gt, color='#3a3a3a', lw=1.8, marker='o', ms=3,
                label='Ground truth', zorder=5)
        for i, (lab, vals) in enumerate(method_curves.items()):
            vals = np.asarray(vals, dtype=float)
            mae = np.nanmean(np.abs(vals - gt))
            ax.plot(days, vals, color=_COLORS[i % len(_COLORS)], lw=1.2,
                    ls='--', marker='.', ms=2.5,
                    label=f'{lab}  (MAE {mae:.3g})')
        ax.set_xlabel('days since landfall')
        ax.set_ylabel(ylabel)
        ax.margins(y=0.08)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                  ncol=2, fontsize=7, columnspacing=1.6, handlelength=1.6)
        if title:
            ax.set_title(title, fontsize=8.5)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_component_curves_grid(curves_obs, method_curves, save_path=None,
                              title=None, ncols=4, weights=None):
    """Per-component curve grid for ONE city-event: each panel shows a
    component's OBSERVED relative curve (solid dark) and one dashed line per
    method in `method_curves` (dict label -> DataFrame [days × k] aligned with
    `curves_obs`).  The baseline r = 1 is drawn as a grey rule.  `weights`
    (optional, length k, positionally aligned with `curves_obs.columns`) is the
    city-aggregation weight; when given it is appended to each panel title."""
    _COLORS = ['#0F4D92', '#4C9F70', '#7B5EA7', '#E28E2C', '#B0413E']
    nature_rc = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 7, 'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.7, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    }
    import math
    k = curves_obs.shape[1]
    days = curves_obs.index.to_numpy()
    nrows = math.ceil(k / ncols)
    with plt.rc_context(nature_rc):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(2.4 * ncols, 1.9 * nrows),
                                 squeeze=False, sharex=True)
        for j in range(nrows * ncols):
            ax = axes[j // ncols][j % ncols]
            if j >= k:
                ax.axis('off')
                continue
            ax.axhline(1.0, color='#BBBBBB', lw=0.7, zorder=1)
            ax.plot(days, curves_obs.iloc[:, j], color='#3a3a3a', lw=1.3,
                    zorder=5, label='observed')
            for i, (lab, dfm) in enumerate(method_curves.items()):
                ax.plot(days, dfm.iloc[:, j], color=_COLORS[i % len(_COLORS)],
                        lw=1.0, ls='--', label=lab)
            ttl = f'component {curves_obs.columns[j]}'
            if weights is not None:
                ttl += f'  ($w$ = {weights[j]:.2f})'
            ax.set_title(ttl, fontsize=7)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   ncol=min(4, 1 + len(method_curves)), fontsize=7,
                   bbox_to_anchor=(0.5, -0.01))
        if title:
            fig.suptitle(title, fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_scatter_intensity_resilience(df, intensity_col, metric_cols, group_col=None,
                                     weight_col=None, ncols=3, save_path=None,
                                     title=None):
    """Each component's resilience metric (y) against its event-level
    Saffir-Simpson arrival intensity (x), pooled over ALL city-events.

    WEIGHTED (weight_col, normally weight_normal — the component's share of its
    city's normal-period activity, the weight the city-level reconstruction
    actually aggregates with).  Components are not interchangeable: one can
    carry a third of a city's activity and another a twentieth, and an
    equal-weight summary would let the small ones outvote the large.  So:

      * POINT AREA is proportional to the component's WITHIN-CITY weight share
        (w / that city's total).  The share, not the raw weight — otherwise
        every component of a large city would simply be bigger than every
        component of a small one, which says nothing about its role.
      * the per-intensity summary is the WEIGHTED median with a weighted
        inter-quartile range (an unweighted box plot beside weighted points
        would contradict the figure's own logic).
      * the reported rho is a WEIGHTED Spearman — ranks of x and y, then a
        weighted Pearson on those ranks.

    weight_col=None falls back to equal weights, in which case all three reduce
    to their ordinary forms.
    """
    import math
    metrics = list(metric_cols)
    n = len(metrics)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    single = (n == 1)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=((7.6, 5.4) if single else
                                      (4.0 * ncols, 3.6 * nrows)),
                             squeeze=False)

    x_raw = df[intensity_col].to_numpy(dtype=float)
    groups = df[group_col].to_numpy() if group_col else None
    uniq = list(dict.fromkeys(groups.tolist())) if groups is not None else [None]

    # Within-city weight share -> point area.  A missing/degenerate weight for a
    # city falls back to equal shares inside that city rather than dropping it.
    if weight_col is not None:
        w = df[weight_col].to_numpy(dtype=float)
        share = np.full(len(df), np.nan)
        for u in uniq:
            m = (groups == u) if groups is not None else np.ones(len(df), bool)
            wu = w[m]
            tot = np.nansum(wu)
            share[m] = (wu / tot if tot > 0 and np.isfinite(tot)
                        else 1.0 / max(m.sum(), 1))
        share = np.nan_to_num(share, nan=1.0 / max(len(df), 1))
    else:
        share = np.full(len(df), 1.0 / max(len(df), 1))
    sizes = 14.0 + 300.0 * (share / max(share.max(), 1e-12))

    def _wspearman(xv, yv, wv):
        rx, ry = rankdata(xv), rankdata(yv)
        mx = np.average(rx, weights=wv)
        my = np.average(ry, weights=wv)
        sx = math.sqrt(np.average((rx - mx) ** 2, weights=wv))
        sy = math.sqrt(np.average((ry - my) ** 2, weights=wv))
        if sx <= 0 or sy <= 0:
            return float('nan')
        return float(np.average((rx - mx) * (ry - my), weights=wv) / (sx * sy))

    def _wquantile(v, wv, qs):
        o = np.argsort(v)
        v, wv = v[o], wv[o]
        cw = np.cumsum(wv) - 0.5 * wv
        tot = wv.sum()
        if tot <= 0:
            return [float('nan')] * len(qs)
        return list(np.interp(qs, cw / tot, v))

    jitter = np.random.default_rng(0).uniform(-0.13, 0.13, size=len(df))
    cmap = plt.get_cmap('tab20')
    levels = sorted(np.unique(x_raw[~np.isnan(x_raw)]).tolist())
    PT_OFF, SUM_OFF = -0.17, 0.20        # points left of the integer, summary right
    for idx, m in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        y = df[m].to_numpy(dtype=float)
        for k, u in enumerate(uniq):
            msk = (groups == u) if groups is not None else np.ones(len(df), bool)
            ax.scatter(x_raw[msk] + PT_OFF + jitter[msk], y[msk], s=sizes[msk],
                       color=cmap(k % 20), alpha=0.75, edgecolor='white',
                       linewidth=0.5, label=str(u), zorder=3)
        # Weighted median + weighted IQR at each intensity level.
        for L in levels:
            sel = (x_raw == L) & ~np.isnan(y)
            if sel.sum() == 0:
                continue
            q1, q2, q3 = _wquantile(y[sel], share[sel], [0.25, 0.5, 0.75])
            ax.plot([L + SUM_OFF] * 2, [q1, q3], color='0.25', lw=6,
                    solid_capstyle='butt', alpha=0.35, zorder=2)
            ax.plot([L + SUM_OFF - 0.09, L + SUM_OFF + 0.09], [q2] * 2,
                    color='black', lw=1.8, zorder=4)
        ok = ~np.isnan(y) & ~np.isnan(x_raw)
        rho = (_wspearman(x_raw[ok], y[ok], share[ok]) if ok.sum() > 2
               else float('nan'))
        wtag = 'weighted ' if weight_col is not None else ''
        ax.set_title(f"{m}\n{wtag}Spearman rho={rho:+.2f} (n={int(ok.sum())})",
                     fontsize=10)
        ax.set_xlabel('Saffir-Simpson arrival intensity (1=ExtraTrop .. 8=Cat5)',
                      fontsize=8)
        ax.set_ylabel(m, fontsize=8)
        ax.set_xticks(levels)
        ax.set_xticklabels([str(int(L)) for L in levels])
        ax.tick_params(labelsize=7)
        if idx == 0 and groups is not None:
            leg = (dict(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5),
                        frameon=False, title='city-event', title_fontsize=9)
                   if single else
                   dict(fontsize=6, loc='best', framealpha=0.6,
                        title='city-event'))
            ax.add_artist(ax.legend(**leg))
            if weight_col is not None:
                # Size key: the point areas actually used, at three shares.
                refs = [0.05, 0.15, 0.30]
                hs = [plt.scatter([], [], s=14.0 + 300.0 * (r / max(share.max(), 1e-12)),
                                  color='0.6', edgecolor='white', linewidth=0.5,
                                  label=f'{r:.0%}') for r in refs]
                ax.legend(handles=hs, loc='upper right', fontsize=7,
                          frameon=False, labelspacing=1.1,
                          title='weight share\nof its city', title_fontsize=7.5)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def vis_scatter_reg_pred(pred_data, summary, res_cols, title=None,
                         save_path=None, ncols=3, stat_label='LOO R²',
                         unit='rank within unit', groups=None):
    """
    Regression diagnostic scatter for ONE city: one panel per resilience metric,
    each plotting the ACTUAL value (y, ground truth) against the leave-one-out
    PREDICTED value (x) — one point per component (labelled with its index).
    `unit` names the axis scale: the rank channel passes de-standardized values,
    so both axes read as the within-unit rank (1..n, the prediction continuous
    between them); 'std value' is the raw/Pearson channel's standardized scale.
    The dashed y=x line is perfect prediction; the panel title carries
    summary['stat'] under `stat_label` (Spearman ρ for the rank channel, R² for
    the raw one) and PASS/FAIL.  An 'insufficient data' metric gets a blank
    panel.

    `groups` (optional) dict metric -> per-point label array (e.g. the city-event
    each pooled component belongs to): when given the points are coloured + a
    legend is drawn, so a pooled scatter distinguishes its different city-events.
    """
    import math
    metrics = list(res_cols)
    ncols = min(ncols, max(1, len(metrics)))
    n = len(metrics)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows),
                             squeeze=False)
    for idx, m in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        pd_m = pred_data.get(m)
        s = summary.loc[m]
        if pd_m is None:
            ax.text(0.5, 0.5, f'{m}\n(insufficient data)', ha='center',
                    va='center', transform=ax.transAxes, fontsize=10, color='grey')
            ax.set_xticks([]); ax.set_yticks([])
            continue
        y_true, y_pred, comp_idx = pd_m
        # Prediction on x, ground truth on y (observed-vs-predicted calibration).
        g = groups.get(m) if isinstance(groups, dict) else None
        if g is not None:
            g = np.asarray(g)
            uniq = list(dict.fromkeys(g.tolist()))
            cmap = plt.get_cmap('tab10')
            for k, u in enumerate(uniq):
                msk = (g == u)
                ax.scatter(np.asarray(y_pred)[msk], np.asarray(y_true)[msk], s=32,
                           color=cmap(k % 10), alpha=0.85, edgecolor='white',
                           linewidth=0.5, label=str(u))
            if idx == 0:
                ax.legend(fontsize=6, loc='best', framealpha=0.6)
        else:
            ax.scatter(y_pred, y_true, s=32, color='#1976D2', alpha=0.85,
                       edgecolor='white', linewidth=0.5)
        for xp, yt, ci in zip(y_pred, y_true, comp_idx):
            ax.annotate(str(int(ci)), (xp, yt), fontsize=6, color='grey',
                        xytext=(2, 2), textcoords='offset points')
        lo = float(min(np.min(y_true), np.min(y_pred)))
        hi = float(max(np.max(y_true), np.max(y_pred)))
        ax.plot([lo, hi], [lo, hi], '--', color='grey', lw=1)      # y = x
        tag = ('PASS' if s['passed'] else 'FAIL') if s['status'] == 'ok' else 'n/a'
        ax.set_title(f"{m}\n{stat_label}={s['stat']:+.2f} [{tag}]", fontsize=10)
        ax.set_xlabel(f'predicted ({unit})', fontsize=8)
        ax.set_ylabel(f'actual ({unit})', fontsize=8)
        ax.tick_params(labelsize=7)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')
    if title:
        # The cross-city title names every training unit, which at 13 units is
        # far wider than a one-panel figure; wrap it to the figure's own width
        # (and save with a tight box) so it is never clipped.
        import textwrap
        per_line = max(40, int(11 * 4.0 * ncols))
        fig.suptitle('\n'.join(textwrap.wrap(title, per_line)), fontsize=11)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def vis_heatmap_pair_r2(mat, title=None, save_path=None, vmax=0.6,
                        xlabel='test', ylabel='train', blocks=None):
    """Square pairwise cross-city R² heatmap: rows = train unit, cols = test unit,
    each cell = the transfer R² of that (train, test) pair.  Diverging colour
    centred at 0 (green = beats the mean, red = worse), clipped to ±vmax; each
    cell annotated (n/a if undefined).

    The DIAGONAL is excluded — a cell where train and test are the same unit is
    that unit's own leave-one-component-out, not a transfer between units, and
    it is both structurally larger than its row and irrelevant to the
    source-selection question the matrix is read for.  Diagonal cells are drawn
    blank (grey) and left unannotated; detection is by label, so it survives any
    row/column reordering.

    `blocks` (optional) [(start, size, cluster_id)] draws a square outline
    around each contiguous community — pass the caller's Louvain partition with
    the matrix already reordered to match, otherwise the boxes are meaningless.
    """
    from matplotlib.patches import Rectangle
    V = mat.to_numpy(dtype=float).copy()
    n_r, n_c = V.shape
    rows, cols = list(mat.index), list(mat.columns)
    diag = np.zeros_like(V, dtype=bool)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if r == c:
                diag[i, j] = True
    V[diag] = np.nan

    fig, ax = plt.subplots(figsize=(1.15 * n_c + 2.6, 1.0 * n_r + 2.0))
    cmap = plt.get_cmap('RdYlGn').copy()
    cmap.set_bad('#d9d9d9')                      # excluded diagonal / undefined
    im = ax.imshow(np.ma.masked_invalid(np.clip(V, -vmax, vmax)), cmap=cmap,
                   vmin=-vmax, vmax=vmax, aspect='auto')
    for i in range(n_r):
        for j in range(n_c):
            if diag[i, j]:
                continue                          # excluded: no number at all
            v = V[i, j]
            ax.text(j, i, 'n/a' if np.isnan(v) else f'{v:+.2f}',
                    ha='center', va='center', fontsize=10, color='black')
    for start, size, cid in (blocks or ()):
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), size, size,
                               fill=False, edgecolor='#111111', linewidth=2.5,
                               zorder=5))
        ax.text(start - 0.42, start - 0.42, f'C{cid}', fontsize=9,
                fontweight='bold', color='#111111', ha='left', va='top',
                zorder=6)
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(cols, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title or 'Pairwise cross-city R²', fontsize=12)
    fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8).set_label('R² (clipped)', fontsize=9)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig

def vis_line_resilience_curves(curves, save_path=None, order=None):
    """
    Per-component relative-activity curves over the disaster period — one
    stacked subplot per component (one row each), the same top-to-bottom
    layout as vis_line_nmf_component_timeline.

    Parameters
    ----------
    curves : DataFrame [n_disaster_days × k]  (resilience_curves output);
             index = days since landfall, values = activity / pre-disaster
             day-type-matched baseline (1.0 = normal).
    order  : list[int] | None  component indices in top-to-bottom order
             (None keeps 0..k-1).  Each subplot is labelled with its TRUE
             component index, so an order sorted by a functional share stays
             readable.

    Each panel: r(d) line, grey baseline at 1.0, black dot at the lowest point.
    A shared y-limit makes drop depths comparable across components.
    """
    k = curves.shape[1]
    order = list(range(k)) if order is None else list(order)
    x = curves.index.to_numpy()
    ymax = max(1.05, np.nanmax(curves.to_numpy()) * 1.05)

    fig, axes = plt.subplots(k, 1, figsize=(9, 1.7 * k), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    for row, comp in enumerate(order):
        ax = axes[row]
        y = curves.iloc[:, comp].to_numpy()
        ax.axhline(1.0, color='grey', linestyle=':', linewidth=1.2)
        ax.plot(x, y, color='#D32F2F', linewidth=1.8, zorder=2)
        if not np.all(np.isnan(y)):
            t = int(np.nanargmin(y))
            ax.plot(x[t], y[t], 'o', color='black', markersize=5, zorder=3)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(f'Comp {comp}', fontsize=10, fontweight='bold', labelpad=4)
        ax.grid(axis='y', linestyle=':', alpha=0.35)
        ax.tick_params(axis='y', labelsize=8)

    axes[-1].set_xticks(x)
    axes[-1].set_xlabel('days since landfall', fontsize=11, labelpad=6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


_OD_SLIDER_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>%TITLE%</title>
<script src="https://unpkg.com/deck.gl@9.0.36/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet"/>
<style>
  body { margin:0; font-family: Arial, Helvetica, sans-serif; }
  #map { position:absolute; inset:0; }
  #panel { position:absolute; top:10px; left:10px; z-index:10; background:rgba(255,255,255,.94);
           padding:10px 14px; border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,.3);
           font-size:13px; max-width:330px; }
  #panel h3 { margin:0 0 6px; font-size:14px; }
  #day-row { margin-top:8px; display:flex; align-items:center; gap:8px; }
  #day-label { font-weight:bold; min-width:52px; }
  input[type=range] { width:170px; }
  .views label { margin-right:10px; }
  #legend { margin-top:6px; color:#444; font-size:11.5px; line-height:1.45; }
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h3>%TITLE%</h3>
  <div class="views">
    <label><input type="radio" name="view" value="Prediction" checked> Prediction</label>
    <label><input type="radio" name="view" value="Ground truth"> Ground truth</label>
    <label><input type="radio" name="view" value="Difference"> Difference</label>
  </div>
  <div id="day-row">
    <button id="prev">&#9664;</button>
    <input type="range" id="day" min="0" max="%MAXDAY%" step="1" value="0"/>
    <button id="next">&#9654;</button>
    <span id="day-label"></span>
  </div>
  <div id="legend"></div>
</div>
<script>
const DATA = %DATA%;
const DAYS = %DAYS%;
const VMAX = %VMAX%;
const state = { view: 'Prediction', day: 0 };

const deckgl = new deck.DeckGL({
  container: 'map', map: maplibregl,
  mapStyle: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
  initialViewState: { longitude: %LON%, latitude: %LAT%, zoom: %ZOOM%, pitch: 35, bearing: 0 },
  controller: true, layers: [],
});

function colors(view, v) {
  if (view === 'Difference')
    return v >= 0 ? [[178, 24, 43, 175], [214, 96, 77, 60]]
                  : [[33, 102, 172, 175], [67, 147, 195, 60]];
  if (view === 'Ground truth') return [[58, 58, 58, 165], [120, 120, 120, 55]];
  return [[15, 77, 146, 165], [90, 140, 200, 55]];
}

function render() {
  const arcs = DATA[state.view][state.day] || [];
  const vmax = Math.max(VMAX[state.view], 1e-9);
  const layer = new deck.ArcLayer({
    id: 'od', data: arcs, greatCircle: false,
    getSourcePosition: d => [d[0], d[1]],
    getTargetPosition: d => [d[2], d[3]],
    getWidth: d => 0.6 + 7.0 * Math.sqrt(Math.abs(d[4]) / vmax),
    widthUnits: 'pixels',
    getSourceColor: d => colors(state.view, d[4])[0],
    getTargetColor: d => colors(state.view, d[4])[1],
    pickable: true,
  });
  deckgl.setProps({ layers: [layer],
    getTooltip: ({object}) => object && ('flow: ' + object[4].toFixed(1)) });
  document.getElementById('day-label').textContent = 'day ' + DAYS[state.day];
  const n = arcs.length;
  document.getElementById('legend').innerHTML =
    '<b>' + state.view + '</b>, day ' + DAYS[state.day] + ' since landfall &middot; top '
    + n + ' OD pairs by |flow|<br/>' +
    (state.view === 'Difference'
      ? 'red: predicted &gt; observed &middot; blue: predicted &lt; observed<br/>'
      : 'arc width &prop; &radic;flow (daily trips)<br/>') +
    '%NOTE%';
}

document.querySelectorAll('input[name=view]').forEach(el =>
  el.addEventListener('change', e => { state.view = e.target.value; render(); }));
const slider = document.getElementById('day');
slider.addEventListener('input', e => { state.day = +e.target.value; render(); });
document.getElementById('prev').addEventListener('click', () => {
  state.day = Math.max(0, state.day - 1); slider.value = state.day; render(); });
document.getElementById('next').addEventListener('click', () => {
  state.day = Math.min(%MAXDAY%, state.day + 1); slider.value = state.day; render(); });
render();
</script>
</body>
</html>
"""


def vis_od_flow_slider_html(frames, day_labels, save_path, title, note=''):
    """Self-contained slider map for the STEP-7 spatial OD forecast: one HTML
    per city-event with a view toggle (Prediction / Ground truth / Difference)
    and a day slider.  `frames` maps each of those three view names to a list
    (one entry per disaster day) of arc rows [o_lon, o_lat, d_lon, d_lat,
    value]; `value` is the daily OD flow (signed for the Difference view).
    Arc width is normalized per view by its own max |value| across all days so
    the slider animates on a fixed scale.  Basemap and deck.gl load from CDN
    (same dependency model as the archived pydeck maps)."""
    import json
    lons, lats = [], []
    vmax = {}
    for name, days in frames.items():
        m = 0.0
        for arcs in days:
            for a in arcs:
                lons += [a[0], a[2]]
                lats += [a[1], a[3]]
                m = max(m, abs(a[4]))
        vmax[name] = m
    if not lons:
        return
    lon0, lat0 = float(np.mean(lons)), float(np.mean(lats))
    span = max(max(lons) - min(lons), max(lats) - min(lats), 1e-3)
    zoom = float(np.clip(np.log2(360.0 / (span * 2.2)), 8.0, 11.5))
    n_days = max(len(d) for d in frames.values())
    html = (_OD_SLIDER_TEMPLATE
            .replace('%TITLE%', title)
            .replace('%DATA%', json.dumps(frames, separators=(',', ':')))
            .replace('%DAYS%', json.dumps(list(day_labels)))
            .replace('%VMAX%', json.dumps(vmax))
            .replace('%MAXDAY%', str(n_days - 1))
            .replace('%LON%', f'{lon0:.4f}')
            .replace('%LAT%', f'{lat0:.4f}')
            .replace('%ZOOM%', f'{zoom:.2f}')
            .replace('%NOTE%', note))
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html)


# ── Transferability: does domain proximity predict RANK transfer? ─────────────
def vis_transferability_map(emb, centroids, W2, T, Tsym, labels, colors,
                            var_ratio, rho, pval, save_path=None):
    """Three panels answering ONE question: do city-events that sit close in
    component-feature space also transfer their cum_loss ORDERING to each other?

      A  map    — PCA of every unit's components, one convex hull per unit, the
                  hulls' centroids joined by edges COLOURED BY the measured
                  pairwise rank transfer.  Proximity is the claim; edge colour
                  is the evidence, so the claim is checkable inside the panel.
      B  proof  — the same two quantities as a scatter (one point per unordered
                  pair) with the Mantel statistic, so the visual impression in A
                  cannot be an artefact of the 2-D projection.
      C  matrix — the raw directed transfer matrix; transfer is ASYMMETRIC and
                  panel A can only draw the symmetrized value.

    `emb` maps code -> [k × 2] PCA scores, `centroids` code -> [2], `W2` /`T` /
    `Tsym` are [n × n] aligned with `labels` (codes in plotting order); `rho`
    and `pval` are the Mantel statistic between W2 and Tsym."""
    from scipy.spatial import ConvexHull
    from matplotlib.colors import TwoSlopeNorm
    codes = list(labels)
    n = len(codes)
    short = {c: labels[c] for c in codes}
    nature_rc = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 7, 'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.7, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    }
    iu = np.triu_indices(n, 1)
    norm = TwoSlopeNorm(vmin=min(-0.7, float(np.nanmin(T))), vcenter=0.0,
                        vmax=max(0.9, float(np.nanmax(T))))
    cmap = plt.cm.RdBu
    with plt.rc_context(nature_rc):
        fig = plt.figure(figsize=(11.0, 3.6))
        # Generous wspace: the panel-A colourbar is drawn inside the first
        # cell and would otherwise sit on top of panel B's y-label.
        gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.52)
        # A — map
        ax = fig.add_subplot(gs[0, 0])
        for c in codes:
            pts = np.asarray(emb[c], dtype=float)
            ax.scatter(pts[:, 0], pts[:, 1], s=13, color=colors[c], alpha=.45,
                       zorder=2, edgecolor='none')
            if len(pts) >= 3:
                h = ConvexHull(pts)
                poly = np.vstack([pts[h.vertices], pts[h.vertices][:1]])
                ax.fill(poly[:, 0], poly[:, 1], color=colors[c], alpha=.10,
                        zorder=1)
                ax.plot(poly[:, 0], poly[:, 1], color=colors[c], lw=.8,
                        alpha=.55, zorder=1)
        for i, a in enumerate(codes):
            for j, b in enumerate(codes):
                if i < j:
                    v = float(Tsym[i, j])
                    ax.plot([centroids[a][0], centroids[b][0]],
                            [centroids[a][1], centroids[b][1]],
                            color=cmap(norm(v)), lw=0.6 + 3.4 * abs(v),
                            zorder=3, solid_capstyle='round', alpha=.92)
        for c in codes:
            ax.scatter(*centroids[c], s=110, color=colors[c], zorder=5,
                       edgecolor='white', linewidth=1.3)
            ax.annotate(short[c], centroids[c], fontsize=6.5, fontweight='bold',
                        xytext=(0, -13), textcoords='offset points',
                        ha='center', color=colors[c], zorder=6)
        ax.set_xlabel(f'PC1 ({var_ratio[0]:.0%} var)')
        ax.set_ylabel(f'PC2 ({var_ratio[1]:.0%} var)')
        ax.set_title('A  hulls = each city-event\'s components;\n'
                     'edge colour = measured pairwise RANK transfer', fontsize=8)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=.042, pad=.02)
        cb.set_label('rank transfer (spearman)', fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        # B — proof
        ax = fig.add_subplot(gs[0, 1])
        for i, a in enumerate(codes):
            for j, b in enumerate(codes):
                if i < j:
                    ax.scatter(W2[i, j], Tsym[i, j], s=34,
                               color=cmap(norm(float(Tsym[i, j]))), zorder=3,
                               edgecolor='#555555', linewidth=.5)
                    ax.annotate(f'{short[a][:3]}–{short[b][:3]}',
                                (W2[i, j], Tsym[i, j]), fontsize=5,
                                xytext=(0, 5), textcoords='offset points',
                                ha='center', color='#555555')
        z = np.polyfit(W2[iu], Tsym[iu], 1)
        xs = np.linspace(W2[iu].min(), W2[iu].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color='#333333', lw=1.1, zorder=2)
        ax.axhline(0, color='#CCCCCC', lw=.7, zorder=1)
        ax.set_xlabel(r'domain distance  $W_2$ (Sinkhorn)')
        ax.set_ylabel('rank transfer (symmetrized)')
        ax.set_title('B  closer $\\Rightarrow$ better rank transfer\n'
                     f'$\\rho_s$ = {rho:+.2f}  (Mantel $P$ = {pval:.3f}, '
                     f'{len(iu[0])} pairs)', fontsize=8)
        ax.margins(.18)
        # C — asymmetry
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(np.where(np.eye(n, dtype=bool), np.nan, T), cmap=cmap,
                  norm=norm)
        for i in range(n):
            for j in range(n):
                if i != j and np.isfinite(T[i, j]):
                    ax.text(j, i, f'{T[i, j]:.2f}', ha='center', va='center',
                            fontsize=5.5,
                            color='white' if abs(T[i, j]) > .55 else '#333333')
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([short[c] for c in codes], fontsize=5.5, rotation=40,
                           ha='right')
        ax.set_yticklabels([short[c] for c in codes], fontsize=5.5)
        ax.set_xlabel('test (target)', fontsize=6.5)
        ax.set_ylabel('train (source)', fontsize=6.5)
        ax.set_title('C  rank transfer is ASYMMETRIC\n'
                     '(same city, different event ≠ close)', fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)),
                        exist_ok=True)
            fig.savefig(save_path, dpi=450, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_w2_decomposition(pairs, contrib, transfer, group_names, group_colors,
                         save_path=None):
    """What is the domain distance MADE OF, and which part of it explains the
    measured transfer?  A squared-Euclidean cost is additive over dimensions,
    so under one coupling the transported cost splits EXACTLY over feature
    groups; both rows are that split, seen two ways.

      row 1  one stacked bar per unordered pair, pairs ordered near -> far;
             segment heights are the groups' exact contributions to that pair's
             W2 and the number above the bar is the measured rank-prediction
             performance between the two units.
      row 2  one scatter per group: x is that group's contribution — the
             ABSOLUTE segment height of row 1, not a share — and y the same
             performance.  A group explains the distance/transfer relation only
             if ITS panel slopes; a large but flat group is inert background,
             which is why shares alone would be misleading here.

    `pairs` are the pair labels in plotting order, `contrib` a DataFrame with
    one column per group aligned to `pairs`, `transfer` the matching
    symmetrized (A->B and B->A averaged) transfer values."""
    from scipy.stats import spearmanr as _spearmanr
    nature_rc = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 7, 'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.7, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    }
    gnames = list(group_names)
    transfer = np.asarray(transfer, dtype=float)
    total = contrib[gnames].sum(axis=1).to_numpy(dtype=float)
    with plt.rc_context(nature_rc):
        fig = plt.figure(figsize=(10.6, 5.6))
        gs = fig.add_gridspec(2, len(gnames), height_ratios=[1.25, 1],
                              hspace=0.5, wspace=0.30)
        ax = fig.add_subplot(gs[0, :])
        x = np.arange(len(pairs))
        bottom = np.zeros(len(pairs))
        for g in gnames:
            v = contrib[g].to_numpy(dtype=float)
            ax.bar(x, v, 0.62, bottom=bottom, color=group_colors[g], label=g)
            bottom += v
        for k in range(len(pairs)):
            ax.text(x[k], total[k] + 0.5, f'{transfer[k]:+.2f}', ha='center',
                    fontsize=6.2, fontweight='bold',
                    color=plt.cm.RdBu(0.5 + 0.5 * np.clip(transfer[k] / 0.9,
                                                          -1, 1)))
        ax.text(0.01, 0.86, 'number above bar = rank-prediction performance',
                transform=ax.transAxes, fontsize=6, color='#555555')
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, fontsize=6.5)
        ax.set_ylabel('$W_2$ contribution (additive, exact)')
        ax.set_title('what pushes each pair apart — pairs ordered near '
                     '$\\rightarrow$ far', fontsize=8)
        ax.legend(ncol=len(gnames), fontsize=6.5, loc='upper left')
        ax.margins(x=0.02)
        axs2 = [fig.add_subplot(gs[1, k]) for k in range(len(gnames))]
        for ax, g in zip(axs2, gnames):
            xs = contrib[g].to_numpy(dtype=float)
            ax.scatter(xs, transfer, s=30, color=group_colors[g], zorder=3,
                       edgecolor='white', linewidth=0.5)
            z = np.polyfit(xs, transfer, 1)
            xr = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xr, np.polyval(z, xr), color='#333333', lw=1.0, zorder=2)
            ax.axhline(0, color='#CCCCCC', lw=0.7, zorder=1)
            r = float(_spearmanr(xs, transfer).statistic)
            ax.set_title(f'{g}   $\\rho_s$ = {r:+.2f}', fontsize=8,
                         color=group_colors[g])
            ax.set_xlabel(f'$W_2$ carried by {g}\n'
                          '(= its segment height above, absolute)')
            ax.margins(0.15)
        axs2[0].set_ylabel('cum_loss rank-prediction performance\n'
                           'between the pair (spearman,\n'
                           'A$\\to$B and B$\\to$A averaged)')
        for ax in axs2[1:]:
            ax.set_yticklabels([])
        lo = min(ax.get_ylim()[0] for ax in axs2)
        hi = max(ax.get_ylim()[1] for ax in axs2)
        for ax in axs2:
            ax.set_ylim(lo, hi)
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)),
                        exist_ok=True)
            fig.savefig(save_path, dpi=450, bbox_inches='tight')
            plt.close(fig)
    return fig


# == NMF decomposition-quality figures =========================================


def vis_nmf_quality(q, title=None, save_path=None):
    """
    One city-event's decomposition-quality report, three panels from the
    nmf_quality_metrics dict `q`.  Everything is FIT-window only (the rows the
    basis was fit on); the disaster period is out of scope by design — the
    figure describes the decomposition itself, not any predictive use of it.

      A  per-slot distribution (KS) error across the fit window — localises
         WHERE the reconstruction distorts the OD-flow shape (each slot's X
         and WH are mean-normalised first, so a uniform over/under-estimate
         does not register);
      B  pooled CDF overlay of those per-slot-normalised values, ground truth
         vs reconstruction (log x; positive entries) — the shape the KS
         compares, seen directly;
      C  component weights w_i = $\|W_i\|\,\|H_i\|$ as a fraction of the largest,
         descending, with the min_comp_frac threshold line; components below
         it are red (too-large-k symptom, the health rule requires none).

    The fit-window relative Frobenius error is annotated in panel A.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    axA, axB, axC = axes

    # A - per-slot KS timeline over the fit window
    ks = np.asarray(q['dist_err_per_slot'], dtype=float)
    axA.plot(np.arange(ks.size), ks, color='#1976D2', lw=1.0)
    axA.axhline(q['dist_err_mean'], color='#D32F2F', linestyle='--', lw=1.2,
                label=f"mean = {q['dist_err_mean']:.3f}")
    axA.set_xlabel('fit-window time slot')
    axA.set_ylabel('per-slot KS distance')
    axA.set_title('A  distribution error over time', loc='left', fontsize=12)
    axA.text(0.02, 0.02,
             f"rel. reconstruction error = {q['rel_err']:.3f}",
             transform=axA.transAxes, fontsize=9, va='bottom')
    axA.legend(frameon=False, fontsize=9)

    # B - pooled CDF overlay
    for vals, color, lbl in ((q['gt_vals'], 'black', 'ground truth'),
                             (q['rec_vals'], '#1976D2', 'reconstruction')):
        v = np.sort(np.asarray(vals, dtype=float))
        if v.size == 0:
            continue
        axB.plot(v, np.arange(1, v.size + 1) / v.size, color=color, lw=1.4,
                 label=lbl, alpha=0.9)
    axB.set_xscale('log')
    axB.set_xlabel('OD flow / slot mean (positive entries, log scale)')
    axB.set_ylabel('cumulative fraction')
    axB.set_title('B  value-distribution shapes (pooled over slots)',
                  loc='left', fontsize=12)
    axB.legend(frameon=False, fontsize=9)

    # C - component weight fractions
    frac = np.asarray(q['comp_frac'], dtype=float)
    thr = q['min_comp_frac']
    colors = ['#D32F2F' if f < thr else '#1976D2' for f in frac]
    axC.bar(np.arange(frac.size), frac, color=colors, edgecolor='white')
    axC.axhline(thr, color='#D32F2F', linestyle='--', lw=1.2,
                label=f'threshold = {thr:.0%} of max')
    axC.set_yscale('log')
    axC.set_xlabel('component (sorted by weight)')
    axC.set_ylabel('$\|W_i\|\,\|H_i\|$ / max (log)')
    axC.set_title(f"C  component weights - {q['n_below']} below threshold",
                  loc='left', fontsize=12)
    axC.legend(frameon=False, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def vis_nmf_quality_summary(df, min_comp_frac, save_path=None):
    """
    Cross-city error dashboard: one row per error family, city-events on x.
    All metrics are FIT-window only (see vis_nmf_quality).

      1  mean per-slot distribution (KS) error
      2  relative reconstruction error

    Component-count health is deliberately NOT shown here: it is a per-unit
    pass/fail check, already drawn as panel C of each unit's own figure, and a
    cross-city bar of it carries no comparison worth making.  `min_comp_frac`
    is kept in the signature so callers need not know that.

    `df` is indexed by city-event code with columns dist_err_mean, rel_err
    and k (k annotates the bars).
    """
    codes = list(df.index)
    x = np.arange(len(codes))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, 0.85 * len(codes)), 7),
                             sharex=True)

    axes[0].bar(x, df['dist_err_mean'], color='#1976D2', edgecolor='white')
    axes[0].set_ylabel('mean per-slot\nKS distance')
    axes[0].set_title('Distribution error', loc='left', fontsize=12)

    axes[1].bar(x, df['rel_err'], color='#455A64', edgecolor='white')
    for xi, (v, k) in enumerate(zip(df['rel_err'], df['k'])):
        axes[1].text(xi, v, f'k={int(k)}', ha='center', va='bottom', fontsize=8)
    axes[1].set_ylabel('relative\nreconstruction error')
    axes[1].set_title('Absolute error', loc='left', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(codes, rotation=45, ha='right')

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def vis_nmf_rank_cv(curves, table, band='k_2se', save_path=None, ncols=5):
    """
    Held-out-entry rank curves, one panel per city-event, each panel captioned
    with its own numbers instead of the figure carrying a separate table.

    `curves` maps city-event code -> (ks, mean_err, sd_err); `table` is a
    DataFrame indexed by the SAME codes in the SAME order supplying n_od,
    k_min, <band>_lo and <band>_hi, and optionally `excluded`.

    Each caption gives the matrix size (OD pairs), the k at the curve's
    minimum, and the `band` range — the k whose error stays within that
    tolerance of the minimum.  Other tolerances are computed too and live in
    the CSV; across all of them the selected rank moves by only a component or
    two, so the caption carries one band rather than crowding in three.

    A unit flagged `excluded` (its band tops out too low to support a usable
    decomposition) keeps its panel — this figure is the record of WHY it was
    dropped — but is greyed and labelled, so the figure shows the full evidence
    and the verdict at once.

    The registry's own k is deliberately NOT drawn: this figure reports what
    each matrix supports, and the comparison against the configured value is a
    separate argument that the numbers here feed rather than settle.

    Panels are expected in OD-pair order (the caller sorts them): the size
    effect — sharp interior minima for the large matrices, none at all for the
    smallest — is what the arrangement has to make visible.
    """
    lo_col, hi_col = band + '_lo', band + '_hi'
    band_label = {'k_1se': 'within 1 SE', 'k_2se': 'within 2 SE',
                  'k_2sd': 'within 2 SD'}.get(band, band)
    codes = list(curves)
    nrow = int(np.ceil(len(codes) / ncols))
    fig, axes = plt.subplots(nrow, ncols, figsize=(3.2 * ncols, 2.95 * nrow),
                             squeeze=False)
    for i, c in enumerate(codes):
        ax = axes[i // ncols][i % ncols]
        ks, mu, sd = (np.asarray(v, dtype=float) for v in curves[c])
        ok = np.isfinite(mu)
        if not ok.any():
            ax.axis('off')
            continue
        dropped = bool(table.loc[c, 'excluded']) if 'excluded' in table else False
        line = '#9E9E9E' if dropped else '#1976D2'
        ax.fill_between(ks[ok], (mu - sd)[ok], (mu + sd)[ok], color=line,
                        alpha=0.18, lw=0)
        ax.plot(ks[ok], mu[ok], color=line, lw=1.3, marker='o', ms=3)
        kmin = int(table.loc[c, 'k_min'])
        lo, hi = int(table.loc[c, lo_col]), int(table.loc[c, hi_col])
        ax.axvline(kmin, color='#D32F2F' if not dropped else '#EF9A9A', lw=1.2)
        rng = f'{lo}' if lo == hi else f'{lo}\u2013{hi}'
        ax.set_title(f"{c}   ({int(table.loc[c, 'n_od'])} OD pairs)"
                     f"{'   [dropped]' if dropped else ''}\n"
                     f"k min = {kmin}    {band_label}: {rng}",
                     fontsize=8.6, linespacing=1.6,
                     color='#9E9E9E' if dropped else 'black')
        ax.tick_params(labelsize=7, colors='0.6' if dropped else 'black')
    for j in range(len(codes), nrow * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.supxlabel('number of components k', fontsize=11)
    fig.supylabel('held-out prediction error (masked entries)', fontsize=11)
    fig.suptitle('Rank selection by held-out-entry cross-validation '
                 '(panels ordered by OD-pair count; red line = k min)',
                 fontsize=13)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


# == STEP-7 rank-channel / parameter figures (publication style) ==============
# Ported into the pipeline on 2026-08-04.  They were previously produced by
# throwaway scripts, so they silently went stale whenever the registry changed
# (they still showed 17 city-events at k~10 long after the rank CV cut the set
# to 13).  Everything they need is in the STEP-7 component parameter table.

_PUB_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 7, 'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': 0.7, 'legend.frameon': False,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
}
_PUB_BLUE, _PUB_DARK, _PUB_GREY = '#0F4D92', '#3a3a3a', '#BBBBBB'


def _pub_grid(n, ncols):
    nrow = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrow, ncols, figsize=(2.05 * ncols, 2.15 * nrow),
                             squeeze=False)
    return fig, axes, nrow


def _pub_save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)


def vis_rank_pred_vs_true(params, save_path=None, ncols=5):
    """Predicted vs observed cum_loss RANK, one panel per city-event.

    The rank channel is the only part of the curve forecast that transfers, so
    this is the figure that shows whether it does: both axes are ranks WITHIN
    the unit, the diagonal is a perfect ordering, and each panel carries its own
    Spearman.  `params` is the STEP-7 component table (needs code, rank_score,
    cum_loss_fit).
    """
    codes = list(params['code'].drop_duplicates())
    with plt.rc_context(_PUB_RC):
        fig, axes, nrow = _pub_grid(len(codes), ncols)
        for i, c in enumerate(codes):
            ax = axes[i // ncols][i % ncols]
            s = params[params.code == c].dropna(subset=['rank_score',
                                                        'cum_loss_fit'])
            if len(s) < 2:
                ax.axis('off')
                continue
            rt, rp = rankdata(s['cum_loss_fit']), rankdata(s['rank_score'])
            n = len(s)
            ax.plot([0.5, n + 0.5], [0.5, n + 0.5], color=_PUB_GREY, lw=0.7,
                    zorder=1)
            ax.scatter(rt, rp, s=13, color=_PUB_BLUE, alpha=0.85, zorder=3,
                       edgecolor='white', linewidth=0.35)
            ax.set_title(c, fontsize=6.5)
            rho = spearmanr(rt, rp).statistic
            ax.text(0.05, 0.93, '$\\rho_s$ = {:+.2f}'.format(rho),
                    transform=ax.transAxes, fontsize=5.8, va='top')
            ticks = [t for t in (1, 5, 10, 15) if t <= n]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xlim(0.4, n + 0.6)
            ax.set_ylim(0.4, n + 0.6)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=5.5)
        for k in range(len(codes), nrow * ncols):
            axes[k // ncols][k % ncols].axis('off')
        fig.supxlabel('observed cumulative-loss rank', fontsize=7)
        fig.supylabel('predicted rank', fontsize=7)
        fig.tight_layout()
        _pub_save(fig, save_path)
    return fig


def vis_rank_to_cumloss_qm(params, save_path=None, ncols=5):
    """From rank to magnitude: the quantile-mapped prediction along the
    predicted rank order, against the observed losses at the same positions.

    Shows what the quantile mapping adds on top of the ordering — the borrowed
    pooled shape, rescaled to the held city's spread and shifted to its
    predicted total — and where it under- or over-shoots.  `params` needs code,
    rank_score, cum_loss_fit and cum_loss_pred.
    """
    codes = list(params['code'].drop_duplicates())
    with plt.rc_context(_PUB_RC):
        fig, axes, nrow = _pub_grid(len(codes), ncols)
        for i, c in enumerate(codes):
            ax = axes[i // ncols][i % ncols]
            s = params[params.code == c].dropna(
                subset=['rank_score', 'cum_loss_fit', 'cum_loss_pred'])
            if len(s) < 2:
                ax.axis('off')
                continue
            pos = rankdata(s['rank_score'])
            o = np.argsort(pos)
            ax.axhline(0.0, color=_PUB_GREY, lw=0.7, zorder=1)
            ax.plot(pos[o], s['cum_loss_pred'].to_numpy()[o], color=_PUB_BLUE,
                    lw=0.8, marker='o', ms=2.6, mfc='white', mew=0.7, zorder=3,
                    label='quantile-mapped prediction')
            ax.scatter(pos, s['cum_loss_fit'], s=12, color=_PUB_DARK, zorder=4,
                       label='observed')
            ax.set_title(c, fontsize=6.5)
            r = pearsonr(s['cum_loss_pred'], s['cum_loss_fit']).statistic
            ax.text(0.05, 0.94, 'r = {:+.2f}'.format(r),
                    transform=ax.transAxes, fontsize=5.8, va='top')
            ax.set_xticks([t for t in (1, 5, 10, 15) if t <= len(s)])
            ax.tick_params(labelsize=5.5)
        for k in range(len(codes), nrow * ncols):
            axes[k // ncols][k % ncols].axis('off')
        fig.supxlabel('predicted rank position (1 = smallest predicted loss)',
                      fontsize=7)
        fig.supylabel('component cumulative loss', fontsize=7)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=2, fontsize=6.5,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout()
        _pub_save(fig, save_path)
    return fig


def vis_alpha_level_relationship(params, save_path=None):
    """The recovery rate / plateau trade-off over every component of every
    city-event: log10 of the fitted rate against the fitted level L.

    This relation is what makes the plateau inversion possible — rate and level
    are not free of each other, so pinning one constrains the other.  Grey lines
    are leave-one-city-out refits, showing the relation is not carried by any
    single unit.  `params` needs the UNGATED full fits (alpha_fit_ungated,
    level_fit_ungated) so the quality gate does not thin the cloud.
    """
    s = params.dropna(subset=['alpha_fit_ungated', 'level_fit_ungated'])
    codes = list(s['code'].drop_duplicates())
    la = np.log10(np.clip(s['alpha_fit_ungated'].to_numpy(float), 1e-3, None))
    lv = s['level_fit_ungated'].to_numpy(float)
    coef = np.polyfit(la, lv, 1)
    r2 = 1.0 - (np.sum((lv - np.polyval(coef, la)) ** 2)
                / np.sum((lv - lv.mean()) ** 2))
    cmap = plt.get_cmap('tab20')
    colors = {c: cmap(i % 20) for i, c in enumerate(codes)}
    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        xs = np.linspace(la.min(), la.max(), 200)
        for held in codes:
            m = (s['code'] != held).to_numpy()
            ax.plot(xs, np.polyval(np.polyfit(la[m], lv[m], 1), xs),
                    color=_PUB_GREY, lw=0.6, alpha=0.8, zorder=2)
        ax.plot(xs, np.polyval(coef, xs), color=_PUB_DARK, lw=1.4, zorder=3)
        for c in codes:
            m = (s['code'] == c).to_numpy()
            ax.scatter(la[m], lv[m], s=16, color=colors[c], alpha=0.85,
                       zorder=4, edgecolor='white', linewidth=0.3, label=c)
        ax.set_xlabel(r"$\log_{10}\,\alpha$")
        ax.set_ylabel(r"recovery level $L$")
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01),
                  ncol=min(5, max(3, (len(codes) + 3) // 4)),
                  fontsize=5.2, handletextpad=0.2, columnspacing=0.8,
                  labelspacing=0.3, borderaxespad=0.0)
        stats = ('$L$ = {:.2f} $-$ {:.2f} $\\log_{{10}}\\alpha$\n'
                 '$R^2$ = {:.2f}\n'
                 'grey: leave-one-city-out fits').format(
                     coef[1], abs(coef[0]), r2)
        ax.text(0.98, 0.70, stats,
                transform=ax.transAxes, fontsize=6, va='top', ha='right')
        fig.tight_layout()
        _pub_save(fig, save_path)
    return fig


def vis_func_vs_time_distribution(ranked, time_col, categories, rho_rank=None,
                                  save_path=None):
    """How one temporal feature and the functional shares CO-DISTRIBUTE across
    every city-event — the scatter behind one row of the pooled correlation
    heatmap.

    That heatmap compresses each (temporal feature, function) pair into a single
    pooled rank correlation.  This figure shows what was compressed: one point
    per component, coloured by city-event, so the shape of the relationship and
    whether it holds in every unit or is carried by one of them are both
    visible.

    BOTH axes are the WITHIN-UNIT NORMALISED RANK — each column ranked inside
    its own city-event, then divided by that unit's component count, so every
    unit contributes the same uniform marginal and the cities with the largest
    raw magnitudes cannot dominate.  `ranked` must already be in that form (the
    caller ranks once and reuses it for the correlation), which is what makes
    the annotated Spearman EXACTLY the coefficient the heatmap reports: the
    number and the point cloud are the same object, not two views that have to
    be trusted to agree.

    Ranks also remove the reason the earlier version needed a log axis: the raw
    ratio features are long-tailed, their ranks are uniform on (0, 1).

      x = `time_col`, y = the component's exposure to each function.
      A faint line per city-event (drawn only where a unit has at least MIN_FIT
      points) says whether the units agree; the dark line is the pooled fit.

    `ranked` needs columns func_<cat> for every category, `time_col`, and code.
    """
    MIN_FIT = 4
    codes = list(pd.unique(ranked['code']))
    cmap = plt.get_cmap('tab20')
    colors = {c: cmap(i % 20) for i, c in enumerate(codes)}
    x_all = ranked[time_col].to_numpy(dtype=float)

    fig = plt.figure(figsize=(16.5, 8.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.42], wspace=0.28,
                          hspace=0.36)
    for j, cat in enumerate(categories):
        ax = fig.add_subplot(gs[j // 3, j % 3])
        y_all = ranked[f'func_{cat}'].to_numpy(dtype=float)
        ok_all = np.isfinite(x_all) & np.isfinite(y_all)
        for c in codes:
            m = (ranked['code'] == c).to_numpy()
            ax.scatter(x_all[m], y_all[m], s=24, color=colors[c], alpha=0.85,
                       edgecolor='white', linewidth=0.4, zorder=3)
            ok = m & ok_all
            if ok.sum() >= MIN_FIT and np.ptp(x_all[ok]) > 0:
                cf = np.polyfit(x_all[ok], y_all[ok], 1)
                xs = np.linspace(x_all[ok].min(), x_all[ok].max(), 20)
                ax.plot(xs, np.polyval(cf, xs), color=colors[c], lw=1.0,
                        alpha=0.5, zorder=2)
        if ok_all.sum() >= 3 and np.ptp(x_all[ok_all]) > 0:
            cf = np.polyfit(x_all[ok_all], y_all[ok_all], 1)
            xs = np.linspace(x_all[ok_all].min(), x_all[ok_all].max(), 50)
            ax.plot(xs, np.polyval(cf, xs), color='0.15', lw=2.0, zorder=4)
        ttl = cat
        if rho_rank is not None and f'func_{cat}' in rho_rank:
            ttl += f'    spearman $\\rho$ = {rho_rank[f"func_{cat}"]:+.2f}'
        ax.set_title(ttl, fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel(f'{time_col}  (within-unit rank)', fontsize=9)
        ax.set_ylabel(f'flow touching {cat}  (within-unit rank)', fontsize=9)
        ax.tick_params(labelsize=8)

    axl = fig.add_subplot(gs[:, 3])
    axl.axis('off')
    handles = [plt.Line2D([], [], marker='o', ls='', color=colors[c], label=c,
                          markersize=6) for c in codes]
    axl.legend(handles=handles, loc='center', frameon=False, fontsize=9,
               title='city-event', title_fontsize=10)

    fig.suptitle(
        f'{time_col} vs functional composition, all city-events pooled\n'
        'both axes = within-unit normalised rank   ·   one point = one component'
        '   ·   faint line = that city-event’s own fit   ·   dark line = pooled fit',
        fontsize=13)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def vis_exposure_vs_cumloss(df, panels, metric='cum_loss', group_col='code',
                            weight_col=None, save_path=None, title=None):
    """One panel per definition of "how hard was this city hit", against the
    same y — so the question the figure answers is which DEFINITION relates to
    the loss, not whether some single chosen one happens to.

    `panels` is a list of (column, axis label, kind) with kind in
    {'discrete', 'continuous'}: discrete columns are jittered and get a weighted
    median per level, continuous ones get a weighted least-squares line.  Every
    panel is WEIGHTED by `weight_col` (normally weight_normal): point area is
    the component's within-city weight share and the reported rho is a weighted
    Spearman, because components carry very unequal shares of a city and an
    equal-weight reading would not describe the city the panel is about.
    """
    import math
    n = len(panels)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.5 * nrows),
                             squeeze=False)

    groups = df[group_col].to_numpy()
    uniq = list(dict.fromkeys(groups.tolist()))
    y = df[metric].to_numpy(dtype=float)

    if weight_col is not None:
        w = df[weight_col].to_numpy(dtype=float)
        share = np.full(len(df), np.nan)
        for u in uniq:
            m = groups == u
            tot = np.nansum(w[m])
            share[m] = (w[m] / tot if tot > 0 else 1.0 / max(m.sum(), 1))
        share = np.nan_to_num(share, nan=1.0 / max(len(df), 1))
    else:
        share = np.full(len(df), 1.0 / max(len(df), 1))
    sizes = 14.0 + 300.0 * (share / max(share.max(), 1e-12))

    def _wspearman(xv, yv, wv):
        rx, ry = rankdata(xv), rankdata(yv)
        mx, my = np.average(rx, weights=wv), np.average(ry, weights=wv)
        sx = math.sqrt(np.average((rx - mx) ** 2, weights=wv))
        sy = math.sqrt(np.average((ry - my) ** 2, weights=wv))
        if sx <= 0 or sy <= 0:
            return float('nan')
        return float(np.average((rx - mx) * (ry - my), weights=wv) / (sx * sy))

    def _wquantile(v, wv, q):
        o = np.argsort(v)
        v, wv = v[o], wv[o]
        tot = wv.sum()
        if tot <= 0:
            return float('nan')
        return float(np.interp(q, (np.cumsum(wv) - 0.5 * wv) / tot, v))

    cmap = plt.get_cmap('tab20')
    rng = np.random.default_rng(0)
    for idx, (col, xlabel, kind) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        x = df[col].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if kind == 'discrete':
            levels = sorted(np.unique(x[np.isfinite(x)]).tolist())
            span = (max(levels) - min(levels)) or 1.0
            jit = rng.uniform(-0.022, 0.022, len(df)) * span
            xp = x + jit
        else:
            xp = x
        for k, u in enumerate(uniq):
            m = groups == u
            ax.scatter(xp[m], y[m], s=sizes[m], color=cmap(k % 20), alpha=0.75,
                       edgecolor='white', linewidth=0.5, label=str(u), zorder=3)
        if kind == 'discrete':
            for L in levels:
                sel = (x == L) & ok
                if sel.sum():
                    med = _wquantile(y[sel], share[sel], 0.5)
                    ax.plot([L - 0.035 * span, L + 0.035 * span], [med] * 2,
                            color='black', lw=2.2, zorder=5)
            ax.set_xticks(levels)
        elif ok.sum() >= 3 and np.ptp(x[ok]) > 0:
            # Weighted least squares, the continuous counterpart of the
            # per-level weighted median.
            cf = np.polyfit(x[ok], y[ok], 1, w=np.sqrt(share[ok]))
            xs = np.linspace(x[ok].min(), x[ok].max(), 50)
            ax.plot(xs, np.polyval(cf, xs), color='black', lw=1.8, zorder=5)
        rho = _wspearman(x[ok], y[ok], share[ok]) if ok.sum() > 2 else float('nan')
        ax.axhline(0.0, color='0.8', lw=0.8, zorder=1)
        ax.set_title(f'{xlabel}\nweighted Spearman rho = {rho:+.2f} '
                     f'(n={int(ok.sum())})', fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(metric, fontsize=9)
        ax.tick_params(labelsize=8)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    handles = [plt.Line2D([], [], marker='o', ls='', color=cmap(k % 20),
                          label=u, markersize=6) for k, u in enumerate(uniq)]
    refs = [0.05, 0.15, 0.30]
    handles += [plt.Line2D([], [], marker='o', ls='', color='0.6',
                           markersize=math.sqrt(14.0 + 300.0 * (r / max(share.max(), 1e-12))),
                           label=f'weight {r:.0%}') for r in refs]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=8.5, title='city-event / point size',
               title_fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig
