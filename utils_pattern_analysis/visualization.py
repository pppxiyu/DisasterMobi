"""
Visualisation helpers for pattern analysis (temporal decay, NMF, Tucker).
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import seaborn as sns
import contextily as cx
import pydeck as pdk
import ipywidgets as widgets
from IPython.display import display, clear_output


# ── Flow time-series ──────────────────────────────────────────────────────────

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

def vis_line_temporal_patterns(
    W, selected_indices=None, normalize=False, moving_avg=None,
    de_baseline=False, de_baseline_plot=False, n_days_back=None,
    frequency=None, slots_per_day=None, interval_hours=3, start_date='2021-04-15',
    stack=False, show_impact_line=True, use_line_types=True, return_series=False,
    output_dir='outputs', tag='',
):
    """
    Line subplots of selected temporal factor columns from W (e.g. NMF W matrix).
    Supports baseline removal, normalization, smoothing, and stacked overlay.
    Saves to: <output_dir>/temporal_patterns<tag>.png
    """
    if de_baseline and (n_days_back is None or frequency is None or slots_per_day is None):
        raise ValueError("de_baseline requires n_days_back, frequency, and slots_per_day.")

    indices = (list(range(W.shape[1])) if selected_indices is None
               else [selected_indices] if isinstance(selected_indices, int)
               else selected_indices)
    if not indices:
        print("No indices selected."); return

    num_plots = 1 if stack else len(indices)
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 7 if stack else 4 * num_plots),
                              sharex=True)
    if num_plots == 1:
        axes = [axes]

    colors     = plt.cm.tab10(np.linspace(0, 1, len(indices)))
    line_types = ['-', (0, (10, 5)), ':']
    all_series = []

    for i, idx in enumerate(indices):
        data = W[:, idx]

        if de_baseline:
            from utils_pattern_analysis.temporal import run_temporal_decay_analysis_pipeline_simplified
            data = np.array(run_temporal_decay_analysis_pipeline_simplified(
                flows=data.tolist(), n_days_back=n_days_back, frequency=frequency,
                slots_per_day=slots_per_day, show_plots=de_baseline_plot,
                start_date=start_date,
            ))

        if normalize:
            m, s = np.mean(data), np.std(data)
            data = (data - m) / s if s > 0 else data - m

        ax = axes[0] if stack else axes[i]
        ls = line_types[i % len(line_types)] if use_line_types else '-'

        if stack and n_days_back and slots_per_day:
            imp_start = len(data) - int(n_days_back * slots_per_day)
            data = data - np.mean(data[:imp_start])

        if not stack:
            ax.plot(data, color='black', linewidth=1.2,
                    alpha=0.4 if moving_avg else 1.0, label='Signal')

        if show_impact_line and n_days_back is not None and slots_per_day is not None:
            imp_start = len(data) - int(n_days_back * slots_per_day)
            ax.axvline(x=imp_start, color='red', linestyle='--', linewidth=2.5,
                       label='Impact Start' if i == 0 else '', alpha=0.8)

        if isinstance(moving_avg, int) and moving_avg > 1:
            smoothed = pd.Series(data).rolling(window=moving_avg, center=True).mean()
            ax.plot(smoothed, color=colors[i] if stack else 'blue', linestyle=ls,
                    linewidth=3, label=f'Comp {idx}' if stack else f'{moving_avg}-step MA')
        elif stack:
            ax.plot(data, color=colors[i], linestyle=ls, linewidth=2, alpha=0.7,
                    label=f'Comp {idx}')

        _tick_step = slots_per_day if slots_per_day else (24 // interval_hours)
        if not stack:
            ax.set_xlim(left=0)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(_tick_step))
            ax.set_ylabel(f"Comp {idx}", fontsize=18, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            if de_baseline or normalize:
                ax.axhline(0, color='gray', linestyle='-', linewidth=1.2, alpha=0.5)

        final = smoothed.values if (isinstance(moving_avg, int) and moving_avg > 1) else data
        all_series.append(final)

    if stack:
        axes[0].set_xlim(left=0)
        axes[0].xaxis.set_major_locator(ticker.MultipleLocator(_tick_step))
        axes[0].set_ylabel("Relative Intensity\n(Centred to Normal Mean)",
                            fontsize=20, labelpad=20)
        axes[0].tick_params(axis='both', which='major', labelsize=16)
        axes[0].axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
        axes[0].grid(True, linestyle=':', alpha=0.4)
        axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                       fontsize=12, ncol=min(len(indices), 6),
                       frameon=False, handlelength=6.0)

    os.makedirs(output_dir, exist_ok=True)
    plt.xlabel(f"Time Step ({interval_hours}H intervals)", fontsize=20, labelpad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'temporal_patterns{tag}.png'), bbox_inches='tight', dpi=150)
    plt.close()

    if return_series:
        return np.array(all_series).T


def vis_heatmap_temporal_signature(
    matrix, first_day="Monday", show_days=True, show_components=True,
    index_range=None, slots_per_day=5, interval_hours=3,
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


# ── Spatial plots ─────────────────────────────────────────────────────────────

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


def vis_line_temporal_signatures(factors, interval_hours=None, subplots=True,
                                  output_dir='outputs', tag=''):
    """Line plots of Tucker temporal factor U3 columns, either as subplots or overlaid.
    Saves to: <output_dir>/temporal_signatures<tag>.png

    Parameters
    ----------
    interval_hours : int, optional
        Hours per time slot (e.g. 3 for 3h data, 2 for 2h data).
        Pass 24 // SLOT_PER_DAY from config.  If None, the axis label omits
        the interval size (safe fallback for exploratory calls).
    """
    os.makedirs(output_dir, exist_ok=True)
    interval_label = f"{interval_hours}h" if interval_hours is not None else "?"
    U_time    = factors[2]
    n_trends  = U_time.shape[1]
    if subplots:
        fig, axes = plt.subplots(n_trends, 1, figsize=(10, 3 * n_trends), sharex=True)
        if n_trends == 1: axes = [axes]
        for i in range(n_trends):
            axes[i].plot(U_time[:, i], color=f'C{i}', linewidth=2)
            axes[i].set_title(f"Temporal Trend {i}", fontweight='bold')
            axes[i].set_ylabel("Intensity")
            axes[i].grid(True, linestyle=':', alpha=0.6)
        plt.xlabel(f"Time Step ({interval_label} intervals)")
    else:
        plt.figure(figsize=(10, 5))
        for i in range(n_trends):
            plt.plot(U_time[:, i], label=f"Trend {i}", linewidth=2)
        plt.title("Temporal Signatures (U3) – Overlaid", fontweight='bold')
        plt.xlabel(f"Time Step ({interval_label} intervals)")
        plt.ylabel("Relative Intensity"); plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'temporal_signatures{tag}.png'), bbox_inches='tight', dpi=150)
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


def vis_heatmap_component_mapping(weights, basis_name='BR', target_name='NO',
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


def vis_heatmap_od_error(X, error_matrix, t_idx, spatial_mapping=None, n=5,
                          output_dir='outputs', tag=''):
    """
    Side-by-side heatmaps of signed percentage error (where X>0) and absolute
    error (where X=0) for a single time step t_idx.
    Saves to: <output_dir>/od_error_t<t_idx><tag>.png
    """
    X_sl  = X[:, :, t_idx]
    E_sl  = error_matrix[:, :, t_idx]
    with np.errstate(divide='ignore', invalid='ignore'):
        pe    = np.nan_to_num((E_sl / X_sl) * 100, nan=0, posinf=0, neginf=0)
    labels = [spatial_mapping[i] if spatial_mapping else f"Tract {i}"
              for i in range(X.shape[0])]

    mask_nz = X_sl > 0
    valid_pe = pe[mask_nz]
    top_idx  = np.argsort(np.abs(valid_pe))[::-1][:n]
    o_idx, d_idx = np.where(mask_nz)
    print(f"### Top {n} Errors (t={t_idx}) ###")
    for i in top_idx:
        print(f"  {labels[o_idx[i]]} → {labels[d_idx[i]]}: PE={valid_pe[i]:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pe_lim = np.percentile(np.abs(pe[mask_nz]), 95) if np.any(mask_nz) else 100
    sns.heatmap(pe, mask=(X_sl == 0), cmap="RdBu_r", center=0,
                vmin=-pe_lim, vmax=pe_lim, ax=axes[0],
                cbar_kws={'label': 'Percentage Error (%)'})
    axes[0].set_title("Percentage Error (X > 0)")

    abs_lim = np.max(np.abs(E_sl[X_sl == 0])) if np.any(X_sl == 0) else 1
    sns.heatmap(E_sl, mask=(X_sl != 0), cmap="mako",
                vmin=-abs_lim, vmax=0, ax=axes[1],
                cbar_kws={'label': 'Absolute Error (Hallucination)'})
    axes[1].set_title("Absolute Error (X = 0)")
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'od_error_t{t_idx}{tag}.png'), bbox_inches='tight', dpi=150)
    plt.close()


# ── OD flow map (pydeck) ─────────────────────────────────────────────────────

def vis_map_od_flow(
    od_matrices, gdfs, frame_titles=None, id_col='aggr_id', min_flow=0,
    cmap_name='plasma', alpha_range=(0.3, 0.8), curve_rad=0.15, arrow_size=0.006,
    max_line_width=15, vmax=None, delay=1.0, save_dir='',
):
    """
    Saves an animated OD flow map as an HTML file (pydeck).
    Interactive playback only works inside Jupyter; otherwise saves the first frame.
    """
    num_frames = len(od_matrices)
    if not isinstance(gdfs, list):
        gdfs = [gdfs] * num_frames
    if frame_titles is None:
        frame_titles = [f"Timestep {i+1}" for i in range(num_frames)]

    max_flow_val = vmax
    if max_flow_val is None:
        max_flow_val = 0
        for od_df in od_matrices:
            tmp   = od_df.copy(); tmp.index.name = 'origin'
            melted = tmp.reset_index().melt(id_vars='origin', var_name='destination', value_name='flow')
            melted['flow'] = pd.to_numeric(melted['flow'], errors='coerce').fillna(0)
            lm    = melted[melted['origin'] != melted['destination']]['flow'].max()
            if not np.isnan(lm) and lm > max_flow_val:
                max_flow_val = lm

    norm = mcolors.Normalize(vmin=min_flow, vmax=max_flow_val, clip=True)
    cmap = cm.get_cmap(cmap_name)
    min_a, max_a = alpha_range

    def get_color(flow):
        fn = norm(flow) if max_flow_val > min_flow else 1.0
        r, g, b = [int(c * 255) for c in cmap(fn)[:3]]
        return [r, g, b, int((min_a + fn * (max_a - min_a)) * 255)]

    frames_data = []
    for idx, (od_df, gdf) in enumerate(zip(od_matrices, gdfs)):
        coords = gdf.set_index(id_col)[['lon', 'lat']]
        tmp = od_df.copy(); tmp.index.name = 'origin'
        edges = tmp.reset_index().melt(id_vars='origin', var_name='destination', value_name='flow')
        edges['flow'] = pd.to_numeric(edges['flow'], errors='coerce').fillna(0)
        edges = edges[(edges['origin'] != edges['destination']) & (edges['flow'] > min_flow)]
        edges = edges.sort_values('flow', ascending=True).dropna()

        for col in ('start_lon','start_lat','end_lon','end_lat'):
            src = 'origin' if 'lon' in col or ('lat' in col and 'start' in col) else 'destination'
            axis = 'lon' if 'lon' in col else 'lat'
            edges[col] = edges[src].map(coords[axis])
        edges = edges.dropna(subset=['start_lon','start_lat','end_lon','end_lat']).copy()

        paths, arrows = [], []
        if not edges.empty:
            edges['color']      = edges['flow'].apply(get_color)
            edges['line_width'] = edges['flow'].apply(
                lambda x: float((norm(x) if max_flow_val > min_flow else 1.0)
                                  * (max_line_width - 1) + 1)
            )
            for _, row in edges.iterrows():
                x0, y0, x2, y2 = row.start_lon, row.start_lat, row.end_lon, row.end_lat
                dx, dy = x2-x0, y2-y0
                nx_v, ny_v = dy, -dx
                cxp = x0 + dx*.5 + nx_v*curve_rad
                cyp = y0 + dy*.5 + ny_v*curve_rad
                t   = np.linspace(0,1,20)
                bx  = (1-t)**2*x0 + 2*(1-t)*t*cxp + t**2*x2
                by  = (1-t)**2*y0 + 2*(1-t)*t*cyp + t**2*y2
                paths.append([[lo, la] for lo, la in zip(bx, by)])
                tx, ty = x2-cxp, y2-cyp
                ln = np.hypot(tx,ty)
                if ln > 0: tx, ty = tx/ln, ty/ln
                al = arrow_size * (1 + row.line_width * .15)
                aw = al * .6
                tip_x, tip_y = x2 - tx*.002, y2 - ty*.002
                bx2, by2    = tip_x - tx*al, tip_y - ty*al
                px, py      = -ty, tx
                arrows.append([[[tip_x,tip_y],
                                 [bx2+px*aw, by2+py*aw],
                                 [bx2-px*aw, by2-py*aw]]])
            edges['path']          = paths
            edges['arrow_polygon'] = arrows

        gdf_base = gdf[[id_col, gdf.geometry.name]].copy()
        if gdf_base.crs != "EPSG:4326":
            gdf_base = gdf_base.to_crs("EPSG:4326")
        frames_data.append({
            "edges":   edges.to_dict(orient="records") if not edges.empty else [],
            "nodes":   coords.reset_index().to_dict(orient="records"),
            "geojson": json.loads(gdf_base.to_json()),
        })

    initial_coords = gdfs[0].set_index(id_col)[['lon','lat']]
    view = pdk.ViewState(longitude=initial_coords['lon'].mean(),
                          latitude=initial_coords['lat'].mean(),
                          zoom=9, pitch=0, bearing=0)

    def render_frame(index):
        fd = frames_data[index]
        layers = [
            pdk.Layer("GeoJsonLayer", fd['geojson'], stroked=True, filled=True,
                      get_fill_color=[150,150,150,40], get_line_color=[100,100,100,150],
                      line_width_min_pixels=1),
            pdk.Layer("PathLayer", fd['edges'], get_path='path', get_color='color',
                      width_units='"pixels"', get_width='line_width'),
            pdk.Layer("PolygonLayer", fd['edges'], get_polygon='arrow_polygon',
                      get_fill_color='color', filled=True, stroked=False),
            pdk.Layer("ScatterplotLayer", fd['nodes'], get_position=['lon','lat'],
                      get_fill_color=[255,255,255,255], get_line_color=[0,0,0,255],
                      stroked=True, filled=True, radius_scale=100,
                      radius_min_pixels=3, radius_max_pixels=12),
        ]
        deck = pdk.Deck(layers=layers, initial_view_state=view,
                        tooltip={"html": "<b>O:</b>{origin} <b>D:</b>{destination} <b>F:</b>{flow}"},
                        map_provider="carto", map_style="light")
        deck.to_html(save_dir)
        print(f"Frame {index} saved to {save_dir}")

    render_frame(0)


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


# ── Distance decay (Figure 7 style) ─────────────────────────────────────────
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


# ── Feature correlation heatmap (split from/to cells) ─────────────────────────


def vis_heatmap_corr_split(rho, pval=None, time_cols=None,
                           categories=None, save_path=None,
                           cmap='RdBu_r', annot_fs=17):
    """
    Row-scaled correlation heatmap with split functional cells.  Rows are any
    metric set (resilience metrics, weekday_ratio, …).

    Time columns stay single cells (pass an empty list when there are none).
    Each category column packs share_from_<cat> in the UPPER half-cell and
    share_to_<cat> in the LOWER half-cell, with value labels (and significance
    stars) for both.  Each row is coloured by its own max |rho| over its
    displayed cells, so rows are independent and no colorbar is drawn —
    colours compare within a row only, numbers compare everywhere.

    Parameters
    ----------
    rho, pval  : DataFrames [metric rows × feature columns]; columns must
                 contain time_cols plus share_from_<cat> / share_to_<cat>
                 for every category.
    time_cols  : list[str] columns shown as single cells (may be empty).
    categories : list[str] functional category names (e.g. SF_CATEGORIES).
    """
    time_cols = time_cols or []
    rows = list(rho.index)
    n_r, n_t, n_c = len(rows), len(time_cols), len(categories)
    n_cols = n_t + n_c

    # Upper/lower value matrices per display cell.  Time cells repeat the same
    # value in both halves so they render as one solid cell.
    Vup = np.full((n_r, n_cols), np.nan)
    Vlo = np.full((n_r, n_cols), np.nan)
    for i, rname in enumerate(rows):
        for j, t in enumerate(time_cols):
            Vup[i, j] = Vlo[i, j] = rho.loc[rname, t]
        for j, c in enumerate(categories):
            Vup[i, n_t + j] = rho.loc[rname, f'share_from_{c}']
            Vlo[i, n_t + j] = rho.loc[rname, f'share_to_{c}']

    # Per-row colour scale over every displayed value (time + from + to).
    allv = np.concatenate([Vup, Vlo], axis=1)
    scale = np.nanmax(np.abs(allv), axis=1, keepdims=True)
    scale = np.where(np.isnan(scale) | (scale == 0), 1.0, scale)
    Cup, Clo = Vup / scale, Vlo / scale

    fine = np.empty((2 * n_r, n_cols))
    fine[0::2, :] = Cup
    fine[1::2, :] = Clo

    fig, ax = plt.subplots(figsize=(2.0 * n_cols + 2.4, 1.75 * n_r + 2.2),
                           constrained_layout=True)
    ax.imshow(fine, cmap=cmap, vmin=-1, vmax=1, aspect='auto',
              extent=[0, n_cols, n_r, 0], interpolation='nearest')
    for spine in ax.spines.values():
        spine.set_visible(False)

    # White main grid, plus a thin half split inside the category region only.
    for x in range(n_cols + 1):
        ax.axvline(x, color='white', lw=2)
    for y in range(n_r + 1):
        ax.axhline(y, color='white', lw=2)
    for i in range(n_r):
        ax.plot([n_t, n_cols], [i + 0.5] * 2, color='white', lw=0.8)

    def _stars(rname, col):
        if pval is None or np.isnan(pval.loc[rname, col]):
            return ''
        p = pval.loc[rname, col]
        return '**' if p < 0.01 else ('*' if p < 0.05 else '')

    for i, rname in enumerate(rows):
        for j, t in enumerate(time_cols):
            v = Vup[i, j]
            if np.isnan(v):
                continue
            ax.text(j + 0.5, i + 0.5, f'{v:.2f}{_stars(rname, t)}',
                    ha='center', va='center', fontsize=annot_fs,
                    color='white' if abs(Cup[i, j]) > 0.6 else 'black')
        for j, c in enumerate(categories):
            x = n_t + j + 0.5
            vu, vl = Vup[i, n_t + j], Vlo[i, n_t + j]
            if not np.isnan(vu):
                ax.text(x, i + 0.27, f'{vu:.2f}{_stars(rname, f"share_from_{c}")}',
                        ha='center', va='center', fontsize=annot_fs,
                        color='white' if abs(Cup[i, n_t + j]) > 0.6 else 'black')
            if not np.isnan(vl):
                ax.text(x, i + 0.73, f'{vl:.2f}{_stars(rname, f"share_to_{c}")}',
                        ha='center', va='center', fontsize=annot_fs,
                        color='white' if abs(Clo[i, n_t + j]) > 0.6 else 'black')

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(list(time_cols) + list(categories), rotation=45,
                       ha='right', fontsize=18)
    ax.set_yticks(np.arange(n_r) + 0.5)
    ax.set_yticklabels(rows, fontsize=18)
    ax.tick_params(length=0)
    ax.text(1.0, 1.01, 'Upper = Outflow,  Lower = Inflow',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=16,
            color='dimgrey')
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def vis_bar_function_by_peakslot(df, categories, ncols=3, save_path=None,
                                 color_from='#1976D2', color_to='#E65100',
                                 group_col='peak_slot',
                                 label_col='peak_slot_label'):
    """
    Grouped bar charts of mean functional shares by a categorical temporal
    feature — one subplot per OCCUPIED group value, bars = the functional
    categories, paired by direction: share_from_<cat> (outflow, one colour)
    vs share_to_<cat> (inflow, other).

    The grouping feature is treated as CATEGORICAL (no ordering assumed) —
    this replaces its use in the rank-correlation analysis.  Works for any
    code/label column pair, e.g. peak_slot/peak_slot_label or
    peak_period/peak_period_label; groups are laid out by ascending code.

    Parameters
    ----------
    df         : per-component feature table with group_col, label_col,
                 share_from_<cat> and share_to_<cat> columns.
    categories : list[str] functional category names (e.g. SF_CATEGORIES).
    """
    import math
    from_cols = [f'share_from_{c}' for c in categories]
    to_cols   = [f'share_to_{c}'   for c in categories]

    groups = sorted(df[group_col].unique())
    n = len(groups)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows),
                             squeeze=False, constrained_layout=True)

    x = np.arange(len(categories))
    bw = 0.38                                   # bar width
    ymax = max(df[from_cols].to_numpy().max(), df[to_cols].to_numpy().max())

    for p, g in enumerate(groups):
        ax = axes[p // ncols][p % ncols]
        grp = df[df[group_col] == g]
        mean_from = grp[from_cols].mean().to_numpy()
        mean_to   = grp[to_cols].mean().to_numpy()
        ax.bar(x - bw / 2, mean_from, bw, color=color_from, label='from (outflow)')
        ax.bar(x + bw / 2, mean_to,   bw, color=color_to,   label='to (inflow)')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=10)
        ax.set_ylim(0, ymax * 1.08)
        ax.set_ylabel('mean share', fontsize=10)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        label = grp[label_col].iloc[0] if label_col in grp else str(g)
        ax.set_title(f'{label}  (n={len(grp)})', fontsize=12)
        if p == 0:
            ax.legend(fontsize=9, frameon=True)

    for p in range(n, nrows * ncols):
        axes[p // ncols][p % ncols].axis('off')
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def vis_scatter_component_features(df, pairs, ncols=2, save_path=None,
                                   color='#1976D2', point_size=90):
    """
    Scatter panels of feature pairs — one point per NMF component, uniform
    colour and size.

    Parameters
    ----------
    df     : per-component feature table containing the pair columns.
    pairs  : list of (x_col, y_col) tuples to plot (e.g. the strongest
             correlations).  weekday_ratio x-axes are drawn in log scale.
    """
    import math
    n = len(pairs)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 5.6 * nrows),
                             squeeze=False, constrained_layout=True)
    try:
        fig.get_layout_engine().set(w_pad=0.25, h_pad=0.25)
    except Exception:
        pass

    # Each (a, b) pair is drawn with b on the x axis and a on the y axis.
    for p, (yc, xc) in enumerate(pairs):
        ax = axes[p // ncols][p % ncols]
        ax.scatter(df[xc], df[yc], s=point_size, color=color, alpha=0.8,
                   edgecolor='white', linewidth=0.6, zorder=2)
        if xc == 'weekday_ratio':
            ax.set_xscale('log')
            ax.axvline(1.0, color='grey', linestyle=':', linewidth=1)
        if yc == 'weekday_ratio':
            ax.set_yscale('log')
            ax.axhline(1.0, color='grey', linestyle=':', linewidth=1)
        ax.set_xlabel(xc, fontsize=21)
        ax.set_ylabel(yc, fontsize=21)
        ax.tick_params(labelsize=18)

    for p in range(n, nrows * ncols):
        axes[p // ncols][p % ncols].axis('off')
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# ── Resilience (disaster drop-and-recovery) plots ─────────────────────────────

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


def vis_bar_resilience_by_peakslot(df, res_cols, ncols=2, save_path=None,
                                   color='#455A64', color_weekend='#8D6E63',
                                   group_col='peak_slot',
                                   label_col='peak_slot_label'):
    """
    Mean resilience metrics by a categorical temporal feature — one subplot
    PER METRIC (metrics have different scales, so they cannot share a y-axis),
    bar height = group mean, group sizes in the x labels.

    The weekend category comes from the FEATURE layer (temporal_features
    assigns weekend-dominated components the 'weekend' label, no time-of-day
    split); its bar is drawn in color_weekend.  Works for any code/label
    column pair, e.g. peak_slot/peak_slot_label or
    peak_period/peak_period_label; groups are laid out by ascending code.
    """
    import math
    codes = sorted(df[group_col].unique())
    groups, labels, colors = [], [], []
    for g in codes:
        grp = df[df[group_col] == g]
        lbl = grp[label_col].iloc[0] if label_col in grp else str(g)
        groups.append(grp)
        labels.append(f"{lbl}\n(n={len(grp)})")
        colors.append(color_weekend if lbl == 'weekend' else color)

    n = len(res_cols)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(0.95 * len(groups) * ncols + 2,
                                      3.2 * nrows),
                             squeeze=False, constrained_layout=True)
    x = np.arange(len(groups))
    for p, col in enumerate(res_cols):
        ax = axes[p // ncols][p % ncols]
        means = [grp[col].mean() for grp in groups]
        ax.bar(x, means, 0.6, color=colors)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(col, fontsize=12)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    for p in range(n, nrows * ncols):
        axes[p // ncols][p % ncols].axis('off')
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


