"""
Visualisation helpers for pattern analysis.

Most functions serve the production NMF pipeline (run_pattern_nmf.py).  The
flow time-series section marked in its header
serves utils/neural_network/temporal_decay.py (via utils.pattern_analysis
.temporal) and can be skipped by production readers.
"""
import os
import numpy as np
from scipy.stats import rankdata, spearmanr, pearsonr, skewnorm
import pandas as pd
import matplotlib
# Headless-safe: every figure here is written to disk, never shown.  Without
# this, a detached run lazily imports the Qt GUI backend at the first
# plt.subplots and can die on a Windows access violation inside Qt.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as _Rectangle, FancyArrowPatch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
import seaborn as sns


# ── Flow time-series (only used by the GRU temporal_decay module) ────────

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


# ── Origin × Destination functional heatmaps (paper Fig. 9) ───────────────────

def vis_heatmap_od_function(M, categories, ncols=None,
                            save_path=None, cmap='YlGnBu'):
    """
    Grid of per-component origin×destination functional heatmaps (paper Fig. 9).

    Parameters
    ----------
    M          : ndarray [k × C × C]  per-component proportions (rows = origin
                 function, cols = destination function).
    categories : list[str]            axis labels (length C).
    ncols      : int | None           panels per row; None picks a WIDE grid
                 (at most two rows) so the panel strip fits a slide.
    save_path  : str | None           PNG path; created if its directory is absent.

    Returns the Matplotlib Figure.
    """
    import math
    M = np.asarray(M, dtype=float)
    k, C = M.shape[0], len(categories)
    # WIDE by default: cap the grid at two rows and balance the columns, so 12
    # components read as 6x2 rather than the old portrait 3x4.  A slide is
    # wider than it is tall, and a square panel strip wastes its width.
    if ncols is None:
        nrows = 1 if k <= 6 else 2
        ncols = math.ceil(k / nrows)
    else:
        ncols = max(1, min(ncols, k))
        nrows = math.ceil(k / ncols)
    vmax = M.max() if M.size and M.max() > 0 else 1.0

    # Slide-size type throughout; the cell values stay because a heat map's
    # numbers are its readout, not a callout, and the wide grid leaves each
    # cell roughly 0.6 in -- room enough for them at this size.
    FS_ANNOT, FS_TICK, FS_AXLABEL, FS_TITLE, FS_CBAR = 14, 16, 20, 21, 16

    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(3.9 * ncols + 1.7, 4.5 * nrows + 0.9),
                             constrained_layout=True)

    im = None
    for pos in range(k):
        r, c = pos // ncols, pos % ncols
        ax = axes[r][c]
        im = ax.imshow(M[pos], cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
        for a in range(C):
            for b in range(C):
                v = M[pos, a, b]
                if v > 0:
                    ax.text(b, a, f'{v:.2f}', ha='center', va='center',
                            fontsize=FS_ANNOT,
                            color='white' if v > 0.6 * vmax else 'black')
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        # Tick labels only on the OUTER edge: the six categories repeat in every
        # panel, so naming them once per row/column is the same information at a
        # fraction of the ink.
        last_row = (r == nrows - 1) or (pos + ncols >= k)
        if last_row:
            ax.set_xticklabels(categories, rotation=45, ha='right',
                               fontsize=FS_TICK)
        else:
            ax.set_xticklabels([])
        ax.set_yticklabels(categories if c == 0 else [], fontsize=FS_TICK)
        ax.set_title(f'Component {pos}', fontsize=FS_TITLE, pad=8)

    for pos in range(k, nrows * ncols):                  # hide unused panels
        axes[pos // ncols][pos % ncols].axis('off')

    # The axis names belong to the whole grid, so they are written once.
    fig.supxlabel('destination', fontsize=FS_AXLABEL)
    fig.supylabel('origin', fontsize=FS_AXLABEL)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75,
                            pad=0.015)
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


def vis_scatter_city_pred(df, pred_col, gt_col='cum_loss_gt', title=None,
                          save_path=None):
    """CITY-level LOO calibration scatter: one point per city-event, predicted
    city cum_loss (x) against the ground-truth city cum_loss (y), both in
    day-equivalents; points labelled with the city-event code.  The dashed y=x
    line is perfect prediction; the title carries the LOO R² and MAE over the
    plotted cities.  Rows where either value is missing are dropped (and the
    stats computed on what is plotted)."""
    from sklearn.metrics import r2_score
    sub = df[[gt_col, pred_col]].dropna()
    y, p = sub[gt_col].to_numpy(float), sub[pred_col].to_numpy(float)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(p, y, s=42, color='#1976D2', alpha=0.85, edgecolor='white',
               linewidth=0.6, zorder=3)
    for code, xp, yt in zip(sub.index, p, y):
        ax.annotate(str(code), (xp, yt), fontsize=6.5, color='#555555',
                    xytext=(3, 3), textcoords='offset points')
    lo, hi = float(min(p.min(), y.min())), float(max(p.max(), y.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '--', color='grey',
            lw=1, zorder=2)
    r2 = r2_score(y, p) if len(y) >= 2 else np.nan
    mae = float(np.mean(np.abs(y - p)))
    head = (title + '\n') if title else ''
    ax.set_title(f'{head}LOO R²={r2:+.2f}   MAE={mae:.2f}   (n={len(y)})',
                 fontsize=10)
    ax.set_xlabel('predicted city cum_loss (day-equivalents)', fontsize=9)
    ax.set_ylabel('actual city cum_loss (day-equivalents)', fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def vis_bar_cross_city_resi_pred(df, gt_col='cum_loss_gt',
                                 pred_cols=(('Prediction (kNN)', 'cum_loss_pred_knn'),
                                            ('Prediction (ridge)', 'cum_loss_pred_ridge')),
                                 baseline_col='cum_loss_baseline', baseline_label='Baseline',
                                 save_path=None, title=None,
                                 ylabel='city cum_loss (day-equivalents)'):
    """Two-panel city-level prediction figure, `pred_cols` = ((label, column), ...)
    with absent columns skipped and `baseline_col` appended as a further method.

    Panel a (hero) — observed city cum_loss as a grey bar per city-event, each
    prediction a marker on a stem dropped to the observed value, so the ERROR is
    the segment rather than a height difference the eye must subtract.  Cities
    are sorted by observed cum_loss DESCENDING: the observed series becomes a
    monotone staircase, which makes the predictors' compression toward the
    middle legible at a glance instead of hidden across an arbitrary order.

    Panel b — the aggregate verdict: leave-one-out R² per method as a horizontal
    bar against a marked zero (the threshold for beating a constant mean), MAE
    printed at the bar end.  Its method labels are drawn IN each method's colour
    and thereby serve as the figure's only legend, so panel a carries none.

    Method names are passed through verbatim — spell them out rather than using
    internal codenames, since this is the reader-facing figure.  `title` is
    optional and normally left unset: the axes label themselves and the caption
    belongs to the manuscript.  Saves a 600 dpi PNG."""
    import textwrap
    from matplotlib.transforms import blended_transform_factory
    from sklearn.metrics import r2_score
    # One neutral family (observed, baseline) + one signal family (the two
    # predictors).  Green/red stay reserved for directional cues elsewhere.
    # The baseline's grey is kept DARKER than the observed bar so the control
    # series never reads as part of the reference bars.
    _PRED_COLORS = ['#0F4D92', '#9A4D8E', '#42949E', '#B64342']
    _OBS_FILL, _OBS_EDGE, _NEUTRAL = '#E3E3E3', '#9A9A9A', '#4D4D4D'

    methods = []                                   # (label, values, colour)
    for i, (lab, col) in enumerate([tuple(p)[:2] for p in pred_cols]):
        if col in df.columns:
            methods.append((lab, df[col].to_numpy(dtype=float),
                            _PRED_COLORS[i % len(_PRED_COLORS)]))
    if baseline_col in df.columns:
        methods.append((baseline_label, df[baseline_col].to_numpy(dtype=float),
                        _NEUTRAL))

    gt = df[gt_col].to_numpy(dtype=float)
    order = np.argsort(-gt)                        # observed, high -> low
    codes = [str(c) for c in np.asarray(df.index)[order]]
    gt_s = gt[order]

    nature_rc = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 7, 'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.8, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    }
    with plt.rc_context(nature_rc):
        fig = plt.figure(figsize=(7.2, 3.4))
        gs = fig.add_gridspec(1, 2, width_ratios=[2.75, 1.25], wspace=0.30)
        ax = fig.add_subplot(gs[0, 0])
        axr = fig.add_subplot(gs[0, 1])

        x = np.arange(len(codes))
        bar_w = 0.76
        ax.bar(x, gt_s, width=bar_w, color=_OBS_FILL, edgecolor=_OBS_EDGE,
               linewidth=0.6, zorder=2, label='Observed')
        n_m = len(methods)
        # Every stem must sit INSIDE its own bar: the span is a fraction of the
        # bar's HALF-width, so a marker can never drift over the neighbouring
        # city and make its stem look mis-assigned.
        span = 0.60 * (bar_w / 2.0)
        offs = (np.linspace(-span, span, n_m) if n_m > 1 else np.array([0.0]))
        marks = ['o', 's', '^', 'D']
        for j, (lab, vals, color) in enumerate(methods):
            v = vals[order]
            xs = x + offs[j]
            # Stem from the observed value to the prediction: the visible
            # segment IS the signed error.
            ax.vlines(xs, gt_s, v, color=color, linewidth=0.7, alpha=0.55,
                      zorder=3)
            ax.plot(xs, v, marks[j % len(marks)], color=color, markersize=3.1,
                    markeredgecolor='white', markeredgewidth=0.35, linestyle='none',
                    zorder=4)
        ax.axhline(0, color='#B0B0B0', linewidth=0.6, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(codes, rotation=45, ha='right', fontsize=6.2)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.set_xlabel('city-event, sorted by observed loss', fontsize=7.5)
        ax.margins(x=0.02, y=0.10)
        ax.tick_params(axis='y', labelsize=6.8)
        ax.text(0.0, 1.04, 'a', transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='bottom')
        # Name the observed series in place — the only direct label panel a needs.
        ax.text(0.035, 1.045, 'grey bar = observed; marker = predicted',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=6.3,
                color='#5A5A5A')

        # Panel b: LOO R² per method.  Method names are drawn INSIDE this axis,
        # above their own bar and in their own colour — they are the figure's
        # only legend, and keeping them inside stops long names from spilling
        # over panel a (which y-tick labels would do).
        r2s, maes = [], []
        for lab, vals, color in methods:
            ok = np.isfinite(vals) & np.isfinite(gt)
            r2s.append(r2_score(gt[ok], vals[ok]) if ok.sum() >= 2 else np.nan)
            maes.append(float(np.mean(np.abs(gt[ok] - vals[ok]))))
        ypos = np.arange(len(methods))[::-1] * 1.0      # first method on top
        axr.barh(ypos, r2s, height=0.42,
                 color=[c for _l, _v, c in methods], edgecolor='none', zorder=3)
        axr.axvline(0, color='#4D4D4D', linewidth=0.8, zorder=4)
        lo, hi = float(np.nanmin(r2s + [0.0])), float(np.nanmax(r2s + [0.0]))
        pad = 0.18 * max(hi - lo, 0.1)
        axr.set_xlim(lo - pad * 1.9, hi + pad * 1.9)
        axr.set_ylim(min(ypos) - 0.55, max(ypos) + 0.72)
        tr = blended_transform_factory(axr.transAxes, axr.transData)
        for yp, r2v, mae, (lab, _v, color) in zip(ypos, r2s, maes, methods):
            axr.text(0.015, yp + 0.20, textwrap.fill(lab, 26), transform=tr,
                     ha='left', va='bottom', fontsize=6.2, color=color,
                     linespacing=1.15)
            # MAE at the far end of the bar, pointing away from zero.
            axr.text(r2v + (pad * 0.22 if r2v >= 0 else -pad * 0.22), yp,
                     f'R² {r2v:+.2f}   MAE {mae:.2f}', va='center',
                     ha='left' if r2v >= 0 else 'right', fontsize=5.9,
                     color='#4D4D4D')
        axr.set_yticks([])
        axr.set_xlabel(f'leave-one-out R²  (n = {len(gt)} city-events)',
                       fontsize=7.5)
        axr.tick_params(axis='x', labelsize=6.5)
        axr.spines['left'].set_visible(False)
        axr.text(0.0, 1.04, 'b', transform=axr.transAxes, fontsize=8,
                 fontweight='bold', va='bottom')

        if title:
            fig.suptitle(textwrap.fill(title, 118), fontsize=8, y=1.06)
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=600, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_bar_curve_mae(df, save_path=None, colors=None, names=None, nrows=2,
                      ylabel='city-curve MAE (fraction of the normal baseline)'):
    """Grouped bar: per city-event, the ERROR of each curve-prediction method as
    side-by-side bars, one bar per column of `df` in column order, y = the mean
    absolute deviation between the predicted and observed city relative curve
    over the disaster window.  Unlike vis_bar_cross_city_resi_pred there is no
    ground-truth bar, because every bar IS an error and lower is better.

    The city-events are dealt over `nrows` stacked rows sharing one y scale:
    13 units x 4 methods in a single row forced a ~5:1 canvas on which nothing
    could be read at slide size.  Both rows keep the SAME x unit width, so a
    short final row leaves whitespace rather than fattening its bars, and bar
    widths stay comparable across rows.

    Each legend entry carries that method's all-unit mean, which is what the
    corner text block used to hold: with four methods it no longer fitted on
    one line, and the number belongs next to the colour key that identifies the
    method anyway.  Colours come from _BAR_METHOD_COLORS by METHOD NAME -- the
    bar's OWN set, see the note there for why it is not the curve page's; an
    explicit `colors` overrides, and unknown labels fall back to the positional
    palette.
    `names` maps code -> the full city-event title for the x tick labels.
    PNG >= 300 dpi."""
    codes = list(df.index)
    methods = list(df.columns)
    names = names or {}
    fallback = colors or ['#0F4D92', '#E28E2C', '#7B5EA7', '#767676', '#4C9F70']
    palette = [(_BAR_METHOD_COLORS[m] if not colors and m in _BAR_METHOD_COLORS
                else fallback[i % len(fallback)])
               for i, m in enumerate(methods)]
    means = {m: float(np.nanmean(df[m].to_numpy(dtype=float))) for m in methods}
    nrows = max(int(nrows), 1)
    per_row = int(np.ceil(len(codes) / nrows)) if codes else 1
    rows = [codes[i * per_row:(i + 1) * per_row] for i in range(nrows)]
    rows = [r for r in rows if r]

    rc = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 15, 'axes.labelsize': 17,
        'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 1.1, 'legend.frameon': False,
    }
    with plt.rc_context(rc):
        fig, axes2d = plt.subplots(len(rows), 1, squeeze=False, sharey=True,
                                   figsize=(1.62 * per_row + 1.5,
                                            3.05 * len(rows) + 1.25))
        axes = axes2d.ravel()
        n = max(len(methods), 1)
        w = 0.8 / n
        top = np.nanmax(df.to_numpy(dtype=float)) if len(df) else 1.0
        for ax, grp in zip(axes, rows):
            x = np.arange(len(grp))
            sub = df.loc[grp]
            for i, m in enumerate(methods):
                vals = sub[m].to_numpy(dtype=float)
                bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, width=w * 0.9,
                              label=m, color=palette[i % len(palette)],
                              edgecolor='#2b2b2b', linewidth=0.7)
                for bar in bars:
                    h = bar.get_height()
                    if np.isnan(h):
                        continue
                    # Rotated, and it has to stay that way: four bars share a
                    # group, so a horizontal "0.064" overruns its neighbours
                    # even at this width (measured, not guessed).
                    ax.annotate(f'{h:.3f}',
                                (bar.get_x() + bar.get_width() / 2, max(h, 0)),
                                xytext=(0, 4), textcoords='offset points',
                                rotation=90, ha='center', va='bottom',
                                fontsize=10.5, color='#3a3a3a')
            ax.set_xticks(x)
            # "Baton Rouge (Ida)" -> two horizontal lines.  A row holds only
            # `per_row` groups, so the widest city name still fits inside its
            # group, and upright ticks read better than rotated ones.
            ax.set_xticklabels([names.get(c, c).replace(' (', '\n(', 1)
                                for c in grp], ha='center')
            # identical x unit width on every row (see the docstring)
            ax.set_xlim(-0.62, per_row - 0.38)
            ax.set_ylim(0, top * 1.34)
        h, l = axes[0].get_legend_handles_labels()
        l = [f'{m}  (mean {means[m]:.3f})' for m in l]
        # supylabel defaults to x=0.02 while tight_layout reserves its own
        # left margin, which left a visible channel between the label and the
        # tick numbers; pin both so the label sits just outside the ticks.
        fig.supylabel(ylabel, fontsize=17, x=0.028)
        # Two columns, not one row: the entries carry a mean each, and four of
        # them abreast make the legend WIDER than the plot area -- which then
        # sets the saved width and pads a dead margin down the left side.
        fig.tight_layout(rect=(0.052, 0.10, 1.0, 1.0))
        fig.legend(h, l, loc='lower center', ncol=min(len(methods), 2),
                   fontsize=14, handlelength=1.5, columnspacing=2.4,
                   labelspacing=0.5, bbox_to_anchor=(0.55, 0.005))
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    return fig

def vis_curves_city_pred(days, gt, method_curves, save_path=None, title=None,
                         ylabel='daily mobility (flow volume per day)',
                         slide=False):
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
        'font.size': 20 if slide else 8,
        'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 1.3 if slide else 0.8, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    }
    lw_gt, lw_m, ms_gt, ms_m = ((3.0, 2.4, 7, 6) if slide
                                else (1.8, 1.2, 3, 2.5))
    # slide mode reads the shared per-method style map; legacy keeps its
    # original position-cycled palette so the archived copies do not change.
    gt = np.asarray(gt, dtype=float)
    with plt.rc_context(nature_rc):
        fig, ax = plt.subplots(figsize=(11.0, 6.0) if slide else (6.0, 3.6))
        ax.plot(days, gt, color='#3a3a3a', lw=lw_gt, marker='o', ms=ms_gt,
                label='ground truth', zorder=5)
        for i, (lab, vals) in enumerate(method_curves.items()):
            vals = np.asarray(vals, dtype=float)
            mae = np.nanmean(np.abs(vals - gt))
            col, dsh = (_curve_style(lab, i) if slide
                        else (_COLORS[i % len(_COLORS)], '--'))
            ax.plot(days, vals, color=col, lw=lw_m, ls=dsh,
                    marker='.', ms=ms_m, label=f'{lab}  (MAE {mae:.3g})')
        ax.set_xlabel('days since landfall', fontsize=24 if slide else 8)
        # rotated, the label is bounded by the AXES HEIGHT, not the width;
        # size it to fit rather than letting it run off the canvas
        fs_y = 24
        if slide:
            h_in = ax.get_position().height * fig.get_figheight()
            fs_y = float(np.clip(0.92 * h_in * 72 / (0.58 * len(ylabel)),
                                 14.0, 24.0))
        ax.set_ylabel(ylabel, fontsize=fs_y if slide else 8)
        ax.margins(y=0.08)
        if slide:
            # inside the axes: the curves rise to the right, so the lower right
            # is the one corner they leave free
            ax.legend(loc='lower right', fontsize=18, handlelength=1.8,
                      borderaxespad=0.8, labelspacing=0.4)
        else:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16),
                      ncol=2, fontsize=7, columnspacing=1.6, handlelength=1.8)
        # `slide` deliberately ignores `title`: the file name carries the
        # city-event, and a single-panel title reads as a figure title.
        if title and not slide:
            ax.set_title(title, fontsize=8.5)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_component_curves_grid(curves_obs, method_curves, save_path=None,
                              title=None, ncols=3, weights=None):
    """Per-component curve grid for ONE city-event: each panel shows a
    component's OBSERVED relative curve (solid dark) and one dashed line per
    method in `method_curves` (dict label -> DataFrame [days × k] aligned with
    `curves_obs`).  The baseline r = 1 is drawn as a grey rule.  `weights`
    (optional, length k, positionally aligned with `curves_obs.columns`) is the
    city-aggregation weight; when given it is appended to each panel title."""
    _COLORS = ['#0F4D92', '#4C9F70', '#7B5EA7', '#E28E2C', '#B0413E']
    import math
    k = curves_obs.shape[1]
    days = curves_obs.index.to_numpy()
    nrows = math.ceil(k / ncols)
    with plt.rc_context(_SLIDE_RC):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.4 * ncols, 3.5 * nrows),
                                 squeeze=False, sharex=True)
        for j in range(nrows * ncols):
            ax = axes[j // ncols][j % ncols]
            if j >= k:
                ax.axis('off')
                continue
            ax.axhline(1.0, color='#BBBBBB', lw=1.4, zorder=1)
            ax.plot(days, curves_obs.iloc[:, j], color='#3a3a3a', lw=2.6,
                    zorder=5, label='observed')
            for i, (lab, dfm) in enumerate(method_curves.items()):
                col, dsh = _curve_style(lab, i)
                ax.plot(days, dfm.iloc[:, j], color=col, lw=2.2, ls=dsh,
                        label=lab)
            ttl = f'component {curves_obs.columns[j]}'
            if weights is not None:
                ttl += f'  ($w$ = {weights[j]:.2f})'
            ax.set_title(ttl)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   ncol=min(4, 1 + len(method_curves)), fontsize=19,
                   bbox_to_anchor=(0.5, -0.012))
        # `title` is accepted for call compatibility but not drawn: the file
        # name carries the city-event and a suptitle is not wanted.
        fig.tight_layout(rect=(0, 0.05, 1, 1.0))
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


def deck_diverging_cmap():
    """The deck's diverging ramp for a signed correlation on [-1, 1]:
    AMBER for negative, white at zero, NYU PURPLE for positive.

    Shared by every Spearman-rho heat map so one quantity keeps one palette,
    and colour-blind legible in a way the red-green ramp it replaced was not.
    A fresh copy each call -- callers set_bad() on it."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('deck_diverging', [
        (0.00, '#7A3B06'), (0.24, '#E28E2C'), (0.44, '#F7E3C6'),
        (0.50, '#FFFFFF'),
        (0.56, '#E4D8EF'), (0.76, '#9A6ABA'), (1.00, '#3F0068')])


def vis_heatmap_pair_transfer(mat, save_path=None, vmax=1.0,
                              xlabel='test', ylabel='train', blocks=None,
                              cbar_label='Spearman ρ', names=None):
    """Square pairwise cross-city transfer heatmap: rows = train unit, cols =
    test unit, each cell = how well that (train, test) pair transfers.  The
    metric is the caller's — name it in `cbar_label` — and `vmax` must match its
    range: 1.0 leaves a correlation untouched, while an unbounded metric needs a
    smaller vmax and the colour then saturates.  Diverging colour centred at 0
    in the deck's own two hues — PURPLE for positive transfer, AMBER for
    negative — which also happens to be legible to red-green colour blindness,
    unlike the red-green ramp this replaced.  Each cell annotated (n/a if
    undefined).

    `names` (optional) maps code -> the full city-event title used on the
    ticks; codes are kept when it is absent.

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

    names = names or {}
    # Point size alone does not decide legibility here: the grid is metres
    # wide on paper and gets scaled DOWN to slide width, so a 16 pt tick
    # projects at about 7 pt.  The figure is kept compact and the type set
    # large against it, which is what survives the projection.
    FS_ANNOT, FS_TICK, FS_AXLABEL, FS_CBAR = 17, 25, 31, 23

    fig, ax = plt.subplots(figsize=(0.80 * n_c + 4.4, 0.72 * n_r + 3.0))
    cmap = deck_diverging_cmap()
    cmap.set_bad('#DEDEDE')                      # excluded diagonal / undefined
    im = ax.imshow(np.ma.masked_invalid(np.clip(V, -vmax, vmax)), cmap=cmap,
                   vmin=-vmax, vmax=vmax, aspect='auto')
    for i in range(n_r):
        for j in range(n_c):
            if diag[i, j]:
                continue                          # excluded: no number at all
            v = V[i, j]
            # both ends of the ramp go dark, so the number flips to white
            # rather than sitting unreadable on deep purple or deep amber
            col = ('#333333' if (np.isnan(v) or abs(v) < 0.58 * vmax)
                   else '#FFFFFF')
            ax.text(j, i, 'n/a' if np.isnan(v) else f'{v:+.2f}',
                    ha='center', va='center', fontsize=FS_ANNOT, color=col)
    for start, size, cid in (blocks or ()):
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), size, size,
                               fill=False, edgecolor='#111111', linewidth=7.0,
                               zorder=5))
        # The block's top-left cell is always ON the diagonal, which is masked
        # and unannotated, so the community label can be set large there
        # without covering a single number.
        ax.text(start, start, f'C{cid}', fontsize=FS_TICK + 8,
                fontweight='bold', color='#111111', ha='center', va='center',
                zorder=6)
    ax.set_xticks(range(n_c))
    ax.set_xticklabels([names.get(c, c) for c in cols], rotation=35,
                       ha='right', fontsize=FS_TICK)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels([names.get(r, r) for r in rows], fontsize=FS_TICK)
    ax.set_xlabel(xlabel, fontsize=FS_AXLABEL)
    ax.set_ylabel(ylabel, fontsize=FS_AXLABEL)
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(cbar_label, fontsize=FS_AXLABEL)
    cb.ax.tick_params(labelsize=FS_CBAR)
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
    """
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


def vis_city_mobility_curves(curves, save_path=None, ncols=5, names=None):
    """Every unit's CITY-LEVEL mobility curve on one page, x aligned on landfall.

    `curves` is {code -> dict(rel, day0, d_nor, i_min)} where `rel` is the
    day-type-normalized daily activity (1.0 = a normal day of the same type),
    `day0` the landfall day index, `d_nor` the index where the normal period
    ends, `i_min` the index of the lowest day of the buffer+disaster stretch.
    `names` maps code -> the full title shown on the panel (defaults to code).

    Per panel: the curve, a reference line at 1.0, a dashed line at landfall, a
    dotted line where the normal period ends, and a marker on the minimum.  Both
    axes are SHARED across panels, so they are labelled ONCE for the whole figure
    (one x label under the grid, one y label on its left) rather than per panel.
    A single figure-level legend explains the marks.  Read it for the SHAPE of
    the recovery, not the depth.  Slide-oriented: wide (5-column) layout, large
    fonts, no figure title, no per-panel text annotations.
    """
    import math
    codes = list(curves)
    names = names or {}
    nrows = math.ceil(len(codes) / ncols)
    rc = {'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
          'svg.fonttype': 'none', 'pdf.fonttype': 42,
          'font.size': 26, 'axes.titlesize': 29, 'axes.labelsize': 28,
          'xtick.labelsize': 22, 'ytick.labelsize': 22,
          'axes.spines.right': False, 'axes.spines.top': False,
          'axes.linewidth': 1.3}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.7 * nrows),
                                 squeeze=False, sharex=True, sharey=True)
        handles = None
        for i, code in enumerate(codes):
            ax = axes[i // ncols][i % ncols]
            c = curves[code]
            rel = np.asarray(c['rel'], dtype=float)
            x = np.arange(len(rel)) - c['day0']
            h_base = ax.axhline(1.0, color='#CCCCCC', lw=1.1,
                                label='normal baseline (1.0)')
            h_lf = ax.axvline(0, color='#B64342', lw=1.8, ls='--',
                              label='landfall')
            h_ne = ax.axvline(c['d_nor'] - c['day0'], color='#B0B0B0', lw=1.1,
                              ls=':', label='normal period ends')
            h_cur, = ax.plot(x, rel, color='#0F4D92', lw=2.4, marker='o',
                             ms=4.0, label='city mobility')
            im = int(c['i_min'])
            h_min, = ax.plot(x[im], rel[im], marker='v', ms=15, color='#111111',
                             zorder=5, ls='', label='minimum')
            ax.set_title(names.get(code, code))
            if handles is None:
                handles = [h_cur, h_min, h_lf, h_ne, h_base]
        for j in range(len(codes), nrows * ncols):
            axes[j // ncols][j % ncols].axis('off')
        # ONE shared axis label for the whole grid, not one per panel.
        fig.supxlabel('days relative to landfall', fontsize=28)
        fig.supylabel('daily activity / normal baseline', fontsize=28)
        # Figure-level legend in the free corner (bottom-right of the last row).
        legax = axes[nrows - 1][ncols - 1]
        if len(codes) < nrows * ncols:
            legax.legend(handles=handles, loc='center', frameon=False,
                         fontsize=25, handlelength=2.4, labelspacing=1.0)
        else:
            fig.legend(handles=handles, loc='lower center', ncol=len(handles),
                       frameon=False, fontsize=25, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_mapping_pca(points, arrows, evr, save_path=None):
    """The rank channel's mapping directions on one panel, no clustering.

    `points`: one row per component -- shade (colormap fraction; light =
    lower cum_loss rank within its own city), pc1, pc2.  Every city's cloud
    is centred on the origin by construction: the rank-z input removes each
    city's mean, so the panel carries direction information only.

    `arrows`: kind ('city' | 'pooled'), name, dx, dy (unit in-plane
    direction), cluster (0 = unclustered).  City arrows draw dashed with an
    edge label, coloured by their DISPLAY transfer community (the pair
    heatmap's Louvain partition -- presentation only); the pooled arrow
    draws solid black, slightly heavier, unlabelled.  Components are one
    uniform grey: the panel is about directions, and shading the points by
    loss rank pulled attention the arrows should have.  All arrows are drawn
    at one fixed length -- the projection loses the out-of-plane magnitude,
    so a drawn length would not be meaningful."""
    from matplotlib.lines import Line2D
    rc = {'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans',
                              'sans-serif'],
          'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 13}
    CL_COLORS = {0: '#57068C', 1: '#0F4D92', 2: '#4C9F70', 3: '#B64342',
                 4: '#E28E2C', 5: '#7B5EA7'}
    POOL_COL, PT_COL = '#111111', '#B8B8B8'
    rad = 0.42 * float(np.abs(points[['pc1', 'pc2']].to_numpy()).max())
    lim = 1.08 * float(np.abs(points[['pc1', 'pc2']].to_numpy()).max())

    with matplotlib.rc_context(rc):
        fig, ax = plt.subplots(figsize=(9.2, 8.6))
        ax.axhline(0, color='#DDDDDD', lw=0.8, zorder=1)
        ax.axvline(0, color='#DDDDDD', lw=0.8, zorder=1)
        ax.scatter(points['pc1'], points['pc2'], s=130, color=PT_COL,
                   edgecolor='#555555', linewidth=0.6, zorder=4)
        cl_col = (arrows['cluster'] if 'cluster' in arrows.columns
                  else pd.Series(0, index=arrows.index))
        for (_, a), k in zip(arrows.iterrows(), cl_col):
            solid = a['kind'] == 'pooled'
            col = POOL_COL if solid else CL_COLORS[int(k) % len(CL_COLORS)]
            ax.annotate('', xy=(a['dx'] * rad, a['dy'] * rad), xytext=(0, 0),
                        zorder=7 if solid else 6,
                        arrowprops=dict(arrowstyle='-|>',
                                        lw=3.2 if solid else 2.2, color=col,
                                        linestyle='-' if solid else '--'))
        cd = arrows[arrows['kind'] == 'city']
        tip = cd[['dx', 'dy']].to_numpy(float) * rad
        unit = tip / np.linalg.norm(tip, axis=1, keepdims=True)
        anc = _declutter_points(unit * (lim * 0.74), 0.62)
        cdk = cl_col[cd.index]
        for nm, tp, an, k in zip(cd['name'], tip, anc, cdk):
            col = CL_COLORS[int(k) % len(CL_COLORS)]
            ax.plot([tp[0], an[0]], [tp[1], an[1]], color=col, lw=0.7,
                    alpha=0.6, zorder=6)
            ax.annotate(nm, an, fontsize=13, ha='center', va='center',
                        color=col, zorder=8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"All {points['code'].nunique()} city-events, "
                     f"{len(points)} components", fontsize=18, pad=8)
        ax.set_xlabel(f'PC1 of within-city standardized features  '
                      f'({evr[0]:.0%})',
                      fontsize=15)
        ax.set_ylabel(f'PC2  ({evr[1]:.0%})', fontsize=15)
        ax.tick_params(labelsize=13)
        leg = [Line2D([0], [0], marker='o', ls='', mfc=PT_COL,
                      mec='#555555', ms=11, label='component'),
               Line2D([0], [0], color=POOL_COL, lw=3.2,
                      label='solid black = pooled mapping direction '
                            '(all cities)')]
        for k in sorted(set(int(x) for x in cl_col
                            [arrows['kind'] == 'city'])):
            leg.append(Line2D([0], [0],
                              color=CL_COLORS[k % len(CL_COLORS)], lw=2.2,
                              ls='--',
                              label=("one city's mapping direction"
                                     if k == 0 else
                                     f"city arrow, heatmap community C{k} "
                                     f"(display partition)")))
        ax.legend(handles=leg, loc='lower left', fontsize=12, framealpha=0.9)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)),
                        exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
    return fig

def _cf_graph_panel(ax, p, pos_color, neg_color, node_color, fdr_q,
                    fs_label):
    """One significance-graph panel: distance carries |rho|, edge COLOUR the
    sign, edge STYLE the FDR verdict.  Edge width is constant so magnitude is
    read once, from the geometry, and not twice."""
    C, Q, labels = p['corr'], p['qval'], p['labels']
    # A near-perfect correlation puts two nodes on top of each other
    # (rho = +0.89 -> d = 0.11, smaller than the marker).  Separate them by
    # one marker width so both stay readable; display-only.
    pos = _declutter_points(p['pos'], 0.14)
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            r = C[i, j]
            if not np.isfinite(r):
                # Undefined correlation (a constant function column in this
                # selection).  Drawn neutral: colouring it would present
                # "no data" as a measured relationship.
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color='#CCCCCC', ls=(0, (1, 2)), lw=1.4, alpha=0.6,
                        zorder=1)
                continue
            sig = np.isfinite(Q[i, j]) and Q[i, j] < fdr_q
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color=pos_color if r > 0 else neg_color,
                    ls='-' if sig else (0, (2.6, 2.2)), lw=2.2,
                    alpha=0.92 if sig else 0.45, zorder=2 if sig else 1)
    ax.scatter(pos[:, 0], pos[:, 1], s=260, color='white',
               edgecolor=node_color, linewidth=2.8, zorder=4)
    _cf_labels(ax, pos, _place_labels(pos, 0.34), labels,
               [node_color] * n, fs_label, zorder=5)


def _cf_labels(ax, pos, tips, labels, colors, fs_label, zorder):
    """Leader line plus label for every node, one shared implementation."""
    for i, lab in enumerate(labels):
        v = tips[i] - pos[i]
        v = v / (np.linalg.norm(v) + 1e-12)
        ax.plot([pos[i, 0], tips[i, 0]], [pos[i, 1], tips[i, 1]],
                color='#9A9A9A', lw=0.8, alpha=0.8, zorder=zorder)
        ax.annotate(lab, tips[i],
                    xytext=(5 * np.sign(v[0]), 5 * np.sign(v[1])),
                    textcoords='offset points',
                    ha='left' if v[0] > 0.15 else
                       'right' if v[0] < -0.15 else 'center',
                    va='bottom' if v[1] > 0.15 else
                       'top' if v[1] < -0.15 else 'center',
                    fontsize=fs_label, color=colors[i], zorder=zorder + 1)


def _declutter_points(pts, min_sep, rounds=60):
    """Push coincident points apart by the smallest amount that makes them
    separately visible.  Two city-events can land on identical coordinates (a
    5-component correlation matrix has few attainable values) and a hidden
    point reads as a missing city.  Deterministic: pairs resolve in index
    order, exact ties break along +x, so reruns give the same picture."""
    P = np.asarray(pts, float).copy()
    for _ in range(rounds):
        moved = False
        for a in range(len(P)):
            for b in range(a + 1, len(P)):
                d = P[b] - P[a]
                r = float(np.hypot(*d))
                if r >= min_sep:
                    continue
                if r < 1e-9:                      # exactly coincident
                    u, r = np.array([1.0, 0.0]), 0.0   # fixed tie-break axis
                else:
                    u = d / r
                shift = (min_sep - r) / 2.0 * u
                if not np.any(shift):
                    continue                      # already apart: don't spin
                P[a] -= shift
                P[b] += shift
                moved = True
        if not moved:
            break
    return P


def _place_labels(pos, offset, avoid=None, n_dir=24):
    """Least-crowded label direction per node.  A node near the layout
    centroid has no meaningful radial direction and a purely radial rule
    would fling its label across the graph, so candidate tips are scored by
    proximity to the other nodes, to any `avoid` points (the overlay clouds —
    in that figure THEY are the crowded region, so omitting them would let a
    label be placed into the densest part of the panel) and to already-placed
    labels; outermost node first, mild outward preference."""
    cen = pos.mean(axis=0)
    angles = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    other = np.vstack([pos] + ([avoid] if avoid is not None and len(avoid)
                               else []))
    placed, tips = [], np.zeros_like(pos)
    for i in np.argsort(-np.linalg.norm(pos - cen, axis=1)):
        best, best_score = None, np.inf
        out = pos[i] - cen
        out = out / (np.linalg.norm(out) + 1e-12)
        for a in angles:
            u = np.array([np.cos(a), np.sin(a)])
            tip = pos[i] + offset * u
            dist = np.hypot(*(other - tip).T)
            dist[i] = np.inf                       # its own node
            score = float(np.sum(1.0 / (dist + 0.05)))
            score += sum(1.6 / (np.hypot(*(tip - t)) + 0.05) for t in placed)
            score -= 0.6 * float(out @ u)
            if score < best_score:
                best, best_score = tip, score
        tips[i] = best
        placed.append(best)
    return tips


def _cf_overlay_panel(ax, p, func_color, label_offset, min_sep, fs_label,
                      base_grey, node_grey, edge_grey):
    """One overlay panel: the group layout as a grey base map with every
    city-event's own layout drawn over it.  The base map carries neither sign
    nor significance -- here it is scenery, and what the reader judges is how
    far each city-event sits from it."""
    pos, cats, labels = p['pos'], p['cats'], p['labels']
    n = len(cats)
    for i in range(n):
        for j in range(i + 1, n):
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color=base_grey, ls='-', lw=1.8, alpha=0.9, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=260, color='white',
               edgecolor=node_grey, linewidth=2.6, zorder=3)
    # Declutter across the whole panel at once, then redraw each city's edges
    # from the nudged coordinates so its graph stays internally consistent.
    cloud = None
    if p['members']:
        stack = np.vstack([q for _c, q in p['members']])
        nudged = _declutter_points(stack, min_sep).reshape(
            len(p['members']), n, 2)
        cloud = nudged.reshape(-1, 2)
        for q in nudged:
            for i in range(n):
                for j in range(i + 1, n):
                    ax.plot([q[i, 0], q[j, 0]], [q[i, 1], q[j, 1]],
                            color=edge_grey, ls='-', lw=0.7, alpha=0.55,
                            zorder=4)
            ax.scatter(q[:, 0], q[:, 1], s=42,
                       color=[func_color[c] for c in cats], alpha=0.85,
                       edgecolor='none', zorder=5)
    _cf_labels(ax, pos, _place_labels(pos, label_offset, avoid=cloud), labels,
               [func_color[c] for c in cats], fs_label, zorder=6)


def vis_cluster_function_graph(panels, over_panels, func_color,
                               save_path=None, pos_color='#B64342',
                               neg_color='#0F4D92', node_color='#272727',
                               fdr_q=0.05, label_offset=0.30, min_sep=0.07):
    """Function co-riding as a distance layout, TWO ROWS per group.

    Top row    the significance graph: distance carries |rho|, edge colour the
               sign, edge style the FDR verdict.
    Bottom row the same layout as a grey base map with every city-event's own
               layout over it, coloured by FUNCTION -- how much the cities in
               that group disagree about where each function belongs.

    `panels` and `over_panels` are the two panel lists the separate figures
    used, in the SAME group order; column i is one group in both rows, which
    is the whole point of stacking them.  Group titles are therefore written
    ONCE, on the top row, and one legend at the foot carries both rows'
    encodings.

    Coordinates are supplied by the caller because every panel must share one
    frame: the per-group layouts are solved from the pooled one and rigidly
    aligned back onto it, which is a fitting decision, not a drawing decision.

    Publication typography (Arial-first stack, editable vector text) is set
    locally rather than globally so the rest of the pipeline's figures keep
    their own defaults.  Fixed margins replace a tight bbox: tight cropping
    keys off the drawn content, which sits asymmetrically in the shared
    window, and would pull the figure-centred legend off-centre."""
    from matplotlib.lines import Line2D
    FS_LABEL, FS_TITLE, FS_LEGEND = 17, 23, 18
    rc = {'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans',
                              'sans-serif'],
          'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': FS_LABEL}
    base_grey, node_grey, edge_grey = '#B8B8B8', '#8A8A8A', '#D0D0D0'
    with matplotlib.rc_context(rc):
        # ONE window for both rows, so a node that moved between the two
        # readings moved on the page too.
        allpos = np.vstack([p['pos'] for p in panels]
                           + [p['pos'] for p in over_panels]
                           + [q for p in over_panels
                              for _c, q in p['members']])
        x0, x1 = allpos[:, 0].min() - 0.75, allpos[:, 0].max() + 0.75
        y0, y1 = allpos[:, 1].min() - 0.45, allpos[:, 1].max() + 0.60

        n = len(panels)
        fig, axes = plt.subplots(2, n, figsize=(4.7 * n, 10.6), squeeze=False,
                                 gridspec_kw={'wspace': 0.02, 'hspace': 0.04})
        for k in range(n):
            _cf_graph_panel(axes[0][k], panels[k], pos_color, neg_color,
                            node_color, fdr_q, FS_LABEL)
            _cf_overlay_panel(axes[1][k], over_panels[k], func_color,
                              label_offset, min_sep, FS_LABEL, base_grey,
                              node_grey, edge_grey)
            # The group title belongs to the COLUMN, not to either panel, so
            # it is written once above the pair.
            axes[0][k].set_title(panels[k]['title'], fontsize=FS_TITLE,
                                 linespacing=1.45, pad=12)
            for r in (0, 1):
                axes[r][k].set_xlim(x0, x1)
                axes[r][k].set_ylim(y0, y1)
                axes[r][k].set_aspect('equal')
                axes[r][k].set_axis_off()

        fig.legend(handles=[
            Line2D([0], [0], color=pos_color, lw=2.6, label='Positive ρ'),
            Line2D([0], [0], color=neg_color, lw=2.6, label='Negative ρ'),
            Line2D([0], [0], color='#767676', lw=2.6,
                   label=f'Significant (q < {fdr_q})'),
            Line2D([0], [0], color='#767676', lw=2.6, ls=(0, (2.6, 2.2)),
                   label='Not Significant'),
            Line2D([0], [0], color=base_grey, lw=2.2,
                   label='Group Layout (Base Map)'),
            Line2D([0], [0], color=edge_grey, lw=1.4,
                   label='One City-Event Layout'),
            Line2D([0], [0], marker='o', ls='', mfc='#767676', mec='none',
                   ms=9,
                   label="Node = That City-Event's Position "
                         "(Colour = Function)")],
            loc='lower center', ncol=4, fontsize=FS_LEGEND, frameon=False,
            handlelength=2.6, columnspacing=2.6, handletextpad=0.8,
            labelspacing=0.7, bbox_to_anchor=(0.5, 0.008))
        fig.subplots_adjust(left=0.008, right=0.992, top=0.93, bottom=0.10)
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)),
                        exist_ok=True)
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
    return fig


def vis_cluster_function_heatmap(panels, save_path=None, vmax=1.0):
    """The co-riding structure as HEATMAPS: the pooled average and each
    cluster, all in ABSOLUTE (CLR) Spearman rho on one shared scale.

    `panels` is a list of dicts with title, labels (already in this panel's
    display order), mat (the 6x6 correlation matrix, same order) and blocks
    (the Louvain function communities as (start, size, id) triples).

    The pooled average sits ALONE on the first row and the clusters share the
    second: the average is the reference the clusters are read against, not a
    fourth peer, and one row of four hid that.

    Every panel is ordered by ITS OWN Louvain communities: a cluster whose
    functions group differently should show that in its row order, and
    forcing one shared order would hide exactly the difference the figure is
    about.  The price is that cells are not vertically aligned between
    panels; the numbers are printed in each cell for that reason.  The
    diagonal is masked — a self-correlation is 1 by definition."""
    cats_n = len(panels[0]['labels'])
    FS_ANNOT, FS_TICK, FS_TITLE, FS_CBAR = 13, 16, 20, 16
    rc = {'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
          'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': FS_TICK}
    with matplotlib.rc_context(rc):
        cmap = deck_diverging_cmap()
        cmap.set_bad('#ECECEC')
        # Two rows on a 2*n_cl column grid: one panel is two columns wide, so
        # the average lands centred on row 1 whatever the cluster count is.
        n_cl = len(panels) - 1
        # hspace has to clear the 45-degree tick labels hanging BELOW the top
        # panel plus the two-line title sitting ABOVE the bottom row; at 0.42
        # the average's ticks landed on the 'Cluster 2' title.
        fig = plt.figure(figsize=(6.1 * n_cl + 1.2, 12.6))
        # wspace clears each panel's y tick labels, which otherwise reach into
        # the panel on its left; hspace clears the top panel's 45-degree x
        # labels plus the two-line title under them.
        gs = fig.add_gridspec(2, 2 * n_cl, wspace=1.15, hspace=0.50,
                              bottom=0.15, top=0.95)
        axes_all = [fig.add_subplot(gs[0, n_cl - 1:n_cl + 1])]
        axes_all += [fig.add_subplot(gs[1, 2 * i:2 * i + 2])
                     for i in range(n_cl)]
        im = None
        for ax, p in zip(axes_all, panels):
            M, labels = np.asarray(p['mat'], float), p['labels']
            Mm = M.copy()
            np.fill_diagonal(Mm, np.nan)
            im = ax.imshow(Mm, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(cats_n))
            ax.set_xticklabels(labels, rotation=45, ha='right',
                               fontsize=FS_TICK)
            ax.set_yticks(range(cats_n))
            ax.set_yticklabels(labels, fontsize=FS_TICK)
            for i in range(cats_n):
                for j in range(cats_n):
                    if i == j or not np.isfinite(M[i, j]):
                        continue
                    ax.text(j, i, f'{M[i, j]:+.2f}', ha='center', va='center',
                            fontsize=FS_ANNOT,
                            color='white' if abs(M[i, j]) > 0.58 * vmax
                                  else '#222222')
            for start, size, _cid in p.get('blocks', []):
                ax.add_patch(_Rectangle((start - 0.5, start - 0.5), size, size,
                                        fill=False, edgecolor='#1A1A1A',
                                        lw=2.6))
            ax.set_title(p['title'], fontsize=FS_TITLE, linespacing=1.4,
                         pad=12)
        # pad clears the 45-degree tick labels below the bottom row; without it
        # the colour bar rides up over the 'Commercial'/'Residential' ticks.
        # Anchored to the figure, not to the axes: an axes-anchored bar rides
        # up into the bottom row's tick labels once the panels are square.
        cax = fig.add_axes([0.32, 0.045, 0.36, 0.017])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.set_label('Spearman ρ between function shares (pooled, CLR)',
                     fontsize=19)
        cb.ax.tick_params(labelsize=FS_CBAR)
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)),
                        exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
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

# Slide-oriented twin of _PUB_RC: the publication sizes (7 pt) are unreadable
# once a 13-panel grid is projected, so the deck figures use these instead.
# One style per METHOD, keyed by the label the pipeline prints, so a method
# looks identical in the component grid and on the city page.  Hue and stroke
# both separate the lines, so neither has to carry it alone, and the forecast
# (blue) never sits next to a similar green baseline again.
# One style per METHOD, keyed by the label the pipeline prints, so a method
# looks identical in the component grid and on the city curve page.  Hue and
# stroke both separate the lines, so neither has to carry it alone.
# These are the CURVE colours.  The MAE bar keeps its OWN set below: a filled
# bar and a 2pt line have different legibility floors -- the recessive grey the
# bar gives the do-nothing baseline all but vanishes as a dashed line here
# (measured 2026-09-03), so the two figures cannot share one table.
_CURVE_STYLE = {
    'ground truth':               ('#3a3a3a', '-'),
    'observed':                   ('#3a3a3a', '-'),
    'proposed pipeline':          ('#E28E2C', '--'),
    'train-mean baseline':        ('#0F4D92', (0, (1, 1.6))),
    'oracle':                     ('#7B5EA7', '-'),
    'city-wise prediction (kNN)': ('#4C9F70', (0, (5, 1.5, 1, 1.5))),
    # A deeper violet than the oracle's muted #7B5EA7 on purpose: the two
    # purples sit in different figures of the same deck and must not read as
    # one method.
    'naive ridge regression':     ('#7030A0', (0, (6, 1.6, 1.4, 1.6))),
}

# Bar-chart colours.  The methods are ORDERED (do nothing -> naive regression
# -> the two real forecasts) and as solid fills there is room to say so: the
# proposed pipeline takes the only warm saturated hue so the eye lands on it,
# and the train-mean baseline takes a neutral grey because it predicts nothing
# and looking inert is the honest reading.  Orange/teal/violet/grey also
# survives deuteranopia, where the orange/green pair this figure used to run on
# does not.
_BAR_METHOD_COLORS = {
    'proposed pipeline':          '#D95F02',
    'city-wise prediction (kNN)': '#1B9E77',
    'naive ridge regression':     '#7030A0',
    'train-mean baseline':        '#6E6E6E',
    'oracle':                     '#7B5EA7',
}
_STYLE_FALLBACK = [('#B0413E', '--'), ('#4C9F70', '-.'), ('#7B5EA7', ':')]

# Reading order for the per-panel MAE stack: increasing sophistication downwards
# (do nothing, then the naive regression, then the pipeline), so every number is
# read against the one above it.  Unlisted labels fall in after these, in the
# order they were plotted.
_MAE_TEXT_ORDER = ['train-mean baseline', 'city-wise prediction (kNN)',
                   'naive ridge regression', 'proposed pipeline', 'oracle']


def _mae_text_row(label):
    return (_MAE_TEXT_ORDER.index(label) if label in _MAE_TEXT_ORDER
            else len(_MAE_TEXT_ORDER))


def _curve_style(label, i):
    return _CURVE_STYLE.get(label, _STYLE_FALLBACK[i % len(_STYLE_FALLBACK)])


_SLIDE_RC = dict(_PUB_RC, **{
    'font.size': 20, 'axes.titlesize': 22, 'axes.labelsize': 24,
    'xtick.labelsize': 17, 'ytick.labelsize': 17, 'axes.linewidth': 1.3,
})


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


def vis_rank_pred_vs_true(params, save_path=None, ncols=5, names=None,
                          obs_col='cum_loss', footnote=None):
    """Predicted vs observed cum_loss RANK, one panel per city-event.

    The rank channel is the only part of the forecast that transfers, so this
    is the figure that shows whether it does: both axes are ranks WITHIN the
    unit, the diagonal is a perfect ordering, and each panel carries its own
    Spearman.  `params` needs code, rank_score and `obs_col` (the observed
    quantity whose ordering is being predicted).

    `footnote` (optional) is one line set below the shared x label -- the
    sweep-level number no single panel can carry.  It goes BELOW the figure
    rectangle, because the layout parks the shared x label at the bottom of
    that rectangle and anything reserved inside it lands on top of the label.
    """
    codes = list(params['code'].drop_duplicates())
    names = names or {}
    with plt.rc_context(_SLIDE_RC):
        nrow = int(np.ceil(len(codes) / ncols))
        fig, axes = plt.subplots(nrow, ncols, figsize=(3.7 * ncols, 3.8 * nrow),
                                 squeeze=False)
        for i, c in enumerate(codes):
            ax = axes[i // ncols][i % ncols]
            s = params[params.code == c].dropna(subset=['rank_score', obs_col])
            if len(s) < 2:
                ax.axis('off')
                continue
            rt, rp = rankdata(s[obs_col]), rankdata(s['rank_score'])
            n = len(s)
            ax.plot([0.5, n + 0.5], [0.5, n + 0.5], color=_PUB_GREY, lw=1.6,
                    zorder=1)
            ax.scatter(rt, rp, s=150, color=_PUB_BLUE, alpha=0.9, zorder=3,
                       edgecolor='white', linewidth=1.3)
            ax.set_title(names.get(c, c))
            rho = spearmanr(rt, rp).statistic
            ax.text(0.05, 0.95, '$\\rho_s$ = {:+.2f}'.format(rho),
                    transform=ax.transAxes, fontsize=19, va='top')
            ticks = [t for t in (1, 5, 10, 15) if t <= n]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xlim(0.4, n + 0.6)
            ax.set_ylim(0.4, n + 0.6)
            ax.set_aspect('equal')
        for k in range(len(codes), nrow * ncols):
            axes[k // ncols][k % ncols].axis('off')
        fig.supxlabel('observed cumulative-loss rank', fontsize=26)
        fig.supylabel('predicted rank', fontsize=26)
        fig.tight_layout()
        if footnote:
            fig.text(0.5, -0.022 * (11.0 / fig.get_size_inches()[1]), footnote,
                     ha='center', va='top', fontsize=22, color='#333333')
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


def vis_centered_spectrum_loo(pred, metrics, save_path=None, names=None,
                              ncol=5,
                              xlabel='centred component cumulative loss'):
    """Leave-one-out grid: does the predicted centred loss spectrum match the
    held-out unit's own?

    `pred` is {code: {'bary': the predicted spectrum as a dense sample, 'true':
    the held unit's centred component cum_loss}} and `metrics` a frame indexed by
    code with n / W1 / skill.  `names` maps code -> the full city-event title
    (defaults to the code).  One panel per unit; both distributions are drawn as
    CUMULATIVE curves because the area between two CDFs IS the 1-Wasserstein
    distance, so the shaded region is the reported W1 rather than a decoration of
    it.  Ticks mark the unit's actual components -- with n = 5..12 the truth is a
    short staircase, and showing it as such keeps the reader from reading a smooth
    fit into the data.  The n in each panel title is that COMPONENT COUNT, and W1
    is in day-equivalents, the unit the shared x label carries.

    Both axes are SHARED across panels, so they are labelled ONCE for the whole
    figure rather than per panel; tick labels are restored on the lowest occupied
    panel of every column, so a short last row never leaves a column reading
    another column's ticks.  One palette across all panels (prediction / truth /
    area), so a colour carries the same meaning everywhere.  Slide-oriented: wide
    grid, large fonts, no figure suptitle."""
    codes = list(pred)
    names = names or {}
    c_pred, c_true, c_area = '#0F4D92', '#B64342', '#C9A227'
    lo = min(min(v['true'].min(), np.min(v['bary'])) for v in pred.values()) - 1
    hi = max(max(v['true'].max(), np.max(v['bary'])) for v in pred.values()) + 1
    xs = np.linspace(lo, hi, 900)
    rc = {'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
          'font.size': 25, 'axes.labelsize': 30, 'axes.titlesize': 22,
          'xtick.labelsize': 22, 'ytick.labelsize': 22, 'axes.linewidth': 1.3,
          'axes.spines.top': False, 'axes.spines.right': False,
          'svg.fonttype': 'none', 'pdf.fonttype': 42}
    nrow = int(np.ceil(len(codes) / ncol))
    with plt.rc_context(rc):
        fig, axes2d = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 5.0 * nrow),
                                   squeeze=False, sharex=True, sharey=True)
        axes = axes2d.ravel()
        for ax, code in zip(axes, codes):
            d = pred[code]
            b = np.sort(np.asarray(d['bary'], dtype=float))
            t = np.sort(np.asarray(d['true'], dtype=float))
            fp = np.searchsorted(b, xs, 'right') / len(b)
            ft = np.searchsorted(t, xs, 'right') / len(t)
            ax.fill_between(xs, fp, ft, color=c_area, alpha=0.42, zorder=1,
                            label='area between = $W_1$')
            ax.plot(xs, fp, color=c_pred, lw=2.8, zorder=3,
                    label='predicted')
            ax.step(np.concatenate([[lo], t, [hi]]),
                    np.concatenate([[0.0], (np.arange(len(t)) + 1) / len(t), [1.0]]),
                    where='post', color=c_true, lw=2.8, zorder=4,
                    label='true')
            ax.plot(t, np.zeros(len(t)), '|', color=c_true, ms=14, mew=2.0,
                    zorder=5, label='its components')
            ax.set_title(
                '{}'.format(names.get(code, code)) + chr(10)
                + '{} components'.format(int(metrics.loc[code, 'n'])) + chr(10)
                + '$W_1$ {:.2f}   skill {:+.2f}'.format(
                    metrics.loc[code, 'W1'], metrics.loc[code, 'skill']),
                linespacing=1.35)
            ax.set_xlim(lo, hi)
            ax.set_ylim(-0.03, 1.03)
        for ax in axes[len(codes):]:
            ax.axis('off')
        # sharex hides tick labels above the bottom ROW; the last row is short,
        # so give every column its ticks back on ITS lowest occupied panel.
        for col in range(ncol):
            occ = [r for r in range(nrow) if r * ncol + col < len(codes)]
            if occ:
                axes2d[occ[-1]][col].xaxis.set_tick_params(labelbottom=True)
        h, l = axes[0].get_legend_handles_labels()
        if len(axes) > len(codes):
            axes[len(codes)].legend(h, l, loc='center', fontsize=24,
                                    frameon=False, handlelength=1.6)
        fig.supxlabel(xlabel, fontsize=30)
        fig.supylabel('cumulative probability', fontsize=30)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_centered_distributions(cen, save_path=None, names=None, storms=None,
                               ncol=5,
                               xlabel='component cumulative loss centred on '
                                      'its own city-event mean'):
    """The OBSERVED centred loss spectra: what the STEP-7 shape channel has to
    transfer, before any prediction enters.

    `cen` is {code: centred component cum_loss array}, `names` maps code -> the
    full city-event title, `storms` code -> a colour.  A pooled panel on top
    (every component of every unit on one cumulative curve, each unit's own
    curve behind it in grey) sets the reference; the grid below gives each unit
    its own panel against that pooled reference as a dashed line, so a unit
    reads as wider or narrower than the pool at a glance.

    Cumulative curves, not histograms: with n = 5..12 components a unit's
    spectrum is a short staircase, and the CDF shows every component as a step
    instead of hiding them in bin edges.  The n in each panel title is that
    COMPONENT COUNT.  Both axes are shared across the grid and therefore
    labelled ONCE for the whole figure; tick labels are restored on the lowest
    occupied panel of each column so a short last row leaves no column without
    ticks.  Slide-oriented: wide grid, large fonts, no figure suptitle."""
    codes = list(cen)
    names = names or {}
    storms = storms or {}
    pooled = np.sort(np.concatenate([np.asarray(cen[c], dtype=float)
                                     for c in codes]))
    lo = float(pooled.min()) - 1.5
    hi = float(pooled.max()) + 1.5
    xs = np.linspace(lo, hi, 900)
    f_pool = np.searchsorted(pooled, xs, 'right') / len(pooled)
    nrow = int(np.ceil(len(codes) / ncol))
    rc = {'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
          'font.size': 31, 'axes.labelsize': 37, 'axes.titlesize': 29,
          'xtick.labelsize': 27, 'ytick.labelsize': 27, 'axes.linewidth': 1.5,
          'axes.spines.top': False, 'axes.spines.right': False,
          'svg.fonttype': 'none', 'pdf.fonttype': 42}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(5.0 * ncol, 4.6 * (nrow + 1)))
        gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1.2] + [1] * nrow,
                              hspace=0.52, wspace=0.16,
                              left=0.075, right=0.995,
                              top=0.985, bottom=0.095)
        # pooled reference panel
        axp = fig.add_subplot(gs[0, :])
        for c in codes:
            v = np.sort(np.asarray(cen[c], dtype=float))
            axp.step(np.concatenate([[lo], v, [hi]]),
                     np.concatenate([[0.0], (np.arange(len(v)) + 1) / len(v),
                                     [1.0]]),
                     where='post', color='#BBBBBB', lw=1.4, zorder=2,
                     label='each city-event on its own' if c == codes[0] else None)
        axp.plot(xs, f_pool, color='#111111', lw=3.2, zorder=3,
                 label='pooled, all {} city-events (n = {})'.format(
                     len(codes), len(pooled)))
        axp.plot(pooled, np.zeros(len(pooled)), '|', color='#555555', ms=13,
                 mew=1.6, zorder=4, label='the {} components'.format(len(pooled)))
        axp.axvline(0, color='#DDDDDD', lw=1.2, zorder=0)
        axp.set_xlim(lo, hi)
        axp.set_ylim(-0.03, 1.03)
        axp.legend(loc='upper left', fontsize=29, frameon=False,
                   handlelength=1.6)

        axes = []
        for i, code in enumerate(codes):
            ax = fig.add_subplot(gs[1 + i // ncol, i % ncol],
                                 sharex=axes[0] if axes else None,
                                 sharey=axes[0] if axes else None)
            axes.append(ax)
            v = np.sort(np.asarray(cen[code], dtype=float))
            ax.plot(xs, f_pool, color='#AAAAAA', lw=2.0, ls='--', zorder=2,
                    label='pooled reference')
            ax.step(np.concatenate([[lo], v, [hi]]),
                    np.concatenate([[0.0], (np.arange(len(v)) + 1) / len(v),
                                    [1.0]]),
                    where='post', color=storms.get(code, '#0F4D92'), lw=3.0,
                    zorder=3, label='this city-event')
            ax.plot(v, np.zeros(len(v)), '|', color=storms.get(code, '#0F4D92'),
                    ms=14, mew=2.0, zorder=4, label='its components')
            ax.axvline(0, color='#DDDDDD', lw=1.2, zorder=0)
            ax.set_title('{}'.format(names.get(code, code)) + chr(10)
                         + '{} components'.format(len(v)), linespacing=1.35)
            ax.set_xlim(lo, hi)
            ax.set_ylim(-0.03, 1.03)
            if i % ncol:
                for lab in ax.get_yticklabels():
                    lab.set_visible(False)
        # every column keeps ticks on ITS lowest occupied panel
        for col in range(ncol):
            occ = [r for r in range(nrow) if r * ncol + col < len(codes)]
            if occ:
                keep = occ[-1] * ncol + col
                for i, ax in enumerate(axes):
                    if i % ncol == col and i != keep:
                        for lab in ax.get_xticklabels():
                            lab.set_visible(False)
        free = len(codes) % ncol
        if free:
            axl = fig.add_subplot(gs[nrow, free:])
            axl.axis('off')
            h, l = axes[0].get_legend_handles_labels()
            axl.legend(h, l, loc='center', fontsize=30, frameon=False,
                       handlelength=1.6)
        # Anchor the two shared labels to the AXES block: supxlabel/supylabel
        # centre on the whole figure, which drifts once the last row is short
        # and a legend slot sits beside it.
        boxes = [a.get_position() for a in [axp] + axes]
        x_lo, x_hi = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
        y_lo, y_hi = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
        fig.text(0.5 * (x_lo + x_hi), y_lo - 0.046, xlabel,
                 ha='center', va='top', fontsize=37)
        fig.text(x_lo - 0.048, 0.5 * (y_lo + y_hi), 'cumulative probability',
                 rotation=90, ha='right', va='center', fontsize=37)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_spread_vs_predictors(target, predictors, save_path=None, storms=None,
                             ncol=5, dim_r2=0.15, gap_after=None,
                             title_fontsize=None,
                             xlabel='candidate predictor',
                             ylabel='within-city loss SD'):
    """Does any candidate predictor track the WIDTH the spread model has to
    predict?  One panel per candidate, all sharing the same y.

    `target` is {code: within-city SD of the centred component cumulative loss}
    -- the quantity ML-B2inc predicts as sigma, and the RESPONSE, so it takes
    the y axis.  `predictors` is an ordered {panel title: {code: value}} and
    supplies x, one candidate per panel, filled row-major; `xlabel` names what
    those candidates have in common, since the panel title already names the
    individual one.

    `gap_after` inserts a narrow blank column after that logical column, so one
    group of candidates can be set apart from the rest without needing a second
    figure or a box drawn around them.

    Each panel carries the least-squares line and its Pearson r with R^2,
    because the question is whether the cloud has usable slope at n = 13.
    Panels at or below `dim_r2` are kept -- a null result is evidence, hiding it
    would turn this into a highlight reel -- but their marks are drawn in grey so
    the eye lands on the candidates that carry signal without reading every R^2
    first.  The TITLE stays in normal ink either way: the dimming ranks the
    evidence, and a reader still has to be able to read what the null result was
    about.  `title_fontsize` overrides the rc size, for two-line titles.

    y is the same in every panel and is therefore labelled ONCE for the whole
    figure; x differs per panel in scale, so each panel keeps its own ticks
    under one shared summary label.  Read these as DESCRIPTIVE: computed on all
    units at once, they show association, not the held-out skill the LOO
    figures report.  Slide-oriented: wide grid, large fonts, no suptitle."""
    items = list(predictors.items())
    codes = list(target)
    nrow = int(np.ceil(len(items) / ncol))
    rc = {'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
          'font.size': 30, 'axes.labelsize': 40, 'axes.titlesize': 32,
          'xtick.labelsize': 26, 'ytick.labelsize': 26, 'axes.linewidth': 1.6,
          'axes.spines.top': False, 'axes.spines.right': False,
          'svg.fonttype': 'none', 'pdf.fonttype': 42}
    with plt.rc_context(rc):
        widths, col_map, g = [], {}, 0
        for j in range(ncol):
            widths.append(1.0)
            col_map[j] = g
            g += 1
            if gap_after is not None and j == gap_after:
                widths.append(0.20)          # blank spacer column
                g += 1
        fig = plt.figure(figsize=(5.4 * sum(widths), max(4.8 * nrow, 6.4)))
        gs = fig.add_gridspec(nrow, len(widths), width_ratios=widths)
        axes = []
        for i, (title, vals) in enumerate(items):
            rw, cl = divmod(i, ncol)
            ax = fig.add_subplot(gs[rw, col_map[cl]],
                                 sharey=axes[0] if axes else None)
            axes.append(ax)
            use = [k for k in codes
                   if np.isfinite(target[k])
                   and np.isfinite(vals.get(k, np.nan))]
            x = np.array([vals[k] for k in use], dtype=float)
            y = np.array([target[k] for k in use], dtype=float)
            rr = np.nan
            if len(use) >= 3 and x.std() > 1e-12 and y.std() > 1e-12:
                rr = float(np.corrcoef(x, y)[0, 1])
            dim = not np.isfinite(rr) or rr ** 2 <= dim_r2
            ax.scatter(x, y, s=230, color='#C8C8C8' if dim else '#39557A',
                       edgecolor='white', linewidth=1.5, zorder=3,
                       alpha=0.85 if dim else 1.0)
            if np.isfinite(rr):
                b1, b0 = np.polyfit(x, y, 1)
                xx = np.linspace(x.min(), x.max(), 50)
                ax.plot(xx, b0 + b1 * xx, lw=2.6, zorder=2,
                        color='#D9D9D9' if dim else '#B64342')
                ax.text(0.035, 0.975, 'r {:+.2f}'.format(rr) + chr(10)
                        + '$R^2$ {:.2f}'.format(rr ** 2),
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=27, linespacing=1.3, zorder=6,
                        color='#A8A8A8' if dim else '#1A1A1A',
                        bbox=dict(boxstyle='round,pad=0.22', fc='white',
                                  ec='none', alpha=0.78))
            ax.set_title(title, color='#1A1A1A', fontsize=title_fontsize)
            if cl:
                for lab in ax.get_yticklabels():
                    lab.set_visible(False)
            if dim:
                ax.tick_params(colors='#A8A8A8')
                for sp in ax.spines.values():
                    sp.set_color('#D0D0D0')
        fig.tight_layout()
        # Size the two shared labels to the PANEL BLOCK and anchor them there:
        # a one-row grid is only a few inches tall, so a fixed point size would
        # be physically longer than the figure and force empty bands.
        boxes = [a.get_position() for a in axes]
        x_lo, x_hi = min(v.x0 for v in boxes), max(v.x1 for v in boxes)
        y_lo, y_hi = min(v.y0 for v in boxes), max(v.y1 for v in boxes)
        W, H = fig.get_figwidth(), fig.get_figheight()
        span_x, span_y = (x_hi - x_lo) * W, (y_hi - y_lo) * H
        fs = float(np.clip(0.92 * span_x * 72 / (0.58 * max(len(xlabel), 1)),
                           24.0, 40.0))
        # The y label stays on ONE line always; if the block is too short to
        # carry it at the shared size it shrinks rather than wrapping.
        fs_y = min(fs, 0.94 * span_y * 72 / (0.58 * max(len(ylabel), 1)))
        fig.text(0.5 * (x_lo + x_hi), y_lo - 0.78 / H, xlabel,
                 ha='center', va='top', fontsize=fs)
        fig.text(x_lo - 0.98 / W, 0.5 * (y_lo + y_hi), ylabel,
                 rotation=90, ha='right', va='center', fontsize=fs_y)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_spread_concept(pooled, scale, save_path=None, units_per_inch=3.1,
                       height=5.2, font_scale=1.0,
                       xlabel='centred cumulative loss'):
    """What the SPREAD channel does, as a picture: the pooled spectrum rescaled.

    `pooled` is every unit's centred component cumulative loss in one array --
    the curve the top panel of the centred-distribution figure draws, with the
    per-unit background curves dropped.  Multiplying that sample by `scale`
    leaves its SHAPE untouched and changes only its WIDTH: exactly the one
    degree of freedom the model's sigma controls.

    ONE curve per figure, and the figure's own horizontal length carries the
    comparison.  `units_per_inch` is held FIXED across calls, so a widened
    spectrum yields a physically wider figure and a narrowed one a physically
    narrower figure; the pair read side by side shows the rescaling at true
    relative size.  Drawing the unscaled spectrum alongside would either clip
    it (the narrowed frame is too small to hold it) or force a common frame
    that destroys exactly the width cue this figure exists to give, so it is
    left out.  No legend.

    `font_scale` multiplies every text size.  A wider figure needs larger
    type to stay readable once it is scaled to fit a slide, so the caller
    raises it in step with the width rather than leaving one file's labels
    shrunk relative to the other's."""
    v = np.sort(np.asarray(pooled, dtype=float) * scale)
    lim = float(np.max(np.abs(v))) * 1.08
    xs = np.linspace(-lim, lim, 1200)
    f = np.searchsorted(v, xs, 'right') / len(v)
    rc = {'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
          'font.size': 24 * font_scale, 'axes.labelsize': 25 * font_scale,
          'xtick.labelsize': 21 * font_scale,
          'ytick.labelsize': 21 * font_scale,
          'axes.linewidth': 1.5 * min(font_scale, 2.0),
          'axes.spines.top': False, 'axes.spines.right': False,
          'svg.fonttype': 'none', 'pdf.fonttype': 42}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(2 * lim / units_per_inch, height))
        ax.plot(xs, f, color='#111111', lw=3.2 * min(font_scale, 2.0),
                zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('cumulative probability')
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_mlb2_pca(block, feature_names, save_path=None, names=None, ncomp=2):
    """The PCA step inside the spread model, opened up: what the components are
    made of.

    ML-B2inc does not use the 22 city-level features directly.  It z-scores them
    across the training units and keeps the leading `ncomp` principal
    components; only those enter the fitted increment as gamma'PC.  This figure
    shows the LOADINGS -- each component's weight on every raw feature -- on a
    symmetric diverging scale, so sign is readable and the components are
    directly comparable.  Each column header carries the share of feature
    variance that component accounts for.  Portrait layout: the 22 features
    run down the panel so each keeps a full horizontal line for its name.

    `block` is {code: the unit's raw feature vector} and `feature_names` labels
    its entries; `names` is accepted so callers can pass city-event titles, but
    the panel is about features, not units.

    DESCRIPTIVE: fitted here on ALL units at once, whereas production refits the
    PCA per fold on the training units, so the axes drift slightly from any
    single fold's.  Sign is arbitrary in any PCA -- a component and its negation
    are the same direction -- so read a column as a contrast pattern, not as
    absolute polarity.  Wide, large type, no figure title."""
    codes = list(block)
    E = np.vstack([np.asarray(block[c], dtype=float) for c in codes])
    mu, sd = E.mean(0), E.std(0)
    sd[sd == 0] = 1.0
    Z = (E - mu) / sd
    Z = Z - Z.mean(0)
    S, Vt = np.linalg.svd(Z, full_matrices=False)[1:]
    evr = (S ** 2) / float(np.sum(S ** 2))
    load = Vt[:ncomp]
    vmax = float(np.max(np.abs(load)))

    rc = {'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
          'font.size': 36, 'axes.labelsize': 40, 'axes.titlesize': 40,
          'xtick.labelsize': 40, 'ytick.labelsize': 36, 'axes.linewidth': 1.5,
          'svg.fonttype': 'none', 'pdf.fonttype': 42}
    with plt.rc_context(rc):
        # Portrait: the features run DOWN the panel and the components across,
        # so each of the 22 names gets a full horizontal line to itself.
        fig, ax = plt.subplots(
            figsize=(4.4 * ncomp + 9.0, 0.86 * len(feature_names) + 2.4))
        im = ax.imshow(load.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       aspect='auto')
        ax.set_xticks(range(ncomp))
        ax.set_xticklabels(['PC{}'.format(i + 1) + chr(10)
                            + '({:.0%})'.format(evr[i]) for i in range(ncomp)])
        ax.xaxis.set_ticks_position('top')
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label('loading', fontsize=38)
        cb.ax.tick_params(labelsize=32)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_centered_spectrum_schematic(bary, true, save_path=None,
                                    xlabel='centred cumulative loss'):
    """One unlabelled panel of the leave-one-out comparison, as a schematic.

    Same three marks as the full grid -- the predicted spectrum, the held-out
    unit's true staircase, and the shaded area between the two cumulative
    curves that IS the reported 1-Wasserstein distance -- but with every
    unit-specific cue removed: no city-event name, no n, no W1 or skill.  It
    exists to show WHAT the metric measures, so the caller supplies one fold's
    arrays purely as an example and the panel must not invite reading anything
    off that particular city.  Legend kept, since without it the three marks
    are unidentifiable.  Small canvas, large type, no title."""
    b = np.sort(np.asarray(bary, dtype=float))
    t = np.sort(np.asarray(true, dtype=float))
    lo = min(b.min(), t.min()) - 1.0
    hi = max(b.max(), t.max()) + 1.0
    xs = np.linspace(lo, hi, 900)
    fp = np.searchsorted(b, xs, 'right') / len(b)
    ft = np.searchsorted(t, xs, 'right') / len(t)
    c_pred, c_true, c_area = '#0F4D92', '#B64342', '#C9A227'
    rc = {'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
          'font.size': 22, 'axes.labelsize': 24,
          'xtick.labelsize': 19, 'ytick.labelsize': 19, 'axes.linewidth': 1.4,
          'axes.spines.top': False, 'axes.spines.right': False,
          'svg.fonttype': 'none', 'pdf.fonttype': 42}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(6.2, 6.6))
        ax.fill_between(xs, fp, ft, color=c_area, alpha=0.42, zorder=1,
                        label='area between = $W_1$')
        ax.plot(xs, fp, color=c_pred, lw=3.0, zorder=3, label='predicted')
        ax.step(np.concatenate([[lo], t, [hi]]),
                np.concatenate([[0.0], (np.arange(len(t)) + 1) / len(t), [1.0]]),
                where='post', color=c_true, lw=3.0, zorder=4, label='true')
        ax.plot(t, np.zeros(len(t)), '|', color=c_true, ms=16, mew=2.2,
                zorder=5, label='components')
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('cumulative probability')
        ax.legend(loc='upper left', fontsize=19, frameon=False,
                  handlelength=1.5, borderpad=0.2, labelspacing=0.35)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
    return fig


def vis_qm_pred_vs_obs(params, save_path=None, ncols=5, names=None):
    """Quantile-mapped prediction against the observed loss, as a plain scatter.

    The companion rank-ordered figure (vis_rank_to_cumloss_qm) puts the
    predicted ORDER on x, which shows where the mapping over- or under-shoots
    along the ordering but never lets a reader read the error off the diagonal.
    This one does: one point per component, observed on x, predicted on y, the
    identity line, and each panel's Pearson r together with a weighted mean
    absolute error and the same error as a percentage of the unit's
    whole-horizon cumulative loss.  `params` needs code, cum_loss_fit,
    cum_loss_pred and weight_normal; `names` maps code -> the full city-event
    title.  Axes are SHARED per panel (square, common limits) so a point's
    distance from the diagonal is the error in day-equivalents."""
    codes = list(params['code'].drop_duplicates())
    names = names or {}
    with plt.rc_context(_SLIDE_RC):
        nrow = int(np.ceil(len(codes) / ncols))
        fig, axes = plt.subplots(nrow, ncols, figsize=(3.7 * ncols, 3.8 * nrow),
                                 squeeze=False)
        for i, c in enumerate(codes):
            ax = axes[i // ncols][i % ncols]
            s = params[params.code == c].dropna(
                subset=['cum_loss_fit', 'cum_loss_pred'])
            if len(s) < 2:
                ax.axis('off')
                continue
            x = s['cum_loss_fit'].to_numpy(dtype=float)
            y = s['cum_loss_pred'].to_numpy(dtype=float)
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            pad = 0.10 * (hi - lo if hi > lo else 1.0)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], color=_PUB_GREY, lw=1.6, zorder=1)
            ax.scatter(x, y, s=150, color=_PUB_BLUE, alpha=0.9, zorder=3,
                       edgecolor='white', linewidth=1.3)
            ax.set_title(names.get(c, c))
            # R^2 against the IDENTITY line, not against a refitted slope:
            # 1 - sum (y-x)^2 / sum (x - xbar)^2.  It therefore penalises the
            # distance from the diagonal the panel actually draws, and goes
            # NEGATIVE when the prediction is worse than this unit's own mean.
            # Pearson r (co-variation, blind to shift and scale), then two
            # error numbers that share ONE numerator, sum_j w_j |y_j - x_j|:
            #   MAE   divides it by sum_j w_j  -> day-equivalents, same units
            #         as both axes, so it reads as a distance from the diagonal;
            #   MAPE  divides it by sum_j w_j |x_j| -> the same error as a share
            #         of the unit's whole-horizon cumulative loss.
            # Both are WEIGHTED by weight_normal, the city-aggregation weight
            # the curve figures use, so a component that carries almost none of
            # the city's flow cannot dominate either number.  Pooling the
            # denominator over the horizon is also what makes the percentage
            # usable at all: cum_loss is SIGNED and crosses zero, so the
            # per-component form divided by quantities near zero -- one
            # component at |observed| ~ 0.07 used to set the whole panel's MAPE
            # while its absolute error was among the smallest.
            wj = s['weight_normal'].to_numpy(dtype=float)
            rp_ = pearsonr(x, y).statistic
            num = float(np.sum(wj * np.abs(y - x)))
            mae_ = num / float(np.sum(wj))
            mape_ = 100.0 * num / float(np.sum(wj * np.abs(x)))
            ax.text(0.965, 0.035,
                    'r = {:+.2f}'.format(rp_) + chr(10)
                    + 'MAE = {:.2f}'.format(mae_) + chr(10)
                    + 'MAPE = {:.0f}%'.format(mape_),
                    transform=ax.transAxes, fontsize=18, va='bottom',
                    ha='right', linespacing=1.35, zorder=4,
                    # lower right, the corner the diagonal point cloud leaves
                    # most open; the white ground covers the units whose
                    # scatter still reaches into it
                    bbox=dict(facecolor='white', alpha=0.78, edgecolor='none',
                              boxstyle='square,pad=0.18'))
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
        for k in range(len(codes), nrow * ncols):
            axes[k // ncols][k % ncols].axis('off')
        fig.supxlabel('observed component cumulative loss', fontsize=26)
        fig.supylabel('quantile-mapped prediction', fontsize=26)
        fig.tight_layout()
        _pub_save(fig, save_path)
    return fig


def vis_city_curves_grid(per_city, save_path=None, ncols=5, names=None):
    """Every city-event's absolute mobility curve on ONE page.

    `per_city` maps code -> (days, ground_truth, {label: values}); `names` maps
    code -> the full city-event title.  Each panel keeps its OWN y scale: the
    units differ by an order of magnitude in flow volume, so a shared y would
    flatten the small cities into invisible lines.  x is shared and therefore
    labelled once; the y label names the quantity once for the page.  A single
    figure-level legend sits in the free grid slot -- with 13 panels on a 5-wide
    grid there are two spare, and using one costs nothing.

    Each legend entry carries that method's page-level error RATE: per unit the
    mean over days of |forecast - observed| / observed, then the mean over
    units (the mean absolute percentage error).  It is computed HERE, from the
    very arrays that are drawn, so the number cannot drift away from the lines;
    and being scale-free it reads the same on these magnitude curves as on the
    relative curves underneath.  The per-panel numbers stay absolute (a MAE in
    flow volume, in each method's own colour), so the page carries both the
    per-unit magnitude of the error and the one comparable rate."""
    codes = list(per_city)
    names = names or {}
    nrow = int(np.ceil(len(codes) / ncols))
    rate = {}
    for code in codes:
        _days, _gt, _lines = per_city[code]
        _gt = np.asarray(_gt, dtype=float)
        for lab, vals in _lines.items():
            vals = np.asarray(vals, dtype=float)
            ok = np.isfinite(_gt) & np.isfinite(vals) & (_gt != 0)
            if ok.any():
                rate.setdefault(lab, []).append(
                    float(np.mean(np.abs(vals[ok] - _gt[ok]) / _gt[ok])))
    rate = {lab: float(np.mean(v)) for lab, v in rate.items()}

    rc = dict(_SLIDE_RC, **{'font.size': 19, 'axes.titlesize': 21,
                            'xtick.labelsize': 15, 'ytick.labelsize': 15})
    with plt.rc_context(rc):
        fig, axes2d = plt.subplots(nrow, ncols, figsize=(4.6 * ncols,
                                                         3.6 * nrow),
                                   squeeze=False, sharex=True)
        axes = axes2d.ravel()
        for ax, code in zip(axes, codes):
            days, gt, lines = per_city[code]
            ax.plot(days, gt, color='#3a3a3a', lw=2.6, marker='o', ms=5,
                    label='ground truth', zorder=5)
            gt = np.asarray(gt, dtype=float)
            # The MAE stack is ordered by _MAE_TEXT_ORDER, NOT by plotting
            # order: the baseline reads first, so every panel is scanned as
            # "what the baseline costs, then what the model saves".
            stack = sorted(lines, key=_mae_text_row)
            for i, (lab, vals) in enumerate(lines.items()):
                vals = np.asarray(vals, dtype=float)
                col, dsh = _curve_style(lab, i)
                ax.plot(days, vals, color=col, lw=2.2, ls=dsh, marker='.',
                        ms=5, label=lab)
                # each method's MAE, in its own colour, so the number needs no
                # separate key; stacked from the top-left downwards
                ax.text(0.03, 0.97 - 0.115 * stack.index(lab),
                        'MAE {:.3g}'.format(np.nanmean(np.abs(vals - gt))),
                        transform=ax.transAxes, color=col, fontsize=16,
                        va='top', ha='left')
            ax.set_title(names.get(code, code))
            ax.margins(y=0.10)
        h, l = axes[0].get_legend_handles_labels()
        l = [f'{lab}  (MAPE {rate[lab]:.1%})' if lab in rate else lab
             for lab in l]
        for ax in axes[len(codes):]:
            ax.axis('off')
        if len(axes) > len(codes):
            spare = len(axes) - len(codes)
            lg = axes[len(codes)].legend(h, l, loc='center', fontsize=20,
                                         frameon=False, handlelength=1.9,
                                         labelspacing=0.6)
            # loc='center' centres the box in the FIRST free cell, so a longer
            # label grows it symmetrically and its left edge creeps back over
            # the last panel.  Re-anchor to the middle of the whole free region
            # instead: length-independent, with a spare cell either side to
            # absorb the growth.
            if spare > 1:
                lg.set_bbox_to_anchor((spare / 2.0, 0.5),
                                      transform=axes[len(codes)].transAxes)
        fig.supxlabel('days since landfall', fontsize=26)
        fig.supylabel('daily mobility magnitude', fontsize=26)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)),
                        exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
    return fig
