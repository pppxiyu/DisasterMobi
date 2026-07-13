"""
Per-component temporal & functional feature quantification, and their
cross-correlation — "do components with a certain temporal rhythm live in a
certain functional space?"

Each NMF component becomes one observation with
  temporal features  (from the temporal factor W, PRE-DISASTER segment only;
  weekend-dominated components form their own 'weekend' category and the
  within-day profile is computed from weekday days only):
      peak_slot      categorical: argmax slot of the weekday profile
                     ('6-8h' … '20-22h') or 'weekend'
      peak_period    categorical: day-period band of the weekday profile
                     (morning_peak 6-10h / midday 10-16h / evening_peak
                     16-20h / night 20-22h) or 'weekend'
      weekday_ratio  mean weekday daily total / mean weekend daily total;
                     >1 weekday-dominated (commute-like), <1 weekend-dominated
  functional features (from the O×D function cross-tab M, Mix/Unknown dropped
  and the remaining categories renormalised to sum 1):
      share_from_<cat>  outflow side, the fraction departing from function <cat>
      share_to_<cat>    inflow side, the fraction arriving at function <cat>
                        (row/column sums, so diagonal same-function flow is
                        counted on both sides)
  resilience features (disaster-period daily activity relative to a weekday/
  weekend-matched pre-disaster baseline; every metric reads higher = worse):
      drop_depth, early_collapse, recovery_day, recovery_deficit, cum_loss

Correlation
-----------
time_function_correlation computes pairwise SPEARMAN rank correlation between
two feature-column groups across components.  Spearman is rank-based and
invariant to monotone transforms (no log needed for weekday_ratio).

Functions
---------
temporal_features(W, n_nor, first_day, slots_per_day, interval_hours) -> DataFrame
functional_features(M, categories, drop) -> DataFrame
component_function_entropy(M, categories, drop, base) -> DataFrame  (outflow/inflow Shannon entropy)
spatial_features(H, distances) -> DataFrame  (loading-weighted mean/std flow distance, km)
socioeconomic_features(H, values, name) -> DataFrame  (loading-weighted median income)
resilience_curves(W, n_nor, first_day, slots_per_day, n_dis, smooth) -> DataFrame
resilience_features(W, n_nor, first_day, slots_per_day, n_dis, smooth) -> DataFrame
recovery_rate_features(W, n_nor, first_day, slots_per_day, n_dis, smooth) -> DataFrame  (exp-recovery rate lambda)
time_function_correlation(df, time_cols, func_cols) -> (rho_df, pval_df)
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
        'Saturday', 'Sunday']

DAY_START_HOUR = 6            # first active slot starts at 06:00 (periodic_trim)

# Day-period bands for the peak_period categorical (hour bounds, end-exclusive).
# A slot belongs to a band when it lies fully inside [start, end).  Band order
# defines the category codes 0..3.
PERIOD_BANDS = [
    ('morning_peak', 6, 10),
    ('midday',       10, 16),
    ('evening_peak', 16, 20),
    ('night',        20, 22),
]


def temporal_features(W, n_nor, first_day, slots_per_day, interval_hours,
                      weekend_ratio_threshold=1.0):
    """
    Quantify each component's temporal signature from the PRE-DISASTER segment.

    Weekend-dominated components (weekday_ratio < weekend_ratio_threshold)
    form their own 'weekend' category in BOTH
    categorical features and get no within-day breakdown.  For the remaining
    (weekday-dominated) components the within-day profile is computed from the
    WEEKDAY normal days only, so weekend days never dilute the slot shapes.

    Parameters
    ----------
    W              : ndarray [time × k]  temporal factors (NMF)
    n_nor          : int    end of the clean normal columns; only W[:n_nor]
                     is used (normal-period rhythm)
    first_day      : str    weekday of the first window day (e.g. 'Monday')
    slots_per_day  : int    active slots per day (= SLOTS_ACTIVE)
    interval_hours : int    hours per slot (= 24 // SLOT_PER_DAY)

    Returns
    -------
    DataFrame indexed by component (0..k-1) with columns:
      peak_slot         int   argmax slot of the weekday profile, or
                              slots_per_day for the weekend category
                              (so weekend sorts after all slots)
      peak_slot_label   str   '6-8h' … '20-22h' or 'weekend'
      peak_period       int   band code 0..3 (PERIOD_BANDS order), or
                              len(PERIOD_BANDS) for weekend
      peak_period_label str   'morning_peak' / 'midday' / 'evening_peak' /
                              'night' / 'weekend'.  Band assignment uses the
                              WIDTH-CORRECTED intensity (mean share per slot
                              inside the band), so wider bands gain no
                              mechanical advantage
      weekday_ratio     float mean weekday daily total / mean weekend daily
                              total (continuous; NaN if the weekend mean is 0,
                              which counts as weekday-dominated)
    """
    W_nor = np.asarray(W, dtype=float)[:n_nor, :]
    n_days = n_nor // slots_per_day
    if n_days * slots_per_day != n_nor:
        raise ValueError(f"n_nor={n_nor} is not a multiple of slots_per_day={slots_per_day}")
    k = W_nor.shape[1]
    # [days × slots × k]
    cube = W_nor.reshape(n_days, slots_per_day, k)

    # ── weekday vs weekend daily totals ───────────────────────────────────────
    daily = cube.sum(axis=1)                                      # [days × k]
    di0 = DAYS.index(first_day.capitalize())
    is_weekday = np.array([((di0 + d) % 7) < 5 for d in range(n_days)])
    wk_mean = daily[is_weekday].mean(axis=0)
    we_mean = daily[~is_weekday].mean(axis=0)
    weekday_ratio = np.divide(wk_mean, we_mean,
                              out=np.full(k, np.nan), where=we_mean != 0)

    # Weekend-dominated components form the 'weekend' category.  A NaN ratio
    # (zero weekend activity) counts as weekday-dominated.
    weekend_type = np.where(np.isnan(weekday_ratio), False,
                            weekday_ratio < weekend_ratio_threshold)

    # ── within-day profile from WEEKDAY normal days only ──────────────────────
    profile = cube[is_weekday].mean(axis=0)                       # [slots × k]
    tot = profile.sum(axis=0, keepdims=True)
    p = np.divide(profile, tot, out=np.zeros_like(profile), where=tot != 0)

    peak_idx = p.argmax(axis=0)                                   # [k]
    slot_label = [
        f"{DAY_START_HOUR + s*interval_hours}-{DAY_START_HOUR + (s+1)*interval_hours}h"
        for s in range(slots_per_day)
    ]

    # Band membership from hour bounds (resolution-independent); band score =
    # mean share per slot inside the band (width-corrected).
    band_means = []
    for _, h0, h1 in PERIOD_BANDS:
        slots = [s for s in range(slots_per_day)
                 if DAY_START_HOUR + s*interval_hours >= h0
                 and DAY_START_HOUR + (s+1)*interval_hours <= h1]
        band_means.append(p[slots, :].mean(axis=0) if slots else np.zeros(k))
    band_idx = np.stack(band_means).argmax(axis=0)                # [k]

    n_band = len(PERIOD_BANDS)
    rows = []
    for j in range(k):
        if weekend_type[j]:
            rows.append((slots_per_day, 'weekend', n_band, 'weekend'))
        else:
            rows.append((int(peak_idx[j]), slot_label[peak_idx[j]],
                         int(band_idx[j]), PERIOD_BANDS[band_idx[j]][0]))

    out = pd.DataFrame(rows, columns=['peak_slot', 'peak_slot_label',
                                      'peak_period', 'peak_period_label'],
                       index=pd.RangeIndex(k, name='component'))
    out['weekday_ratio'] = weekday_ratio
    return out


def functional_features(M, categories, drop=('Mix', 'Unknown')):
    """
    Quantify each component's functional profile from its O×D cross-tab,
    SEPARATELY for the outflow (origin) and inflow (destination) sides.

    Mix/Unknown rows+columns are dropped and the remaining matrix renormalised
    to sum 1, so shares are over substantive function types only.

    Parameters
    ----------
    M          : ndarray [k × C × C]  per-component O×D proportions
                 (build_od_function_matrix output)
    categories : list[str] axis categories of M (length C)
    drop       : categories excluded before renormalisation

    Returns
    -------
    DataFrame indexed by component with columns:
      share_from_<cat>  outflow side: fraction of the component's flow that
                        DEPARTS from function <cat>  (row sums; sum to 1)
      share_to_<cat>    inflow side: fraction that ARRIVES at function <cat>
                        (column sums; sum to 1)
      Full row/column sums include the diagonal, so same-function flow
      a→a contributes to both share_from_a and share_to_a.
      A component whose kept-category mass is 0 gets NaN features.
    """
    M = np.asarray(M, dtype=float)
    keep_idx = [i for i, c in enumerate(categories) if c not in drop]
    kept = [categories[i] for i in keep_idx]
    rows = []
    for comp in range(M.shape[0]):
        sub = M[comp][np.ix_(keep_idx, keep_idx)]
        s = sub.sum()
        if s <= 0:
            rows.append({**{f'share_from_{c}': np.nan for c in kept},
                         **{f'share_to_{c}':   np.nan for c in kept}})
            continue
        sub = sub / s
        origin = sub.sum(axis=1)        # outflow: departing from function a
        dest   = sub.sum(axis=0)        # inflow:  arriving at function b
        rows.append({**{f'share_from_{c}': origin[j] for j, c in enumerate(kept)},
                     **{f'share_to_{c}':   dest[j]   for j, c in enumerate(kept)}})
    return pd.DataFrame(rows, index=pd.RangeIndex(M.shape[0], name='component'))


def component_function_entropy(M, categories=None, drop=(), base=np.e):
    """
    Per-component Shannon entropy of the OUTFLOW (origin) and INFLOW
    (destination) functional distributions, computed from the SAME per-component
    O×D matrix M that the functionality heatmap plots (each M[i] sums to 1).

    For component i:
        outflow distribution = row sums of M[i]    (share DEPARTING per origin function)
        inflow  distribution = column sums of M[i]  (share ARRIVING per dest function)
        entropy_from = H(outflow),  entropy_to = H(inflow)

    H(p) = -Σ p_c ln p_c (nats; base=e). LOWER entropy = the component's flow is
    concentrated on fewer functions (more specialised); HIGHER = spread across
    many.  Both marginals sum to 1, so each H lies in [0, ln K] (K = #categories).

    By DEFAULT all axis categories are kept, matching the heatmap (which includes
    'Mix').  Pass drop=('Mix', 'Unknown') with `categories` to restrict to
    substantive functions — then the marginals match functional_features'
    share_from / share_to and H lies in [0, ln 6].

    Returns DataFrame indexed by component with columns entropy_from, entropy_to;
    a component whose kept mass is 0 gets NaN on both.
    """
    M = np.asarray(M, dtype=float)
    if drop:
        if categories is None:
            raise ValueError("pass `categories` when `drop` is non-empty")
        keep = [i for i, c in enumerate(categories) if c not in drop]
        M = M[np.ix_(np.arange(M.shape[0]), keep, keep)]

    log_base = np.log(base)

    def _H(p):
        s = p.sum()
        if s <= 0:
            return np.nan
        p = p[p > 0] / s
        return float(-np.sum(p * np.log(p)) / log_base)

    rows = [(_H(M[i].sum(axis=1)), _H(M[i].sum(axis=0))) for i in range(M.shape[0])]
    return pd.DataFrame(rows, columns=['entropy_from', 'entropy_to'],
                        index=pd.RangeIndex(M.shape[0], name='component'))


def spatial_features(H, distances):
    """
    Per-component SPATIAL features from the spatial factor H and a per-flow
    distance array — structurally parallel to temporal_/functional_/
    resilience_features (each takes an already-built factor/intermediate and
    returns one row per component).

    Each component is a flow map: H[i, :] are its non-negative loadings over the
    OD flows, and distances[j] is flow j's origin->destination centroid distance
    (km), aligned to H's columns (build via graph_io.build_distance_array).  The
    loading-weighted summary describes how LONG-RANGE the component's flows are:

        mean_distance[i] = Σ_j H[i,j]·d_j / Σ_j H[i,j]
        std_distance[i]  = sqrt(Σ_j H[i,j]·(d_j − mean_i)² / Σ_j H[i,j])
                           (spread of flow distances within the component)

    A component whose loadings sum to <= 0 gets NaN.  Returns a DataFrame indexed
    by component with columns mean_distance, std_distance.
    """
    H = np.asarray(H, dtype=float)
    d = np.asarray(distances, dtype=float)
    rows = []
    for i in range(H.shape[0]):
        w   = H[i]
        tot = w.sum()
        if tot <= 0:
            rows.append((np.nan, np.nan))
            continue
        mean = float(np.sum(w * d) / tot)
        var  = float(np.sum(w * (d - mean) ** 2) / tot)
        rows.append((mean, float(np.sqrt(max(var, 0.0)))))
    return pd.DataFrame(rows, columns=['mean_distance', 'std_distance'],
                        index=pd.RangeIndex(H.shape[0], name='component'))


def socioeconomic_features(H, income_array, name='median_income'):
    """
    Per-component loading-weighted median household income from the spatial factor
    H and a per-flow income array (km-distance analogue spatial_features, but
    NaN-AWARE because the Census suppresses some block-group incomes).

    income_array[j] is flow j's median household income (USD), aligned to H's
    columns (build via graph_io.build_income_array); NaN where unavailable.  For
    component i the value is the loading-weighted mean over flows WITH a valid
    income (missing-income flows are dropped, the weights renormalised):

        value[i] = Σ_{j: valid} H[i,j]·income_j / Σ_{j: valid} H[i,j]

    A component with no valid-income loading mass gets NaN.  `name` sets the single
    output column (e.g. 'median_income_combined' / '_origin' for the two endpoint
    modes).  Returns a DataFrame indexed by component with that one column.
    """
    H = np.asarray(H, dtype=float)
    inc = np.asarray(income_array, dtype=float)
    valid = ~np.isnan(inc)
    inc_safe = np.where(valid, inc, 0.0)            # zeroed so NaN flows don't poison the sum
    rows = []
    for i in range(H.shape[0]):
        wv  = np.where(valid, H[i], 0.0)            # weight only the valid-income flows
        tot = wv.sum()
        rows.append(float(np.sum(wv * inc_safe) / tot) if tot > 0 else np.nan)
    return pd.DataFrame({name: rows},
                        index=pd.RangeIndex(H.shape[0], name='component'))


def _daily_relative_curve(W, n_nor, first_day, slots_per_day, n_dis=None):
    """
    Per-component DISASTER-period daily activity relative to a weekday/weekend-
    matched pre-disaster baseline:

        r_k(d) = daily_k(d) / baseline_k(day-type of d)

    baseline_k(weekday)  = mean normal-half daily total over Mon–Fri days
    baseline_k(weekend)  = mean normal-half daily total over Sat–Sun days

    Parameters
    ----------
    n_nor : int   column where the NORMAL segment ends (baseline uses [0, n_nor)).
    n_dis : int   column where the DISASTER segment starts (defaults to n_nor).
                  Columns in [n_nor, n_dis) — a pre-landfall alert/buffer
                  segment — are excluded from BOTH the baseline and the curve.

    Returns a DataFrame [n_disaster_days × k]; index = days since landfall
    (0 = landfall day).  Components with a zero baseline get NaN.
    """
    # Without n_dis the disaster starts right after the normal segment
    # (no buffer).
    if n_dis is None:
        n_dis = n_nor

    # W and both boundaries must align to whole days, otherwise the
    # day-level reshape below would mix slots from different days.
    W = np.asarray(W, dtype=float)
    n_days = W.shape[0] // slots_per_day
    if n_days * slots_per_day != W.shape[0]:
        raise ValueError("W length is not a multiple of slots_per_day")
    if n_nor % slots_per_day or n_dis % slots_per_day:
        raise ValueError("n_nor / n_dis must be multiples of slots_per_day")
    days_nor = n_nor // slots_per_day
    days_dis = n_dis // slots_per_day

    # Collapse slots into daily totals.  Everything below works at day
    # granularity, which also averages out slot-level noise.
    daily = W.reshape(n_days, slots_per_day, W.shape[1]).sum(axis=1)   # [days × k]

    # Weekday/weekend label per day, projected from the window's first day
    # (valid because the window is contiguous in calendar time).
    di0 = DAYS.index(first_day.capitalize())
    is_wd = np.array([((di0 + d) % 7) < 5 for d in range(n_days)])

    # Day-type-matched baselines from the NORMAL days only — buffer and
    # disaster days never enter the denominator.  Two baselines so that the
    # ordinary weekend dip is not read as a disaster drop.
    base_wd = daily[:days_nor][is_wd[:days_nor]].mean(axis=0)          # [k]
    base_we = daily[:days_nor][~is_wd[:days_nor]].mean(axis=0)

    # Relative value per disaster day, divided by the SAME day-type baseline.
    # The loop starts at days_dis, so buffer days are skipped entirely.
    # A zero baseline (component inactive in the normal segment) yields NaN,
    # not inf — note that small baselines inflate r for emergent components.
    r = np.full((n_days - days_dis, W.shape[1]), np.nan)
    for i, d in enumerate(range(days_dis, n_days)):
        base = base_wd if is_wd[d] else base_we
        r[i] = np.divide(daily[d], base, out=np.full(W.shape[1], np.nan),
                         where=base > 0)
    return pd.DataFrame(r, index=pd.RangeIndex(len(r), name='day_since_landfall'))


def resilience_curves(W, n_nor, first_day, slots_per_day, n_dis=None, smooth=3):
    """
    Smoothed relative-activity curves r_k(d) for the disaster period
    (centred rolling mean over `smooth` days; smooth=1 disables).
    DataFrame [n_disaster_days × k], index = days since landfall.
    See _daily_relative_curve for the n_nor / n_dis (buffer) semantics.
    """
    r = _daily_relative_curve(W, n_nor, first_day, slots_per_day, n_dis=n_dis)
    if smooth and smooth > 1:
        r = r.rolling(smooth, center=True, min_periods=1).mean()
    return r


def resilience_features(W, n_nor, first_day, slots_per_day, n_dis=None, smooth=3,
                        recovery_threshold=0.9):
    """
    Quantify each component's disaster response from its (smoothed) relative
    daily curve r_k(d) — the "drop and come back" pattern.  Every metric reads
    HIGHER = WORSE (early_collapse and recovery_deficit invert the natural
    argmin / recovery-level forms so the whole set shares that direction):

      drop_depth       = 1 − min(r)        how deep it fell (0 none, 1 total
                         stop; NEGATIVE = rose above baseline → emergent pattern)
      early_collapse   = (disaster days − 1) − argmin(r)   how soon it bottomed
                         out: high = bottomed early (on/near landfall), 0 = on
                         the final day
      recovery_day     = 1 + last day with r < recovery_threshold   days until r
                         reaches the threshold AND STAYS there — any later dip
                         below the threshold resets the clock, so oscillating
                         recoveries are not credited early.  0 means never below
                         the threshold (unhurt or emergent).  A value equal to
                         the window length means NOT recovered within the window
                         (right-censored, kept instead of NaN so the hardest-hit
                         components stay in the rank correlation).
      recovery_deficit = 1 − mean of the last 3 days of r   shortfall still left
                         at the window end (0 recovered, >0 not, <0 overshoot)
      cum_loss         = Σ_d (1 − r(d))   NET cumulative deviation from baseline over
                         the disaster window (day-equivalents; >0 = net loss, <0 = net
                         gain).  UNCLIPPED: above-baseline surges cancel below-baseline
                         drops, so the metric is LINEAR/additive across components (a
                         baseline-share-weighted sum of component cum_loss equals the
                         total-curve cum_loss; no Jensen/clipping gap).

    Returns DataFrame indexed by component with all five columns.

    KEEP LONG-TERM (owner decision, 2026-07-12): since 2026-07-12 the analyses
    consume ONLY cum_loss (run_pattern_nmf.RES_COLS = ['cum_loss']); drop_depth,
    early_collapse, recovery_day and recovery_deficit have no consumer anywhere.
    They are deliberately retained here so a retired metric can be restored by
    adding its name back to RES_COLS.  If a dead-code cleanup flags these
    metrics (or this function) as unused, do NOT delete them — surface this
    note and ask the owner to decide.
    """
    r = resilience_curves(W, n_nor, first_day, slots_per_day, n_dis=n_dis,
                          smooth=smooth)
    arr = r.to_numpy()
    n_days, k = arr.shape

    # A zero-baseline component has an all-NaN curve.  Such columns crash
    # nanargmin and make nansum return 0, so they are computed on a stand-in
    # and masked to NaN at the end.
    all_nan = np.isnan(arr).all(axis=0)
    safe = arr.copy()
    safe[:, all_nan] = 1.0

    # Last crossing below the threshold per component (NaN compares False).
    below = safe < recovery_threshold
    last_below = np.where(below.any(axis=0),
                          n_days - 1 - np.argmax(below[::-1], axis=0), -1)

    # early_collapse and recovery_deficit invert argmin(r) / mean-tail(r) so every
    # metric reads HIGHER = WORSE; n_days is the disaster-window length.
    feats = pd.DataFrame({
        'drop_depth':       1.0 - np.nanmin(safe, axis=0),
        'early_collapse':   (n_days - 1) - np.nanargmin(safe, axis=0).astype(float),
        'recovery_day':     (last_below + 1).astype(float),
        'recovery_deficit': 1.0 - np.nanmean(safe[-3:], axis=0),
        'cum_loss':         np.nansum(1.0 - safe, axis=0),
    }, index=pd.RangeIndex(k, name='component'))
    feats.loc[all_nan, :] = np.nan
    return feats


def recovery_rate_features(W, n_nor, first_day, slots_per_day, n_dis=None,
                           smooth=3, max_rate=5.0, min_points=3):
    """
    Exponential-recovery rate λ per component (added 2026-07-12 as the second
    resilience metric next to cum_loss).

    Each component's recovery segment — the smoothed relative curve r_k(d) from
    its lowest point d_min onward — is fitted with the exponential-recovery
    model written on the deficit scale:

        D(d) = D0 · exp(−λ · (d − d_min)),   D(d) = 1 − r(d)

    i.e. Q(t) = Q_target − (Q_target − Q_min)·e^(−λt) with Q_target = 1 (the
    pre-disaster baseline) and D0 fixed at the OBSERVED maximum deficit
    1 − min(r), so only λ is fitted (non-linear least squares, λ bounded to
    [0, max_rate]).  Fixing D0 mirrors the archived StepWiseModel fit and keeps
    the one-parameter fit stable at n ≈ 15 days.

    READING DIRECTION: λ is a RATE — HIGHER = FASTER recovery = MORE resilient,
    the OPPOSITE direction to cum_loss (higher = worse).  1/λ is the e-folding
    time in days; ln(2)/λ is the deficit half-life.

    NaN cases (the metric is undefined, and the affected component is dropped
    from λ-analyses only — every consumer handles metrics independently):
      * the component never fell below baseline (D0 <= 0: nothing to recover);
      * fewer than `min_points` days from d_min to the window end;
      * the all-NaN curve of a zero-baseline component;
      * a failed fit.

    Returns DataFrame ['recovery_lambda'] indexed by component (1/days).
    """
    r = resilience_curves(W, n_nor, first_day, slots_per_day, n_dis=n_dis,
                          smooth=smooth)
    arr = r.to_numpy()
    n_days, k = arr.shape
    lam = np.full(k, np.nan)
    for j in range(k):
        deficit = 1.0 - arr[:, j]
        if np.isnan(deficit).all():
            continue
        d_min = int(np.nanargmax(deficit))
        d0 = deficit[d_min]
        if not np.isfinite(d0) or d0 <= 0:
            continue
        tail = deficit[d_min:]
        ok = np.isfinite(tail)
        t = np.arange(len(tail), dtype=float)[ok]
        y = tail[ok]
        if len(y) < min_points:
            continue
        try:
            popt, _ = curve_fit(lambda t_, l: d0 * np.exp(-l * t_), t, y,
                                p0=[0.3], bounds=(0.0, max_rate), maxfev=10000)
            lam[j] = popt[0]
        except Exception:
            pass
    return pd.DataFrame({'recovery_lambda': lam},
                        index=pd.RangeIndex(k, name='component'))


def time_function_correlation(df, time_cols, func_cols, method='spearman'):
    """
    Pairwise correlation between two feature-column groups across components
    (rows of df).  Rows with NaN in a pair are dropped pairwise.

    method : 'spearman' (rank correlation, default — robust to non-linearity and
             outliers) or 'pearson' (linear correlation on the raw values).

    Returns
    -------
    rho_df  : DataFrame [time_cols × func_cols] correlation coefficient
    pval_df : DataFrame [time_cols × func_cols] two-sided p-values
    """
    corr_fn = {'spearman': spearmanr, 'pearson': pearsonr}.get(method)
    if corr_fn is None:
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    rho = pd.DataFrame(index=time_cols, columns=func_cols, dtype=float)
    pval = pd.DataFrame(index=time_cols, columns=func_cols, dtype=float)
    for t in time_cols:
        for f in func_cols:
            sub = df[[t, f]].dropna()
            # < 3 points, or a constant column (pearson is undefined / spearman NaN),
            # gives NaN — guarding both keeps pearson from raising on constants.
            if len(sub) < 3 or sub[t].nunique() < 2 or sub[f].nunique() < 2:
                rho.loc[t, f], pval.loc[t, f] = np.nan, np.nan
                continue
            r, pv = corr_fn(sub[t], sub[f])
            rho.loc[t, f], pval.loc[t, f] = r, pv
    return rho, pval
