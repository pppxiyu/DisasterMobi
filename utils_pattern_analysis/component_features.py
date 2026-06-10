"""
Per-component temporal & functional feature quantification, and their
cross-correlation — "do components with a certain temporal rhythm live in a
certain functional space?"

Each NMF component becomes one observation with
  temporal features  (from the temporal factor W, PRE-DISASTER segment only):
      peak_slot      argmax of the normalised within-day profile (ordinal 0..S-1)
      am_pm          sum of morning-slot shares (6-12h) minus evening-slot
                     shares (16-22h); >0 morning-type, <0 evening-type
      weekday_ratio  mean weekday daily total / mean weekend daily total;
                     >1 weekday-dominated (commute-like), <1 weekend-dominated
  functional features (from the O×D function cross-tab M, Mix/Unknown dropped
  and the remaining categories renormalised to sum 1):
      share_<cat>    (origin share + destination share) / 2 per category
      diag_share     trace of the renormalised matrix = same-function flow
                     fraction (high = activity stays within one function type)

Correlation
-----------
time_function_correlation computes pairwise SPEARMAN rank correlation between
the temporal and functional feature columns across components (both cities are
pooled by the caller to enlarge the sample).  Spearman is rank-based, so it
handles peak_slot's ordinality and is invariant to monotone transforms (no log
needed for weekday_ratio).

Functions
---------
temporal_features(W, n_nor, first_day, slots_per_day, interval_hours) -> DataFrame
functional_features(M, categories, drop) -> DataFrame
time_function_correlation(df, time_cols, func_cols) -> (rho_df, pval_df)
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
        'Saturday', 'Sunday']

# Hour bounds defining the morning / evening windows for am_pm.  Slots fully
# inside [AM_START, AM_END) count as morning; fully inside [PM_START, PM_END)
# as evening; anything else (midday) is a buffer and counts for neither.
AM_START, AM_END = 6, 12
PM_START, PM_END = 16, 22
DAY_START_HOUR   = 6          # first active slot starts at 06:00 (periodic_trim)


def temporal_features(W, n_nor, first_day, slots_per_day, interval_hours):
    """
    Quantify each component's temporal signature from the PRE-DISASTER segment.

    Parameters
    ----------
    W              : ndarray [time × k]  temporal factors (unified NMF)
    n_nor          : int    column where the disaster period starts; only
                     W[:n_nor] is used (normal-period rhythm)
    first_day      : str    weekday of the first window day (e.g. 'Monday')
    slots_per_day  : int    active slots per day (= SLOTS_ACTIVE)
    interval_hours : int    hours per slot (= 24 // SLOT_PER_DAY)

    Returns
    -------
    DataFrame indexed by component (0..k-1) with columns:
      peak_slot (int), peak_slot_label (str), am_pm (float), weekday_ratio (float)
    """
    W_nor = np.asarray(W, dtype=float)[:n_nor, :]
    n_days = n_nor // slots_per_day
    if n_days * slots_per_day != n_nor:
        raise ValueError(f"n_nor={n_nor} is not a multiple of slots_per_day={slots_per_day}")
    k = W_nor.shape[1]
    # [days × slots × k]
    cube = W_nor.reshape(n_days, slots_per_day, k)

    # ── within-day profile (mean over days, normalised per component) ─────────
    profile = cube.mean(axis=0)                                   # [slots × k]
    tot = profile.sum(axis=0, keepdims=True)
    p = np.divide(profile, tot, out=np.zeros_like(profile), where=tot != 0)

    peak_slot = p.argmax(axis=0)                                  # [k]
    slot_label = [
        f"{DAY_START_HOUR + s*interval_hours}-{DAY_START_HOUR + (s+1)*interval_hours}h"
        for s in range(slots_per_day)
    ]

    # morning / evening slot sets from hour bounds (resolution-independent)
    am_slots = [s for s in range(slots_per_day)
                if DAY_START_HOUR + s*interval_hours >= AM_START
                and DAY_START_HOUR + (s+1)*interval_hours <= AM_END]
    pm_slots = [s for s in range(slots_per_day)
                if DAY_START_HOUR + s*interval_hours >= PM_START
                and DAY_START_HOUR + (s+1)*interval_hours <= PM_END]
    am_pm = p[am_slots, :].sum(axis=0) - p[pm_slots, :].sum(axis=0)   # [k]

    # ── weekday vs weekend daily totals ───────────────────────────────────────
    daily = cube.sum(axis=1)                                      # [days × k]
    di0 = DAYS.index(first_day.capitalize())
    is_weekday = np.array([((di0 + d) % 7) < 5 for d in range(n_days)])
    wk_mean = daily[is_weekday].mean(axis=0)
    we_mean = daily[~is_weekday].mean(axis=0)
    weekday_ratio = np.divide(wk_mean, we_mean,
                              out=np.full(k, np.nan), where=we_mean != 0)

    return pd.DataFrame({
        'peak_slot':       peak_slot,
        'peak_slot_label': [slot_label[s] for s in peak_slot],
        'am_pm':           am_pm,
        'weekday_ratio':   weekday_ratio,
    }, index=pd.RangeIndex(k, name='component'))


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
      diag_share        same-function flow fraction (trace)
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
                         **{f'share_to_{c}':   np.nan for c in kept},
                         'diag_share': np.nan})
            continue
        sub = sub / s
        origin = sub.sum(axis=1)        # outflow: departing from function a
        dest   = sub.sum(axis=0)        # inflow:  arriving at function b
        rows.append({**{f'share_from_{c}': origin[j] for j, c in enumerate(kept)},
                     **{f'share_to_{c}':   dest[j]   for j, c in enumerate(kept)},
                     'diag_share': np.trace(sub)})
    return pd.DataFrame(rows, index=pd.RangeIndex(M.shape[0], name='component'))


def _daily_relative_curve(W, n_nor, first_day, slots_per_day):
    """
    Per-component DISASTER-period daily activity relative to a weekday/weekend-
    matched pre-disaster baseline:

        r_k(d) = daily_k(d) / baseline_k(day-type of d)

    baseline_k(weekday)  = mean normal-half daily total over Mon–Fri days
    baseline_k(weekend)  = mean normal-half daily total over Sat–Sun days

    Returns a DataFrame [n_disaster_days × k]; index = days since landfall
    (0 = landfall day).  Components with a zero baseline get NaN.
    """
    W = np.asarray(W, dtype=float)
    n_days = W.shape[0] // slots_per_day
    if n_days * slots_per_day != W.shape[0]:
        raise ValueError("W length is not a multiple of slots_per_day")
    days_nor = n_nor // slots_per_day
    daily = W.reshape(n_days, slots_per_day, W.shape[1]).sum(axis=1)   # [days × k]

    di0 = DAYS.index(first_day.capitalize())
    is_wd = np.array([((di0 + d) % 7) < 5 for d in range(n_days)])

    base_wd = daily[:days_nor][is_wd[:days_nor]].mean(axis=0)          # [k]
    base_we = daily[:days_nor][~is_wd[:days_nor]].mean(axis=0)

    r = np.full((n_days - days_nor, W.shape[1]), np.nan)
    for i, d in enumerate(range(days_nor, n_days)):
        base = base_wd if is_wd[d] else base_we
        r[i] = np.divide(daily[d], base, out=np.full(W.shape[1], np.nan),
                         where=base > 0)
    return pd.DataFrame(r, index=pd.RangeIndex(len(r), name='day_since_landfall'))


def resilience_curves(W, n_nor, first_day, slots_per_day, smooth=3):
    """
    Smoothed relative-activity curves r_k(d) for the disaster period
    (centred rolling mean over `smooth` days; smooth=1 disables).
    DataFrame [n_disaster_days × k], index = days since landfall.
    """
    r = _daily_relative_curve(W, n_nor, first_day, slots_per_day)
    if smooth and smooth > 1:
        r = r.rolling(smooth, center=True, min_periods=1).mean()
    return r


def resilience_features(W, n_nor, first_day, slots_per_day, smooth=3):
    """
    Quantify each component's disaster response from its (smoothed) relative
    daily curve r_k(d) — the "drop and come back" pattern:

      drop_depth     = 1 − min(r)        how deep it fell (0 none, 1 total
                       stop; NEGATIVE = rose above baseline → emergent pattern)
      trough_day     = argmin(r)         days from landfall to the trough
      recovery_level = mean of the last 3 days of r   (≈1 recovered, <1 not,
                       >1 overshoot)
      cum_loss       = Σ_d max(0, 1 − r(d))   resilience-triangle area: total
                       activity lost over the disaster window (day-equivalents;
                       smaller = more resilient).  Above-baseline excess is NOT
                       credited against losses.

    Returns DataFrame indexed by component with those four columns.
    """
    r = resilience_curves(W, n_nor, first_day, slots_per_day, smooth=smooth)
    arr = r.to_numpy()
    return pd.DataFrame({
        'drop_depth':     1.0 - np.nanmin(arr, axis=0),
        'trough_day':     np.nanargmin(arr, axis=0),
        'recovery_level': np.nanmean(arr[-3:], axis=0),
        'cum_loss':       np.nansum(np.clip(1.0 - arr, 0, None), axis=0),
    }, index=pd.RangeIndex(arr.shape[1], name='component'))


def time_function_correlation(df, time_cols, func_cols):
    """
    Pairwise Spearman correlation between temporal and functional features
    across components (rows of df).  Rows with NaN in a pair are dropped
    pairwise.

    Returns
    -------
    rho_df  : DataFrame [time_cols × func_cols] Spearman rho
    pval_df : DataFrame [time_cols × func_cols] two-sided p-values
    """
    rho = pd.DataFrame(index=time_cols, columns=func_cols, dtype=float)
    pval = pd.DataFrame(index=time_cols, columns=func_cols, dtype=float)
    for t in time_cols:
        for f in func_cols:
            sub = df[[t, f]].dropna()
            if len(sub) < 3:
                rho.loc[t, f], pval.loc[t, f] = np.nan, np.nan
                continue
            r, pv = spearmanr(sub[t], sub[f])
            rho.loc[t, f], pval.loc[t, f] = r, pv
    return rho, pval
