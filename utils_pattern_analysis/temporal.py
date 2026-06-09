"""
Time-series decomposition and disaster-impact modelling.

Provides:
  - Prophet-based baseline decomposition and forecasting
  - GammaModel  / StepWiseModel  for fitting the impact/recovery curve
  - Full pipeline helpers used by both the standalone script and visualization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import curve_fit
from prophet import Prophet


# ── Impact-curve models ───────────────────────────────────────────────────────

class GammaModel:
    """Gamma-PDF shaped impact curve."""

    def __init__(self, A, alpha, beta, t0):
        self.A     = A      # amplitude (negative = drop)
        self.alpha = alpha  # shape  (>1 gives the hump)
        self.beta  = beta   # scale  (stretches recovery)
        self.t0    = t0     # time offset

    def predict(self, t):
        t = np.asanyarray(t)
        shifted = t - self.t0
        mask = shifted > 0
        res  = np.zeros_like(shifted, dtype=float)
        res[mask] = (
            self.A
            * (shifted[mask] ** (self.alpha - 1))
            * np.exp(-shifted[mask] / self.beta)
            / (gamma(self.alpha) * self.beta ** self.alpha)
        )
        return res


class StepWiseModel:
    """Two-stage model: linear drop then exponential recovery."""

    def __init__(self, A, alpha, beta, t0):
        self.A     = A      # value at peak impact (y_min)
        self.alpha = alpha  # exponential recovery rate
        self.beta  = beta   # slope of linear drop (starts from 0)
        self.t0    = t0     # time index of peak impact

    def predict(self, t):
        t   = np.asanyarray(t)
        res = np.zeros_like(t, dtype=float)
        drop_mask = t <= self.t0
        rec_mask  = t > self.t0
        res[drop_mask] = self.beta * t[drop_mask]
        res[rec_mask]  = self.A * np.exp(-self.alpha * (t[rec_mask] - self.t0))
        return res


# ── Curve fitting ─────────────────────────────────────────────────────────────

def fit_gamma_shape(data):
    y_raw = np.array(data)
    is_negative = np.nanmean(y_raw) < 0
    y_fit = np.abs(y_raw)
    t = np.arange(len(y_fit))
    mask = ~np.isnan(y_fit)

    def gamma_func(t, A, a, b, t0):
        s = t - t0
        res = np.zeros_like(s)
        m = s > 1e-5
        res[m] = A * (s[m] ** (a - 1) * np.exp(-s[m] / b)) / (gamma(a) * b ** a)
        return res

    p0 = [np.nansum(y_fit[mask]), 2.0, 5.0, -2.0]
    try:
        popt, _ = curve_fit(
            gamma_func, t[mask], y_fit[mask], p0=p0,
            bounds=([0, 1.001, 0.1, -20], [np.inf, 20, 50, 20]),
            maxfev=10000
        )
        A_final = -popt[0] if is_negative else popt[0]
        return GammaModel(A_final, popt[1], popt[2], popt[3])
    except Exception as e:
        print(f"Gamma fit failed: {e}")
        return None


def fit_stepwise_shape(data):
    """Fit a linear-drop then exponential-recovery curve, robust to NaN/Inf."""
    y_raw = np.array(data)
    t_raw = np.arange(len(y_raw))
    mask  = np.isfinite(y_raw)
    y_data, t_data = y_raw[mask], t_raw[mask]

    if len(y_data) < 2:
        print("Step-wise fit failed: not enough valid data points.")
        return None

    idx_min  = np.argmin(y_data)
    t0_val   = t_data[idx_min]
    y_min    = y_data[idx_min]
    beta_fit = (y_min / t0_val) if t0_val > 0 else 0

    def exp_recovery(t, alpha):
        return y_min * np.exp(-alpha * (t - t0_val))

    rec_mask = t_data >= t0_val
    try:
        popt, _ = curve_fit(
            exp_recovery, t_data[rec_mask], y_data[rec_mask],
            p0=[0.1], bounds=(0, 2)
        )
        return StepWiseModel(A=y_min, alpha=popt[0], beta=beta_fit, t0=float(t0_val))
    except Exception as e:
        print(f"Step-wise fit failed: {e}")
        return None


# ── Prophet helpers ───────────────────────────────────────────────────────────

def decompose_with_prophet(ts_data, start_date, freq='H',
                           daily_seasonality=True, weekly_seasonality=True):
    """
    Fits Prophet and extracts trend, seasonalities, noise, and the clean signal.

    Returns dict with keys: model, clean_ts, trend, seasonalities, noise.
    """
    dates = pd.date_range(start=start_date, periods=len(ts_data), freq=freq)
    df    = pd.DataFrame({'ds': dates, 'y': ts_data})

    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=False,
    )
    model.fit(df)
    forecast = model.predict(df[['ds']])

    seasonalities = {}
    if 'daily'  in forecast.columns: seasonalities['daily']  = forecast['daily'].tolist()
    if 'weekly' in forecast.columns: seasonalities['weekly'] = forecast['weekly'].tolist()

    return {
        'model':        model,
        'clean_ts':     forecast['yhat'].tolist(),
        'trend':        forecast['trend'].tolist(),
        'seasonalities': seasonalities,
        'noise':        (df['y'] - forecast['yhat']).tolist(),
    }


def forecast_with_prophet(model, periods, freq):
    """
    Extrapolates a fitted Prophet model into the future.

    freq is required (no default) — pass PROPHET_FREQ from config so the
    value stays correct when switching between 2h ('2H') and 3h ('3H') resolution.

    Returns dict with keys: dates, forecast, trend, seasonalities,
    lower_bound, upper_bound.
    """
    future   = model.make_future_dataframe(periods=periods, freq=freq, include_history=False)
    forecast = model.predict(future)

    seasonalities = {}
    if 'daily'  in forecast.columns: seasonalities['daily']  = forecast['daily'].tolist()
    if 'weekly' in forecast.columns: seasonalities['weekly'] = forecast['weekly'].tolist()

    return {
        'dates':        forecast['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'forecast':     forecast['yhat'].tolist(),
        'trend':        forecast['trend'].tolist(),
        'seasonalities': seasonalities,
        'lower_bound':  forecast['yhat_lower'].tolist(),
        'upper_bound':  forecast['yhat_upper'].tolist(),
    }


def create_comparison_df(baseline, actual, baseline_upper, baseline_lower):
    """Builds a DataFrame with pct_change and uncertainty bounds."""
    df = pd.DataFrame({
        'baseline':       baseline,
        'actual':         actual,
        'baseline_upper': baseline_upper,
        'baseline_lower': baseline_lower,
    })
    df['pct_change'] = (df['actual'] - df['baseline'])       / df['baseline']
    df['pct_upper']  = (df['actual'] - df['baseline_lower']) / df['baseline_lower']
    df['pct_lower']  = (df['actual'] - df['baseline_upper']) / df['baseline_upper']
    return df


def process_impact_series(data, buffer=0, end_method='first_return', n_consecutive=3):
    """
    Detects and characterises the impact period (values below zero).

    Parameters
    ----------
    end_method : 'first_return' | 'consecutive' | 'end_of_series'
    """
    series = np.array(data)
    impact_indices = np.where(series < -buffer)[0]
    if len(impact_indices) == 0:
        return "No significant impact detected."

    start_idx      = impact_indices[0]
    last_normal    = max(start_idx - 1, 0)
    end_idx        = len(series) - 1
    status         = "Ongoing"

    if end_method == 'first_return':
        rec = np.where(series[start_idx:] >= 0)[0]
        if len(rec) > 0:
            end_idx, status = start_idx + rec[0], "Recovered"

    elif end_method == 'consecutive':
        count = 0
        for i in range(start_idx, len(series)):
            if series[i] >= 0:
                count += 1
                if count == n_consecutive:
                    end_idx, status = i - (n_consecutive - 1), "Recovered"
                    break
            else:
                count = 0

    elif end_method == 'end_of_series':
        end_idx, status = len(series) - 1, "Fixed (End of Series)"

    return {
        "last_normal_point": {
            "index": int(last_normal),
            "value": float(series[last_normal]),
        },
        "impact_period": {
            "start_index": int(start_idx),
            "end_index":   int(end_idx),
            "duration":    int(end_idx - start_idx),
            "status":      status,
        },
    }


# ── Full pipelines ────────────────────────────────────────────────────────────

def run_temporal_decay_analysis_pipeline(
    flows, n_days_back, frequency, slots_per_day, rolling_window_size,
    show_plots, start_date='2021-04-15', output_dir='outputs', **kwargs,
):
    """
    Full decay analysis pipeline:
    1. Split pre/post impact.
    2. Decompose baseline with Prophet.
    3. Forecast expected behaviour.
    4. Compute % change from baseline.
    5. Identify impact period.
    6. Fit StepWise recovery model.

    Returns dict: gamma_model, decay_series, prophet_results,
                  comparison_df, impact_metadata.
    """
    from utils_pattern_analysis.visualization import (
        vis_line_total_flows, vis_line_pct_change, vis_line_recovery_fit,
    )

    split = slots_per_day * n_days_back
    flows_pre, flows_post = flows[:-split], flows[-split:]

    dec = decompose_with_prophet(flows_pre, start_date=start_date, freq=frequency)
    if show_plots:
        for key in ('trend', 'noise', 'clean_ts'):
            vis_line_total_flows(dec[key], output_dir=output_dir, tag=f'_prophet_{key}')
        for s_key, s in dec['seasonalities'].items():
            vis_line_total_flows(s, output_dir=output_dir, tag=f'_prophet_seasonality_{s_key}')

    future = forecast_with_prophet(dec['model'], periods=split, freq=frequency)
    if show_plots:
        vis_line_total_flows(dec['clean_ts'] + future['forecast'],
                             output_dir=output_dir, tag='_prophet_with_forecast')

    df_cmp = create_comparison_df(
        future['forecast'], flows_post, future['upper_bound'], future['lower_bound']
    )
    if show_plots:
        vis_line_pct_change(df_cmp, window=rolling_window_size, output_dir=output_dir)

    df_mv = df_cmp.copy()
    for col in ('pct_upper', 'pct_change'):
        df_mv[col] = df_mv[col].rolling(rolling_window_size, min_periods=rolling_window_size).mean()

    impact_params = {k: v for k, v in kwargs.items()
                     if k in ('end_method', 'n_consecutive', 'buffer')}
    metadata = process_impact_series(df_mv['pct_upper'].tolist(), **impact_params)

    s_idx = metadata["last_normal_point"]["index"]
    e_idx = metadata["impact_period"]["end_index"]
    decay_series = df_mv['pct_change'].tolist()[s_idx: e_idx + 1]

    fit = fit_stepwise_shape(decay_series)
    if fit:
        print("-" * 30)
        print(f"DECAY MODEL RESULTS (End Method: {metadata['impact_period']['status']})")
        print("-" * 30)
        vis_line_recovery_fit(decay_series, fit, output_dir=output_dir)

    return {
        "gamma_model":      fit,
        "decay_series":     decay_series,
        "prophet_results":  dec,
        "comparison_df":    df_mv,
        "impact_metadata":  metadata,
    }


def run_temporal_decay_analysis_pipeline_simplified(
    flows, n_days_back, frequency, slots_per_day, show_plots,
    start_date='2021-04-15', **kwargs,
):
    """
    Lighter version that returns the full de-baselined series (Actual - Baseline)
    without fitting a decay model. Used inside vis_line_temporal_patterns.
    """
    split = int(slots_per_day * n_days_back)
    flows_pre, flows_post = flows[:-split], flows[-split:]

    dec = decompose_with_prophet(flows_pre, start_date=start_date, freq=frequency)

    if show_plots:
        import matplotlib.pyplot as _plt
        from utils_pattern_analysis.visualization import vis_line_total_flows
        for key in ('trend', 'noise'):
            vis_line_total_flows(dec[key], output_dir='outputs', tag=f'_prophet_{key}')

    future = forecast_with_prophet(dec['model'], periods=split, freq=frequency)

    hist_daily  = np.array(dec['seasonalities'].get('daily',  0))
    hist_weekly = np.array(dec['seasonalities'].get('weekly', 0))
    baseline_pre  = (hist_daily + hist_weekly).tolist()
    baseline_post = future['forecast']
    full_baseline = baseline_pre + baseline_post

    if show_plots:
        import os as _os
        import matplotlib.pyplot as _plt
        _plt.figure(figsize=(12, 6))
        _plt.plot(list(flows), label='Actual', alpha=0.4, color='gray')
        _plt.plot(full_baseline, label='Baseline', color='red', linewidth=2)
        _plt.axvline(x=len(flows_pre), color='black', linestyle=':', label='Impact Start')
        _plt.legend(); _plt.tight_layout()
        _os.makedirs('outputs', exist_ok=True)
        _plt.savefig('outputs/flow_series_prophet_baseline_vs_actual.png', bbox_inches='tight', dpi=150)
        _plt.close()

    return [a - b for a, b in zip(list(flows), full_baseline)]
