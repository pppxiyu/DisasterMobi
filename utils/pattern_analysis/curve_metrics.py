"""Shared city-curve reporting metrics; the reporting window does not train models."""

import numpy as np


def curve_error_metrics(observed, predicted, days, start_day=0):
    """Score finite paired days at or after start_day; MAPE is a fraction.

    MAE, normalized RMSE and R-squared share the same finite-pair mask.
    Only MAPE additionally excludes zero observed values. No observations
    return NaN metrics rather than a misleading perfect score.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    days = np.asarray(days)
    if observed.ndim != 1 or observed.shape != predicted.shape or days.shape != observed.shape:
        raise ValueError('City curves and days must be equal-length one-dimensional arrays')
    ok = np.isfinite(observed) & np.isfinite(predicted) & (days >= start_day)
    truth, estimate = observed[ok], predicted[ok]
    result = dict(mae=np.nan, nrmse=np.nan, r2=np.nan, mape=np.nan,
                  n_days=int(ok.sum()))
    if not len(truth):
        return result
    error = estimate - truth
    sd = float(np.std(truth))
    mse = float(np.mean(error ** 2))
    nonzero = truth != 0
    result.update(
        mae=float(np.mean(np.abs(error))),
        nrmse=float(np.sqrt(mse) / sd) if sd > 0 else np.nan,
        r2=float(1 - mse / sd ** 2) if sd > 0 else np.nan,
        mape=float(np.mean(np.abs(error[nonzero] / truth[nonzero])))
        if nonzero.any() else np.nan,
    )
    return result
