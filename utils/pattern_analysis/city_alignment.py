"""Align decoded component forecasts to an independently predicted city loss."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar


def align_city_total_loss(component_losses, city_total_prediction, component_base,
                          component_loading, city_base, decode, tolerance=1e-5):
    """Search one common loss shift, decoding from the immutable original each time.

    Losses are day-equivalents across the entire recovery horizon. ``decode``
    takes a component loss vector and returns (curve DataFrame, levels, rates).
    This function receives no observed recovery values. Normal-period component
    baselines and H row sums restore daily flows, which are divided by the raw
    city's normal baseline. Solver bounds can make the scalar target infeasible;
    return the best evaluated candidate and an explicit residual/convergence flag.
    Never hide an unmatched target or worsen the unaligned scalar residual.
    """
    original = np.asarray(component_losses, dtype=float).copy()
    base = np.asarray(component_base, dtype=float)
    loading = np.asarray(component_loading, dtype=float)
    city = np.asarray(city_base, dtype=float)
    if (original.ndim != 1 or base.ndim != 2 or base.shape[1] != len(original)
            or loading.shape != original.shape or city.shape != (base.shape[0],)):
        raise ValueError('Incompatible loss, baseline or loading shapes')
    if (not all(np.isfinite(v).all() for v in (original, base, loading, city))
            or (base < 0).any() or (loading < 0).any() or (city <= 0).any()
            or not np.isfinite(city_total_prediction) or tolerance <= 0):
        raise ValueError('Alignment requires finite inputs and positive city baselines')
    evaluated = {}

    def candidate(delta):
        delta = float(delta)
        if delta not in evaluated:
            curves, levels, rates = decode(original + delta)
            values = np.asarray(curves, dtype=float)
            if values.shape != base.shape or not np.isfinite(values).all():
                raise ValueError('Curve decoder returned invalid predictions')
            city_curve = (values * base) @ loading / city
            loss = float(np.sum(1.0 - city_curve))
            evaluated[delta] = (loss - city_total_prediction, curves, levels, rates)
        return evaluated[delta]

    initial_gap = candidate(0.0)[0]
    if abs(initial_gap) > tolerance:
        low, high = -1.0, 1.0
        for _ in range(8):
            if candidate(low)[0] * candidate(high)[0] <= 0:
                break
            low *= 2
            high *= 2
        if candidate(low)[0] * candidate(high)[0] <= 0:
            candidate(brentq(lambda delta: candidate(delta)[0], low, high, xtol=1e-8))
        else:
            grid = np.linspace(low, high, 65)
            best = int(np.argmin([abs(candidate(delta)[0]) for delta in grid]))
            optimum = minimize_scalar(lambda delta: candidate(delta)[0] ** 2,
                bounds=(grid[max(0, best-1)], grid[min(64, best+1)]), method='bounded')
            candidate(optimum.x)
    chosen = min(evaluated, key=lambda delta: abs(evaluated[delta][0]))
    gap, curves, levels, rates = candidate(chosen)
    audit = dict(initial_loss_gap=float(initial_gap), delta=float(chosen),
                 final_loss_gap=float(gap), evaluations=len(evaluated),
                 aligned=bool(abs(gap) <= tolerance), tolerance=float(tolerance),
                 city_total_prediction=float(city_total_prediction),
                 final_curve_loss=float(city_total_prediction + gap))
    return original + chosen, curves, levels, rates, audit
