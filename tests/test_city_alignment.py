"""Checks for observed-curve coherence and forecast-only scalar alignment."""
import unittest

import numpy as np
import pandas as pd

from utils.pattern_analysis.component_features import (
    daily_baselines, resilience_curves, resilience_features, recovery_curve_features,
)
from utils.pattern_analysis.city_alignment import align_city_total_loss


class CoherentResponseTests(unittest.TestCase):
    def setUp(self):
        normal = np.array([[10, 3]] * 5 + [[2, 9]] * 2, dtype=float)
        disaster = np.array([[2, 8], [4, 7], [8, 6], [12, 4], [3, 3], [5, 12],
                             [6, 15]], dtype=float)
        self.W = np.vstack([normal, disaster])
        self.loading = np.array([2., 5.])
        self.city = self.W @ self.loading + 7
        self.kw = dict(component_loading=self.loading, city_flow=self.city)

    def curve(self, **kwargs):
        return resilience_curves(self.W, 7, 'Monday', 1, **kwargs)

    def test_coherence_for_both_smoothing_operators(self):
        base = daily_baselines(self.W, 7, 'Monday', 1).to_numpy()
        cb = daily_baselines(self.city[:, None], 7, 'Monday', 1).to_numpy()[:, 0]
        nmf = self.W[7:] @ self.loading
        for smoothing in ('ratio_of_sums', 'mean_of_ratios'):
            corrected = self.curve(smoothing=smoothing, **self.kw)
            if smoothing == 'ratio_of_sums':
                expected = (pd.Series(nmf).rolling(3, center=True, min_periods=1).sum()
                            / pd.Series(cb).rolling(3, center=True, min_periods=1).sum())
            else:
                expected = pd.Series(nmf/cb).rolling(3, center=True, min_periods=1).mean()
            np.testing.assert_allclose((corrected.to_numpy()*base)@self.loading/cb, expected)
            before = self.curve(smoothing=smoothing).to_numpy()
            factors = corrected.to_numpy()/before
            np.testing.assert_allclose(factors[:, 0], factors[:, 1])

    def test_raw_city_future_is_not_a_correction_target(self):
        changed = self.city.copy()
        changed[7:] *= 100
        np.testing.assert_array_equal(self.curve(**self.kw), self.curve(
            component_loading=self.loading, city_flow=changed))
        np.testing.assert_array_equal(self.curve(smooth=1), self.curve(smooth=1, **self.kw))

    def test_training_target_and_oracle_use_corrected_curves(self):
        expected = self.curve(**self.kw)
        feats = resilience_features(self.W, 7, 'Monday', 1, **self.kw)
        np.testing.assert_allclose(feats.cum_loss, (1-expected).sum())
        # The fit consumer must forward the same correction arguments.
        from unittest.mock import patch
        with patch('utils.pattern_analysis.component_features.resilience_curves',
                   return_value=expected) as mock:
            recovery_curve_features(self.W, 7, 'Monday', 1, **self.kw)
        self.assertIs(mock.call_args.kwargs['component_loading'], self.loading)
        self.assertIs(mock.call_args.kwargs['city_flow'], self.city)

    def test_zero_flow_window_and_invalid_context(self):
        self.W[7:] = 0
        np.testing.assert_array_equal(self.curve(**self.kw), np.zeros((7, 2)))
        with self.assertRaises(ValueError):
            self.curve(component_loading=self.loading)
        with self.assertRaises(ValueError):
            self.curve(component_loading=[1], city_flow=self.city)

    def test_subdaily_context_matches_daily_definition(self):
        slots = np.repeat(self.W/2, 2, axis=0)
        city_slots = np.repeat(self.city/2, 2)
        actual = resilience_curves(slots, 14, 'Monday', 2,
            component_loading=self.loading, city_flow=city_slots)
        np.testing.assert_allclose(actual, self.curve(**self.kw))

    def test_production_context_switch_has_no_hidden_state(self):
        import run_pattern_nmf as rpm
        from unittest.mock import patch
        H = self.loading[:, None]
        X = self.city[None, :]
        with patch.object(rpm, 'COMPONENT_AGGREGATE_COHERENCE', False):
            self.assertEqual(rpm._component_curve_kwargs(H, X), {})
        with patch.object(rpm, 'COMPONENT_AGGREGATE_COHERENCE', True):
            kw = rpm._component_curve_kwargs(H, X)
            np.testing.assert_array_equal(kw['component_loading'], self.loading)
            np.testing.assert_array_equal(kw['city_flow'], self.city)


class ForecastAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.loss = np.array([1., 2.])
        self.base = np.array([[2., 8.], [3., 7.], [6., 4.]])
        self.loading = np.array([1., 2.])
        self.city = np.array([20., 18., 16.])

    def decode(self, losses):
        return pd.DataFrame(np.tile(1-losses/3, (3, 1))), losses.copy(), None

    def call(self, target, decode=None):
        return align_city_total_loss(self.loss, target, self.base, self.loading,
                                     self.city, self.decode if decode is None else decode)

    def test_dynamic_identity_shift_and_no_mutation(self):
        before = self.loss.copy()
        losses, curves, levels, rates, audit = self.call(1.3)
        self.assertTrue(audit['aligned'])
        reconstructed_loss = np.sum(1-(curves.to_numpy()*self.base)@self.loading/self.city)
        self.assertAlmostEqual(reconstructed_loss, 1.3, places=8)
        np.testing.assert_array_equal(self.loss, before)
        np.testing.assert_allclose(losses-before, audit['delta'])
        np.testing.assert_array_equal(levels, losses)
        self.assertIsNone(rates)
        np.testing.assert_allclose(np.diff(losses), np.diff(before))

    def test_already_aligned_returns_zero_shift(self):
        curves, _, _ = self.decode(self.loss)
        target = np.sum(1-(curves.to_numpy()*self.base)@self.loading/self.city)
        *_, audit = self.call(target)
        self.assertEqual(audit['delta'], 0)
        self.assertEqual(audit['evaluations'], 1)

    def test_infeasible_target_is_reported_not_claimed_matched(self):
        fixed = self.decode(self.loss)
        *_, audit = self.call(100, lambda losses: fixed)
        self.assertFalse(audit['aligned'])
        self.assertGreater(abs(audit['final_loss_gap']), 1)
        self.assertLessEqual(abs(audit['final_loss_gap']), abs(audit['initial_loss_gap']))

    def test_invalid_prediction_or_geometry_rejected(self):
        with self.assertRaises(ValueError):
            self.call(np.nan)
        with self.assertRaises(ValueError):
            self.call(1, lambda losses: (np.full((3, 2), np.nan), None, None))
        self.city[0] = 0
        with self.assertRaises(ValueError):
            self.call(1)


if __name__ == '__main__':
    unittest.main()
