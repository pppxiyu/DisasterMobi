"""Recovery-definition and fixed-benchmark regression checks."""
import unittest

import numpy as np

from utils.pattern_analysis.component_features import (
    daily_baselines, resilience_curves, resilience_features,
)
from utils.pattern_analysis.curve_metrics import curve_error_metrics


class RecoveryCurveTests(unittest.TestCase):
    def setUp(self):
        # Normal Monday-Sunday, one buffer week, disaster Monday-Sunday.
        self.normal = [10, 10, 10, 10, 10, 2, 2]
        self.W = np.array(self.normal + [999] * 7 + [5, 10, 20, 10, 10, 1, 4],
                          dtype=float)[:, None]

    def curve(self, **kwargs):
        return resilience_curves(self.W, 7, 'Monday', 1, n_dis=14, **kwargs)

    def test_ratio_of_sums_including_calendar_transition_and_edges(self):
        expected = [15/20, 35/30, 40/30, 40/30, 21/22, 15/14, 5/4]
        np.testing.assert_allclose(self.curve().iloc[:, 0], expected)
        np.testing.assert_allclose(daily_baselines(
            self.W, 7, 'Monday', 1, n_dis=14).iloc[:, 0], self.normal)

    def test_historical_benchmark_is_mean_of_daily_ratios(self):
        raw = self.curve(smooth=1)
        np.testing.assert_allclose(self.curve(smoothing='mean_of_ratios'),
                                   raw.rolling(3, center=True, min_periods=1).mean())
        self.assertNotAlmostEqual(self.curve().iloc[4, 0],
                                   self.curve(smoothing='mean_of_ratios').iloc[4, 0])

    def test_raw_ratios_and_downstream_cumulative_loss(self):
        raw = [0.5, 1, 2, 1, 1, 0.5, 2]
        np.testing.assert_allclose(self.curve(smooth=1).iloc[:, 0], raw)
        features = resilience_features(self.W, 7, 'Monday', 1, n_dis=14)
        self.assertAlmostEqual(features['cum_loss'].iloc[0],
                               float((1-self.curve()).sum().iloc[0]))

    def test_same_baseline_reproduces_old_smoothing(self):
        self.W[:7] = 10
        np.testing.assert_allclose(self.curve(), self.curve(smoothing='mean_of_ratios'))

    def test_zero_baseline_and_slots(self):
        self.W[:7] = 0
        self.assertTrue(self.curve().isna().all().all())
        self.setUp()
        slots = np.repeat(self.W / 2, 2, axis=0)
        np.testing.assert_allclose(self.curve(), resilience_curves(
            slots, 14, 'Monday', 2, n_dis=28))

    def test_zero_single_day_baseline_uses_actual_window_flow(self):
        self.W[5:7] = 0
        # Friday-Sunday still has a positive window baseline and all raw flows.
        self.assertAlmostEqual(self.curve().iloc[5, 0], 15/10)
        self.assertTrue(np.isnan(self.curve().iloc[6, 0]))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            self.curve(smoothing='unknown')
        with self.assertRaises(ValueError):
            resilience_curves(self.W, 7, 'Monday', 2, n_dis=14)


class CityMetricTests(unittest.TestCase):
    def test_report_excludes_day_zero_without_changing_arrays(self):
        truth, prediction = np.array([1., 2., 4.]), np.array([100., 1., 6.])
        metrics = curve_error_metrics(truth, prediction, [0, 1, 2], start_day=1)
        self.assertEqual(metrics['n_days'], 2)
        self.assertAlmostEqual(metrics['mape'], .5)
        self.assertAlmostEqual(metrics['mae'], 1.5)
        self.assertAlmostEqual(metrics['r2'], -1.5)
        self.assertEqual(prediction[0], 100)

    def test_missing_zero_and_invalid_arrays(self):
        metrics = curve_error_metrics([0, 2, np.nan], [3, 1, 5], [0, 1, 2])
        self.assertEqual(metrics['n_days'], 2)
        self.assertAlmostEqual(metrics['mape'], .5)
        self.assertAlmostEqual(metrics['mae'], 2)
        self.assertTrue(np.isnan(curve_error_metrics([1], [1], [0], 1)['mape']))
        with self.assertRaises(ValueError):
            curve_error_metrics([1], [1, 2], [0])

    def test_plot_metrics_share_reporting_window_but_plot_all_days(self):
        from utils.pattern_analysis.visualization import vis_city_curves_grid
        import matplotlib.pyplot as plt
        figure = vis_city_curves_grid(
            {'example': (np.array([0, 1, 2]), np.array([1., 2., 4.]),
                         {'proposed method': np.array([100., 1., 6.])})},
            ncols=3, evaluation_start_day=1)
        try:
            self.assertIn('MAE 1.5', [text.get_text() for text in figure.axes[0].texts])
            legend = figure.axes[1].get_legend()
            self.assertTrue(any('MAPE 50.0%' in text.get_text()
                                for text in legend.get_texts()))
            np.testing.assert_array_equal(figure.axes[0].lines[1].get_xdata(), [0, 1, 2])
        finally:
            plt.close(figure)


if __name__ == '__main__':
    unittest.main()
