"""Regression checks for the seven-input spread feature construction."""
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from utils.pattern_analysis.spread_prediction import FUNCTIONS, RIDGE_GRID, PurposeSpreadModel


class SpreadPredictionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(91)
        self.frames = {}
        for i in range(6):
            f = rng.dirichlet(np.ones(6), size=4 + i)
            frame = pd.DataFrame({'share_from_' + c: f[:, j] for j, c in enumerate(FUNCTIONS)})
            for j, c in enumerate(FUNCTIONS):
                frame['share_to_' + c] = f[:, j] * .6
            frame['cum_loss'] = 4 * f[:, 0] - 2 * f[:, 1] + rng.normal(0, .1, len(f)) + i
            self.frames[str(i)] = frame

    def test_seven_features_and_population_variance(self):
        model = PurposeSpreadModel(self.frames)
        features = model.context(('0', '1', '2', '3', '4'), '5')
        self.assertTrue(all(len(v) == 7 for v in features.values()))
        for code, frame in self.frames.items():
            f = np.column_stack([frame['share_from_' + c] + frame['share_to_' + c] for c in FUNCTIONS])
            np.testing.assert_allclose(features[code][:6], np.log(np.var(f, axis=0) + 1e-12))
        self.assertTrue(all(a['target'] not in a['weight_train'] for a in model.audit))

    def test_test_labels_cannot_change_any_context_feature(self):
        train = ('0', '1', '2', '3', '4')
        original = PurposeSpreadModel(self.frames).context(train, '5')
        changed = dict(self.frames)
        changed['5'] = self.frames['5'].drop(columns='cum_loss')
        rebuilt = PurposeSpreadModel(changed).context(train, '5')
        for code in original:
            np.testing.assert_array_equal(original[code], rebuilt[code])

    def test_training_city_own_feature_excludes_own_labels(self):
        train = ('0', '1', '2', '3', '4')
        original = PurposeSpreadModel(self.frames).context(train, '5')
        changed = {c: f.copy() for c, f in self.frames.items()}
        changed['0']['cum_loss'] *= -100
        rebuilt = PurposeSpreadModel(changed).context(train, '5')
        np.testing.assert_array_equal(original['0'], rebuilt['0'])

    def test_ridge_grid_matches_independent_solver_and_city_weights(self):
        model = PurposeSpreadModel(self.frames)
        train = ('0', '1', '2', '3', '4')
        x, y, weights, sd = model.design(train)
        start = 0
        for code in train:
            count = len(model.x[code])
            self.assertAlmostEqual(weights[start:start + count].sum(), len(x) / len(train))
            start += count
        for i in (0, 6, 12):
            fit = Ridge(alpha=RIDGE_GRID[i], fit_intercept=False, solver='svd').fit(x, y, sample_weight=weights)
            np.testing.assert_allclose(model.ridge_grid(train)[i], fit.coef_ / sd, atol=1e-8, rtol=1e-7)


if __name__ == '__main__':
    unittest.main()
