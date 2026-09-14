"""Seven city inputs: six functional log variances and cross-fitted purpose SD.

The purpose learner predicts within-city centred cumulative loss from 15 raw
functional products. Its signed score SD is computed out of city. The city
scale correction retains the original backbone and KDE objective, without PCA.
"""
import itertools

import numpy as np
from scipy.optimize import minimize


FUNCTIONS = ('residential', 'commercial', 'leisure', 'industrial', 'health', 'public')
FEATURE_NAMES = tuple('log_variance_' + c for c in FUNCTIONS) + ('purpose_score_sd',)
RIDGE_GRID = np.logspace(-3, 3, 13)


class PurposeSpreadModel:
    """Dataset-scoped caches; all fitting subsets explicitly exclude validation."""

    def __init__(self, frames):
        self.x, self.y, self.function_variance = {}, {}, {}
        self.grid_cache, self.model_cache, self.scale_cache = {}, {}, {}
        self.audit, self.fit_records = [], []
        for code, frame in frames.items():
            f = np.column_stack([(frame['share_from_' + c] + frame['share_to_' + c])
                                 .to_numpy(float) for c in FUNCTIONS])
            if len(f) < 3 or not np.isfinite(f).all():
                continue
            products = np.column_stack([f[:, a] * f[:, b]
                                        for a, b in itertools.combinations(range(6), 2)])
            self.x[code] = products - products.mean(axis=0)
            self.function_variance[code] = np.log(np.diag(np.cov(f, rowvar=False, ddof=0)) + 1e-12)
            if 'cum_loss' in frame:
                y = frame.cum_loss.to_numpy(float)
                if np.isfinite(y).all():
                    self.y[code] = y - y.mean()

    def design(self, train):
        x = np.vstack([self.x[c] for c in train])
        y = np.concatenate([self.y[c] for c in train])
        weights = np.concatenate([np.full(len(self.x[c]), len(x) / (len(train) * len(self.x[c])))
                                  for c in train])
        sd = np.sqrt(np.average(x ** 2, axis=0, weights=weights))
        sd[sd < 1e-12] = 1.
        return x / sd, y, weights, sd

    def ridge_grid(self, train):
        train = tuple(train)
        if train not in self.grid_cache:
            x, y, weights, sd = self.design(train)
            eig, rotation = np.linalg.eigh(x.T @ (weights[:, None] * x))
            eig = np.maximum(eig, 0.)
            rhs = rotation.T @ (x.T @ (weights * y))
            beta = (rhs[None, :] / (eig[None, :] + RIDGE_GRID[:, None])) @ rotation.T
            self.grid_cache[train] = beta / sd
        return self.grid_cache[train]

    def purpose_fit(self, train):
        train = tuple(train)
        if train not in self.model_cache:
            errors = []
            for held in train:
                rest = tuple(c for c in train if c != held)
                pred = self.x[held] @ self.ridge_grid(rest).T
                errors.append(np.mean((pred - self.y[held][:, None]) ** 2, axis=0))
            scores = np.mean(errors, axis=0)
            best = int(np.argmin(scores))
            self.model_cache[train] = (self.ridge_grid(train)[best], float(RIDGE_GRID[best]))
        return self.model_cache[train]

    def context(self, train, held):
        train = tuple(train)
        if held in train:
            raise ValueError('Validation city must not appear in training cities')
        values = {}
        for code in train + (held,):
            eligible = tuple(c for c in train if c != code)
            beta, alpha = self.purpose_fit(eligible)
            purpose = float(np.std(self.x[code] @ beta, ddof=0))
            values[code] = np.r_[self.function_variance[code], purpose]
            self.audit.append(dict(target=code, meta_train=train, weight_train=eligible,
                                   ridge_alpha=alpha, coefficients=beta.tolist()))
        return values

    def fit_scale(self, train, sbb, vectors, h, lam, pg, outer, validation):
        matrix = np.array([vectors[c] for c in train])
        mean, sd = matrix.mean(axis=0), matrix.std(axis=0)
        sd[sd == 0] = 1.
        center = ((matrix - mean) / sd).mean(axis=0)

        def project(code):
            return (vectors[code] - mean) / sd - center

        design = np.array([project(c) for c in train])
        bary = np.mean([np.quantile(self.y[c], pg) for c in train], axis=0)
        bary -= bary.mean()
        xb = float(np.mean([np.log(sbb[c]) for c in train]))
        b0 = xb - np.log(np.mean([sbb[c] for c in train])) + np.log(bary.std())
        offset = b0 + np.array([np.log(sbb[c]) - xb for c in train])
        values = np.concatenate([self.y[c] for c in train])
        assignment = np.concatenate([np.full(len(self.y[c]), i) for i, c in enumerate(train)]).astype(int)

        def objective(gamma):
            with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
                sigma = np.exp(offset + design @ gamma)
                eps = values / sigma[assignment]
                centres = eps / eps.std()
                delta = eps[:, None] - centres[None, :]
                density = np.exp(-.5 * (delta / h) ** 2).sum(axis=1) / (len(centres) * h * np.sqrt(2 * np.pi))
                loss = np.sum(np.log(sigma[assignment])) - np.sum(np.log(np.maximum(density, 1e-300))) + lam * np.sum(gamma ** 2)
            return float(loss) if np.isfinite(loss) else 1e100

        key = (tuple(train), matrix.tobytes(), tuple(sbb[c] for c in train), h, lam, pg.tobytes())
        if key in self.scale_cache:
            gamma, record = self.scale_cache[key]
        else:
            zero = np.zeros(7)
            initial = objective(zero)
            fit = minimize(objective, zero, method='Nelder-Mead',
                           options=dict(maxiter=20000, maxfev=50000, xatol=1e-4, fatol=1e-7))
            retry = 0
            if not fit.success:
                retry = 1
                fit = minimize(objective, fit.x, method='Nelder-Mead',
                               options=dict(maxiter=100000, maxfev=250000, xatol=1e-4, fatol=1e-7))
            if not fit.success:
                raise RuntimeError(f'Spread fit did not converge: {outer}, {validation}, h={h}, penalty={lam}')
            gamma = fit.x.copy() if fit.fun <= initial else zero
            record = dict(converged=True, iterations=int(fit.nit), retried=retry,
                          objective_gain=initial - objective(gamma), n_features=7)
            self.scale_cache[key] = gamma, record
        self.fit_records.append(dict(outer_held=outer, validation=validation, train=list(train),
                                     bandwidth=h, penalty=lam, **record))
        return gamma, project, b0, xb, bary

    def multiplier(self, held, rest, sbb, pg, h_grid, penalty_grid):
        train = tuple(c for c in rest if c in self.x and c in self.y
                      and np.isfinite(sbb.get(c, np.nan)) and sbb[c] > 0)
        if held not in self.x or len(train) < 4:
            return 1., dict(code=held, fallback='insufficient complete training cities')
        contexts = {c: self.context(tuple(t for t in train if t != c), c) for c in train}
        final = self.context(train, held)
        best, selected, grid = None, None, []
        for h in h_grid:
            for lam in penalty_grid:
                errors = []
                for val in train:
                    tr = tuple(c for c in train if c != val)
                    gamma, project, b0, xb, bary = self.fit_scale(tr, sbb, contexts[val], h, lam, pg, held, val)
                    sigma = np.exp(b0 + np.log(sbb[val]) - xb + project(val) @ gamma)
                    curve = bary * (sigma / bary.std())
                    curve -= curve.mean()
                    errors.append(float(np.abs(curve - np.quantile(self.y[val], pg)).mean()))
                score = float(np.mean(errors))
                grid.append(dict(bandwidth=h, penalty=lam, mean_inner_W1=score))
                if best is None or score < best:
                    best, selected = score, (h, lam)
        gamma, project, _, _, _ = self.fit_scale(train, sbb, final, *selected, pg, held, 'final')
        mult = float(np.exp(project(held) @ gamma))
        info = dict(code=held, train=list(train), bandwidth=selected[0], penalty=selected[1],
                    inner_W1=best, gamma=gamma.tolist(), multiplier=mult, grid=grid,
                    feature_names=list(FEATURE_NAMES), PCA_used=False,
                    feature_rows=[dict(code=c, is_held=c == held, values=final[c].tolist()) for c in train + (held,)])
        return mult, info
