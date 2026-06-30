"""
Multivariate RANK (Ridge) attribution of resilience metrics onto a component's
features — the honest small-n counterpart to the pairwise Spearman block, and
ALIGNED with it: by default both the features and the target are RANK-transformed
before the fit, so the model learns rank associations.  (Spearman is exactly
Pearson on ranks; a single-feature rank fit with a tiny penalty reproduces
Spearman r.)

For each city (~20-25 NMF components) and each resilience metric, a ridge
regression predicts the (ranked) metric from the (ranked) features.  Unlike the
pairwise Spearman correlation, every feature enters ONE model together, so each
standardized coefficient is that feature's PARTIAL rank-association — its effect
after controlling for the others (sign = direction).  A leave-one-out R² reports
whether the model has any out-of-sample (rank) signal at all.  Pass rank=False
to regress on the raw metric VALUES instead.

Why ridge (not plain OLS): the merged functional features are compositional (sum
to a constant), so an OLS design is rank-deficient (collinear) and unstable;
ridge regularisation removes the singularity and damps small-n overfitting.
Why no SHAP: for a linear model SHAP(feature) = coef * (x - mean(x)), so the
standardized coefficient already IS the per-feature contribution (magnitude) and
direction — read it directly, no shap dependency.

Caveats (small n stays small): a passing LOO R² is a small-sample hint, not
proof; a failing model's coefficients only describe the in-sample fit.  Ranking
is global over the city's components (a mild optimism in the LOO check, like the
full-data alpha selection).  Compositional features are best read in groups.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score


MIN_ROWS     = 8                          # below this (after dropna) -> insufficient
RIDGE_ALPHAS = np.logspace(-3, 3, 13)     # ridge penalty grid (RidgeCV LOO-selects)


def fit_resilience_linear(df, target, feature_cols, alphas=RIDGE_ALPHAS,
                          min_rows=MIN_ROWS, rank=True):
    """
    Ridge regression of one resilience `target` on `feature_cols`, ONE city.

    rank=True (default): the features AND the target are rank-transformed
    (average ranks for ties) before the fit, so the coefficients are PARTIAL RANK
    associations — the multivariate generalisation of Spearman (rank=False
    regresses on the raw values instead).  Both features and target are z-scored,
    so a single-feature rank fit reproduces Spearman r exactly and the partial
    coefficients sit on a correlation-like scale.  Rows with NaN in target
    or any feature are dropped.  The ridge penalty alpha is chosen by RidgeCV
    (efficient leave-one-out internally); model quality is a SEPARATE leave-one-
    out R² at that alpha (each fold re-standardises on its own split), which can
    be negative (= worse than predicting the mean rank).

    Returns a dict: target, status ('ok' | 'insufficient_data'), n_used, alpha,
    loo_r2, passed (loo_r2 > 0), coef (Series feature -> standardized coefficient),
    plus y_true / y_pred (LOO predictions) / comp_index for the predicted-vs-actual
    scatter (None when insufficient_data).
    A degenerate case (too few rows, or a constant target) returns
    status='insufficient_data' with an all-zero coef — never raises, so the
    downstream heatmap/CSV still gets a full row.
    """
    feature_cols = list(feature_cols)
    sub = df[[target] + feature_cols].dropna()
    n_used = len(sub)

    base = dict(target=target, n_used=n_used, alpha=np.nan, loo_r2=np.nan,
                passed=False,
                coef=pd.Series(0.0, index=feature_cols, name=target),
                y_true=None, y_pred=None, comp_index=None)
    if n_used < min_rows or sub[target].nunique() < 2:
        return {**base, 'status': 'insufficient_data'}

    X = sub[feature_cols].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)

    # Rank-transform features and target (Spearman = Pearson on ranks) so the
    # model learns RANK associations, consistent with the Spearman block.  Ranks
    # are global over the city's components.
    if rank:
        X = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
        y = rankdata(y)

    # Standardize the target too (ddof=0, matching StandardScaler) so a single-
    # feature fit reproduces Spearman r exactly and the partial coefficients sit
    # on a correlation-like scale comparable to the Spearman heatmap.
    sy = y.std()
    y = (y - y.mean()) / (sy if sy > 0 else 1.0)

    ridge = RidgeCV(alphas=alphas).fit(StandardScaler().fit_transform(X), y)
    alpha = float(ridge.alpha_)

    # Honest generalisation check: standardiser refit inside each LOO fold.
    y_pred = cross_val_predict(
        make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        X, y, cv=LeaveOneOut(),
    )
    loo_r2 = float(r2_score(y, y_pred))

    coef = pd.Series(ridge.coef_, index=feature_cols, name=target)
    return dict(target=target, status='ok', n_used=n_used, alpha=alpha,
                loo_r2=loo_r2, passed=bool(loo_r2 > 0.0), coef=coef,
                y_true=y, y_pred=y_pred, comp_index=sub.index.to_numpy())


def run_city_resilience_linear(df_city, res_cols, feature_cols,
                               alphas=RIDGE_ALPHAS, min_rows=MIN_ROWS, rank=True):
    """
    Run fit_resilience_linear for every resilience metric of ONE city (rank-based
    by default — see fit_resilience_linear; pass rank=False for value regression).

    Returns
    -------
    coef_mat : DataFrame [res_cols x feature_cols]  standardized coefficients
               (rows = metrics, cols = the feature columns).
               Insufficient/failed rows are KEPT (all-zero coef row), never dropped.
    summary  : DataFrame indexed by res_cols, columns
               [n_used, alpha, loo_r2, passed, status].
    pred_data: dict res_col -> (y_true, y_pred_loo, comp_index) or None
               (LOO predicted-vs-actual on the ranked-standardized scale, for the
               regression scatter; None for insufficient_data metrics).
    """
    res_cols     = list(res_cols)
    feature_cols = list(feature_cols)
    results = [fit_resilience_linear(df_city, t, feature_cols,
                                     alphas=alphas, min_rows=min_rows, rank=rank)
               for t in res_cols]

    coef_mat = pd.DataFrame({r['target']: r['coef'] for r in results}).T
    coef_mat = coef_mat.reindex(index=res_cols, columns=feature_cols)
    summary = pd.DataFrame(
        [{k: r[k] for k in ('n_used', 'alpha', 'loo_r2', 'passed', 'status')}
         for r in results],
        index=pd.Index(res_cols, name='metric'),
    )
    pred_data = {r['target']: (None if r['y_true'] is None
                               else (r['y_true'], r['y_pred'], r['comp_index']))
                 for r in results}
    return coef_mat, summary, pred_data


def cross_city_resilience(feats_by_city, res_cols, feature_cols,
                          alphas=RIDGE_ALPHAS, min_rows=MIN_ROWS, rank=True,
                          split=None):
    """
    Cross-city resilience generalisation driven by an explicit train/test split.

    Each city-EVENT is rank/standardized WITHIN ITSELF (Option A, level-robust),
    so absolute-level differences between disasters are normalised away; the units
    are then POOLED.  Behaviour depends on `split` (lists of city-event codes):

      * train and test DISJOINT  -> TRANSFER: fit ONE RidgeCV on the pooled train
        units, predict each test unit.  r2_table columns = test codes.  R² measures
        how well the relation learned on the (unseen-test) train set transfers.
      * train and test the SAME set -> POOLED-LOO: pool the units' components and
        leave-one-component-out CV (RidgeCV picks alpha, then LeaveOneOut).
        r2_table has one column 'pooled_LOO'.  (One unit -> that unit's within-city LOO.)
      * partial overlap -> warns (leakage; not an intended usage), treated as transfer.
      * split is None -> warns and returns empty (caller should set CROSS_CITY_SPLIT).

    Returns
    -------
    r2_table : DataFrame [res_cols x columns]  test / pooled-LOO R².
    pred     : dict column -> {metric -> (y_true, y_pred, comp_index) | None}.
    groups   : dict column -> {metric -> array_of_city_event_per_point | None}.
               Non-None only for the pooled-LOO column (to colour points by unit).
    """
    res_cols     = list(res_cols)
    feature_cols = list(feature_cols)

    def _prep(df, target):
        sub = df[[target] + feature_cols].dropna()
        if len(sub) < min_rows or sub[target].nunique() < 2:
            return None
        X = sub[feature_cols].to_numpy(dtype=float)
        y = sub[target].to_numpy(dtype=float)
        if rank:
            X = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
            y = rankdata(y)
        sy = y.std()
        y = (y - y.mean()) / (sy if sy > 0 else 1.0)
        return StandardScaler().fit_transform(X), y, sub.index.to_numpy()

    if split is None:
        warnings.warn("cross_city_resilience: no train/test split (CROSS_CITY_SPLIT "
                      "is None); skipping the cross-city step.")
        return pd.DataFrame(index=res_cols), {}, {}

    present = set(feats_by_city)
    train = [c for c in split.get('train', []) if c in present]
    test  = [c for c in split.get('test', []) if c in present]
    missing = (set(split.get('train', [])) | set(split.get('test', []))) - present
    if missing:
        warnings.warn(f"cross_city_resilience: split codes not found and ignored: "
                      f"{sorted(missing)}")
    if not train or not test:
        warnings.warn(f"cross_city_resilience: empty train ({train}) or test ({test}) "
                      f"after filtering to available units; skipping.")
        return pd.DataFrame(index=res_cols), {}, {}

    pooled_loo = (set(train) == set(test))
    if not pooled_loo and (set(train) & set(test)):
        warnings.warn(f"cross_city_resilience: train/test partially overlap "
                      f"{sorted(set(train) & set(test))} (leakage); treating as transfer.")

    r2, pred, groups = {}, {}, {}

    if pooled_loo:
        col = 'pooled_LOO'
        r2[col], pred[col], groups[col] = {}, {}, {}
        for m in res_cols:
            parts = [(c, _prep(feats_by_city[c], m)) for c in train]
            parts = [(c, p) for c, p in parts if p is not None]
            if not parts or sum(p[0].shape[0] for _, p in parts) < min_rows:
                r2[col][m], pred[col][m], groups[col][m] = np.nan, None, None
                continue
            X = np.vstack([p[0] for _, p in parts])
            y = np.concatenate([p[1] for _, p in parts])
            cidx = np.concatenate([p[2] for _, p in parts])
            unit = np.concatenate([np.array([c] * p[0].shape[0]) for c, p in parts])
            alpha = float(RidgeCV(alphas=alphas).fit(X, y).alpha_)
            y_pred = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
                                       X, y, cv=LeaveOneOut())
            r2[col][m] = float(r2_score(y, y_pred))
            pred[col][m] = (y, y_pred, cidx)
            groups[col][m] = unit
    else:
        for b in test:                               # pooled train -> each test unit
            r2[b], pred[b], groups[b] = {}, {}, {}
            for m in res_cols:
                tr_parts = [p for p in (_prep(feats_by_city[t], m) for t in train)
                            if p is not None]
                te = _prep(feats_by_city[b], m)
                if not tr_parts or te is None:
                    r2[b][m], pred[b][m], groups[b][m] = np.nan, None, None
                    continue
                Xtr = np.vstack([p[0] for p in tr_parts])
                ytr = np.concatenate([p[1] for p in tr_parts])
                Xte, yte, cidx = te
                ridge = RidgeCV(alphas=alphas).fit(Xtr, ytr)
                ypred = ridge.predict(Xte)
                r2[b][m] = float(r2_score(yte, ypred))
                pred[b][m] = (yte, ypred, cidx)
                groups[b][m] = None
    r2_table = pd.DataFrame(r2).reindex(index=res_cols)
    return r2_table, pred, groups
