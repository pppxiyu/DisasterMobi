"""
Cross-city resilience transfer: leave-one-unit-out and pairwise generalisation
of the component resilience metrics from feature tables (cosine-kNN by default;
see cross_city_resilience).  The within-city Ridge that used to open this module
(fit_resilience_linear / run_city_resilience_linear, a partial-rank attribution
per unit) was retired 2026-08-04 together with the intra-city LOO folder — at
5-12 components per unit its LOO R² was negative throughout.
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
# NOTE: 8 guards a WITHIN-city Ridge, where the rows ARE the regression's whole
# sample.  The cross-city path fits on the pooled training units and uses the
# held unit's rows only to score, so it runs at a lower floor — its callers pass
# run_pattern_nmf.CROSS_CITY_MIN_ROWS explicitly.  A cross-city caller that
# forgets to pass it silently inherits 8 and drops every small unit.
RIDGE_ALPHAS = np.logspace(-3, 3, 13)     # ridge penalty grid (RidgeCV LOO-selects)


def cross_city_resilience(feats_by_city, res_cols, feature_cols,
                          alphas=RIDGE_ALPHAS, min_rows=MIN_ROWS, rank=True,
                          split=None, target_std='within_unit', level_feature_cols=(),
                          model='ridge', pooled_feature_cols=(),
                          rank_features=True, pooled_scale_cols=()):
    """
    Cross-city resilience generalisation driven by an explicit train/test split.

    FEATURES and the TARGET are standardized in one of two method-pinned frames,
    selected by `target_std` — the two paths are DISJOINT (they share no
    standardization step), so read them separately:
      * 'within_unit'  (the RANK / spearman path): EVERY feature is z-scored WITHIN
        its own city-event and the target likewise (Option A, level-robust); with
        rank=True the target is rank-transformed first, and the features too
        unless `rank_features=False` —
        absolute-level differences between disasters are normalised away and R²
        reflects only the within-unit shape.  `pooled_feature_cols` and
        `level_feature_cols` are IGNORED here: a rank only means "more than" WITHIN a
        city, so a pooled frame would fake a cross-city level ranks cannot carry.
      * 'pooled_train' (the RAW / pearson path): the columns in `pooled_feature_cols`
        and the target are standardized on the POOLED TRAIN statistics and the SAME
        applied to the test unit, so each city's absolute LEVEL survives (R² then also
        reflects level transfer; one train unit -> single-unit standardization).  The
        per-event CONSTANT `level_feature_cols` are standardized on the pooled train
        and APPENDED to X (within-unit standardization would zero a constant).  Any
        feature NOT in `pooled_feature_cols` keeps its within-unit z-score.  Under the
        production config every feature is pooled (POOLED_FEATURE_COLS == feature_cols),
        so this path is FULLY pooled and nothing stays within-unit but the target's
        per-metric shape — there is no shared within-unit "trunk" left.
    `model` selects the predictor (same features, same standardized target either way):
      * 'ridge'      -> RidgeCV (global linear; can extrapolate beyond the training y).
      * 'cosine_knn' -> Nadaraya-Watson with a COSINE kernel: each prediction is the
        cosine-similarity-weighted mean of the training targets (negative similarities
        clipped to 0).  Non-parametric/local; bounded to the training-target range
        (cannot extrapolate).
    Behaviour depends on `split` (lists of city-event codes):

      * train and test DISJOINT  -> TRANSFER: fit ONE RidgeCV on the pooled train
        units, predict each test unit.  r2_table columns = test codes.
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
    level_feature_cols = list(level_feature_cols)
    pooled_feature_cols = list(pooled_feature_cols)

    # ---- feature-column bookkeeping --------------------------------------------
    # RAW path only: the columns in `pooled_feature_cols` are standardized on the
    # pooled-train statistics; any remaining column keeps its within-unit z-score.
    # Under the production config every feature is pooled (POOLED_FEATURE_COLS ==
    # feature_cols), so `nonpooled_idx` is empty and the raw path is fully pooled.
    pidx = [feature_cols.index(c) for c in pooled_feature_cols]
    nonpooled_idx = [j for j in range(len(feature_cols)) if j not in pidx]
    # RANK path: columns that must keep a POOLED-TRAIN scale even though every
    # other feature is z-scored within its own unit.  A within-unit z divides out
    # any unit-level constant a column was multiplied by -- z_unit(s*x) ==
    # sign(s)*z_unit(x) exactly -- so a feature carrying a unit-level magnitude,
    # or an interaction between one and a component-level column, cannot survive
    # that frame.  Naming it here keeps it on the pooled scale instead.  The
    # statistics come from the TRAINING units only, so the held-out unit never
    # sets its own scale.
    psidx = [feature_cols.index(c) for c in pooled_scale_cols
             if c in feature_cols]

    def _prep(df, target):
        # Prepare ONE city-event: drop rows with a missing feature/target, skip a
        # unit with too few usable components or a constant target, and (rank mode)
        # rank-transform every feature column and the target WITHIN the unit.  The
        # feature matrix is returned PRE-standardization (ranked-or-raw); each path
        # below standardizes it itself.  L = the per-event LEVEL covariates (raw);
        # P = the pooled feature columns as RAW un-ranked values (the pooled-train
        # path fits on these — a Pearson design pools magnitudes, not ranks).
        sub = df[[target] + feature_cols + level_feature_cols].dropna()
        if len(sub) < min_rows or sub[target].nunique() < 2:
            return None
        X = sub[feature_cols].to_numpy(dtype=float)
        y = sub[target].to_numpy(dtype=float)
        if rank:
            # `rank_features=False` ranks the TARGET only: the channel still
            # predicts an ordering, but the predictors keep their raw shape
            # and are merely z-scored within the unit below.  Splitting the
            # two is what the caller's V0r recipe needs; both stay per-unit.
            if rank_features:
                X = np.column_stack([rankdata(X[:, j])
                                     for j in range(X.shape[1])])
            y = rankdata(y)
        L = (sub[level_feature_cols].to_numpy(dtype=float) if level_feature_cols
             else None)
        P = (sub[pooled_feature_cols].to_numpy(dtype=float) if pooled_feature_cols
             else None)
        return X, y, sub.index.to_numpy(), L, P

    def _within_z(M):
        # z-score each column WITHIN one city-event (ddof=0, matching _zscore).
        return StandardScaler().fit_transform(M)

    def _zscore(y):
        sd = y.std()
        return (y - y.mean()) / (sd if sd > 0 else 1.0)

    def _unit_rows(M):
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.where(n > 0, n, 1.0)

    def _knn_transfer(Xtr, ytr, Xte):
        # Cosine-kernel Nadaraya-Watson: predict each test row as the cosine-similarity-
        # weighted mean of the training targets (negatives clipped; all <=0 -> mean).
        S = np.clip(_unit_rows(Xte) @ _unit_rows(Xtr).T, 0.0, None)
        denom = S.sum(axis=1)
        num = S @ ytr
        out = np.full(num.shape, float(np.mean(ytr)))
        ok = denom > 0
        out[ok] = num[ok] / denom[ok]
        return out

    def _knn_loo(X, y):
        # Leave-one-out cosine-kNN: each point = cosine-weighted mean of the OTHERS.
        S = np.clip(_unit_rows(X) @ _unit_rows(X).T, 0.0, None)
        np.fill_diagonal(S, 0.0)
        denom = S.sum(axis=1)
        num = S @ y
        out = np.full(num.shape, float(np.mean(y)))
        ok = denom > 0
        out[ok] = num[ok] / denom[ok]
        return out

    # ---- the two method-pinned standardization PATHS ---------------------------
    # RANK path  (spearman, target_std='within_unit'): every feature z-scored WITHIN
    #   its city-event, the target likewise; pooled/level machinery inert.
    # RAW path   (pearson,  target_std='pooled_train'): pooled features + target
    #   standardized on the POOLED TRAIN stats (each city's LEVEL survives), the
    #   per-event LEVEL covariate appended, any non-pooled feature kept within-unit.
    def _stdz_transfer(tr_parts, te):
        Xte_raw, yte_raw, _, Lte, Pte = te
        if target_std == 'within_unit':
            Xtr = np.vstack([_within_z(p[0]) for p in tr_parts])
            ytr = np.concatenate([_zscore(p[1]) for p in tr_parts])
            Xte = _within_z(Xte_raw)
            if psidx:
                Rtr = np.vstack([p[0] for p in tr_parts])[:, psidx]
                mu, sd = Rtr.mean(axis=0), Rtr.std(axis=0)
                sd[sd == 0] = 1.0
                Xtr[:, psidx] = (Rtr - mu) / sd
                Xte[:, psidx] = (Xte_raw[:, psidx] - mu) / sd
            return Xtr, ytr, Xte, _zscore(yte_raw)
        # pooled_train (RAW path)
        Xtr = np.vstack([p[0] for p in tr_parts]).astype(float)
        Xte = Xte_raw.astype(float).copy()
        if nonpooled_idx:                            # non-pooled cols -> within-unit z
            Xtr[:, nonpooled_idx] = np.vstack(
                [_within_z(p[0][:, nonpooled_idx]) for p in tr_parts])
            Xte[:, nonpooled_idx] = _within_z(Xte_raw[:, nonpooled_idx])
        if pidx:                                     # pooled cols -> pooled-train z
            Ptr = np.vstack([p[4] for p in tr_parts])
            muP, sdP = Ptr.mean(axis=0), Ptr.std(axis=0)
            sdP[sdP == 0] = 1.0
            Xtr[:, pidx] = (Ptr - muP) / sdP
            Xte[:, pidx] = (Pte - muP) / sdP
        ytr_raw = np.concatenate([p[1] for p in tr_parts])   # target -> pooled-train z
        mu, sd = float(ytr_raw.mean()), float(ytr_raw.std())
        sd = sd if sd > 0 else 1.0
        ytr, yte = (ytr_raw - mu) / sd, (yte_raw - mu) / sd
        if level_feature_cols:                       # LEVEL covariate -> appended
            Ltr = np.vstack([p[3] for p in tr_parts])
            muL, sdL = Ltr.mean(axis=0), Ltr.std(axis=0)
            sdL[sdL == 0] = 1.0
            Xtr = np.hstack([Xtr, (Ltr - muL) / sdL])
            Xte = np.hstack([Xte, (Lte - muL) / sdL])
        return Xtr, ytr, Xte, yte

    def _stdz_pooled_loo(parts):
        # train == test: the "pooled train" is the WHOLE set.  LEVEL covariates are
        # NOT appended here (this branch is the within-set diagnostic, never the
        # cross-city predictor).
        if target_std == 'within_unit':
            X = np.vstack([_within_z(p[0]) for p in parts])
            y = np.concatenate([_zscore(p[1]) for p in parts])
            if psidx:
                R = np.vstack([p[0] for p in parts])[:, psidx]
                mu, sd = R.mean(axis=0), R.std(axis=0)
                sd[sd == 0] = 1.0
                X[:, psidx] = (R - mu) / sd
            return X, y
        X = np.vstack([p[0] for p in parts]).astype(float)
        if nonpooled_idx:
            X[:, nonpooled_idx] = np.vstack(
                [_within_z(p[0][:, nonpooled_idx]) for p in parts])
        if pidx:
            Praw = np.vstack([p[4] for p in parts])
            muP, sdP = Praw.mean(axis=0), Praw.std(axis=0)
            sdP[sdP == 0] = 1.0
            X[:, pidx] = (Praw - muP) / sdP
        y = _zscore(np.concatenate([p[1] for p in parts]))
        return X, y

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
            named = [(c, _prep(feats_by_city[c], m)) for c in train]
            named = [(c, p) for c, p in named if p is not None]
            parts = [p for _, p in named]
            if not parts or sum(p[0].shape[0] for p in parts) < min_rows:
                r2[col][m], pred[col][m], groups[col][m] = np.nan, None, None
                continue
            cidx = np.concatenate([p[2] for p in parts])
            unit = np.concatenate([np.array([c] * p[0].shape[0]) for c, p in named])
            X, y = _stdz_pooled_loo(parts)
            if model == 'cosine_knn':
                y_pred = _knn_loo(X, y)
            else:
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
                Xtr, ytr, Xte, yte = _stdz_transfer(tr_parts, te)
                cidx = te[2]
                if model == 'cosine_knn':
                    ypred = _knn_transfer(Xtr, ytr, Xte)
                else:
                    ypred = RidgeCV(alphas=alphas).fit(Xtr, ytr).predict(Xte)
                r2[b][m] = float(r2_score(yte, ypred))
                pred[b][m] = (yte, ypred, cidx)
                groups[b][m] = None
    r2_table = pd.DataFrame(r2).reindex(index=res_cols)
    return r2_table, pred, groups
