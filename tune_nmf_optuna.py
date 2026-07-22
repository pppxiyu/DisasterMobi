"""
Optuna hyperparameter search for the cross-city resilience prediction, tuned by
LEAVE-ONE-CITY-EVENT-OUT cross-validation WITHIN THE TRAINING SET.

Objective (MINIMISE city-level LOO cum_loss MAE; the study MAXIMISES -MAE): for each
held-out TRAINING city-event, reconstruct its CITY-level cum_loss and compare to the
total-activity-curve ground truth, averaged over the folds.  Which reconstruction is
optimised is chosen by `--method` (2026-07-07, config C):
  * 'component' -> M1 (decompose + component-wise): component cosine-kNN transfer, keep
    the standardized predictions, weight_normal-aggregate to a city score, then a
    1-param city-scale variance-match calibration learned NESTED-LOO on the rest.
  * 'city' (default) -> city-kNN (decompose + city-wise): cosine-kNN over the
    weight_normal-aggregated feature vector, predicting GT directly.
Both use config C: func labelled with a GLOBAL TF-IWF pooled over the training cities'
block groups at the trial's global IWF scale, pooled func + distance, pooled_train
target, model = CROSS_CITY_MODEL, held-out unit at k = K_LOO_TEST (=10).  The training
set is rp.CROSS_CITY_SPLIT['train']; the held-out TEST city-event (BR_Ida) is NEVER
loaded.  NOTE: the M1 objective's variance-match is estimated from ~3 cities -> noisy.

Tuned hyperparameters
---------------------
PER TRAINING CITY-EVENT:  n_behaviors (k) int, l1_reg (sparsity q) float.
GLOBAL:  GLOBAL_IWF_SCALE float — the exponent of the pooled (global) TF-IWF
    that makes func cross-city comparable (config C).
NOT tuned (fixed at the run_pattern_nmf value): landuse_weighting='tf_iwf',
filter_factor=0, context_aware=False, LANDUSE_RESIDENTIAL_WEIGHT,
LANDUSE_DOMINANT_THRESHOLD, windows/buffers, fit segments, first-day calendar,
resilience-metric definitions, and the transfer itself (CROSS_CITY_MODEL + paired
standardization + level/pooled feature columns + K_LOO_TEST).  The tuned k is a
TRAIN-ROLE parameter only (the held-out unit is always at k=10).

Design
------
This script does NOT modify the pipeline.  It imports run_pattern_nmf as a module
(read-only) and reuses its EXACT constants and helper functions, so the per-trial
factorisation + feature path is identical to the production run except for the
overridden knobs.  The city-fixed inputs (geo, graphs, RAW SLD cache) are loaded
ONCE per training city-event; the land-use classification (now partly tuned) is
rescored from that raw cache each trial.  Each trial re-runs (rescore land-use) ->
build_city_matrices -> decompose -> features -> leave-one-city-event-out CV.  The
solver is deterministic (random_state=42 hardcoded in decomposition.py), so each
configuration has a single reproducible score and TPE sees no seed noise.

Run (inside the disaster_mobi env)
----------------------------------
    python tune_nmf_optuna.py --method city --trials 150
    python tune_nmf_optuna.py --method component --trials 150
Each run PERSISTS to cache/optuna/<task>/<task>.db (task = nmf_cityMAE_<method>) and
RESUMES that study if it already exists.  --study overrides the path.  The best
per-city-event k/l1 + GLOBAL_IWF_SCALE are printed at the end for you to
fill back into the registry MANUALLY (BR_Ida, the test unit, is not tuned).
"""
import argparse
import contextlib
import io
import os
import warnings

import numpy as np
import pandas as pd
import optuna

# Read-only import: pulls run_pattern_nmf's constants and helpers WITHOUT running
# main() (guarded by __main__).  Reusing its objects guarantees the trial path
# matches production exactly; no constant is duplicated here.
import run_pattern_nmf as rp
from config import OPTUNA_CACHE_DIR as CACHE_DIR


# Every study is persisted under CACHE_DIR/<task>/<task>.db — i.e.
# .cache/optuna/<task>/<task>.db (.cache/ is gitignored) — so distinct fine-tuning
# tasks keep separate, resumable subfolders.  CACHE_DIR lives in config.py, shared
# with the GRU tuner (run_prediction_training.py), so both use the same cache root.


# ── Search ranges (edit here) ─────────────────────────────────────────────────
# k FLOOR lowered to 8 (= MIN_ROWS): the old floor of 15 was for the 2-city single-
# train probe where one held-out unit (~k rows) faced 7 features (n~=p near-
# interpolation).  The LOO-CV objective now POOLS the other training city-events per
# fold (n >> p), and the new cities are small (Lake Charles ~173 nodes), so a low k
# is admissible.  L1 span brackets the current config.  FILTER is NOT tuned —
# filtering is disabled (filter_factor=0) for every city-event, matching the NMF run.
K_RANGE      = (8, 25)         # n_behaviors per training city-event
L1_RANGE     = (0.0, 2.0)      # l1_reg per training city-event

# GLOBAL_IWF_SCALE search: the exponent of the GLOBAL (pooled-over-training-cities)
# TF-IWF that labels func for the cross-city prediction (config C).  weighting is FIXED
# 'tf_iwf'; residential_weight (1.0) and dominant_threshold (0.4) stay at the rp value.
IWF_SCALE_RANGE = (0.0, 3.0)   # IWF exponent; 0 = no distinctiveness reweight

FEATURE_COLS = ([f'func_{c}' for c in rp.SF_CATEGORIES]
                + ['mean_distance', 'median_income_combined'])
_ALL_SEGMENTS = {'normal', 'buffer', 'disaster'}


def _load_city_fixed(label, key, graph_path, analysis_days, geo_csv):
    """Load the per-city inputs that do NOT depend on any tuned knob: geometry,
    graphs, and the path to the RAW SLD cache.  The land-use CLASSIFICATION
    (shares + dominant labels) is now tuned (RESIDENTIAL_WEIGHT / IWF_SCALE /
    DOMINANT_THRESHOLD), so it is rescored from this raw cache per trial in
    _build_feats, not here."""
    gdf = rp.load_city_geo(label, rp.AGG_LEVEL, geo_csv)
    graphs = rp.load_graphs_trimmed(graph_path, analysis_days, rp.SLOT_PER_DAY,
                                    label=label)
    raw_csv = os.path.join(rp.SPACE_FUNCTION_DIR,
                           f'{key}_block_group_sld_raw.csv')
    # Ensure the raw POI/employment cache exists (present on disk -> instant).
    assert rp.ensure_city_landuse_raw(label, gdf['aggr_id'].tolist(),
                                      raw_csv) is not None
    # BG->income lookup for the median_income_combined predictor (not tuned; the ACS
    # cache is fixed).  Mirrors production's _build_cross_city_feats income source.
    inc_csv = os.path.join(rp.ACS_DATA_DIR,
                           f'{key}_block_group_acs_income_{rp.ACS_INCOME_YEAR}_raw.csv')
    rp.ensure_city_income_raw(label, gdf['aggr_id'].tolist(), inc_csv,
                              year=rp.ACS_INCOME_YEAR)
    income = rp.load_city_income(inc_csv)
    income_by_aggr = dict(zip(income['aggr_id'], income['median_household_income']))
    return dict(label=label, gdf=gdf, graphs=graphs, raw_csv=raw_csv,
                income_by_aggr=income_by_aggr)


def _build_feats(cd, spec, *, k, l1, cat_lookup):
    """Reproduce production's `_build_cross_city_feats` for ONE city at the trial's k / l1,
    labelling func with the passed GLOBAL-IWF `cat_lookup` (config C).  Returns the per-
    component feature table with the 6 merged func_<c> shares, mean_distance,
    median_income_combined, the RES_COLS target (cum_loss), weight_normal (city reconstruction
    weight), and the two LEVEL covariates hurricane_intensity and evac_level.  filter is
    OFF, context is OFF (both fixed) — the plain sklearn decomposition only."""
    X_all, n_nor, mapping = rp.build_city_matrices(
        cd['graphs'], spec['days_window'], spec['days_buffer'] + spec['days_disaster'], 0)
    n_dis = n_nor + spec['days_buffer'] * rp.SLOTS_ACTIVE
    fit_cols = (None if set(spec['fit_segments']) == _ALL_SEGMENTS
                else rp.select_segment_columns(spec['fit_segments'], n_nor, n_dis, X_all.shape[1]))
    # nndsvd needs k <= min(#fit time-slots, #OD pairs); clamp so the factorisation is valid.
    n_fit_cols = X_all.shape[1] if fit_cols is None else len(fit_cols)
    k = max(1, min(k, X_all.shape[0], n_fit_cols))
    W, H, weights = rp.decompose_city(X_all, k, l1_reg=l1, fit_time_cols=fit_cols)

    distances = rp.build_distance_array(mapping, cd['gdf'])
    sf = rp.spatial_features(H, distances)
    M, _ = rp.build_od_function_matrix(H, mapping, cat_lookup, rp.AXIS_CATEGORIES)
    income_array = rp.build_income_array(mapping, cd['income_by_aggr'], mode='combined')
    inc = rp.socioeconomic_features(H, income_array, name='median_income_combined')
    res = rp.resilience_features(W, n_nor, spec['first_day_normal'], rp.SLOTS_ACTIVE, n_dis=n_dis)
    feats = pd.concat([rp.functional_features(M, rp.AXIS_CATEGORIES), sf, inc, res], axis=1)
    feats.insert(0, 'city', cd['label'])
    for c in rp.SF_CATEGORIES:                          # merge outflow+inflow func shares
        feats[f'func_{c}'] = feats[f'share_from_{c}'] + feats[f'share_to_{c}']
    feats['weight_normal'] = W[:n_nor].sum(axis=0) * H.sum(axis=1)
    feats['hurricane_intensity'] = spec['ss_intensity']
    feats['evac_level'] = spec['evac_level']
    return feats


_BAD = -1e3   # objective floor (we MAXIMISE -MAE, so a degenerate config returns a huge -MAE)


def _fold_transfer(held, rest, feats, feats_test):
    """One component-level cosine-kNN transfer fold (production pearson path).  Returns the
    sigma-un-standardized weight_normal city prediction (`pcity`, legacy) and the aggregated
    STANDARDIZED score (`scity`, used by M1), or None."""
    fold = {held: feats_test[held]}
    fold.update({c: feats[c] for c in rest})
    # The tuner's objective only reads cum_loss, so the metric list is pinned
    # here instead of following rp.RES_COLS (whose extra metrics the tuner's
    # feature tables do not compute).
    _, pred, _ = rp.cross_city_resilience(
        fold, ['cum_loss'], FEATURE_COLS, rank=False,
        split={'train': rest, 'test': [held]},
        target_std=rp.CROSS_CITY_METHOD_STD['pearson'][0],
        level_feature_cols=rp.LEVEL_FEATURE_COLS, model=rp.CROSS_CITY_MODEL,
        pooled_feature_cols=rp.POOLED_FEATURE_COLS)
    pm = pred.get(held, {}).get('cum_loss')
    if pm is None:
        return None
    yte_std, ypred_std, cidx = pm
    yte_std = np.asarray(yte_std, float); ypred_std = np.asarray(ypred_std, float)
    rt = feats_test[held].loc[cidx, 'cum_loss'].to_numpy(float)
    s = float(np.std(yte_std)); sig = (float(np.std(rt)) / s) if s > 0 else 1.0
    mu = float(np.mean(rt) - sig * np.mean(yte_std)); pred_raw = ypred_std * sig + mu
    w = feats_test[held].loc[cidx, 'weight_normal'].to_numpy(float); wsum = w.sum()
    return dict(pcity=float((w * pred_raw).sum() / wsum) if wsum > 0 else float(pred_raw.mean()),
                scity=float((w * ypred_std).sum() / wsum) if wsum > 0 else float(ypred_std.mean()))


def _city_vec(feats):
    """weight_normal-weighted mean of FEATURE_COLS + intensity — the city-kNN feature vector."""
    w = feats['weight_normal'].to_numpy(float); sw = w.sum()
    w = w / sw if sw > 0 else np.full(len(w), 1.0 / len(w))
    return np.array([float(w @ feats[col].to_numpy(float)) for col in FEATURE_COLS]
                    + [float(feats['hurricane_intensity'].iloc[0])])


def _reconstruct_city(held, rest, feats, feats_test, gt, method):
    """City-level cum_loss prediction for the held city, EXACTLY as production reconstructs.
    method='city' -> city-kNN over aggregated features; 'component' -> M1 (component transfer
    + nested-LOO variance-match calibration on `rest`).  Returns None if degenerate."""
    if method == 'city':
        Xtr = np.vstack([_city_vec(feats[c]) for c in rest]); ytr = np.array([gt[c] for c in rest])
        mu_x, sd_x = Xtr.mean(0), Xtr.std(0); sd_x[sd_x == 0] = 1.0
        Ztr = (Xtr - mu_x) / sd_x; zte = (_city_vec(feats_test[held]) - mu_x) / sd_x
        u = lambda M: M / np.where(np.linalg.norm(M, axis=-1, keepdims=True) > 0,
                                   np.linalg.norm(M, axis=-1, keepdims=True), 1.0)
        sims = np.clip(u(Ztr) @ u(zte), 0.0, None)
        return float(sims @ ytr / sims.sum()) if sims.sum() > 0 else float(ytr.mean())
    outer = _fold_transfer(held, rest, feats, feats_test)          # method == 'component' (M1)
    if outer is None:
        return None
    s_in, gt_in = [], []
    for s2 in rest:                                                # nested LOO on the rest cities
        inn = _fold_transfer(s2, [c for c in rest if c != s2], feats, feats_test)
        if inn is not None:
            s_in.append(inn['scity']); gt_in.append(gt[s2])
    if len(s_in) < 2:
        return outer['pcity']
    s_in, gt_in = np.array(s_in), np.array(gt_in); sd_s = s_in.std()
    return float(gt_in.mean() + (outer['scity'] - s_in.mean())
                 * (gt_in.std() / (sd_s if sd_s > 0 else 1.0)))


def make_objective(cities, gt, method):
    """cities: {code -> (city_fixed, spec)} TRAINING set only.  gt: {code -> total-curve
    cum_loss} (trial-independent).  Objective MAXIMISES -MAE where MAE is the mean city-level
    leave-one-city-event-out error of the chosen reconstruction (`method`: 'component'=M1 /
    'city'=city-kNN), the global TF-IWF recipe at the trial's scale, pooled func."""
    codes = list(cities)

    def objective(trial):
        iwf_scale = trial.suggest_float('cross_city_iwf_scale', *IWF_SCALE_RANGE)
        hp = {code: dict(k=trial.suggest_int(f'k_{code}', *K_RANGE),
                         l1=trial.suggest_float(f'l1_{code}', *L1_RANGE)) for code in codes}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                # GLOBAL TF-IWF pooled over the training cities' block groups at this trial's
                # scale (config C recipe), shared cat lookup for every city.
                iwf = rp.pooled_iwf_weights([cf['raw_df'] for cf, _ in cities.values()],
                                            residential_weight=rp.LANDUSE_RESIDENTIAL_WEIGHT,
                                            iwf_scale=iwf_scale)
                luts = {code: rp.category_lookup_from_landuse(rp.classify_dominant_function(
                            cf['raw_df'], residential_weight=rp.LANDUSE_RESIDENTIAL_WEIGHT,
                            weighting='tf_iwf', dominant_threshold=rp.LANDUSE_DOMINANT_THRESHOLD,
                            iwf=iwf))
                        for code, (cf, sp) in cities.items()}
                feats = {code: _build_feats(cf, sp, k=hp[code]['k'], l1=hp[code]['l1'],
                                            cat_lookup=luts[code])
                         for code, (cf, sp) in cities.items()}
                feats_test = {code: _build_feats(cf, sp, k=rp.K_LOO_TEST, l1=hp[code]['l1'],
                                                 cat_lookup=luts[code])
                              for code, (cf, sp) in cities.items()}
                for c, f in feats.items():
                    trial.set_user_attr(f'keff_{c}', int(len(f)))
                errs = []
                for held in codes:                     # LOO over the training cities
                    rest = [c for c in codes if c != held]
                    pc = _reconstruct_city(held, rest, feats, feats_test, gt, method)
                    if pc is None:
                        return _BAD
                    errs.append(abs(pc - gt[held]))
        except Exception as e:
            trial.set_user_attr('error', repr(e)[:200])
            return _BAD
        mae = float(np.mean(errs))
        trial.set_user_attr('MAE', mae)
        return -mae                                    # maximise -> minimise city MAE
    return objective


def _print_callback(study, trial):
    p = trial.params
    mae = trial.user_attrs.get('MAE')
    mae_s = f"{mae:.3f}" if mae is not None else "  x  "
    iwf = p.get('cross_city_iwf_scale')
    print(f"[trial {trial.number:>3}] city-MAE={mae_s}  "
          f"iwf={iwf:.2f}" if iwf is not None else f"[trial {trial.number:>3}] city-MAE={mae_s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trials', type=int, default=200,
                    help='number of Optuna trials (5 cities x 2 per-city k/l1 + 2 '
                         'global = 12-dim space)')
    ap.add_argument('--timeout', type=int, default=None, help='wall-clock seconds cap')
    ap.add_argument('--seed', type=int, default=42, help='TPE sampler seed')
    ap.add_argument('--method', default='city', choices=['component', 'city'],
                    help="reconstruction to optimise the city-level cum_loss MAE of: "
                         "'component' = M1 (decompose + component-wise), "
                         "'city' = city-kNN (decompose + city-wise, default)")
    ap.add_argument('--study', default=None,
                    help='override the sqlite path (default: '
                         'cache/optuna/<task>/<task>.db, auto-created and resumed)')
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore')

    # Load ONLY the training set (CROSS_CITY_SPLIT['train']) from the registry; the
    # held-out TEST city-event (BR_Ida) is never touched by the tuner.
    reg = {c['code']: c for c in rp.CITY_EVENTS}
    train_codes = [c for c in rp.CROSS_CITY_SPLIT['train'] if c in reg]
    if len(train_codes) < 2:
        raise SystemExit(f"need >=2 training city-events for LOO-CV; got {train_codes}")
    print(f"Loading {len(train_codes)} training city-events (geo, graphs, raw SLD "
          f"cache) once: {', '.join(train_codes)}")
    cities, gt = {}, {}
    for code in train_codes:
        e = reg[code]
        cf = _load_city_fixed(e['label'], e['key'], e['graph'],
                              e['analysis_days'], e['geo'])
        cf['raw_df'] = pd.read_csv(cf['raw_csv'])          # for the global TF-IWF each trial
        spec = dict(days_window=e['window'], days_buffer=e['buffer'],
                    days_disaster=e['disaster'], fit_segments=e['fit_segments'],
                    first_day_normal=e['first_day_normal'], ss_intensity=e['ss_intensity'],
                    evac_level=e['evac_level'])
        cities[code] = (cf, spec)
        # Ground-truth city cum_loss = TOTAL-activity-curve cum_loss (no decomposition),
        # independent of k/l1/iwf, so compute it ONCE per city.
        X_all, n_nor, _ = rp.build_city_matrices(
            cf['graphs'], e['window'], e['buffer'] + e['disaster'], 0)
        n_dis = n_nor + e['buffer'] * rp.SLOTS_ACTIVE
        gt[code] = float(rp.resilience_features(
            X_all.sum(axis=0).reshape(-1, 1), n_nor, e['first_day_normal'],
            rp.SLOTS_ACTIVE, n_dis=n_dis)['cum_loss'].iloc[0])

    objective = make_objective(cities, gt, args.method)

    # Persist under cache/optuna/<task>/<task>.db; --study overrides.  Task name carries the
    # METHOD so the two reconstructions ('component'=M1, 'city'=city-kNN) keep separate,
    # resumable studies and never mix objectives.
    task = f'nmf_cityMAE_{args.method}'
    db_path = args.study or os.path.join(CACHE_DIR, task, f'{task}.db')
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    print(f"Study '{task}'  ->  {db_path}")

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(
        direction='maximize', sampler=sampler,          # maximises -MAE (i.e. minimises MAE)
        study_name=task, storage=f'sqlite:///{db_path}', load_if_exists=True)

    print(f"Leave-one-city-event-out city-level cum_loss MAE "
          f"[method={args.method} -> {'M1 component-wise' if args.method=='component' else 'city-kNN'}], "
          f"{len(train_codes)} folds, {args.trials} trials ...\n")
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                   callbacks=[_print_callback], show_progress_bar=False)

    # ── Report ────────────────────────────────────────────────────────────────
    bt = study.best_trial
    print("\n" + "=" * 72)
    print(f"BEST  —  method={args.method}  city-level LOO cum_loss MAE = {-study.best_value:.4f}")
    print("-" * 72)
    print(f"  GLOBAL  landuse_weighting = tf_iwf (fixed)   "
          f"GLOBAL_IWF_SCALE = {bt.params['cross_city_iwf_scale']:.3f}")
    print("  (n_behaviors is the EFFECTIVE k actually used — clamped to the city's OD-matrix "
          "size when\n   the suggested k was too large; fill THIS value into the registry.  "
          "filter_factor=0.)")
    for code in train_codes:
        ksug = bt.params[f'k_{code}']
        keff = bt.user_attrs.get(f'keff_{code}', ksug)
        ktag = f"{keff:>3}" + (f" (<-{ksug})" if keff != ksug else "      ")
        print(f"  [{code:<11}] n_behaviors={ktag}  l1_reg={bt.params[f'l1_{code}']:.3f}")
    print("  (BR_Ida = test unit, NOT tuned; keep its registry k/l1.)")
    print("=" * 72)

    # Top-5 leaderboard (k shown is the effective/clamped value; value = -MAE).
    done = [t for t in study.trials if t.value is not None and t.value > _BAD]
    top = sorted(done, key=lambda t: t.value, reverse=True)[:5]
    print("\nTop trials (city-MAE):")
    for t in top:
        p = t.params
        ks = "  ".join(f"{c}:k={t.user_attrs.get(f'keff_{c}', p[f'k_{c}'])},l1={p[f'l1_{c}']:.2f}"
                       for c in train_codes)
        print(f"  #{t.number:>3}  MAE={-t.value:.4f}  iwf={p['cross_city_iwf_scale']:.2f}  {ks}")
    print(f"\n({len(done)} non-degenerate trials of {len(study.trials)})")


if __name__ == '__main__':
    main()
