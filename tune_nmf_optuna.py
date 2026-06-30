"""
Optuna hyperparameter search for the cross-city resilience prediction, tuned by
LEAVE-ONE-CITY-EVENT-OUT cross-validation WITHIN THE TRAINING SET.

Objective (maximise): the mean leave-one-city-event-out Pearson cross-city R2.
The training set is rp.CROSS_CITY_SPLIT['train'] (read from the registry); the
held-out TEST city-event (BR_Ida) is NEVER loaded here.  For each fold one
training city-event is held out, the OTHER training city-events are pooled to fit
one RidgeCV, and that model predicts the held-out unit (this is exactly the
transfer mode of cross_city_resilience(split={'train': rest, 'test': [held]},
rank=False)).  The objective averages the held-out R2 over (folds x metrics);
`--target cum_loss` (default) optimises cum_loss only, `--target all` averages all
5 RES_COLS, else a single named metric.

Tuned hyperparameters
---------------------
PER TRAINING CITY-EVENT (separate value for each):
    n_behaviors   (k)            int
    l1_reg        (sparsity q)   float
GLOBAL (one shared value, the land-use classification recipe):
    LANDUSE_WEIGHTING            {'tf_iwf','raw_share'}
    LANDUSE_IWF_SCALE            float   (TF-IWF distinctiveness exponent)
NOT tuned (fixed at the run_pattern_nmf value): filter_factor=0 (NO OD filtering,
for every city-event), context_aware=False (so lambda_ctx and FLOW_FEATURE_MODE are
irrelevant and not searched), LANDUSE_RESIDENTIAL_WEIGHT, LANDUSE_DOMINANT_THRESHOLD,
windows/buffers, fit segments, first-day calendar, resilience-metric definitions,
and the regression itself.

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
    python tune_nmf_optuna.py --trials 200
    python tune_nmf_optuna.py --target cum_loss --trials 200 --timeout 3600
Each run PERSISTS to cache/optuna/<task>/<task>.db (task = nmf_loocv_<target>) and
RESUMES that study if it already exists.  --study overrides the path.  The best
per-city-event k/l1 + global weighting/iwf_scale are printed at the end for you to
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
# fold (n >> p), and the new cities are small (Panama City ~124 nodes), so a low k
# is admissible.  L1 span brackets the current config.  FILTER is NOT tuned —
# filtering is disabled (filter_factor=0) for every city-event, matching the NMF run.
K_RANGE      = (8, 25)         # n_behaviors per training city-event
L1_RANGE     = (0.0, 2.0)      # l1_reg per training city-event

# Land-use classification — tuned GLOBALLY (one shared recipe across all training
# city-events).  Only weighting + iwf_scale; residential_weight (default 1.0) and
# dominant_threshold (default 0.4) stay fixed at the run_pattern_nmf value.
IWF_SCALE_RANGE = (0.0, 3.0)   # IWF exponent; 0 = no distinctiveness reweight

FEATURE_COLS = [f'func_{c}' for c in rp.SF_CATEGORIES] + ['mean_distance']
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
    return dict(label=label, gdf=gdf, graphs=graphs, raw_csv=raw_csv)


def _build_feats(cd, spec, *, k, filter_factor, l1, context, lambda_ctx, flow_mode,
                 residential_weight, iwf_scale, dominant_threshold,
                 weighting=None):
    """Reproduce run_pattern_nmf.main()'s per-city `feats` assembly for ONE city
    with the trial's hyperparameters.  Returns the per-component feature table
    carrying the 6 merged func_<c> shares, mean_distance, and the 5 RES_COLS
    resilience targets (the cross-city regression inputs+targets)."""
    # Land-use classification (now tuned) — rescored from the raw cache each trial.
    # `landuse` (share_<cat> cols) feeds the context-Y target; `cat_lookup`
    # (dominant labels) feeds the func cross-tab.  `weighting` is now tuned globally
    # ('tf_iwf'/'raw_share'); None falls back to the run_pattern_nmf default.
    landuse = rp.load_city_landuse(
        cd['raw_csv'], residential_weight=residential_weight,
        weighting=weighting if weighting is not None else rp.LANDUSE_WEIGHTING,
        iwf_scale=iwf_scale, dominant_threshold=dominant_threshold)
    cat_lookup = rp.category_lookup_from_landuse(landuse)

    X_all, n_nor, mapping = rp.build_city_matrices(
        cd['graphs'], spec['days_window'],
        spec['days_buffer'] + spec['days_disaster'], filter_factor)
    n_dis = n_nor + spec['days_buffer'] * rp.SLOTS_ACTIVE

    fit_cols = (None if set(spec['fit_segments']) == _ALL_SEGMENTS
                else rp.select_segment_columns(spec['fit_segments'], n_nor, n_dis,
                                               X_all.shape[1]))

    # decompose_city factorises X_all.T (and, with fit_cols, only those time rows),
    # so NMF's nndsvd init requires k <= min(#fit time-slots, #OD pairs).  Small/
    # sparse cities have far fewer OD pairs than the searched k ceiling (Houma keeps
    # only ~13 OD pairs after filtering, fewer at higher filter), so clamp k down to
    # keep the factorisation valid.  The EFFECTIVE k is read back as len(feats).
    n_fit_cols = X_all.shape[1] if fit_cols is None else len(fit_cols)
    k = max(1, min(k, X_all.shape[0], n_fit_cols))

    if context:
        W, H, weights = rp.decompose_city_context(
            X_all, k, mapping, landuse, lambda_ctx=lambda_ctx,
            feature_mode=flow_mode, l1_reg=l1, fit_time_cols=fit_cols)
    else:
        W, H, weights = rp.decompose_city(
            X_all, k, l1_reg=l1, fit_time_cols=fit_cols)

    # Spatial (mean_distance) + functional (share_from/share_to) features.
    distances = rp.build_distance_array(mapping, cd['gdf'])
    sf = rp.spatial_features(H, distances)
    M, _ = rp.build_od_function_matrix(H, mapping, cat_lookup,
                                       rp.AXIS_CATEGORIES)

    # Resilience targets (the 5 RES_COLS, higher = worse, from resilience_features).
    res = rp.resilience_features(W, n_nor, spec['first_day_normal'],
                                 rp.SLOTS_ACTIVE, n_dis=n_dis)

    feats = pd.concat([rp.functional_features(M, rp.AXIS_CATEGORIES), sf, res],
                      axis=1)
    feats.insert(0, 'city', cd['label'])
    # Merge same-category outflow+inflow into one feature (as analysis_resilience_linear).
    for c in rp.SF_CATEGORIES:
        feats[f'func_{c}'] = feats[f'share_from_{c}'] + feats[f'share_to_{c}']
    return feats


def make_objective(cities, target):
    """cities: dict code -> (city_fixed, spec) for the TRAINING set only (the test
    city-event is never passed in).  target: a single resilience metric, or 'all'
    to average all RES_COLS.  The objective is the mean leave-one-city-event-out
    Pearson cross-city R2 over (folds x metrics)."""
    codes = list(cities)
    metrics = list(rp.RES_COLS) if target == 'all' else [target]

    def objective(trial):
        # GLOBAL land-use recipe (shared).  context_aware is FIXED False, so
        # flow_mode / lambda_ctx are irrelevant and not searched.
        weighting = trial.suggest_categorical('landuse_weighting',
                                              ['tf_iwf', 'raw_share'])
        iwf_scale = trial.suggest_float('iwf_scale', *IWF_SCALE_RANGE)
        # PER training city-event: k / l1.  filter is NOT tuned — filtering is OFF
        # (filter_factor=0) for every city, matching the NMF run.
        hp = {code: dict(k=trial.suggest_int(f'k_{code}', *K_RANGE),
                         l1=trial.suggest_float(f'l1_{code}', *L1_RANGE))
              for code in codes}

        # Per-metric R2 over the LOO folds; default -1 (a NaN/missing fold counts
        # as a failed transfer) so a degenerate config cannot win on missing data.
        fold_r2 = {m: [] for m in metrics}
        try:
            # Suppress the helpers' progress prints (exceptions still propagate).
            with contextlib.redirect_stdout(io.StringIO()):
                feats = {code: _build_feats(
                    cf, sp, k=hp[code]['k'], filter_factor=0,
                    l1=hp[code]['l1'], context=False, lambda_ctx=0.0, flow_mode='sum',
                    residential_weight=rp.LANDUSE_RESIDENTIAL_WEIGHT,
                    iwf_scale=iwf_scale, dominant_threshold=rp.LANDUSE_DOMINANT_THRESHOLD,
                    weighting=weighting)
                    for code, (cf, sp) in cities.items()}
                # Effective per-city k (= #components = rows of feats), which may be
                # below the suggested k_<code> when that city's OD matrix is small;
                # this is the value to fill into the registry, so stash it.
                for c, f in feats.items():
                    trial.set_user_attr(f'keff_{c}', int(len(f)))
                # Leave-one-city-event-out CV: pool the REST -> predict `held`.
                # Explicit per-city folds (transfer mode), NOT the pooled-component
                # mode (that would be leave-one-COMPONENT-out, not leave-one-city-out).
                for held in codes:
                    rest = [c for c in codes if c != held]
                    r2_table, _, _ = rp.cross_city_resilience(
                        feats, rp.RES_COLS, FEATURE_COLS, rank=False,
                        split={'train': rest, 'test': [held]})
                    for m in metrics:
                        v = (r2_table.loc[m, held]
                             if held in r2_table.columns else np.nan)
                        fold_r2[m].append(-1.0 if (v is None or pd.isna(v))
                                          else max(float(v), -1.0))
        except Exception as e:                       # degenerate config -> bad score
            trial.set_user_attr('error', repr(e)[:200])
            return -1.0

        # Per-metric mean over folds (stashed for the report); objective = grand
        # mean over all (fold x metric) cells.  target='all' averages all RES_COLS
        # so the search cannot trade the others away for one metric.
        all_vals = []
        for m in metrics:
            trial.set_user_attr(f'R2_{m}', float(np.mean(fold_r2[m])))
            all_vals.extend(fold_r2[m])
        return float(np.mean(all_vals))
    return objective


_ABBR = {'drop_depth': 'drop', 'early_collapse': 'earl', 'recovery_day': 'rcvD',
         'recovery_deficit': 'rcvF', 'cum_loss': 'cuml'}


def _print_callback(study, trial):
    a, p = trial.user_attrs, trial.params
    # Per-metric mean LOO-CV R2 (unambiguous 4-char abbrev for a compact line).
    ms = " ".join(f"{_ABBR.get(k[3:], k[3:][:4])}={v:+.2f}"
                  for k, v in a.items() if k.startswith('R2_'))
    val = trial.value if trial.value is not None else float('nan')
    iwf = p.get('iwf_scale')
    print(f"[trial {trial.number:>3}] LOO-CV R2={val:+.3f}  {ms}  "
          f"w={p.get('landuse_weighting')} iwf={iwf:.2f}"
          if iwf is not None else
          f"[trial {trial.number:>3}] LOO-CV R2={val:+.3f}  {ms}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trials', type=int, default=200,
                    help='number of Optuna trials (5 cities x 2 per-city k/l1 + 2 '
                         'global = 12-dim space)')
    ap.add_argument('--timeout', type=int, default=None, help='wall-clock seconds cap')
    ap.add_argument('--seed', type=int, default=42, help='TPE sampler seed')
    ap.add_argument('--target', default='cum_loss',
                    help="resilience metric to optimise (default cum_loss), or "
                         "'all' = mean over the 5")
    ap.add_argument('--study', default=None,
                    help='override the sqlite path (default: '
                         'cache/optuna/<task>/<task>.db, auto-created and resumed)')
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore')

    if args.target not in list(rp.RES_COLS) + ['all']:
        raise SystemExit(f"--target must be 'all' or one of {rp.RES_COLS}")

    # Load ONLY the training set (CROSS_CITY_SPLIT['train']) from the registry; the
    # held-out TEST city-event (BR_Ida) is never touched by the tuner.
    reg = {c['code']: c for c in rp.CITY_EVENTS}
    train_codes = [c for c in rp.CROSS_CITY_SPLIT['train'] if c in reg]
    if len(train_codes) < 2:
        raise SystemExit(f"need >=2 training city-events for LOO-CV; got {train_codes}")
    print(f"Loading {len(train_codes)} training city-events (geo, graphs, raw SLD "
          f"cache) once: {', '.join(train_codes)}")
    cities = {}
    for code in train_codes:
        e = reg[code]
        cf = _load_city_fixed(e['label'], e['key'], e['graph'],
                              e['analysis_days'], e['geo'])
        cities[code] = (cf, dict(days_window=e['window'], days_buffer=e['buffer'],
                                 days_disaster=e['disaster'],
                                 fit_segments=e['fit_segments'],
                                 first_day_normal=e['first_day_normal']))

    objective = make_objective(cities, args.target)

    # Persist under cache/optuna/<task>/<task>.db so each LOO-CV target gets its own
    # resumable subfolder; --study overrides the full path.
    task = f'nmf_loocv_{args.target}'
    db_path = args.study or os.path.join(CACHE_DIR, task, f'{task}.db')
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    print(f"Study '{task}'  ->  {db_path}")

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(
        direction='maximize', sampler=sampler,
        study_name=task,
        storage=f'sqlite:///{db_path}',
        load_if_exists=True)

    obj_label = ('mean over all 5 metrics' if args.target == 'all'
                 else args.target)
    print(f"Leave-one-city-event-out CV, Pearson cross-city R2 [{obj_label}], "
          f"{len(train_codes)} folds, {args.trials} trials ...\n")
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                   callbacks=[_print_callback], show_progress_bar=False)

    # ── Report ────────────────────────────────────────────────────────────────
    bt = study.best_trial
    print("\n" + "=" * 72)
    print(f"BEST  —  LOO-CV Pearson cross-city R2 [{obj_label}] = {study.best_value:+.4f}")
    # Per-metric mean LOO R2 of the best trial (watch for any collapse).
    pm = {k[3:]: v for k, v in bt.user_attrs.items() if k.startswith('R2_')}
    if pm:
        print("        per-metric mean LOO R2:  "
              + "  ".join(f"{k}={v:+.3f}" for k, v in pm.items()))
    print("-" * 72)
    print(f"  GLOBAL  LANDUSE_WEIGHTING = {bt.params['landuse_weighting']:<9}  "
          f"LANDUSE_IWF_SCALE = {bt.params['iwf_scale']:.3f}")
    print("  (n_behaviors is the EFFECTIVE k actually used — clamped to the city's "
          "OD-matrix size when the\n   suggested k was too large; fill this value "
          "into the registry.)")
    print("  (filtering is OFF for every city-event: filter_factor=0.)")
    for code in train_codes:
        ksug = bt.params[f'k_{code}']
        keff = bt.user_attrs.get(f'keff_{code}', ksug)
        ktag = f"{keff:>3}" + (f" (<-{ksug})" if keff != ksug else "      ")
        print(f"  [{code:<11}] n_behaviors={ktag}  "
              f"l1_reg={bt.params[f'l1_{code}']:.3f}")
    print("=" * 72)

    # Top-5 leaderboard for context (k shown is the effective/clamped value).
    done = [t for t in study.trials if t.value is not None]
    top = sorted(done, key=lambda t: t.value, reverse=True)[:5]
    print("\nTop trials:")
    for t in top:
        p = t.params
        ks = "  ".join(f"{c}:k={t.user_attrs.get(f'keff_{c}', p[f'k_{c}'])},"
                       f"l1={p[f'l1_{c}']:.2f}"
                       for c in train_codes)
        print(f"  #{t.number:>3}  {t.value:+.4f}  "
              f"w={p['landuse_weighting']} iwf={p['iwf_scale']:.2f}  {ks}")
    print(f"\n({len(done)} completed trials)")


if __name__ == '__main__':
    main()
