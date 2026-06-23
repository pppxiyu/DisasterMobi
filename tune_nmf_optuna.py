"""
Optuna hyperparameter search for the cross-city resilience prediction.

Objective (maximise): the CROSS-CITY test R2 of the `cum_loss` resilience metric,
averaged over the two transfer directions (train BR / test FM, and the reverse).
This is exactly the number cross_city_resilience() reports for cum_loss; the
tuned knobs all feed into it through the per-component feature table `feats`.

Tuned hyperparameters
---------------------
PER CITY (separate value for Baton Rouge and Fort Myers):
    N_BEHAVIORS   (k)            int
    FILTER_FACTOR (OD filter)    float
    L1_REG        (sparsity q)   float
    LAMBDA_CTX    (context pull)  float  (only when CONTEXT_AWARE is on)
SHARED across both cities:
    CONTEXT_AWARE                bool             (structural switch)
    FLOW_FEATURE_MODE            {'sum','outer'}  (only when CONTEXT_AWARE is on)
    LANDUSE_RESIDENTIAL_WEIGHT   float            (housing-unit <-> job scale)
    LANDUSE_IWF_SCALE            float            (TF-IWF distinctiveness exponent)
    LANDUSE_DOMINANT_THRESHOLD   float            (top share for a hard label, else 'Mix')
The three LANDUSE_* knobs reshape the on-the-fly land-use classification, which
feeds the func_<c> features ALWAYS (via the dominant-category O*D cross-tab) and
the context-Y target when CONTEXT_AWARE is on.  They are kept SHARED (a consistent
classification recipe across cities; also limits overfitting on the 2-city probe).

Design
------
This script does NOT modify the pipeline.  It imports run_pattern_nmf as a module
(read-only) and reuses its EXACT constants and helper functions, so the per-trial
factorisation + feature path is identical to the production run except for the
overridden knobs.  Everything that is NOT tuned (windows, buffers, fit segments,
first-day calendar, the 'tf_iwf' weighting mode, resilience-metric definitions, the
regression itself) stays at its run_pattern_nmf value.

The city-fixed inputs (geo, graphs, and the RAW SLD cache) are loaded ONCE; the
land-use CLASSIFICATION is now tuned, so it is rescored from that raw cache each
trial.  Each trial re-runs (rescore land-use) -> build_city_matrices -> decompose
-> features ->
cross_city_resilience.  The solver is deterministic (random_state=42 is hardcoded
in decomposition.py), so each hyperparameter configuration has a single,
reproducible score and Optuna's TPE sampler sees no seed noise.

Run (inside the disaster_mobi env)
----------------------------------
    python tune_nmf_optuna.py --trials 60
    python tune_nmf_optuna.py --trials 40 --baseline-only      # skip context (faster)
    python tune_nmf_optuna.py --trials 200 --timeout 3600 --study optuna_nmf.db
The best hyperparameters are printed at the end.
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


# ── Search ranges (edit here) ─────────────────────────────────────────────────
# k FLOOR is 15 (not the MIN_ROWS=8 minimum): with 7 features, k=8 makes the
# cross-city ridge fit ~8 rows against 7 predictors (near-interpolation), which
# the search exploits to inflate a single metric.  k>=15 keeps n >> p so the fit
# is non-degenerate.  Filter/L1/Lambda spans bracket the current config (FILTER
# 3/1, L1 0.5/0.5, LAMBDA 0.1/0.1).
K_RANGE      = (15, 25)        # N_BEHAVIORS per city (floor raised to avoid n~=p overfit)
FILTER_RANGE = (0.5, 5.0)      # FILTER_FACTOR per city
L1_RANGE     = (0.0, 2.0)      # L1_REG per city
LAMBDA_RANGE = (1e-3, 1.0)     # LAMBDA_CTX per city (log scale; context only)

# Land-use classification knobs (SHARED across both cities).  Defaults are
# RESIDENTIAL_WEIGHT=1.0, IWF_SCALE=1.0, DOMINANT_THRESHOLD=0.4.
RESIDENTIAL_WEIGHT_RANGE = (0.2, 5.0)   # log; housing-unit <-> job equivalence scale
IWF_SCALE_RANGE          = (0.0, 3.0)   # IWF exponent; 0 = no distinctiveness reweight
DOMINANT_THRESHOLD_RANGE = (0.2, 0.6)   # share needed for a hard label (6 cats, uniform~0.17)

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
                 residential_weight, iwf_scale, dominant_threshold):
    """Reproduce run_pattern_nmf.main()'s per-city `feats` assembly for ONE city
    with the trial's hyperparameters.  Returns the per-component feature table
    carrying the 6 merged func_<c> shares, mean_distance, and the 5 RES_COLS
    resilience targets (the cross-city regression inputs+targets)."""
    # Land-use classification (now tuned) — rescored from the raw cache each trial.
    # `landuse` (share_<cat> cols) feeds the context-Y target; `cat_lookup`
    # (dominant labels) feeds the func cross-tab.  weighting stays 'tf_iwf'.
    landuse = rp.load_city_landuse(
        cd['raw_csv'], residential_weight=residential_weight,
        weighting=rp.LANDUSE_WEIGHTING, iwf_scale=iwf_scale,
        dominant_threshold=dominant_threshold)
    cat_lookup = rp.category_lookup_from_landuse(landuse)

    X_all, n_nor, mapping = rp.build_city_matrices(
        cd['graphs'], spec['days_window'],
        spec['days_buffer'] + spec['days_disaster'], filter_factor)
    n_dis = n_nor + spec['days_buffer'] * rp.SLOTS_ACTIVE

    fit_cols = (None if set(spec['fit_segments']) == _ALL_SEGMENTS
                else rp.select_segment_columns(spec['fit_segments'], n_nor, n_dis,
                                               X_all.shape[1]))

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


def make_objective(cities, target, context_choices=(True, False)):
    """cities: dict code -> (city_fixed, spec).  target: resilience metric name.
    context_choices: the CONTEXT_AWARE values to search ((False,) = baseline only)."""
    def objective(trial):
        context = trial.suggest_categorical('context_aware', list(context_choices))
        flow_mode = (trial.suggest_categorical('flow_mode', ['sum', 'outer'])
                     if context else 'sum')
        # Land-use classification knobs — SHARED across cities (one recipe).
        landuse_hp = dict(
            residential_weight=trial.suggest_float('residential_weight',
                                                   *RESIDENTIAL_WEIGHT_RANGE, log=True),
            iwf_scale=trial.suggest_float('iwf_scale', *IWF_SCALE_RANGE),
            dominant_threshold=trial.suggest_float('dominant_threshold',
                                                   *DOMINANT_THRESHOLD_RANGE),
        )
        hp = {}
        for code in cities:
            hp[code] = dict(
                k=trial.suggest_int(f'k_{code}', *K_RANGE),
                filter_factor=trial.suggest_float(f'filter_{code}', *FILTER_RANGE),
                l1=trial.suggest_float(f'l1_{code}', *L1_RANGE),
                context=context, flow_mode=flow_mode,
                lambda_ctx=(trial.suggest_float(f'lambda_{code}', *LAMBDA_RANGE,
                                                log=True) if context else 0.0),
                **landuse_hp,
            )
        try:
            # Suppress the helpers' progress prints to keep the trial log clean
            # (exceptions still propagate; only stdout is muted).
            with contextlib.redirect_stdout(io.StringIO()):
                feats = {code: _build_feats(cf, sp, **hp[code])
                         for code, (cf, sp) in cities.items()}
                r2_table, _ = rp.cross_city_resilience(feats, rp.RES_COLS,
                                                       FEATURE_COLS)
        except Exception as e:                       # degenerate config -> bad score
            trial.set_user_attr('error', repr(e)[:200])
            return -1.0

        # Aggregate cross-city R2.  target='all' -> mean over ALL 5 RES_COLS (so
        # the search cannot trade the other metrics away for one); else the single
        # named metric.  Each metric x direction is clipped to [-1, 1] so one
        # catastrophic transfer cannot dominate; a NaN cell (too few rows after
        # dropna) counts as -1.  Per-metric means are stashed for the report.
        metrics = list(rp.RES_COLS) if target == 'all' else [target]
        all_vals = []
        for m in metrics:
            mv = []
            for d in r2_table.columns:               # 'BR->FM', 'FM->BR'
                v = r2_table.loc[m, d]
                v = -1.0 if (v is None or pd.isna(v)) else max(float(v), -1.0)
                mv.append(v)
            trial.set_user_attr(f'R2_{m}', float(np.mean(mv)))
            all_vals.extend(mv)
        return float(np.mean(all_vals))
    return objective


_ABBR = {'drop_depth': 'drop', 'early_collapse': 'earl', 'recovery_day': 'rcvD',
         'recovery_deficit': 'rcvF', 'cum_loss': 'cuml'}


def _print_callback(study, trial):
    a, p = trial.user_attrs, trial.params
    # Per-metric mean cross-city R2 (unambiguous 4-char abbrev for a compact line).
    ms = " ".join(f"{_ABBR.get(k[3:], k[3:][:4])}={v:+.2f}"
                  for k, v in a.items() if k.startswith('R2_'))
    val = trial.value if trial.value is not None else float('nan')
    ks = f"k=({p.get('k_BR')},{p.get('k_FM')})"
    print(f"[trial {trial.number:>3}] score={val:+.3f}  {ms}  "
          f"ctx={p.get('context_aware')} {ks}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trials', type=int, default=60, help='number of Optuna trials')
    ap.add_argument('--timeout', type=int, default=None, help='wall-clock seconds cap')
    ap.add_argument('--seed', type=int, default=42, help='TPE sampler seed')
    ap.add_argument('--target', default='all',
                    help="resilience metric to optimise, or 'all' = mean over the 5")
    ap.add_argument('--baseline-only', action='store_true',
                    help='force CONTEXT_AWARE=False (skip the slow context solver)')
    ap.add_argument('--study', default=None,
                    help='sqlite path to persist/resume the study, e.g. optuna_nmf.db')
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore')

    if args.target not in list(rp.RES_COLS) + ['all']:
        raise SystemExit(f"--target must be 'all' or one of {rp.RES_COLS}")

    print("Loading city-fixed inputs (geo, graphs, raw SLD cache) once ...")
    br = _load_city_fixed('Baton Rouge', 'Baton_Rouge', rp.BR_GRAPH_PATH,
                          rp.BR_ANALYSIS_DAYS, rp.BR_GEO_CSV)
    fm = _load_city_fixed('Fort Myers', 'Fort_Myers', rp.FM_GRAPH_PATH,
                          rp.FM_ANALYSIS_DAYS, rp.FM_GEO_CSV)
    cities = {
        'BR': (br, dict(days_window=rp.DAYS_WINDOW_BR, days_buffer=rp.DAYS_BUFFER_BR,
                        days_disaster=rp.DAYS_DISASTER_IN_WIN_BR,
                        fit_segments=rp.NMF_FIT_SEGMENTS_BR,
                        first_day_normal=rp.FIRST_DAY_BR_NORMAL)),
        'FM': (fm, dict(days_window=rp.DAYS_WINDOW_FM, days_buffer=rp.DAYS_BUFFER_FM,
                        days_disaster=rp.DAYS_DISASTER_IN_WIN_FM,
                        fit_segments=rp.NMF_FIT_SEGMENTS_FM,
                        first_day_normal=rp.FIRST_DAY_FM_NORMAL)),
    }

    objective = make_objective(cities, args.target,
                               context_choices=((False,) if args.baseline_only
                                                else (True, False)))

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(
        direction='maximize', sampler=sampler,
        study_name=f'nmf_crosscity_{args.target}',
        storage=(f'sqlite:///{args.study}' if args.study else None),
        load_if_exists=bool(args.study))

    obj_label = ('mean over all 5 metrics' if args.target == 'all'
                 else args.target)
    print(f"Optimising cross-city R2 [{obj_label}] (mean of both directions), "
          f"{args.trials} trials"
          f"{' [baseline only]' if args.baseline_only else ''} ...\n")
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                   callbacks=[_print_callback], show_progress_bar=False)

    # ── Report ────────────────────────────────────────────────────────────────
    bt = study.best_trial
    ctx = bt.params['context_aware']
    print("\n" + "=" * 72)
    print(f"BEST  —  objective [{obj_label}] cross-city R2 = {study.best_value:+.4f}")
    # Per-metric mean cross-city R2 of the best trial (watch for any collapse).
    pm = {k[3:]: v for k, v in bt.user_attrs.items() if k.startswith('R2_')}
    if pm:
        print("        per-metric mean R2:  "
              + "  ".join(f"{k}={v:+.3f}" for k, v in pm.items()))
    print("-" * 72)
    print(f"  CONTEXT_AWARE (both cities)      = {ctx}")
    if ctx:
        print(f"  FLOW_FEATURE_MODE (both)         = {bt.params['flow_mode']}")
    print(f"  LANDUSE_RESIDENTIAL_WEIGHT (both) = {bt.params['residential_weight']:.3f}")
    print(f"  LANDUSE_IWF_SCALE (both)          = {bt.params['iwf_scale']:.3f}")
    print(f"  LANDUSE_DOMINANT_THRESHOLD (both) = {bt.params['dominant_threshold']:.3f}")
    for code, name in (('BR', 'Baton Rouge'), ('FM', 'Fort Myers')):
        line = (f"  [{name:<11}] N_BEHAVIORS={bt.params[f'k_{code}']:>3}  "
                f"FILTER_FACTOR={bt.params[f'filter_{code}']:.3f}  "
                f"L1_REG={bt.params[f'l1_{code}']:.3f}")
        if ctx:
            line += f"  LAMBDA_CTX={bt.params[f'lambda_{code}']:.4f}"
        print(line)
    print("=" * 72)

    # Top-5 leaderboard for context.
    done = [t for t in study.trials if t.value is not None]
    top = sorted(done, key=lambda t: t.value, reverse=True)[:5]
    print("\nTop trials:")
    for t in top:
        p = t.params
        extra = (f" flow={p.get('flow_mode')}" if p.get('context_aware') else "")
        print(f"  #{t.number:>3}  {t.value:+.4f}  ctx={p['context_aware']}{extra}  "
              f"k=({p['k_BR']},{p['k_FM']})  filt=({p['filter_BR']:.2f},{p['filter_FM']:.2f})  "
              f"l1=({p['l1_BR']:.2f},{p['l1_FM']:.2f})  "
              f"lu(rw={p['residential_weight']:.2f},iwf={p['iwf_scale']:.2f},dom={p['dominant_threshold']:.2f})")
    print(f"\n({len(done)} completed trials)")


if __name__ == '__main__':
    main()
