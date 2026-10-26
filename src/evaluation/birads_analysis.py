"""BIRADS in depth: per-category and binarised targets, pre-ultrasound information,
and the gain of the image embedding over each baseline.

Questions:
1. Does modelling each BIRADS category on its own pay off? Dedicated one-vs-rest
   binary models vs. the per-class ROC-AUC read off the multiclass model.
2. Does a binarised BIRADS make sense? Cuts >=4a, >=4b, >=4c (the dataset has no 4d),
   as dedicated binary models and as scores derived from the multiclass model
   (sum of P(class >= cut)) and from an ordinal regression on the category rank.
3. How is the predicted score distributed within each true category? (OOF scores
   saved to oof_predictions.csv; plotted by src/evaluation/birads_figures.py)
4. How well does information collected before the ultrasound (Age, Signs, Symptoms)
   predict BIRADS?
5. Does the embedding add to the baselines (acquisition, pre-US, both), and does
   pre-US add to the embedding? Paired corrected t-tests, as in significance_tests.py.

All targets, feature sets and methods share the same folds within a CV scheme: they
are built once from the multiclass BIRADS (BIRADS 1, 4 "normal" cases, dropped), so
every comparison is paired fold by fold, and the multiclass embedding / acquisition
runs reproduce acquisition_confound exactly.

Feature sets: acquisition, pre-US, both, the embedding (mae_cls, 768-d) alone and early-fused
with the baselines, and the embedding late-fused (its nested-CV out-of-fold score as a feature)
with pre-US and with acquisition + pre-US.

Outputs in outputs/embedding_to_tabular_experiments/{folder_name}/:
- results_folds.csv     per (target, method, feature_set, scheme, fold) metrics
- results.csv           mean/std over valid folds
- oof_predictions.csv   out-of-fold scores per case
- significance_tests.csv
- config.json

Usage:
    python -m src.evaluation.birads_analysis --folder-name birads_smoke --quick
    python -m src.evaluation.birads_analysis --folder-name birads_analysis
    python -m src.evaluation.birads_analysis --folder-name birads_analysis --retest
"""
import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.evaluation.acquisition_confound import make_folds
from src.evaluation.metadata_only_baseline import load_metadata
from src.evaluation.significance_tests import benjamini_hochberg, corrected_t
from src.models.embedding_to_clinical_shared import load_embeddings, safe_pr_auc, safe_roc_auc
from src.models.train_embedding_to_clinical_gbm import OUT_ROOT, make_models
from src.preprocessing.clinical_targets import load_raw_clinical

CLASSES = ['2', '3', '4a', '4b', '4c', '5']
CUTS = ['4a', '4b', '4c']
METADATA_COLS = ['width', 'height', 'pixel_size']
FEATURE_SETS = ['acquisition', 'pre_us', 'acquisition+pre_us',
                'embedding', 'embedding+pre_us', 'embedding+acquisition+pre_us']
# Late fusion: the embedding enters as its own out-of-fold score (nested CV inside the
# training fold), so 768 dims do not drown the ~25 pre-US columns in the tree splits.
STACKED_SETS = ['emb_score+pre_us', 'emb_score+acquisition+pre_us']
N_INNER = 5
SCHEMES = ['stratified', 'group_pixel_size']
ALPHA = 0.05

# (name, minuend, subtrahend): gain = minuend - subtrahend, paired by fold.
COMPARISONS = [
    ('pre_us vs dummy', 'pre_us', 'dummy'),
    ('acquisition+pre_us vs acquisition', 'acquisition+pre_us', 'acquisition'),
    ('acquisition+pre_us vs pre_us', 'acquisition+pre_us', 'pre_us'),
    ('embedding vs acquisition', 'embedding', 'acquisition'),
    ('embedding+pre_us vs pre_us', 'embedding+pre_us', 'pre_us'),
    ('embedding+pre_us vs embedding', 'embedding+pre_us', 'embedding'),
    ('embedding+acquisition+pre_us vs acquisition+pre_us', 'embedding+acquisition+pre_us', 'acquisition+pre_us'),
    ('emb_score+pre_us vs pre_us', 'emb_score+pre_us', 'pre_us'),
    ('emb_score+pre_us vs embedding', 'emb_score+pre_us', 'embedding'),
    ('emb_score+acquisition+pre_us vs acquisition+pre_us', 'emb_score+acquisition+pre_us', 'acquisition+pre_us'),
]


def binary_targets(birads: pd.Series) -> dict:
    rank = birads.map(CLASSES.index)
    targets = {f'BIRADS_is_{c}': (birads == c).astype(int) for c in CLASSES}
    targets.update({f'BIRADS_ge_{c}': (rank >= CLASSES.index(c)).astype(int) for c in CUTS})
    return targets


def pre_us_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Age (NaN when not available) plus one-hot Signs/Symptoms; 'not available' is
    kept as its own indicator, since missingness itself is batch-dependent."""
    age = pd.to_numeric(raw_df['Age'].where(raw_df['Age'] != 'not available'), errors='coerce')
    feats = pd.concat([age.rename('Age'),
                       raw_df['Signs'].str.get_dummies(sep='&').add_prefix('Signs_'),
                       raw_df['Symptoms'].str.get_dummies(sep='&').add_prefix('Symptoms_')], axis=1)
    feats.index = raw_df['Image_filename']
    return feats.astype(float)


def derived_scores(probas: np.ndarray) -> dict:
    """One-vs-rest and >=cut scores read off a multiclass probability matrix (columns = CLASSES)."""
    out = {f'BIRADS_is_{c}': probas[:, i] for i, c in enumerate(CLASSES)}
    out.update({f'BIRADS_ge_{c}': probas[:, CLASSES.index(c):].sum(axis=1) for c in CUTS})
    return out


def binary_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    p = np.column_stack([1 - s, s])
    return {'roc_auc': safe_roc_auc(y, p), 'pr_auc': safe_pr_auc(y, p)}


def fit_predict(kind: str, X_train, y_train, X_test, n_classes: int = len(CLASSES),
                model_name: str = 'catboost') -> np.ndarray:
    """Multiclass: probability matrix (n_classes columns); binary: P(positive); regression: prediction."""
    model = make_models(kind)[model_name].fit(X_train, y_train)
    if kind == 'regression':
        return model.predict(X_test)
    probas = model.predict_proba(X_test)
    if kind == 'binary':
        return probas[:, 1]
    # An inner split of a grouped fold may lack a class: keep one column per class anyway.
    full = np.zeros((len(X_test), n_classes))
    full[:, np.asarray(model.classes_).astype(int)] = probas
    return full


def stacked_fit_predict(kind: str, X_emb, X_side, y, train, test, n_classes: int = len(CLASSES),
                        model_name: str = 'catboost') -> np.ndarray:
    """Late fusion: the embedding model's score on the training rows comes from an inner
    CV (never from a model that saw the row), its score on the test rows from a model fit
    on the whole training fold; the same model is then fit on [embedding score, side features]."""
    inner = StratifiedKFold(N_INNER, shuffle=True, random_state=42)
    strat = y[train].astype(int)
    emb_train = None
    for itr, ite in inner.split(strat, strat):
        s = fit_predict(kind, X_emb[train][itr], y[train][itr], X_emb[train][ite], n_classes, model_name)
        if emb_train is None:
            emb_train = np.zeros((len(train),) + s.shape[1:])
        emb_train[ite] = s
    emb_test = fit_predict(kind, X_emb[train], y[train], X_emb[test], n_classes, model_name)
    stack = lambda e, side: np.column_stack([e.reshape(len(side), -1), side])
    return fit_predict(kind, stack(emb_train, X_side[train]), y[train], stack(emb_test, X_side[test]), n_classes,
                       model_name)


def run_set(feature_set: str, X: np.ndarray, rank: np.ndarray, cases: np.ndarray, folds: list,
            scheme: str, X_emb: np.ndarray = None) -> tuple:
    """Fits every target/method for one feature set on all folds. Returns (metric rows, oof rows).
    With X_emb given, X holds the side features and the embedding is late-fused."""
    predict = ((lambda kind, y, tr, te: stacked_fit_predict(kind, X_emb, X, y, tr, te)) if X_emb is not None
               else (lambda kind, y, tr, te: fit_predict(kind, X[tr], y[tr], X[te])))
    bins = {n: y.to_numpy() for n, y in binary_targets(pd.Series(np.array(CLASSES)[rank])).items()}
    rows, oof = [], []

    def add(target, method, fold, n_test, valid, **metrics):
        rows.append({'target': target, 'method': method, 'feature_set': feature_set, 'scheme': scheme,
                     'fold': fold, 'n_test': n_test, 'valid': valid, **metrics})

    for fold, (train, test) in enumerate(folds):
        base_oof = {'case': cases[test], 'birads': np.array(CLASSES)[rank[test]], 'feature_set': feature_set,
                    'scheme': scheme, 'fold': fold}

        valid = len(np.unique(rank[train])) == len(CLASSES) and len(np.unique(rank[test])) > 1
        if valid:
            probas = predict('multiclass', rank, train, test)
            add('BIRADS', 'multiclass', fold, len(test), True, roc_auc=safe_roc_auc(rank[test], probas),
                pr_auc=safe_pr_auc(rank[test], probas),
                balanced_accuracy=balanced_accuracy_score(rank[test], probas.argmax(1)))
            for name, s in derived_scores(probas).items():
                ok = len(np.unique(bins[name][test])) > 1
                add(name, 'from_multiclass', fold, len(test), ok, **(binary_metrics(bins[name][test], s) if ok else {}))
                oof.append(pd.DataFrame({**base_oof, 'target': name, 'method': 'from_multiclass', 'score': s}))
        else:
            add('BIRADS', 'multiclass', fold, len(test), False)

        s = predict('regression', rank.astype(float), train, test)
        add('BIRADS', 'ordinal', fold, len(test), True, spearman=spearmanr(rank[test], s).statistic)
        oof.append(pd.DataFrame({**base_oof, 'target': 'BIRADS', 'method': 'ordinal', 'score': s}))
        for name in [f'BIRADS_ge_{c}' for c in CUTS] + [f'BIRADS_is_{c}' for c in CLASSES]:
            ok = len(np.unique(bins[name][test])) > 1
            # A one-vs-rest score from a rank regression is |rank - class| (closer = more likely).
            score = s if name.startswith('BIRADS_ge_') else -np.abs(s - CLASSES.index(name.split('_')[-1]))
            add(name, 'from_ordinal', fold, len(test), ok, **(binary_metrics(bins[name][test], score) if ok else {}))

        for name, y in bins.items():
            ok = len(np.unique(y[train])) == 2 and len(np.unique(y[test])) > 1
            if not ok:
                add(name, 'dedicated', fold, len(test), False)
                continue
            s = predict('binary', y, train, test)
            add(name, 'dedicated', fold, len(test), True, **binary_metrics(y[test], s),
                balanced_accuracy=balanced_accuracy_score(y[test], (s >= 0.5).astype(int)))
            oof.append(pd.DataFrame({**base_oof, 'target': name, 'method': 'dedicated', 'score': s}))
    return rows, oof


def significance(folds_df: pd.DataFrame) -> pd.DataFrame:
    """Paired corrected t-tests on ROC-AUC (Spearman for the ordinal BIRADS).
    Families for BH: (comparison, scheme, primary) for feature-set comparisons; (method pair,
    feature set, scheme) for dedicated vs. derived."""
    df = folds_df.copy()
    df['metric'] = np.where(df['method'] == 'ordinal', df['spearman'], df['roc_auc'])
    df = df[df['valid']]
    n_total = folds_df.drop_duplicates(['scheme', 'fold']).groupby('scheme')['n_test'].sum()
    key = ['target', 'method', 'scheme', 'fold']
    wide = df.pivot_table(index=key + ['n_test'], columns='feature_set', values='metric').reset_index()
    # Dummy: ROC-AUC 0.5 (constant score), Spearman 0.
    wide['dummy'] = np.where(wide['method'] == 'ordinal', 0.0, 0.5)

    rows = []
    for (target, method, scheme), g in wide.groupby(['target', 'method', 'scheme']):
        for name, a, b in COMPARISONS:
            v = g.dropna(subset=[a, b])
            if len(v) < 2:
                continue
            rows.append({'family': name, 'target': target, 'method': method, 'scheme': scheme,
                         'a_mean': v[a].mean(), 'b_mean': v[b].mean(),
                         **corrected_t((v[a] - v[b]).to_numpy(), v['n_test'].to_numpy(float), int(n_total[scheme]))})

    # Dedicated binary vs. the same target derived from the multiclass / ordinal model, per feature set.
    by_method = df.pivot_table(index=['target', 'feature_set', 'scheme', 'fold', 'n_test'], columns='method',
                               values='metric').reset_index()
    for other in ['from_multiclass', 'from_ordinal']:
        for (target, fs, scheme), g in by_method.groupby(['target', 'feature_set', 'scheme']):
            if other not in g or 'dedicated' not in g:
                continue
            v = g.dropna(subset=['dedicated', other])
            if len(v) < 2:
                continue
            rows.append({'family': f'dedicated vs {other} | {fs}', 'target': target, 'method': f'dedicated - {other}',
                         'scheme': scheme, 'feature_set': fs, 'a_mean': v['dedicated'].mean(), 'b_mean': v[other].mean(),
                         **corrected_t((v['dedicated'] - v[other]).to_numpy(), v['n_test'].to_numpy(float),
                                       int(n_total[scheme]))})

    res = pd.DataFrame(rows)
    # The models fit directly (multiclass, ordinal, dedicated binaries) are the primary tests;
    # scores read off the multiclass/ordinal models are a separate, secondary BH family.
    res['primary'] = res['method'].isin(['multiclass', 'ordinal', 'dedicated'])
    res['q'] = res.groupby(['family', 'scheme', 'primary'], group_keys=False)['p'].apply(benjamini_hochberg)
    res['significant'] = res['q'] < ALPHA
    return res.sort_values(['family', 'scheme', 'p'])


def summarize(folds_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'spearman'] if m in folds_df]
    keys = ['target', 'method', 'feature_set', 'scheme']
    valid = folds_df[folds_df['valid']]
    g = valid.groupby(keys)[metrics]
    res = g.mean().add_suffix('_mean').join(g.std().add_suffix('_std'))
    res = res.join(folds_df.groupby(keys)['valid'].sum().rename('n_valid_folds')).reset_index()
    return res.dropna(axis=1, how='all')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder-name', required=True)
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--n-jobs', type=int, default=4)
    p.add_argument('--quick', action='store_true', help='Stratified only, 3 folds, to validate the pipeline.')
    p.add_argument('--retest', action='store_true',
                   help="Skip training: rebuild significance_tests.csv from the folder's results_folds.csv.")
    args = p.parse_args()
    if args.retest:
        out_dir = OUT_ROOT / args.folder_name
        significance(pd.read_csv(out_dir / 'results_folds.csv')).to_csv(out_dir / 'significance_tests.csv', index=False)
        return
    schemes = ['stratified'] if args.quick else SCHEMES
    n_splits = 3 if args.quick else args.n_splits
    out_dir = OUT_ROOT / args.folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_clinical()
    emb = load_embeddings('mae', 'cls')
    birads = raw_df.set_index('Image_filename')['BIRADS'].astype(str).reindex(emb.index)
    birads = birads[birads.isin(CLASSES)]
    cases = birads.index
    rank = birads.map(CLASSES.index).to_numpy()

    meta = load_metadata(raw_df).reindex(cases)
    pre = pre_us_features(raw_df).reindex(cases)
    blocks = {'acquisition': meta[METADATA_COLS], 'pre_us': pre, 'embedding': emb.loc[cases]}
    inputs = {fs: pd.concat([blocks[b] for b in fs.split('+')], axis=1).to_numpy(float) for fs in FEATURE_SETS}

    X_emb = inputs['embedding']
    for fs in STACKED_SETS:
        inputs[fs] = inputs[fs.removeprefix('emb_score+')]
    folds = {scheme: make_folds(scheme, rank, meta, n_splits) for scheme in schemes}
    tasks = [(fs, scheme) for scheme in schemes for fs in FEATURE_SETS + STACKED_SETS]
    # Stacked and embedding sets first: they dominate the runtime.
    cost = lambda fs: (N_INNER + 1) * X_emb.shape[1] if fs in STACKED_SETS else inputs[fs].shape[1]
    tasks.sort(key=lambda t: -cost(t[0]))
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_set)(fs, inputs[fs], rank, cases.to_numpy(), folds[scheme], scheme,
                         X_emb if fs in STACKED_SETS else None)
        for fs, scheme in tasks)

    folds_df = pd.DataFrame([r for rows, _ in results for r in rows])
    oof = pd.concat([d for _, oofs in results for d in oofs], ignore_index=True)
    res = summarize(folds_df)
    sig = significance(folds_df)

    folds_df.to_csv(out_dir / 'results_folds.csv', index=False)
    res.to_csv(out_dir / 'results.csv', index=False)
    oof.to_csv(out_dir / 'oof_predictions.csv', index=False)
    sig.to_csv(out_dir / 'significance_tests.csv', index=False)
    (out_dir / 'config.json').write_text(json.dumps({
        'embedding': 'mae_cls', 'model': 'catboost', 'n_splits': n_splits, 'schemes': schemes,
        'feature_sets': FEATURE_SETS + STACKED_SETS, 'n_inner': N_INNER, 'classes': CLASSES, 'cuts': CUTS, 'n_cases': len(cases),
        'class_counts': birads.value_counts().to_dict(), 'pre_us_cols': pre.columns.tolist(),
        'metadata_cols': METADATA_COLS,
    }, indent=2))

    with pd.option_context('display.width', 250, 'display.max_rows', 500):
        main_rows = res[res['method'].isin(['multiclass', 'ordinal', 'dedicated'])]
        print(main_rows.pivot_table(index=['target', 'method'], columns=['scheme', 'feature_set'],
                                    values=[c for c in ['roc_auc_mean'] if c in res]).round(3).to_string())
    print(f'\nSaved to {out_dir}')


if __name__ == '__main__':
    main()
