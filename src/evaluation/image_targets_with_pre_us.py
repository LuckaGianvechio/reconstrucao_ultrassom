"""Does pre-ultrasound information (Age, Signs, Symptoms: the 'history' group) add to
the image embedding when predicting the specialist's image annotations ('image' group,
23 targets)? Pathology is left out on purpose: it is obtained after the exam.

Same feature sets as src/evaluation/birads_analysis.py, extended from BIRADS to every
image target: acquisition, pre-US, both, the embedding (mae_cls, 768-d) alone, early
fusion (concatenated) and late fusion (the embedding's nested-CV out-of-fold score as a
feature next to pre-US). Folds are built per target exactly as in acquisition_confound
(so with --model catboost the embedding and acquisition runs reproduce it) and shared by
all feature sets of that target: every comparison is paired fold by fold.

Outputs in outputs/embedding_to_tabular_experiments/{folder_name}/:
- results_folds.csv, results.csv, significance_tests.csv, summary.csv, config.json

Usage:
    python -m src.evaluation.image_targets_with_pre_us --folder-name image_targets_with_pre_us_lgbm --model lightgbm
    python -m src.evaluation.image_targets_with_pre_us --folder-name image_targets_with_pre_us
    python -m src.evaluation.image_targets_with_pre_us --folder-name image_targets_with_pre_us --retest
"""
import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.evaluation.acquisition_confound import make_folds
from src.evaluation.birads_analysis import (
    COMPARISONS, FEATURE_SETS, METADATA_COLS, N_INNER, SCHEMES, STACKED_SETS, binary_metrics, fit_predict,
    pre_us_features, stacked_fit_predict,
)
from src.evaluation.metadata_only_baseline import load_metadata
from src.evaluation.significance_tests import benjamini_hochberg, corrected_t
from src.models.embedding_to_clinical_shared import load_embeddings, safe_pr_auc, safe_roc_auc
from src.models.train_embedding_to_clinical_gbm import MIN_CLASS_COUNT, OUT_ROOT, build_targets, target_group
from src.preprocessing.clinical_targets import load_raw_clinical

ALPHA = 0.05


def run_task(name: str, kind: str, y: np.ndarray, X: np.ndarray, X_emb, feature_set: str, scheme: str,
             folds: list, model_name: str) -> list:
    n_classes = len(np.unique(y))
    rows = []
    for fold, (train, test) in enumerate(folds):
        row = {'target': name, 'kind': kind, 'feature_set': feature_set, 'scheme': scheme, 'fold': fold,
               'n_test': len(test)}
        # Same validity rule as acquisition_confound: every class in train, more than one in test.
        row['valid'] = len(np.unique(y[train])) == n_classes and len(np.unique(y[test])) > 1
        if row['valid']:
            s = (stacked_fit_predict(kind, X_emb, X, y, train, test, n_classes, model_name)
                 if feature_set in STACKED_SETS else fit_predict(kind, X[train], y[train], X[test], n_classes, model_name))
            row.update(binary_metrics(y[test], s) if kind == 'binary'
                       else {'roc_auc': safe_roc_auc(y[test], s), 'pr_auc': safe_pr_auc(y[test], s)})
        rows.append(row)
    return rows


def significance(folds_df: pd.DataFrame) -> pd.DataFrame:
    """Paired corrected t on ROC-AUC per (target, scheme, comparison); BH over the targets
    within each (comparison, scheme). Dummy = 0.5."""
    df = folds_df[folds_df['valid']]
    n_total = folds_df.drop_duplicates(['target', 'scheme', 'fold']).groupby(['target', 'scheme'])['n_test'].sum()
    wide = df.pivot_table(index=['target', 'scheme', 'fold', 'n_test'], columns='feature_set',
                          values='roc_auc').reset_index()
    wide['dummy'] = 0.5
    rows = []
    for (target, scheme), g in wide.groupby(['target', 'scheme']):
        for family, a, b in COMPARISONS:
            v = g.dropna(subset=[a, b])
            if len(v) < 2:
                continue
            rows.append({'family': family, 'target': target, 'scheme': scheme, 'a_mean': v[a].mean(),
                         'b_mean': v[b].mean(),
                         **corrected_t((v[a] - v[b]).to_numpy(), v['n_test'].to_numpy(float),
                                       int(n_total[(target, scheme)]))})
    res = pd.DataFrame(rows)
    res['q'] = res.groupby(['family', 'scheme'], group_keys=False)['p'].apply(benjamini_hochberg)
    res['significant'] = res['q'] < ALPHA
    return res.sort_values(['family', 'scheme', 'p'])


def summarize(folds_df: pd.DataFrame, sig: pd.DataFrame) -> tuple:
    valid = folds_df[folds_df['valid']]
    g = valid.groupby(['target', 'kind', 'feature_set', 'scheme'])[['roc_auc', 'pr_auc']]
    res = g.mean().add_suffix('_mean').join(g.std().add_suffix('_std')).reset_index()
    mean_auc = res.groupby(['scheme', 'feature_set'])['roc_auc_mean'].mean().unstack('feature_set')
    counts = sig.groupby(['family', 'scheme']).agg(
        n=('target', 'size'), mean_gain=('gain', 'mean'), positive=('gain', lambda s: int((s > 0).sum())),
        significant_pos=('significant', lambda s: int((s & (sig.loc[s.index, 'gain'] > 0)).sum())),
        significant_neg=('significant', lambda s: int((s & (sig.loc[s.index, 'gain'] < 0)).sum()))).reset_index()
    return res, mean_auc, counts


def write_tests(folds_df: pd.DataFrame, out_dir):
    sig = significance(folds_df)
    res, mean_auc, counts = summarize(folds_df, sig)
    res.to_csv(out_dir / 'results.csv', index=False)
    sig.to_csv(out_dir / 'significance_tests.csv', index=False)
    counts.to_csv(out_dir / 'summary.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_rows', 300):
        print('Mean ROC-AUC over image targets:\n', mean_auc.round(3).to_string())
        print('\n', counts.to_string(index=False, float_format='%.3f'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder-name', required=True)
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--n-jobs', type=int, default=6)
    p.add_argument('--model', default='catboost', choices=['catboost', 'lightgbm'],
                   help='lightgbm is ~8x faster on the raw embedding (2026-09-26), for a first look.')
    p.add_argument('--retest', action='store_true',
                   help="Skip training: rebuild the tests and summaries from the folder's results_folds.csv.")
    args = p.parse_args()
    out_dir = OUT_ROOT / args.folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.retest:
        write_tests(pd.read_csv(out_dir / 'results_folds.csv'), out_dir)
        return

    raw_df = load_raw_clinical()
    emb = load_embeddings('mae', 'cls')
    meta = load_metadata(raw_df).reindex(emb.index)
    pre = pre_us_features(raw_df).reindex(emb.index)
    blocks = {'acquisition': meta[METADATA_COLS], 'pre_us': pre, 'embedding': emb}
    inputs = {fs: pd.concat([blocks[b] for b in fs.removeprefix('emb_score+').split('+')], axis=1)
              for fs in FEATURE_SETS + STACKED_SETS}
    targets = [t for t in build_targets(raw_df, emb.index, max(MIN_CLASS_COUNT, args.n_splits))
               if target_group(t[0]) == 'image' and t[1] != 'regression']

    tasks = []
    for name, kind, y in targets:
        for scheme in SCHEMES:
            folds = make_folds(scheme, y.to_numpy(), meta.loc[y.index], args.n_splits)
            X_emb = emb.loc[y.index].to_numpy(float)
            for fs in FEATURE_SETS + STACKED_SETS:
                tasks.append((name, kind, y.to_numpy(), inputs[fs].loc[y.index].to_numpy(float),
                              X_emb if fs in STACKED_SETS else None, fs, scheme, folds, args.model))
    # Late fusion first (nested CV: 6 embedding fits per fold), then the embedding sets.
    cost = lambda t: ((N_INNER + 1) * 768 if t[5] in STACKED_SETS else t[3].shape[1]) * (
        1 if t[1] == 'binary' else len(np.unique(t[2])))
    tasks.sort(key=lambda t: -cost(t))
    print(f'{len(targets)} image targets, {len(tasks)} tasks')

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(run_task)(*t) for t in tasks)
    folds_df = pd.DataFrame([r for rows in results for r in rows])
    folds_df.to_csv(out_dir / 'results_folds.csv', index=False)
    (out_dir / 'config.json').write_text(json.dumps({
        'embedding': 'mae_cls', 'model': args.model, 'n_splits': args.n_splits, 'schemes': SCHEMES,
        'feature_sets': FEATURE_SETS + STACKED_SETS, 'n_inner': N_INNER, 'targets': [t[0] for t in targets],
        'pre_us_cols': pre.columns.tolist(), 'metadata_cols': METADATA_COLS,
    }, indent=2))
    write_tests(folds_df, out_dir)
    print(f'\nSaved to {out_dir}')


if __name__ == '__main__':
    main()
