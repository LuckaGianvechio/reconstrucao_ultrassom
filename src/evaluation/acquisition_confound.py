"""Separate acquisition from image content in the embedding -> clinical results
(follow-up of src/models/train_embedding_to_clinical_gbm.py, whose negative-control
targets were predicted better than the image ones).

(a) Increment over acquisition: CatBoost on embedding + acquisition metadata
    (image width, height, Pixel_size) vs. on the metadata alone. The gain is what the
    embedding adds beyond the acquisition batch.
(b) Grouped CV: folds are built so that no acquisition batch appears in both train
    and test, with batch = image size (52 groups) or, stricter, Pixel_size
    (28 groups; one Pixel_size spans several image sizes of the same device/setting).
    A model can no longer score by recognising the batch.

Every feature set (metadata, embedding, embedding+metadata) and the Dummy baseline
are evaluated on the same folds within each CV scheme. With grouped folds, a fold is
skipped (NaN) when the training split lacks a class or the test split has only one.

Outputs in outputs/embedding_to_tabular_experiments/{folder_name}/:
- results_folds.csv   per (target, scheme, fold) metrics
- results.csv         per (target, scheme): mean/std over valid folds per feature set
- summary_by_group.csv  mean over targets per (target_group, scheme)

Usage:
    python -m src.evaluation.acquisition_confound --folder-name acquisition_confound
"""
import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.evaluation.metadata_only_baseline import load_metadata
from src.models.embedding_to_clinical_shared import load_embeddings
from src.models.train_embedding_to_clinical_gbm import (
    MIN_CLASS_COUNT, OUT_ROOT, TARGET_GROUPS, build_targets, make_models, score, target_group,
)
from src.preprocessing.clinical_targets import load_raw_clinical

METADATA_COLS = ['width', 'height', 'pixel_size']
FEATURE_SETS = ['metadata', 'embedding', 'embedding+metadata']
SCHEMES = ['stratified', 'group_size', 'group_pixel_size']
METRICS = ['roc_auc', 'pr_auc', 'balanced_accuracy']


def make_folds(scheme: str, y: np.ndarray, meta: pd.DataFrame, n_splits: int) -> list:
    if scheme == 'stratified':
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=42).split(y, y))
    groups = (meta['width'].astype(str) + 'x' + meta['height'].astype(str) if scheme == 'group_size'
              else meta['pixel_size'].astype(str))
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=42).split(y, y, groups.to_numpy()))


def run_target(name: str, kind: str, y: np.ndarray, X_emb: np.ndarray, meta: pd.DataFrame,
               scheme: str, n_splits: int) -> list:
    X_meta = meta[METADATA_COLS].to_numpy(float)
    inputs = {'metadata': X_meta, 'embedding': X_emb, 'embedding+metadata': np.hstack([X_emb, X_meta])}
    n_classes = len(np.unique(y))

    rows = []
    for fold, (train, test) in enumerate(make_folds(scheme, y, meta, n_splits)):
        row = {'target': name, 'kind': kind, 'scheme': scheme, 'fold': fold, 'n_test': len(test),
               'test_positive_rate': np.nan if kind == 'multiclass' else y[test].mean()}
        valid = len(np.unique(y[train])) == n_classes and len(np.unique(y[test])) > 1
        row['valid'] = valid
        if valid:
            dummy = DummyClassifier(strategy='most_frequent').fit(X_emb[train], y[train])
            row.update({f'dummy_{m}': v for m, v in score(kind, y[test], dummy, X_emb[test]).items() if m in METRICS})
            for source, X in inputs.items():
                model = make_models(kind)['catboost'].fit(X[train], y[train])
                row.update({f'{source}_{m}': v for m, v in score(kind, y[test], model, X[test]).items()
                            if m in METRICS})
        rows.append(row)
    return rows


def summarize(folds_df: pd.DataFrame) -> tuple:
    valid = folds_df[folds_df['valid']]
    metric_cols = [f'{s}_{m}' for s in ['dummy'] + FEATURE_SETS for m in METRICS if f'{s}_{m}' in valid]
    grouped = valid.groupby(['target', 'kind', 'scheme'])[metric_cols]
    res = grouped.mean().add_suffix('_mean').join(grouped.std().add_suffix('_std'))
    res = res.join(folds_df.groupby(['target', 'kind', 'scheme'])['valid'].sum().rename('n_valid_folds')).reset_index()
    res['target_group'] = res['target'].map(target_group)
    for m in ['roc_auc', 'pr_auc']:
        res[f'gain_vs_metadata_{m}'] = res[f'embedding+metadata_{m}_mean'] - res[f'metadata_{m}_mean']
    res = res.sort_values(['target_group', 'target', 'scheme'])

    cls = res[res['target_group'].isin(TARGET_GROUPS)]
    cols = [f'{s}_roc_auc_mean' for s in FEATURE_SETS] + ['gain_vs_metadata_roc_auc', 'dummy_pr_auc_mean'] + \
           [f'{s}_pr_auc_mean' for s in FEATURE_SETS]
    summary = (cls.groupby(['target_group', 'scheme'])[cols].mean()
               .join(cls.groupby(['target_group', 'scheme']).size().rename('n_targets')).reset_index())
    return res, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder-name', required=True)
    p.add_argument('--embedding', default='mae:cls', help='model:pooling')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--n-jobs', type=int, default=4)
    args = p.parse_args()
    out_dir = OUT_ROOT / args.folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_clinical()
    emb = load_embeddings(*args.embedding.split(':'))
    meta = load_metadata(raw_df).reindex(emb.index)
    targets = [t for t in build_targets(raw_df, emb.index, max(MIN_CLASS_COUNT, args.n_splits))
               if t[1] != 'regression']

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_target)(name, kind, y.to_numpy(), emb.loc[y.index].to_numpy(), meta.loc[y.index],
                            scheme, args.n_splits)
        for name, kind, y in targets for scheme in SCHEMES
    )
    folds_df = pd.DataFrame([r for rows in results for r in rows])
    res, summary = summarize(folds_df)

    folds_df.to_csv(out_dir / 'results_folds.csv', index=False)
    res.to_csv(out_dir / 'results.csv', index=False)
    summary.to_csv(out_dir / 'summary_by_group.csv', index=False)
    (out_dir / 'config.json').write_text(json.dumps({
        'embedding': args.embedding, 'model': 'catboost', 'n_splits': args.n_splits, 'schemes': SCHEMES,
        'feature_sets': FEATURE_SETS, 'metadata_cols': METADATA_COLS, 'n_targets': len(targets),
    }, indent=2))

    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(summary.to_string(index=False, float_format='%.3f'))
    print(f'\nSaved to {out_dir}')


if __name__ == '__main__':
    main()
