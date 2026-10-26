"""How much specialist-annotated clinical information do ultrasound image
embeddings carry? Sweep of embedding -> clinical-target prediction over:

- embeddings: USF-MAE (mean / CLS pooling, 768-d) and SD-VAE (mean pooling, 4-d;
  flattened latent, 4096-d)
- dimensionality reduction: raw, PCA and UMAP at the same n_components, fit on
  the training fold only
- models: CatBoost, XGBoost, LightGBM, MLP, against a Dummy baseline
- validation: stratified k-fold (KFold for the Age regression); UMAP repeated
  over several seeds per fold and averaged, since it is stochastic

Targets come from src/preprocessing/clinical_targets.py (Age, single-label
multiclass columns, and one binary target per multi-label value), each tagged
with a group (see `target_group`): 'image' (annotated from the image), 'pathology',
'history' (not visible in the image: a negative control) and 'sentinel'
("not available"/"not applicable" labels, which describe report completeness and
are left out of summary_by_combo.csv).

Outputs in outputs/embedding_to_tabular_experiments/{folder_name}/:
- results_folds.csv   one row per (combination, fold), UMAP seeds averaged
- results_long.csv    mean/std over folds, plus delta vs. baseline on the same folds
- summary_by_combo.csv  classification metrics averaged over targets per
                        (target_group, embedding, reducer, n_components, model)
- best_by_target.csv  best combination per target (max over the grid, so optimistic)
- config.json

Usage:
    python -m src.models.train_embedding_to_clinical_gbm --folder-name smoke_test --quick
    python -m src.models.train_embedding_to_clinical_gbm --folder-name gbm_pca_umap_sweep
    python -m src.models.train_embedding_to_clinical_gbm --folder-name gbm_pca_umap_sweep --resummarize
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, precision_score,
    r2_score, recall_score, root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from src.models.embedding_to_clinical_shared import (
    apply_pca, apply_umap, load_embeddings, safe_pr_auc, safe_roc_auc,
)
from src.preprocessing.clinical_targets import (
    load_raw_clinical, multilabel_targets, regression_targets, singlelabel_targets,
)


OUT_ROOT = Path('outputs/embedding_to_tabular_experiments')

EMBEDDINGS = [('mae', 'mean'), ('mae', 'cls'), ('vae', 'mean'), ('vae', 'flat')]
MODEL_NAMES = ['catboost', 'xgboost', 'lightgbm', 'mlp']
REDUCER_DIMS = [10, 30, 50]
QUICK_REDUCER_DIMS = [10]
QUICK_TARGETS = ['Age', 'Shape', 'BIRADS', 'Halo', 'Margin_spiculated', 'Diagnosis_Fibroadenoma',
                 'Signs_not available']

MIN_SAMPLES = 30
MIN_CLASS_COUNT = 5
SENTINEL_MARKERS = ('not available', 'not applicable')

# What a target describes. Only 'image' is annotated by the specialist from the image
# itself; 'history' (symptoms, signs, age) is not visible in it and serves as a
# negative control: high scores there point to a confound, not image content.
IMAGE_TARGETS = {'BIRADS', 'Shape', 'Echogenicity', 'Posterior_features', 'Halo', 'Calcifications',
                 'Skin_thickening'}
IMAGE_PREFIXES = ('Margin_', 'Tissue_', 'Interpretation_')
PATHOLOGY_TARGETS = {'Verification'}
PATHOLOGY_PREFIXES = ('Diagnosis_',)
TARGET_GROUPS = ['image', 'pathology', 'history']


def target_group(name: str) -> str:
    if any(m in name for m in SENTINEL_MARKERS):
        return 'sentinel'
    if name in IMAGE_TARGETS or name.startswith(IMAGE_PREFIXES):
        return 'image'
    if name in PATHOLOGY_TARGETS or name.startswith(PATHOLOGY_PREFIXES):
        return 'pathology'
    return 'history'

MLP_HIDDEN_LAYER_SIZES = (64,)
GBM_PARAMS = dict(learning_rate=0.08, max_depth=4)

RANKING_METRICS = {
    'classification': ['balanced_accuracy', 'roc_auc', 'pr_auc'],
    'regression': ['r2'],
}
GROUP_KEYS = ['embedding', 'target', 'kind', 'is_sentinel_target', 'n_samples', 'n_classes',
              'reducer', 'n_components', 'model']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder-name', required=True,
                   help='Results go to outputs/embedding_to_tabular_experiments/<folder-name>/')
    p.add_argument('--quick', action='store_true',
                   help='Reduced grid to validate the pipeline: raw/pca10/umap10, 1 UMAP seed, 3 folds, '
                        f'targets {QUICK_TARGETS}.')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--umap-seeds', type=int, default=3)
    p.add_argument('--n-jobs', type=int, default=4,
                   help='Parallel worker processes. Each peaks around 1-1.5 GB (raw vae_flat fits).')
    p.add_argument('--resummarize', action='store_true',
                   help='Skip training: rebuild summary_by_combo.csv / best_by_target.csv from the '
                        "folder's existing results_long.csv.")
    args = p.parse_args()
    if args.quick:
        args.n_splits, args.umap_seeds = 3, 1
    return args


def build_targets(raw_df: pd.DataFrame, index: pd.Index, min_class_count: int) -> list:
    """Returns [(name, kind, y)] with y aligned to `index`, missing labels dropped
    and multiclass labels encoded to 0..K-1."""
    targets = []

    for name, y in regression_targets(raw_df).items():
        y = y.reindex(index).dropna()
        if len(y) >= MIN_SAMPLES:
            targets.append((name, 'regression', y.astype(float)))

    for name, y in singlelabel_targets(raw_df).items():
        y = y.reindex(index).dropna()
        counts = y.value_counts()
        y = y[y.isin(counts[counts >= min_class_count].index)]
        if len(y) < MIN_SAMPLES or y.nunique() < 2:
            continue
        if y.nunique() == 2:
            # Positive class = minority, so precision/recall/PR-AUC describe the rarer finding.
            targets.append((name, 'binary', (y == y.value_counts().idxmin()).astype(int)))
        else:
            targets.append((name, 'multiclass', pd.Series(LabelEncoder().fit_transform(y), index=y.index)))

    multi_df = multilabel_targets(raw_df).reindex(index).dropna()
    for name in multi_df.columns:
        y = multi_df[name].astype(int)
        if len(y) >= MIN_SAMPLES and y.value_counts().min() >= min_class_count:
            targets.append((name, 'binary', y))

    return targets


def reducer_settings(n_features: int, dims: list) -> list:
    settings = [('raw', n_features)]
    for reducer in ('pca', 'umap'):
        settings += [(reducer, k) for k in dims if k < n_features]
    return settings


def make_models(kind: str) -> dict:
    if kind == 'regression':
        return {
            'catboost': CatBoostRegressor(iterations=150, depth=4, learning_rate=0.08, random_seed=42,
                                          thread_count=1, verbose=False),
            'xgboost': XGBRegressor(n_estimators=150, **GBM_PARAMS, random_state=42, n_jobs=1, verbosity=0),
            'lightgbm': LGBMRegressor(n_estimators=150, **GBM_PARAMS, random_state=42, n_jobs=1, verbose=-1),
            'mlp': make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES, alpha=1e-3, max_iter=1000,
                early_stopping=True, random_state=42,
            )),
        }
    return {
        'catboost': CatBoostClassifier(iterations=150, depth=4, learning_rate=0.08, random_seed=42,
                                       auto_class_weights='Balanced', thread_count=1, verbose=False),
        'xgboost': XGBClassifier(n_estimators=150, **GBM_PARAMS, random_state=42, n_jobs=1, verbosity=0),
        'lightgbm': LGBMClassifier(n_estimators=150, **GBM_PARAMS, random_state=42, n_jobs=1,
                                   class_weight='balanced', verbose=-1),
        'mlp': make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES, alpha=1e-3, max_iter=1000,
            early_stopping=True, random_state=42,
        )),
    }


def fit(model_name: str, model, kind: str, X, y):
    # XGBoost has no class_weight option; balanced sample weights match the others.
    if model_name == 'xgboost' and kind != 'regression':
        model.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    else:
        model.fit(X, y)
    return model


def score(kind: str, y_test, model, X_test) -> dict:
    if kind == 'regression':
        preds = model.predict(X_test)
        return {'r2': r2_score(y_test, preds), 'mae': mean_absolute_error(y_test, preds),
                'rmse': root_mean_squared_error(y_test, preds)}

    preds = np.asarray(model.predict(X_test)).ravel().astype(int)
    probas = model.predict_proba(X_test)
    average = 'binary' if kind == 'binary' else 'macro'
    return {
        'balanced_accuracy': balanced_accuracy_score(y_test, preds),
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, average=average, zero_division=0),
        'recall': recall_score(y_test, preds, average=average, zero_division=0),
        'f1': f1_score(y_test, preds, average=average, zero_division=0),
        'roc_auc': safe_roc_auc(y_test, probas),
        'pr_auc': safe_pr_auc(y_test, probas),
    }


def reduce(reducer: str, n_components: int, X_train, X_test, seed: int):
    if reducer == 'pca':
        X_train, X_test, _ = apply_pca(X_train, X_test, n_components)
    elif reducer == 'umap':
        X_train, X_test, _ = apply_umap(X_train, X_test, n_components, random_state=seed)
    return X_train, X_test


def run_task(embedding: str, X: np.ndarray, name: str, kind: str, y: np.ndarray,
             settings: list, n_splits: int, umap_seeds: int) -> list:
    splitter = (KFold if kind == 'regression' else StratifiedKFold)(n_splits=n_splits, shuffle=True,
                                                                     random_state=42)
    folds = list(splitter.split(X, y))
    base = {'embedding': embedding, 'target': name, 'kind': kind,
            'is_sentinel_target': any(m in name for m in SENTINEL_MARKERS),
            'n_samples': len(y), 'n_classes': np.nan if kind == 'regression' else len(np.unique(y))}

    rows = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

        baseline = (DummyRegressor(strategy='mean') if kind == 'regression'
                    else DummyClassifier(strategy='most_frequent')).fit(X_train, y_train)
        baseline_metrics = score(kind, y_test, baseline, X_test)

        for reducer, n_components in settings:
            seeds = range(umap_seeds) if reducer == 'umap' else [0]
            for seed in seeds:
                X_train_r, X_test_r = reduce(reducer, n_components, X_train, X_test, seed)
                key = {**base, 'reducer': reducer, 'n_components': n_components, 'fold': fold, 'seed': seed}
                rows.append({**key, 'model': 'baseline', **baseline_metrics})
                for model_name, model in make_models(kind).items():
                    fit(model_name, model, kind, X_train_r, y_train)
                    rows.append({**key, 'model': model_name, **score(kind, y_test, model, X_test_r)})
    return rows


def aggregate(raw_rows: pd.DataFrame) -> tuple:
    """Average UMAP seeds within each fold, add paired deltas vs. the baseline of the
    same fold, then take mean/std over folds."""
    metric_cols = [c for c in raw_rows.columns if c not in GROUP_KEYS + ['fold', 'seed']]
    folds_df = raw_rows.groupby(GROUP_KEYS + ['fold'], dropna=False)[metric_cols].mean().reset_index()

    ranking = sorted({m for ms in RANKING_METRICS.values() for m in ms} & set(metric_cols))
    fold_keys = ['embedding', 'target', 'reducer', 'n_components', 'fold']
    baseline = folds_df[folds_df['model'] == 'baseline'][fold_keys + ranking]
    folds_df = folds_df.merge(baseline, on=fold_keys, suffixes=('', '_baseline'))
    for m in ranking:
        folds_df[f'delta_{m}'] = folds_df[m] - folds_df[f'{m}_baseline']
    folds_df = folds_df.drop(columns=[f'{m}_baseline' for m in ranking])

    value_cols = metric_cols + [f'delta_{m}' for m in ranking]
    grouped = folds_df.groupby(GROUP_KEYS, dropna=False)[value_cols]
    long_df = grouped.mean().add_suffix('_mean').join(grouped.std().add_suffix('_std')).reset_index()
    long_df = long_df.dropna(axis=1, how='all')
    return folds_df, long_df


def summarize(long_df: pd.DataFrame) -> tuple:
    long_df['target_group'] = long_df['target'].map(target_group)
    cls = long_df[(long_df['kind'] != 'regression') & long_df['target_group'].isin(TARGET_GROUPS)]
    cols = ['roc_auc_mean', 'pr_auc_mean', 'balanced_accuracy_mean', 'precision_mean', 'recall_mean',
            'accuracy_mean', 'delta_roc_auc_mean', 'delta_pr_auc_mean', 'delta_balanced_accuracy_mean']
    keys = ['target_group', 'embedding', 'reducer', 'n_components', 'model']
    summary = (cls.groupby(keys)[cols].mean()
               .join(cls.groupby(keys).size().rename('n_targets'))
               .reset_index().sort_values(['target_group', 'roc_auc_mean'], ascending=[True, False]))

    models = long_df[long_df['model'] != 'baseline'].copy()
    models['rank_metric'] = np.where(models['kind'] == 'regression', models.get('r2_mean'), models.get('roc_auc_mean'))
    best = (models.sort_values('rank_metric', ascending=False).groupby('target', as_index=False).head(1)
            .sort_values('rank_metric', ascending=False))
    best_cols = ['target', 'target_group', 'kind', 'n_samples', 'n_classes', 'embedding', 'reducer',
                 'n_components', 'model', 'rank_metric', 'roc_auc_mean', 'pr_auc_mean',
                 'balanced_accuracy_mean', 'delta_pr_auc_mean', 'r2_mean']
    return summary, best[[c for c in best_cols if c in best.columns]]


def write_summaries(long_df: pd.DataFrame, out_dir: Path):
    summary, best = summarize(long_df)
    long_df.to_csv(out_dir / 'results_long.csv', index=False)
    summary.to_csv(out_dir / 'summary_by_combo.csv', index=False)
    best.to_csv(out_dir / 'best_by_target.csv', index=False)

    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print("\n=== 'image' targets: mean over targets (top 15 by ROC-AUC) ===")
        print(summary[summary['target_group'] == 'image'].head(15).to_string(index=False, float_format='%.3f'))
        print('\n=== Best combination per target (max over the grid: optimistic) ===')
        print(best.to_string(index=False, float_format='%.3f'))


def main():
    args = parse_args()
    out_dir = OUT_ROOT / args.folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resummarize:
        write_summaries(pd.read_csv(out_dir / 'results_long.csv'), out_dir)
        return

    started = time.time()

    dims = QUICK_REDUCER_DIMS if args.quick else REDUCER_DIMS
    min_class_count = max(MIN_CLASS_COUNT, args.n_splits)
    raw_df = load_raw_clinical()

    tasks = []
    for model, pooling in EMBEDDINGS:
        embedding = f'{model}_{pooling}'
        X = load_embeddings(model, pooling)
        settings = reducer_settings(X.shape[1], dims)
        targets = build_targets(raw_df, X.index, min_class_count)
        if args.quick:
            targets = [t for t in targets if t[0] in QUICK_TARGETS]
        print(f'{embedding}: {X.shape[1]} dims, {len(targets)} targets, settings={settings}')
        tasks += [(embedding, X.loc[y.index].to_numpy(), name, kind, y.to_numpy(), settings)
                  for name, kind, y in targets]

    # Longest first (raw fits dominate, scaling with n_features and n_classes) so no
    # single slow task is left running alone at the end.
    tasks.sort(key=lambda t: t[1].shape[1] * (2 if t[3] == 'regression' else len(np.unique(t[4]))),
               reverse=True)

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_task)(embedding, X, name, kind, y, settings, args.n_splits, args.umap_seeds)
        for embedding, X, name, kind, y, settings in tasks
    )
    raw_rows = pd.DataFrame([row for rows in results for row in rows])

    folds_df, long_df = aggregate(raw_rows)
    folds_df.to_csv(out_dir / 'results_folds.csv', index=False)
    write_summaries(long_df, out_dir)

    import catboost, lightgbm, sklearn, umap, xgboost
    config = {
        'folder_name': args.folder_name, 'quick': args.quick, 'n_splits': args.n_splits,
        'umap_seeds': args.umap_seeds, 'embeddings': [f'{m}_{p}' for m, p in EMBEDDINGS],
        'reducer_dims': dims, 'models': MODEL_NAMES, 'min_samples': MIN_SAMPLES,
        'min_class_count': min_class_count, 'n_tasks': len(tasks),
        'finished_at': datetime.now().isoformat(timespec='seconds'),
        'runtime_seconds': round(time.time() - started, 1),
        'versions': {lib.__name__: lib.__version__ for lib in (catboost, lightgbm, sklearn, umap, xgboost)},
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'\nSaved to {out_dir} ({config["runtime_seconds"]}s)')


if __name__ == '__main__':
    main()
