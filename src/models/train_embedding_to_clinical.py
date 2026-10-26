"""Predict clinical features FROM pooled image embeddings (the inverse
direction of src/models/train_catboost.py, which predicts embeddings from
clinical features and is documented to fail — see ONBOARDING.md and
docs/vae-experiment-log.md).

For every clinical target (regression: Age; single-label multiclass: BIRADS,
Shape, Echogenicity, ...; multi-label binary: one classifier per
Interpretation/Margin/Signs/Symptoms/Tissue_composition/Diagnosis label),
trains two models on the chosen pooled embedding — CatBoost (tree-based) and
an MLP (the standard baseline for a binary/multiclass classification problem
over a fixed-size feature vector) — and compares both against a trivial
baseline (predict the mean / majority class), so any apparent skill is
visible relative to chance rather than raw accuracy alone (several targets
are heavily imbalanced, e.g. Calcifications ~88% "no").

Classification targets report accuracy, precision, recall, F1 and ROC AUC
(all vs. the same baseline); "metric"/"delta" columns use balanced accuracy
specifically, since it's the one that stays meaningful under class imbalance
and is what targets are ranked by. Regression (Age) reports R², MAE, RMSE.

Optionally, --pca-variance or --umap-components reduces the embedding before
training (mutually exclusive with each other). Both are fit on the TRAINING
split only (per target, since the valid-row subset differs slightly per
target after dropping missing labels) and only transform the test split —
never fit on test data.

--pca-variance keeps just enough principal components to explain that
fraction of variance (see docs/embedding-to-clinical-log.md: this generally
hurt performance, since the embeddings' dominant-variance directions are not
the ones that carry clinically relevant signal). --umap-components asks for
an actual, unsupervised nonlinear manifold learning method instead of a
linear projection — unlike t-SNE (which has no principled out-of-sample
transform: it optimizes a fixed embedding for the data it was given, so
"transforming" new points would require rerunning the whole optimization
with the test points included, leaking them into the fit), UMAP genuinely
supports fit-on-train/transform-on-test via its learned nearest-neighbor
graph + embedding, which is why it's used here instead.

Usage:
    python -m src.models.train_embedding_to_clinical --model mae --pooling mean
    python -m src.models.train_embedding_to_clinical --model mae --pooling cls
    python -m src.models.train_embedding_to_clinical --model vae --pooling mean
    python -m src.models.train_embedding_to_clinical --model mae --pooling mean --pca-variance 0.7
    python -m src.models.train_embedding_to_clinical --model mae --pooling mean --umap-components 30
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, root_mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.models.embedding_to_clinical_shared import apply_pca, apply_umap, load_embeddings, safe_roc_auc
from src.preprocessing.clinical_targets import (
    load_raw_clinical, regression_targets, singlelabel_targets, multilabel_targets,
)


OUT_DIR = Path('outputs/embedding_to_clinical')

MIN_SAMPLES = 30  # skip targets with too few valid (non-missing) rows to split meaningfully
MIN_CLASS_COUNT = 5  # for multiclass targets, drop classes with fewer examples than this

MLP_HIDDEN_LAYER_SIZES = (64,)  # small on purpose: ~200 training samples, up to 768-dim input


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['mae', 'vae'], required=True)
    p.add_argument('--pooling', choices=['mean', 'cls'], default='mean')
    p.add_argument('--pca-variance', type=float, default=None,
                    help='If set (e.g. 0.7), reduce the embedding with PCA before training, keeping '
                         'just enough components to explain that fraction of variance (fit on the '
                         'training split only). Omit for no PCA (raw embedding).')
    p.add_argument('--umap-components', type=int, default=None,
                    help='If set (e.g. 30), reduce the embedding with UMAP to this many components '
                         'before training (fit on the training split only). Mutually exclusive with '
                         '--pca-variance.')
    args = p.parse_args()
    if args.pca_variance is not None and args.umap_components is not None:
        p.error('--pca-variance and --umap-components are mutually exclusive')
    return args


def apply_dim_reduction(X_train, X_test, pca_variance, umap_components):
    """Dispatch to whichever dimensionality reduction (if any) was requested.
    Returns (X_train, X_test, n_components) — n_components is None if neither was set."""
    if pca_variance is not None:
        return apply_pca(X_train, X_test, pca_variance)
    if umap_components is not None:
        return apply_umap(X_train, X_test, umap_components)
    return X_train, X_test, None


def make_classifiers() -> dict:
    return {
        'catboost': CatBoostClassifier(iterations=150, depth=4, learning_rate=0.08, random_seed=42,
                                        auto_class_weights='Balanced', verbose=False),
        # MLP needs scaled inputs; small hidden layer + early stopping to avoid overfitting
        # with ~200 training samples and up to 768-dim embeddings.
        'mlp': make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES, alpha=1e-3, max_iter=1000,
            early_stopping=True, random_state=42,
        )),
    }


def make_regressors() -> dict:
    return {
        'catboost': CatBoostRegressor(iterations=150, depth=4, learning_rate=0.08, random_seed=42, verbose=False),
        'mlp': make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES, alpha=1e-3, max_iter=1000,
            early_stopping=True, random_state=42,
        )),
    }


def eval_regression(y_test, preds_by_model: dict, baseline_preds) -> dict:
    metrics = {'metric_baseline': r2_score(y_test, baseline_preds)}
    for name, preds in preds_by_model.items():
        metrics[f'metric_{name}'] = r2_score(y_test, preds)
        metrics[f'mae_{name}'] = mean_absolute_error(y_test, preds)
        metrics[f'rmse_{name}'] = root_mean_squared_error(y_test, preds)
        metrics[f'delta_{name}'] = metrics[f'metric_{name}'] - metrics['metric_baseline']
    return metrics


def eval_classification(y_test, results_by_model: dict, baseline_preds, baseline_probas, average: str) -> dict:
    """results_by_model: {model_name: (preds, probas_or_None)}"""
    metrics = {
        'metric_baseline': balanced_accuracy_score(y_test, baseline_preds),
        'accuracy_baseline': accuracy_score(y_test, baseline_preds),
        'precision_baseline': precision_score(y_test, baseline_preds, average=average, zero_division=0),
        'recall_baseline': recall_score(y_test, baseline_preds, average=average, zero_division=0),
        'f1_baseline': f1_score(y_test, baseline_preds, average=average, zero_division=0),
        'roc_auc_baseline': safe_roc_auc(y_test, baseline_probas),
    }
    for name, (preds, probas) in results_by_model.items():
        metrics[f'metric_{name}'] = balanced_accuracy_score(y_test, preds)
        metrics[f'accuracy_{name}'] = accuracy_score(y_test, preds)
        metrics[f'precision_{name}'] = precision_score(y_test, preds, average=average, zero_division=0)
        metrics[f'recall_{name}'] = recall_score(y_test, preds, average=average, zero_division=0)
        metrics[f'f1_{name}'] = f1_score(y_test, preds, average=average, zero_division=0)
        metrics[f'roc_auc_{name}'] = safe_roc_auc(y_test, probas)
        metrics[f'delta_{name}'] = metrics[f'metric_{name}'] - metrics['metric_baseline']
    return metrics


def run_regression_target(name, X, y, pca_variance=None, umap_components=None):
    valid = y.notna()
    X_valid, y_valid = X.loc[valid.index[valid]], y[valid]
    if len(y_valid) < MIN_SAMPLES:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)

    X_train, X_test, n_components = apply_dim_reduction(X_train, X_test, pca_variance, umap_components)

    preds_by_model = {}
    for model_name, model in make_regressors().items():
        model.fit(X_train, y_train)
        preds_by_model[model_name] = model.predict(X_test)

    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)

    metrics = eval_regression(y_test, preds_by_model, baseline_preds)
    return {'target': name, 'kind': 'regression', 'n_samples': len(y_valid), 'n_classes': None,
            'n_components': n_components, **metrics}


def run_multiclass_target(name, X, y, pca_variance=None, umap_components=None):
    valid = y.notna()
    X_valid, y_valid = X.loc[valid.index[valid]], y[valid]

    counts = y_valid.value_counts()
    keep_classes = counts[counts >= MIN_CLASS_COUNT].index
    mask = y_valid.isin(keep_classes)
    X_valid, y_valid = X_valid[mask], y_valid[mask]

    if len(y_valid) < MIN_SAMPLES or y_valid.nunique() < 2:
        return None

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_valid)

    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded,
    )

    X_train, X_test, n_components = apply_dim_reduction(X_train, X_test, pca_variance, umap_components)

    results_by_model = {}
    for model_name, model in make_classifiers().items():
        model.fit(X_train, y_train)
        preds = np.asarray(model.predict(X_test)).ravel()
        probas = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        results_by_model[model_name] = (preds, probas)

    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_probas = baseline.predict_proba(X_test)

    metrics = eval_classification(y_test, results_by_model, baseline_preds, baseline_probas, average='macro')
    return {'target': name, 'kind': 'multiclass', 'n_samples': len(y_valid), 'n_classes': y_valid.nunique(),
            'n_components': n_components, **metrics}


def run_binary_target(name, X, y, pca_variance=None, umap_components=None):
    if len(y) < MIN_SAMPLES:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_test, n_components = apply_dim_reduction(X_train, X_test, pca_variance, umap_components)

    results_by_model = {}
    for model_name, model in make_classifiers().items():
        model.fit(X_train, y_train)
        preds = np.asarray(model.predict(X_test)).ravel()
        probas = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        results_by_model[model_name] = (preds, probas)

    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_probas = baseline.predict_proba(X_test)

    metrics = eval_classification(y_test, results_by_model, baseline_preds, baseline_probas, average='binary')
    return {'target': name, 'kind': 'binary', 'n_samples': len(y), 'n_classes': 2,
            'n_components': n_components, **metrics}


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X = load_embeddings(args.model, args.pooling)
    raw_df = load_raw_clinical()

    rows = []

    for name, y in regression_targets(raw_df).items():
        print(f'[regression] {name}')
        result = run_regression_target(name, X, y, pca_variance=args.pca_variance,
                                        umap_components=args.umap_components)
        if result:
            rows.append(result)
        else:
            print(f'  skipped (too few valid samples)')

    for name, y in singlelabel_targets(raw_df).items():
        print(f'[multiclass] {name}')
        result = run_multiclass_target(name, X, y, pca_variance=args.pca_variance,
                                        umap_components=args.umap_components)
        if result:
            rows.append(result)
        else:
            print(f'  skipped (too few valid samples/classes)')

    multi_df = multilabel_targets(raw_df)
    for col in multi_df.columns:
        print(f'[binary] {col}')
        # align by Image_filename index, drop any embedding rows missing for this target
        common_idx = X.index.intersection(multi_df.index)
        X_aligned = X.loc[common_idx]
        y_aligned = multi_df.loc[common_idx, col]
        result = run_binary_target(col, X_aligned, y_aligned, pca_variance=args.pca_variance,
                                    umap_components=args.umap_components)
        if result:
            rows.append(result)
        else:
            print(f'  skipped (too few samples)')

    results_df = pd.DataFrame(rows).sort_values('delta_catboost', ascending=False)
    if args.pca_variance is not None:
        suffix = f'_pca{int(args.pca_variance * 100)}'
        label = f', PCA={args.pca_variance:.0%} variance'
    elif args.umap_components is not None:
        suffix = f'_umap{args.umap_components}'
        label = f', UMAP={args.umap_components} components'
    else:
        suffix = ''
        label = ', no dim. reduction'
    out_path = OUT_DIR / f'{args.model}_{args.pooling}{suffix}_results.csv'
    results_df.to_csv(out_path, index=False)

    print()
    print(f'=== {args.model} / {args.pooling}{label} ===')

    is_classification = results_df['kind'] != 'regression'
    cls_df = results_df[is_classification].sort_values('delta_catboost', ascending=False)
    reg_df = results_df[~is_classification]

    if not cls_df.empty:
        print('\n--- Classification targets (balanced accuracy / precision / recall / f1 / ROC AUC) ---')
        for model_name in ['catboost', 'mlp']:
            print(f'\n[{model_name}]')
            cols = ['target', 'kind', 'n_samples', 'n_classes', 'accuracy_baseline', f'accuracy_{model_name}',
                    f'precision_{model_name}', f'recall_{model_name}', f'f1_{model_name}',
                    f'roc_auc_{model_name}', f'delta_{model_name}']
            print(cls_df[cols].to_string(index=False))

    if not reg_df.empty:
        print('\n--- Regression targets (R² / MAE / RMSE) ---')
        for model_name in ['catboost', 'mlp']:
            print(f'\n[{model_name}]')
            cols = ['target', 'n_samples', 'metric_baseline', f'metric_{model_name}',
                    f'mae_{model_name}', f'rmse_{model_name}', f'delta_{model_name}']
            print(reg_df[cols].to_string(index=False))

    print()
    print('Mean delta over all targets: catboost=%.4f, mlp=%.4f' % (
        results_df['delta_catboost'].mean(), results_df['delta_mlp'].mean(),
    ))
    print('Saved to', out_path)


if __name__ == '__main__':
    main()
