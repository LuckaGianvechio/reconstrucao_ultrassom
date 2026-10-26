"""MAE-only variant of src/models/train_embedding_to_clinical.py that replaces
mean/CLS pooling with a per-patch-position PCA compression, as suggested by
the user: instead of collapsing the 196 patch tokens into one 768-dim vector
by averaging (losing all spatial detail), fit a separate PCA for each of the
196 patch positions — across the training images' 768-dim token at that
position — keeping just enough components to explain --pca-variance of that
position's variance, then concatenate all positions' compressed vectors into
one feature vector per image (sum of N_i components, N_i possibly different
per position).

This trades the "one vector, no spatial detail" of mean-pooling for "still
some detail per position, but far more features than samples" — the known
risk being severe overfitting when the concatenated feature count vastly
exceeds the ~200 training images available. Only CatBoost is tried here (no
MLP): an MLP already underperformed CatBoost on the much lower-dimensional
raw/mean embeddings (see docs/embedding-to-clinical-log.md), so it is
expected to fare even worse with thousands of features and ~200 samples.

Each per-position PCA is fit ONLY on the training filenames for that target's
split (never on test filenames), exactly as in the post-hoc PCA already
supported by train_embedding_to_clinical.py's --pca-variance.

Usage:
    python -m src.models.train_embedding_to_clinical_patch_pca --pca-variance 0.999
    python -m src.models.train_embedding_to_clinical_patch_pca --pca-variance 0.95
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.embedding_pooling import load_mae_patch_tensor
from src.preprocessing.clinical_targets import (
    load_raw_clinical, regression_targets, singlelabel_targets, multilabel_targets,
)
from src.models.train_embedding_to_clinical import (
    MIN_SAMPLES, MIN_CLASS_COUNT, make_classifiers, make_regressors,
    eval_regression, eval_classification,
)


MAE_JSON = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/embeddings_per_patch.json')
OUT_DIR = Path('outputs/embedding_to_clinical')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pca-variance', type=float, required=True,
                    help='Fraction of variance to keep per patch position (e.g. 0.999, 0.95).')
    return p.parse_args()


def build_patch_pca_features(train_filenames, test_filenames, patch_tensor: dict, variance: float):
    """Fit one PCA per patch position on the TRAINING filenames only, transform
    both splits, then concatenate all 196 positions' compressed vectors.
    Returns (X_train, X_test, n_components_per_position: list[int])."""
    n_positions = next(iter(patch_tensor.values())).shape[0]  # 196

    train_blocks, test_blocks, n_components = [], [], []
    for pos in range(n_positions):
        X_train_pos = np.stack([patch_tensor[f][pos] for f in train_filenames])  # (n_train, 768)
        X_test_pos = np.stack([patch_tensor[f][pos] for f in test_filenames])

        pca = PCA(n_components=variance, svd_solver='full', random_state=42)
        train_blocks.append(pca.fit_transform(X_train_pos))
        test_blocks.append(pca.transform(X_test_pos))
        n_components.append(pca.n_components_)

    X_train = np.concatenate(train_blocks, axis=1)
    X_test = np.concatenate(test_blocks, axis=1)
    return X_train, X_test, n_components


def run_regression_target(name, patch_tensor, y, pca_variance):
    valid = y.notna()
    valid_filenames = valid.index[valid].tolist()
    y_valid = y[valid]
    if len(y_valid) < MIN_SAMPLES:
        return None

    train_f, test_f, y_train, y_test = train_test_split(
        valid_filenames, y_valid.values, test_size=0.2, random_state=42,
    )
    X_train, X_test, n_components = build_patch_pca_features(train_f, test_f, patch_tensor, pca_variance)

    model = make_regressors()['catboost']
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)

    metrics = eval_regression(y_test, {'catboost': preds}, baseline_preds)
    return {'target': name, 'kind': 'regression', 'n_samples': len(y_valid),
            'n_features': sum(n_components), **metrics}


def run_multiclass_target(name, patch_tensor, y, pca_variance):
    valid = y.notna()
    y_valid = y[valid]

    counts = y_valid.value_counts()
    keep_classes = counts[counts >= MIN_CLASS_COUNT].index
    y_valid = y_valid[y_valid.isin(keep_classes)]

    if len(y_valid) < MIN_SAMPLES or y_valid.nunique() < 2:
        return None

    valid_filenames = y_valid.index.tolist()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_valid)

    train_f, test_f, y_train, y_test = train_test_split(
        valid_filenames, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded,
    )
    X_train, X_test, n_components = build_patch_pca_features(train_f, test_f, patch_tensor, pca_variance)

    model = make_classifiers()['catboost']
    model.fit(X_train, y_train)
    preds = np.asarray(model.predict(X_test)).ravel()
    probas = model.predict_proba(X_test)

    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_probas = baseline.predict_proba(X_test)

    metrics = eval_classification(y_test, {'catboost': (preds, probas)}, baseline_preds, baseline_probas,
                                   average='macro')
    return {'target': name, 'kind': 'multiclass', 'n_samples': len(y_valid),
            'n_features': sum(n_components), **metrics}


def run_binary_target(name, patch_tensor, y, pca_variance):
    common = [f for f in y.index if f in patch_tensor]
    y = y.loc[common]
    if len(y) < MIN_SAMPLES:
        return None

    train_f, test_f, y_train, y_test = train_test_split(
        common, y.values, test_size=0.2, random_state=42, stratify=y.values,
    )
    X_train, X_test, n_components = build_patch_pca_features(train_f, test_f, patch_tensor, pca_variance)

    model = make_classifiers()['catboost']
    model.fit(X_train, y_train)
    preds = np.asarray(model.predict(X_test)).ravel()
    probas = model.predict_proba(X_test)

    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    baseline_probas = baseline.predict_proba(X_test)

    metrics = eval_classification(y_test, {'catboost': (preds, probas)}, baseline_preds, baseline_probas,
                                   average='binary')
    return {'target': name, 'kind': 'binary', 'n_samples': len(y), 'n_features': sum(n_components), **metrics}


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Loading per-patch tensor (196 x 768 per image) from {MAE_JSON} ...')
    patch_tensor = load_mae_patch_tensor(MAE_JSON)
    raw_df = load_raw_clinical()

    rows = []
    t0 = time.time()

    for name, y in regression_targets(raw_df).items():
        print(f'[regression] {name}')
        result = run_regression_target(name, patch_tensor, y, args.pca_variance)
        if result:
            rows.append(result)
            print(f'  n_features={result["n_features"]}, delta_catboost={result["delta_catboost"]:.4f}')
        else:
            print('  skipped (too few valid samples)')

    for name, y in singlelabel_targets(raw_df).items():
        print(f'[multiclass] {name}')
        result = run_multiclass_target(name, patch_tensor, y, args.pca_variance)
        if result:
            rows.append(result)
            print(f'  n_features={result["n_features"]}, delta_catboost={result["delta_catboost"]:.4f}')
        else:
            print('  skipped (too few valid samples/classes)')

    multi_df = multilabel_targets(raw_df)
    for col in multi_df.columns:
        print(f'[binary] {col}')
        result = run_binary_target(col, patch_tensor, multi_df[col], args.pca_variance)
        if result:
            rows.append(result)
            print(f'  n_features={result["n_features"]}, delta_catboost={result["delta_catboost"]:.4f}')
        else:
            print('  skipped (too few samples)')

    results_df = pd.DataFrame(rows).sort_values('delta_catboost', ascending=False)
    variance_pct = int(round(args.pca_variance * 1000))  # e.g. 999 for 0.999, 950 for 0.95
    out_path = OUT_DIR / f'mae_patchpca{variance_pct}_results.csv'
    results_df.to_csv(out_path, index=False)

    print()
    print(f'=== MAE patch-position PCA (variance={args.pca_variance:.1%}) ===')
    cols = ['target', 'kind', 'n_samples', 'n_features', 'accuracy_baseline', 'accuracy_catboost',
            'precision_catboost', 'recall_catboost', 'f1_catboost', 'roc_auc_catboost', 'delta_catboost']
    is_classification = results_df['kind'] != 'regression'
    print(results_df[is_classification][cols].to_string(index=False))

    reg_df = results_df[~is_classification]
    if not reg_df.empty:
        print()
        print(reg_df[['target', 'n_samples', 'n_features', 'metric_baseline', 'metric_catboost',
                       'mae_catboost', 'rmse_catboost', 'delta_catboost']].to_string(index=False))

    print()
    print('Mean delta_catboost over all targets: %.4f' % results_df['delta_catboost'].mean())
    print('Mean n_features: %.0f' % results_df['n_features'].mean())
    print('Total runtime: %.1fs' % (time.time() - t0))
    print('Saved to', out_path)


if __name__ == '__main__':
    main()
