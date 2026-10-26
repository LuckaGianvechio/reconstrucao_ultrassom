"""Confound check for the embedding -> clinical sweep: how well can each target be
predicted from acquisition metadata alone (image width, height and Pixel_size,
which act as a proxy for the ultrasound device / export protocol), with the same
folds and CatBoost setup as src/models/train_embedding_to_clinical_gbm.py?

A target the embedding predicts no better than this is not evidence of image
content. Writes metadata_only_baseline.csv next to the sweep's results.

Usage:
    python -m src.evaluation.metadata_only_baseline --folder-name gbm_pca_umap_sweep
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from src.models.train_embedding_to_clinical_gbm import (
    MIN_CLASS_COUNT, OUT_ROOT, build_targets, make_models, score, target_group,
)
from src.preprocessing.clinical_targets import load_raw_clinical

IMAGES_DIR = 'dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/'

FEATURE_SETS = {
    'metadata': ['width', 'height', 'pixel_size'],
    'size': ['width', 'height'],
    'pixel_size': ['pixel_size'],
}


def load_metadata(raw_df: pd.DataFrame) -> pd.DataFrame:
    meta = pd.DataFrame({f: Image.open(IMAGES_DIR + f).size for f in raw_df['Image_filename']},
                        index=['width', 'height']).T
    meta['pixel_size'] = pd.to_numeric(raw_df.set_index('Image_filename')['Pixel_size'], errors='coerce')
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder-name', required=True)
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--compare-embedding', default='mae_cls')
    args = p.parse_args()
    folder = OUT_ROOT / args.folder_name

    raw_df = load_raw_clinical()
    meta = load_metadata(raw_df)

    rows = []
    for name, kind, y in build_targets(raw_df, meta.index, max(MIN_CLASS_COUNT, args.n_splits)):
        if kind == 'regression':
            continue
        yy = y.to_numpy()
        folds = list(StratifiedKFold(args.n_splits, shuffle=True, random_state=42).split(yy, yy))
        row = {'target': name, 'target_group': target_group(name)}
        for set_name, cols in FEATURE_SETS.items():
            X = meta.loc[y.index, cols].to_numpy(float)
            scores = [score(kind, yy[test], make_models(kind)['catboost'].fit(X[train], yy[train]), X[test])
                      for train, test in folds]
            for metric in ('roc_auc', 'pr_auc'):
                values = [s[metric] for s in scores]
                row[f'{set_name}_{metric}'] = np.mean(values)
                row[f'{set_name}_{metric}_std'] = np.std(values, ddof=1)
        rows.append(row)

    long_df = pd.read_csv(folder / 'results_long.csv')
    raw = long_df[(long_df['reducer'] == 'raw') & (long_df['embedding'] == args.compare_embedding)]
    emb = raw[raw['model'] == 'catboost'][['target', 'roc_auc_mean', 'roc_auc_std', 'pr_auc_mean']].rename(
        columns={'roc_auc_mean': 'embedding_roc_auc', 'roc_auc_std': 'embedding_roc_auc_std',
                 'pr_auc_mean': 'embedding_pr_auc'})
    dummy = raw[raw['model'] == 'baseline'][['target', 'pr_auc_mean']].rename(
        columns={'pr_auc_mean': 'dummy_pr_auc'})
    res = pd.DataFrame(rows).merge(emb, on='target').merge(dummy, on='target')
    res['embedding_minus_metadata'] = res['embedding_roc_auc'] - res['metadata_roc_auc']
    res = res.sort_values(['target_group', 'embedding_minus_metadata'], ascending=[True, False])
    res.to_csv(folder / 'metadata_only_baseline.csv', index=False)

    summary_cols = ['dummy_pr_auc', 'size_roc_auc', 'pixel_size_roc_auc', 'metadata_roc_auc', 'embedding_roc_auc',
                    'embedding_minus_metadata', 'size_pr_auc', 'pixel_size_pr_auc', 'metadata_pr_auc',
                    'embedding_pr_auc']
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(res[['target', 'target_group'] + summary_cols].to_string(index=False, float_format='%.3f'))
        print(res.groupby('target_group')[summary_cols].mean().round(3).to_string())


if __name__ == '__main__':
    main()
