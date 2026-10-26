"""Is a 'history' target (signs, symptoms) a clean negative control for the
embedding -> clinical experiments? Predict it from the specialist's IMAGE
annotations alone (BIRADS, Shape, Echogenicity, Posterior_features, Halo,
Calcifications, Skin_thickening, Margin_*, Interpretation_*, Tissue_*, one-hot),
with the same folds and CatBoost as the embedding runs.

If the annotations already predict it, the target is linked to visible lesion
features and an embedding predicting it is not evidence of a confound. Only the
excess of the embedding over the annotations is.

Writes history_from_annotations.csv into the given results folder.

Usage:
    python -m src.evaluation.history_from_annotations --folder-name acquisition_confound
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.models.train_embedding_to_clinical_gbm import (
    IMAGE_PREFIXES, MIN_CLASS_COUNT, OUT_ROOT, SENTINEL_MARKERS, build_targets, make_models, score, target_group,
)
from src.preprocessing.clinical_targets import load_raw_clinical, multilabel_targets, singlelabel_targets

IMAGE_SINGLELABEL = ['BIRADS', 'Shape', 'Echogenicity', 'Posterior_features', 'Halo', 'Calcifications',
                     'Skin_thickening']


def image_annotation_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    single = pd.DataFrame(singlelabel_targets(raw_df))[IMAGE_SINGLELABEL]
    multi = multilabel_targets(raw_df, min_positive_rate=0, max_positive_rate=1)
    multi = multi[[c for c in multi.columns
                   if c.startswith(IMAGE_PREFIXES) and not any(m in c for m in SENTINEL_MARKERS)]]
    return pd.get_dummies(single, dtype=int).join(multi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder-name', required=True)
    p.add_argument('--n-splits', type=int, default=5)
    args = p.parse_args()

    raw_df = load_raw_clinical()
    X_ann = image_annotation_features(raw_df)
    rows = []
    for name, kind, y in build_targets(raw_df, X_ann.index, max(MIN_CLASS_COUNT, args.n_splits)):
        if target_group(name) != 'history' or kind == 'regression':
            continue
        X, yy = X_ann.loc[y.index].to_numpy(float), y.to_numpy()
        folds = StratifiedKFold(args.n_splits, shuffle=True, random_state=42).split(yy, yy)
        aucs = [score(kind, yy[test], make_models(kind)['catboost'].fit(X[train], yy[train]), X[test])['roc_auc']
                for train, test in folds]
        rows.append({'target': name, 'n_features': X.shape[1], 'annotations_roc_auc_mean': np.mean(aucs),
                     'annotations_roc_auc_std': np.std(aucs, ddof=1)})

    res = pd.DataFrame(rows)
    res.to_csv(OUT_ROOT / args.folder_name / 'history_from_annotations.csv', index=False)
    print(res.to_string(index=False, float_format='%.3f'))


if __name__ == '__main__':
    main()
