"""Test TabPFN (via the PriorLabs cloud API) on a sample of embedding
dimensions, using the same single-table framing as src/models/train_catboost.py
(one row per patch token, x_pos/y_pos kept as features) - but scoped to a
handful of dimensions because of the API's daily credit quota (~23 full-context
dimensions/day on the free tier).

Trains on all patients except one held-out image (case001 by default) and
predicts that image's embedding at the sampled dimensions, leave-one-out style,
matching src/models/train_catboost_per_position.py's evaluation setup so
results are directly comparable.

Usage:
    export TABPFN_TOKEN="..."
    python -m src.models.train_tabpfn_sample --image case001.png
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from tabpfn_client import TabPFNRegressor, init


DATA_PATH = Path('dataset_breast/processed_dataset_embeddings.csv')
N_DIMS_TOTAL = 768
SAMPLE_STEP = 48  # matches src/evaluation/plot_dimension_distributions.py: dims 0, 48, 96, ..., 720


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='case001.png')
    args = p.parse_args()

    init()

    df = pd.read_csv(DATA_PATH)
    embedding_cols = [c for c in df.columns if c.startswith('emb_')]
    feature_cols = [c for c in df.columns if c not in embedding_cols and c != 'patient_id']
    feature_cols_clean = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)).strip('_') for c in feature_cols]
    rename_map = dict(zip(feature_cols, feature_cols_clean))

    case_id = int(''.join(filter(str.isdigit, Path(args.image).stem)))
    train_mask = df['patient_id'] != case_id
    test_mask = df['patient_id'] == case_id

    X_train = df.loc[train_mask, feature_cols].rename(columns=rename_map)
    X_test = df.loc[test_mask, feature_cols].rename(columns=rename_map)
    print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')

    sample_dims = list(range(0, N_DIMS_TOTAL, SAMPLE_STEP))  # 0, 48, ..., 720 (16 dims)
    sample_cols = [f'emb_{d}' for d in sample_dims]

    real = df.loc[test_mask, sample_cols].to_numpy(dtype=np.float32)  # (197, 16)
    predicted = np.zeros_like(real)

    for i, (d, col) in enumerate(zip(sample_dims, sample_cols)):
        t0 = time.time()
        model = TabPFNRegressor()
        model.fit(X_train, df.loc[train_mask, col])
        pred = model.predict(X_test)
        predicted[:, i] = pred
        print(f'  [{i + 1}/{len(sample_dims)}] dim {d} done in {time.time() - t0:.1f}s')

    mae = mean_absolute_error(real, predicted)
    rmse = root_mean_squared_error(real, predicted)
    r2 = r2_score(real, predicted)

    std_real = real.std(axis=0)
    std_pred = predicted.std(axis=0)
    ratio = std_pred / std_real

    print()
    print(f'=== TabPFN (sample of {len(sample_dims)} dims) - {args.image} (leave-one-out) ===')
    print(f'MAE:  {mae:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'R2:   {r2:.6f}')
    print()
    print('Token-to-token std ratio (predicted/real), across sampled dims:')
    print(f'  mean:   {ratio.mean():.4f}')
    print(f'  median: {np.median(ratio):.4f}')
    print(f'  per-dim: {dict(zip(sample_dims, np.round(ratio, 3)))}')

    out_path = Path('outputs/evaluation') / f'{Path(args.image).stem}_tabpfn_sample_predictions.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predicted, columns=sample_cols).to_csv(out_path, index=False)
    print(f'\nSaved predictions to {out_path}')


if __name__ == '__main__':
    main()
