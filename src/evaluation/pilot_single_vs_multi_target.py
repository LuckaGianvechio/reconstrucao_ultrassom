"""Pilot test: for a single fixed patch position, compare one CatBoost MultiRMSE
model (768 targets jointly) against 768 independent single-target CatBoost
models, using the same patient-level 80/20 train/test split as
src/models/train_catboost.py.

This is a cheap (768 fits, not 197x768) way to check whether fragmenting the
per-position regressor further, into one model per (position, dimension) pair,
would actually help - before committing to the full 197x768 experiment.

Usage:
    python -m src.evaluation.pilot_single_vs_multi_target
    python -m src.evaluation.pilot_single_vs_multi_target --patient-id 1 --token-index 1
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path('dataset_breast/processed_dataset_embeddings.csv')

CATBOOST_PARAMS = dict(iterations=150, depth=6, learning_rate=0.08, l2_leaf_reg=3.0, random_seed=42, verbose=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patient-id', type=int, default=1, help='Whose token position to use as the pilot position')
    p.add_argument('--token-index', type=int, default=1, help='0=CLS, 1..196=patch tokens')
    args = p.parse_args()

    df = pd.read_csv(DATA_PATH)
    embedding_cols = [c for c in df.columns if c.startswith('emb_')]
    feature_cols = [c for c in df.columns if c not in embedding_cols and c not in ('patient_id', 'x_pos', 'y_pos')]
    feature_cols_clean = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)).strip('_') for c in feature_cols]
    rename_map = dict(zip(feature_cols, feature_cols_clean))

    ref_rows = df[df['patient_id'] == args.patient_id].reset_index(drop=True)
    x_pos, y_pos = ref_rows.loc[args.token_index, ['x_pos', 'y_pos']]
    print(f'Pilot position: x_pos={x_pos}, y_pos={y_pos} (from patient {args.patient_id}, token {args.token_index})')

    position_df = df[(df['x_pos'] == x_pos) & (df['y_pos'] == y_pos)].reset_index(drop=True)
    print(f'Rows at this position: {len(position_df)} (one per patient)')

    unique_patients = df['patient_id'].drop_duplicates().to_numpy()
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42, shuffle=True)

    train_mask = position_df['patient_id'].isin(train_patients)
    test_mask = position_df['patient_id'].isin(test_patients)

    X = position_df[feature_cols].rename(columns=rename_map)
    y = position_df[embedding_cols]

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    print(f'Train: {len(X_train)} patients | Test: {len(X_test)} patients')

    # Model A: one MultiRMSE model, 768 targets jointly
    model_multi = CatBoostRegressor(loss_function='MultiRMSE', **CATBOOST_PARAMS)
    model_multi.fit(X_train, y_train)
    pred_multi = model_multi.predict(X_test)

    mae_multi = mean_absolute_error(y_test, pred_multi)
    rmse_multi = root_mean_squared_error(y_test, pred_multi)
    r2_multi = r2_score(y_test, pred_multi)

    # Model B: 768 independent single-target models
    pred_single = np.zeros((len(X_test), len(embedding_cols)), dtype=np.float64)
    for i, col in enumerate(embedding_cols):
        model = CatBoostRegressor(loss_function='RMSE', **CATBOOST_PARAMS)
        model.fit(X_train, y_train[col])
        pred_single[:, i] = model.predict(X_test)
        if i % 128 == 0:
            print(f'  [{i + 1}/{len(embedding_cols)}] single-target models trained')

    mae_single = mean_absolute_error(y_test, pred_single)
    rmse_single = root_mean_squared_error(y_test, pred_single)
    r2_single = r2_score(y_test, pred_single)

    print()
    print(f'=== Position (x={x_pos}, y={y_pos}) - {len(X_test)} test patients ===')
    print(f'{"":20s} {"MAE":>10s} {"RMSE":>10s} {"R2":>10s}')
    print(f'{"MultiRMSE (1 model)":20s} {mae_multi:10.6f} {rmse_multi:10.6f} {r2_multi:10.6f}')
    print(f'{"768 single-target":20s} {mae_single:10.6f} {rmse_single:10.6f} {r2_single:10.6f}')


if __name__ == '__main__':
    main()
