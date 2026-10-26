"""Alternative regression framing: instead of one shared model predicting all
768 embedding dimensions across all 197 patch positions (which collapses
patch-to-patch variance almost entirely - see src/evaluation/compare_token_variance.py
and the analysis that led to it), train one CatBoost MultiRMSE model PER patch
position (197 models total). Each model predicts the full 768-dim embedding at
that fixed position using only clinical features (x_pos/y_pos are no longer
needed as inputs - the position is now implicit in which of the 197 models is
used).

Evaluated via leave-one-out on a single image: each of the 197 position models
is trained on the other 255 patients and used to predict the held-out image's
embedding at that position, then all 197 predictions are compared to the real
per-patch embedding.

Usage:
    python -m src.models.train_catboost_per_position --image case001.png
"""
import argparse
import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


DATA_PATH = Path('dataset_breast/processed_dataset_embeddings.csv')
OUT_JSON = Path('dataset_breast/infered_embeddings_per_position.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='case001.png')
    args = p.parse_args()

    df = pd.read_csv(DATA_PATH)
    embedding_cols = [c for c in df.columns if c.startswith('emb_')]
    feature_cols = [c for c in df.columns if c not in embedding_cols and c not in ('patient_id', 'x_pos', 'y_pos')]
    feature_cols_clean = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)).strip('_') for c in feature_cols]
    rename_map = dict(zip(feature_cols, feature_cols_clean))

    case_id = int(''.join(filter(str.isdigit, Path(args.image).stem)))

    case_rows = df[df['patient_id'] == case_id].reset_index(drop=True)  # 197 rows, already in token order
    if case_rows.empty:
        raise ValueError(f'No rows found for {args.image} (patient_id={case_id})')

    n_tokens = len(case_rows)
    real_emb = case_rows[embedding_cols].to_numpy(dtype=np.float32)  # (197, 768)
    predicted = np.zeros_like(real_emb)

    for token_idx in range(n_tokens):
        x_pos = case_rows.loc[token_idx, 'x_pos']
        y_pos = case_rows.loc[token_idx, 'y_pos']

        position_mask = (df['x_pos'] == x_pos) & (df['y_pos'] == y_pos)
        train_mask = position_mask & (df['patient_id'] != case_id)

        X_train = df.loc[train_mask, feature_cols].rename(columns=rename_map)
        y_train = df.loc[train_mask, embedding_cols]
        X_test = case_rows.loc[[token_idx], feature_cols].rename(columns=rename_map)

        model = CatBoostRegressor(
            loss_function='MultiRMSE',
            iterations=150,
            depth=6,
            learning_rate=0.08,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False,
        )
        model.fit(X_train, y_train)
        predicted[token_idx] = np.asarray(model.predict(X_test)).reshape(-1)

        if token_idx % 20 == 0:
            print(f'  [{token_idx + 1}/{n_tokens}] position (x={x_pos}, y={y_pos}) trained on {train_mask.sum()} patients')

    mae = mean_absolute_error(real_emb, predicted)
    rmse = root_mean_squared_error(real_emb, predicted)
    r2 = r2_score(real_emb, predicted)

    print()
    print(f'=== Per-position CatBoost - {args.image} (leave-one-out) ===')
    print(f'MAE:  {mae:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'R2:   {r2:.6f}')

    std_real = real_emb.std(axis=0)
    std_pred = predicted.std(axis=0)
    ratio = std_pred / std_real
    print()
    print('Token-to-token std ratio (predicted/real), across 768 dims:')
    print(f'  mean:   {ratio.mean():.4f}')
    print(f'  median: {np.median(ratio):.4f}')

    arr = predicted.astype(np.float32)[np.newaxis, :, :]  # (1, 197, 768)
    payload = {
        'b64': base64.b64encode(arr.tobytes()).decode('ascii'),
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            existing = json.load(f)
    existing[args.image] = payload
    with open(OUT_JSON, 'w') as f:
        json.dump(existing, f, ensure_ascii=False)
    print(f'\nSaved predicted embedding to {OUT_JSON}')


if __name__ == '__main__':
    main()
