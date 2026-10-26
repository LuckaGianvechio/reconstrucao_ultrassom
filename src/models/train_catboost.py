"""Train a CatBoost multi-output regressor to predict per-patch MAE embeddings
from tabular clinical features.

Standalone reproduction of the CatBoost cell in notebooks/predictions.ipynb (same
hyperparameters, same patient-level train/test split, same evaluation metrics),
scoped to CatBoost only.

Usage:
    python -m src.models.train_catboost
"""
import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


DATA_PATH = Path('dataset_breast/processed_dataset_embeddings.csv')
MODEL_OUT = Path('catboost_embeddings.joblib')
INFERRED_JSON = Path('dataset_breast/infered_embeddings.json')


def evaluate_regressor(model, X_test, y_test, name):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'=== {name} ===')
    print(f'MAE: {mae:.6f}')
    print(f'RMSE: {rmse:.6f}')
    print(f'R2: {r2:.6f}')
    return preds


def main():
    df = pd.read_csv(DATA_PATH)
    print(f'Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas')
    print(f'Pacientes unicos: {df["patient_id"].nunique()}')
    print(f'Linhas por paciente: min={df.groupby("patient_id").size().min()}, '
          f'max={df.groupby("patient_id").size().max()}')

    embedding_cols = [c for c in df.columns if c.startswith('emb_')]
    feature_cols = [c for c in df.columns if c not in embedding_cols and c != 'patient_id']
    feature_cols_clean = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)).strip('_') for c in feature_cols]
    rename_map = dict(zip(feature_cols, feature_cols_clean))

    X = df[feature_cols].copy().rename(columns=rename_map)
    y = df[embedding_cols].copy().astype(float)

    unique_patients = df['patient_id'].drop_duplicates().to_numpy()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=42, shuffle=True,
    )
    train_mask = df['patient_id'].isin(train_patients)
    test_mask = df['patient_id'].isin(test_patients)

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    print(f'Treino: {len(X_train)} linhas | Teste: {len(X_test)} linhas')
    print(f'Pacientes treino: {len(train_patients)} | Pacientes teste: {len(test_patients)}')

    base_cat = CatBoostRegressor(
        loss_function='MultiRMSE',
        iterations=150,
        depth=6,
        learning_rate=0.08,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False,
    )
    model = MultiOutputRegressor(base_cat, n_jobs=-1)
    model.fit(X_train, y_train)

    evaluate_regressor(model, X_test, y_test, 'CatBoost')

    joblib.dump(model, MODEL_OUT)
    print(f'Modelo salvo em {MODEL_OUT}')

    # Predicted embeddings for every patient, in the same JSON schema as
    # embeddings_per_patch.json, for use by serve/reconstruct_and_compile.py
    all_predictions = {}
    for patient_id in sorted(df['patient_id'].unique()):
        patient_rows = df[df['patient_id'] == patient_id]
        patient_X = X.loc[patient_rows.index]
        predicted = model.predict(patient_X)

        arr = np.asarray(predicted, dtype=np.float32)[np.newaxis, :, :]  # (1, n_tokens, 768)
        payload = {
            'b64': base64.b64encode(arr.tobytes()).decode('ascii'),
            'shape': list(arr.shape),
            'dtype': str(arr.dtype),
        }
        all_predictions[f'case{int(patient_id):03d}.png'] = payload

    INFERRED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(INFERRED_JSON, 'w') as f:
        json.dump(all_predictions, f, ensure_ascii=False)
    print(f'Previsoes salvas em {INFERRED_JSON} ({len(all_predictions)} imagens)')


if __name__ == '__main__':
    main()
