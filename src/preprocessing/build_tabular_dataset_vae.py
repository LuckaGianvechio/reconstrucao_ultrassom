"""Build dataset_breast/processed_dataset_embeddings_vae.csv.

VAE analogue of src/preprocessing/build_tabular_dataset.py: reuses the same
clinical one-hot preprocessing, but merges it with the fixed-size SD-VAE
latent embeddings produced by serve/generate_embeddings_vae.py
(embeddings_vae.json) instead of the MAE per-patch embeddings.

Unlike the MAE case, the VAE latent has no CLS-like summary token — it is a
purely spatial (4, H', W') grid (H'=W'=32 for the default 256x256 fixed
size). Each image contributes H'*W' rows (one per spatial position), all
sharing the same clinical features but carrying their own 4-dim embedding
(emb_0..emb_3) and grid position (x_pos, y_pos).

Usage:
    python -m src.preprocessing.build_tabular_dataset_vae
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.build_tabular_dataset import build_clinical_features


CLINICAL_XLSX = Path('dataset_breast/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx')
DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')
EMBEDDINGS_JSON = DATA_DIR / 'embeddings_vae.json'
OUT_CSV = Path('dataset_breast/processed_dataset_embeddings_vae.csv')

EMBEDDING_DIM = 4  # VAE latent channels


def decode_embedding_record(rec) -> np.ndarray:
    shape = tuple(rec['shape'])
    dtype = rec.get('dtype', 'float16')
    arr = np.frombuffer(base64.b64decode(rec['b64']), dtype=np.dtype(dtype)).reshape(shape)
    return arr.astype(np.float32)


def flatten_latent(latent: np.ndarray):
    """latent: (1, C, H, W) -> (tokens (H*W, C), x_pos (H*W,), y_pos (H*W,)).

    Row-major flattening: token index i = row * W + col, so x_pos=col, y_pos=row.
    """
    channels = latent.squeeze(0)  # (C, H, W)
    _, h, w = channels.shape
    tokens = channels.transpose(1, 2, 0).reshape(h * w, -1)  # (H*W, C)
    y_pos, x_pos = np.divmod(np.arange(h * w), w)
    return tokens, x_pos, y_pos


def main():
    raw_df = pd.read_excel(CLINICAL_XLSX)
    clinical_features = build_clinical_features(raw_df)

    with open(EMBEDDINGS_JSON) as f:
        embeddings = json.load(f)

    emb_cols = [f'emb_{d}' for d in range(EMBEDDING_DIM)]
    per_image_dfs = []
    skipped = 0

    for i, row in raw_df.iterrows():
        filename = row['Image_filename']
        if filename not in embeddings:
            skipped += 1
            continue

        latent = decode_embedding_record(embeddings[filename])  # (1, 4, H, W)
        tokens, x_pos, y_pos = flatten_latent(latent)
        n_tokens = tokens.shape[0]

        clinical_block = pd.DataFrame(
            np.tile(clinical_features.loc[i].to_numpy(), (n_tokens, 1)),
            columns=clinical_features.columns,
        )

        combined = pd.concat([
            pd.DataFrame({'patient_id': row['CaseID'], 'x_pos': x_pos, 'y_pos': y_pos}),
            clinical_block,
            pd.DataFrame(tokens, columns=emb_cols),
        ], axis=1)
        per_image_dfs.append(combined)

    if skipped:
        print(f'Skipped {skipped} images with no embedding in {EMBEDDINGS_JSON}')

    df = pd.concat(per_image_dfs, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'Wrote {OUT_CSV} with {len(df)} rows, {df.shape[1]} columns, '
          f'{df["patient_id"].nunique()} patients')


if __name__ == '__main__':
    main()
