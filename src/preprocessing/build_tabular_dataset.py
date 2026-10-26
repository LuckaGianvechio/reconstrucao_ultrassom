"""Build dataset_breast/processed_dataset_embeddings.csv.

Reproduces the clinical one-hot preprocessing from notebooks/data_exploration.ipynb
and merges it, per token, with the per-patch MAE embeddings produced by
serve/generate_embeddings.py (embeddings_per_patch.json). Each ultrasound image
contributes 197 rows (1 CLS token + 196 patch tokens for the 14x14 grid at
patch16/224px), all sharing the same clinical features but carrying their own
768-dim embedding (emb_0..emb_767) and patch position (x_pos, y_pos; CLS is -1,-1).

IMPORTANT: MAE shuffles its patch sequence per image (random_masking uses
argsort(rand) even at mask_ratio=0.0), so a token's index is NOT its spatial
position. Tokens are restored to spatial order with the saved `ids_restore`
before x_pos/y_pos are assigned - see
src/preprocessing/mae_tokens.py::restore_spatial_order. Without that
step, x_pos/y_pos label a different image region in every row.

Usage:
    python -m src.preprocessing.build_tabular_dataset
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.mae_tokens import restore_spatial_order


CLINICAL_XLSX = Path('dataset_breast/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx')
DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')
EMBEDDINGS_JSON = DATA_DIR / 'embeddings_per_patch.json'
OUT_CSV = Path('dataset_breast/processed_dataset_embeddings.csv')

PATCH_GRID_SIZE = 14  # 224 / patch16


def build_clinical_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """One-hot/impute clinical columns, same transformations as data_exploration.ipynb."""
    inter_dummies = raw_df["Interpretation"].str.get_dummies(sep="&").add_prefix("Interpretation_")
    marg_dummies = (raw_df["Margin"]
                    .apply(lambda x: x.split(" - ")[1] if " - " in x else x)
                    .str.get_dummies(sep="&").add_prefix("Margin_"))
    sigs_dummies = raw_df["Signs"].str.get_dummies(sep="&").add_prefix("Signs_")
    symp_dummies = raw_df["Symptoms"].str.get_dummies(sep="&").add_prefix("Symptoms_")
    tis_dummies = raw_df["Tissue_composition"].str.get_dummies(sep="&").add_prefix("Tissue_")
    diag_dummies = raw_df["Diagnosis"].str.get_dummies(sep="&").add_prefix("Diagnosis_")

    useful_raw_features = ["Shape", "Echogenicity", "Posterior_features", "Halo",
                            "Calcifications", "Skin_thickening", "BIRADS", "Verification"]
    class_features_dummies = pd.get_dummies(raw_df[useful_raw_features], dtype=int)

    age_numeric = pd.to_numeric(raw_df["Age"].replace("not available", pd.NA), errors="coerce")
    median_age = age_numeric.median()
    age_without_na = age_numeric.fillna(median_age)
    age_na_dummy = raw_df["Age"].eq("not available").astype(int).rename("Age_NA")

    processed_df = pd.concat([
        pd.DataFrame({"Age": age_without_na, "Age_NA": age_na_dummy}),
        class_features_dummies,
        inter_dummies,
        marg_dummies,
        sigs_dummies,
        symp_dummies,
        tis_dummies,
        diag_dummies,
    ], axis=1)

    bool_cols = processed_df.select_dtypes(include=['bool']).columns
    processed_df[bool_cols] = processed_df[bool_cols].astype(int)

    return processed_df


def decode_embedding_record(rec) -> np.ndarray:
    shape = tuple(rec['shape'])
    dtype = rec.get('dtype', 'float16')
    arr = np.frombuffer(base64.b64decode(rec['b64']), dtype=np.dtype(dtype)).reshape(shape)
    return arr.astype(np.float32)


def token_positions(n_tokens: int):
    """CLS token (index 0) gets (-1, -1); patch tokens get their (col, row) in the 14x14 grid."""
    patch_idx = np.arange(n_tokens) - 1
    x_pos = np.where(np.arange(n_tokens) == 0, -1, patch_idx % PATCH_GRID_SIZE)
    y_pos = np.where(np.arange(n_tokens) == 0, -1, patch_idx // PATCH_GRID_SIZE)
    return x_pos, y_pos


def main():
    raw_df = pd.read_excel(CLINICAL_XLSX)
    clinical_features = build_clinical_features(raw_df)

    with open(EMBEDDINGS_JSON) as f:
        embeddings = json.load(f)

    emb_cols = [f'emb_{d}' for d in range(768)]
    per_image_dfs = []
    skipped = 0

    for i, row in raw_df.iterrows():
        filename = row['Image_filename']
        if filename not in embeddings:
            skipped += 1
            continue

        rec = embeddings[filename]
        tokens = decode_embedding_record(rec).squeeze(0)  # (n_tokens, 768), MAE-shuffled
        tokens = restore_spatial_order(tokens, rec.get('ids_restore'))
        n_tokens = tokens.shape[0]
        x_pos, y_pos = token_positions(n_tokens)

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
