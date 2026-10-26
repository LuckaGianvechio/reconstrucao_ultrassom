"""Extract clinical-feature prediction targets from the BrEaST clinical xlsx,
for the embeddings-to-clinical-features experiment (predicting clinical
features FROM image embeddings — the inverse of
src/preprocessing/build_tabular_dataset.py, which predicts embeddings FROM
clinical features and is documented to fail in ONBOARDING.md /
docs/vae-experiment-log.md).

Three kinds of targets:
- Regression: Age (numeric, "not available" -> NaN, no imputation since it's
  the target here, not a feature).
- Single-label multiclass: the raw categorical value (not one-hot), with the
  "not applicable"/"not available" sentinel turned into NaN.
- Multi-label binary: one binary target per label, reusing the same
  str.get_dummies(sep="&") logic as
  src/preprocessing/build_tabular_dataset.py::build_clinical_features (kept
  separate/duplicated here rather than importing that function directly,
  since that function fuses everything - including Age imputation aimed at
  using it as a *feature* - into one wide one-hot frame not suited for
  per-column targets).
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


CLINICAL_XLSX = Path('dataset_breast/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx')

SENTINELS = {'not available', 'not applicable'}

SINGLELABEL_COLUMNS = [
    'BIRADS', 'Shape', 'Echogenicity', 'Posterior_features', 'Halo',
    'Calcifications', 'Skin_thickening', 'Verification',
]

MULTILABEL_COLUMNS = ['Interpretation', 'Margin', 'Signs', 'Symptoms', 'Tissue_composition', 'Diagnosis']


def load_raw_clinical() -> pd.DataFrame:
    return pd.read_excel(CLINICAL_XLSX)


def regression_targets(raw_df: pd.DataFrame) -> Dict[str, pd.Series]:
    age = pd.to_numeric(raw_df['Age'].where(raw_df['Age'] != 'not available', other=np.nan), errors='coerce')
    age.index = raw_df['Image_filename']
    return {'Age': age}


def singlelabel_targets(raw_df: pd.DataFrame) -> Dict[str, pd.Series]:
    targets = {}
    for col in SINGLELABEL_COLUMNS:
        series = raw_df[col].astype(str)
        series = series.where(~series.isin(SENTINELS), other=np.nan)
        series.index = raw_df['Image_filename']
        targets[col] = series
    return targets


def multilabel_targets(raw_df: pd.DataFrame, min_positive_rate: float = 0.05,
                        max_positive_rate: float = 0.95) -> pd.DataFrame:
    """One binary (0/1) column per label across MULTILABEL_COLUMNS, filtered to
    labels whose positive rate falls within [min_positive_rate, max_positive_rate]
    (drops near-constant labels, including the sentinel-derived ones like
    "Diagnosis_not applicable" when they're too rare/common to be useful)."""
    margin_base = raw_df['Margin'].apply(lambda x: x.split(' - ')[1] if ' - ' in x else x)

    blocks = {
        'Interpretation': raw_df['Interpretation'].str.get_dummies(sep='&').add_prefix('Interpretation_'),
        'Margin': margin_base.str.get_dummies(sep='&').add_prefix('Margin_'),
        'Signs': raw_df['Signs'].str.get_dummies(sep='&').add_prefix('Signs_'),
        'Symptoms': raw_df['Symptoms'].str.get_dummies(sep='&').add_prefix('Symptoms_'),
        'Tissue_composition': raw_df['Tissue_composition'].str.get_dummies(sep='&').add_prefix('Tissue_'),
        'Diagnosis': raw_df['Diagnosis'].str.get_dummies(sep='&').add_prefix('Diagnosis_'),
    }

    combined = pd.concat(blocks.values(), axis=1)
    combined.index = raw_df['Image_filename']

    positive_rate = combined.mean(axis=0)
    keep_cols = positive_rate[(positive_rate >= min_positive_rate) & (positive_rate <= max_positive_rate)].index
    return combined[keep_cols]
