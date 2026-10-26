"""Shared pieces of the embedding -> clinical-features experiments
(src/models/train_embedding_to_clinical.py and train_embedding_to_clinical_gbm.py):
embedding loading, train-only dimensionality reduction, and ranking metrics
that return NaN instead of raising on degenerate folds."""
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize


MAE_JSON = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/embeddings_per_patch.json')
VAE_JSON = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/embeddings_vae.json')


def load_embeddings(model: str, pooling: str) -> pd.DataFrame:
    # Deferred: embedding_pooling pulls in torch (~1.3 GB RSS) via serve.inference,
    # which would otherwise be loaded again in every parallel worker process.
    from src.preprocessing.embedding_pooling import load_mae_pooled, load_vae_flat, load_vae_pooled

    if model == 'vae':
        if pooling == 'mean':
            vectors = load_vae_pooled(VAE_JSON)
        elif pooling == 'flat':
            vectors = load_vae_flat(VAE_JSON)
        else:
            raise ValueError(f"VAE supports pooling 'mean' or 'flat', got {pooling!r}")
    else:
        vectors = load_mae_pooled(MAE_JSON, pooling=pooling)

    df = pd.DataFrame.from_dict(vectors, orient='index')
    df.columns = [f'emb_{i}' for i in range(df.shape[1])]
    return df


def apply_pca(X_train, X_test, n_components):
    """Fit PCA on X_train only, transform both splits. `n_components` is an int
    (fixed dimensionality) or a float in (0, 1) (fraction of variance to keep).
    Returns (X_train, X_test, n_components_kept)."""
    pca = PCA(n_components=n_components, svd_solver='full', random_state=42)
    X_train_t = pca.fit_transform(X_train)
    X_test_t = pca.transform(X_test)
    return X_train_t, X_test_t, pca.n_components_


def apply_umap(X_train, X_test, n_components: int, random_state: int = 42):
    """Fit UMAP on X_train only (after StandardScaler, since UMAP is distance-based),
    transform both splits. Returns (X_train, X_test, n_components)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_neighbors = min(15, len(X_train) - 1)
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state, n_jobs=1)
    X_train_t = reducer.fit_transform(X_train_scaled)
    X_test_t = reducer.transform(X_test_scaled)
    return X_train_t, X_test_t, n_components


def safe_roc_auc(y_test, probas):
    """roc_auc_score can raise (e.g. a class missing from y_test in a small split);
    return NaN instead of crashing the whole sweep over targets."""
    if probas is None:
        return np.nan
    try:
        if probas.shape[1] == 2:
            return roc_auc_score(y_test, probas[:, 1])
        return roc_auc_score(y_test, probas, multi_class='ovr', average='macro',
                             labels=np.arange(probas.shape[1]))
    except ValueError:
        return np.nan


def safe_pr_auc(y_test, probas):
    """Average precision (area under the precision-recall curve). Binary: on the
    positive class. Multiclass: one-vs-rest, macro-averaged. NaN on failure."""
    if probas is None:
        return np.nan
    try:
        if probas.shape[1] == 2:
            return average_precision_score(y_test, probas[:, 1])
        y_bin = label_binarize(y_test, classes=np.arange(probas.shape[1]))
        return average_precision_score(y_bin, probas, average='macro')
    except ValueError:
        return np.nan
