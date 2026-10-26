"""Pool per-image embeddings (MAE per-patch tokens, VAE spatial latent) into a
single vector per image, for the embeddings-to-clinical-features experiment
(see src/models/train_embedding_to_clinical.py).

MAE mean-pooling reproduces the exact convention already used in
serve/inference.py::encode_image (`enc[:, 1:, :].mean(dim=1)`, i.e. mean over
the 196 patch tokens, excluding the CLS token at index 0). CLS-pooling just
takes that index-0 token directly.

The VAE latent has no CLS-like token, so only mean (global average) pooling
is supported for it.

IMPORTANT - MAE token order. MAE's `random_masking` shuffles the patch sequence
with `argsort(rand)` even at mask_ratio=0.0 (see mae/models_mae.py), and the
shuffle is drawn PER IMAGE. So token index `i` in embeddings_per_patch.json is a
DIFFERENT spatial patch in every image. `serve/generate_embeddings.py` saves the
`ids_restore` permutation that undoes this, and anything that treats a token
index as a spatial position MUST apply it first (see `restore_spatial_order`).

Mean- and CLS-pooling are exempt: the mean over patches is permutation-invariant,
and the CLS token is prepended AFTER masking, so it is always index 0. Do not
"fix" `load_mae_pooled` - its output is already correct.
"""
import json
from pathlib import Path
from typing import Dict, Literal

import numpy as np

from serve.inference import base64_to_embedding
from src.preprocessing.mae_tokens import restore_spatial_order
from src.preprocessing.build_tabular_dataset_vae import decode_embedding_record, flatten_latent


def load_mae_pooled(json_path: Path, pooling: Literal['mean', 'cls'] = 'mean') -> Dict[str, np.ndarray]:
    """Load embeddings_per_patch.json and pool each image's (1, 197, 768) tokens
    into a single 768-dim vector."""
    with open(json_path) as f:
        records = json.load(f)

    pooled = {}
    for filename, rec in records.items():
        tokens = base64_to_embedding(rec['b64'], tuple(rec['shape']), rec.get('dtype', 'float16'))
        tokens = tokens.squeeze(0)  # (197, 768)
        if pooling == 'mean':
            pooled[filename] = tokens[1:].mean(axis=0)
        elif pooling == 'cls':
            pooled[filename] = tokens[0]
        else:
            raise ValueError(f"pooling must be 'mean' or 'cls' for MAE, got {pooling!r}")
    return pooled


def load_mae_patch_tensor(json_path: Path) -> Dict[str, np.ndarray]:
    """Load embeddings_per_patch.json without pooling: each image's 196 patch
    tokens (CLS at index 0 excluded), kept as a (196, 768) array IN SPATIAL ORDER.
    Used by src/models/train_embedding_to_clinical_patch_pca.py, which compresses
    each patch position separately via PCA instead of collapsing all positions
    with a single mean/CLS vector.

    The spatial reordering is essential here and was missing originally: without
    it, "position i" is a different spatial patch in every image, so fitting a PCA
    across images at a fixed position would mix unrelated image regions."""
    with open(json_path) as f:
        records = json.load(f)

    tensors = {}
    for filename, rec in records.items():
        tokens = base64_to_embedding(rec['b64'], tuple(rec['shape']), rec.get('dtype', 'float16'))
        spatial = restore_spatial_order(tokens.squeeze(0), rec.get('ids_restore'))
        tensors[filename] = spatial[1:]  # (196, 768), CLS dropped
    return tensors


def load_vae_pooled(json_path: Path) -> Dict[str, np.ndarray]:
    """Load embeddings_vae.json and mean-pool each image's (1, 4, H, W) latent
    into a single 4-dim vector (global average per channel)."""
    with open(json_path) as f:
        records = json.load(f)

    pooled = {}
    for filename, rec in records.items():
        latent = decode_embedding_record(rec)  # (1, 4, H, W)
        tokens, _, _ = flatten_latent(latent)  # (H*W, 4)
        pooled[filename] = tokens.mean(axis=0)
    return pooled


def load_vae_flat(json_path: Path) -> Dict[str, np.ndarray]:
    """Load embeddings_vae.json and flatten each image's (1, 4, H, W) latent into a
    single 4*H*W vector. Valid as a fixed-length feature vector because the VAE
    latent is spatially aligned across images (no token shuffling, unlike MAE)."""
    with open(json_path) as f:
        records = json.load(f)

    return {filename: decode_embedding_record(rec).reshape(-1) for filename, rec in records.items()}
