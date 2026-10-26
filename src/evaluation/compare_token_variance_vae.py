"""VAE analogue of src/evaluation/compare_token_variance.py.

Confirms, across all 4 VAE latent channels, whether the CatBoost prediction
collapses the token-to-token (spatial-position-to-position) variance that
the real latent has — same diagnostic that revealed the core problem for
the MAE embeddings (see ONBOARDING.md), now repeated for the VAE latent
space to check if the same structural issue (position alone not being
predictive of content across different patients) shows up here too.

Usage:
    python -m src.evaluation.compare_token_variance_vae
    python -m src.evaluation.compare_token_variance_vae --image case002.png
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.preprocessing.build_tabular_dataset_vae import decode_embedding_record, flatten_latent


DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')
EMB_REAL = DATA_DIR / 'embeddings_vae.json'
EMB_PRED = Path('dataset_breast/infered_embeddings_vae.json')
OUT_DIR = Path('outputs/evaluation')

EMBEDDING_DIM = 4


def load_real_tokens(path: Path, key: str) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    latent = decode_embedding_record(data[key])  # (1, 4, H, W)
    tokens, _, _ = flatten_latent(latent)
    return tokens  # (n_tokens, 4)


def load_pred_tokens(path: Path, key: str) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    rec = data[key]
    arr = np.frombuffer(base64.b64decode(rec['b64']), dtype=np.dtype(rec.get('dtype', 'float32')))
    arr = arr.reshape(tuple(rec['shape'])).astype(np.float32)
    return arr.squeeze(0)  # (n_tokens, 4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='case001.png')
    args = p.parse_args()

    real_tokens = load_real_tokens(EMB_REAL, args.image)  # (n_tokens, 4)
    pred_tokens = load_pred_tokens(EMB_PRED, args.image)  # (n_tokens, 4)

    std_real = real_tokens.std(axis=0)  # (4,) variance across spatial positions, per channel
    std_pred = pred_tokens.std(axis=0)  # (4,)

    ratio = std_pred / std_real

    print(f'Image: {args.image}')
    print(f'std_real (per channel): {std_real}')
    print(f'std_pred (per channel): {std_pred}')
    print()
    print('ratio = std_pred / std_real, per channel:', ratio)
    print(f'  mean:   {ratio.mean():.4f}')
    print(f'  max:    {ratio.max():.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    lo, hi = 0, max(std_real.max(), std_pred.max())
    ax.scatter(std_real, std_pred, s=40, alpha=0.8, color='blue')
    ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=1, label='y=x (no collapse)')
    ax.set_xlabel('std across spatial positions - real latent')
    ax.set_ylabel('std across spatial positions - predicted (CatBoost)')
    ax.set_title('Per-channel position-to-position std: predicted vs real')
    ax.legend()

    ax = axes[1]
    ax.bar(range(EMBEDDING_DIM), ratio, color='blue', alpha=0.7)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='ratio=1 (no collapse)')
    ax.set_xlabel('latent channel')
    ax.set_ylabel('std_pred / std_real')
    ax.set_title('Variance-collapse ratio per VAE latent channel')
    ax.legend()

    fig.suptitle(f'{args.image} - position-to-position variance collapse in CatBoost predictions (VAE latent)')
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name_noext = Path(args.image).stem
    out_path = OUT_DIR / f'{name_noext}_token_variance_collapse_vae.png'
    fig.savefig(out_path, dpi=140)
    print()
    print('Saved', out_path)


if __name__ == '__main__':
    main()
