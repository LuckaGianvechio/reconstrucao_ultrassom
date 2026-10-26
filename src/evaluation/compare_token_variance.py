"""Confirm, across all 768 embedding dimensions, that the CatBoost prediction
collapses the token-to-token (patch-to-patch) variance that the real embedding
has.

For a given image, computes std across the 197 tokens for both the real and
predicted embeddings, per dimension, and reports how much smaller the
predicted std is on average (plus a scatter plot of predicted vs real std).

Usage:
    python -m src.evaluation.compare_token_variance
    python -m src.evaluation.compare_token_variance --image case002.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from serve.inference import base64_to_embedding


DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')
EMB_REAL = DATA_DIR / 'embeddings_per_patch.json'
EMB_PRED = Path('dataset_breast/infered_embeddings.json')
OUT_DIR = Path('outputs/evaluation')


def load_record(path: Path, key: str) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    rec = data[key]
    arr = base64_to_embedding(rec['b64'], tuple(rec['shape']), rec.get('dtype', 'float16'))
    return arr.squeeze(0)  # (n_tokens, 768)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='case001.png')
    args = p.parse_args()

    real_tokens = load_record(EMB_REAL, args.image)  # (197, 768)
    pred_tokens = load_record(EMB_PRED, args.image)  # (197, 768)

    std_real = real_tokens.std(axis=0)  # (768,) variance across the 197 tokens, per dimension
    std_pred = pred_tokens.std(axis=0)  # (768,)

    ratio = std_pred / std_real  # how much of the real token-to-token variance survives in the prediction

    print(f'Image: {args.image}')
    print(f'std_real: mean={std_real.mean():.4f}, median={np.median(std_real):.4f}')
    print(f'std_pred: mean={std_pred.mean():.4f}, median={np.median(std_pred):.4f}')
    print()
    print('ratio = std_pred / std_real, across all 768 dimensions:')
    print(f'  mean:   {ratio.mean():.4f}')
    print(f'  median: {np.median(ratio):.4f}')
    print(f'  p90:    {np.percentile(ratio, 90):.4f}')
    print(f'  max:    {ratio.max():.4f}')
    print(f'  dims with ratio < 0.10: {(ratio < 0.10).sum()} / 768')
    print(f'  dims with ratio < 0.25: {(ratio < 0.25).sum()} / 768')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    lo, hi = 0, max(std_real.max(), std_pred.max())
    ax.scatter(std_real, std_pred, s=8, alpha=0.6, color='blue')
    ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=1, label='y=x (no collapse)')
    ax.set_xlabel('std across 197 tokens - real embedding')
    ax.set_ylabel('std across 197 tokens - predicted (CatBoost)')
    ax.set_title('Per-dimension token-to-token std: predicted vs real')
    ax.legend()

    ax = axes[1]
    ax.hist(ratio, bins=40, color='blue', alpha=0.7)
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, label='ratio=1 (no collapse)')
    ax.set_xlabel('std_pred / std_real')
    ax.set_ylabel('number of dimensions')
    ax.set_title('Distribution of the variance-collapse ratio (768 dims)')
    ax.legend()

    fig.suptitle(f'{args.image} - token-to-token variance collapse in CatBoost predictions')
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name_noext = Path(args.image).stem
    out_path = OUT_DIR / f'{name_noext}_token_variance_collapse.png'
    fig.savefig(out_path, dpi=140)
    print()
    print('Saved', out_path)


if __name__ == '__main__':
    main()
