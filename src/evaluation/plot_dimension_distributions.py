"""4x4 grid comparing, for 16 evenly-spaced embedding dimensions, the
distribution of predicted (CatBoost) values across an image's 197 tokens
against the real embedding's distribution at that same dimension.

Dimensions sampled: 0, 48, 96, ..., 720 (768 / 16 = 48 apart), i.e. the 1st,
49th, 97th, ... coordinate in 1-indexed terms.

Usage:
    python -m src.evaluation.plot_dimension_distributions
    python -m src.evaluation.plot_dimension_distributions --image case002.png
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

N_DIMS_TOTAL = 768
GRID_SIZE = 4
STEP = N_DIMS_TOTAL // (GRID_SIZE * GRID_SIZE)  # 48


def load_record(path: Path, key: str) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    rec = data[key]
    arr = base64_to_embedding(rec['b64'], tuple(rec['shape']), rec.get('dtype', 'float16'))
    return arr.squeeze(0)  # (n_tokens, 768)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='case001.png')
    p.add_argument('--bins', type=int, default=20)
    args = p.parse_args()

    real_tokens = load_record(EMB_REAL, args.image)  # (197, 768)
    pred_tokens = load_record(EMB_PRED, args.image)  # (197, 768)

    dims = [i * STEP for i in range(GRID_SIZE * GRID_SIZE)]  # 0, 48, 96, ..., 720

    fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(20, 16))

    for ax, d in zip(axes.flat, dims):
        real_vals = real_tokens[:, d]
        pred_vals = pred_tokens[:, d]

        lo = min(real_vals.min(), pred_vals.min())
        hi = max(real_vals.max(), pred_vals.max())
        bins = np.linspace(lo, hi, args.bins)

        ax.hist(real_vals, bins=bins, color='black', alpha=0.4, label='real')
        ax.hist(pred_vals, bins=bins, color='blue', alpha=0.5, label='predicted (CatBoost)')
        ax.set_title(f'dim {d} (coord {d + 1})', fontsize=10)
        ax.tick_params(labelsize=7)

    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'{args.image} - distribution across 197 tokens, every {STEP}th embedding dimension', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name_noext = Path(args.image).stem
    out_path = OUT_DIR / f'{name_noext}_dimension_distributions.png'
    fig.savefig(out_path, dpi=130)
    print('Saved', out_path)


if __name__ == '__main__':
    main()
