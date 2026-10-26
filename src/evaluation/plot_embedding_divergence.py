"""Visualize, dimension by dimension, how a perturbed embedding and a CatBoost
prediction diverge from the real embedding of a single token.

The real embedding is plotted as a flat reference line at 0 (all 768
dimensions normalized to their own deviation-from-real, in units of that
dimension's std - the same std used by serve/perturbation_test.py). For each
dimension: a red point at the fraction=1.0 synthetic perturbation's deviation,
a blue point at the CatBoost prediction's deviation, connected by a thin line.

Usage:
    python -m src.evaluation.plot_embedding_divergence
    python -m src.evaluation.plot_embedding_divergence --image case002.png --token-index 5
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
    p.add_argument('--token-index', type=int, default=1,
                   help='0 = CLS token, 1..196 = patch tokens. Default 1 (first real patch)')
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42,
                   help='Must match the seed used by serve/perturbation_test.py to reproduce the same noise draw')
    args = p.parse_args()

    real_tokens = load_record(EMB_REAL, args.image)  # (197, 768)
    pred_tokens = load_record(EMB_PRED, args.image)  # (197, 768)

    std = real_tokens.std(axis=0)  # (768,), same reference used to calibrate perturbation fractions

    # Reproduce the exact same noise draw as serve/perturbation_test.py::build_row
    rng = np.random.default_rng(args.seed)
    noise_all = rng.normal(0.0, 1.0, size=real_tokens.shape) * (args.fraction * std)  # (197, 768)
    noise_token = noise_all[args.token_index]

    real_token = real_tokens[args.token_index]
    pred_token = pred_tokens[args.token_index]

    dims = np.arange(768)
    red_y = noise_token / std                       # perturbed deviation from real, in std units
    blue_y = (pred_token - real_token) / std         # predicted deviation from real, in std units

    fig, ax = plt.subplots(figsize=(22, 7))
    ax.vlines(dims, np.minimum(red_y, blue_y), np.maximum(red_y, blue_y), colors='gray', linewidth=0.4, alpha=0.6)
    ax.axhline(0, color='black', linewidth=1.5, label='real embedding (reference)', zorder=3)
    ax.scatter(dims, red_y, color='red', s=10, label=f'perturbed (fraction={args.fraction})', zorder=4)
    ax.scatter(dims, blue_y, color='blue', s=10, label='predicted (CatBoost)', zorder=4)

    # Light red marks on the left axis showing the mean of the red and blue points
    mean_red = red_y.mean()
    mean_blue = blue_y.mean()
    label_offsets = {'perturbed': -14, 'predicted': 14}
    for name, mean_val in (('perturbed', mean_red), ('predicted', mean_blue)):
        ax.axhline(mean_val, xmin=0.0, xmax=0.02, color='red', alpha=0.5, linewidth=3, zorder=5)
        ax.annotate(f'{name} mean={mean_val:.3f}', xy=(0, mean_val), xycoords=('axes fraction', 'data'),
                    xytext=(-8, label_offsets[name]), textcoords='offset points', ha='right', va='center',
                    color='red', alpha=0.8, fontsize=8, clip_on=False,
                    arrowprops=dict(arrowstyle='-', color='red', alpha=0.5, lw=0.7))

    ax.set_xlabel('Embedding dimension (0-767)')
    ax.set_ylabel('Deviation from real value (std units)')
    ax.set_title(f'{args.image} - token index {args.token_index} - divergence from real embedding')
    ax.legend(loc='upper right')
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name_noext = Path(args.image).stem
    out_path = OUT_DIR / f'{name_noext}_token{args.token_index}_divergence.png'
    fig.savefig(out_path, dpi=150)
    print('Saved', out_path)


if __name__ == '__main__':
    main()
