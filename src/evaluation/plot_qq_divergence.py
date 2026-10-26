"""QQ plot (against a standard normal) of the perturbed vs. predicted deviations
from the real embedding, for a single token.

Both series are already expressed in std units (deviation from the real value,
divided by that dimension's std - same convention as plot_embedding_divergence.py).
Since the synthetic perturbation (fraction=1.0) is drawn i.i.d. from
N(0, std_d^2), its deviation in std units is ~N(0, 1): its QQ plot against a
standard normal should hug the y=x reference line closely. Comparing the
CatBoost prediction's QQ curve against that same reference shows whether its
per-dimension error behaves like unstructured Gaussian noise or departs from
it (heavier/lighter tails, skew).

Usage:
    python -m src.evaluation.plot_qq_divergence --image case001.png --token-index 0
    python -m src.evaluation.plot_qq_divergence --image case001.png --token-index 1
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
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


def compute_deviations(image_name, token_index, fraction, seed):
    real_tokens = load_record(EMB_REAL, image_name)
    pred_tokens = load_record(EMB_PRED, image_name)
    std = real_tokens.std(axis=0)

    rng = np.random.default_rng(seed)
    noise_all = rng.normal(0.0, 1.0, size=real_tokens.shape) * (fraction * std)

    red_y = noise_all[token_index] / std
    blue_y = (pred_tokens[token_index] - real_tokens[token_index]) / std
    return red_y, blue_y


def plot_qq(image_name, token_index, fraction, seed):
    red_y, blue_y = compute_deviations(image_name, token_index, fraction, seed)

    osm_red, osr_red = stats.probplot(red_y, dist='norm', fit=False)
    osm_blue, osr_blue = stats.probplot(blue_y, dist='norm', fit=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(osm_red, osr_red, color='red', s=12, alpha=0.7, label=f'perturbed (fraction={fraction})')
    ax.scatter(osm_blue, osr_blue, color='blue', s=12, alpha=0.7, label='predicted (CatBoost)')

    lo = min(osm_red.min(), osm_blue.min(), osr_red.min(), osr_blue.min())
    hi = max(osm_red.max(), osm_blue.max(), osr_red.max(), osr_blue.max())
    ax.plot([lo, hi], [lo, hi], color='black', linewidth=1, linestyle='--', label='y=x (standard normal)')

    ax.set_xlabel('Theoretical quantiles (standard normal)')
    ax.set_ylabel('Empirical quantiles (deviation from real, std units)')
    token_label = 'CLS' if token_index == 0 else f'patch {token_index}'
    ax.set_title(f'{image_name} - token {token_index} ({token_label}) - QQ plot vs normal')
    ax.legend(loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()

    name_noext = Path(image_name).stem
    out_path = OUT_DIR / f'{name_noext}_token{token_index}_qq.png'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('Saved', out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='case001.png')
    p.add_argument('--token-index', type=int, action='append',
                   help='Repeatable. Defaults to [0, 1] if omitted')
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    token_indices = args.token_index or [0, 1]
    for token_index in token_indices:
        plot_qq(args.image, token_index, args.fraction, args.seed)


if __name__ == '__main__':
    main()
