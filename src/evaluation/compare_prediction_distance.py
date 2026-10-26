"""Measure how far a predicted embedding is from the real one, in units of
per-dimension standard deviation.

Uses the same reference std as serve/perturbation_test.py: the std of each of
the 768 dimensions computed across the 197 tokens (CLS + patches) of the image
itself. Expressing the prediction error in these units lets it be compared
directly against the perturbation calibration grid (fractions 0.1, 0.25, 0.5,
0.75, 1, 5, 10): e.g. an "equivalent fraction" of 3.0 means the prediction is,
on average, as far from the real embedding as a fraction=3.0 synthetic
perturbation would be.

Usage:
    python -m src.evaluation.compare_prediction_distance
    python -m src.evaluation.compare_prediction_distance --image case002.png
"""
import argparse
import json
from pathlib import Path

import numpy as np

from serve.inference import base64_to_embedding


DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')
EMB_REAL = DATA_DIR / 'embeddings_per_patch.json'
EMB_PRED = Path('dataset_breast/infered_embeddings.json')


def load_record(path: Path, key: str) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    rec = data[key]
    arr = base64_to_embedding(rec['b64'], tuple(rec['shape']), rec.get('dtype', 'float16'))
    return arr.squeeze(0)  # (n_tokens, 768)


CALIBRATION_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0, 5.0, 10.0]


def measure_image(image_name: str) -> dict:
    real = load_record(EMB_REAL, image_name)   # (197, 768)
    pred = load_record(EMB_PRED, image_name)   # (197, 768)

    std = real.std(axis=0)  # (768,), same reference used to calibrate perturbation fractions
    diff = pred - real      # (197, 768)

    # Elementwise error, in units of that dimension's std
    dist_in_std = diff / std  # (197, 768)
    abs_dist = np.abs(dist_in_std)

    # Per-token "equivalent fraction": ||diff_row|| / ||std||, comparable to the
    # fraction used to scale synthetic Gaussian noise in serve/perturbation_test.py
    std_norm = np.linalg.norm(std)
    per_token_equiv_fraction = np.linalg.norm(diff, axis=1) / std_norm  # (197,)
    mean_equiv = per_token_equiv_fraction.mean()

    return {
        'image': image_name,
        'elemwise_mean': abs_dist.mean(),
        'elemwise_median': np.median(abs_dist),
        'elemwise_p90': np.percentile(abs_dist, 90),
        'elemwise_max': abs_dist.max(),
        'cls_equiv_fraction': per_token_equiv_fraction[0],
        'patches_mean_equiv_fraction': per_token_equiv_fraction[1:].mean(),
        'mean_equiv_fraction': mean_equiv,
        'closest_calibration_fraction': min(CALIBRATION_FRACTIONS, key=lambda f: abs(f - mean_equiv)),
    }


def print_single(result: dict):
    print(f'Image: {result["image"]}')
    print()
    print('Elementwise |error| / std_d (across all 197x768 values):')
    print(f'  mean:   {result["elemwise_mean"]:.3f}')
    print(f'  median: {result["elemwise_median"]:.3f}')
    print(f'  p90:    {result["elemwise_p90"]:.3f}')
    print(f'  max:    {result["elemwise_max"]:.3f}')
    print()
    print('Per-token equivalent fraction (||pred - real|| / ||std||):')
    print(f'  CLS token:         {result["cls_equiv_fraction"]:.3f}')
    print(f'  mean over patches: {result["patches_mean_equiv_fraction"]:.3f}')
    print()
    print(f'Mean equivalent fraction across all tokens: {result["mean_equiv_fraction"]:.3f}')
    print(f'Closest value in the perturbation calibration grid: fraction={result["closest_calibration_fraction"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='case001.png')
    p.add_argument('--batch-images', type=int, default=None,
                   help='If set, ignore --image and process the first N images (case001, case002, ...)')
    args = p.parse_args()

    if args.batch_images:
        images = [f'case{i:03d}.png' for i in range(1, args.batch_images + 1)]
        rows = [measure_image(name) for name in images]

        import pandas as pd
        df = pd.DataFrame(rows).set_index('image')
        pd.set_option('display.width', 160)
        pd.set_option('display.float_format', lambda v: f'{v:.3f}')
        print(df)
        print()
        print('Average across images:')
        print(df.mean(numeric_only=True))
        return

    print_single(measure_image(args.image))


if __name__ == '__main__':
    main()
