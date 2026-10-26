"""Phase 2 — SD-VAE sensitivity test: perturb the encoder latent and decode.

Encodes ultrasound image(s) (not tumor masks) to their native-resolution VAE
latent (4, H/8, W/8), decodes the unperturbed latent to check reconstruction
fidelity (same as serve/vae_reconstruction_baseline.py), then adds
independent Gaussian noise to every spatial position/channel and decodes
each perturbed version.

The noise for channel c is drawn per-position as N(0, (fraction * std_c)^2),
where std_c is the standard deviation of channel c across all spatial
positions of that same image's latent. `fraction` is swept over a grid to
calibrate how much perturbation the decoder tolerates before the
reconstruction degrades — the same calibration idea as
serve/perturbation_test.py (MAE), applied to the VAE's purely spatial latent
(there is no CLS-like token here, so there is no `--target cls` mode).

Single-image usage (full-resolution calibration grid):
    python -m serve.vae_perturbation_test
    python -m serve.vae_perturbation_test --image case002.png --fractions 0.02,0.1,0.3 --seed 7

Batch usage (compiled comparison grid across many images):
    python -m serve.vae_perturbation_test --batch-images 10
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from serve.vae_inference import load_vae_model, encode_image_vae, decode_embedding_vae
from serve.reconstruct_and_compile import make_captioned_strip
from src.evaluation.image_metrics import all_metrics


DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')
OUT_DIR = Path('outputs/vae_perturbation_v0')

DEFAULT_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0, 5.0, 10.0]

CAPTION_FONT_SIZE = 28
CAPTION_LABEL_HEIGHT = 40


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='case001.png', help='Filename inside DATA_DIR (must not be a *_tumor.png mask)')
    p.add_argument('--fractions', default=','.join(str(f) for f in DEFAULT_FRACTIONS),
                   help='Comma-separated list of noise fractions (relative to per-channel std)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-images', type=int, default=None,
                   help='If set, ignore --image and instead process the first N ultrasound images '
                        '(sorted by filename, tumor masks excluded), compiling one comparison grid image')
    p.add_argument('--cell-size', type=int, default=None,
                   help='Per-cell thumbnail size (pixels) used only in batch mode. Defaults to each image\'s '
                        'original resolution')
    p.add_argument('--out-dir', default=None)
    return p.parse_args()


def list_dataset_images(limit=None):
    files = sorted(p.name for p in DATA_DIR.iterdir()
                    if p.suffix.lower() == '.png'
                    and not p.name.endswith('_tumor.png')
                    and '_other' not in p.name)
    if limit is not None:
        files = files[:limit]
    return files


def build_row(image_name, model, fractions, seed, device, cell_size=None, save_individual=False, out_dir=None):
    """Encode/decode one image and return (row_images, row_captions, metric_rows).

    Captions: <image_stem>, reconstruction, fraction=<f> for each f in fractions.
    metric_rows: one dict per fraction (plus fraction=0.0 for the unperturbed
    reconstruction) with image/fraction/mse/psnr/ssim, for later analysis.
    """
    out_dir = out_dir or OUT_DIR
    name_noext = Path(image_name).stem
    img_path = DATA_DIR / image_name
    orig_img = Image.open(img_path).convert('RGB')

    latent = encode_image_vae(orig_img, model, device=device)  # (1, 4, H', W')
    recon_original = decode_embedding_vae(latent, model, device=device, crop_to=orig_img.size)

    target_size = (cell_size, cell_size) if cell_size else orig_img.size

    if save_individual:
        recon_original.save(out_dir / f'{name_noext}_reconstruction.png')

    channels = latent[0]  # (4, H', W')
    std = channels.std(axis=(1, 2), keepdims=True)  # (4, 1, 1), per-channel std across spatial positions

    rng = np.random.default_rng(seed)

    row_images = [orig_img.resize(target_size), recon_original.resize(target_size)]
    row_captions = [name_noext, 'reconstruction']
    metric_rows = [{'image': name_noext, 'fraction': 0.0, **all_metrics(orig_img, recon_original)}]

    for fraction in fractions:
        noise = rng.normal(0.0, 1.0, size=channels.shape) * (fraction * std)  # (4, H', W')
        latent_perturbed = latent + noise[np.newaxis, :, :, :]

        recon = decode_embedding_vae(latent_perturbed, model, device=device, crop_to=orig_img.size)
        if save_individual:
            recon.save(out_dir / f'{name_noext}_perturbed_f{fraction}.png')

        row_images.append(recon.resize(target_size))
        row_captions.append(f'fraction={fraction}')
        metric_rows.append({'image': name_noext, 'fraction': fraction, **all_metrics(orig_img, recon)})

    return row_images, row_captions, metric_rows


def main():
    args = parse_args()
    fractions = [float(f) for f in args.fractions.split(',') if f.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_vae_model(device=args.device)

    all_metric_rows = []

    if args.batch_images:
        images = list_dataset_images(limit=args.batch_images)
        print(f'Processing {len(images)} images in batch mode')
        for i, image_name in enumerate(images):
            name_noext = Path(image_name).stem
            print(f'  [{i + 1}/{len(images)}] {image_name}')
            row_images, row_captions, metric_rows = build_row(
                image_name, model, fractions, args.seed, args.device, cell_size=args.cell_size,
                out_dir=out_dir,
            )
            all_metric_rows.extend(metric_rows)
            compiled = make_captioned_strip(row_images, row_captions,
                                             label_height=CAPTION_LABEL_HEIGHT, font_size=CAPTION_FONT_SIZE)
            out_path = out_dir / f'{name_noext}_grid.png'
            compiled.save(out_path)
        print('Saved one compiled comparison image per case to', out_dir)
    else:
        if args.image.endswith('_tumor.png'):
            raise ValueError('Refusing to run on a tumor mask; pass the original ultrasound image filename')

        name_noext = Path(args.image).stem
        row_images, row_captions, metric_rows = build_row(
            args.image, model, fractions, args.seed, args.device, cell_size=None, save_individual=True,
            out_dir=out_dir,
        )
        all_metric_rows.extend(metric_rows)

        baseline = make_captioned_strip(row_images[:2], row_captions[:2],
                                         label_height=CAPTION_LABEL_HEIGHT, font_size=CAPTION_FONT_SIZE)
        baseline.save(out_dir / f'{name_noext}_original_vs_reconstruction.png')
        print('Saved baseline reconstruction check to', out_dir / f'{name_noext}_original_vs_reconstruction.png')

        grid = make_captioned_strip(row_images, row_captions,
                                     label_height=CAPTION_LABEL_HEIGHT, font_size=CAPTION_FONT_SIZE)
        grid_path = out_dir / f'{name_noext}_grid.png'
        grid.save(grid_path)
        print('Saved calibration grid to', grid_path)

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_path = out_dir / 'metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print('\nMean metrics per fraction:')
    print(metrics_df.groupby('fraction')[['mse', 'psnr', 'ssim']].mean().to_string())
    print('Saved metrics to', metrics_path)


if __name__ == '__main__':
    main()
