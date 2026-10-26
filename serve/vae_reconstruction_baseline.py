"""Phase 1 — SD-VAE autoencoder fidelity baseline.

Encodes ultrasound image(s) (not tumor masks) at native resolution (padded to
a multiple of 8) with the Stable Diffusion VAE, decodes them right back with
no perturbation and no regressor involved, and measures how well the decoder
reconstructs the original image (MSE/PSNR/SSIM) — the first, simplest check
before testing noise tolerance (phase 2) or tabular regression (phase 3).

Usage:
    python -m serve.vae_reconstruction_baseline --image case001.png
    python -m serve.vae_reconstruction_baseline --batch-images 10
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from serve.vae_inference import load_vae_model, encode_image_vae, decode_embedding_vae
from serve.reconstruct_and_compile import make_captioned_strip
from src.evaluation.image_metrics import all_metrics


DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')
OUT_DIR = Path('outputs/vae_reconstruction_v0')

CAPTION_FONT_SIZE = 28
CAPTION_LABEL_HEIGHT = 40


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='case001.png', help='Filename inside DATA_DIR (must not be a *_tumor.png mask)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-images', type=int, default=None,
                    help='If set, ignore --image and process the first N ultrasound images '
                         '(sorted by filename, tumor masks excluded)')
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


def process_image(image_name, model, device, out_dir):
    name_noext = Path(image_name).stem
    img_path = DATA_DIR / image_name
    orig_img = Image.open(img_path).convert('RGB')

    latent = encode_image_vae(orig_img, model, device=device)
    recon = decode_embedding_vae(latent, model, device=device, crop_to=orig_img.size)

    orig_img.save(out_dir / f'{name_noext}_original.png')
    recon.save(out_dir / f'{name_noext}_reconstruction.png')

    strip = make_captioned_strip([orig_img, recon], ['original', 'reconstruction'],
                                  label_height=CAPTION_LABEL_HEIGHT, font_size=CAPTION_FONT_SIZE)
    strip.save(out_dir / f'{name_noext}_compiled.png')

    metrics = all_metrics(orig_img, recon)
    metrics['image'] = name_noext
    metrics['latent_shape'] = str(latent.shape)
    return metrics


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_vae_model(device=args.device)

    if args.batch_images:
        images = list_dataset_images(limit=args.batch_images)
    else:
        if args.image.endswith('_tumor.png'):
            raise ValueError('Refusing to run on a tumor mask; pass the original ultrasound image filename')
        images = [args.image]

    rows = []
    for i, image_name in enumerate(images):
        print(f'[{i + 1}/{len(images)}] {image_name}')
        rows.append(process_image(image_name, model, args.device, out_dir))

    df = pd.DataFrame(rows)[['image', 'mse', 'psnr', 'ssim', 'latent_shape']]
    summary_path = out_dir / 'summary.csv'
    df.to_csv(summary_path, index=False)
    print(df.to_string(index=False))
    print('\nMeans:', df[['mse', 'psnr', 'ssim']].mean().to_dict())
    print('Saved summary to', summary_path)


if __name__ == '__main__':
    main()
