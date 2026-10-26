"""Sensitivity test: perturb the per-patch embedding of one or more images and decode.

Encodes ultrasound image(s) (not tumor masks) to their 197 per-patch
embeddings (CLS + 14x14 patches, 768-dim each), decodes the unperturbed
embedding to check reconstruction fidelity, then adds independent Gaussian
noise to every token's embedding and decodes each perturbed version.

The noise for dimension d is drawn per-token as N(0, (fraction * std_d)^2),
where std_d is the standard deviation of dimension d across the 197 tokens of
that same image. `fraction` is swept over a grid to calibrate how much
perturbation the decoder tolerates before the reconstruction degrades.

Single-image usage (full-resolution calibration grid):
    python -m serve.perturbation_test
    python -m serve.perturbation_test --image case002.png --fractions 0.02,0.1,0.3 --seed 7

Batch usage (compiled comparison grid across many images):
    python -m serve.perturbation_test --batch-images 50
"""
import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from serve.inference import load_model, encode_image, decode_embedding
from serve.reconstruct_and_compile import make_captioned_strip


DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')
OUT_DIR = Path('outputs/perturbation_v1')

DEFAULT_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0, 5.0, 10.0]
DEFAULT_OUT_DIRS = {'all': Path('outputs/perturbation_v1'), 'cls': Path('outputs/perturbation_v2')}

CAPTION_FONT_SIZE = 28
CAPTION_LABEL_HEIGHT = 40


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='case001.png', help='Filename inside DATA_DIR (must not be a *_tumor.png mask)')
    p.add_argument('--fractions', default=','.join(str(f) for f in DEFAULT_FRACTIONS),
                   help='Comma-separated list of noise fractions (relative to per-dimension std)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-images', type=int, default=None,
                   help='If set, ignore --image and instead process the first N ultrasound images '
                        '(sorted by filename, tumor masks excluded), compiling one comparison grid image')
    p.add_argument('--cell-size', type=int, default=None,
                   help='Per-cell thumbnail size (pixels) used only in batch mode. Defaults to each image\'s '
                        'original resolution')
    p.add_argument('--target', choices=['all', 'cls'], default='all',
                   help='"all" perturbs every one of the 197 tokens (default). "cls" perturbs only the '
                        'CLS token (index 0), leaving all 196 patch tokens untouched')
    p.add_argument('--out-dir', default=None, help='Overrides the default output directory for --target')
    return p.parse_args()


def list_dataset_images(limit=None):
    files = sorted(p.name for p in DATA_DIR.iterdir()
                    if p.suffix.lower() == '.png'
                    and not p.name.endswith('_tumor.png')
                    and '_other' not in p.name)
    if limit is not None:
        files = files[:limit]
    return files


def build_row(image_name, model, fractions, seed, device, cell_size=None, save_individual=False,
              target='all', out_dir=None):
    """Encode/decode one image and return (row_images, row_captions) for a comparison strip.

    Captions: <image_stem>, reconstruction, fraction=<f> for each f in fractions.

    target='all' perturbs every one of the 197 tokens; target='cls' perturbs only
    the CLS token (index 0), leaving all patch tokens equal to the real embedding.
    """
    out_dir = out_dir or OUT_DIR
    name_noext = Path(image_name).stem
    img_path = DATA_DIR / image_name
    orig_img = Image.open(img_path).convert('RGB')

    emb, ids_restore = encode_image(orig_img, model, device=device, pooled=False)  # (1, 197, 768)
    recon_original = decode_embedding(emb, model, device=device, ids_restore=ids_restore)

    target_size = (cell_size, cell_size) if cell_size else orig_img.size

    if save_individual:
        recon_original.save(out_dir / f'{name_noext}_reconstruction.png')

    tokens = emb.squeeze(0)  # (197, 768)
    std = tokens.std(axis=0)  # (768,), per-dimension std across the 197 tokens of this image

    rng = np.random.default_rng(seed)

    row_images = [orig_img.resize(target_size), recon_original.resize(target_size)]
    row_captions = [name_noext, 'reconstruction']

    token_mask = np.ones((tokens.shape[0], 1), dtype=np.float32)
    if target == 'cls':
        token_mask[:] = 0.0
        token_mask[0] = 1.0  # only the CLS token (index 0) receives noise

    for fraction in fractions:
        noise = rng.normal(0.0, 1.0, size=tokens.shape) * (fraction * std) * token_mask  # (197, 768)
        emb_perturbed = emb + noise[np.newaxis, :, :]

        recon = decode_embedding(emb_perturbed, model, device=device, ids_restore=ids_restore)
        if save_individual:
            recon.save(out_dir / f'{name_noext}_perturbed_f{fraction}.png')

        row_images.append(recon.resize(target_size))
        row_captions.append(f'fraction={fraction}')

    return row_images, row_captions


def main():
    args = parse_args()
    fractions = [float(f) for f in args.fractions.split(',') if f.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIRS[args.target]
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = os.environ.get('MAE_CHECKPOINT', './USF-MAE_full_pretrain_43dataset_100epochs.pt')
    model = load_model(ckpt, device=args.device)
    model.eval()

    if args.batch_images:
        images = list_dataset_images(limit=args.batch_images)
        print(f'Processing {len(images)} images in batch mode (target={args.target})')
        for i, image_name in enumerate(images):
            name_noext = Path(image_name).stem
            print(f'  [{i + 1}/{len(images)}] {image_name}')
            row_images, row_captions = build_row(
                image_name, model, fractions, args.seed, args.device, cell_size=args.cell_size,
                target=args.target, out_dir=out_dir,
            )
            compiled = make_captioned_strip(row_images, row_captions,
                                             label_height=CAPTION_LABEL_HEIGHT, font_size=CAPTION_FONT_SIZE)
            out_path = out_dir / f'{name_noext}_grid.png'
            compiled.save(out_path)
        print('Saved one compiled comparison image per case to', out_dir)
        return

    if args.image.endswith('_tumor.png'):
        raise ValueError('Refusing to run on a tumor mask; pass the original ultrasound image filename')

    name_noext = Path(args.image).stem
    row_images, row_captions = build_row(
        args.image, model, fractions, args.seed, args.device, cell_size=None, save_individual=True,
        target=args.target, out_dir=out_dir,
    )

    baseline = make_captioned_strip(row_images[:2], row_captions[:2],
                                     label_height=CAPTION_LABEL_HEIGHT, font_size=CAPTION_FONT_SIZE)
    baseline.save(out_dir / f'{name_noext}_original_vs_reconstruction.png')
    print('Saved baseline reconstruction check to', out_dir / f'{name_noext}_original_vs_reconstruction.png')

    grid = make_captioned_strip(row_images, row_captions,
                                 label_height=CAPTION_LABEL_HEIGHT, font_size=CAPTION_FONT_SIZE)
    grid_path = out_dir / f'{name_noext}_grid.png'
    grid.save(grid_path)
    print('Saved calibration grid to', grid_path)


if __name__ == '__main__':
    main()
