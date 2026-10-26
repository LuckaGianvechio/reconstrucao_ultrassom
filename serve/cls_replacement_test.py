"""Replace the CLS token entirely (not just add noise) and decode, to see how
much the decoder's spatial reconstruction actually depends on it.

Two replacements are tested:
- zero: CLS token set to the all-zeros vector.
- random: CLS token set to an independent random draw per dimension, from
  N(mean_d, std_d) estimated from that image's own 196 patch tokens.

Usage:
    python -m serve.cls_replacement_test --image case008.png
"""
import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from serve.inference import load_model, encode_image, decode_embedding
from serve.reconstruct_and_compile import make_captioned_strip


DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')
OUT_DIR = Path('outputs/perturbation_v2')

CAPTION_FONT_SIZE = 28
CAPTION_LABEL_HEIGHT = 40


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='case008.png')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = os.environ.get('MAE_CHECKPOINT', './USF-MAE_full_pretrain_43dataset_100epochs.pt')
    model = load_model(ckpt, device=args.device)
    model.eval()

    name_noext = Path(args.image).stem
    orig_img = Image.open(DATA_DIR / args.image).convert('RGB')

    emb, ids_restore = encode_image(orig_img, model, device=args.device, pooled=False)  # (1, 197, 768)
    recon_original = decode_embedding(emb, model, device=args.device, ids_restore=ids_restore)

    patch_tokens = emb.squeeze(0)[1:]  # (196, 768), excludes CLS
    mean_d = patch_tokens.mean(axis=0)
    std_d = patch_tokens.std(axis=0)

    rng = np.random.default_rng(args.seed)
    random_cls = rng.normal(mean_d, std_d)  # (768,), one independent draw per dimension

    emb_zero = emb.copy()
    emb_zero[0, 0, :] = 0.0

    emb_random = emb.copy()
    emb_random[0, 0, :] = random_cls

    recon_zero = decode_embedding(emb_zero, model, device=args.device, ids_restore=ids_restore)
    recon_random = decode_embedding(emb_random, model, device=args.device, ids_restore=ids_restore)

    target_size = orig_img.size
    row_images = [
        orig_img.resize(target_size),
        recon_original.resize(target_size),
        recon_zero.resize(target_size),
        recon_random.resize(target_size),
    ]
    row_captions = [name_noext, 'reconstruction', 'CLS = zero vector', 'CLS = random vector']

    grid = make_captioned_strip(row_images, row_captions,
                                 label_height=CAPTION_LABEL_HEIGHT, font_size=CAPTION_FONT_SIZE)
    out_path = OUT_DIR / f'{name_noext}_cls_replacement.png'
    grid.save(out_path)
    print('Saved', out_path)


if __name__ == '__main__':
    main()
