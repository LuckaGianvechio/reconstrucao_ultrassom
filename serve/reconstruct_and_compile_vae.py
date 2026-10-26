"""VAE analogue of serve/reconstruct_and_compile.py.

Decodes, for each case with a predicted (regressor) VAE latent, the original
fixed-size latent (embeddings_vae.json, produced by
serve/generate_embeddings_vae.py) and the CatBoost-inferred latent
(infered_embeddings_vae.json, produced by src/models/train_catboost_vae.py),
and compiles [original, decoder, infered] comparison strips.

Usage:
    python -m serve.reconstruct_and_compile_vae --limit N
"""
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from serve.vae_inference import load_vae_model, decode_embedding_vae
from serve.reconstruct_and_compile import make_captioned_strip
from src.preprocessing.build_tabular_dataset_vae import decode_embedding_record


DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')
EMB_ORIG = DATA_DIR / 'embeddings_vae.json'
EMB_INF = Path('dataset_breast/infered_embeddings_vae.json')
OUT_DIR = Path('outputs/vae_reconstructions')
COMPILED_DIR = OUT_DIR / 'compiled'

VAE_FIXED_SIZE = 256  # must match serve/generate_embeddings_vae.py


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPILED_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    if not path.exists():
        print('Missing:', path)
        return {}
    return json.load(path.open())


def to_latent_grid(arr: np.ndarray) -> np.ndarray:
    """Normalize either (1, C, H, W) [original] or (1, n_tokens, C) [regressor
    output, row-major flattened, square grid] into (1, C, H, W)."""
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        _, n_tokens, c = arr.shape
        side = int(round(n_tokens ** 0.5))
        assert side * side == n_tokens, f'Expected a square token grid, got n_tokens={n_tokens}'
        grid = arr.reshape(side, side, c).transpose(2, 0, 1)  # (C, H, W)
        return grid[np.newaxis, ...]
    raise ValueError(f'Unexpected latent array shape {arr.shape}')


def decode_emb_record(rec, model):
    arr = decode_embedding_record(rec)  # (1, C, H, W) or (1, n_tokens, C)
    latent = to_latent_grid(arr)
    return decode_embedding_vae(latent, model, device='cpu')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit', type=int, default=None, help='Only process the first N images (sorted by filename)')
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    model = load_vae_model(device='cpu')

    orig_json = load_json(EMB_ORIG)
    inf_json = load_json(EMB_INF)

    keys = sorted(inf_json.keys())
    if args.limit:
        keys = keys[:args.limit]
    print('Found', len(keys), 'inferred embeddings')

    for k in keys:
        print('Processing', k)
        name_noext = Path(k).stem

        orig_img_path = DATA_DIR / k
        if not orig_img_path.exists():
            print('  Original image missing, skipping:', orig_img_path)
            continue

        orig_img = Image.open(orig_img_path).convert('RGB')
        target_size = (VAE_FIXED_SIZE, VAE_FIXED_SIZE)

        if k in orig_json:
            try:
                img_decoded = decode_emb_record(orig_json[k], model)
            except Exception as e:
                print('  Failed decoding original embedding:', e)
                img_decoded = Image.new('RGB', target_size, (200, 200, 200))
        else:
            img_decoded = Image.new('RGB', target_size, (200, 200, 200))

        try:
            img_infer = decode_emb_record(inf_json[k], model)
        except Exception as e:
            print('  Failed decoding inferred embedding:', e)
            img_infer = Image.new('RGB', target_size, (220, 220, 220))

        images = [orig_img.resize(target_size), img_decoded.resize(target_size), img_infer.resize(target_size)]
        captions = ['original', 'decoder', 'infered']
        compiled = make_captioned_strip(images, captions)
        compiled.save(COMPILED_DIR / f'{name_noext}_compiled.png')

    print('Done. Outputs in', OUT_DIR)


if __name__ == '__main__':
    main()
