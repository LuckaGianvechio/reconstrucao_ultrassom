"""Generate fixed-size VAE latent embeddings for dataset_breast (exclude *_tumor.png)
and save a JSON file (same schema as embeddings_per_patch.json) beside the images folder.

Used for phase 3 (tabular regression): every image is resized to
VAE_FIXED_SIZE x VAE_FIXED_SIZE before encoding, so every latent has the
same (4, VAE_FIXED_SIZE/8, VAE_FIXED_SIZE/8) shape and can be laid out as
consistent columns in the tabular dataset (see
src/preprocessing/build_tabular_dataset_vae.py). For native-resolution
fidelity checks, see serve/vae_reconstruction_baseline.py and
serve/vae_perturbation_test.py instead.

Usage: python -m serve.generate_embeddings_vae
"""
import re
import json
from pathlib import Path
from PIL import Image

from serve.vae_inference import load_vae_model, encode_image_vae
from serve.inference import embedding_to_base64


DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')
OUT_PATH = DATA_DIR / 'embeddings_vae.json'

VAE_FIXED_SIZE = 256


def main():
    model = load_vae_model(device='cpu')

    records = {}
    files = sorted([p for p in DATA_DIR.iterdir() if re.fullmatch(r'case\d+\.png', p.name)])
    for p in files:
        print('Processing', p.name)
        img = Image.open(p)
        emb = encode_image_vae(img, model, device='cpu', fixed_size=VAE_FIXED_SIZE)
        b64, shape, dtype = embedding_to_base64(emb)
        records[p.name] = {'b64': b64, 'shape': list(shape), 'dtype': dtype}

    with open(OUT_PATH, 'w') as f:
        json.dump(records, f)

    print('Wrote', OUT_PATH, 'with', len(records), 'items')


if __name__ == '__main__':
    main()
