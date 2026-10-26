"""Generate per-patch embeddings for dataset_breast (exclude *_tumor.png)
and save a JSON file with base64-encoded embeddings beside the images folder.

Usage: set `MAE_CHECKPOINT` env var or put checkpoint at repo root.
"""
import os
import re
import json
from pathlib import Path
from PIL import Image

from serve.inference import load_model, encode_image, embedding_to_base64


DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks')
OUT_PATH = DATA_DIR / 'embeddings_per_patch.json'


def main():
    ckpt = os.environ.get('MAE_CHECKPOINT', './USF-MAE_full_pretrain_43dataset_100epochs.pt')
    model = load_model(ckpt, device='cpu')
    model.eval()

    records = {}
    files = sorted([p for p in DATA_DIR.iterdir() if re.fullmatch(r'case\d+\.png', p.name)])
    for p in files:
        print('Processing', p.name)
        img = Image.open(p)
        res = encode_image(img, model, device='cpu', pooled=False)
        # res may be (emb, ids_restore) or emb
        if isinstance(res, tuple):
            emb, ids_restore = res
        else:
            emb = res
            ids_restore = None
        b64, shape, dtype = embedding_to_base64(emb)
        rec = {'b64': b64, 'shape': list(shape), 'dtype': dtype}
        if ids_restore is not None:
            rec['ids_restore'] = ids_restore.tolist()
        records[p.name] = rec

    with open(OUT_PATH, 'w') as f:
        json.dump(records, f)

    print('Wrote', OUT_PATH, 'with', len(records), 'items')


if __name__ == '__main__':
    main()
