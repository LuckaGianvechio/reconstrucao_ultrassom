"""Quick CPU smoke test for the serve wrappers.

Usage: set MAE_CHECKPOINT to a local checkpoint `.pt` and run on a machine
with PyTorch CPU installed.
"""
import os
from PIL import Image

from serve.inference import load_model, encode_image, decode_embedding


def main():
    ckpt = os.environ.get('MAE_CHECKPOINT')
    if not ckpt or not os.path.exists(ckpt):
        print('Set MAE_CHECKPOINT to a valid path to the .pt checkpoint to run smoke test.')
        return

    model = load_model(ckpt, device='cpu')
    print('Model loaded')
    # find a sample image from repo if present
    sample = None
    for root, dirs, files in os.walk('..'):
        for f in files:
            if f.lower().endswith('.png') or f.lower().endswith('.jpg'):
                sample = os.path.join(root, f)
                break
        if sample:
            break

    if not sample:
        print('No sample image found to run encode test.')
        return

    img = Image.open(sample)
    emb = encode_image(img, model, device='cpu', pooled=True)
    print('Embedding shape:', emb.shape, 'dtype:', emb.dtype)


if __name__ == '__main__':
    main()
