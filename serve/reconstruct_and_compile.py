import os
import json
import base64
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from serve.inference import load_model, base64_to_embedding, decode_embedding


DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks')
# embeddings_per_patch.json is produced inside the images folder by generate_embeddings.py
EMB_ORIG = DATA_DIR / 'embeddings_per_patch.json'
EMB_INF = Path('dataset_breast/infered_embeddings.json')
OUT_DIR = Path('outputs/reconstructions')
COMPILED_DIR = OUT_DIR / 'compiled'


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPILED_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    if not path.exists():
        print('Missing:', path)
        return {}
    return json.load(path.open())


def save_image(pil_img: Image.Image, path: Path):
    pil_img.save(path)


def decode_emb_record(rec, model):
    # rec: {b64, shape, dtype, ids_restore?}
    b64 = rec.get('b64') or rec.get('embedding_b64')
    shape = tuple(rec.get('shape'))
    dtype = rec.get('dtype', 'float16')
    arr = base64.b64decode(b64)
    emb = np.frombuffer(arr, dtype=np.dtype(dtype)).reshape(shape)
    # convert to float32 for model
    emb = emb.astype(np.float32)
    ids_restore = rec.get('ids_restore')
    if ids_restore is not None:
        ids_restore = np.array(ids_restore, dtype=np.int64)
    img = decode_embedding(emb, model, device='cpu', ids_restore=ids_restore)
    return img


def make_captioned_strip(images, captions, label_height=20, padding=8):
    # images: list of PIL Images, captions: list of str
    widths, heights = zip(*(im.size for im in images))
    total_w = sum(widths) + padding * (len(images) - 1)
    max_h = max(heights)
    # font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    caption_h = label_height
    out_h = max_h + caption_h + 6
    out = Image.new('RGB', (total_w, out_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(out)

    x = 0
    for im, cap in zip(images, captions):
        out.paste(im, (x, 0))
        # draw caption centered under image
        w, h = im.size
        # Pillow version differences: use font.getsize if available
        try:
            text_w, text_h = draw.textsize(cap, font=font)
        except Exception:
            try:
                text_w, text_h = font.getsize(cap)
            except Exception:
                text_w, text_h = (len(cap) * 6, 10)
        tx = x + (w - text_w) // 2
        ty = max_h + 2
        draw.text((tx, ty), cap, fill=(0, 0, 0), font=font)
        x += w + padding

    return out


def main():
    ensure_dirs()
    model = load_model(os.environ.get('MAE_CHECKPOINT', './USF-MAE_full_pretrain_43dataset_100epochs.pt'), device='cpu')

    orig_json = load_json(EMB_ORIG)
    inf_json = load_json(EMB_INF)

    keys = sorted(inf_json.keys())
    print('Found', len(keys), 'inferred embeddings')

    for k in keys:
        print('Processing', k)
        base_name = Path(k).name
        name_noext = Path(base_name).stem

        orig_img_path = DATA_DIR / base_name
        if not orig_img_path.exists():
            print('  Original image missing, skipping:', orig_img_path)
            continue

        # save original image
        orig_img = Image.open(orig_img_path).convert('RGB')
        out_orig = OUT_DIR / f'{name_noext}_original.png'
        save_image(orig_img, out_orig)

        # decode original embedding if present
        if base_name in orig_json:
            try:
                img_decoded = decode_emb_record(orig_json[base_name], model)
            except Exception as e:
                print('  Failed decoding original embedding:', e)
                img_decoded = Image.new('RGB', orig_img.size, (200, 200, 200))
        else:
            img_decoded = Image.new('RGB', orig_img.size, (200, 200, 200))

        out_dec = OUT_DIR / f'{name_noext}_decoder.png'
        save_image(img_decoded.convert('RGB'), out_dec)

        # decode inferred embedding
        try:
            img_infer = decode_emb_record(inf_json[base_name], model)
        except Exception as e:
            print('  Failed decoding inferred embedding:', e)
            img_infer = Image.new('RGB', orig_img.size, (220, 220, 220))

        out_inf = OUT_DIR / f'{name_noext}_infered.png'
        save_image(img_infer.convert('RGB'), out_inf)

        # resize decoded/infer to match original if sizes differ
        target_size = orig_img.size
        images = [orig_img, img_decoded, img_infer]
        images_resized = [im.resize(target_size) for im in images]

        captions = ['original', 'decoder', 'infered']
        compiled = make_captioned_strip(images_resized, captions)
        compiled_out = COMPILED_DIR / f'{name_noext}_compiled.png'
        compiled.save(compiled_out)

    print('Done. Outputs in', OUT_DIR)


if __name__ == '__main__':
    main()
