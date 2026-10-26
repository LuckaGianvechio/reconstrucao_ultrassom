"""What do the images actually look like if you only do UMAP -> inverse_transform?

src/models/train_tabular_to_umap_inverse.py answered this numerically for the
MAE pooled embedding (R2 ~ 0.08 in the original 768-dim space, i.e. the
projection alone destroys most of the signal), but that embedding is a mean over
196 patch tokens and therefore cannot be decoded back into an image at all.

The SD-VAE latent CAN be decoded, so it is the right place to *see* the answer.
Each image here is a fixed-size (1, 4, 32, 32) latent = 4096 dims, encoded by
serve/generate_embeddings_vae.py. The pipeline is:

    true latent (4096d) -> StandardScaler -> UMAP(n) -> inverse_transform
                        -> unscale -> reshape (1,4,32,32) -> VAE decoder -> image

NO regression is involved anywhere. This isolates the UMAP round-trip loss by
itself, which is the hard ceiling for anything routed through the manifold.

Each output strip shows three panels, which is the whole point of the script:
  1. `original`        - the source ultrasound image.
  2. `vae_roundtrip`   - decode(true latent). The VAE's OWN reconstruction, i.e.
                         the real ceiling: no manifold can beat this, because
                         every route goes through the same decoder.
  3. `umap_roundtrip`  - decode(inverse_transform(project(true latent))).

Comparing panel 3 against panel 2 (not against panel 1) is what isolates the
UMAP loss from the VAE's own reconstruction loss.

The UMAP is fit on the TRAINING split only and the images rendered are from the
TEST split, so what is shown is genuine out-of-sample behaviour - a UMAP fit on
all 256 images would flatter itself by having seen the image it reconstructs.

Usage:
    python -m src.models.umap_roundtrip_reconstruction --n-components 5 --n-images 6
    python -m src.models.umap_roundtrip_reconstruction --n-components 2 10 50 --n-images 4
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import umap

from serve.vae_inference import load_vae_model, decode_embedding_vae
from serve.reconstruct_and_compile import make_captioned_strip
from src.preprocessing.build_tabular_dataset_vae import decode_embedding_record
from src.evaluation.image_metrics import all_metrics


DATA_DIR = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/'
                'BrEaST-Lesions_USG-images_and_masks')
VAE_JSON = DATA_DIR / 'embeddings_vae.json'
OUT_DIR = Path('outputs/umap_roundtrip_reconstruction')

# Same split convention as src/models/train_tabular_to_umap.py, so the train/test
# membership of a given image matches across the two experiments.
TEST_SIZE = 0.2
SPLIT_SEED = 42

# The latents in embeddings_vae.json were encoded with
# serve/vae_inference.py::preprocess_image_vae_fixed(size=256), so every decoded
# image comes back at this resolution.
FIXED_SIZE = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-components', type=int, nargs='+', default=[2, 5, 10, 50],
                   help='UMAP dimensionalities to round-trip through.')
    p.add_argument('--n-images', type=int, default=6,
                   help='How many TEST images to render strips for.')
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = json.load(open(VAE_JSON))
    filenames = sorted(records)
    latents = np.stack([decode_embedding_record(records[f])[0] for f in filenames])  # (N,4,32,32)
    latent_shape = latents.shape[1:]
    flat = latents.reshape(len(filenames), -1)  # (N, 4096)
    print(f'{len(filenames)} latents of shape {latent_shape} -> {flat.shape[1]} dims')

    train_idx, test_idx = train_test_split(
        np.arange(len(filenames)), test_size=TEST_SIZE, random_state=SPLIT_SEED,
    )
    print(f'train {len(train_idx)} | test {len(test_idx)}')

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(flat[train_idx])
    test_scaled = scaler.transform(flat[test_idx])

    print('Loading SD-VAE decoder ...')
    vae = load_vae_model(device=args.device)

    show_idx = test_idx[:args.n_images]
    # These latents come from preprocess_image_vae_fixed(size=FIXED_SIZE), so the
    # decoder returns FIXED_SIZE x FIXED_SIZE regardless of the source resolution.
    # The original is resized to match, otherwise it is not comparable at all
    # (and `crop_to` would not help: it crops, it cannot upscale).
    originals = {i: Image.open(DATA_DIR / filenames[i]).convert('RGB')
                        .resize((FIXED_SIZE, FIXED_SIZE)) for i in show_idx}

    # Panel 2: the VAE's own reconstruction of the true latent - the ceiling that
    # no manifold route can beat, since every route ends in this same decoder.
    vae_recon = {i: decode_embedding_vae(latents[i][None], vae, device=args.device)
                 for i in show_idx}

    rows = []
    for n_components in args.n_components:
        print(f'\n=== UMAP {n_components} components ===')
        reducer = umap.UMAP(n_components=n_components,
                            n_neighbors=min(15, len(train_idx) - 1),
                            random_state=42)
        reducer.fit(train_scaled)
        z_test = reducer.transform(test_scaled)
        back = scaler.inverse_transform(reducer.inverse_transform(z_test))  # (n_test, 4096)

        # Latent-space fidelity over the WHOLE test split (not just the rendered ones).
        ss_res = ((flat[test_idx] - back) ** 2).sum()
        ss_tot = ((flat[test_idx] - flat[train_idx].mean(axis=0)) ** 2).sum()
        latent_r2 = 1.0 - ss_res / ss_tot
        print(f'latent-space R2 over {len(test_idx)} test images: {latent_r2:+.4f}')

        pos_in_test = {img_i: k for k, img_i in enumerate(test_idx)}
        for i in show_idx:
            recon_latent = back[pos_in_test[i]].reshape(latent_shape)
            umap_img = decode_embedding_vae(recon_latent[None], vae, device=args.device)

            name = Path(filenames[i]).stem
            strip = make_captioned_strip(
                [originals[i], vae_recon[i], umap_img],
                ['original', 'VAE roundtrip (teto)', f'UMAP-{n_components} roundtrip'],
                label_height=40, font_size=28,
            )
            strip.save(OUT_DIR / f'{name}_umap{n_components}.png')

            # Against the VAE reconstruction, so the number is UMAP loss alone;
            # against the original it would also carry the VAE's own loss.
            vs_vae = all_metrics(vae_recon[i], umap_img)
            vs_orig = all_metrics(originals[i], umap_img)
            rows.append({'image': name, 'n_components': n_components,
                         'latent_r2_testsplit': latent_r2,
                         **{f'{k}_vs_vae': v for k, v in vs_vae.items()},
                         **{f'{k}_vs_original': v for k, v in vs_orig.items()}})
            print(f'  {name}: SSIM vs VAE-recon = {vs_vae["ssim"]:.4f}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'metrics.csv', index=False)

    print('\n=== SSIM of the UMAP round-trip against the VAE reconstruction ===')
    print(df.groupby('n_components')
            .agg(latent_r2=('latent_r2_testsplit', 'first'),
                 ssim_vs_vae=('ssim_vs_vae', 'mean'),
                 psnr_vs_vae=('psnr_vs_vae', 'mean'),
                 ssim_vs_original=('ssim_vs_original', 'mean'))
            .to_string())
    print(f'\nStrips and metrics.csv saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
