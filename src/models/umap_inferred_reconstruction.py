"""Render the INFERRED reconstructions: tabular clinical data -> image.

src/models/umap_roundtrip_reconstruction.py showed what the UMAP round-trip does
to an image with no regression involved (answer: it destroys it). This script
adds the regression step back, completing the chain the project originally set
out to build:

    tabular clinical features -> CatBoost -> UMAP coords -> inverse_transform
                              -> latent (1,4,32,32) -> SD-VAE decoder -> image

Each strip has four panels, left to right, which is the point of the script -
each one adds exactly one source of loss on top of the previous:

  1. `original`          - the source ultrasound image.
  2. `VAE roundtrip`     - decode(true latent). Loss: the VAE encoder/decoder only.
                           This is the hard ceiling; every route ends in this decoder.
  3. `UMAP roundtrip`    - decode(inverse(project(true latent))). Adds: the UMAP
                           projection loss, with NO regression involved.
  4. `UMAP inferido`     - decode(inverse(CatBoost(tabular))). Adds: the regression
                           error on top of panel 3.

Reading panels 3 -> 4 isolates how much the regressor contributes, separately from
how much the projection already destroyed on its own. Only 2 and 5 components are
supported in practice: umap-learn's inverse_transform runs out of memory at 10
dims and overflows Qhull at 50 (see docs/tabular-to-umap-log.md).

UMAP, StandardScaler and CatBoost are all fit on the TRAINING split only; the
images rendered come from the TEST split.

Usage:
    python -m src.models.umap_inferred_reconstruction --n-components 2 5
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
from src.preprocessing.build_tabular_dataset import build_clinical_features
from src.preprocessing.build_tabular_dataset_vae import decode_embedding_record
from src.preprocessing.clinical_targets import load_raw_clinical
from src.evaluation.image_metrics import all_metrics
from src.models.train_tabular_to_umap import make_regressor
from src.models.umap_roundtrip_reconstruction import (
    DATA_DIR, VAE_JSON, FIXED_SIZE, TEST_SIZE, SPLIT_SEED,
)


OUT_DIR = Path('outputs/umap_inferred_reconstruction')

# The four cases already rendered by umap_roundtrip_reconstruction.py, so the
# user compares the same images across both experiments.
DEFAULT_CASES = ['case229.png', 'case007.png', 'case080.png', 'case207.png']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-components', type=int, nargs='+', default=[2, 5],
                   help='Only 2 and 5 are viable; inverse_transform fails above that.')
    p.add_argument('--cases', nargs='+', default=DEFAULT_CASES)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = json.load(open(VAE_JSON))
    raw_df = load_raw_clinical()
    clinical = build_clinical_features(raw_df)
    clinical.index = raw_df['Image_filename']

    filenames = sorted(f for f in records if f in clinical.index)
    latents = np.stack([decode_embedding_record(records[f])[0] for f in filenames])
    latent_shape = latents.shape[1:]
    flat = latents.reshape(len(filenames), -1)
    X = clinical.loc[filenames].to_numpy(dtype=float)
    print(f'{len(filenames)} images | tabular {X.shape[1]} feats | latent {flat.shape[1]} dims')

    train_idx, test_idx = train_test_split(
        np.arange(len(filenames)), test_size=TEST_SIZE, random_state=SPLIT_SEED,
    )
    pos_in_test = {img_i: k for k, img_i in enumerate(test_idx)}

    show_idx = []
    for case in args.cases:
        i = filenames.index(case)
        if i not in pos_in_test:
            print(f'  skipping {case}: it is in the TRAINING split, not test')
            continue
        show_idx.append(i)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(flat[train_idx])
    test_scaled = scaler.transform(flat[test_idx])

    print('Loading SD-VAE decoder ...')
    vae = load_vae_model(device=args.device)

    originals = {i: Image.open(DATA_DIR / filenames[i]).convert('RGB')
                       .resize((FIXED_SIZE, FIXED_SIZE)) for i in show_idx}
    vae_recon = {i: decode_embedding_vae(latents[i][None], vae, device=args.device)
                 for i in show_idx}

    rows = []
    for n_components in args.n_components:
        print(f'\n=== UMAP {n_components} components ===')
        reducer = umap.UMAP(n_components=n_components,
                            n_neighbors=min(15, len(train_idx) - 1),
                            random_state=42)
        z_train = reducer.fit_transform(train_scaled)
        z_test = reducer.transform(test_scaled)

        # Panel 3: no regression, the projection's own loss.
        back_true = scaler.inverse_transform(reducer.inverse_transform(z_test))

        # Panel 4: the actual inference chain, tabular -> coords -> latent.
        model = make_regressor()
        model.fit(X[train_idx], z_train)
        z_pred = np.asarray(model.predict(X[test_idx]))
        back_pred = scaler.inverse_transform(reducer.inverse_transform(z_pred))

        for i in show_idx:
            k = pos_in_test[i]
            img_roundtrip = decode_embedding_vae(
                back_true[k].reshape(latent_shape)[None], vae, device=args.device)
            img_inferred = decode_embedding_vae(
                back_pred[k].reshape(latent_shape)[None], vae, device=args.device)

            name = Path(filenames[i]).stem
            strip = make_captioned_strip(
                [originals[i], vae_recon[i], img_roundtrip, img_inferred],
                ['original', 'VAE roundtrip (teto)',
                 f'UMAP-{n_components} roundtrip', f'UMAP-{n_components} inferido (tabular)'],
                label_height=40, font_size=24,
            )
            strip.save(OUT_DIR / f'{name}_umap{n_components}_inferred.png')

            m_inf = all_metrics(vae_recon[i], img_inferred)
            m_rt = all_metrics(vae_recon[i], img_roundtrip)
            rows.append({'image': name, 'n_components': n_components,
                         'ssim_roundtrip_vs_vae': m_rt['ssim'],
                         'ssim_inferred_vs_vae': m_inf['ssim'],
                         'psnr_inferred_vs_vae': m_inf['psnr']})
            print(f'  {name}: SSIM roundtrip={m_rt["ssim"]:.4f} -> inferido={m_inf["ssim"]:.4f}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'metrics.csv', index=False)
    print('\n=== SSIM contra a reconstrução do VAE ===')
    print(df.groupby('n_components')[['ssim_roundtrip_vs_vae', 'ssim_inferred_vs_vae']]
            .mean().round(4).to_string())
    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()
