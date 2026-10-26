"""The MAE counterpart of src/models/umap_inferred_reconstruction.py.

That script ran the reconstruction chain on the SD-VAE latent, because the MAE
*pooled* embedding (a mean over 196 patch tokens) cannot be decoded at all. The
MAE does have a decodable representation, though: the full per-patch token
tensor (1, 197, 768) that serve/inference.py::decode_embedding consumes. That is
151,296 dims per image - 37x wider than the VAE's 4,096 - which is the whole
point of running it: it tests whether the UMAP round-trip failure was specific
to the VAE latent or is a general property of squeezing an image representation
through 2-5 manifold coordinates.

Chain (identical in structure to the VAE version):

    tabular clinical features -> CatBoost -> UMAP coords -> inverse_transform
                              -> tokens (1,197,768) -> MAE decoder -> image

Four panels per strip, each adding exactly one source of loss over the previous:

  1. `original`        - the source image, resized to the MAE's 224x224.
     (Tokens are first restored to true spatial order via `ids_restore`; see
     `_spatial_tokens` - without that every panel decodes to a scrambled mosaic.)
  2. `MAE roundtrip`   - decode(true tokens). Loss: the MAE encoder/decoder only.
                         The ceiling - every route ends in this same decoder.
  3. `UMAP roundtrip`  - decode(inverse(project(true tokens))). Adds the UMAP
                         projection loss, with NO regression involved.
  4. `UMAP inferido`   - decode(inverse(CatBoost(tabular))). Adds regression error.

Only 2 and 5 components are viable: umap-learn's inverse_transform OOMs at 10
dims and overflows Qhull at 50 (see docs/tabular-to-umap-log.md).

Everything (StandardScaler, UMAP, CatBoost) is fit on the TRAINING split only;
rendered images come from the TEST split.

Usage:
    python -m src.models.umap_inferred_reconstruction_mae --n-components 2 5
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import umap

from serve.inference import load_model, base64_to_embedding, decode_embedding
from serve.reconstruct_and_compile import make_captioned_strip
from src.preprocessing.build_tabular_dataset import build_clinical_features
from src.preprocessing.clinical_targets import load_raw_clinical
from src.evaluation.image_metrics import all_metrics
from src.models.train_tabular_to_umap import make_regressor, MAE_JSON, TEST_SIZE, SPLIT_SEED


DATA_DIR = MAE_JSON.parent
OUT_DIR = Path('outputs/umap_inferred_reconstruction_mae')

# The MAE operates at 224x224 (patch16 -> 14x14 grid -> 196 patches + CLS).
MAE_SIZE = 224

DEFAULT_CASES = ['case229.png', 'case007.png', 'case080.png', 'case207.png']


def _spatial_tokens(rec) -> np.ndarray:
    """Return an image's (197, 768) tokens in TRUE SPATIAL ORDER.

    This reordering is mandatory, not cosmetic. MAE's `random_masking` shuffles
    the patch sequence with `argsort(rand)` even at mask_ratio=0.0 (see
    mae/models_mae.py), and the shuffle is drawn PER IMAGE. So token index `i` in
    embeddings_per_patch.json is a different spatial patch in every image.
    Flattening the tensor without undoing that would make UMAP dimension j mean a
    different patch for each image - and decoding it would produce a scrambled
    mosaic instead of an ultrasound.

    `ids_restore` (saved alongside the embedding by serve/generate_embeddings.py)
    is the permutation that undoes it, applied the same way MAE's own
    forward_decoder does. The CLS token is exempt: it is prepended AFTER masking,
    so index 0 is always CLS.

    Once tokens are in spatial order they can be decoded with ids_restore=None,
    since decode_embedding then gathers with arange (the identity) - verified to
    be byte-identical to passing the shuffled tokens with their real ids_restore.
    """
    emb = base64_to_embedding(rec['b64'], tuple(rec['shape']),
                              rec.get('dtype', 'float16'))[0]  # (197, 768)
    ids_restore = rec.get('ids_restore')
    if ids_restore is None:
        raise ValueError('embedding record has no ids_restore; cannot recover spatial order')
    ids = np.asarray(ids_restore, dtype=np.int64).ravel()       # (196,)
    return np.concatenate([emb[:1], emb[1:][ids]], axis=0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-components', type=int, nargs='+', default=[2, 5])
    p.add_argument('--cases', nargs='+', default=DEFAULT_CASES)
    p.add_argument('--device', default='cpu')
    p.add_argument('--checkpoint', default=os.environ.get(
        'MAE_CHECKPOINT', './USF-MAE_full_pretrain_43dataset_100epochs.pt'))
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = json.load(open(MAE_JSON))
    raw_df = load_raw_clinical()
    clinical = build_clinical_features(raw_df)
    clinical.index = raw_df['Image_filename']

    filenames = sorted(f for f in records if f in clinical.index)
    # float32 throughout: at 151,296 dims the float64 copies StandardScaler would
    # otherwise make are ~310MB each, which this machine does not have to spare.
    tokens = np.stack([_spatial_tokens(records[f]) for f in filenames]).astype(np.float32)
    token_shape = tokens.shape[1:]
    flat = tokens.reshape(len(filenames), -1)
    X = clinical.loc[filenames].to_numpy(dtype=float)
    print(f'{len(filenames)} images | tabular {X.shape[1]} feats | tokens {token_shape} '
          f'-> {flat.shape[1]} dims ({flat.nbytes / 1e6:.0f} MB)')

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
    train_scaled = scaler.fit_transform(flat[train_idx]).astype(np.float32)
    test_scaled = scaler.transform(flat[test_idx]).astype(np.float32)

    print('Loading MAE model ...')
    model = load_model(args.checkpoint, device=args.device)

    originals = {i: Image.open(DATA_DIR / filenames[i]).convert('RGB')
                       .resize((MAE_SIZE, MAE_SIZE)) for i in show_idx}
    mae_recon = {i: decode_embedding(tokens[i], model, device=args.device) for i in show_idx}

    rows = []
    for n_components in args.n_components:
        print(f'\n=== UMAP {n_components} components ===')
        reducer = umap.UMAP(n_components=n_components,
                            n_neighbors=min(15, len(train_idx) - 1),
                            random_state=42)
        z_train = reducer.fit_transform(train_scaled)
        z_test = reducer.transform(test_scaled)

        back_true = scaler.inverse_transform(reducer.inverse_transform(z_test))

        regressor = make_regressor()
        regressor.fit(X[train_idx], z_train)
        z_pred = np.asarray(regressor.predict(X[test_idx]))
        back_pred = scaler.inverse_transform(reducer.inverse_transform(z_pred))

        # Token-space fidelity over the whole test split, comparable to the
        # latent-space R2 reported for the VAE.
        ss_res = ((flat[test_idx] - back_true) ** 2).sum()
        ss_tot = ((flat[test_idx] - flat[train_idx].mean(axis=0)) ** 2).sum()
        print(f'token-space R2 (roundtrip, {len(test_idx)} test imgs): {1 - ss_res / ss_tot:+.4f}')

        for i in show_idx:
            k = pos_in_test[i]
            img_rt = decode_embedding(back_true[k].reshape(token_shape).astype(np.float32),
                                      model, device=args.device)
            img_inf = decode_embedding(back_pred[k].reshape(token_shape).astype(np.float32),
                                       model, device=args.device)

            name = Path(filenames[i]).stem
            strip = make_captioned_strip(
                [originals[i], mae_recon[i], img_rt, img_inf],
                ['original', 'MAE roundtrip (teto)',
                 f'UMAP-{n_components} roundtrip', f'UMAP-{n_components} inferido (tabular)'],
                label_height=40, font_size=24,
            )
            strip.save(OUT_DIR / f'{name}_umap{n_components}_inferred.png')

            m_rt = all_metrics(mae_recon[i], img_rt)
            m_inf = all_metrics(mae_recon[i], img_inf)
            rows.append({'image': name, 'n_components': n_components,
                         'ssim_roundtrip_vs_mae': m_rt['ssim'],
                         'ssim_inferred_vs_mae': m_inf['ssim'],
                         'psnr_inferred_vs_mae': m_inf['psnr']})
            print(f'  {name}: SSIM roundtrip={m_rt["ssim"]:.4f} -> inferido={m_inf["ssim"]:.4f}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'metrics.csv', index=False)
    print('\n=== SSIM contra a reconstrucao do MAE ===')
    print(df.groupby('n_components')[['ssim_roundtrip_vs_mae', 'ssim_inferred_vs_mae']]
            .mean().round(4).to_string())
    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()
