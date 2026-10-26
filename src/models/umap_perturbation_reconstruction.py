"""Sensitivity of the inferred reconstruction to perturbations of the tabular row.

Takes ONE patient, perturbs their clinical feature vector, and decodes the image
that the full inference chain produces for each perturbed version:

    tabular row (perturbed) -> CatBoost -> UMAP coords -> inverse_transform
                            -> tokens (1,197,768) -> MAE decoder -> image

Two perturbation families:

  * GAUSSIAN, calibrated as a fraction of each feature's training standard
    deviation - the same convention serve/perturbation_test.py already uses on
    the embedding side, so the two sensitivity tests are directly comparable.
  * SINGLE-FEATURE FLIPS of clinically meaningful one-hot columns (shape, margin,
    BI-RADS, ...). For a one-hot column a flip IS the smallest meaningful change;
    a fractional nudge has no clinical interpretation.

WHY THE CONTROL MATTERS. The inferred reconstructions are known to be nearly
featureless (see LOGS.md), so "the images all look the same" is the expected
outcome and proves nothing on its own. What makes the result interpretable is the
reference distance printed alongside: how far apart the predictions for two
DIFFERENT real patients are, under the same metric. If perturbing one patient's
features moves the image less than the typical gap between two real patients,
the chain is insensitive to the clinical input - which is a statement about the
model, not about the perturbation being too small.

Everything (StandardScaler, UMAP, CatBoost) is fit on the TRAINING split only,
and the patient rendered must come from the TEST split.

Usage:
    python -m src.models.umap_perturbation_reconstruction --case case080.png
    python -m src.models.umap_perturbation_reconstruction --case case080.png --n-components 2
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

from serve.inference import load_model, decode_embedding
from serve.reconstruct_and_compile import make_captioned_strip
from src.preprocessing.build_tabular_dataset import build_clinical_features
from src.preprocessing.clinical_targets import load_raw_clinical
from src.preprocessing.mae_tokens import restore_spatial_order
from src.evaluation.image_metrics import all_metrics
from src.models.train_tabular_to_umap import make_regressor, MAE_JSON, TEST_SIZE, SPLIT_SEED
from src.models.umap_inferred_reconstruction_mae import MAE_SIZE, _spatial_tokens


OUT_DIR = Path('outputs/umap_perturbation_reconstruction')

# Fractions of each feature's training std to inject as gaussian noise.
FRACTIONS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

# Substrings of clinically meaningful one-hot columns to flip one at a time.
FLIP_PATTERNS = ['Shape_', 'Margin_', 'BIRADS_', 'Echogenicity_', 'Diagnosis_']

N_NOISE_DRAWS = 5  # draws per fraction for the metrics (images use the first draw)


def diff_map(base: Image.Image, other: Image.Image, gain: int = 8) -> Image.Image:
    """|other - base| amplified `gain`x, as a grayscale image.

    Necessary because the perturbed reconstructions differ from the base by only
    a couple of grey levels out of 255 - at native contrast the panels look
    identical and the reader cannot tell "no change" from "small change". The
    amplified map makes the spatial pattern of the change visible; the honest
    magnitude stays in the printed mean-absolute-difference, not in this image.
    """
    a = np.asarray(base.convert('L')).astype(np.int16)
    b = np.asarray(other.convert('L')).astype(np.int16)
    d = np.clip(np.abs(b - a) * gain, 0, 255).astype(np.uint8)
    return Image.fromarray(d).convert('RGB')


def stack_strips(top: Image.Image, bottom: Image.Image) -> Image.Image:
    """Stack two captioned strips vertically into one image."""
    w = max(top.width, bottom.width)
    out = Image.new('RGB', (w, top.height + bottom.height), (255, 255, 255))
    out.paste(top, (0, 0))
    out.paste(bottom, (0, top.height))
    return out


def mad(base: Image.Image, other: Image.Image) -> float:
    """Mean absolute difference in grey levels (0-255) - the plain-language
    magnitude that SSIM alone can overstate for subtle, spread-out changes."""
    a = np.asarray(base.convert('L')).astype(float)
    b = np.asarray(other.convert('L')).astype(float)
    return float(np.abs(b - a).mean())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', default='case080.png', help='Filename; must be in the TEST split.')
    p.add_argument('--n-components', type=int, default=5, help='Only 2 or 5 are viable.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--checkpoint', default='./USF-MAE_full_pretrain_43dataset_100epochs.pt')
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = json.load(open(MAE_JSON))
    raw_df = load_raw_clinical()
    clinical = build_clinical_features(raw_df)
    clinical.index = raw_df['Image_filename']

    filenames = sorted(f for f in records if f in clinical.index)
    tokens = np.stack([_spatial_tokens(records[f]) for f in filenames]).astype(np.float32)
    token_shape = tokens.shape[1:]
    flat = tokens.reshape(len(filenames), -1)
    feature_names = list(clinical.columns)
    X = clinical.loc[filenames].to_numpy(dtype=float)

    train_idx, test_idx = train_test_split(
        np.arange(len(filenames)), test_size=TEST_SIZE, random_state=SPLIT_SEED)
    pos_in_test = {i: k for k, i in enumerate(test_idx)}

    case_i = filenames.index(args.case)
    if case_i not in pos_in_test:
        raise SystemExit(f'{args.case} is in the TRAINING split; pick a test-split case.')

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(flat[train_idx])
    test_scaled = scaler.transform(flat[test_idx])

    reducer = umap.UMAP(n_components=args.n_components,
                        n_neighbors=min(15, len(train_idx) - 1), random_state=42)
    z_train = reducer.fit_transform(train_scaled)

    regressor = make_regressor()
    regressor.fit(X[train_idx], z_train)

    print('Loading MAE model ...')
    model = load_model(args.checkpoint, device=args.device)

    def predict_image(feature_rows: np.ndarray):
        """tabular rows -> decoded images (the full inference chain)."""
        z = np.asarray(regressor.predict(np.atleast_2d(feature_rows)))
        back = scaler.inverse_transform(reducer.inverse_transform(z))
        return [decode_embedding(b.reshape(token_shape).astype(np.float32),
                                 model, device=args.device) for b in back]

    base_row = X[case_i]
    base_img = predict_image(base_row)[0]
    original = Image.open(MAE_JSON.parent / args.case).convert('RGB').resize((MAE_SIZE, MAE_SIZE))

    # CONTROL: how far apart are the predictions for different real patients?
    # This is the yardstick every perturbation number below is measured against.
    other_test = [i for i in test_idx if i != case_i][:12]
    other_imgs = predict_image(X[other_test])
    between_patient = [all_metrics(base_img, im)['ssim'] for im in other_imgs]
    ctrl_mean, ctrl_min = float(np.mean(between_patient)), float(np.min(between_patient))
    print(f'\nCONTROLE - SSIM entre a predicao deste paciente e a de {len(other_test)} outros:')
    print(f'  media={ctrl_mean:.4f}  minimo={ctrl_min:.4f}')
    print('  (uma perturbacao so e relevante se derrubar o SSIM ate perto desta faixa)\n')

    std = X[train_idx].std(axis=0)
    rows = []
    grid_imgs, grid_caps = [original, base_img], ['original', 'predicao base']
    grid_diffs, grid_dcaps = [original, diff_map(base_img, base_img)], ['', 'diff (base=0)']

    print('=== Perturbacao gaussiana (fracao do desvio-padrao por feature) ===')
    for frac in FRACTIONS:
        if frac == 0.0:
            continue
        ssims = []
        for draw in range(N_NOISE_DRAWS):
            rng = np.random.default_rng(1000 + draw)
            pert = base_row + rng.normal(0.0, 1.0, size=base_row.shape) * frac * std
            img = predict_image(pert)[0]
            ssims.append(all_metrics(base_img, img)['ssim'])
            if draw == 0:
                grid_imgs.append(img)
                grid_caps.append(f'ruido {frac}x sd')
                grid_diffs.append(diff_map(base_img, img))
                grid_dcaps.append(f'mad={mad(base_img, img):.2f}/255')
        rows.append({'kind': 'gaussian', 'label': f'{frac}x_sd',
                     'ssim_vs_base_mean': float(np.mean(ssims)),
                     'ssim_vs_base_std': float(np.std(ssims)),
                     'mad_vs_base': mad(base_img, grid_imgs[-1]),
                     'ssim_between_patients_mean': ctrl_mean})
        print(f'  ruido {frac:>4}x sd -> SSIM vs base = {np.mean(ssims):.4f} '
              f'(+-{np.std(ssims):.4f})  mad = {mad(base_img, grid_imgs[-1]):.2f}/255')

    stack_strips(
        make_captioned_strip(grid_imgs, grid_caps, label_height=40, font_size=20),
        make_captioned_strip(grid_diffs, grid_dcaps, label_height=40, font_size=20),
    ).save(OUT_DIR / f'{Path(args.case).stem}_gaussian_umap{args.n_components}.png')

    print('\n=== Flip de uma feature clinica por vez ===')
    flip_cols = [j for j, n in enumerate(feature_names)
                 if any(n.startswith(p) for p in FLIP_PATTERNS)]
    # Evaluate EVERY flip first, then render the most influential ones. Rendering
    # in column order would show Shape_*/Echogenicity_* (the least sensitive) and
    # silently omit the strongest effects, which is what an earlier version did.
    flip_results = []
    for j in flip_cols:
        pert = base_row.copy()
        pert[j] = 1.0 - pert[j]          # one-hot: flip is the smallest meaningful change
        img = predict_image(pert)[0]
        ssim = all_metrics(base_img, img)['ssim']
        flip_results.append((feature_names[j], ssim, mad(base_img, img), img))
        rows.append({'kind': 'flip', 'label': feature_names[j],
                     'ssim_vs_base_mean': ssim, 'ssim_vs_base_std': 0.0,
                     'mad_vs_base': mad(base_img, img),
                     'ssim_between_patients_mean': ctrl_mean})
        print(f'  {feature_names[j]:<40} SSIM = {ssim:.4f}  mad = {mad(base_img, img):.2f}/255')

    flip_results.sort(key=lambda r: r[1])          # lowest SSIM = most influential
    top = flip_results[:6]
    stack_strips(
        make_captioned_strip([original, base_img] + [r[3] for r in top],
                             ['original', 'predicao base'] + [r[0][:22] for r in top],
                             label_height=40, font_size=18),
        make_captioned_strip([original, diff_map(base_img, base_img)] +
                             [diff_map(base_img, r[3]) for r in top],
                             ['', 'diff (base=0)'] + [f'mad={r[2]:.2f}/255' for r in top],
                             label_height=40, font_size=18),
    ).save(OUT_DIR / f'{Path(args.case).stem}_flips_umap{args.n_components}.png')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f'{Path(args.case).stem}_umap{args.n_components}_metrics.csv', index=False)
    print(f'\nSSIM minimo obtido por qualquer perturbacao: {df.ssim_vs_base_mean.min():.4f}')
    print(f'SSIM medio entre pacientes diferentes (controle): {ctrl_mean:.4f}')
    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()
