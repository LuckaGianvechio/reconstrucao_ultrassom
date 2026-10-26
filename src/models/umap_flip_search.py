"""How many feature flips does it take to maximally move the inferred reconstruction?

Follow-up to src/models/umap_perturbation_reconstruction.py, which showed that a
SINGLE flip moves the decoded image by only ~2% of the grey scale. The question
here is how that grows: flip 2 features, 5, 10 - where does it saturate, and does
it ever reach the distance between two different real patients?

⚠️⚠️ `umap.inverse_transform` IS BATCH-DEPENDENT - READ BEFORE EDITING.
The same UMAP coordinate inverted alone vs. inside a batch of 52 yields vectors
that differ by a RELATIVE distance > 1.0 (i.e. completely different, not a
rounding artefact). Mechanism, in umap_.py::inverse_transform: the optimisation
schedule comes from `make_epochs_per_sample(graph.data, n_epochs)`, normalised by
the MAXIMUM edge weight over the whole batch, and `optimize_layout_inverse`
consumes one shared `rng_state` sequentially across the batch. So how many epochs
and which negative samples each point receives depend on what else was inverted
alongside it. It is deterministic for a fixed batch and arbitrary across batches.

Consequence: every inversion here is done ONE POINT AT A TIME, which is the only
way it is a well-defined function of its input. An earlier version of this script
scored candidates in batches and verified them individually, producing the
contradiction that step 1 measured dist=1.11 during the search and dist=0.00 in
the final curve.

METHOD. Greedy forward search: starting from the patient's real row, repeatedly
add the single flip that moves the prediction furthest, and record the curve.

The search scores candidates by displacement in UMAP COORDINATE space (5 dims),
which needs no inversion at all. This is a proxy, and deliberately so: a solo
inversion costs ~10s, so scoring ~58 candidates over 8 steps by inversion would
take over an hour. Only the chosen sequence and the controls are actually
inverted (one point at a time) and decoded. The output table prints the z-space
and token-space distances side by side so the proxy can be audited - if they
disagree, the greedy ranking is not trustworthy.

TWO CONTROLS, both necessary:

  * RANDOM-k: flip k features at random, several draws. If random flips move the
    prediction as far as the greedily chosen ones, then there is no specific
    feature structure being exploited - the effect is just "more perturbation,
    more change", and the greedy ranking means nothing.
  * BETWEEN-PATIENT: the distance between predictions for two different real
    patients. This is the yardstick: flips are only interesting if enough of them
    can move one patient's reconstruction as far as simply being another patient.

⚠️ OFF-MANIFOLD CAVEAT. Flipping one-hot columns freely produces rows no real
patient could have (two Shape values at once, or none). Past a couple of flips the
input is outside the training distribution, and a tree ensemble's behaviour there
is extrapolation, not prediction. The `--coherent` mode addresses this: it changes
whole categorical GROUPS to another valid level (exactly one active per group), so
every perturbed row remains a clinically possible patient. Compare the two modes -
if only the free-flip mode produces large changes, the effect lives off-manifold
and says little about real clinical variation.

Usage:
    python -m src.models.umap_flip_search --case case080.png --max-flips 12
    python -m src.models.umap_flip_search --case case080.png --coherent
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
from src.evaluation.image_metrics import all_metrics
from src.models.train_tabular_to_umap import make_regressor, MAE_JSON, TEST_SIZE, SPLIT_SEED
from src.models.umap_inferred_reconstruction_mae import MAE_SIZE, _spatial_tokens
from src.models.umap_perturbation_reconstruction import diff_map, stack_strips, mad


OUT_DIR = Path('outputs/umap_flip_search')

FLIP_PREFIXES = ['Shape_', 'Margin_', 'BIRADS_', 'Echogenicity_', 'Diagnosis_',
                 'Posterior_features_', 'Halo_', 'Calcifications_', 'Tissue_']
N_RANDOM_DRAWS = 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', default='case080.png')
    p.add_argument('--n-components', type=int, default=5)
    p.add_argument('--max-flips', type=int, default=8)
    p.add_argument('--n-control', type=int, default=10,
                   help='How many other patients for the between-patient control. Each costs '
                        'one ~10s solo inversion.')
    p.add_argument('--n-random', type=int, default=4,
                   help='Random-k draws per k. Each costs one ~10s solo inversion.')
    p.add_argument('--coherent', action='store_true',
                   help='Change whole one-hot GROUPS to another valid level instead of '
                        'flipping individual columns, keeping every row a possible patient.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--checkpoint', default='./USF-MAE_full_pretrain_43dataset_100epochs.pt')
    return p.parse_args()


def group_of(name: str) -> str:
    """Categorical group a one-hot column belongs to ('Shape_oval' -> 'Shape')."""
    return name.split('_')[0]


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
    case_i = filenames.index(args.case)
    if case_i not in set(test_idx):
        raise SystemExit(f'{args.case} is in the TRAINING split; pick a test-split case.')

    scaler = StandardScaler()
    reducer = umap.UMAP(n_components=args.n_components,
                        n_neighbors=min(15, len(train_idx) - 1), random_state=42)
    z_train = reducer.fit_transform(scaler.fit_transform(flat[train_idx]))
    regressor = make_regressor()
    regressor.fit(X[train_idx], z_train)

    def to_z(rows: np.ndarray) -> np.ndarray:
        """tabular rows -> predicted UMAP coordinates. No inversion, so this is
        cheap AND batch-independent (CatBoost predicts each row on its own)."""
        return np.asarray(regressor.predict(np.atleast_2d(rows)))

    def to_tokens_solo(row: np.ndarray) -> np.ndarray:
        """One tabular row -> predicted token vector, inverting ONE point at a
        time. Never batch this: see the batch-dependence warning above."""
        z = to_z(row)
        return scaler.inverse_transform(reducer.inverse_transform(z[:1]))[0]

    base_row = X[case_i]
    base_z = to_z(base_row)[0]
    base_tok = to_tokens_solo(base_row)
    z_scale = np.linalg.norm(base_z) or 1.0
    tok_scale = np.linalg.norm(base_tok)

    def zdist(z: np.ndarray) -> float:
        """Relative displacement in UMAP coordinate space (the greedy proxy)."""
        return float(np.linalg.norm(z - base_z) / z_scale)

    def dist(tok: np.ndarray) -> float:
        """Relative L2 distance from the base prediction, in token space."""
        return float(np.linalg.norm(tok - base_tok) / tok_scale)

    # CONTROL: how far apart are predictions for different real patients?
    # Inverted one at a time, like everything else, so it is comparable.
    others = [i for i in test_idx if i != case_i][:args.n_control]
    between = [dist(to_tokens_solo(X[i])) for i in others]
    between_z = [zdist(to_z(X[i])[0]) for i in others]
    ctrl_mean, ctrl_min = float(np.mean(between)), float(np.min(between))
    ctrl_z = float(np.mean(between_z))
    print(f'CONTROLE entre pacientes ({len(others)} outros, invertidos solo):')
    print(f'  token: media={ctrl_mean:.4f}  minima={ctrl_min:.4f}')
    print(f'  z    : media={ctrl_z:.4f}\n')

    cand = [j for j, n in enumerate(feature_names)
            if any(n.startswith(p) for p in FLIP_PREFIXES)]

    def apply_change(row, j):
        """Free flip, or coherent group change when --coherent."""
        out = row.copy()
        if not args.coherent:
            out[j] = 1.0 - out[j]
            return out
        g = group_of(feature_names[j])
        for k in cand:                      # zero the whole group, activate only j
            if group_of(feature_names[k]) == g:
                out[k] = 0.0
        out[j] = 1.0
        return out

    # ---- greedy forward search ----
    print(f'=== Busca gulosa ({"grupos coerentes" if args.coherent else "flips livres"}) ===')
    chosen, chosen_rows, cur = [], [], base_row.copy()
    used_groups = set()
    for step in range(args.max_flips):
        pool = [j for j in cand if j not in chosen]
        if args.coherent:
            pool = [j for j in pool if group_of(feature_names[j]) not in used_groups]
        if not pool:
            break
        zs = to_z(np.stack([apply_change(cur, j) for j in pool]))
        d = [zdist(z) for z in zs]
        best = int(np.argmax(d))
        j = pool[best]
        cur = apply_change(cur, j)
        chosen.append(j)
        chosen_rows.append(cur.copy())
        used_groups.add(group_of(feature_names[j]))
        print(f'  +{step+1:2d} {feature_names[j]:<45} zdist={d[best]:.4f} '
              f'({d[best]/ctrl_z*100:5.1f}% do controle em z)')

    # ---- random-k control ----
    print('\n=== Controle random-k (mesma quantidade, features sorteadas) ===')
    rng = np.random.default_rng(0)
    rand_curve = {}
    for k in range(1, len(chosen) + 1):
        ds = []
        for _ in range(args.n_random):
            row = base_row.copy()
            picks = rng.choice(cand, size=k, replace=False)
            if args.coherent:
                seen, picks2 = set(), []
                for j in picks:
                    if group_of(feature_names[j]) not in seen:
                        seen.add(group_of(feature_names[j])); picks2.append(j)
                picks = picks2
            for j in picks:
                row = apply_change(row, j)
            ds.append(dist(to_tokens_solo(row)))
        rand_curve[k] = (float(np.mean(ds)), float(np.std(ds)))
        print(f'  k={k:2d}  random dist={np.mean(ds):.4f} (+-{np.std(ds):.4f})')

    # ---- decode the greedy sequence ----
    print('\nLoading MAE model (decodificando so a sequencia escolhida) ...')
    model = load_model(args.checkpoint, device=args.device)
    original = Image.open(MAE_JSON.parent / args.case).convert('RGB').resize((MAE_SIZE, MAE_SIZE))
    base_img = decode_embedding(base_tok.reshape(token_shape).astype(np.float32),
                                model, device=args.device)

    rows, imgs, caps = [], [original, base_img], ['original', 'base (0 flips)']
    diffs, dcaps = [original, diff_map(base_img, base_img)], ['', 'diff (base=0)']
    show_steps = [s for s in (1, 2, 3, 4, 6, 8) if s <= len(chosen)]
    for k in range(1, len(chosen) + 1):
        tok = to_tokens_solo(chosen_rows[k - 1])
        img = decode_embedding(tok.reshape(token_shape).astype(np.float32),
                               model, device=args.device)
        m = all_metrics(base_img, img)
        rows.append({'k_flips': k, 'feature_added': feature_names[chosen[k - 1]],
                     'z_dist_greedy': zdist(to_z(chosen_rows[k - 1])[0]),
                     'token_dist_greedy': dist(tok),
                     'token_dist_random_mean': rand_curve[k][0],
                     'token_dist_random_std': rand_curve[k][1],
                     'token_dist_between_patients': ctrl_mean,
                     'ssim_vs_base': m['ssim'], 'mad_vs_base': mad(base_img, img)})
        if k in show_steps:
            imgs.append(img); caps.append(f'{k} flips')
            diffs.append(diff_map(base_img, img))
            dcaps.append(f'mad={mad(base_img, img):.2f}/255')

    suffix = 'coherent' if args.coherent else 'free'
    stack_strips(make_captioned_strip(imgs, caps, label_height=40, font_size=20),
                 make_captioned_strip(diffs, dcaps, label_height=40, font_size=20)
                 ).save(OUT_DIR / f'{Path(args.case).stem}_{suffix}_umap{args.n_components}.png')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f'{Path(args.case).stem}_{suffix}_umap{args.n_components}.csv', index=False)
    print('\n=== Curva ===')
    print(df[['k_flips', 'z_dist_greedy', 'token_dist_greedy', 'token_dist_random_mean',
              'ssim_vs_base', 'mad_vs_base']].round(4).to_string(index=False))
    corr = df['z_dist_greedy'].corr(df['token_dist_greedy'])
    print(f'\nAuditoria do proxy: correlacao z-dist vs token-dist = {corr:.3f} '
          f'({"proxy confiavel" if corr > 0.7 else "PROXY RUIM - ranking guloso suspeito"})')
    print(f'\nDistancia entre pacientes diferentes (controle): {ctrl_mean:.4f}')
    print(f'Maximo alcancado por {len(chosen)} flips: {df.token_dist_greedy.max():.4f} '
          f'({df.token_dist_greedy.max()/ctrl_mean*100:.1f}% do controle)')
    print(f'\nSaved to {OUT_DIR}')


if __name__ == '__main__':
    main()
