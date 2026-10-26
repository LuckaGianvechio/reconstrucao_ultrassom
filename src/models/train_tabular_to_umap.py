"""Predict UMAP-projected image embeddings from tabular clinical features.

This revisits the ORIGINAL direction of the project (tabular -> embedding),
which failed in its raw form (R2~0 for MAE per-patch, R2=0.054 for the SD-VAE;
see ONBOARDING.md and docs/vae-experiment-log.md). The root cause diagnosed
there was that a given patch/latent position does not carry the same visual
content across different patients, so the regressor collapses to the mean.

The new idea (user's): instead of predicting the full embedding, predict the
embedding *projected onto a low-dimensional UMAP manifold*, fit on the training
split only. Reconstruction through the decoder is not possible from such a
projection (UMAP has no exact inverse), but a good regression here would still
be an interesting result on its own: it would mean the tabular clinical data
determines *where on the image manifold* a lesion lies, even if it cannot
determine the exact embedding.

Two design decisions, both deliberate:

1. IMAGE LEVEL, NOT PER PATCH. The per-patch setup is exactly what was
   diagnosed as structurally broken (positions are not comparable across
   patients); running UMAP over patch tokens would only reparametrise that same
   misaligned space. Here each image contributes ONE pooled vector (mean over
   the 196 patch tokens, or the CLS token), so there is no "position" left to
   collapse, and the sample count is 256 images = 256 rows.

2. TWO CONTROLS, without which the numbers are uninterpretable:
   - LOWER CONTROL (`--n-components 0`, reported as `raw`): tabular -> the raw
     768-dim pooled embedding, no UMAP. If UMAP "works" but the raw pooled
     embedding works just as well, the gain came from pooling, not from UMAP.
   - UPPER BOUND (`source='embedding'`): the pooled embedding itself -> its own
     UMAP coordinates. If not even the original embedding can predict its own
     out-of-sample UMAP coordinates, then a low R2 from the tabular side says
     nothing about the tabular data - it says the UMAP coordinates are noise
     out of sample.

Every UMAP (and the StandardScaler feeding it) is fit on the TRAINING split
only and applied to the test split, exactly as in
src/models/train_embedding_to_clinical.py::apply_umap. Each configuration is
repeated over several UMAP seeds, because with ~204 training samples a single
projection can look predictable by chance; the spread across seeds is reported.

Usage:
    python -m src.models.train_tabular_to_umap --pooling mean
    python -m src.models.train_tabular_to_umap --pooling cls
    python -m src.models.train_tabular_to_umap --pooling mean --n-components 2 5 --seeds 1
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import umap

from src.preprocessing.build_tabular_dataset import build_clinical_features
from src.preprocessing.clinical_targets import load_raw_clinical
from src.preprocessing.embedding_pooling import load_mae_pooled


MAE_JSON = Path('dataset_breast/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/'
                'BrEaST-Lesions_USG-images_and_masks/embeddings_per_patch.json')
OUT_DIR = Path('outputs/tabular_to_umap')

TEST_SIZE = 0.2
SPLIT_SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pooling', choices=['mean', 'cls'], default='mean',
                   help='How to pool the 197 MAE tokens into one vector per image.')
    p.add_argument('--n-components', type=int, nargs='+', default=[2, 5, 10, 30],
                   help='UMAP output dimensionalities to test (0 means "no UMAP", the raw control).')
    p.add_argument('--seeds', type=int, default=3,
                   help='How many UMAP seeds to repeat each configuration with.')
    return p.parse_args()


def make_regressor():
    """One CatBoost with the native MultiRMSE loss (a single multi-output model,
    not 768 wrapped single-output ones) - same hyperparameters as the original
    tabular->embedding experiment in src/models/train_catboost.py, so the
    comparison against that failed baseline stays apples-to-apples."""
    return CatBoostRegressor(
        loss_function='MultiRMSE',
        iterations=150,
        depth=6,
        learning_rate=0.08,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False,
    )


def project_umap(Y_train: np.ndarray, Y_test: np.ndarray, n_components: int, seed: int):
    """Fit StandardScaler + UMAP on the TRAINING embeddings only, transform both
    splits. Returns the projected (train, test) targets."""
    scaler = StandardScaler()
    Y_train_scaled = scaler.fit_transform(Y_train)
    Y_test_scaled = scaler.transform(Y_test)

    n_neighbors = min(15, len(Y_train) - 1)
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=seed)
    return reducer.fit_transform(Y_train_scaled), reducer.transform(Y_test_scaled)


def eval_multioutput(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-dimension R2 summarised several ways, plus a scale-free correlation.

    `r2_mean` is the headline number (mean over output dimensions). `r2_median`
    and `dims_r2_positive` guard against a single lucky/unlucky dimension
    dominating the average. `pearson_mean` is reported separately because a
    regressor can get the *direction* of a coordinate right while systematically
    shrinking its amplitude (the classic mean-collapse signature) - that shows
    up as R2<=0 but a clearly positive correlation."""
    n_dims = y_true.shape[1]
    r2_per_dim = np.array([r2_score(y_true[:, d], y_pred[:, d]) for d in range(n_dims)])

    pearson = []
    for d in range(n_dims):
        # A constant prediction (or a constant target dimension) makes Pearson
        # undefined; numpy would emit a divide-by-zero warning instead.
        if np.std(y_true[:, d]) < 1e-12 or np.std(y_pred[:, d]) < 1e-12:
            pearson.append(np.nan)
        else:
            pearson.append(np.corrcoef(y_true[:, d], y_pred[:, d])[0, 1])
    pearson = np.array(pearson)
    # All-NaN happens for the DummyRegressor baseline, whose predictions are
    # constant, so correlation is undefined rather than zero - report NaN
    # instead of letting np.nanmean warn about an empty slice.
    pearson_mean = float(np.nanmean(pearson)) if not np.all(np.isnan(pearson)) else float('nan')

    return {
        'r2_mean': float(r2_per_dim.mean()),
        'r2_median': float(np.median(r2_per_dim)),
        'r2_best_dim': float(r2_per_dim.max()),
        'dims_r2_positive': int((r2_per_dim > 0).sum()),
        'n_dims': n_dims,
        'pearson_mean': pearson_mean,
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(root_mean_squared_error(y_true, y_pred)),
    }


def run_config(X_train, X_test, Y_train_emb, Y_test_emb, n_components, seed, source):
    """One (target space, feature source) combination.

    `source='tabular'` is the experiment itself; `source='embedding'` is the
    upper-bound control, where the features are the pooled embedding instead of
    the clinical table - it answers "are these UMAP coordinates predictable at
    all out of sample?" before we blame the tabular data for a low R2."""
    if n_components == 0:
        Y_train, Y_test = Y_train_emb, Y_test_emb  # raw control: the 768-dim embedding itself
    else:
        Y_train, Y_test = project_umap(Y_train_emb, Y_test_emb, n_components, seed)

    features_train = X_train if source == 'tabular' else Y_train_emb
    features_test = X_test if source == 'tabular' else Y_test_emb

    model = make_regressor()
    model.fit(features_train, Y_train)
    preds = np.asarray(model.predict(features_test))

    baseline = DummyRegressor(strategy='mean')
    baseline.fit(features_train, Y_train)
    baseline_preds = np.asarray(baseline.predict(features_test))

    metrics = eval_multioutput(Y_test, preds)
    baseline_metrics = eval_multioutput(Y_test, baseline_preds)

    return {
        'source': source,
        'n_components': n_components if n_components else 768,
        'target_space': 'raw_embedding' if n_components == 0 else f'umap{n_components}',
        'umap_seed': seed,
        **metrics,
        'r2_mean_baseline': baseline_metrics['r2_mean'],
        'delta_r2': metrics['r2_mean'] - baseline_metrics['r2_mean'],
    }


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('ignore', category=UserWarning)

    raw_df = load_raw_clinical()
    clinical = build_clinical_features(raw_df)
    clinical.index = raw_df['Image_filename']

    print(f'Loading MAE embeddings ({args.pooling}-pooled) from {MAE_JSON} ...')
    pooled = load_mae_pooled(MAE_JSON, pooling=args.pooling)

    filenames = [f for f in clinical.index if f in pooled]
    X = clinical.loc[filenames].to_numpy(dtype=float)
    Y = np.stack([pooled[f] for f in filenames]).astype(np.float32)
    print(f'{len(filenames)} images | tabular features: {X.shape[1]} | embedding dims: {Y.shape[1]}')

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=SPLIT_SEED,
    )
    print(f'Train: {len(X_train)} | Test: {len(X_test)}')

    rows = []
    t0 = time.time()

    for n_components in args.n_components:
        # The raw 768-dim control has no UMAP in it, so seeds would only produce
        # identical rows - run it once.
        seeds = [0] if n_components == 0 else list(range(args.seeds))
        # For the raw 768-dim control, source='embedding' would be the embedding
        # predicting itself (an identity map) - trivially perfect and meaningless,
        # so the upper-bound control only applies to the UMAP target spaces.
        sources = ('tabular',) if n_components == 0 else ('tabular', 'embedding')
        for seed in seeds:
            for source in sources:
                label = 'raw768' if n_components == 0 else f'umap{n_components}'
                print(f'[{source:>9} -> {label:<8} seed={seed}] ', end='', flush=True)
                result = run_config(X_train, X_test, Y_train, Y_test, n_components, seed, source)
                rows.append(result)
                print(f'r2_mean={result["r2_mean"]:+.4f} '
                      f'pearson={result["pearson_mean"]:+.4f} '
                      f'dims_pos={result["dims_r2_positive"]}/{result["n_dims"]}')

    results_df = pd.DataFrame(rows)
    out_path = OUT_DIR / f'mae_{args.pooling}_tabular_to_umap.csv'
    results_df.to_csv(out_path, index=False)

    print()
    print(f'=== tabular -> UMAP-projected MAE embedding ({args.pooling}-pooled) ===')
    summary = (results_df
               .groupby(['source', 'target_space', 'n_components'], as_index=False)
               .agg(r2_mean=('r2_mean', 'mean'),
                    r2_std=('r2_mean', 'std'),
                    pearson_mean=('pearson_mean', 'mean'),
                    dims_r2_positive=('dims_r2_positive', 'mean'),
                    n_dims=('n_dims', 'first'),
                    delta_r2=('delta_r2', 'mean'))
               .sort_values(['source', 'n_components']))
    print(summary.to_string(index=False))

    print()
    print('Total runtime: %.1fs' % (time.time() - t0))
    print('Saved to', out_path)


if __name__ == '__main__':
    main()
