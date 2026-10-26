"""Close the loop on src/models/train_tabular_to_umap.py: invert the UMAP
projection back into the original 768-dim embedding space.

WHY THIS EXISTS. train_tabular_to_umap.py measures R2 in UMAP space (R2~0.29)
and compares it against a control measured in embedding space (R2~0.08). Those
two numbers live in different spaces, so the comparison is suggestive but not
strictly valid: R2 averaged over 5 UMAP coordinates and R2 averaged over 768
embedding dimensions are not the same quantity. umap-learn's
`inverse_transform` lets us map a predicted UMAP coordinate back to a 768-dim
vector, so every number here is computed in ONE space - the original embedding
space - making the comparison apples-to-apples.

WHAT THIS DOES *NOT* DO. This does not enable image reconstruction. The
irreversible step in the pipeline is the POOLING, not the UMAP: the pooled
embedding is the mean of 196 patch tokens, and no inverse recovers 196 vectors
from their average. Even a perfect UMAP inverse lands on a single 768-dim
vector, while the MAE decoder needs all 196 tokens. `inverse_transform` is also
only an approximation (it interpolates among training points), so it adds its
own loss on top.

Three quantities, all as R2 in the original 768-dim embedding space:

  1. `umap_roundtrip_ceiling` - take the TRUE test embedding, project it to
     UMAP, invert it straight back. No regression involved. This isolates how
     much information the UMAP projection itself destroys, and is the hard
     ceiling for anything routed through this manifold.
  2. `tabular_via_umap` - the real experiment: tabular -> predicted UMAP
     coordinates -> inverse_transform -> 768-dim vector.
  3. `direct_tabular` - the control already measured in train_tabular_to_umap.py
     (tabular -> 768-dim embedding directly, no UMAP), recomputed here so it is
     produced by the exact same code path and metric.

If (2) beats (3), routing through the manifold genuinely helps even after
paying the inversion cost. If (2) merely approaches (1), the bottleneck is the
projection, not the regression.

Usage:
    python -m src.models.train_tabular_to_umap_inverse --pooling mean
    python -m src.models.train_tabular_to_umap_inverse --pooling cls
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import umap

from src.preprocessing.build_tabular_dataset import build_clinical_features
from src.preprocessing.clinical_targets import load_raw_clinical
from src.preprocessing.embedding_pooling import load_mae_pooled
from src.models.train_tabular_to_umap import MAE_JSON, SPLIT_SEED, TEST_SIZE, make_regressor


OUT_DIR = Path('outputs/tabular_to_umap')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pooling', choices=['mean', 'cls'], default='mean')
    p.add_argument('--n-components', type=int, nargs='+', default=[2, 5, 10, 30])
    p.add_argument('--seeds', type=int, default=3)
    return p.parse_args()


def r2_in_embedding_space(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-dimension R2 over the 768 embedding dimensions, summarised. Kept
    separate from train_tabular_to_umap.eval_multioutput because here the number
    of dimensions is always 768, so the dimension-count caveats do not apply."""
    n_dims = y_true.shape[1]
    r2_per_dim = np.array([r2_score(y_true[:, d], y_pred[:, d]) for d in range(n_dims)])
    return {
        'r2_mean': float(r2_per_dim.mean()),
        'r2_median': float(np.median(r2_per_dim)),
        'dims_r2_positive': int((r2_per_dim > 0).sum()),
    }


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('ignore')

    raw_df = load_raw_clinical()
    clinical = build_clinical_features(raw_df)
    clinical.index = raw_df['Image_filename']

    pooled = load_mae_pooled(MAE_JSON, pooling=args.pooling)
    filenames = [f for f in clinical.index if f in pooled]
    X = clinical.loc[filenames].to_numpy(dtype=float)
    Y = np.stack([pooled[f] for f in filenames]).astype(np.float32)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=SPLIT_SEED,
    )
    print(f'{len(filenames)} images | train {len(X_train)} | test {len(X_test)} | dims {Y.shape[1]}')

    rows = []
    t0 = time.time()

    # Control (3): straight tabular -> 768-dim embedding, no manifold involved.
    print('[direct_tabular -> 768d] ', end='', flush=True)
    direct = make_regressor()
    direct.fit(X_train, Y_train)
    direct_metrics = r2_in_embedding_space(Y_test, np.asarray(direct.predict(X_test)))
    baseline = DummyRegressor(strategy='mean').fit(X_train, Y_train)
    baseline_r2 = r2_in_embedding_space(Y_test, np.asarray(baseline.predict(X_test)))['r2_mean']
    rows.append({'route': 'direct_tabular', 'n_components': 768, 'umap_seed': -1,
                 **direct_metrics, 'r2_baseline': baseline_r2})
    print(f'r2={direct_metrics["r2_mean"]:+.4f}')

    for n_components in args.n_components:
        for seed in range(args.seeds):
            scaler = StandardScaler()
            Y_train_scaled = scaler.fit_transform(Y_train)
            Y_test_scaled = scaler.transform(Y_test)

            reducer = umap.UMAP(n_components=n_components,
                                n_neighbors=min(15, len(Y_train) - 1),
                                random_state=seed)
            Z_train = reducer.fit_transform(Y_train_scaled)
            Z_test = reducer.transform(Y_test_scaled)

            # (1) Ceiling: true embedding -> UMAP -> straight back, no regression.
            back_true = scaler.inverse_transform(reducer.inverse_transform(Z_test))
            ceiling = r2_in_embedding_space(Y_test, back_true)
            rows.append({'route': 'umap_roundtrip_ceiling', 'n_components': n_components,
                         'umap_seed': seed, **ceiling, 'r2_baseline': baseline_r2})

            # (2) The experiment: tabular -> predicted UMAP coords -> inverse.
            model = make_regressor()
            model.fit(X_train, Z_train)
            Z_pred = np.asarray(model.predict(X_test))
            back_pred = scaler.inverse_transform(reducer.inverse_transform(Z_pred))
            via_umap = r2_in_embedding_space(Y_test, back_pred)
            rows.append({'route': 'tabular_via_umap', 'n_components': n_components,
                         'umap_seed': seed, **via_umap, 'r2_baseline': baseline_r2})

            print(f'[n={n_components:<3} seed={seed}] ceiling={ceiling["r2_mean"]:+.4f}  '
                  f'via_umap={via_umap["r2_mean"]:+.4f}')

    results_df = pd.DataFrame(rows)
    out_path = OUT_DIR / f'mae_{args.pooling}_umap_inverse.csv'
    results_df.to_csv(out_path, index=False)

    print()
    print(f'=== All R2 measured in the ORIGINAL 768-dim embedding space ({args.pooling}-pooled) ===')
    summary = (results_df
               .groupby(['route', 'n_components'], as_index=False)
               .agg(r2_mean=('r2_mean', 'mean'),
                    r2_std=('r2_mean', 'std'),
                    dims_r2_positive=('dims_r2_positive', 'mean'))
               .sort_values(['route', 'n_components']))
    print(summary.to_string(index=False))
    print()
    print(f'DummyRegressor baseline r2: {baseline_r2:+.4f}')
    print('Total runtime: %.1fs' % (time.time() - t0))
    print('Saved to', out_path)


if __name__ == '__main__':
    main()
