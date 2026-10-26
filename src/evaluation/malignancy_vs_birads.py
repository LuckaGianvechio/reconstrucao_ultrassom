"""Does our model give better malignancy risk scores than the radiologist's BIRADS?

Target: Classification (malignant = 1, benign = 0) on the 252 cases of
src/evaluation/birads_analysis.py (BIRADS 2..5; the 4 "normal" BIRADS 1 cases dropped).
Same folds as that analysis (built from the BIRADS rank), so every model and the
radiologist are scored on identical test folds and compared fold by fold.

1. Model vs. radiologist: CatBoost trained directly on malignancy (acquisition, pre-US,
   embedding, late fusion), ROC-AUC against the BIRADS rank used as-is as a score.
2. Increment over BIRADS: the BIRADS rank as a feature, alone and with pre-US and/or the
   embedding's out-of-fold score (late fusion), against the BIRADS rank alone.
3. Grey zone: within each BIRADS category that holds both classes (4b, 4c, 5) and in
   4a-4c pooled, ROC-AUC of the out-of-fold malignancy score over same-fold pairs only
   (see within_fold_auc), with a bootstrap CI.
   This is where BIRADS itself cannot rank cases, so it is where a model could add to it.

Caveat: benign BIRADS 2-3 cases were confirmed by follow-up, not biopsy, and BIRADS
decided who was biopsied (verification bias in BIRADS's favour).

Outputs in outputs/embedding_to_tabular_experiments/{folder_name}/:
- results_folds.csv, results.csv, oof_predictions.csv, significance_tests.csv,
  within_category.csv, config.json

Usage:
    python -m src.evaluation.malignancy_vs_birads --folder-name malignancy_vs_birads
    python -m src.evaluation.malignancy_vs_birads --folder-name malignancy_vs_birads --rewithin
"""
import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from src.evaluation.acquisition_confound import make_folds
from src.evaluation.birads_analysis import (
    CLASSES, METADATA_COLS, N_INNER, SCHEMES, binary_metrics, fit_predict, pre_us_features, stacked_fit_predict,
)
from src.evaluation.metadata_only_baseline import load_metadata
from src.evaluation.significance_tests import benjamini_hochberg, corrected_t
from src.models.embedding_to_clinical_shared import load_embeddings
from src.models.train_embedding_to_clinical_gbm import OUT_ROOT
from src.preprocessing.clinical_targets import load_raw_clinical

RADIOLOGIST = 'radiologist'  # the BIRADS rank itself, no model
FEATURE_SETS = ['acquisition', 'pre_us', 'acquisition+pre_us', 'embedding',
                'birads', 'birads+pre_us']
# Late fusion: the embedding enters as its out-of-fold score next to these side features.
STACKED_SETS = ['emb_score+pre_us', 'emb_score+acquisition+pre_us', 'emb_score+birads', 'emb_score+birads+pre_us']
GREY_ZONES = {'4b': ['4b'], '4c': ['4c'], '5': ['5'], '4a-4c': ['4a', '4b', '4c']}
N_BOOT = 2000
ALPHA = 0.05

COMPARISONS = (
    [('model vs radiologist', fs, RADIOLOGIST)
     for fs in ['pre_us', 'acquisition+pre_us', 'embedding', 'emb_score+pre_us', 'emb_score+acquisition+pre_us']] +
    [('increment over BIRADS', fs, RADIOLOGIST)
     for fs in ['birads', 'birads+pre_us', 'emb_score+birads', 'emb_score+birads+pre_us']] +
    [('embedding over pre-US', 'emb_score+pre_us', 'pre_us'),
     ('embedding over BIRADS+pre-US', 'emb_score+birads+pre_us', 'birads+pre_us')]
)


def run_set(feature_set: str, X: np.ndarray, y: np.ndarray, X_emb, folds: list, scheme: str) -> tuple:
    rows, oof = [], []
    for fold, (train, test) in enumerate(folds):
        if feature_set == RADIOLOGIST:
            s = X[test, 0]
        elif feature_set in STACKED_SETS:
            s = stacked_fit_predict('binary', X_emb, X, y, train, test)
        else:
            s = fit_predict('binary', X[train], y[train], X[test])
        rows.append({'feature_set': feature_set, 'scheme': scheme, 'fold': fold, 'n_test': len(test),
                     **binary_metrics(y[test], s)})
        oof.append(pd.DataFrame({'idx': test, 'feature_set': feature_set, 'scheme': scheme, 'fold': fold, 'score': s}))
    return rows, oof


def significance(folds_df: pd.DataFrame) -> pd.DataFrame:
    wide = folds_df.pivot_table(index=['scheme', 'fold', 'n_test'], columns='feature_set', values='roc_auc').reset_index()
    rows = []
    for scheme, g in wide.groupby('scheme'):
        for family, a, b in COMPARISONS:
            rows.append({'family': family, 'scheme': scheme, 'a': a, 'b': b, 'a_mean': g[a].mean(), 'b_mean': g[b].mean(),
                         **corrected_t((g[a] - g[b]).to_numpy(), g['n_test'].to_numpy(float), int(g['n_test'].sum()))})
    res = pd.DataFrame(rows)
    res['q'] = res.groupby(['family', 'scheme'], group_keys=False)['p'].apply(benjamini_hochberg)
    res['significant'] = res['q'] < ALPHA
    return res


def within_fold_auc(y: np.ndarray, s: np.ndarray, fold: np.ndarray) -> float:
    """AUC over (malignant, benign) pairs from the same fold only: out-of-fold scores from
    different fold models are not on a common scale, and pooling them inside a single
    BIRADS category would partly rank folds instead of cases. Ties count 1/2."""
    num = den = 0.0
    for f in np.unique(fold):
        m = fold == f
        pos, neg = s[m & (y == 1)], s[m & (y == 0)]
        if len(pos) and len(neg):
            diff = pos[:, None] - neg[None, :]
            num += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            den += diff.size
    return num / den if den else np.nan


def bootstrap_auc(y: np.ndarray, s: np.ndarray, fold: np.ndarray, rng) -> tuple:
    """Within-fold AUC and a percentile 95% CI, resampling cases within each class."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots.append(within_fold_auc(y[idx], s[idx], fold[idx]))
    return within_fold_auc(y, s, fold), *np.nanpercentile(boots, [2.5, 97.5])


def within_category(oof: pd.DataFrame, y: np.ndarray, birads: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for (fs, scheme), g in oof.groupby(['feature_set', 'scheme']):
        g = g.set_index('idx').sort_index()
        s, fold = g['score'].to_numpy(), g['fold'].to_numpy()
        for zone, cats in GREY_ZONES.items():
            m = np.isin(birads, cats)
            auc, lo, hi = bootstrap_auc(y[m], s[m], fold[m], rng)
            rows.append({'zone': zone, 'feature_set': fs, 'scheme': scheme, 'n': int(m.sum()),
                         'n_malignant': int(y[m].sum()), 'roc_auc': auc, 'ci_low': lo, 'ci_high': hi})
    return pd.DataFrame(rows)


def rebuild_within(out_dir) -> pd.DataFrame:
    """Recompute within_category.csv from the saved oof_predictions.csv (no training)."""
    oof = pd.read_csv(out_dir / 'oof_predictions.csv')
    first = oof.drop_duplicates('case').reset_index(drop=True)
    idx = pd.Series(first.index, index=first['case'])
    oof['idx'] = oof['case'].map(idx)
    return within_category(oof, first['malignant'].to_numpy(), first['birads'].astype(str).to_numpy())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder-name', required=True)
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--n-jobs', type=int, default=4)
    p.add_argument('--rewithin', action='store_true',
                   help="Skip training: rebuild within_category.csv from the folder's oof_predictions.csv.")
    args = p.parse_args()
    out_dir = OUT_ROOT / args.folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.rewithin:
        rebuild_within(out_dir).to_csv(out_dir / 'within_category.csv', index=False)
        return

    raw_df = load_raw_clinical()
    emb = load_embeddings('mae', 'cls')
    clinical = raw_df.set_index('Image_filename').reindex(emb.index)
    clinical = clinical[clinical['BIRADS'].astype(str).isin(CLASSES)]
    assert clinical['Classification'].isin(['benign', 'malignant']).all()
    cases = clinical.index
    birads = clinical['BIRADS'].astype(str).to_numpy()
    rank = clinical['BIRADS'].astype(str).map(CLASSES.index).to_numpy()
    y = (clinical['Classification'] == 'malignant').astype(int).to_numpy()

    meta = load_metadata(raw_df).reindex(cases)
    blocks = {'acquisition': meta[METADATA_COLS], 'pre_us': pre_us_features(raw_df).reindex(cases),
              'embedding': emb.loc[cases], 'birads': pd.DataFrame({'birads_rank': rank}, index=cases)}
    inputs = {fs: pd.concat([blocks[b] for b in fs.removeprefix('emb_score+').split('+')], axis=1).to_numpy(float)
              for fs in FEATURE_SETS + STACKED_SETS}
    inputs[RADIOLOGIST] = rank[:, None].astype(float)
    X_emb = inputs['embedding']

    # Same folds as birads_analysis: stratified on the BIRADS rank.
    folds = {scheme: make_folds(scheme, rank, meta, args.n_splits) for scheme in SCHEMES}
    tasks = [(fs, scheme) for scheme in SCHEMES for fs in [RADIOLOGIST] + FEATURE_SETS + STACKED_SETS]
    # Stacked and embedding sets first: they dominate the runtime.
    cost = lambda fs: (N_INNER + 1) * X_emb.shape[1] if fs in STACKED_SETS else inputs[fs].shape[1]
    tasks.sort(key=lambda t: -cost(t[0]))
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_set)(fs, inputs[fs], y, X_emb, folds[scheme], scheme) for fs, scheme in tasks)

    folds_df = pd.DataFrame([r for rows, _ in results for r in rows])
    oof = pd.concat([d for _, oofs in results for d in oofs], ignore_index=True)
    g = folds_df.groupby(['feature_set', 'scheme'])[['roc_auc', 'pr_auc']]
    res = g.mean().add_suffix('_mean').join(g.std().add_suffix('_std')).reset_index()
    sig = significance(folds_df)
    within = within_category(oof, y, birads)
    oof = oof.assign(case=cases.to_numpy()[oof['idx']], birads=birads[oof['idx']], malignant=y[oof['idx']])

    folds_df.to_csv(out_dir / 'results_folds.csv', index=False)
    res.to_csv(out_dir / 'results.csv', index=False)
    oof.drop(columns='idx').to_csv(out_dir / 'oof_predictions.csv', index=False)
    sig.to_csv(out_dir / 'significance_tests.csv', index=False)
    within.to_csv(out_dir / 'within_category.csv', index=False)
    (out_dir / 'config.json').write_text(json.dumps({
        'target': 'Classification == malignant', 'embedding': 'mae_cls', 'model': 'catboost',
        'n_splits': args.n_splits, 'schemes': SCHEMES, 'feature_sets': [RADIOLOGIST] + FEATURE_SETS + STACKED_SETS,
        'n_inner': N_INNER, 'n_cases': len(cases), 'n_malignant': int(y.sum()), 'n_boot': N_BOOT,
        'grey_zones': GREY_ZONES,
    }, indent=2))

    with pd.option_context('display.width', 220, 'display.max_rows', 200):
        print(res.pivot(index='feature_set', columns='scheme', values='roc_auc_mean').round(3).to_string())
        print(sig[['family', 'scheme', 'a', 'b', 'a_mean', 'b_mean', 'gain', 'p', 'q', 'significant']]
              .to_string(index=False, float_format='%.3f'))
        print(within.pivot_table(index=['zone', 'feature_set'], columns='scheme', values='roc_auc').round(3).to_string())
    print(f'\nSaved to {out_dir}')


if __name__ == '__main__':
    main()
