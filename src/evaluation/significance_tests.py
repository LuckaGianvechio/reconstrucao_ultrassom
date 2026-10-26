"""Is the global strategy's ROC-AUC gain over the acquisition baseline significant?

For each target and CV scheme in acquisition_confound/results_folds.csv (embedding =
mae_cls raw CatBoost and acquisition baseline evaluated on the same folds):
- estimator: mean of the per-fold paired differences (embedding - acquisition);
- standard error: Nadeau & Bengio (2003) corrected resampled t, i.e. the fold-difference
  variance scaled by (1/k + n_test/n_train) instead of 1/k, because CV folds share training
  data and are not independent;
- reference distribution: Student t with k-1 degrees of freedom (k = valid folds);
- p two-sided for H0: gain = 0; q = Benjamini-Hochberg within each (target group, scheme).

Only the global strategy is tested: it was chosen on the mean over targets before looking
at any single target, unlike the per-variable best (max over 88 combinations).

Writes acquisition_confound/significance_tests.csv.

Usage:
    python -m src.evaluation.significance_tests
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.models.train_embedding_to_clinical_gbm import OUT_ROOT, target_group

CONFOUND = OUT_ROOT / 'acquisition_confound'
ALPHA = 0.05


def corrected_t(diffs: np.ndarray, n_test: np.ndarray, n_total: int) -> dict:
    k = len(diffs)
    sd = diffs.std(ddof=1)
    ratio = np.mean(n_test / (n_total - n_test))
    se = np.sqrt((1 / k + ratio) * sd ** 2)
    t = diffs.mean() / se if se > 0 else np.inf * np.sign(diffs.mean())
    return {'k': k, 'gain': diffs.mean(), 'sd_diff': sd, 'se_naive': sd / np.sqrt(k), 'se_corrected': se,
            't': t, 'p': 2 * stats.t.sf(abs(t), df=k - 1)}


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    order = p.sort_values()
    m = len(order)
    q = (order * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
    return q.reindex(p.index)


def run() -> pd.DataFrame:
    folds = pd.read_csv(CONFOUND / 'results_folds.csv')
    rows = []
    for (target, scheme), g in folds.groupby(['target', 'scheme']):
        n_total = int(g['n_test'].sum())
        # A multiclass ROC-AUC is undefined (NaN) on a grouped fold missing a class in its test split.
        v = g[g['valid']].dropna(subset=['embedding_roc_auc', 'metadata_roc_auc'])
        diffs = (v['embedding_roc_auc'] - v['metadata_roc_auc']).to_numpy()
        rows.append({'target': target, 'target_group': target_group(target), 'scheme': scheme,
                     'embedding_roc_auc': v['embedding_roc_auc'].mean(), 'embedding_sd': v['embedding_roc_auc'].std(ddof=1),
                     'metadata_roc_auc': v['metadata_roc_auc'].mean(),
                     **corrected_t(diffs, v['n_test'].to_numpy(float), n_total)})
    df = pd.DataFrame(rows)
    df['q'] = df.groupby(['target_group', 'scheme'], group_keys=False)['p'].apply(benjamini_hochberg)
    df['significant'] = df['q'] < ALPHA
    df['above_margin'] = df['gain'].abs() > df['embedding_sd']
    return df.sort_values(['target_group', 'scheme', 'p'])


def main():
    df = run()
    df.to_csv(CONFOUND / 'significance_tests.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_rows', 200):
        cols = ['target', 'k', 'gain', 'sd_diff', 'se_corrected', 't', 'p', 'q', 'significant', 'above_margin']
        for (grp, scheme), g in df.groupby(['target_group', 'scheme']):
            print(f'\n=== {grp} / {scheme} ===')
            print(g[cols].to_string(index=False, float_format='%.4f'))
        summary = df.groupby(['target_group', 'scheme']).agg(
            n=('target', 'size'), significant_pos=('significant', lambda s: int((s & (df.loc[s.index, 'gain'] > 0)).sum())),
            significant_neg=('significant', lambda s: int((s & (df.loc[s.index, 'gain'] < 0)).sum())),
            above_margin=('above_margin', lambda s: int((s & (df.loc[s.index, 'gain'] > 0)).sum())))
        print('\n', summary.to_string())


if __name__ == '__main__':
    main()
