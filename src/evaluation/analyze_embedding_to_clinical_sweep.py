"""Answer the sweep's comparison questions from
outputs/embedding_to_tabular_experiments/{folder}/results_long.csv
(produced by src/models/train_embedding_to_clinical_gbm.py):

- embeddings: USF-MAE vs. SD-VAE
- reducers: raw vs. PCA vs. UMAP at the same n_components
- models: CatBoost vs. XGBoost vs. LightGBM vs. MLP

Each comparison is paired by target (same folds everywhere) and restricted to one
target group ('image' by default). Where the other factors are averaged out, that
happens within each target first, so the Wilcoxon / Friedman tests see one
independent value per target.

Writes comparison_*.csv and paired_tests.csv into the same folder.

Usage:
    python -m src.evaluation.analyze_embedding_to_clinical_sweep --folder-name gbm_pca_umap_sweep
"""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from src.models.train_embedding_to_clinical_gbm import OUT_ROOT, target_group

METRICS = ['roc_auc_mean', 'pr_auc_mean', 'balanced_accuracy_mean', 'precision_mean', 'recall_mean',
           'accuracy_mean']
MODELS = ['catboost', 'xgboost', 'lightgbm', 'mlp']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--folder-name', required=True)
    p.add_argument('--group', default='image', help="Target group to analyse ('image', 'pathology', 'history').")
    return p.parse_args()


def setting(df: pd.DataFrame) -> pd.Series:
    return df['reducer'] + df['n_components'].astype(int).astype(str).where(df['reducer'] != 'raw', '')


def paired(wide: pd.DataFrame, a: str, b: str, comparison: str, metric: str) -> dict:
    pair = wide[[a, b]].dropna()
    diff = pair[a] - pair[b]
    p = wilcoxon(pair[a], pair[b]).pvalue if (diff != 0).any() else 1.0
    return {'comparison': comparison, 'metric': metric, 'a': a, 'b': b, 'n_targets': len(pair),
            'mean_a': pair[a].mean(), 'mean_b': pair[b].mean(), 'mean_diff': diff.mean(),
            'a_wins': int((diff > 0).sum()), 'b_wins': int((diff < 0).sum()), 'wilcoxon_p': p}


def main():
    args = parse_args()
    folder = OUT_ROOT / args.folder_name
    df = pd.read_csv(folder / 'results_long.csv')
    df['target_group'] = df['target'].map(target_group)
    df = df[(df['target_group'] == args.group) & (df['kind'] != 'regression') & (df['model'] != 'baseline')].copy()
    df['setting'] = setting(df)
    tests = []

    # Embeddings: mean over models, per target, for raw and for each embedding's best setting.
    per_target = df.groupby(['embedding', 'setting', 'target'])[METRICS].mean().reset_index()
    by_setting = per_target.groupby(['embedding', 'setting'])[METRICS].mean().reset_index()
    best_setting = by_setting.loc[by_setting.groupby('embedding')['roc_auc_mean'].idxmax()]
    emb_table = pd.concat([
        by_setting[by_setting['setting'] == 'raw'].assign(which='raw'),
        best_setting.assign(which='best setting (chosen on the mean over targets)'),
    ]).sort_values(['which', 'roc_auc_mean'], ascending=[False, False])
    emb_table.to_csv(folder / 'comparison_embeddings.csv', index=False)

    best_rows = per_target.merge(best_setting[['embedding', 'setting']], on=['embedding', 'setting'])
    for metric in ['roc_auc_mean', 'pr_auc_mean']:
        wide = best_rows.pivot(index='target', columns='embedding', values=metric)
        for a, b in [('mae_mean', 'vae_flat'), ('mae_cls', 'vae_flat'), ('mae_mean', 'mae_cls'),
                     ('vae_flat', 'vae_mean')]:
            tests.append(paired(wide, a, b, 'embedding (best setting each)', metric))

    # Reducers: per embedding, mean over models per target.
    red_table = (per_target[per_target['embedding'] != 'vae_mean']
                 .groupby(['embedding', 'setting'])[METRICS].mean().reset_index()
                 .sort_values(['embedding', 'roc_auc_mean'], ascending=[True, False]))
    red_table.to_csv(folder / 'comparison_reducers.csv', index=False)
    for embedding in ['mae_mean', 'mae_cls', 'vae_flat']:
        sub = per_target[per_target['embedding'] == embedding]
        for metric in ['roc_auc_mean', 'pr_auc_mean']:
            wide = sub.pivot(index='target', columns='setting', values=metric)
            for k in (10, 30, 50):
                tests.append(paired(wide, f'pca{k}', f'umap{k}', f'reducer ({embedding})', metric))
                tests.append(paired(wide, 'raw', f'pca{k}', f'reducer ({embedding})', metric))
                tests.append(paired(wide, 'raw', f'umap{k}', f'reducer ({embedding})', metric))

    # Models: per (embedding, setting), then pooled over settings per target.
    model_table = (df.groupby(['embedding', 'setting', 'model'])[METRICS].mean().reset_index()
                   .sort_values(['embedding', 'setting', 'roc_auc_mean'], ascending=[True, True, False]))
    model_table.to_csv(folder / 'comparison_models.csv', index=False)
    friedman = []
    for embedding in ['mae_mean', 'mae_cls', 'vae_flat']:
        sub = df[(df['embedding'] == embedding) & (df['setting'] == 'raw')]
        wide = sub.pivot(index='target', columns='model', values='roc_auc_mean')[MODELS].dropna()
        friedman.append({'embedding': embedding, 'setting': 'raw', 'n_targets': len(wide),
                         **{f'mean_{m}': wide[m].mean() for m in MODELS},
                         'friedman_p': friedmanchisquare(*[wide[m] for m in MODELS]).pvalue})
        for a, b in combinations(MODELS, 2):
            tests.append(paired(wide, a, b, f'model ({embedding}, raw)', 'roc_auc_mean'))
    pd.DataFrame(friedman).to_csv(folder / 'comparison_models_friedman.csv', index=False)

    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(folder / 'paired_tests.csv', index=False)

    with pd.option_context('display.width', 220, 'display.max_columns', 20, 'display.max_rows', 200):
        fmt = '%.3f'
        print(f"=== group '{args.group}': embeddings ===")
        print(emb_table[['which', 'embedding', 'setting'] + METRICS].to_string(index=False, float_format=fmt))
        print('\n=== reducers (mean over models) ===')
        print(red_table[['embedding', 'setting'] + METRICS[:3]].to_string(index=False, float_format=fmt))
        print('\n=== models (raw) ===')
        print(pd.DataFrame(friedman).to_string(index=False, float_format=fmt))
        print('\n=== paired tests ===')
        print(tests_df.to_string(index=False, float_format=fmt))


if __name__ == '__main__':
    main()
