"""Which strategies (embedding, reduction, model) win most often when the best
combination is chosen per variable, in the stratified-CV sweep
(gbm_pca_umap_sweep/results_long.csv)?

Three views, per target group and overall:
1. winners: how often each level is the argmax of ROC-AUC (88 combinations per variable);
2. near-best: the combinations within one fold-std of the winner's ROC-AUC are treated as
   tied with it; for each level, its share of those near-best slots divided by its share of
   the grid (enrichment > 1 = over-represented among the top, 1 = what chance would give);
3. per-variable vs. global: how much the per-variable winner beats the single global
   strategy (mae_cls, raw, CatBoost) on the same variable, against the winner's fold std.

Writes report/tables/tab_estrategias_vencedoras_*.csv.

Usage:
    python -m src.evaluation.analyze_best_per_variable
"""
import pandas as pd

from src.models.train_embedding_to_clinical_gbm import OUT_ROOT, target_group

SWEEP = OUT_ROOT / 'gbm_pca_umap_sweep'
TABLES = OUT_ROOT / 'report' / 'tables'
GROUPS = ['image', 'pathology', 'history', 'sentinel']
MODEL_PT = {'catboost': 'CatBoost', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'mlp': 'MLP'}
GLOBAL = ('mae_cls', 'raw', 'catboost')


def load() -> pd.DataFrame:
    df = pd.read_csv(SWEEP / 'results_long.csv')
    df = df[(df['kind'] != 'regression') & (df['model'] != 'baseline')].copy()
    df['group'] = df['target'].map(target_group)
    df['setting'] = df['reducer'] + df['n_components'].astype(int).astype(str).where(df['reducer'] != 'raw', '')
    df['model'] = df['model'].map(MODEL_PT)
    return df


FACTORS = {'embedding': 'embedding', 'método de redução': 'reducer', 'redução (com k)': 'setting', 'modelo': 'model'}


def winners(df: pd.DataFrame) -> pd.DataFrame:
    best = df.loc[df.groupby('target')['roc_auc_mean'].idxmax()]
    rows = []
    for label, col in FACTORS.items():
        for scope in ['todos (sem sentinela)'] + GROUPS:
            sub = best[best.group != 'sentinel'] if scope.startswith('todos') else best[best.group == scope]
            for level, n in sub[col].value_counts().items():
                rows.append({'fator': label, 'grupo': scope, 'nivel': level, 'vitorias': int(n), 'n_variaveis': len(sub)})
    return pd.DataFrame(rows)


def near_best(df: pd.DataFrame) -> pd.DataFrame:
    best = df.loc[df.groupby('target')['roc_auc_mean'].idxmax()].set_index('target')
    threshold = best['roc_auc_mean'] - best['roc_auc_std']
    tied = df[df['roc_auc_mean'] >= df['target'].map(threshold)]
    rows = []
    for label, col in FACTORS.items():
        for scope in ['todos (sem sentinela)'] + GROUPS:
            pick = (lambda d: d[d.group != 'sentinel']) if scope.startswith('todos') else (lambda d, s=scope: d[d.group == s])
            grid, top = pick(df), pick(tied)
            grid_share = grid[col].value_counts(normalize=True)
            top_share = top[col].value_counts(normalize=True).reindex(grid_share.index, fill_value=0)
            present = top.groupby(col)['target'].nunique().reindex(grid_share.index, fill_value=0)
            n_vars = grid['target'].nunique()
            for level in grid_share.index:
                rows.append({'fator': label, 'grupo': scope, 'nivel': level,
                             'participacao_na_grade': grid_share[level], 'participacao_entre_quase_melhores': top_share[level],
                             'enriquecimento': top_share[level] / grid_share[level],
                             'variaveis_com_o_nivel_entre_quase_melhores': int(present[level]), 'n_variaveis': n_vars})
    return pd.DataFrame(rows)


def per_variable_vs_global(df: pd.DataFrame) -> pd.DataFrame:
    best = df.loc[df.groupby('target')['roc_auc_mean'].idxmax()].set_index('target')
    emb, setting, model = GLOBAL
    glob = df[(df.embedding == emb) & (df.setting == setting) & (df.model == MODEL_PT[model])].set_index('target')
    out = pd.DataFrame({
        'grupo': best['group'], 'melhor_combinacao': best['embedding'] + ' · ' + best['setting'] + ' · ' + best['model'],
        'roc_auc_melhor': best['roc_auc_mean'], 'dp_melhor': best['roc_auc_std'],
        'roc_auc_global': glob['roc_auc_mean'].reindex(best.index),
    })
    out['vantagem_sobre_global'] = out['roc_auc_melhor'] - out['roc_auc_global']
    out['vantagem_supera_dp'] = out['vantagem_sobre_global'] > out['dp_melhor']
    out['n_quase_melhores'] = df[df['roc_auc_mean'] >= df['target'].map(best['roc_auc_mean'] - best['roc_auc_std'])] \
        .groupby('target').size().reindex(best.index)
    out['global_entre_quase_melhores'] = out['roc_auc_global'] >= out['roc_auc_melhor'] - out['dp_melhor']
    return out.reset_index().sort_values(['grupo', 'vantagem_sobre_global'], ascending=[True, False])


def main():
    df = load()
    TABLES.mkdir(parents=True, exist_ok=True)
    w, nb, pv = winners(df), near_best(df), per_variable_vs_global(df)
    w.to_csv(TABLES / 'tab_estrategias_vencedoras_contagem.csv', index=False)
    nb.to_csv(TABLES / 'tab_estrategias_vencedoras_quase_melhores.csv', index=False)
    pv.to_csv(TABLES / 'tab_estrategias_vencedoras_vs_global.csv', index=False)

    with pd.option_context('display.width', 220, 'display.max_rows', 300, 'display.max_columns', 20):
        for label in FACTORS:
            print(f'\n=== {label} ===')
            a = w[w.fator == label].pivot_table(index='nivel', columns='grupo', values='vitorias', fill_value=0)
            b = nb[(nb.fator == label) & (nb.grupo == 'todos (sem sentinela)')].set_index('nivel')
            print(a.join(b[['participacao_na_grade', 'participacao_entre_quase_melhores', 'enriquecimento',
                            'variaveis_com_o_nivel_entre_quase_melhores']]).round(2).to_string())
        print('\n=== melhor por variável vs. estratégia global ===')
        print(pv.round(3).to_string(index=False))
        print('\nvantagem média por grupo:', pv.groupby('grupo')[['vantagem_sobre_global']].mean().round(3).to_dict())
        print('vantagem supera dp:', int(pv.vantagem_supera_dp.sum()), 'de', len(pv),
              '| global entre as quase-melhores:', int(pv.global_entre_quase_melhores.sum()), 'de', len(pv),
              '| mediana de combinações quase-melhores:', pv.n_quase_melhores.median())


if __name__ == '__main__':
    main()
