"""Consolidated report of the embedding -> clinical-variable experiments.

Reads the saved results (no training) of
- outputs/embedding_to_tabular_experiments/gbm_pca_umap_sweep/   (full grid, stratified CV)
- outputs/embedding_to_tabular_experiments/acquisition_confound/ (best config, stratified
  and acquisition-grouped CV, embedding + acquisition, history <- annotations)
and writes report.md, figures/*.png and tables/*.csv into
outputs/embedding_to_tabular_experiments/report/.

Usage:
    python -m src.evaluation.build_report
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare

from src.evaluation import analyze_best_per_variable as winners_analysis
from src.evaluation import significance_tests
from src.evaluation.metadata_only_baseline import load_metadata
from src.models.train_embedding_to_clinical_gbm import OUT_ROOT, build_targets, target_group
from src.preprocessing.clinical_targets import (
    MULTILABEL_COLUMNS, SINGLELABEL_COLUMNS, load_raw_clinical, multilabel_targets,
)

SWEEP = OUT_ROOT / 'gbm_pca_umap_sweep'
CONFOUND = OUT_ROOT / 'acquisition_confound'
OUT = OUT_ROOT / 'report'
FIG, TAB = OUT / 'figures', OUT / 'tables'

GROUPS = ['image', 'pathology', 'history', 'sentinel']
GROUP_PT = {'image': 'Imagem', 'pathology': 'Patologia', 'history': 'Histórico', 'sentinel': 'Sentinela'}
EMBEDDINGS = ['mae_cls', 'mae_mean', 'vae_flat', 'vae_mean']
SETTINGS = ['raw', 'pca10', 'pca30', 'pca50', 'umap10', 'umap30', 'umap50']
MODELS = ['catboost', 'xgboost', 'lightgbm', 'mlp']
MODEL_PT = {'catboost': 'CatBoost', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'mlp': 'MLP'}
SCHEME_PT = {'stratified': 'CV estratificada', 'group_size': 'CV agrupada por tamanho',
             'group_pixel_size': 'CV agrupada por Pixel_size'}
BEST = ('mae_cls', 'raw', 'catboost')

# Reference palette (dataviz skill, references/palette.md): first three categorical
# slots, validated all-pairs; neutrals for baselines.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
CATEGORICAL = [BLUE, ORANGE, AQUA, '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
BASELINE_GRAY, DUMMY_GRAY = '#898781', '#c3c2b7'
SURFACE, INK, INK2, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
SEQUENTIAL = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
              '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']

DESCRIPTIONS = {
    'Age': 'Idade da paciente (anos)',
    'Tissue_composition': 'Composição do tecido mamário de fundo (gordura / fibroglandular, homogêneo / heterogêneo; lactante)',
    'Signs': 'Sinais no exame físico (palpável, retração de pele ou mamilo, cicatriz, vermelhidão...)',
    'Symptoms': 'Sintomas e histórico relatados (histórico familiar, terapia hormonal, descarga papilar...)',
    'Shape': 'Forma da lesão (léxico BI-RADS)',
    'Margin': 'Margem da lesão: circunscrita ou não circunscrita (angular, espiculada, indistinta, microlobulada)',
    'Echogenicity': 'Ecogenicidade da lesão',
    'Posterior_features': 'Características acústicas posteriores (sombra, reforço, combinada)',
    'Halo': 'Presença de halo ecogênico',
    'Calcifications': 'Calcificações (na massa, intraductais, indefinidas)',
    'Skin_thickening': 'Espessamento de pele',
    'Interpretation': 'Interpretação do radiologista (cisto, fibroadenoma, suspeita de malignidade...)',
    'BIRADS': 'Categoria BI-RADS (1–5, com 4a/4b/4c)',
    'Verification': 'Como o diagnóstico foi confirmado (biópsia ou acompanhamento)',
    'Diagnosis': 'Diagnóstico final (histopatológico ou por acompanhamento)',
    'Classification': 'Benigno / maligno / normal',
    'Pixel_size': 'Tamanho do pixel (mm) — parâmetro de aquisição',
}
COLUMN_GROUP = {'Age': 'history', 'Signs': 'history', 'Symptoms': 'history', 'Verification': 'pathology',
                'Diagnosis': 'pathology', 'Classification': 'pathology', 'Pixel_size': '—'}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2,
    'axes.edgecolor': AXIS, 'axes.linewidth': 1, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlecolor': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 1, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'lines.linewidth': 2, 'lines.solid_capstyle': 'round', 'savefig.dpi': 150,
})


def short(name: str) -> str:
    for old, new in [('of no special type (NST)', 'NST'), ('Ductal carcinoma in situ (DCIS)', 'DCIS'),
                     ('/hormonal contraception', ''), (' of breast/ovarian cancer', ''),
                     ('Complex cyst / Non-simple cyst', 'Complex cyst'), ('heterogeneous: predominantly', 'het. pred.'),
                     ('homogeneous:', 'homog.'), ('Interpretation_', 'Interp: '), ('Diagnosis_', 'Diag: '),
                     ('Margin_', 'Margin: '), ('Tissue_', 'Tissue: '), ('Signs_', 'Signs: '),
                     ('Symptoms_', 'Symptoms: ')]:
        name = name.replace(old, new)
    return name


def fmt(x, d=3):
    return '—' if pd.isna(x) else f'{x:.{d}f}'


COLUMN_PT = {
    'fator': 'Fator', 'nivel': 'Nível', 'vitorias': 'Vitórias (34 variáveis)', 'vitorias_imagem': 'Vitórias (23 de imagem)',
    'participacao_na_grade': 'Participação na grade', 'participacao_entre_quase_melhores': 'Participação entre quase-melhores',
    'enriquecimento': 'Enriquecimento',
    'target': 'Alvo', 'grupo': 'Grupo', 'target_group': 'Grupo', 'scheme': 'CV', 'roc_auc': 'ROC-AUC',
    'roc_auc_dp': 'dp', 'aquisicao_roc_auc': 'Aquisição', 'ganho': 'Ganho', 'agrup_roc_auc': 'ROC-AUC agrup.',
    'agrup_roc_auc_dp': 'dp agrup.', 'agrup_aquisicao_roc_auc': 'Aquisição agrup.', 'agrup_ganho': 'Ganho agrup.',
    'pr_auc': 'PR-AUC', 'dummy_pr_auc': 'PR-AUC Dummy', 'precision': 'Precision', 'recall': 'Recall',
    'accuracy': 'Accuracy', 'embedding': 'Embedding', 'reducao': 'Redução', 'modelo': 'Modelo',
    'metadata_roc_auc_mean': 'Só aquisição', 'embedding_roc_auc_mean': 'Só embedding',
    'embedding+metadata_roc_auc_mean': 'Embedding + aquisição', 'gain_vs_metadata_roc_auc': 'Ganho (emb+aq − aq)',
    'n_targets': 'nº alvos', 'anotacoes_roc_auc': 'Só anotações', 'anotacoes_dp': 'dp',
    'embedding_estrat': 'Embedding estrat.', 'embedding_estrat_dp': 'dp', 'embedding_agrup': 'Embedding agrup.',
    'embedding_agrup_dp': 'dp', 'comparison': 'Comparação', 'a': 'A', 'b': 'B', 'n_targets_': 'n', 'mean_a': 'Média A',
    'mean_b': 'Média B', 'a_wins': 'A vence', 'b_wins': 'B vence', 'wilcoxon_p': 'p (Wilcoxon)',
}
VALUE_PT = {**GROUP_PT, **SCHEME_PT}


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(COLUMN_PT.get(c, c) for c in cols) + ' |', '|' + '---|' * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c, v in zip(cols, row):
            if c == 'wilcoxon_p':
                cells.append(p_label(v).replace('p ', '').replace('= ', ''))
            elif isinstance(v, float):
                cells.append(fmt(v))
            else:
                cells.append(VALUE_PT.get(v, str(v)) if c in ('grupo', 'target_group', 'scheme', 'Grupo') else str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def save(fig, name: str) -> str:
    fig.savefig(FIG / name, bbox_inches='tight')
    plt.close(fig)
    return f'figures/{name}'


def legend_marker(color: str, label: str, marker: str = 'o'):
    return Line2D([], [], color=color, marker=marker, linestyle='none', markersize=8 if marker == '|' else 6.5,
                  markeredgewidth=2.5 if marker == '|' else 1.2, markeredgecolor=color if marker == '|' else SURFACE,
                  label=label)


def finish(fig, title: str, legend_ax=None, handles=None, ncol: int = 3):
    """Title on top, legend centered at the bottom, space reserved in inches so it
    works for figures of any height."""
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.12 / h, ha='left', va='top', fontweight='bold')
    if handles is None and legend_ax is not None:
        handles = legend_ax.get_legend_handles_labels()[0]
    bottom = 0.45 / h if handles else 0
    fig.tight_layout(rect=[0, bottom, 1, 1 - 0.45 / h])
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, 0.05 / h), fontsize=8.5)


def p_label(p: float) -> str:
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TAB / name, index=False)
    return df


# ---------------------------------------------------------------- data

def load():
    long = pd.read_csv(SWEEP / 'results_long.csv')
    long['target_group'] = long['target'].map(target_group)
    long['setting'] = long['reducer'] + long['n_components'].astype(int).astype(str).where(long['reducer'] != 'raw', '')
    conf = pd.read_csv(CONFOUND / 'results.csv')
    conf['target_group'] = conf['target'].map(target_group)
    return {
        'long': long,
        'cls': long[(long['kind'] != 'regression') & (long['model'] != 'baseline')],
        'meta': pd.read_csv(SWEEP / 'metadata_only_baseline.csv'),
        'tests': pd.read_csv(SWEEP / 'paired_tests.csv'),
        'conf': conf,
        'conf_summary': pd.read_csv(CONFOUND / 'summary_by_group.csv'),
        'history': pd.read_csv(CONFOUND / 'history_from_annotations.csv'),
        'sig': significance_tests.run(),
        'raw_df': load_raw_clinical(),
    }


# ---------------------------------------------------------------- section 2: data

def dictionary_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    index = pd.Index(raw_df['Image_filename'])
    targets = {name: (kind, y) for name, kind, y in build_targets(raw_df, index, 5)}
    kept_multi = set(multilabel_targets(raw_df).columns)
    all_multi = multilabel_targets(raw_df, min_positive_rate=0, max_positive_rate=1)
    prefix = {'Tissue_composition': 'Tissue'}

    rows = []
    for col in DESCRIPTIONS:
        group = COLUMN_GROUP.get(col, 'image')
        if col == 'Age':
            kind, y = targets['Age']
            rows.append([col, DESCRIPTIONS[col], group, 'numérica', 'regressão', '—', '—', len(y)])
        elif col in SINGLELABEL_COLUMNS:
            values = raw_df[col].astype(str)
            counts = values[~values.isin(['not available', 'not applicable'])].value_counts()
            kind, y = targets[col]
            used = counts[counts >= 5]
            dropped = ', '.join(f'{k} ({v})' for k, v in counts[counts < 5].items()) or '—'
            model = '1 modelo multiclasse' if kind == 'multiclass' else '1 modelo binário'
            rows.append([col, DESCRIPTIONS[col], group, 'rótulo único', model,
                         ', '.join(f'{k} ({v})' for k, v in used.items()), dropped, len(y)])
        elif col in MULTILABEL_COLUMNS:
            p = prefix.get(col, col) + '_'
            labels = all_multi[[c for c in all_multi.columns if c.startswith(p)]].sum()
            used = [f'{c[len(p):]} ({int(v)})' for c, v in labels.items() if c in kept_multi]
            dropped = [f'{c[len(p):]} ({int(v)})' for c, v in labels.items() if c not in kept_multi]
            n_multi = int(raw_df[col].astype(str).str.contains('&').sum())
            rows.append([col, DESCRIPTIONS[col], f'{group} (+ sentinela)' if any('not ' in u for u in used) else group,
                         f'multirrótulo `&` ({n_multi} casos com > 1 rótulo)', '1 binário por rótulo',
                         ', '.join(used), ', '.join(dropped) or '—', len(raw_df)])
        elif col == 'Classification':
            counts = raw_df[col].value_counts()
            rows.append([col, DESCRIPTIONS[col], group, 'rótulo único', '**não usada como alvo**',
                         ', '.join(f'{k} ({v})' for k, v in counts.items()), '—', len(raw_df)])
        else:
            rows.append([col, DESCRIPTIONS[col], group, 'numérica', 'só no baseline de aquisição', '—', '—',
                         len(raw_df)])
    return pd.DataFrame(rows, columns=['Variável', 'Descrição', 'Grupo', 'Tipo da coluna', 'Modelagem',
                                       'Classes / rótulos usados (n)', 'Descartados (n)', 'n válido'])


def targets_table(data) -> pd.DataFrame:
    raw_df = data['raw_df']
    rows = []
    for name, kind, y in build_targets(raw_df, pd.Index(raw_df['Image_filename']), 5):
        if kind == 'binary':
            dist = f'{y.mean():.1%} positivos ({int(y.sum())})'
        elif kind == 'multiclass':
            dist = f'{y.nunique()} classes'
        else:
            dist = f'média {y.mean():.1f}, dp {y.std():.1f}'
        rows.append([name, target_group(name), kind, len(y), dist])
    df = pd.DataFrame(rows, columns=['Alvo', 'Grupo', 'Tipo', 'n', 'Distribuição'])
    df['_order'] = df['Grupo'].map(GROUPS.index)
    return df.sort_values(['_order', 'Alvo']).drop(columns='_order')


def fig_prevalence(data) -> str:
    raw_df = data['raw_df']
    targets = build_targets(raw_df, pd.Index(raw_df['Image_filename']), 5)
    binary = pd.DataFrame([(n, target_group(n), y.mean()) for n, k, y in targets if k == 'binary'],
                          columns=['target', 'group', 'prev'])
    multi = [(n, y) for n, k, y in targets if k == 'multiclass']

    heights = [len(binary[binary.group == g]) for g in GROUPS] + [len(multi) * 1.4]
    fig, axes = plt.subplots(len(heights), 1, figsize=(8, 0.26 * sum(heights) + 2.5),
                             gridspec_kw={'height_ratios': heights})
    for ax, g in zip(axes, GROUPS):
        sub = binary[binary.group == g].sort_values('prev')
        ax.barh([short(t) for t in sub.target], sub.prev, height=0.6, color=BLUE)
        for yv, p in enumerate(sub.prev):
            ax.text(p + 0.01, yv, f'{p:.0%}', va='center', fontsize=8, color=INK2)
        ax.set_xlim(0, 1)
        ax.set_title(f'{GROUP_PT[g]} — alvos binários: prevalência da classe positiva', loc='left')
        ax.grid(axis='y', visible=False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))

    ax = axes[-1]
    raw_values = {c: raw_df[c].astype(str) for c in ['BIRADS', 'Shape', 'Echogenicity', 'Posterior_features']}
    for i, (name, _) in enumerate(multi):
        counts = raw_values[name][~raw_values[name].isin(['not available', 'not applicable'])].value_counts()
        counts = counts[counts >= 5].sort_index()
        left = 0
        for j, (cls, v) in enumerate(counts.items()):
            share = v / counts.sum()
            ax.barh(i, share - 0.004, left=left, height=0.6, color=CATEGORICAL[j % len(CATEGORICAL)])
            if share > 0.07:
                ax.text(left + share / 2, i, f'{cls}\n{share:.0%}', ha='center', va='center', fontsize=7,
                        color='white' if j in (0, 1, 5, 6, 7) else INK)
            left += share
    ax.set_yticks(range(len(multi)), [n for n, _ in multi])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title('Imagem — alvos multiclasse: proporção de cada classe (rótulos só nas fatias > 7%)', loc='left')
    ax.grid(visible=False)
    fig.tight_layout()
    return save(fig, 'fig01_distribuicao_alvos.png')


# ---------------------------------------------------------------- section 7: comparisons

def setting_matrix(cls: pd.DataFrame, group: str) -> pd.DataFrame:
    sub = cls[cls.target_group == group]
    per_target = sub.groupby(['embedding', 'setting', 'target'])['roc_auc_mean'].mean().reset_index()
    return per_target.groupby(['embedding', 'setting'])['roc_auc_mean'].mean().unstack().reindex(
        index=EMBEDDINGS, columns=SETTINGS)


def fig_heatmap(data) -> tuple:
    cmap = LinearSegmentedColormap.from_list('seq', SEQUENTIAL)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    tables = []
    for ax, g in zip(axes.flat, GROUPS):
        m = setting_matrix(data['cls'], g)
        tables.append(m.assign(grupo=g).reset_index())
        im = ax.imshow(m.to_numpy(float), cmap=cmap, vmin=0.5, vmax=0.85, aspect='auto')
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iat[i, j]
                if pd.isna(v):
                    ax.text(j, i, 'n/a', ha='center', va='center', fontsize=8, color=INK2)
                else:
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8,
                            color='white' if v > 0.7 else INK)
        ax.set_xticks(range(len(SETTINGS)), SETTINGS, rotation=35, ha='right')
        ax.set_yticks(range(len(EMBEDDINGS)), EMBEDDINGS)
        ax.grid(False)
        n = data['cls'][data['cls'].target_group == g].target.nunique()
        ax.set_title(f'{GROUP_PT[g]} ({n} alvos)', loc='left')
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.88, bottom=0.1, wspace=0.28, hspace=0.45)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='ROC-AUC médio (média dos 4 modelos)')
    fig.suptitle('ROC-AUC médio por embedding × redução de dimensionalidade — CV estratificada', x=0.01,
                 y=0.97, ha='left', fontweight='bold')
    table = pd.concat(tables)
    save_table(table, 'tab_embedding_x_reducao.csv')
    return save(fig, 'fig02_heatmap_embedding_reducao.png'), table


def fig_pca_umap(data) -> str:
    fig, axes = plt.subplots(len(GROUPS), 3, figsize=(10, 9.5), sharex=True, sharey='row')
    for r, g in enumerate(GROUPS):
        m = setting_matrix(data['cls'], g)
        for c, emb in enumerate(['mae_cls', 'mae_mean', 'vae_flat']):
            ax = axes[r, c]
            for method, color, label in [('pca', BLUE, 'PCA'), ('umap', ORANGE, 'UMAP')]:
                ys = [m.loc[emb, f'{method}{k}'] for k in (10, 30, 50)]
                ax.plot([10, 30, 50], ys, color=color, marker='o', markersize=6, markeredgecolor=SURFACE,
                        markeredgewidth=2, label=label)
            ax.axhline(m.loc[emb, 'raw'], color=BASELINE_GRAY, linewidth=1.5, label='sem redução (raw)')
            ax.set_xticks([10, 30, 50])
            if r == 0:
                ax.set_title(emb)
            if c == 0:
                ax.set_ylabel(f'{GROUP_PT[g]}\nROC-AUC médio')
            if r == len(GROUPS) - 1:
                ax.set_xlabel('nº de componentes (k)')
    finish(fig, 'PCA vs. UMAP com o mesmo nº de componentes (média dos 4 modelos; eixo y comum por linha) '
                '— CV estratificada', legend_ax=axes[0, 0])
    return save(fig, 'fig03_pca_vs_umap.png')


def fig_models(data) -> tuple:
    cls = data['cls'][data['cls'].setting == 'raw']
    fig, axes = plt.subplots(len(GROUPS), 3, figsize=(10, 8.5), sharex=True)
    rows = []
    for r, g in enumerate(GROUPS):
        for c, emb in enumerate(['mae_cls', 'mae_mean', 'vae_flat']):
            ax = axes[r, c]
            wide = cls[(cls.target_group == g) & (cls.embedding == emb)].pivot(
                index='target', columns='model', values='roc_auc_mean')[MODELS].dropna()
            means = wide.mean()
            se = wide.std(ddof=1) / np.sqrt(len(wide))
            p = friedmanchisquare(*[wide[mdl] for mdl in MODELS]).pvalue if len(wide) >= 3 else np.nan
            ypos = np.arange(len(MODELS))[::-1]
            ax.errorbar(means, ypos, xerr=se, fmt='o', color=BLUE, markersize=7, markeredgecolor=SURFACE,
                        markeredgewidth=2, ecolor=BLUE, elinewidth=1.5, capsize=0)
            ax.set_yticks(ypos, [MODEL_PT[mdl] for mdl in MODELS])
            ax.set_title(f'{emb}  ·  Friedman {p_label(p)}' if r == 0 else f'Friedman {p_label(p)}',
                         fontsize=9, loc='left')
            if c == 0:
                ax.set_ylabel(f'{GROUP_PT[g]} ({len(wide)})')
            if r == len(GROUPS) - 1:
                ax.set_xlabel('ROC-AUC médio ± erro-padrão entre alvos')
            rows += [{'grupo': g, 'embedding': emb, 'modelo': MODEL_PT[mdl], 'roc_auc_medio': means[mdl],
                      'erro_padrao': se[mdl], 'n_alvos': len(wide), 'friedman_p': p} for mdl in MODELS]
    finish(fig, 'Modelos de inferência na embedding bruta — CV estratificada')
    table = save_table(pd.DataFrame(rows), 'tab_modelos.csv')
    return save(fig, 'fig04_modelos.png'), table


def dot_rows(ax, labels, series, title=None, dodge=False):
    """series: list of (values, errors_or_None, color, label, marker). With dodge, each
    series gets its own lane inside the row, so error bars never overlap."""
    y = np.arange(len(labels))
    offsets = np.linspace(-0.22, 0.22, len(series)) if dodge else np.zeros(len(series))
    if not dodge:
        stacked = np.vstack([s[0] for s in series])
        ax.hlines(y, np.nanmin(stacked, axis=0), np.nanmax(stacked, axis=0), color=GRID, linewidth=3, zorder=1)
    for off, (values, errors, color, label, marker) in zip(offsets, series):
        if errors is not None:
            ax.errorbar(values, y + off, xerr=errors, fmt='none', ecolor=color, elinewidth=1.5,
                        alpha=1 if dodge else 0.55, zorder=2)
        if marker == '|':
            ax.scatter(values, y + off, color=color, s=110, marker=marker, linewidths=2.5, label=label, zorder=3)
        else:
            ax.scatter(values, y + off, color=color, s=42, marker=marker, edgecolors=SURFACE, linewidths=1.5,
                       label=label, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.grid(axis='y', visible=False)
    if title:
        ax.set_title(title, loc='left')


def grouped_figure(groups_rows: dict, width: float = 9, ncols: int = 1):
    heights = [max(len(v), 1) for v in groups_rows.values()]
    fig, axes = plt.subplots(len(heights), ncols, figsize=(width, 0.27 * sum(heights) + 1.2 * len(heights)),
                             gridspec_kw={'height_ratios': heights}, squeeze=False, sharex=True)
    return fig, axes


def fig_mae_vs_vae(data) -> str:
    cls = data['cls']
    mae = cls[(cls.embedding == 'mae_cls') & (cls.setting == 'raw') & (cls.model == 'catboost')].set_index('target')
    vae = cls[(cls.embedding == 'vae_flat') & (cls.setting == 'pca10') & (cls.model == 'catboost')].set_index('target')
    rows = {g: mae[mae.target_group == g].sort_values('roc_auc_mean').index.tolist() for g in GROUPS}
    fig, axes = grouped_figure(rows)
    for ax, (g, targets) in zip(axes[:, 0], rows.items()):
        dot_rows(ax, [short(t) for t in targets], [
            (vae.loc[targets, 'roc_auc_mean'].to_numpy(), None, ORANGE, 'SD-VAE (vae_flat + PCA 10)', 'o'),
            (mae.loc[targets, 'roc_auc_mean'].to_numpy(), None, BLUE, 'USF-MAE (mae_cls, raw)', 'o'),
        ], title=GROUP_PT[g])
        ax.axvline(0.5, color=DUMMY_GRAY, linewidth=1)
    axes[-1, 0].set_xlabel('ROC-AUC (CatBoost, CV estratificada)')
    finish(fig, 'USF-MAE vs. SD-VAE por variável (cada embedding na sua melhor redução)', legend_ax=axes[0, 0],
           ncol=2)
    return save(fig, 'fig05_mae_vs_vae.png')


# ---------------------------------------------------------------- section 8: best per variable

def best_per_variable(data) -> pd.DataFrame:
    cls = data['cls']
    best = cls.loc[cls.groupby('target')['roc_auc_mean'].idxmax()].set_index('target')
    meta = data['meta'].set_index('target')
    df = pd.DataFrame({
        'grupo': best['target_group'],
        'embedding': best['embedding'], 'reducao': best['setting'], 'modelo': best['model'].map(MODEL_PT),
        'roc_auc': best['roc_auc_mean'], 'roc_auc_dp': best['roc_auc_std'], 'pr_auc': best['pr_auc_mean'],
        'aquisicao_roc_auc': meta['metadata_roc_auc'], 'aquisicao_pr_auc': meta['metadata_pr_auc'],
        'dummy_pr_auc': meta['dummy_pr_auc'],
    })
    df['ganho'] = df['roc_auc'] - df['aquisicao_roc_auc']
    df['ganho_supera_margem'] = df['ganho'].abs() > df['roc_auc_dp']
    df.loc['Calcifications', ['pr_auc', 'dummy_pr_auc', 'aquisicao_pr_auc']] = np.nan
    return df.reset_index().rename(columns={'index': 'target'})


def fig_best_per_variable(best: pd.DataFrame) -> str:
    rows = {g: best[best.grupo == g].sort_values('ganho').target.tolist() for g in GROUPS}
    fig, axes = grouped_figure(rows, width=10)
    b = best.set_index('target')
    for ax, (g, targets) in zip(axes[:, 0], rows.items()):
        sub = b.loc[targets]
        dot_rows(ax, [short(t) for t in targets], [
            (sub.aquisicao_roc_auc.to_numpy(), None, BASELINE_GRAY, 'Baseline aquisição (largura, altura, Pixel_size)', 'o'),
            (sub.roc_auc.to_numpy(), sub.roc_auc_dp.to_numpy(), BLUE, 'Melhor combinação da grade (± dp entre folds)', 'o'),
        ], title=GROUP_PT[g])
        for yv, (_, r) in enumerate(sub.iterrows()):
            ax.text(1.005, yv, f'{r.embedding} · {r.reducao} · {r.modelo}   ({r.ganho:+.2f})',
                    transform=ax.get_yaxis_transform(), va='center', fontsize=7.5, color=INK2)
        ax.axvline(0.5, color=DUMMY_GRAY, linewidth=1)
        ax.set_xlim(0.35, 1.0)
    axes[-1, 0].set_xlabel('ROC-AUC (CV estratificada)')
    finish(fig, 'Melhor combinação por variável vs. baseline de aquisição — máximo sobre 88 combinações (otimista)',
           legend_ax=axes[0, 0], ncol=2)
    return save(fig, 'fig06_melhor_por_variavel.png')


def strategy_frequency() -> tuple:
    """Per factor level: wins as per-variable best, and enrichment among the combinations
    tied with the best (within its fold std). Sentinel targets excluded."""
    grid = winners_analysis.load()
    wins = winners_analysis.winners(grid)
    near = winners_analysis.near_best(grid)
    vs_global = winners_analysis.per_variable_vs_global(grid)
    scope = 'todos (sem sentinela)'
    w_all = wins[wins.grupo == scope].set_index(['fator', 'nivel'])['vitorias']
    w_img = wins[wins.grupo == 'image'].set_index(['fator', 'nivel'])['vitorias']
    nb = near[near.grupo == scope].set_index(['fator', 'nivel'])
    table = pd.DataFrame({
        'vitorias': w_all.reindex(nb.index, fill_value=0).astype(int),
        'vitorias_imagem': w_img.reindex(nb.index, fill_value=0).astype(int),
        'participacao_na_grade': nb['participacao_na_grade'],
        'participacao_entre_quase_melhores': nb['participacao_entre_quase_melhores'],
        'enriquecimento': nb['enriquecimento'],
    }).reset_index()
    table = table[table.fator != 'redução (com k)'].sort_values(['fator', 'vitorias'], ascending=[True, False])
    save_table(table, 'tab_estrategias_vencedoras.csv')
    return table, vs_global


# ---------------------------------------------------------------- section 9: global best

def global_best_table(data) -> pd.DataFrame:
    cls = data['cls']
    emb, setting, model = BEST
    g = cls[(cls.embedding == emb) & (cls.setting == setting) & (cls.model == model)].set_index('target')
    conf = data['conf'].set_index(['target', 'scheme'])
    sig = data['sig'].set_index(['target', 'scheme'])
    rows = []
    for t in g.index:
        s, p = conf.loc[(t, 'stratified')], conf.loc[(t, 'group_pixel_size')]
        ts, tp = sig.loc[(t, 'stratified')], sig.loc[(t, 'group_pixel_size')]
        rows.append({
            'target': t, 'grupo': g.at[t, 'target_group'], 'tipo': g.at[t, 'kind'],
            'roc_auc': s['embedding_roc_auc_mean'], 'roc_auc_dp': s['embedding_roc_auc_std'],
            'pr_auc': s['embedding_pr_auc_mean'], 'precision': g.at[t, 'precision_mean'],
            'recall': g.at[t, 'recall_mean'], 'accuracy': g.at[t, 'accuracy_mean'],
            'balanced_accuracy': s['embedding_balanced_accuracy_mean'],
            'dummy_pr_auc': s['dummy_pr_auc_mean'],
            'aquisicao_roc_auc': s['metadata_roc_auc_mean'], 'aquisicao_pr_auc': s['metadata_pr_auc_mean'],
            'ganho': s['embedding_roc_auc_mean'] - s['metadata_roc_auc_mean'],
            'agrup_roc_auc': p['embedding_roc_auc_mean'], 'agrup_roc_auc_dp': p['embedding_roc_auc_std'],
            'agrup_pr_auc': p['embedding_pr_auc_mean'], 'agrup_dummy_pr_auc': p['dummy_pr_auc_mean'],
            'agrup_aquisicao_roc_auc': p['metadata_roc_auc_mean'], 'agrup_aquisicao_pr_auc': p['metadata_pr_auc_mean'],
            'agrup_ganho': p['embedding_roc_auc_mean'] - p['metadata_roc_auc_mean'],
            'agrup_folds_validos': int(p['n_valid_folds']),
            'p': ts['p'], 'q': ts['q'], 'agrup_p': tp['p'], 'agrup_q': tp['q'],
            'ep_corrigido': ts['se_corrected'], 'agrup_ep_corrigido': tp['se_corrected'],
        })
    df = pd.DataFrame(rows)
    df['ganho_supera_margem'] = df['ganho'].abs() > df['roc_auc_dp']
    df['agrup_ganho_supera_margem'] = df['agrup_ganho'].abs() > df['agrup_roc_auc_dp']
    df['significativo'] = df['q'] < significance_tests.ALPHA
    df['agrup_significativo'] = df['agrup_q'] < significance_tests.ALPHA
    # Precision/recall/accuracy come from the sweep, where Calcifications' positive class was inverted.
    df.loc[df.target == 'Calcifications', ['precision', 'recall', 'accuracy']] = np.nan
    df['_o'] = df.grupo.map(GROUPS.index)
    return df.sort_values(['_o', 'agrup_ganho'], ascending=[True, False]).drop(columns='_o')


def fig_global_best(gb: pd.DataFrame, metric: str) -> str:
    rows = {g: gb[gb.grupo == g].sort_values('agrup_ganho').target.tolist() for g in GROUPS}
    fig, axes = grouped_figure(rows, width=12, ncols=2)
    t = gb.set_index('target')
    for r, (g, targets) in enumerate(rows.items()):
        sub = t.loc[targets]
        for c, (prefix, scheme) in enumerate([('', 'stratified'), ('agrup_', 'group_pixel_size')]):
            ax = axes[r, c]
            if metric == 'roc_auc':
                series = [
                    (sub[f'{prefix}aquisicao_roc_auc'].to_numpy(), None, BASELINE_GRAY, 'Baseline aquisição', 'o'),
                    (sub[f'{prefix}roc_auc'].to_numpy(), sub[f'{prefix}roc_auc_dp'].to_numpy(),
                     BLUE if c == 0 else ORANGE, 'Embedding (± dp entre folds)', 'o'),
                ]
                ref = 0.5
            else:
                series = [
                    (sub[f'{prefix}dummy_pr_auc'].to_numpy(), None, '#a9a8a1', 'Dummy (prevalência)', '|'),
                    (sub[f'{prefix}aquisicao_pr_auc'].to_numpy(), None, BASELINE_GRAY, 'Baseline aquisição', 'o'),
                    (sub[f'{prefix}pr_auc'].to_numpy(), None, BLUE if c == 0 else ORANGE, 'Embedding', 'o'),
                ]
                ref = None
            sig = sub[f'{prefix}significativo'].to_numpy()
            dot_rows(ax, [short(x) + ('  *' if s else '') for x, s in zip(targets, sig)], series,
                     title=f'{GROUP_PT[g]} — {SCHEME_PT[scheme]}' if r == 0 else GROUP_PT[g])
            if ref is not None:
                ax.axvline(ref, color=DUMMY_GRAY, linewidth=1)
            ax.set_xlim((0.2, 1.0) if metric == 'roc_auc' else (0, 1.0))
            flag = sub[f'{prefix}ganho_supera_margem'].to_numpy()
            for lbl, f in zip(ax.get_yticklabels(), flag):
                if f and metric == 'roc_auc':
                    lbl.set_fontweight('bold')
                    lbl.set_color(INK)
            if c == 1:
                ax.tick_params(labelleft=True)
    name = 'ROC-AUC' if metric == 'roc_auc' else 'PR-AUC'
    for ax in axes[-1]:
        ax.set_xlabel(name)
        ax.tick_params(labelbottom=True)
    err = ' (± dp entre folds)' if metric == 'roc_auc' else ''
    handles = ([legend_marker('#a9a8a1', 'Dummy (prevalência)', '|')] if metric == 'pr_auc' else []) + [
        legend_marker(BASELINE_GRAY, 'Baseline aquisição (largura, altura, Pixel_size)'),
        legend_marker(BLUE, f'Embedding — CV estratificada{err}'),
        legend_marker(ORANGE, f'Embedding — CV agrupada por Pixel_size{err}'),
    ]
    note = '  ·  negrito = ganho acima da margem (dp)  ·  * = ganho significativo (q < 0.05)' if metric == 'roc_auc' else ''
    finish(fig, f'Melhor estratégia global (mae_cls · raw · CatBoost) por variável vs. baselines — {name}{note}',
           handles=handles, ncol=len(handles))
    return save(fig, f'fig0{7 if metric == "roc_auc" else 8}_melhor_global_{metric}.png')


# ---------------------------------------------------------------- section 10: confound

def fig_confound(data) -> tuple:
    s = data['conf_summary']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    sets = [('metadata', BASELINE_GRAY, 'Só aquisição'), ('embedding', BLUE, 'Só embedding'),
            ('embedding+metadata', AQUA, 'Embedding + aquisição')]
    schemes = ['stratified', 'group_size', 'group_pixel_size']
    for ax, g in zip(axes, ['image', 'pathology', 'history']):
        sub = s[s.target_group == g].set_index('scheme').loc[schemes]
        x = np.arange(len(schemes))
        w = 0.26
        for i, (key, color, label) in enumerate(sets):
            vals = sub[f'{key}_roc_auc_mean'].to_numpy()
            ax.bar(x + (i - 1) * w, vals - 0.5, bottom=0.5, width=w - 0.02, color=color, label=label)
            for xv, v in zip(x + (i - 1) * w, vals):
                ax.text(xv, v + 0.005, f'{v:.2f}', ha='center', fontsize=7.5, color=INK2)
        ax.set_xticks(x, ['estratificada', 'agrup. tamanho', 'agrup. Pixel_size'])
        ax.set_ylim(0.5, 0.88)
        ax.set_title(f'{GROUP_PT[g]} ({int(sub.n_targets.iloc[0])} alvos)', loc='left')
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('ROC-AUC médio (eixo começa em 0.5 = acaso)')
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Fator de confusão de aquisição: embedding vs. aquisição, com e sem separar os lotes '
                 '(mae_cls · raw · CatBoost)', x=0.02, ha='left', fontweight='bold')
    fig.tight_layout()
    table = s[s.target_group.isin(['image', 'pathology', 'history'])][
        ['target_group', 'scheme', 'metadata_roc_auc_mean', 'embedding_roc_auc_mean',
         'embedding+metadata_roc_auc_mean', 'gain_vs_metadata_roc_auc', 'n_targets']]
    table = table.assign(_g=table.target_group.map(GROUPS.index), _s=table.scheme.map(schemes.index)) \
        .sort_values(['_g', '_s']).drop(columns=['_g', '_s'])
    save_table(table, 'tab_fator_confusao.csv')
    return save(fig, 'fig09_fator_confusao.png'), table


def history_table(data) -> pd.DataFrame:
    h = data['history'].set_index('target')
    conf = data['conf'].set_index(['target', 'scheme'])
    df = pd.DataFrame({
        'anotacoes_roc_auc': h['annotations_roc_auc_mean'], 'anotacoes_dp': h['annotations_roc_auc_std'],
        'embedding_estrat': [conf.loc[(t, 'stratified'), 'embedding_roc_auc_mean'] for t in h.index],
        'embedding_estrat_dp': [conf.loc[(t, 'stratified'), 'embedding_roc_auc_std'] for t in h.index],
        'embedding_agrup': [conf.loc[(t, 'group_pixel_size'), 'embedding_roc_auc_mean'] for t in h.index],
        'embedding_agrup_dp': [conf.loc[(t, 'group_pixel_size'), 'embedding_roc_auc_std'] for t in h.index],
    }, index=h.index).sort_values('anotacoes_roc_auc')
    return save_table(df.reset_index(), 'tab_historico_anotacoes.csv')


def fig_history(ht: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    dot_rows(ax, [short(t) for t in ht.target], [
        (ht.anotacoes_roc_auc.to_numpy(), ht.anotacoes_dp.to_numpy(), AQUA, 'Só anotações de imagem do especialista', 'o'),
        (ht.embedding_estrat.to_numpy(), ht.embedding_estrat_dp.to_numpy(), BLUE, 'Embedding — CV estratificada', 'o'),
        (ht.embedding_agrup.to_numpy(), ht.embedding_agrup_dp.to_numpy(), ORANGE, 'Embedding — CV agrupada por Pixel_size', 'o'),
    ], dodge=True)
    ax.grid(axis='y', visible=False)
    for yv in np.arange(len(ht) - 1) + 0.5:
        ax.axhline(yv, color=GRID, linewidth=1)
    ax.axvline(0.5, color=DUMMY_GRAY, linewidth=1)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel('ROC-AUC (± dp entre folds)')
    finish(fig, 'Alvos de histórico: previstos pelas anotações de imagem tanto quanto pela embedding?',
           legend_ax=ax, ncol=3)
    return save(fig, 'fig10_historico_vs_anotacoes.png')


# ---------------------------------------------------------------- report text

def mean_of(gb, group, col):
    return gb[gb.grupo == group][col].mean()


def sig_text(gb, group, prefix) -> str:
    rows = gb[(gb.grupo == group) & gb[f'{prefix}significativo'] & (gb[f'{prefix}ganho'] > 0)]
    if rows.empty:
        return 'nenhuma variável é significativa'
    names = ', '.join(f'`{t}`' for t in rows.target)
    return f'{len(rows)} variável{"" if len(rows) == 1 else "s"} {"é" if len(rows) == 1 else "são"} significativa{"" if len(rows) == 1 else "s"} ({names})'


def p_text(gb, group, prefix) -> str:
    rows = gb[(gb.grupo == group) & (gb[f'{prefix}p'] < significance_tests.ALPHA) & (gb[f'{prefix}ganho'] > 0)]
    return f'{len(rows)} têm p < 0.05 ({", ".join(f"`{t}`" for t in rows.target)})'


def significance_section(data, gb) -> str:
    sig = data['sig']
    ex = sig[(sig.target == 'Posterior_features') & (sig.scheme == 'stratified')].iloc[0]
    folds = pd.read_csv(CONFOUND / 'results_folds.csv')
    exf = folds[(folds.target == 'Posterior_features') & (folds.scheme == 'stratified')]
    diffs = ', '.join(f'{d:.3f}' for d in exf.embedding_roc_auc - exf.metadata_roc_auc)

    counts = []
    for g in GROUPS:
        for scheme in ['stratified', 'group_size', 'group_pixel_size']:
            s = sig[(sig.target_group == g) & (sig.scheme == scheme)]
            pos = s.gain > 0
            counts.append({'grupo': g, 'scheme': scheme, 'n': len(s),
                           'acima da margem': int((s.above_margin & pos).sum()),
                           'p < 0.05': int(((s.p < 0.05) & pos).sum()),
                           'q < 0.05': int((s.significant & pos).sum())})
    counts = save_table(pd.DataFrame(counts), 'tab_testes_contagem.csv')

    img = gb[gb.grupo == 'image'].sort_values('agrup_p')
    tab = pd.DataFrame({
        'target': img.target, 'Ganho estrat.': img.ganho, 'EP corr. estrat.': img.ep_corrigido,
        'p estrat.': img.p, 'q estrat.': img.q, 'Ganho agrup.': img.agrup_ganho,
        'EP corr. agrup.': img.agrup_ep_corrigido, 'p agrup.': img.agrup_p, 'q agrup.': img.agrup_q,
    })
    return f"""### 9.1 Teste de significância do ganho sobre a aquisição

O critério "acima da margem" (ganho maior que 1 desvio-padrão da embedding entre folds) é uma
heurística. Para testar formalmente **H₀: ganho = 0**, usamos um teste t pareado com correção para
validação cruzada, aplicado só à estratégia global (escolhida pela média, antes de olhar cada
variável; testar a melhor-por-variável daria p-valores enviesados pela seleção entre 88 combinações):

1. **Estimador**: média das diferenças de ROC-AUC fold a fold (embedding − aquisição), já que os
   dois modelos foram avaliados nas mesmas partições.
2. **Erro-padrão corrigido (Nadeau & Bengio, 2003)**: os folds compartilham dados de treino e não são
   independentes, então a variância das diferenças é multiplicada por (1/k + n_teste/n_treino) em vez
   de 1/k.
3. **Distribuição de referência**: t de Student com k − 1 graus de liberdade (k = folds válidos, em
   geral 5), em vez da normal, porque o erro-padrão é estimado com poucos valores (valor crítico 2.78
   em vez de 1.96).
4. **Múltiplos testes**: q-valores de Benjamini–Hochberg (FDR) dentro de cada grupo e esquema de CV;
   "ganho significativo" = q < 0.05.

Exemplo — `Posterior_features`, CV estratificada: diferenças por fold {diffs}; média {ex.gain:.3f};
erro-padrão ingênuo {ex.se_naive:.3f}, corrigido {ex.se_corrected:.3f}; t = {ex.t:.2f} com {int(ex.k) - 1} g.l.;
p = {ex.p:.4f}; q = {ex.q:.3f}.

Contagem de variáveis com ganho positivo em cada critério (tabela: `tables/tab_testes_contagem.csv`):

{md_table(counts)}

O teste é bem mais exigente que a margem. Dois motivos: o erro-padrão corrigido é maior que o
ingênuo, e a correção de FDR sobre 23 variáveis eleva o limiar. Com 5 folds o teste tem pouco poder,
então "não significativo" aqui quer dizer "não detectável com este tamanho de amostra", não "sem
ganho". Os resultados mais consistentes são `Posterior_features` (significativo na estratificada,
p < 0.05 na agrupada) e `Skin_thickening` (p = {gb.set_index('target').at['Skin_thickening', 'agrup_p']:.4f} na agrupada, q = {gb.set_index('target').at['Skin_thickening', 'agrup_q']:.3f}).

Variáveis de imagem, ordenadas pelo p na CV agrupada por `Pixel_size` (todas as variáveis e
esquemas: `tables/tab_testes_significancia.csv`):

{md_table(tab)}
"""


def build_markdown(data, figs, tabs) -> str:
    raw_df, cls, tests = data['raw_df'], data['cls'], data['tests']
    meta = load_metadata(raw_df)
    gb, best = tabs['global'], tabs['best']
    heat = tabs['heat']
    img = heat[heat.grupo == 'image'].set_index('embedding')
    models = tabs['models']
    m_img = models[(models.grupo == 'image') & (models.embedding == 'mae_cls')].set_index('modelo')
    conf = tabs['confound'].set_index(['target_group', 'scheme'])
    robust = gb[(gb.grupo == 'image') & gb.agrup_ganho_supera_margem & (gb.agrup_ganho > 0)]
    lost = gb[(gb.grupo == 'image') & gb.ganho_supera_margem & ~gb.agrup_ganho_supera_margem]
    vsg = tabs['vs_global']
    emb_test = tests[(tests.comparison == 'embedding (best setting each)') & (tests.metric == 'roc_auc_mean')].set_index(['a', 'b'])
    wins = best[best.grupo == 'image']
    g_img = ('image', 'stratified'), ('image', 'group_pixel_size')
    md = f"""# Relatório consolidado — informação clínica nas embeddings de ultrassom

Gerado por `python -m src.evaluation.build_report` a partir dos resultados salvos em
`outputs/embedding_to_tabular_experiments/` (`gbm_pca_umap_sweep/` e `acquisition_confound/`).
Números, tabelas e figuras saem direto dos CSVs; nada foi retreinado. Registro cronológico dos
experimentos: `LOGS.md` (2026-09-26 e 2026-09-28).

## 0. Resumo

- **USF-MAE carrega muito mais informação clínica que o SD-VAE.** Nas {cls[cls.target_group == 'image'].target.nunique()} variáveis anotadas a partir da imagem, ROC-AUC médio de
  {img.at['mae_cls', 'raw']:.3f} (`mae_cls`, sem redução) vs. {img.loc['vae_flat', SETTINGS].max():.3f} (`vae_flat`, na sua melhor redução); o MAE vence em
  {int(emb_test.at[('mae_cls', 'vae_flat'), 'a_wins'])} de {int(emb_test.at[('mae_cls', 'vae_flat'), 'n_targets'])} variáveis. `mae_cls` ≈ `mae_mean`.
- **Reduzir a dimensionalidade piora o MAE** em todos os grupos; nas variáveis de imagem, com o mesmo
  nº de componentes, **PCA é melhor que UMAP** nas 9 comparações (3 embeddings × 3 valores de k) (Fig. 2–3).
  Nos grupos menores (patologia, histórico) a ordem PCA/UMAP varia.
- **Os três gradient boostings empatam** (CatBoost {m_img.at['CatBoost', 'roc_auc_medio']:.3f}, XGBoost {m_img.at['XGBoost', 'roc_auc_medio']:.3f},
  LightGBM {m_img.at['LightGBM', 'roc_auc_medio']:.3f}); o MLP fica atrás ({m_img.at['MLP', 'roc_auc_medio']:.3f}) (Fig. 4).
- **Melhor estratégia global: `mae_cls` · sem redução · CatBoost** — ROC-AUC médio {conf.at[g_img[0], 'embedding_roc_auc_mean']:.3f} nas variáveis
  de imagem (CV estratificada) e {conf.at[g_img[1], 'embedding_roc_auc_mean']:.3f} com os lotes de aquisição separados entre treino e teste.
- **Parte do desempenho é reconhecimento do aparelho/lote**: só largura, altura e `Pixel_size` da
  imagem já dão {conf.at[g_img[0], 'metadata_roc_auc_mean']:.3f}; separar os lotes custa {conf.at[g_img[0], 'embedding_roc_auc_mean'] - conf.at[g_img[1], 'embedding_roc_auc_mean']:.3f} à embedding.
- **{len(robust)} variáveis de imagem mantêm ganho sobre a aquisição acima da margem (1 dp entre folds) mesmo
  com os lotes separados**: {', '.join(f'`{x}`' for x in robust.target)} (Fig. 7).
- **No teste de hipótese formal (seção 9.1) o quadro é bem mais modesto**: com correção para múltiplos
  testes, {sig_text(gb, 'image', '')} na CV estratificada e {sig_text(gb, 'image', 'agrup_')} na CV agrupada por `Pixel_size`.
  Sem a correção, {p_text(gb, 'image', 'agrup_')} na CV agrupada. Com 5 folds o teste tem pouco poder: ausência de
  significância aqui não é evidência de ausência de ganho.
- **Tinham ganho acima da margem na CV estratificada e o perdem com os lotes separados**:
  {', '.join(f'`{x}`' for x in lost.target) or '—'}.
- Os sinais clínicos (`Signs_*`) não são controles negativos: as anotações de imagem do
  especialista já os preveem tão bem quanto a embedding (Fig. 10).

## 1. Pergunta e desenho

**Pergunta.** Quanta informação sobre as variáveis que o especialista anota olhando a imagem
(BI-RADS, forma, margem, ecogenicidade, diagnóstico...) as embeddings de imagens de ultrassom
carregam, e como isso varia entre modelos de embedding, estratégias de redução de dimensionalidade
e modelos de inferência?

**Medida.** Desempenho de classificadores treinados sobre a embedding para prever cada variável,
em validação cruzada. Informação que a embedding carrega → classificador acerta acima do acaso.

**Por que baselines.** Acima do acaso não basta: a embedding pode acertar reconhecendo o aparelho de
aquisição em vez da lesão. Por isso cada resultado é comparado com (i) um classificador trivial
(Dummy) e (ii) um classificador que só vê variáveis de aquisição da imagem (largura, altura,
`Pixel_size`). O ganho sobre (ii) é a parte atribuível ao conteúdo.

## 2. Dados e dicionário de variáveis

Dataset BrEaST: **{len(raw_df)} imagens de ultrassom mamário = {raw_df.CaseID.nunique()} pacientes** (uma imagem por paciente,
então validação cruzada por imagem é por paciente). As imagens vêm em **{meta.groupby(['width', 'height']).ngroups} tamanhos diferentes** e
**{meta.pixel_size.nunique()} valores de `Pixel_size`** ({meta.pixel_size.min():.4f}–{meta.pixel_size.max():.4f} mm), o que reflete aparelho e protocolo de exportação.

**Grupos de variáveis:**

| Grupo | O que é | Papel no relatório |
|---|---|---|
| Imagem (`image`) | anotado pelo especialista a partir da imagem | pergunta principal |
| Patologia (`pathology`) | diagnóstico confirmado e forma de confirmação | diagnóstico não é visto diretamente, mas se relaciona com a imagem |
| Histórico (`history`) | sinais, sintomas, idade | não está na imagem; usado para verificar o fator de confusão |
| Sentinela (`sentinel`) | rótulos "not available" / "not applicable" | medem preenchimento do laudo, não a lesão; só como referência |

**Dicionário das colunas clínicas** (tabela completa: `tables/tab_dicionario_variaveis.csv`):

{md_table(tabs['dictionary'])}

Regras de modelagem: colunas de **rótulo único** → um único modelo multiclasse com todas as classes
(ou binário, se sobram 2 classes; classe positiva = minoritária); classes com < 5 casos descartadas.
Colunas **multirrótulo** (valores separados por `&`, como no notebook de pré-processamento) → um
classificador binário independente por rótulo, mantendo rótulos com 5–95% de positivos.

**Alvos derivados** ({len(tabs['targets'])}; tabela: `tables/tab_alvos.csv`):

{md_table(tabs['targets'])}

![Distribuição dos alvos]({figs['prevalence']})

*Fig. 1 — Desbalanceamento dos alvos. Vários diagnósticos têm só 13–30 positivos, o que torna suas
métricas instáveis.*

## 3. Modelos de embedding

| Embedding | Modelo | Pré-processamento | Saída do modelo | Pooling | Dimensão |
|---|---|---|---|---|---|
| `mae_cls` | USF-MAE — ViT-B/16 MAE pré-treinado em ultrassom (`USF-MAE_full_pretrain_43dataset_100epochs.pt`) | redimensiona para 224×224, normalização ImageNet | 197 tokens × 768 (1 CLS + 196 patches) | token CLS | 768 |
| `mae_mean` | USF-MAE | idem | idem | média dos 196 patches | 768 |
| `vae_flat` | SD-VAE — `stabilityai/sd-vae-ft-mse` (autoencoder do Stable Diffusion) | redimensiona para 256×256 | latente 4 × 32 × 32 | latente achatado | 4096 |
| `vae_mean` | SD-VAE | idem | idem | média espacial por canal | 4 |

O MAE é um encoder semântico treinado para reconstruir patches mascarados; o SD-VAE é um
autoencoder de compressão, treinado para reconstrução fiel de baixo nível. Os dois redimensionam a
imagem **sem preservar a proporção** — por isso o formato original da imagem (aparelho) fica
impresso na embedding (seção 10). Os tokens do MAE vêm embaralhados por imagem; mean e CLS são
imunes a isso (ver `LOGS.md`, 2026-09-15).

## 4. Estratégias de redução de dimensionalidade

| Estratégia | Detalhe |
|---|---|
| `raw` | embedding inteira, sem redução |
| PCA k ∈ {{10, 30, 50}} | ajustado só no fold de treino |
| UMAP k ∈ {{10, 30, 50}} | `StandardScaler` + UMAP (`n_neighbors`=15), ajustado só no fold de treino, média de 3 seeds por fold |

Mesmo nº de componentes para PCA e UMAP, para comparação direta. `vae_mean` (4-d) só aparece sem
redução.

## 5. Modelos de inferência

| Modelo | Configuração | Desbalanceamento de classes |
|---|---|---|
| CatBoost | 150 iterações, profundidade 4, lr 0.08 | `auto_class_weights='Balanced'` |
| XGBoost | 150 árvores, profundidade 4, lr 0.08 | `sample_weight` balanceado |
| LightGBM | 150 árvores, profundidade 4, lr 0.08 | `class_weight='balanced'` |
| MLP | `StandardScaler` + 1 camada oculta de 64, `alpha`=1e-3, early stopping | — |
| Dummy (baseline) | prevê a classe majoritária | — |

## 6. Protocolo de avaliação

- **CV estratificada** 5 folds (`StratifiedKFold`, `random_state=42`) — as mesmas partições para
  todas as embeddings, reduções e modelos de uma variável (comparações pareadas por fold).
- **CV agrupada** (`StratifiedGroupKFold`, 5 folds) — nenhum lote de aquisição em treino e teste ao
  mesmo tempo; lote = tamanho da imagem (52 grupos) ou, mais rigoroso, `Pixel_size` (28 grupos).
  Rodada só para a melhor estratégia global.
- **Métricas**: ROC-AUC (principal; multiclasse = um-contra-o-resto, média macro), PR-AUC (average
  precision; o Dummy vale a prevalência), precision, recall, accuracy e balanced accuracy (limiar
  padrão 0.5, sem ajuste).
- **Margem de erro**: desvio-padrão da métrica da embedding entre os 5 folds; o baseline de
  aquisição é tratado como erro 0. Ganho "acima da margem" = |ganho| > desvio (heurística).
- **Teste de hipótese** (só para a estratégia global): t pareado por fold com erro-padrão corrigido
  para CV (Nadeau–Bengio), t de Student com k − 1 g.l., q-valores de Benjamini–Hochberg; "ganho
  significativo" = q < 0.05. Detalhes na seção 9.1.
- **Ressalvas**: 256 casos e 5 folds dão desvios de 0.04–0.26 por variável; a "melhor combinação por
  variável" é o máximo de 88 combinações e, portanto, otimista.

## 7. Resultados comparativos — estratégias e modelos

![Heatmap embedding x redução]({figs['heatmap']})

*Fig. 2 — ROC-AUC médio de cada embedding × redução, por grupo de variáveis (média dos 4 modelos).
Tabela: `tables/tab_embedding_x_reducao.csv`.*

![PCA vs UMAP]({figs['pca_umap']})

*Fig. 3 — PCA (azul) vs. UMAP (laranja) com o mesmo nº de componentes; linha cinza = embedding sem
redução; eixo y comum dentro de cada linha. No MAE a embedding inteira é sempre melhor; nas variáveis
de imagem o PCA supera o UMAP em todo k, e no VAE o PCA ajuda um pouco.*

![Modelos]({figs['models']})

*Fig. 4 — ROC-AUC médio de cada modelo na embedding bruta (± erro-padrão entre variáveis). Tabela:
`tables/tab_modelos.csv`.*

![MAE vs VAE]({figs['mae_vae']})

*Fig. 5 — USF-MAE vs. SD-VAE em cada variável (CatBoost; cada embedding na sua melhor redução).*

Testes pareados por variável (Wilcoxon, grupo imagem; tabela completa em
`gbm_pca_umap_sweep/paired_tests.csv`):

{md_table(tests[tests.metric == 'roc_auc_mean'][['comparison', 'a', 'b', 'n_targets', 'mean_a', 'mean_b', 'a_wins', 'b_wins', 'wilcoxon_p']]
          .loc[lambda d: d.comparison.str.startswith('embedding') | (d.comparison.str.startswith('reducer') & d.b.str.startswith('umap') & d.a.str.startswith('pca')) | d.comparison.str.startswith('model (mae_cls')])}

## 8. Melhor estratégia por variável vs. baseline de aquisição

![Melhor por variável]({figs['best']})

*Fig. 6 — Para cada variável, a melhor das 88 combinações (embedding · redução · modelo) vs. o
baseline de aquisição. Por ser o máximo de muitas tentativas, superestima o desempenho real,
sobretudo nas variáveis com poucos positivos. À direita, a combinação vencedora e o ganho.*

Vitórias por fator, no grupo imagem: embedding — {', '.join(f'{k} {v}' for k, v in wins.embedding.value_counts().items())};
redução — {', '.join(f'{k} {v}' for k, v in wins.reducao.value_counts().items())}; modelo — {', '.join(f'{k} {v}' for k, v in wins.modelo.value_counts().items())}.

{md_table(best[['target', 'grupo', 'embedding', 'reducao', 'modelo', 'roc_auc', 'roc_auc_dp', 'aquisicao_roc_auc', 'ganho', 'pr_auc', 'dummy_pr_auc']])}

### 8.1 Quais estratégias aparecem mais entre as melhores por variável

Contagem nas {int(vsg[vsg.grupo != 'sentinel'].shape[0])} variáveis não-sentinela. Como 5 folds deixam o vencedor mal definido, a tabela
também considera as **quase-melhores**: combinações com ROC-AUC dentro de um desvio-padrão (entre folds)
da melhor, ou seja, empatadas com ela dentro do ruído. Em cada variável há, na mediana,
**{vsg.n_quase_melhores.median():.0f} das 88 combinações** nessa faixa. **Enriquecimento** = participação do nível entre as
quase-melhores ÷ participação na grade (1 = o que o acaso daria; a grade não é equilibrada: `vae_mean`
tem 4 das 88 combinações, a redução `raw` 16 das 88).

{md_table(tabs['freq'])}

- **Embedding**: os dois poolings do USF-MAE ganham quase tudo e estão super-representados entre as
  quase-melhores; o SD-VAE aparece bem abaixo do acaso.
- **Redução**: sem redução (`raw`) é a mais frequente e a mais enriquecida; o UMAP vence uma única vez
  e fica abaixo do acaso; o PCA fica perto do acaso.
- **Modelo**: CatBoost lidera, XGBoost e LightGBM empatam logo atrás e o MLP fica muito abaixo do acaso.

**Escolher a estratégia por variável compensa?** A estratégia global (`mae_cls` · raw · CatBoost)
está entre as quase-melhores em **{int(vsg.global_entre_quase_melhores.sum())} de {len(vsg)}** variáveis, e a vantagem do vencedor
individual sobre ela passa do desvio-padrão em só **{int(vsg.vantagem_supera_dp.sum())}**:
{', '.join(f"`{t}` (+{v:.3f})" for t, v in vsg[vsg.vantagem_supera_dp][['target', 'vantagem_sobre_global']].itertuples(index=False))}.
Vantagem média: {vsg[vsg.grupo == 'image'].vantagem_sobre_global.mean():.3f} nas variáveis de imagem e {vsg[vsg.grupo == 'pathology'].vantagem_sobre_global.mean():.3f} em patologia — puxada por
diagnósticos com 13–30 positivos, onde o máximo de 88 tentativas é especialmente otimista. Tabelas:
`tables/tab_estrategias_vencedoras*.csv` (`python -m src.evaluation.analyze_best_per_variable`).

## 9. Melhor estratégia global por variável vs. baseline de aquisição

A estratégia global é a de maior ROC-AUC médio nas variáveis de imagem: **`mae_cls` · sem redução ·
CatBoost** — escolhida pela média, não variável a variável, então não tem o otimismo da seção 8.

![Melhor global ROC-AUC]({figs['global_roc']})

*Fig. 7 — Por variável: baseline de aquisição (cinza) vs. embedding (± dp entre folds), na CV
estratificada (esquerda) e na CV agrupada por `Pixel_size` (direita). Variáveis em negrito: ganho
sobre a aquisição maior que a margem. Ordenadas pelo ganho na CV agrupada.*

![Melhor global PR-AUC]({figs['global_pr']})

*Fig. 8 — O mesmo em PR-AUC, com a prevalência (Dummy) como referência.*

Média por grupo — melhor estratégia global:

| Grupo | ROC-AUC (estrat.) | Aquisição (estrat.) | ROC-AUC (agrup.) | Aquisição (agrup.) | PR-AUC (estrat.) | Dummy PR-AUC | Variáveis com ganho > margem (agrup.) |
|---|---|---|---|---|---|---|---|
""" + '\n'.join(
        f"| {GROUP_PT[g]} | {mean_of(gb, g, 'roc_auc'):.3f} | {mean_of(gb, g, 'aquisicao_roc_auc'):.3f} | "
        f"{mean_of(gb, g, 'agrup_roc_auc'):.3f} | {mean_of(gb, g, 'agrup_aquisicao_roc_auc'):.3f} | "
        f"{mean_of(gb, g, 'pr_auc'):.3f} | {mean_of(gb, g, 'dummy_pr_auc'):.3f} | "
        f"{int((gb[gb.grupo == g].agrup_ganho_supera_margem & (gb[gb.grupo == g].agrup_ganho > 0)).sum())} de {int((gb.grupo == g).sum())} |"
        for g in GROUPS) + f"""

Tabela consolidada por variável (tabela completa, com todas as colunas: `tables/tab_melhor_global.csv`):

{md_table(gb[['target', 'grupo', 'roc_auc', 'roc_auc_dp', 'aquisicao_roc_auc', 'ganho', 'agrup_roc_auc', 'agrup_roc_auc_dp', 'agrup_aquisicao_roc_auc', 'agrup_ganho', 'pr_auc', 'dummy_pr_auc', 'precision', 'recall', 'accuracy']])}

""" + significance_section(data, gb) + f"""
## 10. Fator de confusão de aquisição

![Fator de confusão]({figs['confound']})

*Fig. 9 — Média por grupo: só aquisição, só embedding e embedding + aquisição, nos três esquemas de
CV. Juntar a aquisição à embedding não acrescenta nada (a embedding já contém essa informação);
separar os lotes derruba mais a aquisição que a embedding.*

{md_table(tabs['confound'])}

![Histórico vs anotações]({figs['history']})

*Fig. 10 — Alvos de histórico previstos só pelas anotações de imagem do especialista vs. pela
embedding. Sinais clínicos (`Signs_*`) são tão previsíveis pelas anotações quanto pela embedding:
se relacionam com o que é visível (lesão palpável é maior e mais suspeita) e não servem de
controle negativo. `family history` é o único com excesso não explicado.*

{md_table(tabs['history'])}

## 11. Limitações e próximos passos

- **Tamanho amostral**: 256 casos, 5 folds; diagnósticos com 13–30 positivos têm desvio alto e
  recall 0 no limiar padrão (o modelo ordena, mas nunca prevê a classe).
- **Otimismo da seção 8**: máximo de 88 combinações por variável.
- **Poder estatístico baixo**: com 5 folds e correção para 23 testes, só ganhos grandes e estáveis
  entre folds chegam a q < 0.05. Repetir a CV com várias sementes (k-fold repetido) aumentaria o
  poder do teste sem precisar de mais dados.
- **CV agrupada só para a melhor estratégia**: as comparações de embedding, redução e modelo (seção
  7) estão na CV estratificada.
- **`Calcifications`**: no sweep a classe positiva ficou invertida ("sem calcificação"); PR-AUC,
  precision, recall e accuracy desse alvo vindos do sweep ficam de fora (—). O ROC-AUC não é
  afetado, e os valores da seção 9 vêm do experimento de 2026-09-28, já corrigido.
- **`Classification` (benigno / maligno) não foi usada como alvo** — é provavelmente a variável
  clínica mais relevante e fica como próximo alvo natural.
- **Próximos passos**: embedding só da lesão, recortada pela máscara (`Mask_tumor_filename`), com CV
  agrupada por `Pixel_size`; incluir `Classification`; ajuste de limiar para precision/recall.

## Apêndice — arquivos

| Seção | Fonte |
|---|---|
| 2 | planilha clínica (`clinical_targets.py`), imagens (tamanhos) |
| 7, 8 | `gbm_pca_umap_sweep/results_long.csv`, `paired_tests.csv` |
| 8, 9 | `gbm_pca_umap_sweep/metadata_only_baseline.csv`, `acquisition_confound/results.csv` |
| 10 | `acquisition_confound/summary_by_group.csv`, `history_from_annotations.csv` |
"""
    return md


def records(df: pd.DataFrame) -> list:
    return json.loads(df.to_json(orient='records'))


def write_html(data, tabs) -> Path:
    """Interactive version of the report: report_template.html with the report's data inlined."""
    cls = data['cls']
    pick = lambda emb, setting: cls[(cls.embedding == emb) & (cls.setting == setting) & (cls.model == 'catboost')] \
        .set_index('target')['roc_auc_mean']
    mae, vae = pick('mae_cls', 'raw'), pick('vae_flat', 'pca10')
    mae_vae = pd.DataFrame({'target': mae.index, 'grupo': mae.index.map(target_group),
                            'mae': mae.values, 'vae': vae.reindex(mae.index).values})
    conf = tabs['confound'].set_index(['target_group', 'scheme'])
    gb = tabs['global']
    robust = gb[(gb.grupo == 'image') & gb.agrup_ganho_supera_margem & (gb.agrup_ganho > 0)]
    heat = tabs['heat'].rename(columns={'grupo': 'group'})
    img = heat[heat.group == 'image'].set_index('embedding')
    payload = {
        'summary': {
            'n_images': int(len(data['raw_df'])), 'n_image_targets': int((gb.grupo == 'image').sum()),
            'mae': float(img.at['mae_cls', 'raw']), 'vae': float(img.loc['vae_flat', SETTINGS].max()),
            'best_strat': float(conf.at[('image', 'stratified'), 'embedding_roc_auc_mean']),
            'best_grouped': float(conf.at[('image', 'group_pixel_size'), 'embedding_roc_auc_mean']),
            'acq_strat': float(conf.at[('image', 'stratified'), 'metadata_roc_auc_mean']),
            'n_robust': int(len(robust)),
        },
        'dictionary': records(tabs['dictionary']),
        'targets': records(tabs['targets']),
        'heat': records(heat),
        'models': records(tabs['models']),
        'mae_vae': records(mae_vae),
        'best': records(tabs['best']),
        'global': records(gb),
        'confound': records(tabs['confound']),
        'history': records(tabs['history']),
        'freq': records(tabs['freq']),
        'vs_global': records(tabs['vs_global']),
    }
    template = (Path(__file__).parent / 'report_template.html').read_text()
    out = OUT / 'relatorio.html'
    out.write_text(template.replace('__REPORT_DATA__', json.dumps(payload, ensure_ascii=False)))
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    data = load()

    tabs = {
        'dictionary': save_table(dictionary_table(data['raw_df']), 'tab_dicionario_variaveis.csv'),
        'targets': save_table(targets_table(data), 'tab_alvos.csv'),
        'best': save_table(best_per_variable(data), 'tab_melhor_por_variavel.csv'),
        'global': save_table(global_best_table(data), 'tab_melhor_global.csv'),
        'history': history_table(data),
    }
    tabs['freq'], tabs['vs_global'] = strategy_frequency()
    data['sig'].to_csv(CONFOUND / 'significance_tests.csv', index=False)
    save_table(data['sig'], 'tab_testes_significancia.csv')
    figs = {'prevalence': fig_prevalence(data)}
    figs['heatmap'], tabs['heat'] = fig_heatmap(data)
    figs['pca_umap'] = fig_pca_umap(data)
    figs['models'], tabs['models'] = fig_models(data)
    figs['mae_vae'] = fig_mae_vs_vae(data)
    figs['best'] = fig_best_per_variable(tabs['best'])
    figs['global_roc'] = fig_global_best(tabs['global'], 'roc_auc')
    figs['global_pr'] = fig_global_best(tabs['global'], 'pr_auc')
    figs['confound'], tabs['confound'] = fig_confound(data)
    figs['history'] = fig_history(tabs['history'])

    (OUT / 'report.md').write_text(build_markdown(data, figs, tabs))
    html = write_html(data, tabs)
    print(f'Report: {OUT / "report.md"}  ({len(figs)} figures, {len(list(TAB.glob("*.csv")))} tables); page: {html}')


if __name__ == '__main__':
    main()
