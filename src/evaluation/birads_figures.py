"""Figures for the BIRADS analysis (src/evaluation/birads_analysis.py). Reads only the
saved CSVs and writes PNGs into birads_analysis/figures/.

Usage:
    python -m src.evaluation.birads_figures
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.birads_analysis import CLASSES, CUTS
from src.evaluation.build_report import (
    AQUA, AXIS, BASELINE_GRAY, BLUE, GRID, INK, INK2, ORANGE, SCHEME_PT, SURFACE, legend_marker,
)
from src.models.train_embedding_to_clinical_gbm import OUT_ROOT

DIR = OUT_ROOT / 'birads_analysis'
FIG = DIR / 'figures'
YELLOW = '#eda100'

# Fixed entity -> color, never by rank. Combinations without the embedding are drawn as
# baselines; late fusion is the fair "embedding + pre-US" (see birads_analysis docstring).
SETS = {
    'acquisition': ('Aquisição', BASELINE_GRAY),
    'pre_us': ('Pré-US', ORANGE),
    'embedding': ('Embedding', BLUE),
    'embedding+pre_us': ('Embedding + pré-US (fusão precoce)', YELLOW),
    'emb_score+pre_us': ('Embedding + pré-US (fusão tardia)', AQUA),
}
TARGET_PT = {'BIRADS': 'Multiclasse (macro)', **{f'BIRADS_ge_{c}': f'≥ {c}' for c in CUTS},
             **{f'BIRADS_is_{c}': f'BIRADS {c}' for c in CLASSES}}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'figure.dpi': 150,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
})


def save(fig, name: str):
    fig.savefig(FIG / name, bbox_inches='tight')
    plt.close(fig)
    print(FIG / name)


def finish(fig, title: str, handles, ncol: int = 3):
    h = fig.get_figheight()
    legend_rows = -(-len(handles) // ncol)
    fig.suptitle(title, x=0.01, y=1 - 0.12 / h, ha='left', va='top', fontweight='bold')
    fig.tight_layout(rect=[0, (0.15 + 0.25 * legend_rows) / h, 1, 1 - 0.3 / h])
    fig.legend(handles=handles, loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, 0.02 / h), fontsize=8.5,
               frameon=False)


def dot_panel(ax, rows: pd.DataFrame, targets: list, sets: list, marker='o', filled=True, offset=0.0):
    """Horizontal dot + std whisker per (target, feature set); targets on y."""
    step = 0.7 / max(len(sets), 1)
    for j, fs in enumerate(sets):
        color = SETS[fs][1]
        for i, t in enumerate(targets):
            r = rows[(rows['target'] == t) & (rows['feature_set'] == fs)]
            if r.empty:
                continue
            y = i - 0.35 + step * (j + 0.5) + offset
            m, s = r['roc_auc_mean'].iloc[0], r['roc_auc_std'].iloc[0]
            ax.plot([m - s, m + s], [y, y], color=color, lw=1.5, alpha=0.5, solid_capstyle='round')
            ax.plot(m, y, marker, ms=6.5, color=color if filled else SURFACE, mec=color if not filled else SURFACE,
                    mew=1.5 if not filled else 1.2, zorder=3)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks(range(len(targets)), [TARGET_PT[t] for t in targets])
    ax.invert_yaxis()
    ax.set_xlabel('ROC-AUC (média ± dp entre folds)')


def fig_cuts(res: pd.DataFrame):
    targets = ['BIRADS'] + [f'BIRADS_ge_{c}' for c in CUTS]
    rows = res[res['method'].isin(['multiclass', 'dedicated'])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, scheme in zip(axes, ['stratified', 'group_pixel_size']):
        dot_panel(ax, rows[rows['scheme'] == scheme], targets, list(SETS))
        ax.set_title(SCHEME_PT[scheme], loc='left', fontsize=9.5, color=INK2)
    finish(fig, 'BIRADS multiclasse vs. binarizado em três cortes', [legend_marker(c, l) for l, c in SETS.values()])
    save(fig, 'fig1_cortes.png')


def fig_categories(res: pd.DataFrame):
    targets = [f'BIRADS_is_{c}' for c in CLASSES]
    sets = ['pre_us', 'embedding', 'emb_score+pre_us']
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharex=True, sharey=True)
    for ax, scheme in zip(axes, ['stratified', 'group_pixel_size']):
        r = res[res['scheme'] == scheme]
        dot_panel(ax, r[r['method'] == 'dedicated'], targets, sets, filled=True, offset=-0.05)
        dot_panel(ax, r[r['method'] == 'from_multiclass'], targets, sets, filled=False, offset=0.05)
        ax.set_title(SCHEME_PT[scheme], loc='left', fontsize=9.5, color=INK2)
    handles = [legend_marker(SETS[s][1], SETS[s][0]) for s in sets] + [
        plt.Line2D([], [], marker='o', ls='none', color=INK2, mec=SURFACE, ms=6.5, label='Modelo binário dedicado'),
        plt.Line2D([], [], marker='o', ls='none', color=SURFACE, mec=INK2, mew=1.5, ms=6.5,
                   label='Lido do modelo multiclasse')]
    finish(fig, 'ROC-AUC por categoria do BIRADS (um-contra-o-resto)', handles, ncol=3)
    save(fig, 'fig2_categorias.png')


def fig_score_distribution(oof: pd.DataFrame, scheme: str):
    """Out-of-fold score per true category: P(>= cut) from the dedicated binary models
    and the ordinal regression score."""
    sets = ['pre_us', 'embedding', 'emb_score+pre_us']
    panels = [(f'BIRADS_ge_{c}', 'dedicated', f'P(BIRADS ≥ {c})') for c in CUTS] + \
             [('BIRADS', 'ordinal', 'Score ordinal (posto 0–5)')]
    d = oof[oof['scheme'] == scheme]
    fig, axes = plt.subplots(len(panels), len(sets), figsize=(11, 2.4 * len(panels)), sharex=True)
    rng = np.random.default_rng(0)
    for i, (target, method, ylabel) in enumerate(panels):
        for j, fs in enumerate(sets):
            ax = axes[i, j]
            g = d[(d['target'] == target) & (d['method'] == method) & (d['feature_set'] == fs)]
            data = [g.loc[g['birads'].astype(str) == c, 'score'].to_numpy() for c in CLASSES]
            color = SETS[fs][1]
            bp = ax.boxplot(data, positions=range(len(CLASSES)), widths=0.55, patch_artist=True, showfliers=False,
                            medianprops={'color': INK, 'lw': 1.5}, whiskerprops={'color': INK2},
                            capprops={'color': INK2}, boxprops={'edgecolor': INK2, 'lw': 0.8})
            for b in bp['boxes']:
                b.set_facecolor(color)
                b.set_alpha(0.35)
            for k, v in enumerate(data):
                ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=color, alpha=0.7, lw=0, zorder=3)
            if method == 'dedicated':
                cut = CLASSES.index(target.split('_')[-1])
                ax.axvline(cut - 0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
                ax.set_ylim(-0.03, 1.03)
            ax.set_xticks(range(len(CLASSES)), CLASSES)
            if i == 0:
                ax.set_title(SETS[fs][0], loc='left', fontsize=9.5, color=INK2)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(panels) - 1:
                ax.set_xlabel('BIRADS verdadeiro')
    h = fig.get_figheight()
    fig.suptitle(f'Distribuição do score predito (fora do fold) por categoria — {SCHEME_PT[scheme]}',
                 x=0.01, y=1 - 0.12 / h, ha='left', va='top', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.4 / h])
    save(fig, f'fig3_distribuicao_score_{scheme}.png')


def fig_gains(sig: pd.DataFrame):
    """Gain with the corrected 95% CI for the pre-registered comparisons, per target."""
    families = [('pre_us vs dummy', 'Pré-US − acaso', ORANGE),
                ('embedding+pre_us vs pre_us', 'Emb. + pré-US (precoce) − pré-US', YELLOW),
                ('emb_score+pre_us vs pre_us', 'Emb. + pré-US (tardia) − pré-US', AQUA),
                ('emb_score+acquisition+pre_us vs acquisition+pre_us', 'Emb. + aq. + pré-US (tardia) − aq. + pré-US', BLUE)]
    targets = ['BIRADS'] + [f'BIRADS_ge_{c}' for c in CUTS]
    s = sig[sig['method'].isin(['multiclass', 'dedicated']) & sig['target'].isin(targets)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    step = 0.7 / len(families)
    for ax, scheme in zip(axes, ['stratified', 'group_pixel_size']):
        for j, (fam, _, color) in enumerate(families):
            for i, t in enumerate(targets):
                r = s[(s['family'] == fam) & (s['target'] == t) & (s['scheme'] == scheme)]
                if r.empty:
                    continue
                r = r.iloc[0]
                half = stats.t.ppf(0.975, r['k'] - 1) * r['se_corrected']
                y = i - 0.35 + step * (j + 0.5)
                ax.plot([r['gain'] - half, r['gain'] + half], [y, y], color=color, lw=1.5, alpha=0.5)
                ax.plot(r['gain'], y, 'o' if r['significant'] else 'o', ms=6.5,
                        color=color if r['significant'] else SURFACE, mec=SURFACE if r['significant'] else color,
                        mew=1.2 if r['significant'] else 1.5, zorder=3)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(range(len(targets)), [TARGET_PT[t] for t in targets])
        ax.invert_yaxis()
        ax.set_xlabel('Ganho de ROC-AUC (IC 95% corrigido, Nadeau–Bengio)')
        ax.set_title(SCHEME_PT[scheme], loc='left', fontsize=9.5, color=INK2)
    handles = [legend_marker(c, l) for _, l, c in families] + [
        plt.Line2D([], [], marker='o', ls='none', color=INK2, mec=SURFACE, ms=6.5, label='q < 0.05 (cheio)'),
        plt.Line2D([], [], marker='o', ls='none', color=SURFACE, mec=INK2, mew=1.5, ms=6.5, label='q ≥ 0.05 (vazado)')]
    finish(fig, 'Ganho de cada fonte de informação sobre o BIRADS', handles, ncol=2)
    save(fig, 'fig4_ganhos.png')


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(DIR / 'results.csv')
    oof = pd.read_csv(DIR / 'oof_predictions.csv')
    sig = pd.read_csv(DIR / 'significance_tests.csv')
    fig_cuts(res)
    fig_categories(res)
    for scheme in ['stratified', 'group_pixel_size']:
        fig_score_distribution(oof, scheme)
    fig_gains(sig)


if __name__ == '__main__':
    main()
