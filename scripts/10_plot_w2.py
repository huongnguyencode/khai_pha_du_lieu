"""Export optional PNG/SVG figures from measured W2 results (Matplotlib)."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nhanes-w2-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/w2'


def main():
    result = json.loads((OUT / 'results.json').read_text())
    colors = ['#555555', '#e69f00', '#0072b2', '#cc79a7', '#009e73']
    summaries = result['summary']
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar([r['model'].split('_')[0] for r in summaries],
                  [r['macro_f1'] for r in summaries], color=colors,
                  yerr=[r['macro_f1_std'] for r in summaries], capsize=4)
    for bar, row in zip(bars, summaries):
        ax.text(bar.get_x()+bar.get_width()/2, .015, f"{row['macro_f1']:.4f}",
                ha='center', va='bottom', color='white', fontweight='bold')
    ax.set(ylabel='Macro-F1 (mean outer folds; bars = SD)', xlabel='Controlled ablation',
           title='NHANES W2: paired repeated nested CV', ylim=(0, .6))
    ax.spines[['right', 'top']].set_visible(False)
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(OUT / f'macro_f1_comparison.{extension}', dpi=160)
    plt.close(fig)

    data = list(csv.DictReader((OUT / 'oof_predictions.csv').open()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for color, summary in zip(colors, summaries):
        part = [r for r in data if r['model'] == summary['model']]
        truth = np.array([int(r['true_target']) for r in part])
        for axis, name in enumerate(('egfr_score', 'uacr_score')):
            target = truth // 2 if axis == 0 else truth % 2
            score = np.array([float(r[name]) for r in part])
            order = np.argsort(-score, kind='stable')
            ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(score)-1]
            tp = np.cumsum(target[order])[ends]
            recall = tp / target.sum()
            precision = tp / (ends+1)
            axes[axis].step(np.r_[0, recall], np.r_[1, precision], where='pre',
                            label=summary['model'], color=color, linewidth=1.3, alpha=.85)
    for axis, ax in enumerate(axes):
        ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1),
               title=('Low eGFR', 'High uACR')[axis])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Pooled OOF precision-recall; repeated observations are dependent')
    fig.tight_layout()
    for extension in ('png', 'svg'):
        fig.savefig(OUT / f'precision_recall_curves.{extension}', dpi=160)
    plt.close(fig)
    (OUT / 'figure_environment.json').write_text(json.dumps({'matplotlib': matplotlib.__version__}, indent=2)+'\n')
    print('Saved macro-F1 and precision-recall figures (PNG + SVG).')


if __name__ == '__main__':
    main()
