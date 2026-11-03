"""Compare paired, complete L1 utility stores from the same training epoch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from adavip.manifeel.full_sensor_schema import CONDITION_NAMES
from adavip.manifeel.stage1_dataset import UtilityLabelStore
from adavip.manifeel.utility_label_store import atomic_json


def summarize_pair(original: np.ndarray, dropout: np.ndarray) -> dict:
    """Summarize paired raw signed scores without clipping or fitting statistics."""
    original, dropout = np.asarray(original), np.asarray(dropout)
    if original.shape != dropout.shape or original.ndim != 1 or not len(original):
        raise ValueError('Expected aligned nonempty one-dimensional scores')
    if not np.isfinite(original).all() or not np.isfinite(dropout).all():
        raise ValueError('Scores must be finite')
    varying = original.std() > 0 and dropout.std() > 0
    return {
        'windows': len(original), 'original_mean': float(original.mean()),
        'dropout_mean': float(dropout.mean()), 'delta_mean': float((dropout-original).mean()),
        'original_std': float(original.std()), 'dropout_std': float(dropout.std()),
        'original_negative_fraction': float((original < 0).mean()),
        'dropout_negative_fraction': float((dropout < 0).mean()),
        'sign_agreement': float((np.sign(original) == np.sign(dropout)).mean()),
        'pearson': float(np.corrcoef(original, dropout)[0, 1]) if varying else None,
        'spearman': float(spearmanr(original, dropout).statistic) if varying else None,
    }


def validate_pair(original: UtilityLabelStore, dropout: UtilityLabelStore) -> None:
    """Reject differing windows, epoch, normalizer, draw streams or scoring protocols."""
    np.testing.assert_array_equal(original.index, dropout.index)
    np.testing.assert_array_equal(original.task_ids, dropout.task_ids)
    left, right = original.manifest, dropout.manifest
    if left.get('teacher_protocol') != 'original_snapshot' or right.get('teacher_protocol') != 'dropout10':
        raise ValueError('Expected original L1 and input-dropout10 snapshot stores')
    for key in ('dimensions', 'conditions', 'protocol', 'draws', 'reference_manifest_sha256',
                'mask_semantics', 'validity', 'label_normalization'):
        if left[key] != right[key]:
            raise ValueError(f'Comparison protocol differs: {key}')
    for key in ('normalizer_sha256',):
        if left['teacher'][key] != right['teacher'][key]:
            raise ValueError(f'Comparison teacher differs: {key}')
    for key in ('epoch', 'global_step', 'weights'):
        if left['teacher']['metadata'][key] != right['teacher']['metadata'][key]:
            raise ValueError(f'Comparison checkpoint differs: {key}')
    if left['runtime'] != right['runtime']:
        raise ValueError('Comparison scoring runtime differs')


def compare_stores(original_root: Path, dropout_root: Path, output: Path) -> dict:
    """Verify complete stores and export pooled/task tables, plots and checkpoint provenance."""
    original, dropout = UtilityLabelStore(original_root), UtilityLabelStore(dropout_root)
    validate_pair(original, dropout)
    output.mkdir(parents=True, exist_ok=True)
    tasks = {int(task['task_index']): task['task_id'] for task in original.manifest['tasks']}
    rows, losses = [], []
    original_losses = np.load(original_root/'weighted_losses.npy')
    dropout_losses = np.load(dropout_root/'weighted_losses.npy')
    for task in [-1, *sorted(tasks)]:
        selected = np.ones(len(original.index), dtype=bool) if task == -1 else original.index['task_index'] == task
        for dimension, name in enumerate(CONDITION_NAMES[1:]):
            rows.append({'task_index': task, 'task_id': tasks.get(task, 'all'), 'dimension': name,
                         **summarize_pair(original.utility[selected, dimension], dropout.utility[selected, dimension])})
        losses.append({'task_index': task, 'task_id': tasks.get(task, 'all'),
                       'original_full_loss': float(original_losses[selected, 0].mean()),
                       'dropout_full_loss': float(dropout_losses[selected, 0].mean()),
                       'top_utility_agreement': float((original.utility[selected].argmax(1) ==
                                                      dropout.utility[selected].argmax(1)).mean())})
    for filename, records in (('utility_comparison.csv', rows), ('full_loss_comparison.csv', losses)):
        with (output/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    result = {'status': 'complete', 'epoch': original.manifest['teacher']['metadata']['epoch'],
              'windows': len(original.index), 'original': str(original_root), 'dropout': str(dropout_root),
              'original_checkpoint': original.manifest['checkpoint'],
              'dropout_checkpoint': dropout.manifest['checkpoint'], 'paired_protocol_verified': True,
              'utility_rows': rows, 'loss_rows': losses,
              'interpretation': 'Signed masked-minus-full denoising MSE on training demonstrations; '
                                'smaller utility measures less loss increase under removal, not rollout success. '
                                'Overlapping windows are not independent samples.'}
    plot_comparison(rows, tasks, output, result['epoch'])
    lines = [f'# L1 E{result["epoch"]} EMA: input dropout comparison', '',
             f'{len(original.index):,} paired windows, K64, strict FP32; both checkpoints use the same epoch/step.', '',
             '| Removal | No dropout | Dropout10 | Difference | Negative % (no / yes) |',
             '|---|---:|---:|---:|---:|']
    for row in rows[:9]:
        lines.append(f'| {row["dimension"]} | {row["original_mean"]:.6g} | {row["dropout_mean"]:.6g} | '
                     f'{row["delta_mean"]:+.6g} | {100*row["original_negative_fraction"]:.2f} / '
                     f'{100*row["dropout_negative_fraction"]:.2f} |')
    lines += ['', f'Full-input loss: {losses[0]["original_full_loss"]:.6g} (no dropout), '
              f'{losses[0]["dropout_full_loss"]:.6g} (dropout10).', '', result['interpretation'], '',
              f'Original checkpoint: `{original.manifest["checkpoint"]["path"]}`', '',
              f'Dropout checkpoint: `{dropout.manifest["checkpoint"]["path"]}`', '']
    (output/'comparison.md').write_text('\n'.join(lines))
    atomic_json(output/'COMPLETE.json', result)
    return result


def plot_comparison(rows: list[dict], tasks: dict, output: Path, epoch: int) -> None:
    """Save exportable pooled utility, negative fraction and per-task difference plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True,
                                gridspec_kw={'height_ratios': [1, 1, 2]})
    x = np.arange(9)
    for offset, prefix, label in ((-.2, 'original', 'No dropout'), (.2, 'dropout', 'Input dropout10')):
        axes[0].bar(x+offset, [row[f'{prefix}_mean'] for row in rows[:9]], .4, label=label)
        axes[1].bar(x+offset, [100*row[f'{prefix}_negative_fraction'] for row in rows[:9]], .4, label=label)
    for axis in axes[:2]:
        axis.set_xticks(x, CONDITION_NAMES[1:])
        axis.legend()
        axis.axhline(0, color='black', linewidth=.5)
    axes[0].set_ylabel('Mean signed utility (MSE)')
    axes[0].set_title(f'L1 E{epoch} EMA, paired K64 complete windows')
    axes[1].set_ylabel('Negative utility (%)')
    matrix = np.array([[row['delta_mean'] for row in rows if row['task_index'] == task] for task in sorted(tasks)])
    limit = max(float(np.abs(matrix).max()), 1e-12)
    picture = axes[2].imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
    axes[2].set_xticks(x, CONDITION_NAMES[1:])
    axes[2].set_yticks(np.arange(len(tasks)), [tasks[task] for task in sorted(tasks)])
    axes[2].set_title('Mean utility difference: dropout10 minus no dropout')
    figure.colorbar(picture, ax=axes[2], label='MSE difference')
    figure.savefig(output/'utility_comparison.png', dpi=180)
    figure.savefig(output/'utility_comparison.pdf')
    plt.close(figure)


def main() -> None:
    """Compare two completed stores without requiring a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--dropout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare_stores(args.original, args.dropout, args.output)
    print(json.dumps({key: result[key] for key in ('status', 'epoch', 'windows', 'paired_protocol_verified')}))


if __name__ == '__main__':
    main()
