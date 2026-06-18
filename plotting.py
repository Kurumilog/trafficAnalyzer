from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _make_confusion_matrix(y_true, y_pred, output_path):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    labels = ['Normal', 'Attack']
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix')

    for row in range(2):
        for col in range(2):
            ax.text(col, row, f'{matrix[row, col]}', ha='center', va='center',
                    color='white' if matrix[row, col] > matrix.max() * 0.5 else 'black',
                    fontsize=12, fontweight='bold')

    _save_figure(fig, output_path)


def plot_results(re_test, y_test, threshold, W, X_test_sc, f1, K, output_dir='.', prefix='anomaly'):
    """Generate a set of English-only figures that summarize the detector."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    re_norm_test = re_test[y_test == 0]
    re_atk_test = re_test[y_test == 1]
    predicted = (re_test > threshold).astype(int)
    sample_size = min(3000, len(re_test))
    sample_indices = np.linspace(0, len(re_test) - 1, sample_size, dtype=int)

    fig1, ax1 = plt.subplots(figsize=(14, 5))
    colors = np.where(y_test[sample_indices] == 1, '#d95f5f', '#4c78a8')
    ax1.scatter(sample_indices, re_test[sample_indices], c=colors, s=10, alpha=0.65, linewidths=0)
    ax1.axhline(threshold, color='#d62728', linewidth=2, linestyle='--', label=f'Threshold = {threshold:.4f}')
    ax1.set_title(f'Reconstruction error by flow | K={K} | F1={f1:.2f}')
    ax1.set_xlabel('Flow index')
    ax1.set_ylabel('Reconstruction error')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.grid(True, which='both', alpha=0.15)
    _save_figure(fig1, output_dir / f'{prefix}_reconstruction_error.png')

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    lower = max(np.percentile(re_test, 1), 1e-4)
    upper = max(np.percentile(re_test, 99.7), lower * 10)
    bins = np.logspace(np.log10(lower), np.log10(upper), 70)
    ax2.hist(re_norm_test, bins=bins, alpha=0.65, color='#4c78a8', label='Normal', density=True)
    ax2.hist(re_atk_test, bins=bins, alpha=0.65, color='#d95f5f', label='Attack', density=True)
    ax2.axvline(threshold, color='#d62728', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xscale('log')
    ax2.set_title('Reconstruction error distribution')
    ax2.set_xlabel('Reconstruction error')
    ax2.set_ylabel('Density')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', alpha=0.15)
    _save_figure(fig2, output_dir / f'{prefix}_error_distribution.png')

    fig3, ax3 = plt.subplots(figsize=(8, 7))
    proj = X_test_sc @ W[:, :2]
    projection_sample = np.random.choice(len(proj), min(2500, len(proj)), replace=False)
    proj_colors = np.where(y_test[projection_sample] == 1, '#d95f5f', '#4c78a8')
    ax3.scatter(proj[projection_sample, 0], proj[projection_sample, 1], c=proj_colors, s=12, alpha=0.65, linewidths=0)
    ax3.set_title('First two PCA components')
    ax3.set_xlabel('Principal component 1')
    ax3.set_ylabel('Principal component 2')
    ax3.grid(True, alpha=0.15)
    _save_figure(fig3, output_dir / f'{prefix}_pca_projection.png')

    _make_confusion_matrix(y_test, predicted, output_dir / f'{prefix}_confusion_matrix.png')

    print(f"Plots saved to: {output_dir.resolve()}")
