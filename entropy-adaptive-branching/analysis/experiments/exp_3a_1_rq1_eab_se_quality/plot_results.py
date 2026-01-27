"""
Plot results from RQ1: EAB+SE Quality Evaluation
Generates the same plots as SE-alone experiment for direct comparison.
"""

import sys
import json
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import roc_curve, auc

# Set pastel style
sns.set_palette("pastel")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

experiment_dir = Path(__file__).parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = experiment_dir / "config.yaml"
    else:
        config_path = Path(config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_results(config: Dict[str, Any]):
    results_dir = experiment_dir / config['output']['results_dir']
    with open(results_dir / "raw_results.json") as f:
        return json.load(f)['results']


def plot_roc_curve(results, save_dir, config):
    """ROC curve for SE predicting incorrect answers."""
    se = np.array([r['se_uncertainty_score'] for r in results])
    y_true = 1 - np.array([r['best_sample_correct'] for r in results])

    fpr, tpr, _ = roc_curve(y_true, se)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color=sns.color_palette()[0], linewidth=2.5,
             label=f'SE with EAB (AUROC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: SE → Incorrect Answers (EAB Generation)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: roc_curve.png (AUROC = {roc_auc:.3f})")


def plot_distribution_comparison(results, save_dir):
    """Distribution of SE scores by correctness."""
    correct = [r['se_uncertainty_score'] for r in results if r['best_sample_correct']]
    incorrect = [r['se_uncertainty_score'] for r in results if not r['best_sample_correct']]

    plt.figure(figsize=(8, 5))
    plt.hist(correct, bins=20, alpha=0.7, label=f'Correct (n={len(correct)})',
             color=sns.color_palette("pastel")[2])
    plt.hist(incorrect, bins=20, alpha=0.7, label=f'Incorrect (n={len(incorrect)})',
             color=sns.color_palette("pastel")[3])

    plt.axvline(np.mean(correct), color='darkgreen', linestyle='--', linewidth=2,
                label=f'Mean Correct: {np.mean(correct):.3f}')
    plt.axvline(np.mean(incorrect), color='darkred', linestyle='--', linewidth=2,
                label=f'Mean Incorrect: {np.mean(incorrect):.3f}')

    plt.xlabel('Semantic Uncertainty Score')
    plt.ylabel('Frequency')
    plt.title('SE Distribution by Answer Correctness (EAB Samples)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: distribution_comparison.png")


def plot_cluster_accuracy(results, save_dir):
    """Accuracy by number of semantic clusters."""
    # Group by cluster count
    cluster_accuracy = {}
    for r in results:
        n = r['se_n_clusters']
        if n not in cluster_accuracy:
            cluster_accuracy[n] = []
        cluster_accuracy[n].append(1 if r['best_sample_correct'] else 0)

    clusters = sorted(cluster_accuracy.keys())
    accuracies = [np.mean(cluster_accuracy[c]) for c in clusters]
    counts = [len(cluster_accuracy[c]) for c in clusters]

    overall_acc = np.mean([r['best_sample_correct'] for r in results])

    plt.figure(figsize=(10, 5))
    bars = plt.bar(clusters, accuracies, color=sns.color_palette("pastel")[1], alpha=0.8)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    plt.axhline(overall_acc, color='red', linestyle='--', linewidth=2,
                label=f'Overall Accuracy: {overall_acc:.1%}')

    plt.xlabel('Number of Semantic Clusters')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Number of Semantic Clusters (EAB Samples)')
    plt.xticks(clusters)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'cluster_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: cluster_accuracy.png")


def plot_eab_metrics(results, save_dir):
    """Additional plots for EAB-specific metrics."""

    # 1. Distribution of samples per question
    num_samples = [r['num_samples'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Samples distribution
    axes[0].hist(num_samples, bins=30, color=sns.color_palette("pastel")[4], alpha=0.8)
    axes[0].axvline(np.mean(num_samples), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(num_samples):.1f}')
    axes[0].set_xlabel('Number of Samples Generated')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('EAB: Samples per Question')
    axes[0].legend()

    # Branches distribution
    num_branches = [r['eab_num_branches'] for r in results]
    axes[1].hist(num_branches, bins=30, color=sns.color_palette("pastel")[5], alpha=0.8)
    axes[1].axvline(np.mean(num_branches), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(num_branches):.1f}')
    axes[1].set_xlabel('Number of Branches')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('EAB: Branches per Question')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / 'eab_generation_stats.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: eab_generation_stats.png")


def main():
    parser = argparse.ArgumentParser(description="Plot RQ1 results")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    print("=" * 70)
    print("PLOTTING RQ1 RESULTS")
    print("=" * 70)

    config = load_config(args.config)
    results = load_results(config)

    results_dir = experiment_dir / config['output']['results_dir']
    save_dir = results_dir / "figures"
    save_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nGenerating plots for {len(results)} questions...")
    print(f"Clustering threshold: δ = {config['semantic_entropy']['default_threshold']}")
    print(f"EAB entropy threshold: τ = {config['eab']['entropy_threshold']}")

    # Generate all plots
    plot_roc_curve(results, save_dir, config)
    plot_distribution_comparison(results, save_dir)
    plot_cluster_accuracy(results, save_dir)
    plot_eab_metrics(results, save_dir)

    print(f"\n✓ All figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
